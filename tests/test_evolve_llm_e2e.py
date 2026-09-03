"""The LLM proposer inside the evolve loop, end to end through the REAL runtime +
scripts/evolve.py on the test_evolve_e2e task card, the model_endpoint FAKE answering
a fixed sequence (PH_MODEL_ENDPOINT_FAKE): round 1 a tunables answer becomes the
round's try (proposer "llm", summary kept); round 2 a card whose ref is outside its
dir is rejected by the doctor, the exact text goes back to the model, its repaired
card passes the doctor + the dry instantiation + the one-seed preflight, is mounted
for the trial suite and wins; round 3 a stub executor whose act() returns the wrong
shape raises on the preflight seed -- the traceback goes back, a non-JSON repair and a
second stub exhaust the 3 attempts, the round is an honest none with the last reason;
round 4 the model answers none: two none rounds in a row end the loop (status done)
long before the 6 rounds asked. Every attempt's raw answer is in the audit file.
``PH_LLM_E2E=1`` adds one REAL DeepSeek call on the recycle_cans brief (prompt size)."""

from __future__ import annotations

import json
import os

import pytest
from test_evolve_e2e import _CARD, EMB, TASK
from test_mission_e2e import _Runtime, _kinds

from board import store as bs

_EXEC = ("from harness.skill_executor import InprocExecutor\n\n\n"
         "class _E(InprocExecutor):\n    def bind(self, env, target=None):\n        pass\n\n"
         "    def act(self, obs):\n        return {act}\n\n\n"
         "class _P:\n    def make_driver(self, spec):\n        return _E()\n\n\n"
         "def provider(**params):\n    return _P()\n")
GOOD, BAD_SHAPE = _EXEC.format(act="(0.0,)"), _EXEC.format(act="(0.0, 0.0)")
MANIFEST = (f'needs_sim = true\n[executors.{{to}}]\nskill = "grab"\nembodiment = "{EMB}"\n'
            'ref = "{name}:provider"\ntransport = "inproc"\n')


def _card(name, to, code, ref=None, node=None):
    pay = {"name": name, "to": to, "ref": ref or f"{name}:provider",
           "files": {"manifest.toml": MANIFEST.format(to=to, name=name), "__init__.py": code}}
    if node:
        pay["node"] = node
    return {"kind": "card", "payload": pay, "summary": f"写执行器 {name}。", "rationale": "code-as-policy"}


CANNED = [
    {"kind": "tunables", "payload": {"ref": "test_evolve_e2e:policy_provider",
                                     "path": ["tunables", "stall_k"], "to": 28},
     "summary": "两颗种子都死在 grab-0。", "rationale": "先把 stall_k 调低"},
    _card("grab_llm", "llm", GOOD, ref="grab_other:provider"),      # round 2, attempt 1: ref outside the dir
    _card("grab_llm", "llm", GOOD),                                  # round 2, attempt 2: repaired
    _card("grab_stub", "stub", BAD_SHAPE, node="grab-0"),            # round 3, attempt 1: raises on the seed
    "not json at all",                                               # round 3, attempt 2
    _card("grab_stub", "stub", BAD_SHAPE, node="grab-0"),            # round 3, attempt 3
    {"kind": "none", "payload": {}, "summary": "没有值得试的。", "rationale": "两颗种子都过了"},   # round 4
]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    rt = _Runtime(runs, card=_CARD, canned=CANNED, mode="evolution",
                  env={"PH_CANDIDATES_ROOT": str(runs / "candidates")})
    rt.campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    try:
        rt.rows = rt.run({"kind": "evolve", "task": TASK, "seeds": [1, 2], "rounds": 6, "arm": "auto"})[1]
        yield rt
    finally:
        rt.stop()




def test_llm_answers_drive_the_rounds_repair_from_the_exact_error_and_stop_honestly(runtime):
    doc = json.loads(runtime.campaign.read_text())
    assert doc["status"] == "done" and doc["cursor"] == 4 and len(doc["rounds"]) == 4   # 6 asked: 2 tries + 2 none
    r1, r2, r3, r4 = doc["rounds"]
    # round 1: the tunables answer is the try; summary / provenance on the row, never the key
    assert (r1["proposer"], r1["tried"]["kind"], r1["tried"]["node"]) == ("llm", "tunables", "grab-0")
    assert (r1["tried"]["detail"]["path"], r1["tried"]["detail"]["to"]) == (["tunables", "stall_k"], 28)
    assert r1["llm"]["summary"] == "两颗种子都死在 grab-0。" and r1["llm"]["rationale"] == "先把 stall_k 调低"
    assert len(r1["llm"]["prompt_sha"]) == 64 and len(r1["llm"]["raw_sha"]) == 64
    assert r1["llm"]["model"].startswith("fake(") and r1["llm"]["reason"] is None
    assert (r1["before"], r1["after"], r1["published"]) == (0, 0, False)
    audits = {p.name: json.loads(p.read_text()) for p in (runtime.campaign.parent / "llm").glob("round-*.json")}
    assert set(audits) == {f"round-{r}.json" for r in (1, 2, 3, 4)}
    # call 1 carries the brief alone (one call: the decision came with its payload); the code
    # material a coder needs sits in the audit's ``materials`` (call 2's static message)
    a1 = audits["round-1.json"]
    assert a1["calls"] == 1 and [m["role"] for m in a1["messages"]] == ["system", "user"]
    proj = {**a1["brief"], **a1["materials"]}
    assert a1["messages"][1]["content"].startswith("Round input:\n" + json.dumps(a1["brief"], sort_keys=True)[:200])
    assert proj["first_death"]["node"] == "grab-0" and proj["first_death"]["embodiment"] == EMB
    assert proj["card_template"]["ref"] == "<name>:provider"   # a tmp candidates root: no package prefix
    assert f'skill = "grab"\nembodiment = "{EMB}"\nref = "<name>:provider"\ntransport = "inproc"' \
        in proj["card_template"]["manifest.toml"]
    assert "handshake()" in proj["executor_contract"] and "bind(env, target=None)" in proj["executor_contract"]
    assert "class GeometricGraspExecutor(InprocExecutor)" in proj["reference_card"]["__init__.py"]
    assert "[executors.geometric]" in proj["reference_card"]["manifest.toml"]
    assert proj["scripted_driver_source"] is None and "primitives" not in proj   # the fake embodiment has neither
    assert proj["first_death"]["modules"] == []   # ... nor a _STAGES table: nothing to patch
    # round 2: the bad ref is rejected by the doctor, the text goes back verbatim, the repair wins
    assert (r2["proposer"], r2["tried"]["kind"], r2["tried"]["detail"]["to"]) == ("llm", "card", "llm")
    assert r2["tried"]["detail"]["path"] == str(runtime.runs / "candidates" / "grab_llm")
    assert (r2["before"], r2["after"], r2["published"]) == (0, 2, True)
    a2 = audits["round-2.json"]
    assert len(a2["attempts"]) == 1 and a2["attempts"][0]["reason"].startswith("doctor:ref 'grab_other:provider'")
    assert "must name a provider inside grab_llm" in a2["attempts"][0]["reason"]
    assert a2["calls"] == 2 and a2["messages"][2]["role"] == "assistant" and a2["messages"][3]["role"] == "user"
    assert a2["attempts"][0]["reason"] in a2["messages"][3]["content"] and a2["raw"] == json.dumps(CANNED[2])
    assert doc["applied"]["cards"]["llm"]["ref"] == "grab_llm:provider"
    rec = json.loads((runtime.session / "skills" / f"{r2['tried']['detail']['digest']}.json").read_text())
    assert rec["bindings"][EMB]["policies"]["llm"]["ref"] == "grab_llm:provider"
    # round 3: the stub's act() shape blows up on the preflight seed -> traceback back; three
    # rejected attempts -> an honest none carrying the last reason, files kept for the operator
    assert r3["proposer"] == "llm" and r3["tried"]["kind"] == "none" and r3["outcome"] == "none"
    a3 = audits["round-3.json"]
    assert [a["reason"][:10] for a in a3["attempts"]] == ["preflight:", "ValueError", "preflight:"]
    assert "Traceback" in a3["attempts"][0]["reason"] and "expected a 1-dim action" in a3["attempts"][0]["reason"]
    assert "no JSON object" in a3["attempts"][1]["reason"]
    assert r3["tried"]["detail"]["reason"].startswith("llm: 3 answers rejected; last: preflight:")
    assert "expected a 1-dim action" in r3["needs"][1] and r3["needs"][0] == "proposal"
    assert r3["tried"]["detail"]["path"] == str(runtime.runs / "candidates" / "grab_stub")
    assert (runtime.runs / "candidates" / "grab_stub" / "__init__.py").read_text() == BAD_SHAPE
    assert (r3["before"], r3["after"], r3["published"], r3["parent"]) == (2, 2, False, 2)
    assert r3["llm"]["reason"] is None and r3["llm"]["summary"] == "写执行器 grab_stub。"
    # round 4: an honest none from the model -> the second none in a row ends the loop
    assert r4["proposer"] == "llm" and r4["tried"]["kind"] == "none" and r4["needs"] == ["proposal"]
    assert r4["tried"]["detail"]["reason"] == "llm: 两颗种子都过了"
    # the round rows ride rsi_step / rsi_series; the live log said what the LLM was doing
    steps = _kinds(runtime.rows, "rsi_step")
    assert [s["proposer"] for s in steps] == ["llm"] * 4 and steps[0]["llm"] == r1["llm"]
    assert [s["proposer"] for s in bs.rsi_series(runtime.session, TASK)] == ["llm"] * 4
    assert any(m["text"].startswith("LLM 分析第") for m in doc["live"]["messages"])


def test_parse_unwraps_a_payload_nested_under_its_kind():
    """Seen live from DeepSeek: ``{"payload": {"executor": {"to": "alt"}}}``."""
    from scripts.evolve_llm import _parse
    ans = _parse(json.dumps({"kind": "executor", "payload": {"executor": {"to": "alt"}}, "summary": "换"}))
    assert ans["payload"] == {"to": "alt"} and ans["rationale"] == ""
    assert _parse(json.dumps({"decision": "patch", "summary": "改"}))["payload"] == {}   # call 1's bare decision
    with pytest.raises(ValueError, match="decision must be"):
        _parse(json.dumps({"goal": "a planner reply", "nodes": []}))


def test_dry_run_refuses_a_stub_that_is_no_executor(tmp_path, monkeypatch):
    from scripts import evolve_llm
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "stubcard").mkdir()
    (tmp_path / "stubcard" / "__init__.py").write_text(
        "class _P:\n    def make_driver(self, spec):\n        return object()\n\ndef provider(**p):\n    return _P()\n")
    why = evolve_llm.dry_run("stubcard:provider", {})
    assert why.startswith("doctor:stubcard:provider make_driver() returned object, not a StepExecutor")
    assert evolve_llm.dry_run("plugins.candidates.grasp_geometric_robocasa:provider", {}) is None


def recycle_cans_projection() -> tuple[dict, dict]:
    """The recycle_cans brief on a synthetic 'died at drop-can1 with reach_stall' round
    (the production shape, no simulator): the robocasa driver source + primitives ride."""
    from harness.manifest import discover
    from scripts import evolve_llm
    from scripts import harness_runtime as hr
    binding = discover().task_bindings["recycle_cans"]
    records = hr._binding_records(binding)
    seed = {"success": False, "first_death": "drop-can1", "failure_mode": "reach_stall", "keyframes": [],
            "fault": {"kind": "node_failure", "node": "drop-can1", "msg": "node 'drop-can1' failed"},
            "trail": [{"id": n, "ok": n != "drop-can1", "steps": 100, "failure_mode": None}
                      for n in ("nav-can1", "grasp-can1", "carry-can1", "drop-can1")],
            "nodes": {n: {"skill": n.replace("-", "_"), "success": n != "drop-can1", "executor": "scripted"}
                      for n in ("nav-can1", "grasp-can1", "carry-can1", "drop-can1")}}
    before = {"count": 0, "seeds": {"4243": seed, "4244": dict(seed)}}
    doc = {"task": "recycle_cans", "seeds": [4243, 4244], "cursor": 0, "rounds": [], "applied": {}}
    return evolve_llm.rsi_projection(doc, before, records, "robocasa", "scripted", binding,
                                     ["seed 4243 task.fault {...}"] * 10), before


def test_recycle_cans_brief_carries_the_drop_driver_source_and_stays_bounded(tmp_path):
    from scripts import evolve_llm
    proj, _ = recycle_cans_projection()
    assert proj["first_death"] == {**proj["first_death"], "skill": "drop_can1", "embodiment": "robocasa",
                                   "task": "drop_can1"}
    assert proj["card_template"]["ref"] == "plugins.candidates.<name>:provider"
    assert 'skill = "drop_can1"\nembodiment = "robocasa"' in proj["card_template"]["manifest.toml"]
    src = proj["scripted_driver_source"]
    assert src.index("class PointPlaceDriver") < src.index("class ClusterDropDriver(X.PointPlaceDriver)")
    assert proj["first_death"]["modules"] == ["plugins.embodiment_robocasa.recycle_driver", "plugins.embodiment_robocasa.stage_extras"]
    assert proj["primitives"]["ref"] == "plugins.embodiment_robocasa.drivers"
    assert proj["primitives"]["constants"]["ADIM"] == 12 and proj["primitives"]["constants"]["GRIP"] == 6
    assert any(k.startswith("_arm_action(env, goal_world, grip") for k in proj["primitives"]["functions"])
    assert proj["obs_keys"][0] == "robot0_base_pos" and "out[6] = a[11]" in proj["action_order"]
    assert len(json.dumps(proj, sort_keys=True)) <= evolve_llm.PROMPT_CHARS


@pytest.mark.skipif(not os.environ.get("PH_LLM_E2E"), reason="opt-in: one real DeepSeek call")
def test_real_deepseek_reads_the_recycle_cans_brief_within_budget(tmp_path):
    """One real round-1 call on the recycle_cans brief: the prompt fits ~12k tokens and the
    answer parses (any kind; a card is doctor-checked in the tmp candidates root)."""
    from scripts import evolve_llm
    monkey = pytest.MonkeyPatch()
    monkey.setattr(evolve_llm, "CANDIDATES_ROOT", tmp_path / "candidates")
    try:
        proj, before = recycle_cans_projection()
        ep = evolve_llm.endpoint()
        tried, row = evolve_llm.llm_propose(ep, proj, before, 1, tmp_path / "llm")
    finally:
        monkey.undo()
    print("usage", row["usage"], "reason", row["reason"], "tried", tried and tried["kind"])
    assert row["usage"] and row["usage"]["prompt"] <= 13000, row
    assert row["reason"] is None and row["summary"]


def _repeat_proj(history: list) -> tuple[dict, dict]:
    """A round whose first death is grab-0: two knobs, three bound executors, and the
    campaign history the model must not repeat."""
    proj = {"first_death": {"node": "grab-0", "skill": "grab", "executor": "scripted",
                            "executors": {"scripted": {}, "alt": {}, "geometric": {}},
                            "tunables": {"ref": "r", "path": ["tunables"],
                                         "values": {"hover_dz": 0.10, "stall_k": 40},
                                         "hints": {"reach_stall": ["hover_dz"]}}},
            "history": history, "this_round": {"per_seed": []}}
    before = {"seeds": {"1": {"first_death": "grab-0",
                              "nodes": {"grab-0": {"skill": "grab", "executor": "scripted"}}}}}
    return proj, before


def _fake(tmp_path, canned, name="canned.json"):
    from scripts import evolve_llm
    f = tmp_path / name
    f.write_text(json.dumps(canned))
    return evolve_llm.load_provider(evolve_llm.FAKE_REF, {"path": str(f)})


def test_a_knob_and_direction_the_campaign_already_tried_is_sent_back_with_what_is_left(tmp_path, monkeypatch):
    """drop_edge_margin 0.10->0.15 then 0.10->0.20 (same knob, same direction) is what the
    live model did despite the prompt rule: the harness rejects it through the repair loop."""
    from scripts import evolve, evolve_llm
    monkeypatch.setattr(evolve, "mount_params", lambda ref: {"tunables": {"hover_dz": 0.10, "stall_k": 40}})
    proj, before = _repeat_proj([{"round": 1, "tried": {"kind": "tunables", "node": "grab-0", "detail": {
        "ref": "r", "path": ["tunables", "hover_dz"], "from": 0.10, "to": 0.15}}}])
    ep = _fake(tmp_path, [
        {"kind": "tunables", "payload": {"ref": "r", "path": ["tunables", "hover_dz"], "to": 0.20},
         "summary": "再放大一点。", "rationale": "同一个 knob 同一个方向"},
        {"kind": "tunables", "payload": {"ref": "r", "path": ["tunables", "stall_k"], "to": 28},
         "summary": "换 stall_k。", "rationale": "换没试过的"}])
    tried, row = evolve_llm.llm_propose(ep, proj, before, 2, tmp_path / "llm")
    audit = json.loads((tmp_path / "llm" / "round-2.json").read_text())
    why = audit["attempts"][0]["reason"]
    assert "hover_dz up was already tried in this campaign (tried: hover_dz up)" in why
    assert "untried instead: hover_dz down, stall_k down, stall_k up" in why
    assert '"reach_stall": ["hover_dz"]' in why          # the hints say which knob to reach for
    assert why in audit["messages"][3]["content"]        # verbatim, through the existing repair loop
    # the second answer names a knob nothing tried: it is this round's try
    assert (tried["kind"], tried["detail"]["path"], tried["detail"]["to"]) == ("tunables", ["tunables", "stall_k"], 28)
    assert tried["detail"]["from"] == 40 and row["reason"] is None and len(audit["attempts"]) == 1


def test_an_executor_already_tried_is_rejected_this_round_too_and_the_round_ends_honestly(tmp_path):
    """An executor switch history already ran, then the same answer again after an unrelated
    rejection: the round's own attempts count as tried, so three answers exhaust the round."""
    from scripts import evolve_llm
    proj, before = _repeat_proj([{"round": 1, "tried": {"kind": "card", "node": "grab-0",
                                                        "detail": {"to": "alt", "ref": "c:provider"}}}])
    ans = lambda to, node=None: {"kind": "executor", "payload": {"to": to, **({"node": node} if node else {})},
                                 "summary": f"换 {to}。", "rationale": "-"}
    ep = _fake(tmp_path, [ans("alt"), ans("geometric", "nope"), ans("geometric")], name="exec.json")
    tried, _ = evolve_llm.llm_propose(ep, proj, before, 2, tmp_path / "llm")
    audit = json.loads((tmp_path / "llm" / "round-2.json").read_text())
    reasons = [a["reason"] for a in audit["attempts"]]
    assert "alt was already tried in this campaign (tried: alt)" in reasons[0]
    assert "untried instead: geometric" in reasons[0]
    assert "no node the suite ran ('nope')" in reasons[1]           # burns geometric for this round
    assert "geometric was already tried in this campaign (tried: alt, geometric)" in reasons[2]
    assert tried["kind"] == "none" and tried["detail"]["reason"].startswith("llm: 3 answers rejected")
