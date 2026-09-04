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

import importlib
import json
import os
import re

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
     "summary": "两颗种子都死在 grab-0。", "rationale": "先把 stall_k 调低",
     "layer": "parameter", "notes": "grab-0 的 stall_k 还没试过下调。"},
    _card("grab_llm", "llm", GOOD, ref="grab_other:provider"),      # round 2, attempt 1: ref outside the dir
    _card("grab_llm", "llm", GOOD),                                  # round 2, attempt 2: repaired
    _card("grab_stub", "stub", BAD_SHAPE, node="grab-0"),            # round 3, attempt 1: raises on the seed
    "not json at all",                                               # round 3, attempt 2
    _card("grab_stub", "stub2", BAD_SHAPE, node="grab-0"),           # round 3, attempt 3 (a payload
                                                                     # identical to attempt 1 would be
                                                                     # a repeat, not a third attempt)
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
    # the diagnosis rides the round row (and the sealed step): which layer, what it taught
    assert (r1["layer"], r1["notes"]) == ("parameter", "grab-0 的 stall_k 还没试过下调。")
    assert r1["tried"]["detail"]["layer"] == "parameter"
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
    # the card came WITH its payload on call 1 (no material asked for): the rejection carries
    # the material the model wrote blind, inserted after the system message
    assert a2["calls"] == 2 and [m["role"] for m in a2["messages"]] == ["system", "user", "user", "assistant", "user"]
    assert a2["messages"][1]["content"].startswith("Materials (static):") and "executor_contract" in a2["messages"][1]["content"]
    assert a2["attempts"][0]["reason"] in a2["messages"][4]["content"] and a2["raw"] == json.dumps(CANNED[2])
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
    assert (steps[0]["layer"], steps[0]["notes"]) == ("parameter", r1["notes"])
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


def recycle_cans_projection(doc_extra=None, trace=None) -> tuple[dict, dict]:
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
                      | ({"trace": trace} if trace and n == "drop-can1" else {})
                      for n in ("nav-can1", "grasp-can1", "carry-can1", "drop-can1")],
            "nodes": {n: {"skill": n.replace("-", "_"), "success": n != "drop-can1", "executor": "scripted"}
                      for n in ("nav-can1", "grasp-can1", "carry-can1", "drop-can1")}}
    before = {"count": 0, "seeds": {"4243": seed, "4244": dict(seed)}}
    doc = {"task": "recycle_cans", "seeds": [4243, 4244], "cursor": 0, "rounds": [], "applied": {},
           **(doc_extra or {})}
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
    # `functions` is what an edit's `old` is copied from: every class of every editable
    # module, not just the target node's stage (the live model invented a snippet for the
    # drop stage while the target node was nav, because only nav's methods were here)
    import inspect

    from plugins.embodiment_robocasa import stage_extras
    fns = proj["functions"]
    # the key names the module to send as `module` (round 102 sent PointPlaceDriver._act
    # verbatim against the wrong one), and the source carries no "NNNN| " prefix to strip
    x = "plugins.embodiment_robocasa.stage_extras:"
    assert {"plugins.embodiment_robocasa.recycle_driver:ClusterDropDriver._drop_point",
            x + "PointPlaceDriver.act"} <= set(fns)
    assert fns[x + "PointPlaceDriver.act"] == inspect.getsource(stage_extras.PointPlaceDriver.act)
    assert not re.match(r"\s*\d+\| ", fns[x + "PointPlaceDriver.act"])
    assert len(json.dumps(proj, sort_keys=True)) <= evolve_llm.PROMPT_CHARS


@pytest.mark.skipif(not os.environ.get("PH_LLM_E2E"), reason="opt-in: one real DeepSeek call")
def test_real_deepseek_reads_the_recycle_cans_brief_within_budget(tmp_path):
    """One real round-1 call on the recycle_cans brief: the prompt fits the budget and the
    answer parses (any kind; a card is doctor-checked in the tmp candidates root)."""
    from scripts import evolve_llm
    from scripts.evolve_llm import PROMPT_CHARS
    monkey = pytest.MonkeyPatch()
    monkey.setattr(evolve_llm, "CANDIDATES_ROOT", tmp_path / "candidates")
    try:
        proj, before = recycle_cans_projection()
        ep = evolve_llm.endpoint()
        tried, row = evolve_llm.llm_propose(ep, proj, before, 1, tmp_path / "llm")
    finally:
        monkey.undo()
    print("usage", row["usage"], "reason", row["reason"], "tried", tried and tried["kind"])
    # call 1 is the brief alone (~6k); a card / patch decision adds call 2's materials, which
    # carry the full text of every editable module -- PROMPT_CHARS/2.5 tokens at the ceiling
    # (measured 42k on the production round: brief + materials + the repair)
    assert row["usage"] and row["usage"]["prompt"] <= PROMPT_CHARS // 2, row
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


def test_none_is_rejected_while_something_is_untried_and_accepted_on_the_third_answer(tmp_path, monkeypatch):
    """The live model answered none twice while the brief still offered untried knobs: none
    now comes back with the concrete list, and an answer off that list is the round's try."""
    from scripts import evolve, evolve_llm
    monkeypatch.setattr(evolve, "mount_params", lambda ref: {"tunables": {"hover_dz": 0.10, "stall_k": 40}})
    proj, before = _repeat_proj([])
    proj["first_death"]["modules"] = ["plugins.x.mod"]
    none = {"kind": "none", "payload": {}, "summary": "没辙了。", "rationale": "参数都不管用"}
    ep = _fake(tmp_path, [none,
                          {"kind": "tunables", "payload": {"ref": "r", "path": ["tunables", "stall_k"], "to": 28},
                           "summary": "试 stall_k。", "rationale": "还没试过"}], name="none.json")
    tried, _ = evolve_llm.llm_propose(ep, proj, before, 2, tmp_path / "llm")
    audit = json.loads((tmp_path / "llm" / "round-2.json").read_text())
    why = audit["attempts"][0]["reason"]
    assert "none is only allowed when nothing is left to try" in why and "untried on grab-0" in why
    assert "tunables hover_dz down, tunables hover_dz up, tunables stall_k down, tunables stall_k up" in why
    assert "executor alt, executor geometric, patch plugins.x.mod" in why
    assert why in audit["messages"][3]["content"]        # the existing repair loop carries it back
    assert audit["brief"]["untried"][0] == "tunables hover_dz down"   # call 1 saw the same list
    assert (tried["kind"], tried["detail"]["path"]) == ("tunables", ["tunables", "stall_k"])
    assert len(audit["attempts"]) == 1


def test_none_is_accepted_at_once_when_nothing_is_left_to_try(tmp_path):
    from scripts import evolve_llm
    proj, before = _repeat_proj([])
    proj["first_death"] |= {"executors": {"scripted": {}}, "tunables": {"ref": "r", "path": [], "values": {}}}
    ep = _fake(tmp_path, [{"kind": "none", "payload": {}, "summary": "都过了。", "rationale": "无事可试"}],
               name="empty.json")
    tried, _ = evolve_llm.llm_propose(ep, proj, before, 2, tmp_path / "llm")
    audit = json.loads((tmp_path / "llm" / "round-2.json").read_text())
    assert audit["brief"]["untried"] == [] and audit["attempts"] == [] and audit["calls"] == 1
    assert tried["kind"] == "none" and tried["detail"]["reason"] == "llm: 无事可试"


def test_three_nones_are_taken_honestly_on_the_third_attempt(tmp_path):
    """The model insists: rejected twice (an explanation is still an answer), accepted third."""
    from scripts import evolve_llm
    proj, before = _repeat_proj([])
    none = lambda why: {"kind": "none", "payload": {}, "summary": "没辙。", "rationale": why}
    ep = _fake(tmp_path, [none("参数都不管用"), none("每个 knob 都解释过了"), none("坚持 none")], name="3none.json")
    tried, _ = evolve_llm.llm_propose(ep, proj, before, 2, tmp_path / "llm")
    audit = json.loads((tmp_path / "llm" / "round-2.json").read_text())
    assert len(audit["attempts"]) == 2 and audit["calls"] == 3
    assert tried["kind"] == "none" and tried["detail"]["reason"] == "llm: 坚持 none"


def test_an_unbounded_run_waits_instead_of_stopping_and_only_cancel_ends_it(tmp_path):
    """The console's 开始/继续 sends no ``rounds``: the loop must never end itself.

    Two model answers, then nothing but ``none`` -- a bounded run would stop after
    MAX_NONE; unbounded it enters ``waiting`` (backoff, live phase + message) and
    keeps the campaign ``running`` until the operator's cancel marker lands.
    """
    from test_mission_e2e import SESSION, _wait

    canned = [CANNED[0]] + [{"kind": "none", "payload": {},
                             "summary": "没有可试的。", "rationale": "都试过了"}] * 12
    runs = tmp_path / "runs"
    runs.mkdir()
    rt = _Runtime(runs, card=_CARD, canned=canned, mode="evolution",
                  env={"PH_CANDIDATES_ROOT": str(runs / "candidates"),
                       "PH_NONE_BACKOFF_S": "3"})   # 3 s, 6 s, ... instead of 60 s
    campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    try:
        bs.submit_brief(runs, json.dumps({"kind": "evolve", "task": TASK,
                                          "seeds": [1, 2], "arm": "auto"}), session=SESSION)
        doc = lambda: json.loads(campaign.read_text()) if campaign.exists() else {}
        _wait(lambda: (doc().get("live") or {}).get("phase") == "waiting", 120,
              "the loop to throttle instead of stopping")
        d = doc()
        assert d["status"] == "running", d["status"]          # never done on its own
        assert "按停止结束" in d["live"]["message"], d["live"]["message"]
        assert d["live"]["wait_s"] >= 3 and d["live"]["nones"] >= 1
        brief = sorted(p.name for p in (rt.session / "processing").iterdir())[0]
        assert bs.cancel_brief(rt.session, brief).get("error") is None
        _wait(lambda: (rt.session / "cancelled" / brief).exists(), 60, "cancel to land")
        # the read says stopped even if the loop was killed before it could write
        assert bs.rsi_run(rt.session, TASK)["status"] in ("cancelled", "stopped")
        assert bs.rsi_run(rt.session, TASK)["open_brief"] is None
    finally:
        rt.stop()


# ── Zetta: the layer ladder, the failure clusters and the lab notebook ────────────

def _exhausted_proj(extra_history=()):
    """grab-0 with every (knob, direction) already tried: the brief's ``exhausted`` flag is
    set, so the parameter layer is closed there."""
    hist = [{"round": i, "tried": {"kind": "tunables", "node": "grab-0", "detail": {
        "ref": "r", "path": ["tunables", k], "from": 1.0, "to": to}}}
        for i, (k, to) in enumerate((("hover_dz", 2.0), ("hover_dz", 0.5),
                                     ("stall_k", 2.0), ("stall_k", 0.5)), 1)]
    return _repeat_proj([*hist, *extra_history])


def test_a_parameter_layer_answer_is_refused_once_the_knobs_are_exhausted(tmp_path):
    """Zetta's top-down rule: with the parameter layer closed the answer must come from a
    higher one -- the rejection names them, and a state-layer answer is the round's try."""
    from scripts import evolve_llm
    proj, before = _exhausted_proj()
    assert "parameter layer is CLOSED" in evolve_llm.brief(proj)["exhausted"]
    ep = _fake(tmp_path, [
        {"kind": "tunables", "payload": {"ref": "r", "path": ["tunables", "hover_dz"], "to": 0.2},
         "summary": "再调 hover_dz。", "rationale": "参数", "layer": "parameter"},
        {"kind": "executor", "payload": {"to": "alt"}, "summary": "换执行器。", "rationale": "几何不对",
         "layer": "state", "notes": "drop 点在底盘 1.0 m 外，抓取段底盘不能动。"}], name="layer.json")
    tried, _ = evolve_llm.llm_propose(ep, proj, before, 2, tmp_path / "llm")
    why = json.loads((tmp_path / "llm" / "round-2.json").read_text())["attempts"][0]["reason"]
    assert "the parameter layer is CLOSED on grab-0" in why
    assert "evaluation, plan, state, recovery" in why and "is the predicate / oracle right" in why
    assert "hover_dz up, hover_dz down" not in why or "already tried" in why
    # the higher-layer answer is taken, with its layer and its notebook note on the try
    assert (tried["kind"], tried["detail"]["to"]) == ("executor", "alt")
    assert tried["detail"]["layer"] == "state"
    assert tried["detail"]["notes"] == "drop 点在底盘 1.0 m 外，抓取段底盘不能动。"


def test_a_tunables_answer_is_a_parameter_answer_even_unlabelled(tmp_path):
    """The live model does not label its layer: a knob change IS the parameter layer."""
    from scripts import evolve_llm
    proj, before = _exhausted_proj()
    ep = _fake(tmp_path, [{"kind": "tunables", "payload": {"ref": "r", "path": ["tunables", "stall_k"], "to": 9},
                           "summary": "调 stall_k。", "rationale": "-"}] * 3, name="unlabelled.json")
    tried, _ = evolve_llm.llm_propose(ep, proj, before, 2, tmp_path / "llm")
    audit = json.loads((tmp_path / "llm" / "round-2.json").read_text())
    assert "the parameter layer is CLOSED on grab-0" in audit["attempts"][0]["reason"]
    assert tried["kind"] == "none" and tried["detail"]["reason"].startswith("llm: ")


def test_an_invalid_layer_is_rejected_and_the_ladder_named():
    from scripts.evolve_llm import _parse
    with pytest.raises(ValueError, match="layer must be evaluation|plan|state|recovery|parameter"):
        _parse(json.dumps({"kind": "none", "payload": {}, "summary": "x", "layer": "vibes"}))


def test_the_brief_carries_the_clusters_the_first_missing_milestone_the_divergence_and_the_notebook():
    """The recycle_cans round as Zetta reads it: each seed a milestone chain indexed by its
    first missing milestone, the seeds clustered by (milestone, failure_mode), the numeric
    divergence from the successful reference at that milestone, and the last rounds' notes."""
    from scripts import evolve_llm
    trace = {"start": {"d_base_target": 1.028, "d_eef_target": 0.626, "step": 1},
             "stall": {"d_base_target": 1.028, "d_eef_target": 0.572, "step": 65},
             "end": {"d_base_target": 1.028, "d_eef_target": 0.574, "step": 65}}
    notes = [{"round": i, "notes": f"第 {i} 轮：调参没用。"} for i in range(1, 13)]
    proj, _ = recycle_cans_projection(
        {"reference": {"drop-can1": {"d_base_target": 0.35, "d_eef_target": {"mean": 0.05}}},
         "rounds": [{"round": n["round"], "tried": {"kind": "none", "node": "drop-can1", "detail": {}},
                     "before": 0, "after": 0, "published": False, "notes": n["notes"]} for n in notes]},
        trace=trace)
    assert proj["clusters"] == [{"milestone": "drop-can1", "failure_mode": "reach_stall",
                                 "seeds": [4243, 4244], "size": 2}]
    assert proj["target"]["cluster"] == proj["clusters"][0]
    row = proj["this_round"]["per_seed"][0]
    assert row["first_missing_milestone"] == "drop-can1"
    assert row["divergence"]["d_base_target"] == {"seed": 1.028, "reference": 0.35, "delta": 0.678}
    assert row["divergence"]["d_eef_target"]["delta"] == 0.524   # a distribution row: its mean
    # a notebook row is 假设+观测：the note plus what the round tried, measured and was judged
    assert len(proj["notebook"]) == 10 and [n["round"] for n in proj["notebook"]] == list(range(3, 13))
    assert proj["notebook"][0] == {**notes[2], "tried": "none drop-can1",
                                   "measured": None, "verdict": None}
    assert proj["layers"]["plan"].startswith("is the node graph right")
    b = evolve_llm.brief(proj)   # call 1 sees all of it
    assert b["clusters"] == proj["clusters"] and b["notebook"] == proj["notebook"]
    assert b["this_round"]["per_seed"][0]["divergence"] == row["divergence"]
    assert b["layers"] == proj["layers"] and b["target"]["cluster"] == proj["clusters"][0]


def test_the_brief_renders_the_gradient_the_accepted_stack_and_what_score_means():
    """Round 86 of the live campaign: the model wrote the right structural fix, both seeds
    then died EARLIER (drop-can1 -> nav-can1), the success count stayed 0/2 and the round was
    filed "same". Successes are flat for whole campaigns, so the brief now says out loud what
    moved, what is already accepted, and that milestones are the gradient."""
    from scripts import evolve_llm
    chain = [{"id": n} for n in ("nav-can1", "grasp-can1", "carry-can1", "drop-can1")]
    per = lambda dead: [{"seed": s, "success": False, "first_death": dead, "nodes": chain}
                        for s in (4243, 4244)]
    proj, _ = recycle_cans_projection({"rounds": [
        {"round": 85, "tried": {"kind": "card", "node": "drop-can1", "detail": {"layer": "plan", "to": "fix"}},
         "before": 0, "after": 0, "published": True, "score": {"before": [0, 9, 0], "after": [0, 11, 0]},
         "per_seed": per("nav-can1"), "after_seeds": per("drop-can1")},
        {"round": 86, "tried": {"kind": "patch", "node": "drop-can1", "detail": {"layer": "plan"}},
         "before": 0, "after": 0, "published": False, "score": {"before": [0, 11, 0], "after": [0, 9, 0]},
         "per_seed": per("drop-can1"), "after_seeds": per("nav-can1")}]})
    b = evolve_llm.brief(proj)
    assert b["last_outcome"] == (
        "上一轮（第 86 轮）：plan patch，score (0, 11, 0) → (0, 9, 0)，"
        "种子 4243 的死亡点从 drop-can1 前移到 nav-can1 = 变差；"
        "种子 4244 的死亡点从 drop-can1 前移到 nav-can1 = 变差（未接受，本轮仍从已接受状态出发）")
    assert "唯一的梯度就是里程碑位" in b["score_definition"]
    # the accepted stack: the published round, named as the baseline this round starts from
    assert b["accepted_stack"]["changes"] == [{"round": 85, "kind": "card", "node": "drop-can1", "to": "fix"}]
    assert b["accepted_stack"]["note"].startswith("已接受的改动（本轮从它们之上出发")
    rules = evolve_llm._RULES
    assert "PREFER A CHANGE THAT ADVANCES THE FURTHEST-REACHED MILESTONE" in rules
    assert "NEVER trade a node that already passes for the target node" in rules
    assert "proposing one of them again is refused" in rules and "functions[<module>:<Class>.<method>]" in \
        evolve_llm.PAYLOAD_BY_KIND["patch"]["edits"][0]["old"] + rules


def test_first_missing_milestone_is_the_first_node_not_completed():
    from scripts.evolve_llm import first_missing
    trail = [{"id": "nav", "ok": True}, {"id": "recover-drop", "ok": True},
             {"id": "drop", "ok": False}, {"id": "placed", "ok": None}]
    assert first_missing(trail) == "drop" and first_missing([{"id": "a", "ok": True}]) is None
    assert first_missing([]) is None


def test_the_first_death_row_keeps_its_geometry_and_upstream_for_the_brief():
    """The causal evidence rides the seed row, not only the trace: without ``geometry``
    (d_base_point vs the arm's reach_max) and ``upstream`` (the segment that parked the
    base there) "the base never moved" is a fact with no author and no reach to compare
    it against. Both are on the first-death row alone. ``failure_mode`` rides only when
    the row HAS the key -- a node whose executor sealed none must not read as "no stall"."""
    from scripts.evolve_llm import _seed_row
    geo = {"d_base_point": 1.028, "reach_max": 0.664}
    up = {"node": "carry-can1", "skill": "carry", "steps": 174, "trace_end": {"d_base_target": 0.644}}
    s = {"first_death": "drop-can1", "trail": [
        {"id": "carry-can1", "ok": True, "steps": 174, "failure_mode": None,
         "trace_end": {"d_base_target": 0.644}},
        {"id": "drop-can1", "ok": False, "steps": 65, "trace": {"end": {}},
         "geometry": geo, "upstream": up}]}
    trail = _seed_row(4243, s, {})["trail"]
    assert set(trail[0]) == {"id", "ok", "steps", "failure_mode"}
    assert set(trail[1]) == {"id", "ok", "steps", "trace", "geometry", "upstream"}   # no key
    assert trail[1]["geometry"] == geo and trail[1]["upstream"] == up


# ── the experiment report: absence is a reading, and the notebook carries observations ──

def _round_131() -> dict:
    """Round 131 of runs/session-robocasa-rsi/campaigns/evolve-recycle_cans, verbatim
    (tests/fixtures/evolve_recycle_rounds.json): the candidate ran on 4243 and left NO
    per-step series (d_eef_min_after null -- 363 of the campaign's 365 trial rows are this),
    and on 4244 drop-can1 was never reached at all (the seed dies at nav-can1)."""
    from pathlib import Path
    return json.loads((Path(__file__).parent / "fixtures"
                       / "evolve_recycle_rounds.json").read_text())["131"]


def test_the_trial_line_says_measured_nothing_instead_of_inventing_a_divergence():
    """378 rounds rendered 14 distinct sentences and not ONE distance number: ``if num(a)``
    dropped the before value with the missing after one, and ``_divergent`` on an empty
    after-series returned the baseline's first step -- "第 1 步起与基线不同" in 355 of the
    365 trial rows, about steps nobody measured. The absence itself is the finding."""
    from scripts.evolve_llm import _trial_line
    line = _trial_line(_round_131()["trial_evidence"])
    a, b = line.split("；")
    assert "第 1 步起与基线不同" not in line          # nothing was measured to diverge
    assert "d_eef 最小 基线 0.430 → 本轮测不到" in a  # the before value survives the absence
    assert "d_base 最小 基线 1.028 → 本轮测不到" in a
    assert "跑到第 117 步（基线 65 步）" in a and "没留下逐步 trace，是测不到、不是没变化" in a
    # 4244 never reached drop-can1: 8 rounds told the model its change "had no effect"
    assert "种子 4244" in b and "根本没执行到" in b and "你的改动没有生效" not in line


def test_the_line_leads_with_the_failure_mode_change_and_the_notebook_pairs_it_with_the_note():
    """A failure_mode transition is a reading only when the after side ANSWERED. With the
    key present (the candidate executor reported its own None) the line leads with the
    change; with the key ABSENT (``InprocExecutor.diagnostics`` -> {}) it must say 测不到.
    Over evolve-recycle_cans' 588 rounds the judged node read None on 364 of the 365
    candidate trial rows -- 347 at the segment cap -- while the scripted baseline rows
    carried the stall in 580 rounds, so an unconditional "reach_stall→无" told every
    candidate round the stall was cured. And a notebook row is a HYPOTHESIS: 10 rounds in a row wrote almost
    the same sentence with no observation next to it, so each row now carries what was
    tried, what was measured and how the round was judged."""
    from scripts.evolve_llm import _notebook, _trial_line
    diff = {"steps_before": 65, "steps_after": 300, "base_moved": False,
            "d_eef_min_before": 0.430, "d_eef_min_after": 0.021,
            "phase_changed": [], "first_divergent_step": 12,
            "ok_after": False, "failure_mode_before": "reach_stall",
            "failure_mode_after": None}
    ev = {"node": "drop-can1", "exception": None, "seeds": [{"seed": 4243, "diff": diff}]}
    assert _trial_line(ev).startswith("种子 4243 在 drop-can1 failure_mode reach_stall→无，跑到第 300 步")
    mute = {k: v for k, v in diff.items() if k != "failure_mode_after"}   # nobody answered
    line = _trial_line({**ev, "seeds": [{"seed": 4243, "diff": mute}]})
    assert "failure_mode 基线 reach_stall → 本轮测不到" in line and "→无" not in line
    r = _round_131()
    rounds = [{"round": 131, "notes": r["notes"], "trial_evidence": r["trial_evidence"],
               "tried": {"kind": "card", "node": r["tried_node"]},
               "accepted_reason": r["was"]["reason"]},
              {"round": 132, "notes": "同一个结论又写了一遍。", "tried": {"kind": "none", "node": None}},
              {"round": 133, "tried": {"kind": "patch", "node": "nav-can1"}}]   # no note: no row
    nb = _notebook(rounds)
    assert [n["round"] for n in nb] == [131, 132]
    assert nb[0]["tried"] == "card drop-can1" and "根本没执行到" in nb[0]["measured"]
    assert nb[0]["verdict"] == "regressed: 4243/recover-drop-can1"
    assert nb[1]["measured"] is None and nb[1]["verdict"] is None


def test_the_history_carries_what_the_model_said_and_how_the_round_ended():
    """573 of 588 rounds had ``parent: 0`` and the history rows named only the knob/path:
    the model could not see its own earlier answers, so 77 none rounds were refused as
    "repeated the same rejected answer". ``llm.summary`` and ``outcome`` are already on
    the index row -- projecting them costs no new storage."""
    proj, _ = recycle_cans_projection({"rounds": [
        {"round": 1, "tried": {"kind": "patch", "node": "drop-can1", "detail": {"module": "m"}},
         "before": 0, "after": 0, "published": False, "outcome": "same",
         "accepted_reason": "no score change", "llm": {"summary": "把投放点夹进臂展。"}}]})
    h, = proj["history"]
    assert h["summary"] == "把投放点夹进臂展。" and h["outcome"] == "same"
    assert h["verdict"] == "no score change"


class _Truncated:
    """An endpoint whose answer ran out of room mid-string. ``last_finish`` is the endpoint
    SAYING so; ``last_usage`` at the cap is the approximation for one that does not."""
    identity, images = "fake:truncated", False
    last_usage, last_finish = {"prompt": 10, "completion": 4096}, None

    def __init__(self, usage=..., finish=None):
        if usage is not ...:
            self.last_usage = usage
        self.last_finish = finish

    def chat(self, messages, **kw):
        return '{"kind": "patch", "summary": "夹紧投放点", "payload": {"name": "p", "modu'


def test_the_validation_layer_stops_burning_attempts_on_its_own_bookkeeping(tmp_path, monkeypatch):
    """The five ways a live round lost an attempt to nothing but our own accounting:
    a patch written before any source was shown (108 attempt-0 refusals, 18.6M tokens),
    a snippet copied right but pointed at the wrong module (46 of the 136 "occurs nowhere"
    refusals have their ``old`` sitting exactly once in a sibling module),
    a missing one-line ``summary`` (22 attempts / 11 rounds), a bookkeeping id in
    ``stuck.tried`` (~185 of 205 rows), and a reply cut off at ``max_tokens`` (24 rounds)."""
    from scripts import evolve_llm

    # 1. a patch always gets the material first, however complete its payload looks
    assert evolve_llm._needs_material({"kind": "patch", "payload": {
        "name": "p", "module": "m", "edits": [{"old": "a", "new": "b"}]}}) is True
    assert evolve_llm._needs_material({"kind": "card", "payload": {"name": "c", "files": {}}}) is False

    # 2. the right code, the wrong module -- and retyped, so the exact-once test misses it
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "modhere.py").write_text("def here():\n    return 1\n")
    (tmp_path / "modthere.py").write_text(
        "class T:\n    def act(self, obs):\n        self._n += 1\n        return obs\n")
    importlib.invalidate_caches()
    fd = {"modules": ["modhere", "modthere"]}
    retyped = "def act(self, obs):\n  self._n += 1\n  return obs"   # the model's own indentation
    assert (tmp_path / "modthere.py").read_text().count(retyped) == 0
    why = evolve_llm._elsewhere([{"old": retyped, "new": "x"}], "modhere", fd)
    assert "Three consecutive lines" in why and "MAY belong there" in why
    assert "send `module`: 'modthere'" in why and "def act(self, obs):" in why
    assert "WITHOUT the `NNNN| ` line-number prefixes" in why   # the text it shows IS numbered
    assert evolve_llm._elsewhere([{"old": "def nowhere(self):", "new": "x"}], "modhere", fd) == ""
    # ONE line in common is not evidence of another module: retyped with a tab, so it is
    # not even a substring of modthere, the old any-single-line fallback named it anyway
    assert evolve_llm._elsewhere([{"old": "\treturn obs", "new": "x"}], "modhere", fd) == ""

    # 3. a missing summary is filled in, not sent back
    ans = evolve_llm._parse(json.dumps({"kind": "executor", "payload": {"to": "alt"},
                                        "rationale": "  换成 geometric 执行器  "}))
    assert ans["summary"] == "换成 geometric 执行器"
    assert evolve_llm._parse(json.dumps({"kind": "none", "payload": {}}))["summary"] == "none"

    # 4. stuck.tried keeps the real switches and drops the per-round patch ids
    rounds = [{"round": i, "before": [0, 0, 0], "after": [0, 0, 0], "published": False,
               "outcome": "same", "tried": {"kind": "executor", "node": "drop-can1",
                                            "detail": {"to": to}}}
              for i, to in enumerate([f"patch_r{n}" for n in range(1, 7)] + ["alt_drop"], 1)]
    proj, _ = recycle_cans_projection({"rounds": rounds, "cursor": len(rounds)})
    assert proj["stuck"]["tried"] == ["executor alt_drop"]

    # 5. a reply cut off at the cap says so, instead of "Unterminated string"
    p2, before = _exhausted_proj()
    tried, _ = evolve_llm.llm_propose(_Truncated(), p2, before, 3, tmp_path / "llm", max_tokens=4096)
    audit = json.loads((tmp_path / "llm" / "round-3.json").read_text())
    assert "被截断" in audit["attempts"][0]["reason"]
    assert tried["kind"] == "none" and "被截断" in tried["detail"]["reason"]
    # ...and finish_reason says it even when the endpoint reports no usage at all, which
    # the token count alone reads as "not truncated"
    ep = _Truncated(usage=None, finish="length")
    tried, _ = evolve_llm.llm_propose(ep, p2, before, 4, tmp_path / "llm", max_tokens=4096)
    assert "被截断" in tried["detail"]["reason"]
    assert "被截断" not in evolve_llm.llm_propose(
        _Truncated(usage=None), p2, before, 5, tmp_path / "llm",
        max_tokens=4096)[0]["detail"]["reason"]
