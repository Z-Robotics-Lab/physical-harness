"""Explicit model replies through the real evolve runtime and fake robot world.

The model inspects parameters/source, measures a neutral parameter change, repairs
an invalid card reference, and earns paired development acceptance with an action
that changes the independently verified world. Two later malformed executors fail
on real action-shape checks before the model stops; no completed after suite is
invented. Source and history inspections preserve exact code and missing measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_evolve_e2e import _CARD, EMB, TASK
from test_mission_e2e import _kinds, _Runtime

from board import store as bs
from scripts.evolve_evidence import compact_brief, inspect_evidence

_EXEC = ("from harness.skill_executor import InprocExecutor\n\n\n"
         "class _E(InprocExecutor):\n    def bind(self, env, target=None):\n        pass\n\n"
         "    def act(self, obs):\n        return {act}\n\n\n"
         "class _P:\n    def make_driver(self, spec):\n        return _E()\n\n\n"
         "def provider(**params):\n    return _P()\n")
GOOD, BAD_SHAPE = _EXEC.format(act="(2.0,)"), _EXEC.format(act="(0.0, 0.0)")
MANIFEST = (f'needs_sim = true\n[executors.{{to}}]\nskill = "grab"\nembodiment = "{EMB}"\n'
            'ref = "{name}:provider"\ntransport = "inproc"\n')


def _card(name, to, code, ref=None, node="grab-0"):
    pay = {"name": name, "to": to, "ref": ref or f"{name}:provider",
           "files": {"manifest.toml": MANIFEST.format(to=to, name=name), "__init__.py": code}}
    if node:
        pay["node"] = node
    return {"kind": "card", "payload": pay, "summary": f"写执行器 {name}。", "rationale": "code-as-policy"}


CANNED = [
    {"op": "inspect", "args": {"view": "parameter", "node": "grab-0", "parameter": "stall_k"}},
    {"kind": "tunables", "payload": {"ref": "test_evolve_e2e:policy_provider",
                                     "path": ["tunables", "stall_k"], "to": 28, "node": "grab-0"},
     "summary": "两颗种子都死在 grab-0。", "rationale": "先把 stall_k 调低",
     "layer": "parameter", "notes": "grab-0 的 stall_k 还没试过下调。"},
    {"op": "inspect", "args": {"view": "source", "node": "grab-0", "module": "test_evolve_e2e", "symbol": "_Driver"}},
    _card("grab_llm", "llm", GOOD, ref="grab_other:provider"),      # round 2, attempt 1: ref outside the dir
    _card("grab_llm", "llm", GOOD),                                  # round 2, attempt 2: repaired
    {"op": "inspect", "args": {"view": "source", "node": "grab-0", "module": "grab_llm", "symbol": "_E"}},
    _card("grab_stub", "stub", BAD_SHAPE, node="grab-0"),            # round 3, attempt 1: raises on the seed
    "not json at all",                                               # round 3, attempt 2
    _card("grab_stub2", "stub2", BAD_SHAPE, node="grab-0"),          # round 3, attempt 3: fresh immutable
                                                                     # artifact name for the new manifest
    {"kind": "none", "payload": {}, "summary": "没有值得试的。", "rationale": "两颗种子都过了"},   # round 4
]


def _params_card(runs, ref="test_evolve_e2e:policy_provider", nested=True):
    """Declare the fake's legal knob through the same plugin mount seam."""
    path = Path(runs) / "plugins" / "params"
    path.mkdir(parents=True)
    section = '[mounts."policy.driver".params' + ('.tunables]' if nested else ']')
    (path / "manifest.toml").write_text(
        'enabled = false\n[mounts."policy.driver"]\n' + f'ref = "{ref}"\n' + section + '\nstall_k = 40\n')


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    _params_card(runs)
    rt = _Runtime(runs, card=_CARD, canned=CANNED, mode="evolution",
                  env={"PH_CANDIDATES_ROOT": str(runs / "candidates")})
    rt.campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    try:
        rt.rows = rt.run({"kind": "evolve", "task": TASK, "seeds": [1, 2], "rounds": 6, "arm": "auto",
                          "confirm_seeds": 0, "max_model_calls": 12,
                          "max_input_bytes": 200000, "max_probe_episodes": 4})[1]
        yield rt
    finally:
        rt.stop()


def test_llm_answers_drive_the_rounds_repair_from_the_exact_error_and_stop_honestly(runtime):
    doc = json.loads(runtime.campaign.read_text())
    assert doc["status"] == "done" and doc["cursor"] == 3 and len(doc["rounds"]) == 3
    r1, r2, r3 = doc["rounds"]
    assert (r1["proposer"], r1["tried"]["kind"], r1["tried"]["node"]) == ("llm", "tunables", "grab-0")
    assert (r1["tried"]["detail"]["path"], r1["tried"]["detail"]["to"]) == (["tunables", "stall_k"], 28)
    assert r1["llm"]["summary"] == "两颗种子都死在 grab-0。" and r1["llm"]["rationale"] == "先把 stall_k 调低"
    assert (r1["before"], r1["after"], r1["published"]) == (0, 0, False)
    audits = {p.name: json.loads(p.read_text()) for p in (runtime.campaign.parent / "llm").glob("round-*.json")}
    assert set(audits) == {f"round-{r}.json" for r in (1, 2, 3)}
    a1, a2, a3 = (audits[f"round-{r}.json"] for r in (1, 2, 3))
    assert [a["calls"] for a in (a1, a2, a3)] == [2, 3, 5]
    assert all(a["materials"] == {} for a in (a1, a2, a3))
    assert a1["brief"]["driver_index"]["grab-0"]["skill"] == "grab"
    for audit in (a1, a2, a3):
        assert len(audit["prompt_sha"]) == len(audit["raw_sha"]) == 64
        assert audit["model"].startswith("fake(")
        for request in audit["requests"]:
            assert [m["role"] for m in request["messages"]] == ["system", "user"]
            assert "module_sources" not in json.loads(request["messages"][1]["content"])["state"]

    # A real source read is scoped to the selected node; a rejected reference
    # returns to this same model before it produces the valid copied artifact.
    source = next(event["result"]["data"] for event in a2["events"]
                  if event.get("view") == "inspect")
    assert source["module"] == "test_evolve_e2e" and "class _Driver:" in source["code"]
    assert "reference_card" not in source and "GeometricGraspExecutor" not in json.dumps(source)
    assert len(a2["attempts"]) == 1
    reason = a2["attempts"][0]["reason"]
    assert reason.startswith("doctor:ref 'grab_other:provider'")
    assert "must name a provider inside grab_llm" in reason
    assert reason in json.dumps(a2["requests"][-1]["messages"], ensure_ascii=False)
    assert (r2["tried"]["kind"], r2["tried"]["detail"]["to"]) == ("card", "llm")
    assert (r2["before"], r2["after"], r2["accepted"], r2["published"]) == (0, 2, True, False)
    assert doc["applied"]["cards"]["llm"]["ref"] == "grab_llm:provider"
    assert "digest" not in r2["tried"]["detail"]
    assert r2["learning"]["probes"][0]["accepted"] is False
    assert r2["learning"]["full_evaluations"] == 1

    # The two malformed executors really reach the environment shape check.
    # The intervening parse failure spends a model call, never an episode.
    reasons = [attempt["reason"] for attempt in a3["attempts"]]
    assert len(reasons) == 3
    assert "expected a 1-dim action" in reasons[0] and "expected a 1-dim action" in reasons[2]
    assert "no JSON object" in reasons[1]
    assert all(reason in json.dumps(a3["requests"][i + 2]["messages"], ensure_ascii=False)
               for i, reason in enumerate(reasons))
    assert len(r3["learning"]["probes"]) == 2
    assert all(p["error"] and p["accepted"] is False for p in r3["learning"]["probes"])
    assert r3["llm"]["status"] == "abstained" and r3["llm"]["stop_reason"] == "model_stop"
    assert r3["tried"]["kind"] == "none" and r3["trial"] is None
    assert (r3["before"], r3["after"], r3["published"], r3["parent"]) == (2, None, False, 2)
    assert (runtime.runs / "candidates" / "grab_stub" / "__init__.py").read_text() == BAD_SHAPE
    assert (runtime.runs / "candidates" / "grab_stub2" / "__init__.py").read_text() == BAD_SHAPE
    steps = _kinds(runtime.rows, "rsi_step")
    for step, row in zip(steps, (r1, r2, r3), strict=True):
        for key in ("llm", "learning", "policy", "run_budget"):
            assert step[key] == row[key]
    assert [s["proposer"] for s in bs.rsi_series(runtime.session, TASK)] == ["llm"] * 3


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


def test_recycle_cans_inspection_exposes_exact_source_and_installed_contract():
    import inspect
    from scripts import evolve_llm
    from plugins.embodiment_robocasa import stage_extras

    proj, _ = recycle_cans_projection()
    fd = proj["drivers"]["drop-can1"]
    assert fd["skill"] == "drop_can1" and fd["embodiment"] == "robocasa"
    assert fd["modules"] == ["plugins.embodiment_robocasa.recycle_driver", "plugins.embodiment_robocasa.stage_extras"]
    contract = evolve_llm._contract(evolve_llm._card_package(evolve_llm.CANDIDATES_ROOT), fd["skill"], fd["embodiment"])
    assert contract["card_template"]["ref"] == "plugins.candidates.<name>:provider"
    assert 'skill = "drop_can1"\nembodiment = "robocasa"' in contract["card_template"]["manifest.toml"]
    assert "StepExecutor" in contract["executor_contract"]
    primitives = evolve_llm._primitives(fd["source_ref"])
    assert primitives["primitives"]["constants"]["ADIM"] == 12
    assert primitives["primitives"]["constants"]["GRIP"] == 6
    assert any(k.startswith("_arm_action(env, goal_world, grip") for k in primitives["primitives"]["functions"])
    assert primitives["obs_keys"][0] == "robot0_base_pos" and "out[6] = a[11]" in primitives["action_order"]
    symbol = "plugins.embodiment_robocasa.stage_extras:PointPlaceDriver.act"
    page = inspect_evidence(proj, {"view": "source", "node": "drop-can1", "symbol": symbol})
    assert page["complete"] and page["code_read"]
    assert page["data"]["code"] == inspect.getsource(stage_extras.PointPlaceDriver.act)
    assert page["data"]["symbol"] == symbol
    assert "drop-can1" in page["data"]["owner_nodes"]
    assert len(json.dumps(page).encode()) < 8000


def _repeat_proj(history: list) -> tuple[dict, dict]:
    """A round whose first death is grab-0: two knobs, three bound executors, and the
    campaign history the model must not repeat."""
    proj = {"first_death": {"node": "grab-0", "skill": "grab", "executor": "scripted",
                            "executors": {"scripted": {}, "alt": {}, "geometric": {}},
                            "tunables": {"ref": "test_evolve_e2e:policy_provider", "path": ["tunables"],
                                         "values": {"hover_dz": 0.10, "stall_k": 40},
                                         "hints": {"reach_stall": ["hover_dz"]}}},
            "history": [{"experiments": {"before": "fixture-baseline"}, "after": 0, **row} for row in history],
            "experiment_id": "fixture-baseline", "this_round": {"per_seed": []}}
    before = {"seeds": {"1": {"first_death": "grab-0",
                              "nodes": {"grab-0": {"skill": "grab", "executor": "scripted"}}}}}
    proj["drivers"] = {"grab-0": proj["first_death"]}
    proj["first_death"] = {"node": "grab-0"}
    return proj, before


def _fake(tmp_path, canned, name="canned.json"):
    from scripts import evolve_llm
    f = tmp_path / name
    f.write_text(json.dumps(canned))
    return evolve_llm.load_provider(evolve_llm.FAKE_REF, {"path": str(f)})


def test_no_round_cap_still_obeys_shared_budget_after_model_abstention(tmp_path):
    from test_evolve_e2e import LLM_NONE

    rt = _Runtime(tmp_path, card=_CARD, canned=LLM_NONE, mode="evolution")
    try:
        rt.run({"kind": "evolve", "task": TASK, "seeds": [1, 2], "rounds": 0,
                "max_model_calls": 2})
        campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
        doc = json.loads(campaign.read_text())
        assert doc["status"] == "done" and doc["cursor"] == 2
        assert doc["continuous"] is False and doc["stop_reason"] == "budget_exhausted"
        assert doc["run_budget"]["used"]["model_calls"] == 2
        row = doc["rounds"][0]
        assert row["llm"]["status"] == "abstained" and row["outcome"] == "none"
        assert row["trial"] is None and row["after"] is None and row["after_seeds"] == []
        audit = json.loads((campaign.parent / "llm" / "round-1.json").read_text())
        assert audit["calls"] == 1 and audit["attempts"] == []
        second = doc["rounds"][1]
        assert second["llm"]["stop_reason"] == "model_stop" and second["after"] is None
        assert second["usage"]["episode_attempts"] == 0
        assert second["cycle_budget"]["limits"]["model_calls"] == 1
        assert not (campaign.parent / "llm" / "round-3.json").exists()
    finally:
        rt.stop()


# ── Structured observations and evidence reads ────────────────────────────────

def _parameter_history_proj(extra_history=()):
    """Prior measured values constrain duplicates, not the remaining numeric domain."""
    hist = [{"round": i, "tried": {"kind": "tunables", "node": "grab-0", "detail": {
        "ref": "test_evolve_e2e:policy_provider", "path": ["tunables", k], "from": 1.0, "to": to}}}
        for i, (k, to) in enumerate((("hover_dz", 2.0), ("hover_dz", 0.5),
                                     ("stall_k", 2.0), ("stall_k", 0.5)), 1)]
    return _repeat_proj([*hist, *extra_history])


def test_an_invalid_layer_is_rejected_and_the_ladder_named():
    from scripts.evolve_llm import _parse
    with pytest.raises(ValueError, match="layer must be evaluation|plan|state|recovery|parameter"):
        _parse(json.dumps({"kind": "none", "payload": {}, "summary": "x", "layer": "vibes"}))


def test_compact_observations_keep_history_and_trace_addressable():
    trace = {"start": {"d_base_target": 1.028, "d_eef_target": 0.626, "step": 1},
             "end": {"d_base_target": 1.028, "d_eef_target": 0.574, "step": 65}}
    proj, _ = recycle_cans_projection(
        {"reference": {"drop-can1": {"d_base_target": 0.35}},
         "rounds": [{"round": i, "tried": {"kind": "none", "node": "drop-can1", "detail": {}},
                     "before": 0, "after": None, "published": False} for i in range(1, 13)]}, trace=trace)
    brief = compact_brief(proj)
    assert brief["history_index"]["local_rounds"] == 12
    assert set(brief["driver_index"]) == set(proj["drivers"])
    assert not {"target", "layers", "notebook"} & brief.keys()
    assert brief["observations"][0]["first_death"] == "drop-can1"
    assert proj["this_round"]["per_seed"][0]["divergence"]["d_base_target"] == {"seed": 1.028, "reference": 0.35, "delta": 0.678}
    history = inspect_evidence(proj, {"view": "history"})["data"]["history"]
    assert [r["round"] for r in history] == list(range(1, 13))
    rows = inspect_evidence(proj, {"view": "trace", "node": "drop-can1", "seed": 4243})["data"]
    assert rows[0]["nodes"][0]["trace"] == trace


def test_last_outcome_is_structured_and_incomplete_retests_stay_unknown():
    outcome = {"round": 86, "outcome": "none", "accepted": False,
               "accepted_reason": "model stopped", "before_score": [0, 1], "after_score": None}
    proj, _ = recycle_cans_projection({"last_outcome": outcome, "rounds": [
        {"round": 86, "tried": {"kind": "none", "node": None, "detail": {}},
         "before": 0, "after": None, "published": False, "outcome": "none"}]})
    assert compact_brief(proj)["last_outcome"] == outcome
    history = inspect_evidence(proj, {"view": "history"})["data"]["history"]
    assert history[0]["after"] is None and history[0]["outcome"] == "none"


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
    assert set(trail[0]) == {"id", "ok", "steps", "failure_mode", "trace_end"}
    assert set(trail[1]) == {"id", "ok", "steps", "trace", "geometry", "upstream"}   # no key
    assert trail[1]["geometry"] == geo and trail[1]["upstream"] == up


# ── Missing measurements remain unknown through history inspection ────────────

def _round_131() -> dict:
    """Round 131 of runs/session-robocasa-rsi/campaigns/evolve-recycle_cans, verbatim
    (tests/fixtures/evolve_recycle_rounds.json): the candidate ran on 4243 and left NO
    per-step series (d_eef_min_after null -- 363 of the campaign's 365 trial rows are this),
    and on 4244 drop-can1 was never reached at all (the seed dies at nav-can1)."""
    from pathlib import Path
    return json.loads((Path(__file__).parent / "fixtures"
                       / "evolve_recycle_rounds.json").read_text())["131"]


def test_history_inspection_preserves_missing_candidate_measurements():
    ev = _round_131()["trial_evidence"]
    page = inspect_evidence({"trial_evidence": ev}, {"view": "history"})
    assert page["complete"] and page["data"]["trial_evidence"] == ev
    first, second = page["data"]["trial_evidence"]["seeds"]
    assert first["diff"]["d_eef_min_before"] == 0.430
    assert first["diff"]["d_eef_min_after"] is None
    assert first.get("trace", {}).get("series") in (None, [])
    assert second["diff"]["steps_after"] is None


def test_history_inspection_distinguishes_missing_and_explicit_null_diagnostics():
    diff = {"steps_before": 65, "steps_after": 300, "failure_mode_before": "reach_stall", "failure_mode_after": None}
    mute = {k: v for k, v in diff.items() if k != "failure_mode_after"}
    history = [{"round": 1, "trial_evidence": {"seeds": [{"seed": 4243, "diff": diff}]}},
               {"round": 2, "trial_evidence": {"seeds": [{"seed": 4243, "diff": mute}]}}]
    page = inspect_evidence({"history": history}, {"view": "history"})
    a, b = page["data"]["history"]
    assert a["trial_evidence"]["seeds"][0]["diff"]["failure_mode_after"] is None
    assert "failure_mode_after" not in b["trial_evidence"]["seeds"][0]["diff"]


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
