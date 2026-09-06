"""Mission lane, unit level: the PlanRecord library planner, mission
decomposition validation, the FakeEndpoint reply sequence, and a composed graph
(tasks with real goals) through Covered and the workload's composed-graph entry."""

from __future__ import annotations

import json

from harness import protocol
from harness.events import SessionLog
from harness.manifest import discover
from plugins.model_endpoint import fake_provider
from plugins.planner_library import provider as library_provider
from plugins.planner_vlm import provider as vlm_provider
from plugins.task import workload
from plugins.task.planner_stack import CATALOGUE, ORACLES, StackPlanner
from plugins.task.validate import validate_plan
from scripts.harness_runtime import _compose, task_brief
from test_task_seam import _CountingPlanner, _RolloutFake, _task_kernel

CT_PLAN = StackPlanner().plan({"task": "clear_table", "catalogue": CATALOGUE,
                               "oracles": ORACLES, "scene": {}})
_CT_BINDING = discover().task_bindings["clear_table"]


def _plan_record(lower: float, plan_id: str) -> dict:
    return {"kind": "plan", "id": plan_id, "task": "clear_table", "goal": ["holding(can)"],
            "graph": CT_PLAN, "embodiment": "robosuite", "arm": "scripted",
            "evidence": {"n": 12, "k": 12, "L_mean": 2.0, "seed_blocks": [], "sessions": []},
            "rule": {"theta": 0.8, "n_min": 10, "lower": lower}, "published_from": []}


def _brief(**extra) -> dict:
    return {"task": "clear_table", "catalogue": CATALOGUE, "oracles": ORACLES, "scene": {},
            "embodiment": "robosuite", "arm": "scripted", **extra}


# --- library planner --------------------------------------------------------

def test_library_planner_hit_returns_the_best_record_graph():
    plans = [_plan_record(0.71, "weak"), _plan_record(0.87, "strong"),
             {**_plan_record(0.99, "other-arm"), "arm": "pi05"},
             {**_plan_record(0.99, "other-task"), "task": "stack"}]
    p = library_provider(inner="plugins.task.planner_stack:provider")
    out = p.plan(_brief(plans=plans))
    assert out["planner"] == {"provider": "library", "plan_id": "strong"}
    assert out["nodes"] == CT_PLAN["nodes"] and out["verify"] == CT_PLAN["verify"]
    assert validate_plan(out, CATALOGUE, ORACLES)[0]


def test_library_planner_miss_delegates_to_the_inner_planner():
    p = library_provider(inner="plugins.task.planner_stack:provider")
    out = p.plan(_brief(plans=[{**_plan_record(0.9, "x"), "embodiment": "robocasa"}]))
    assert out["nodes"] == CT_PLAN["nodes"] and "planner" not in out
    assert p.plan(_brief())["nodes"] == CT_PLAN["nodes"]
    assert p.deterministic is True


# --- FakeEndpoint sequence ---------------------------------------------------

def test_fake_endpoint_consumes_a_json_list_in_order_across_instances(tmp_path):
    f = tmp_path / "seq.json"
    f.write_text(json.dumps(["first", {"tasks": []}]))
    a, b = fake_provider(path=str(f)), fake_provider(path=str(f))
    assert a.chat([]) == "first"
    assert json.loads(b.chat([])) == {"tasks": []}
    try:
        a.chat([])
    except ValueError as exc:
        assert "exhausted" in str(exc)
    else:
        raise AssertionError("third call should be refused")
    one = tmp_path / "one.json"
    one.write_text(json.dumps({"nodes": []}))
    ep = fake_provider(path=str(one))
    assert ep.chat([]) == ep.chat([]) == one.read_text()


# --- decomposition -----------------------------------------------------------

_KNOWN = [{"task": "clear_table", "goal": ["holding(object)"], "description": "clear"},
          {"task": "stack", "goal": ["on(object,target)"], "description": "stack"}]
_PREDS = {"holding": ("object",), "on": ("object", "target"), "present": ("object",)}


def _decomposer(tmp_path, *replies):
    f = tmp_path / "dec.json"
    f.write_text(json.dumps(list(replies)))
    return vlm_provider(endpoint="plugins.model_endpoint:fake_provider",
                        endpoint_params={"path": str(f)})


def _mission_brief() -> dict:
    return {"mission": "clear the table then stack", "known_tasks": _KNOWN,
            "predicates": _PREDS, "objects": ["can", "milk", "cubeA", "cubeB"]}


def test_decompose_accepts_known_tasks_over_catalogue_predicates(tmp_path):
    reply = {"tasks": [{"id": "t1", "task": "clear_table", "goal": ["holding(can)"]},
                       {"id": "t2", "task": "stack", "goal": ["on(cubeA, cubeB)"]}],
             "rationale": "clear first"}
    out = _decomposer(tmp_path, reply).decompose(_mission_brief())
    assert [t["task"] for t in out["tasks"]] == ["clear_table", "stack"]
    assert out["tasks"][1]["goal"] == ["on(cubeA,cubeB)"]
    assert len(out["prompt_sha"]) == 64 and out["rationale"] == "clear first"
    proj = protocol.mission_projection(_KNOWN, _PREDS, ["b", "a"])
    assert list(proj) == ["known_tasks", "predicates", "objects", "output_schema"]
    assert proj["objects"] == ["a", "b"] and proj["predicates"]["on"] == ["object", "target"]


def test_decompose_refuses_an_unknown_predicate_and_an_unknown_task(tmp_path):
    bad_pred = {"tasks": [{"id": "t1", "task": "stack", "goal": ["levitating(cubeA)"]}]}
    bad_task = {"tasks": [{"id": "t1", "task": "fly", "goal": ["on(cubeA, cubeB)"]}]}
    d = _decomposer(tmp_path, bad_pred, bad_task, "not json {")
    for needle in ("unknown predicate 'levitating'", "unknown task 'fly'", "Expecting"):
        try:
            d.decompose(_mission_brief())
        except ValueError as exc:
            assert needle in str(exc), (needle, exc)
        else:
            raise AssertionError(f"expected refusal: {needle}")


# --- composed graph: Covered bites ------------------------------------------

def _composed(goal_preds: list[str]) -> dict:
    return _compose("clear the table", [({"id": "ct", "goal": goal_preds}, CT_PLAN)],
                    {"provider": "mission"})


def test_composed_graph_with_real_goals_passes_covered_and_a_missing_ensures_fails():
    brief = {**task_brief("clear_table", _CT_BINDING), "arm": "scripted"}
    records = workload._records(brief, CATALOGUE)
    facts, objects = workload._sigma0(brief, {}, {}, records)
    brief = {**brief, "facts": facts, "objects": objects}
    good = _composed(["holding(can)", "holding(milk)"])
    ok, msg = validate_plan(good, CATALOGUE, ORACLES)
    assert ok, msg
    assert good["nodes"][0] == {"id": "ct.pick-can", "task": "ct", "skill": "pick",
                                "args": {"object": "can"}, "after": []}
    assert workload._graph_problems(good, None, (), brief, CATALOGUE, 1) == []
    bad = _composed(["holding(can)", "on(can,milk)"])
    assert validate_plan(bad, CATALOGUE, ORACLES)[0]      # typed: fine
    problems = workload._graph_problems(bad, None, (), brief, CATALOGUE, 1)
    assert problems == ["covered: task 'ct' goal on(can,milk) is ensured by none of its nodes"]
    # graph_sha ignores provenance keys only
    assert workload.protocol.graph_sha(good) == workload.protocol.graph_sha({**good, "planner": {"x": 1}})
    assert workload.protocol.graph_sha(good) != workload.protocol.graph_sha(bad)


def test_validate_plan_refuses_a_malformed_tasks_block_and_an_unknown_node_task():
    plan = {**CT_PLAN, "tasks": [{"id": "ct", "goal": "holding(can)"}]}
    ok, msg = validate_plan(plan, CATALOGUE, ORACLES)
    assert not ok and "tasks[0]" in msg
    plan = {**CT_PLAN, "tasks": [{"id": "ct", "goal": []}],
            "nodes": [{**n, "task": "zz"} for n in CT_PLAN["nodes"]]}
    ok, msg = validate_plan(plan, CATALOGUE, ORACLES)
    assert not ok and "unknown task 'zz'" in msg


def test_workload_runs_a_composed_graph_without_asking_the_planner(monkeypatch):
    log = SessionLog()
    planner = _CountingPlanner()
    kernel = _task_kernel(planner, log=log)
    monkeypatch.setattr(workload, "_governed_rollout", _RolloutFake([True, True]))
    brief = {**task_brief("clear_table", _CT_BINDING), "arm": "scripted",
             "graph": _composed(["holding(can)", "holding(milk)"])}
    out = workload.run(brief, kernel, seed=7, max_actuations=4)
    assert out["success"] is True and planner.briefs == []
    row = next(r["data"] for r in log.rows() if r["kind"] == "task.plan")
    assert row["legal"] is True and row["planner"] == {"provider": "mission"}
    assert row["graph_sha"] == workload.protocol.graph_sha(brief["graph"])
    assert sorted(out["nodes"]) == ["ct.pick-can", "ct.pick-milk"]
