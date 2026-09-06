"""The recycle_cans mission card: plan shape + binding wiring (base lane --
the card is pure data + a deterministic planner; the live mission is smoked
through the resident robocasa runtime, evidence in local-archive/robocasa-adapt/
missions/)."""

from __future__ import annotations

from harness.config import resolve_plan
from harness.manifest import discover
from harness.registry import load_provider
from plugins.mission_recycle_cans import planner as P
from plugins.task.validate import NODE_KINDS, validate_plan
from profiles import base_profile


def test_plan_validates_against_the_real_validator():
    plan = P.RecycleCansPlanner().plan({"task": "recycle_cans"})
    ok, msg = validate_plan(plan, P.CATALOGUE, P.ORACLES)
    assert ok, msg
    ids = [n["id"] for n in plan["nodes"]]
    assert len(ids) == 32 and len(set(ids)) == 32
    kinds = [n.get("kind", "manipulate") for n in plan["nodes"]]
    # per can: nav/grasp/carry/drop segments + at/grasped/placed verifies
    assert kinds.count("segment") == 16 and kinds.count("verify") == 12
    assert kinds.count("perceive") == 2 and kinds.count("decide") == 2
    assert set(kinds) <= NODE_KINDS
    # the sweep re-perceive sits after the four chains, before the report
    assert ids[-2:] == ["sweep", "report"]


def test_every_segment_retasks_and_every_predicate_is_declared():
    plan = P.RecycleCansPlanner().plan({"task": "recycle_cans"})
    for n in plan["nodes"]:
        kind = n.get("kind", "manipulate")
        if kind == "segment":
            assert n["skill"] in P.SEGMENT_SPECS, f"segment {n['id']} not re-tasked"
            assert "task" in P.SEGMENT_SPECS[n["skill"]]
        elif kind in ("perceive", "decide", "verify"):
            assert n["skill"] in P.PREDICATES, f"{n['id']} names no predicate"


def test_determinism_plan_and_replan_are_byte_identical():
    import json
    p1 = json.dumps(P.RecycleCansPlanner().plan({"task": "recycle_cans"}),
                    sort_keys=True)
    p2 = json.dumps(P.RecycleCansPlanner().plan(
        {"task": "recycle_cans",
         "fault": {"kind": "node_failure", "node": "grasp-can2"}}),
        sort_keys=True)
    assert p1 == p2


def test_binding_folds_and_base_sha_is_untouched():
    from test_manifest import _LLM_BASE_SHA

    reg = discover()
    b = reg.task_bindings.get("recycle_cans")
    assert b is not None, "recycle_cans not discovered"
    assert b.get("episodic") is True
    for key in ("env", "percept", "policy", "planner", "catalogue", "oracles",
                "predicates", "episode", "segment_specs"):
        assert key in b, f"binding missing {key}"
    assert resolve_plan(base_profile()).sha() == _LLM_BASE_SHA


def test_every_ref_resolves_base_clean():
    reg = discover()
    b = reg.task_bindings["recycle_cans"]
    assert hasattr(load_provider(b["policy"]), "make_driver")
    assert hasattr(load_provider(b["planner"]), "plan")
    assert hasattr(load_provider(b["env"]), "make_env")
    assert hasattr(load_provider(b["percept"]), "object_estimate")
    for ref in P.PREDICATES.values():
        assert callable(load_provider(ref))
