"""The pack_lunch mission card: plan shape + the temperature decide + binding
wiring (base lane; live mission smoked through the resident robocasa runtime)."""

from __future__ import annotations

from harness.config import resolve_plan
from harness.manifest import discover
from harness.registry import load_provider
from plugins.mission_pack_lunch import planner as P
from plugins.task.validate import NODE_KINDS, validate_plan
from profiles import base_profile


def test_plan_validates_against_the_real_validator():
    plan = P.PackLunchPlanner().plan({"task": "pack_lunch"})
    ok, msg = validate_plan(plan, P.CATALOGUE, P.ORACLES)
    assert ok, msg
    ids = [n["id"] for n in plan["nodes"]]
    assert len(ids) == 31 and len(set(ids)) == 31
    kinds = [n.get("kind", "manipulate") for n in plan["nodes"]]
    assert kinds.count("segment") == 16 and kinds.count("verify") == 12
    assert kinds.count("perceive") == 1 and kinds.count("decide") == 2
    assert set(kinds) <= NODE_KINDS


def test_sort_temp_is_the_deterministic_assignment():
    """The decide node's temperature->container mapping is a pure function of
    the item names and equals the authored TARGET table (decision and driver
    stage table can never disagree)."""
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class _Ctx:
        episode: Any
        nodes_out: dict

    facts = {"objects": {o: [0.0, 0.0, 1.0]
                         for o in (*P.ITEMS, "tupperware0", "tupperware1")}}
    out = P.sort_temp()({}, _Ctx(None, {"survey": {"success": True,
                                                   "facts": facts}}))
    assert out["success"] is True
    assert out["decision"]["assignment"] == P.TARGET == {
        "hot0": "tupperware0", "hot1": "tupperware0",
        "cold0": "tupperware1", "cold1": "tupperware1"}
    # no survey -> the decide honestly refuses
    assert P.sort_temp()({}, _Ctx(None, {}))["success"] is False


def test_every_segment_retasks_and_every_predicate_is_declared():
    plan = P.PackLunchPlanner().plan({"task": "pack_lunch"})
    for n in plan["nodes"]:
        kind = n.get("kind", "manipulate")
        if kind == "segment":
            assert n["skill"] in P.SEGMENT_SPECS, f"segment {n['id']} not re-tasked"
            assert "task" in P.SEGMENT_SPECS[n["skill"]]
        elif kind in ("perceive", "decide", "verify"):
            assert n["skill"] in P.PREDICATES, f"{n['id']} names no predicate"


def test_determinism_plan_and_replan_are_byte_identical():
    import json
    p1 = json.dumps(P.PackLunchPlanner().plan({"task": "pack_lunch"}),
                    sort_keys=True)
    p2 = json.dumps(P.PackLunchPlanner().plan(
        {"task": "pack_lunch",
         "fault": {"kind": "node_failure", "node": "carry-cold0"}}),
        sort_keys=True)
    assert p1 == p2


def test_binding_folds_and_base_sha_is_untouched():
    from test_manifest import _LLM_BASE_SHA

    reg = discover()
    b = reg.task_bindings.get("pack_lunch")
    assert b is not None, "pack_lunch not discovered"
    assert b.get("episodic") is True
    for key in ("env", "percept", "policy", "planner", "catalogue", "oracles",
                "predicates", "episode", "segment_specs"):
        assert key in b, f"binding missing {key}"
    assert resolve_plan(base_profile()).sha() == _LLM_BASE_SHA


def test_every_ref_resolves_base_clean():
    reg = discover()
    b = reg.task_bindings["pack_lunch"]
    assert hasattr(load_provider(b["policy"]), "make_driver")
    assert hasattr(load_provider(b["planner"]), "plan")
    assert hasattr(load_provider(b["env"]), "make_env")
    assert hasattr(load_provider(b["percept"]), "object_estimate")
    for ref in P.PREDICATES.values():
        assert callable(load_provider(ref))
