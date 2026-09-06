"""The steam_prep mission card: plan shape (incl. the STRUCTURAL temporal
order) + binding wiring (base lane). The card is graph-first: its faucet driver
gap is the xfail frontier in tests/test_robocasa_missions.py, not here."""

from __future__ import annotations

from harness.config import resolve_plan
from harness.manifest import discover
from harness.registry import load_provider
from plugins.mission_steam_prep import planner as P
from plugins.task.validate import NODE_KINDS, validate_plan
from profiles import base_profile


def test_plan_validates_against_the_real_validator():
    plan = P.SteamPrepPlanner().plan({"task": "steam_prep"})
    ok, msg = validate_plan(plan, P.CATALOGUE, P.ORACLES)
    assert ok, msg
    ids = [n["id"] for n in plan["nodes"]]
    assert len(ids) == 21 and len(set(ids)) == 21
    kinds = [n.get("kind", "manipulate") for n in plan["nodes"]]
    assert kinds.count("segment") == 9 and kinds.count("verify") == 9
    assert kinds.count("perceive") == 1 and kinds.count("decide") == 2
    assert set(kinds) <= NODE_KINDS


def test_temporal_order_is_structural():
    """MultistepSteaming's cross-step constraint ("the vegetable was in the
    sink WHILE the water ran") is enforced by the linear verify order: the
    in-sink verify sits strictly between water-on and water-off."""
    plan = P.SteamPrepPlanner().plan({"task": "steam_prep"})
    ids = [n["id"] for n in plan["nodes"]]
    assert (ids.index("water-on") < ids.index("veg-in-sink")
            < ids.index("water-off") < ids.index("veg-in-pot")
            < ids.index("pot-on-burner"))


def test_every_segment_retasks_and_every_predicate_is_declared():
    plan = P.SteamPrepPlanner().plan({"task": "steam_prep"})
    for n in plan["nodes"]:
        kind = n.get("kind", "manipulate")
        if kind == "segment":
            assert n["skill"] in P.SEGMENT_SPECS, f"segment {n['id']} not re-tasked"
            assert "task" in P.SEGMENT_SPECS[n["skill"]]
        elif kind in ("perceive", "decide", "verify"):
            assert n["skill"] in P.PREDICATES, f"{n['id']} names no predicate"


def test_determinism_plan_and_replan_are_byte_identical():
    import json
    p1 = json.dumps(P.SteamPrepPlanner().plan({"task": "steam_prep"}),
                    sort_keys=True)
    p2 = json.dumps(P.SteamPrepPlanner().plan(
        {"task": "steam_prep",
         "fault": {"kind": "node_failure", "node": "faucet-on"}}),
        sort_keys=True)
    assert p1 == p2


def test_binding_folds_and_base_sha_is_untouched():
    from test_manifest import _LLM_BASE_SHA

    reg = discover()
    b = reg.task_bindings.get("steam_prep")
    assert b is not None, "steam_prep not discovered"
    assert b.get("episodic") is True
    for key in ("env", "percept", "policy", "planner", "catalogue", "oracles",
                "predicates", "episode", "segment_specs"):
        assert key in b, f"binding missing {key}"
    assert resolve_plan(base_profile()).sha() == _LLM_BASE_SHA


def test_every_ref_resolves_base_clean():
    reg = discover()
    b = reg.task_bindings["steam_prep"]
    assert hasattr(load_provider(b["policy"]), "make_driver")
    assert hasattr(load_provider(b["planner"]), "plan")
    assert hasattr(load_provider(b["env"]), "make_env")
    assert hasattr(load_provider(b["percept"]), "object_estimate")
    for ref in P.PREDICATES.values():
        assert callable(load_provider(ref))
