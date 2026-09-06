"""The kitchen_thaw mission card (M7 on RoboCasa): plan shape + binding wiring.

Base-lane (neither robosuite nor robocasa needed): the card is PURE DATA + a
deterministic planner, so its graph validates against the REAL ``validate_plan``,
its binding folds through ``discover()`` with a base sha unmoved, and every ref it
names load_provider-resolves base-clean (the robocasa modules import robocasa lazily
inside their methods). The LIVE mission -- one persistent episode, six driven
segments, live-state verify, in-episode replan -- is smoked end-to-end through the
runtime in the robocasa venv (local-archive/robocasa-adapt/phase4.md), not here.
"""

from __future__ import annotations

from harness.config import resolve_plan
from harness.manifest import discover
from harness.registry import load_provider
from plugins.mission_kitchen_thaw import planner as P
from plugins.task.validate import NODE_KINDS, validate_plan
from profiles import base_profile


def test_plan_validates_against_the_real_validator():
    plan = P.KitchenThawPlanner().plan({"task": "kitchen_thaw"})
    ok, msg = validate_plan(plan, P.CATALOGUE, P.ORACLES)
    assert ok, msg
    ids = [n["id"] for n in plan["nodes"]]
    assert len(ids) == 15 and len(set(ids)) == 15
    kinds = [n.get("kind", "manipulate") for n in plan["nodes"]]
    assert kinds.count("segment") == 6 and kinds.count("verify") == 6
    assert kinds.count("perceive") == 1 and kinds.count("decide") == 2
    # every declared kind is a real base handler kind
    assert set(kinds) <= NODE_KINDS


def test_every_segment_retasks_and_every_predicate_is_declared():
    plan = P.KitchenThawPlanner().plan({"task": "kitchen_thaw"})
    for n in plan["nodes"]:
        kind = n.get("kind", "manipulate")
        if kind == "segment":
            assert n["skill"] in P.SEGMENT_SPECS, f"segment {n['id']} not re-tasked"
            assert "task" in P.SEGMENT_SPECS[n["skill"]]
        elif kind in ("perceive", "decide", "verify"):
            assert n["skill"] in P.PREDICATES, f"{n['id']} names no predicate"


def test_determinism_plan_and_replan_are_byte_identical():
    import json
    p1 = json.dumps(P.KitchenThawPlanner().plan({"task": "kitchen_thaw"}), sort_keys=True)
    # a replan (the base loop threads a fault) emits the SAME graph: the planner is a
    # pure fn of the task; the in-episode retry is the loop re-running a failed node.
    p2 = json.dumps(P.KitchenThawPlanner().plan(
        {"task": "kitchen_thaw", "fault": {"kind": "node_failure", "node": "nav-micro"}}),
        sort_keys=True)
    assert p1 == p2


def test_binding_folds_and_base_sha_is_untouched():
    from test_manifest import _LLM_BASE_SHA

    reg = discover()
    b = reg.task_bindings.get("kitchen_thaw")
    assert b is not None, "kitchen_thaw not discovered"
    # the persistence + second-sim declarations the runtime threads
    assert b.get("episodic") is True
    for key in ("env", "percept", "policy", "planner", "catalogue", "oracles",
                "predicates", "episode", "segment_specs"):
        assert key in b, f"binding missing {key}"
    # A task-only card adds no mounts to the current LLM-only base manifest.
    assert resolve_plan(base_profile()).sha() == _LLM_BASE_SHA


def test_every_ref_resolves_base_clean():
    # catalogue/oracles/predicates/episode/segment_specs are module attributes;
    # policy/planner/env/percept are factories -- all import base-clean (robocasa is
    # lazy inside its methods, so resolving a ref never drags the simulator in).
    reg = discover()
    b = reg.task_bindings["kitchen_thaw"]
    # load_provider CALLS the named factory and hands back the provider instance
    assert hasattr(load_provider(b["policy"]), "make_driver")
    assert hasattr(load_provider(b["planner"]), "plan")
    assert hasattr(load_provider(b["env"]), "make_env")
    assert hasattr(load_provider(b["percept"]), "object_estimate")
    # every kindful predicate factory resolves to a (node, ctx) callable
    for ref in P.PREDICATES.values():
        assert callable(load_provider(ref))


def test_grasp_verify_is_secure_dz_shaped_not_the_bare_latch(monkeypatch):
    """The `grasped`/`at-micro` verifies must mean HOLDING, not touching.

    robocasa's ``check_obj_grasped`` is contact + fingers closed with no lift
    term, so it reads True with the hand shut around meat still resting on the
    shelf -- 7 of 7 constructible synthetic controls in
    scripts/probe_grasp_predicate.py. The branch logic that fixes it is scored
    here on fakes; the live reading needs the sim.
    """
    calls: dict = {}

    def fake_load(ref, params=None):
        if ref.endswith("obj_grasped"):
            return lambda env: env["latch"]
        if ref.endswith("obj_grasped_secure"):
            def secure(env, z0):
                calls["z0"] = z0
                return env["latch"] and env["z"] > z0 + 0.08
            return secure
        raise AssertionError(ref)

    monkeypatch.setattr(P, "load_provider", fake_load)

    class _Ctx:
        def __init__(self, env, out):
            self.episode = type("E", (), {"env": env})()
            self.nodes_out = out

    surveyed = {"survey": {"facts": {"meat_pos": [0.0, 0.0, 1.00]}}}

    def score(env, out):
        return P._secure_grasp_verify()({}, _Ctx(env, out))["success"]

    # closed around it, never lifted -> the latch says yes, the verify says no
    assert score({"latch": True, "z": 1.00}, surveyed) is False
    assert calls["z0"] == 1.00  # the SURVEYED resting z is the reference
    # risen a full SECURE_DZ off the surveyed z -> a real hold
    assert score({"latch": True, "z": 1.09}, surveyed) is True
    # risen but no longer latched -> not held
    assert score({"latch": False, "z": 1.30}, surveyed) is False
    # knocked to a LOWER shelf and regrasped there: the segment's own sealed
    # SECURE_DZ success is the alternative z-evidence, latch still required now
    lower = dict(surveyed, grasp={"success": True})
    assert score({"latch": True, "z": 0.80}, lower) is True
    assert score({"latch": False, "z": 0.80}, lower) is False
    # unsurveyed -> no reference, no claim (never a free True)
    assert score({"latch": True, "z": 9.99}, {}) is False
