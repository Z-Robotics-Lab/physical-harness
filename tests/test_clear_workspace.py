"""M7: the ``clear_workspace`` persistent-episode mission card -- the ≥12-node
graph (four KINDS), the fault-adaptive planner, and the FULL run() loop driving
four sub-goals sequentially through ONE fake persistent world with the REAL card
predicates (survey/plan_order/verify_placed/sweep/report) reading its live state.

No simulation here (the test_inventory_build fake手法): a fake world + fake driver
stand in and ONLY the inner drive (``governed.governed_segment``) is monkeypatched,
so the real fault-adaptive ClearWorkspacePlanner, the persistent EpisodeContext,
the segment wrapper, live-state verify, in-episode replan, and span sealing all
run for real. The ONE real headless episode is the sim smoke, reported separately.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from harness import Kernel
from harness.definitions import CAPABILITIES
from harness.registry import load_provider
from plugins.clear_workspace.planner import (
    CATALOGUE, EPISODE, ORACLES, PREDICATES, SEGMENT_SPECS,
    ClearWorkspacePlanner, _pos_key)
from plugins.graphs import InMemorySkillGraph
from plugins.rsi import governed
from plugins.task import workload
from plugins.task.validate import NODE_KINDS, validate_plan

_OBJECTS = ("milk", "bread", "cereal", "can")


# ── the planner graph: shape + kinds + determinism + fault-adaptive routing ───

def _plan(**over) -> dict:
    return ClearWorkspacePlanner().plan(
        {"task": "clear_workspace", "scene": {}, "catalogue": CATALOGUE, **over})


def test_plan_validates_and_is_the_twelve_node_persistent_graph():
    plan = _plan()
    ok, msg = validate_plan(plan, CATALOGUE, ORACLES)
    assert ok and msg == ""
    ids = [n["id"] for n in plan["nodes"]]
    assert ids == ["survey", "plan-order",
                   "clear-milk", "verify-milk", "clear-bread", "verify-bread",
                   "clear-cereal", "verify-cereal", "clear-can", "verify-can",
                   "sweep", "report"]
    kinds = [n.get("kind", "manipulate") for n in plan["nodes"]]
    assert kinds.count("segment") == 4 and kinds.count("verify") == 4
    assert kinds.count("perceive") == 2 and kinds.count("decide") == 2
    assert "manipulate" not in kinds  # a wholly kindful mission
    assert set(kinds) <= NODE_KINDS
    # segments name the ONE `clear` skill routed by object; verifies name the ONE
    # live predicate parameterised by object
    seg = [n["args"]["object"] for n in plan["nodes"]
           if n.get("kind") == "segment"]
    assert seg == list(_OBJECTS)
    # each segment's terminal is gated `lifted`; report is gated so verify is never empty
    assert plan["verify"][-1] == {"after": "report", "predicate": "reported"}
    assert all(v["predicate"] == "lifted" for v in plan["verify"][:-1])


def test_plan_is_deterministic():
    assert json.dumps(_plan(), sort_keys=True) == json.dumps(_plan(), sort_keys=True)


def test_verify_fault_drops_the_object_but_keeps_the_rest():
    p = ClearWorkspacePlanner()
    brief = {"task": "clear_workspace"}
    p.plan(brief)
    nxt = p.plan({**brief, "fault": {"kind": "node_failure", "node": "verify-cereal"}})
    ids = [n["id"] for n in nxt["nodes"]]
    assert "clear-cereal" not in ids and "verify-cereal" not in ids
    assert "clear-milk" in ids and "clear-can" in ids  # the others survive


def test_replan_keeps_a_done_clear_node_for_a_dropped_object():
    # The real loop's fault carries nodes_done; a dropped object whose segment
    # already succeeded keeps that node verbatim (validate_plan replan-stability
    # refuses a graph that loses finished work), while its failing verify goes.
    p = ClearWorkspacePlanner()
    brief = {"task": "clear_workspace"}
    original = p.plan(brief)
    completed = next(i for i, node in enumerate(original["nodes"]) if node["id"] == "clear-cereal")
    done = original["nodes"][:completed + 1]
    nxt = p.plan({**brief, "fault": {
        "kind": "node_failure", "node": "verify-cereal",
        "nodes_done": [node["id"] for node in done]}})
    ids = [n["id"] for n in nxt["nodes"]]
    assert "clear-cereal" in ids and "verify-cereal" not in ids
    kept = next(n for n in nxt["nodes"] if n["id"] == "clear-cereal")
    assert kept["skill"] == "clear" and kept["args"] == {"object": "cereal"} \
        and kept["kind"] == "segment"
    # still verify-covered, and the whole graph clears the hardened validator
    assert {"after": "clear-cereal", "predicate": "lifted"} in nxt["verify"]
    ok, msg = validate_plan(nxt, CATALOGUE, ORACLES, done=done)
    assert ok, msg


def test_segment_fault_retries_then_skips_past_budget():
    p = ClearWorkspacePlanner()
    brief = {"task": "clear_workspace"}
    p.plan(brief)
    f = {"kind": "node_failure", "node": "clear-can"}
    assert "clear-can" in [n["id"] for n in p.plan({**brief, "fault": f})["nodes"]]  # retry
    assert "clear-can" not in [n["id"] for n in p.plan({**brief, "fault": f})["nodes"]]  # skip


def test_wrong_task_refused():
    with pytest.raises(ValueError, match="clear_workspace"):
        ClearWorkspacePlanner().plan({"task": "stack"})


# ── the FULL loop: four sub-goals through ONE fake world, REAL card predicates ──
# The fake world separates two consequences the mission reads distinctly: `lifted`
# (a segment's grasp+lift terminal, read by embodiment.success) and `placed` (the
# object is in its bin, read by the live not_in_bin verify). The frozen policy in
# reality lifts but can not carry to the bin, so `placed` stays empty unless a test
# stages it -- exactly the honest-null the design anticipates.

class _World:
    def __init__(self) -> None:
        self.resets = self.closes = 0
        self.steps = 0
        self.lifted: set[str] = set()
        self.placed: set[str] = set()
        self.object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        self.obs = {_pos_key(o): [0.0, 0.0, 0.85] for o in _OBJECTS}

    def reset(self):
        self.resets += 1
        return self.obs

    def close(self):
        self.closes += 1

    def not_in_bin(self, pos, oid):  # the live geometric oracle, faked by a set
        name = ["milk", "bread", "cereal", "can"][oid]
        return name not in self.placed


class _Embodiment:
    def __init__(self, world: _World) -> None:
        self.world = world
        self.makes = 0

    def make_env(self, spec):
        self.makes += 1
        return self.world

    def tasks(self):
        return ("clearall", *(f"clear{o}" for o in _OBJECTS))

    def object_key(self, spec):
        import plugins.embodiment_robosuite.env as _env  # the real per-object map
        return _env.object_key(spec)

    def success(self, obs, spec, start_z):  # segment terminal == lifted
        import plugins.embodiment_robosuite.env as _env
        obj = _env.object_key(spec).replace("_pos", "").lower()
        return obj in self.world.lifted


class _Driver:
    def __init__(self) -> None:
        self.k = 0
        self.retargets: list = []

    def observe_once(self, obs):
        return obs

    def retarget(self, target):
        self.retargets.append(list(target))


WORLD: _World
EMB: _Embodiment
DRIVER: _Driver
_ATTEMPTS: dict[str, int] = {}
_LIFT_ON_ATTEMPT: dict[str, int] = {}


def _fresh(placed=()) -> None:
    global WORLD, EMB, DRIVER
    WORLD = _World()
    WORLD.placed = set(placed)
    EMB = _Embodiment(WORLD)
    DRIVER = _Driver()
    _ATTEMPTS.clear()
    _LIFT_ON_ATTEMPT.clear()


def epi_embodiment():
    return EMB


class _Policy:
    def make_driver(self, spec):
        return DRIVER


def epi_policy():
    return _Policy()


def _fake_drive(env, obs, driver, spec, bundle, *, step_budget):
    import plugins.embodiment_robosuite.env as _env
    obj = _env.object_key(spec).replace("_pos", "").lower()
    _ATTEMPTS[obj] = _ATTEMPTS.get(obj, 0) + 1
    if _ATTEMPTS[obj] >= _LIFT_ON_ATTEMPT.get(obj, 1):
        env.lifted.add(obj)
    env.steps += 10
    return {"obs": env.obs, "steps": 10,
            "stages": [{"name": "grasp", "success": obj in env.lifted}]}


class _FakeScene:
    def snapshot(self, obs):
        return {"frame": "world", "t": 0.0, "nodes": [], "relations": []}


class _FakeExecutor:
    def map(self, fn, items, *, workers):
        return [fn(item) for item in items]


def _kernel() -> Kernel:
    k = Kernel(CAPABILITIES)
    k.provide("task.planner", ClearWorkspacePlanner(), ref="plugins.clear_workspace.planner:provider")
    k.provide("graph.scene", _FakeScene(), ref="tests.fakes:scene")
    k.provide("graph.skill", InMemorySkillGraph(), ref="plugins.graphs:skill_graph_provider")
    k.provide("embodiment.env", epi_embodiment(), ref="test_clear_workspace:epi_embodiment")
    k.provide("policy.driver", epi_policy(), ref="test_clear_workspace:epi_policy")
    k.provide("exec.rollouts", _FakeExecutor(), ref="tests.fakes:executor")
    return k


def _brief(**over) -> dict:
    b = {"task": "clear_workspace", "catalogue": CATALOGUE, "oracles": ORACLES,
         "predicates": dict(PREDICATES), "episodic": True,
         "episode": dict(EPISODE), "segment_specs": SEGMENT_SPECS}
    b.update(over)
    return b


def test_honest_null_all_four_segments_run_in_one_world(monkeypatch):
    # nothing placed: every live verify fires False, the planner drops each object
    # after one attempt, all four segments still drive sequentially in ONE world,
    # and the mission completes with a faithful cleared=0 report.
    _fresh()
    monkeypatch.setattr(governed, "governed_segment", _fake_drive)
    out = workload.run(_brief(), _kernel(), seed=42, max_replans=8, max_actuations=40)

    assert EMB.makes == 1 and WORLD.resets == 1 and WORLD.closes == 1  # ONE world
    # all four sub-goals drove exactly once, in order, in the same env
    assert _ATTEMPTS == {"milk": 1, "bread": 1, "cereal": 1, "can": 1}
    assert len(DRIVER.retargets) == 4  # re-aimed once per sub-goal, no teleport
    # each verify fired on the live state and failed -> a replan per object
    assert out["replans"] == 4
    assert [f["node"] for f in out["faults"]] == \
        ["verify-milk", "verify-bread", "verify-cereal", "verify-can"]
    # the report is a faithful account of an empty-bin world (honest null == success)
    rep = out["nodes"]["report"]
    assert rep["success"] is True and rep["decision"]["cleared"] == 0
    assert out["success"] is True


def test_span_seals_are_monotonic_across_one_cursor(monkeypatch):
    _fresh()
    monkeypatch.setattr(governed, "governed_segment", _fake_drive)
    out = workload.run(_brief(), _kernel(), seed=1, max_replans=8, max_actuations=40)
    spans = [(out["nodes"][f"clear-{o}"]["governance"]["entered_env_step"],
              out["nodes"][f"clear-{o}"]["governance"]["exited_env_step"])
             for o in _OBJECTS]
    # one shared cursor: each sub-goal's window starts where the previous ended,
    # the world never rewinds between sub-goals
    assert spans == [(0, 10), (10, 20), (20, 30), (30, 40)]


def test_a_placed_object_verifies_true_on_live_state(monkeypatch):
    # stage milk into its bin: verify-milk reads the live not_in_bin and PASSES,
    # so milk is never dropped; the report cross-checks cleared=1
    _fresh(placed={"milk"})
    monkeypatch.setattr(governed, "governed_segment", _fake_drive)
    out = workload.run(_brief(), _kernel(), seed=5, max_replans=8, max_actuations=40)
    assert "verify-milk" not in [f["node"] for f in out["faults"]]
    assert out["nodes"]["verify-milk"]["success"] is True
    assert out["nodes"]["report"]["decision"]["placed"] == ["milk"]
    assert out["nodes"]["report"]["decision"]["cleared"] == 1


def test_slipped_grasp_retries_in_the_same_world(monkeypatch):
    # clear-bread fails to lift on attempt 1 (grasp slip), the base loop re-drives
    # it in the SAME world (no reset), it lifts on attempt 2
    _fresh()
    _LIFT_ON_ATTEMPT["bread"] = 2
    monkeypatch.setattr(governed, "governed_segment", _fake_drive)
    out = workload.run(_brief(), _kernel(), seed=7, max_replans=10, max_actuations=40)
    assert _ATTEMPTS["bread"] == 2  # re-driven once
    assert EMB.makes == 1 and WORLD.resets == 1  # never a second world
    # the first bread fault is the segment (grasp), not the verify
    bread_faults = [f["node"] for f in out["faults"] if "bread" in f["node"]]
    assert bread_faults[0] == "clear-bread"
