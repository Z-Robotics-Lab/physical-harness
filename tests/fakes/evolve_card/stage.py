"""The fake task behind the evolve e2e: reach then grab in a stdlib env. The scripted
driver never lands the grab because ``Driver.STOP`` (a CARRY_STOP-like knob) is too
large; an edit of the copy that lowers it below 0.5 makes grab succeed."""

from __future__ import annotations

import time

from harness import fakes, protocol
from harness.skill_library import segment_specs

EMB = "fakes.evolve_card.stage:env_provider"
RECORDS = {
    "reach": {"id": "reach", "name": "reach", "kind": "segment", "args": {},
              "bindings": {EMB: {"task": "reach"}}},
    "grab": {"id": "grab", "name": "grab", "kind": "segment", "args": {},
             "bindings": {EMB: {"policies": {"scripted": {"task": "grab"}}}}},
}
CATALOGUE = {"reach": {}, "grab": {}}
ORACLES = ("seg_ok",)
EPISODE = {"task": "reach", "horizon": 20}
SEGMENT_SPECS = segment_specs({k: protocol.SkillRecordV0.from_dict(v) for k, v in RECORDS.items()}, EMB)

_OBS = {"robot0_gripper_qpos": [0.03, -0.03], "robot0_gripper_qvel": [0.0, 0.0],
        "robot0_joint_vel": [0.0] * 7, "robot0_eef_pos": [0.0, 0.0, 1.0],
        "cubeA_pos": [0.0, 0.0, 0.1]}


class _Handle(fakes._FakeEnvHandle):
    """Synthetic 128px ``frame()`` for the media recorder; slowed so a cancel lands mid-run."""

    def reset(self):
        time.sleep(0.2)
        super().reset()
        self.achieved = set()
        return dict(_OBS)

    def step(self, action):
        assert len(action) == 1, f"expected a 1-dim action, got {action!r}"
        self.t += 1
        if action[0] == 1.0:
            self.achieved.add("reach")
        elif action[0] == 2.0 and "reach" in self.achieved:
            self.achieved.add("grab")
        return dict(_OBS), 0.0, False, {}


class Env:
    def make_env(self, spec):
        return _Handle()

    def tasks(self):
        return ("reach", "grab")

    def object_key(self, spec):
        raise AssertionError("heterogeneous segment path never reads object_key")

    def success(self, obs, spec, start_z):
        return True

    def terminal_success(self, obs, spec, start_z, env=None):
        return {"reach", "grab"} <= env.achieved


class Driver:
    """Each segment drives STEPS env steps (so the recorder sees frames)."""
    STEPS = 8
    STOP = 0.65   # the loaded standoff: grab only lands below 0.5
    n = 0
    task = "reach"

    @property
    def exhausted(self):
        return self.n >= self.STEPS

    def observe_once(self, obs):
        pass

    def on_handback(self):
        pass

    def act(self, obs):
        self.n += 1
        if self.task == "reach":
            return (1.0,)
        return (2.0,) if self.STOP < 0.5 else (0.0,)

    def enter_segment(self, env, spec, executor=None):
        self.n, self.task = 0, spec.task

    def segment_success(self, env):
        return self.task in env.achieved


class Policies:
    def make_driver(self, spec):
        return Driver()


class Planner:
    identity = "evolve_card:fixed"

    def plan(self, brief):
        return {"goal": "reach then grab",
                "nodes": [{"id": "reach-0", "skill": "reach", "kind": "segment", "args": {}, "after": []},
                          {"id": "grab-0", "skill": "grab", "kind": "segment", "args": {}, "after": ["reach-0"]}],
                "verify": [{"after": "reach-0", "predicate": "seg_ok"}, {"after": "grab-0", "predicate": "seg_ok"}],
                "rationale": "fixed"}


def env_provider():
    return Env()


def policy_provider(**params):
    return Policies()


def planner_provider():
    return Planner()
