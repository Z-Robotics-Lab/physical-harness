"""A fake embodiment-style stage driver for the patch tests: a ``_STAGES`` table (the
convention scripts/evolve_llm reads), a stage class carrying the constant a patch
changes (``GrabStage.STOP``, a CARRY_STOP-like knob: grab only lands below 0.5, the
installed 0.65 never does) and a composite driver with the native-executor seam
(``enter_segment(env, spec, executor=)``, stage_extras.CompositeStageDriver's shape).
The env / planner / catalogue ride test_evolve_e2e's (reach then grab)."""

from __future__ import annotations

from test_evolve_e2e import CATALOGUE, EPISODE, ORACLES, _Env, _Planner  # noqa: F401 -- re-exported by the card

from harness import protocol
from harness.skill_library import segment_specs

EMB = "fakes.patch_stage:env_provider"
RECORDS = {
    "reach": {"id": "reach", "name": "reach", "kind": "segment", "args": {},
              "bindings": {EMB: {"task": "reach"}}},
    "grab": {"id": "grab", "name": "grab", "kind": "segment", "args": {},
             "bindings": {EMB: {"policies": {"scripted": {"task": "grab"}}}}},
}
SEGMENT_SPECS = segment_specs({k: protocol.SkillRecordV0.from_dict(v) for k, v in RECORDS.items()}, EMB)


class ReachStage:
    def act(self, env, obs):
        env.reached = True
        return (0.0,)

    def done(self, env):
        return bool(getattr(env, "reached", False))


class GrabStage:
    # the loaded standoff: the scripted value never closes the grab
    STOP = 0.65

    def __init__(self, target):
        self.target = target

    def act(self, env, obs):
        if self.STOP < 0.5:
            env.grabbed = True
        return (0.0,)

    def done(self, env):
        return bool(getattr(env, "grabbed", False))


_STAGES = {"reach": (lambda: ReachStage(), 8), "grab": (lambda: GrabStage("cube"), 8)}


class Driver:
    def __init__(self):
        self.n, self._cap, self._env, self._stage, self._ex, self._native = 0, 0, None, None, None, False

    @property
    def exhausted(self):
        return self._stage is None or self.n >= self._cap or bool(self._stage.done(self._env))

    def observe_once(self, obs):
        pass

    def on_handback(self):
        pass

    def act(self, obs):
        self.n += 1
        return self._ex.act(obs) if self._native else self._stage.act(self._env, obs)

    def enter_segment(self, env, spec, executor=None):
        factory, self._cap = _STAGES[spec.task]
        self._env, self._stage, self._ex, self.n = env, factory(), executor, 0
        self._native = executor is not None and executor.handshake()["transport"] == "inproc"
        if executor is not None:
            executor.reset()
        if self._native:
            executor.bind(env, target=getattr(self._stage, "target", None))

    def segment_success(self, env):
        return bool(self._stage.done(env))


class Policies:
    def make_driver(self, spec):
        return Driver()


def env_provider():
    return _Env()


def policy_provider(**params):
    return Policies()


def planner_provider():
    return _Planner()
