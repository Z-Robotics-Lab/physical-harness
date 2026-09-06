"""M7: the generic in-episode persistent-mission runner -- ONE sim episode threaded
through the plan graph, sub-goals driven sequentially on the SAME world, live-state
verify, in-episode consequence-carrying replan.

No simulation here (the test_inventory_build fake手法): a fake world + fake driver
stand in and ONLY the inner drive (``governed.governed_segment``) is monkeypatched,
so the REAL ``run()`` loop, the persistent ``EpisodeContext``, the segment wrapper
(retarget, cursor, terminal), governance-per-segment and span sealing all run for
real. Byte-identity of the EXTRACTED drive vs ``governed_rollout`` is a robosuite
golden proved out-of-band (docs/project-documentation.md §3), not this file.
"""

from __future__ import annotations

import json

import pytest

from harness import Kernel
from harness.definitions import CAPABILITIES
from harness.events import SessionLog
from plugins.graphs import InMemorySkillGraph
from plugins.rsi import governed
from plugins.task import workload
from plugins.task.validate import validate_plan

# ── the persistent world + policy, shared across every sub-goal (module globals so
#    the load_provider REFS the runner rebuilds close over the SAME instance the
#    test asserts on -- run() constructs env/driver through the ref, not the mount) ─

_OBJ = {"clear_a": "a", "clear_b": "b"}
_KEY = {"clear_a": "a_pos", "clear_b": "b_pos"}


class _World:
    """The ONE persistent env: reset once, stepped many, closed once. ``placed`` is
    the consequence a segment writes and a later verify/segment reads -- the world
    carrying forward, never a reset preview."""

    def __init__(self) -> None:
        self.resets = self.steps = self.closes = 0
        self.placed: set[str] = set()
        self.obs = {"a_pos": [0.1, 0.1, 0.9], "b_pos": [0.2, 0.2, 0.9]}

    def reset(self):
        self.resets += 1
        return self.obs

    def step(self, action):
        self.steps += 1
        return self.obs, 0.0, False, {}

    def close(self):
        self.closes += 1


class _Embodiment:
    def __init__(self, world: _World) -> None:
        self.world = world
        self.makes = 0

    def make_env(self, spec):
        self.makes += 1
        return self.world

    def tasks(self):
        return ("clear_a", "clear_b")

    def object_key(self, spec):
        return _KEY[spec.task]

    def success(self, obs, spec, start_z):
        # terminal = this sub-goal's object is now in its bin (a live-world read)
        return _OBJ[spec.task] in self.world.placed


class _Driver:
    def __init__(self) -> None:
        self.k = 0
        self.target = None
        self.retargets: list = []

    def observe_once(self, obs):
        self.target = obs

    def retarget(self, target):
        self.target = target
        self.retargets.append(target)


# per-test singletons + drive bookkeeping, reset by _fresh()
WORLD: _World
EMB: _Embodiment
DRIVER: _Driver
_ATTEMPTS: dict[str, int] = {}
_PLACE_ON_ATTEMPT: dict[str, int] = {}
_VERIFY_CALLS: dict[str, int] = {}


def _fresh() -> None:
    global WORLD, EMB, DRIVER
    WORLD = _World()
    EMB = _Embodiment(WORLD)
    DRIVER = _Driver()
    _ATTEMPTS.clear()
    _PLACE_ON_ATTEMPT.clear()
    _VERIFY_CALLS.clear()


def epi_embodiment():
    return EMB


class _Policy:
    def make_driver(self, spec):
        return DRIVER


def epi_policy():
    return _Policy()


# ── the faked inner drive: places this sub-goal's object into the shared world on
#    its N-th attempt, proving both persistence and in-episode retry ──────────────

def _fake_drive(env, obs, driver, spec, bundle, *, step_budget):
    obj = _OBJ[spec.task]
    _ATTEMPTS[obj] = _ATTEMPTS.get(obj, 0) + 1
    if _ATTEMPTS[obj] >= _PLACE_ON_ATTEMPT.get(obj, 1):
        env.placed.add(obj)
    env.steps += 10
    ok = obj in env.placed
    return {"obs": env.obs, "steps": 10, "stages": [{"name": "grasp", "success": ok}]}


# ── the card vocabulary: a two-sub-goal persistent mission ───────────────────────

_CATALOGUE = {"clear": {"object": str}, "inbin_a": {}, "inbin_b": {}}
_ORACLES = ("cleared",)
_SEGMENT_SPECS = {"clear": {"task_by_object": {"a": "clear_a", "b": "clear_b"}}}


def inbin_a():
    def run(node, ctx):
        _VERIFY_CALLS["a"] = _VERIFY_CALLS.get("a", 0) + 1
        return {"success": "a" in ctx.episode.env.placed}
    return run


def inbin_b():
    def run(node, ctx):
        _VERIFY_CALLS["b"] = _VERIFY_CALLS.get("b", 0) + 1
        return {"success": "b" in ctx.episode.env.placed}
    return run


def inbin_b_flaky():
    def run(node, ctx):
        _VERIFY_CALLS["b"] = _VERIFY_CALLS.get("b", 0) + 1
        return {"success": _VERIFY_CALLS["b"] >= 2}  # fails first, passes after replan
    return run


_PREDICATES = {"inbin_a": "test_persistent_mission:inbin_a",
               "inbin_b": "test_persistent_mission:inbin_b"}


class _EpisodicPlanner:
    def plan(self, brief):
        return json.loads(json.dumps({
            "goal": "clear the workspace in ONE persistent episode",
            "nodes": [
                {"id": "clear-a", "skill": "clear", "kind": "segment",
                 "args": {"object": "a"}, "after": []},
                {"id": "verify-a", "skill": "inbin_a", "kind": "verify",
                 "args": {}, "after": ["clear-a"]},
                {"id": "clear-b", "skill": "clear", "kind": "segment",
                 "args": {"object": "b"}, "after": ["verify-a"]},
                {"id": "verify-b", "skill": "inbin_b", "kind": "verify",
                 "args": {}, "after": ["clear-b"]},
            ],
            "verify": [
                {"after": "clear-a", "predicate": "cleared"},
                {"after": "clear-b", "predicate": "cleared"},
            ],
        }, sort_keys=True))

    @property
    def identity(self) -> str:
        return "fake_episodic_planner@v1"


class _FakeScene:
    def snapshot(self, obs):
        return {"frame": "world", "t": 0.0, "nodes": [], "relations": []}


class _FakeExecutor:
    def map(self, fn, items, *, workers):
        return [fn(item) for item in items]


def _kernel(skill_graph=None, log=None) -> Kernel:
    k = Kernel(CAPABILITIES, log=log)
    k.provide("task.planner", _EpisodicPlanner(), ref="tests.fakes:planner")
    k.provide("graph.scene", _FakeScene(), ref="tests.fakes:scene")
    k.provide("graph.skill", skill_graph or InMemorySkillGraph(),
              ref="plugins.graphs:skill_graph_provider")
    k.provide("embodiment.env", epi_embodiment(),
              ref="test_persistent_mission:epi_embodiment")
    k.provide("policy.driver", epi_policy(),
              ref="test_persistent_mission:epi_policy")
    k.provide("exec.rollouts", _FakeExecutor(), ref="tests.fakes:executor")
    return k


def _brief(**over) -> dict:
    b = {"task": "clearall", "catalogue": _CATALOGUE, "oracles": _ORACLES,
         "predicates": dict(_PREDICATES), "episodic": True,
         "episode": {"task": "clear_a", "horizon": 100},
         "segment_specs": _SEGMENT_SPECS}
    b.update(over)
    return b


# ── the plan validates with the new `segment` kind ───────────────────────────────

def test_segment_kind_plan_validates():
    plan = _EpisodicPlanner().plan({})
    ok, msg = validate_plan(plan, _CATALOGUE, _ORACLES)
    assert ok and msg == ""
    kinds = [n.get("kind", "manipulate") for n in plan["nodes"]]
    assert kinds == ["segment", "verify", "segment", "verify"]


# ── sub-goal sequencing on ONE shared world, consequences persist ────────────────

def test_one_persistent_episode_many_subgoals(monkeypatch):
    _fresh()
    monkeypatch.setattr(governed, "governed_segment", _fake_drive)
    out = workload.run(_brief(), _kernel(), seed=42, max_actuations=20)

    assert out["success"] is True and out["replans"] == 0
    # ONE make_env, ONE reset, ONE close for the whole mission -- not per node
    assert EMB.makes == 1 and WORLD.resets == 1 and WORLD.closes == 1
    # both sub-goals ran in the SAME world; the consequences accumulated in it
    assert WORLD.placed == {"a", "b"}
    # the driver was retargeted once per segment (re-aimed at each object), and its
    # grasp clock restarted (poked to 0 on the way in) -- the shared driver threaded
    assert len(DRIVER.retargets) == 2
    # the live-state verify nodes each read the world once and passed
    assert _VERIFY_CALLS == {"a": 1, "b": 1}


# ── the seal: per-sub-goal env-step span off the shared cursor, monotonic ────────

def test_sealing_carries_the_subgoal_step_span(monkeypatch):
    _fresh()
    monkeypatch.setattr(governed, "governed_segment", _fake_drive)
    out = workload.run(_brief(), _kernel(), seed=1, max_actuations=20)

    a = out["nodes"]["clear-a"]["governance"]
    b = out["nodes"]["clear-b"]["governance"]
    # each segment seals the env-step window it consumed off the ONE cursor
    assert a["entered_env_step"] == 0 and a["exited_env_step"] == 10
    assert b["entered_env_step"] == 10 and b["exited_env_step"] == 20
    # the cursor only ever advances -- the world never rewinds between sub-goals
    assert a["exited_env_step"] == b["entered_env_step"]


# ── in-episode SEGMENT retry: a failed sub-goal re-drives the SAME world, no reset ─

def test_failed_segment_retries_in_the_same_episode(monkeypatch):
    _fresh()
    _PLACE_ON_ATTEMPT["b"] = 2  # clear-b fails its first drive, succeeds on retry
    monkeypatch.setattr(governed, "governed_segment", _fake_drive)
    out = workload.run(_brief(), _kernel(), seed=7, max_replans=2, max_actuations=20)

    assert out["success"] is True and out["replans"] == 1
    fault = out["faults"][0]
    assert fault["kind"] == "node_failure" and fault["node"] == "clear-b"
    # the retry re-entered the SAME persistent world: no second make_env, no reset,
    # and clear-a (already placed) was never re-driven -- b was placed on attempt 2
    assert EMB.makes == 1 and WORLD.resets == 1 and WORLD.closes == 1
    assert _ATTEMPTS == {"a": 1, "b": 2} and WORLD.placed == {"a", "b"}


# ── in-episode VERIFY replan: a flaky live verify reroutes into the SAME world ────

def test_flaky_live_verify_replans_in_episode(monkeypatch):
    _fresh()
    monkeypatch.setattr(governed, "governed_segment", _fake_drive)
    brief = _brief(predicates={**_PREDICATES,
                               "inbin_b": "test_persistent_mission:inbin_b_flaky"})
    out = workload.run(brief, _kernel(), seed=3, max_replans=2, max_actuations=20)

    assert out["success"] is True and out["replans"] == 1
    assert out["faults"][0]["node"] == "verify-b"
    # replan re-entered the SAME world (no new env), and only verify-b re-ran --
    # every finished segment was skipped, so the drive count stays at one per object
    assert EMB.makes == 1 and WORLD.resets == 1 and _ATTEMPTS == {"a": 1, "b": 1}


# ── governance mounts PER SEGMENT: an established skill assembles that sub-goal's
#    bundle; a sub-goal with no matching skill seals a null bundle + the span ──────

def _clear_a_record(established=True) -> dict:
    return {
        "kind": "grasp_recovery", "policy": "scripted", "task": "clear_a",
        "heldout_judgement_established": established,
        "preconditions": {"feature": "privileged.stack_z_residual", "op": "gt",
                          "threshold": 0.04, "dwell": 1, "arm_after": 10,
                          "reducer": "value"},
        "recovery": {"name": "regrasp", "strategy": "regrasp",
                     "program": [["descend", 10, 0.0, 0.0], ["close", 14, 0.0, 0.0]],
                     "sensor_sd": 0.02, "max_invocations": 1},
    }


def test_governance_mounts_per_segment(monkeypatch):
    _fresh()
    monkeypatch.setattr(governed, "governed_segment", _fake_drive)
    graph = InMemorySkillGraph()
    graph.publish(_clear_a_record())  # only clear_a has an established skill
    out = workload.run(_brief(), _kernel(graph), seed=5, max_actuations=20)

    a = out["nodes"]["clear-a"]["governance"]
    b = out["nodes"]["clear-b"]["governance"]
    # clear-a's sub-goal task (clear_a) mounted the bundle its skill earned
    assert a["bundle_sha"] is not None and a["skills"] and a["critic_budget"] >= 1
    # clear-b's task (clear_b) has no established skill -> honest null bundle, but the
    # span is still sealed: governance is per-segment, bundle-or-not
    assert b["bundle_sha"] is None and b["skills"] == []
    assert "entered_env_step" in b and "exited_env_step" in b


# ── abort floor: a spent horizon seals an honest partial, never drives a dead env ─

def test_exhausted_horizon_aborts_the_segment_honestly(monkeypatch):
    _fresh()
    monkeypatch.setattr(governed, "governed_segment", _fake_drive)
    # horizon 5 < the 10 steps clear-a consumes: clear-b enters with the cursor spent
    out = workload.run(_brief(episode={"task": "clear_a", "horizon": 5}),
                       _kernel(), seed=9, max_replans=1, max_actuations=20)

    assert out["success"] is False
    # clear-a drove (cursor 0->10); clear-b saw an exhausted cursor and aborted with
    # zero steps rather than stepping a dead env
    b = out["nodes"]["clear-b"]
    assert b["success"] is False and b["steps"] == 0 and b["stages"] == []
    assert _ATTEMPTS == {"a": 1}  # clear-b never reached the drive
    assert WORLD.closes == 1  # still closed exactly once at mission end


# ── a segment with no persistent episode is a loud refusal, not a silent no-op ────

def test_segment_without_episode_refuses():
    ctx = workload.NodeCtx(seed=0, env_ref="x", policy_ref="y", skills=(),
                           nodes_out={}, predicates={}, episode=None)
    with pytest.raises(ValueError, match="persistent episode"):
        workload._segment({"id": "s", "skill": "clear", "args": {"object": "a"}}, ctx)


# ── heterogeneous episodic driver: sub-goals are DIFFERENT behaviours (nav / grasp
#    / place / close), the driver binds the live world + THIS sub-goal's spec and
#    reports its OWN stage terminal -- NOT one retargetable grasp over N objects.
#    This is the robocasa kitchen_thaw shape; the base opts into it via the
#    driver's ``enter_segment`` and reads ``segment_success`` (never object_key /
#    score_terminal). ────────────────────────────────────────────────────────────

class _HetDriver:
    """The composite kitchen driver's contract on a fake: binds the world + spec
    per sub-goal, dispatches nothing (the fake stage is here), reports done off
    ``_done`` keyed by the RE-TASKED sub-goal task."""

    def __init__(self, done: dict) -> None:
        self.k = 0
        self.entered: list[str] = []     # spec.task per segment, in order
        self._done = done

    def observe_once(self, obs):
        return None

    def enter_segment(self, env, spec):
        self.entered.append(spec.task)
        self.k = 0

    def segment_success(self, env) -> bool:
        return bool(self._done.get(self.entered[-1], True))


HET_DRIVER: _HetDriver


class _HetEmbodiment:
    """object_key / success RAISE -- proving the heterogeneous branch calls neither
    (an obs-only retargetable driver would)."""

    def __init__(self, world: _World) -> None:
        self.world = world
        self.makes = 0

    def make_env(self, spec):
        self.makes += 1
        return self.world

    def tasks(self):
        return ("kitchen",)

    def object_key(self, spec):
        raise AssertionError("object_key must NOT be called on the heterogeneous "
                             "segment path (the driver self-targets from the env)")

    def success(self, obs, spec, start_z):
        raise AssertionError("score_terminal/success must NOT be called on the "
                             "heterogeneous path (segment_success is the truth)")


def het_embodiment():
    return _HET_EMB


class _HetPolicy:
    def make_driver(self, spec):
        return HET_DRIVER


def het_policy():
    return _HetPolicy()


def _het_drive(env, obs, driver, spec, bundle, *, step_budget):
    """The faked inner drive: just steps the shared world; success comes from the
    driver's segment_success, not this return (no 'success' key)."""
    env.steps += 5
    return {"obs": env.obs, "steps": 5, "stages": []}


class _HetPlanner:
    def plan(self, brief):
        return json.loads(json.dumps({
            "goal": "a two-sub-goal heterogeneous kitchen mission",
            "nodes": [
                {"id": "walk", "skill": "walk", "kind": "segment", "args": {}, "after": []},
                {"id": "at", "skill": "at", "kind": "verify", "args": {}, "after": ["walk"]},
                {"id": "grab", "skill": "grab", "kind": "segment", "args": {}, "after": ["at"]},
                {"id": "held", "skill": "held", "kind": "verify", "args": {}, "after": ["grab"]},
            ],
            "verify": [{"after": "walk", "predicate": "staged"},
                       {"after": "grab", "predicate": "staged"}],
        }, sort_keys=True))

    @property
    def identity(self):
        return "het_planner@v1"


def _always():
    return lambda node, ctx: {"success": True}


_HET_CATALOGUE = {"walk": {}, "at": {}, "grab": {}, "held": {}}
_HET_ORACLES = ("staged",)
_HET_PREDICATES = {"at": "test_persistent_mission:_always",
                   "held": "test_persistent_mission:_always"}
#: each segment skill -> its distinct sub-goal task (the robocasa SEGMENT_SPECS shape)
_HET_SEGMENT_SPECS = {"walk": {"task": "go"}, "grab": {"task": "pick"}}


def _het_kernel(planner=None) -> Kernel:
    k = Kernel(CAPABILITIES)
    k.provide("task.planner", planner or _HetPlanner(), ref="tests.fakes:planner")
    k.provide("graph.scene", _FakeScene(), ref="tests.fakes:scene")
    k.provide("graph.skill", InMemorySkillGraph(),
              ref="plugins.graphs:skill_graph_provider")
    k.provide("embodiment.env", het_embodiment(),
              ref="test_persistent_mission:het_embodiment")
    k.provide("policy.driver", het_policy(),
              ref="test_persistent_mission:het_policy")
    k.provide("exec.rollouts", _FakeExecutor(), ref="tests.fakes:executor")
    return k


def _het_brief(**over) -> dict:
    b = {"task": "kitchen", "catalogue": _HET_CATALOGUE, "oracles": _HET_ORACLES,
         "predicates": dict(_HET_PREDICATES), "episodic": True,
         "episode": {"task": "kitchen", "horizon": 500},
         "segment_specs": _HET_SEGMENT_SPECS}
    b.update(over)
    return b


def _het_fresh(done: dict) -> None:
    global _HET_EMB, HET_DRIVER
    _HET_EMB = _HetEmbodiment(_World())
    HET_DRIVER = _HetDriver(done)


def test_heterogeneous_segments_bind_env_and_report_own_terminal(monkeypatch):
    _het_fresh(done={"go": True, "pick": True})
    monkeypatch.setattr(governed, "governed_segment", _het_drive)
    out = workload.run(_het_brief(), _het_kernel(), seed=11, max_actuations=20)

    assert out["success"] is True and out["replans"] == 0
    # ONE world for the whole mission (no per-segment make/reset/close)
    assert _HET_EMB.makes == 1 and _HET_EMB.world.resets == 1 and _HET_EMB.world.closes == 1
    # each segment re-tasked its own behaviour and the driver bound it in order --
    # object_key / success (which would have raised) were never on this path
    assert HET_DRIVER.entered == ["go", "pick"]
    # both segment nodes sealed success from segment_success (not score_terminal)
    assert out["nodes"]["walk"]["success"] and out["nodes"]["grab"]["success"]


def test_heterogeneous_segment_failure_replans_in_the_same_world(monkeypatch):
    # the second sub-goal's stage never reaches done -> its segment fails -> the base
    # loop faults and replans into the SAME world (the kitchen_thaw nav-micro shape)
    _het_fresh(done={"go": True, "pick": False})
    monkeypatch.setattr(governed, "governed_segment", _het_drive)
    out = workload.run(_het_brief(), _het_kernel(), seed=12, max_replans=2, max_actuations=20)

    assert out["success"] is False and out["replans"] == 2
    assert out["faults"][0]["kind"] == "node_failure" and out["faults"][0]["node"] == "grab"
    # no new world on replan; walk (already done) was never re-driven, grab retried
    # ONCE as-is -- the deterministic planner's second identical graph is refused
    # (protocol.replan_progress no_progress), never driven
    assert _HET_EMB.makes == 1 and _HET_EMB.world.resets == 1 and _HET_EMB.world.closes == 1
    assert HET_DRIVER.entered.count("go") == 1 and HET_DRIVER.entered.count("pick") == 2
    assert out["faults"][-1]["kind"] == "no_progress"


def test_segment_retry_reuses_valid_graph_without_calling_planner(monkeypatch):
    """One controller miss retries in-place before graph-level replanning."""
    class _FailPickOnce(_HetDriver):
        def __init__(self):
            super().__init__({"go": True, "pick": True})
            self.pick_checks = 0

        def segment_success(self, env) -> bool:
            if self.entered[-1] != "pick":
                return True
            self.pick_checks += 1
            return self.pick_checks > 1

    class _CountingHetPlanner(_HetPlanner):
        def __init__(self):
            self.calls = 0

        def plan(self, brief):
            self.calls += 1
            return super().plan(brief)

    global _HET_EMB, HET_DRIVER
    _HET_EMB = _HetEmbodiment(_World())
    HET_DRIVER = _FailPickOnce()
    planner = _CountingHetPlanner()
    monkeypatch.setattr(governed, "governed_segment", _het_drive)

    out = workload.run(_het_brief(), _het_kernel(planner), seed=13,
                       max_replans=0, max_actuations=20, segment_retries=1)

    assert out["success"] is True
    assert out["replans"] == 0 and planner.calls == 1
    assert HET_DRIVER.entered == ["go", "pick", "pick"]
    assert out["faults"][0]["node"] == "grab"


def test_replanner_observes_the_world_left_by_failed_sampling(monkeypatch):
    _fresh()
    _PLACE_ON_ATTEMPT['b'] = 2
    seen = []
    original_plan = _EpisodicPlanner.plan

    def drive(*args, **kwargs):
        result = _fake_drive(*args, **kwargs)
        WORLD.obs['placed'] = sorted(WORLD.placed)
        return result

    def snapshot(self, obs):
        return {'frame': 'world', 'nodes': [], 'relations': [],
                'has_observation': 'a_pos' in obs, 'placed': list(obs.get('placed', []))}

    def plan(self, brief):
        seen.append(json.loads(json.dumps(brief['scene'])))
        return original_plan(self, brief)

    monkeypatch.setattr(governed, 'governed_segment', drive)
    monkeypatch.setattr(_FakeScene, 'snapshot', snapshot)
    monkeypatch.setattr(_EpisodicPlanner, 'plan', plan)
    out = workload.run(_brief(), _kernel(), seed=42, max_replans=2, max_actuations=20)
    assert out['success'] is True
    assert [scene['has_observation'] for scene in seen] == [True, True]
    assert [scene['placed'] for scene in seen] == [[], ['a']]
    assert WORLD.resets == WORLD.closes == 1
    assert _ATTEMPTS == {'a': 1, 'b': 2}


@pytest.mark.parametrize('phase', ['scene', 'planner', 'dispatch', 'verify_seal', 'terminal_seal'])
def test_persistent_episode_closes_on_each_raising_execution_path(monkeypatch, phase):
    _fresh()
    monkeypatch.setattr(governed, 'governed_segment', _fake_drive)
    error = RuntimeError(f'fixture {phase}')

    def fail(*args, **kwargs):
        raise error

    kernel_log = SessionLog()
    kernel = _kernel(log=kernel_log)
    if phase == 'scene':
        monkeypatch.setattr(_FakeScene, 'snapshot', fail)
    elif phase == 'planner':
        monkeypatch.setattr(_EpisodicPlanner, 'plan', fail)
    elif phase == 'dispatch':
        monkeypatch.setitem(workload._KIND_HANDLERS, 'segment', fail)
    else:
        original = kernel.note
        def note(kind, data):
            if kind == ('task.verify' if phase == 'verify_seal' else 'task.terminal_observation'):
                raise error
            return original(kind, data)
        monkeypatch.setattr(kernel, 'note', note)
    with pytest.raises(RuntimeError) as raised:
        workload.run(_brief(), kernel, seed=42, max_actuations=20)
    assert raised.value is error
    assert WORLD.closes == 1
    assert not any(r['kind'] == 'task.plan_complete' for r in kernel_log.rows())


def test_cleanup_attempts_every_factory_without_hiding_the_execution_error(monkeypatch):
    from types import SimpleNamespace
    _fresh()
    closed, contexts = [], []
    error = RuntimeError('fixture executor failed')

    def close_env():
        WORLD.closes += 1
        closed.append('env')
        raise OSError('fixture environment close failed')

    def close_first():
        closed.append('first')
        raise OSError('fixture factory close failed')

    def dispatch(node, ctx):
        contexts.append(ctx.episode)
        ctx.episode.factories.update(first=SimpleNamespace(close=close_first),
                                     second=SimpleNamespace(close=lambda: closed.append('second')))
        raise error

    monkeypatch.setattr(WORLD, 'close', close_env)
    monkeypatch.setitem(workload._KIND_HANDLERS, 'segment', dispatch)
    with pytest.raises(RuntimeError) as raised:
        workload.run(_brief(), _kernel(), seed=42)
    assert raised.value is error
    assert 'cleanup also failed' in str(error.__notes__)
    assert closed == ['env', 'first', 'second']
    contexts[0].close()
    assert WORLD.closes == 1 and closed == ['env', 'first', 'second']


def test_cleanup_error_on_success_is_not_sealed_as_a_completed_plan(monkeypatch):
    _fresh()
    monkeypatch.setattr(governed, 'governed_segment', _fake_drive)
    def fail_close():
        WORLD.closes += 1
        raise OSError('fixture close failed')
    monkeypatch.setattr(WORLD, 'close', fail_close)
    kernel_log = SessionLog()
    kernel = _kernel(log=kernel_log)
    with pytest.raises(OSError, match='fixture close failed'):
        workload.run(_brief(), kernel, seed=42, max_actuations=20)
    assert WORLD.closes == 1
    assert not any(r['kind'] == 'task.plan_complete' for r in kernel_log.rows())


@pytest.mark.parametrize('phase', ['reset', 'driver', 'observe'])
def test_episode_initialization_failure_releases_its_acquired_environment(monkeypatch, phase):
    _fresh()
    error = RuntimeError(f'fixture {phase} initialization failed')
    def fail(*args, **kwargs):
        raise error
    if phase == 'reset':
        monkeypatch.setattr(WORLD, 'reset', fail)
    elif phase == 'driver':
        monkeypatch.setattr(governed, 'make_driver', fail)
    else:
        monkeypatch.setattr(DRIVER, 'observe_once', fail)
    with pytest.raises(RuntimeError) as raised:
        workload.run(_brief(), _kernel(), seed=42)
    assert raised.value is error
    assert EMB.makes == WORLD.closes == 1
