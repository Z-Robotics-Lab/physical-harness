"""Shared stage machinery for the three composite missions (recycle_cans /
pack_lunch / steam_prep) -- everything they need beyond ``drivers.py``, WITHOUT
touching it: an object-addressed navigate, a point-targeted place family, and the
generic composite ``policy.driver`` adapter each mission's driver file arms with
its own stage table.

Same discipline as ``drivers.py``: closed-loop P controllers over LIVE privileged
state, deterministic given the seeded scene, robocasa imported lazily inside
methods so every module here stays base-importable. The composite adapter is a
verbatim sibling of ``kitchen_driver.KitchenThawDriver`` (the M7
episodic-segment protocol: enter_segment / act / exhausted / segment_success),
parameterised by stage table instead of copied per mission.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plugins.embodiment_robocasa import drivers as D


class NavToObjectDriver(D.NavigateDriver):
    """NavigateDriver whose dock is computed NEAR A NAMED OBJECT on the fixture
    (``compute_robot_base_placement_pose(..., ref_object=...)``): a long counter
    has one generic dock but the can/tupperware sits at a specific run of it.

    The EMPTY leg is heading-first: while far, drive FACING the travel direction
    (the wheeled base cannot strafe -- holding the dock yaw across a long
    cross-kitchen leg turns most of the error into a dead vy channel), settle to
    the dock yaw inside the last stretch; on a stall (window displacement under
    2 cm over 20 steps: wedged on the spawn dock's furniture) reverse straight
    back 25 steps and resume -- under the parent's shared plateau watchdog
    (``_watch``: no approach for stall_k steps far from the dock seals
    "nav_stall" for the planner's redock, the reverse included). Measured on recycle_cans scratch seeds: the
    parent's hold-dock-yaw empty leg moved 0.03-0.13 m on 4/6 cross-kitchen
    legs; heading-first completes a 4.2 m leg (seed 4243, 136 steps) and leaves
    near-dock legs untouched. No path planning -- a mid-route blocking counter
    is still the charter's honest failure surface. The carry leg is the parent's
    stow+arc recipe verbatim."""

    #: stall detector + recovery knobs (probe-measured; geometry, not per-scene).
    STALL_WIN = 20
    STALL_EPS = 0.02
    REV_STEPS = 25
    REV_V = -0.6
    HEADING_FAR = 0.6   # beyond this, face the travel direction, not the dock

    def __init__(self, fixture_name: str, obj_name: str, carry: bool = False):
        super().__init__(fixture_name, carry=carry)
        self.obj_name = obj_name
        self._hist: list = []
        self._rev = 0

    def _target(self, env):
        if self._goal is None:
            from robocasa.utils.env_utils import compute_robot_base_placement_pose

            fx = D._fixture(env, self.fixture_name)
            pos, ori = compute_robot_base_placement_pose(
                env, fx, ref_object=self.obj_name)
            self._goal = (np.asarray(pos[:2], float), float(ori[2]))
        return self._goal

    def act(self, env, obs):
        if self.carry:
            return super().act(env, obs)
        gxy, gyaw = self._target(env)
        xy, _ = D._base_pose(env)
        self._hist.append(xy.copy())
        vec = np.asarray(gxy) - xy
        d = float(np.linalg.norm(vec))
        self._watch(env, d)
        if self._rev > 0:
            self._rev -= 1
        elif (len(self._hist) > self.STALL_WIN and d > 0.45
              and np.linalg.norm(self._hist[-1]
                                 - self._hist[-1 - self.STALL_WIN]) < self.STALL_EPS):
            self._rev = self.REV_STEPS
            self._hist.clear()
        if self._rev > 0:
            a = D._zero()
            a[7] = self.REV_V
            a[D.MODE] = D.GRIP_CLOSE  # +1 == base mode
            return a
        heading = (float(np.arctan2(vec[1], vec[0]))
                   if d > self.HEADING_FAR else gyaw)
        return D._base_action(env, gxy, heading, grip=D.GRIP_OPEN)


class PointPlaceDriver:
    """Place the held object at a WORLD POINT: over -> lower -> release ->
    retreat, the ``drivers.PlaceDriver`` phase chain re-targeted from a fixture's
    interior centroid to ``_drop_point(env)`` (subclass-supplied). done() is the
    subclass's own live truth."""

    RELEASE_TICKS = 6

    def __init__(self, obj_name: str):
        t = D.tunables()
        self.obj_name = obj_name
        self.phase = "over"
        self._ticks = 0
        self._reach_tol = t["reach_tol"]
        self._over_dz = t["drop_over_dz"]
        self._stall = D.StallDetector(t["stall_k"])
        self.failure_mode = None  # "reach_stall": over/lower made no progress
        self._trace = D.Trace()   # over/lower geometry (diagnostics only)

    # -- subclass surface ------------------------------------------------------
    def _drop_point(self, env) -> np.ndarray:
        raise NotImplementedError

    def done(self, env) -> bool:
        raise NotImplementedError

    def diagnostics(self, env) -> dict[str, Any]:
        """Live terminal details used to explain a bounded place failure."""
        return {"phase": self.phase, "failure_mode": self.failure_mode,
                "trace": self._trace.dump(env)}

    # -- the shared phase chain ------------------------------------------------
    def act(self, env, obs):
        c = np.asarray(self._drop_point(env), float)
        eef = D._eef(env)
        self._trace.step += 1
        if self.phase == "over":
            goal = np.array([c[0], c[1], c[2] + self._over_dz])
            self._trace.at("start", env, goal)
            if np.linalg.norm((eef - goal)[:2]) < self._reach_tol:
                self.phase = "lower"
            elif self._stall.update(float(np.linalg.norm(goal - eef))):
                self._trace.at("stall", env)
                self.failure_mode = "reach_stall"
            return D._arm_action(env, goal, D.GRIP_CLOSE)
        if self.phase == "lower":
            self._trace.at("start", env, c)
            if eef[2] - c[2] < 0.04:
                self.phase = "release"
            elif self._stall.update(float(np.linalg.norm(c - eef))):
                self._trace.at("stall", env)
                self.failure_mode = "reach_stall"
            return D._arm_action(env, c, D.GRIP_CLOSE)
        if self.phase == "release":
            self._ticks += 1
            if self._ticks > self.RELEASE_TICKS:
                self.phase = "retreat"
            return D._arm_action(env, np.array([c[0], c[1], c[2] + 0.02]),
                                 D.GRIP_OPEN)
        # retreat: up AND back toward the base, gripper open -- straight-up alone
        # leaves the eef ~0.2 m over the dropped object, inside gripper_obj_far's
        # 0.25 m release gate (measured: the drop landed, only the gate held out)
        xy, psi = D._base_pose(env)
        back = xy + 0.35 * np.array([np.cos(psi), np.sin(psi)])
        return D._arm_action(env, np.array([back[0], back[1], c[2] + 0.35]),
                             D.GRIP_OPEN)


class ReceptaclePlaceDriver(PointPlaceDriver):
    """Place into a receptacle OBJECT (tupperware/pot) -- target its LIVE body
    pose (a nudged receptacle is followed), drop from just above its rim."""

    RIM_DZ = 0.10  # release height above the receptacle centre

    def __init__(self, obj_name: str, receptacle: str):
        super().__init__(obj_name)
        self.receptacle = receptacle

    def _drop_point(self, env) -> np.ndarray:
        p = D._obj_pos(env, self.receptacle)
        return np.array([p[0], p[1], p[2] + self.RIM_DZ])

    def done(self, env) -> bool:
        import robocasa.utils.object_utils as OU

        return bool(OU.check_obj_in_receptacle(env, self.obj_name, self.receptacle)
                    and OU.gripper_obj_far(env, obj_name=self.obj_name))

    def diagnostics(self, env) -> dict[str, Any]:
        import robocasa.utils.object_utils as OU

        return {
            **super().diagnostics(env),
            "inside": bool(OU.check_obj_in_receptacle(
                env, self.obj_name, self.receptacle)),
            "released": bool(OU.gripper_obj_far(env, obj_name=self.obj_name)),
        }


class CompositeStageDriver:
    """The generic composite ``policy.driver`` for a heterogeneous persistent
    mission: one instance threaded through the whole episode, re-armed per
    sub-goal by ``enter_segment`` (spec.task -> its stage driver via the
    mission's own table). Protocol-identical to ``kitchen_driver.
    KitchenThawDriver`` -- factored here because three missions would otherwise
    carry three verbatim copies."""

    def __init__(self, stages: dict[str, tuple[Any, int]], identity: str) -> None:
        self._stages = stages
        self._identity = identity
        self._env: Any = None
        self._stage: Any = None
        self._cap: int = 0
        self.k: int = 0
        #: an arm's executor bound for THIS segment by ``enter_segment(..., executor=)``
        #: (same seam as KitchenThawDriver): inproc = a code-as-policy card that binds
        #: the live env itself, raw obs in / 12-dim action out; else the pi0.5 contract.
        self._executor: Any = None
        self._native: bool = False
        self._prompt: str = ""

    # --- obs-only PolicyDriver surface ---------------------------------------
    def observe_once(self, obs) -> np.ndarray:
        return np.zeros(D.ADIM)

    def act(self, obs) -> np.ndarray:
        if self._native:
            a = self._executor.act(obs)
        elif self._executor is not None:
            from plugins.embodiment_robocasa import vla_io
            a = vla_io.lerobot_to_env(self._executor.act(vla_io.build_obs(obs, self._prompt)))
        else:
            a = self._stage.act(self._env, obs)
        self.k += 1
        return a

    @property
    def exhausted(self) -> bool:
        if self._stage is None:
            return True
        return (self.k >= self._cap or bool(self._stage.done(self._env))
                or getattr(self._stage, "failure_mode", None) is not None)

    def retarget(self, target) -> None:
        """No-op: the stage drivers self-target off the live env, never a pose."""

    def on_handback(self) -> None:
        """No-op: no critic-recovery bundle mounts over these sub-goals yet."""

    def make_recovery(self, recovery, obs, draw, spec):
        """The 12-dim recovery actor for the ACTIVE stage (governed.py's
        ``make_recovery`` seam, same shape as KitchenThawDriver's)."""
        from plugins.embodiment_robocasa.recovery import RobocasaRecoveryActor

        return RobocasaRecoveryActor.for_stage(self._env, self._stage, recovery)

    @property
    def identity(self) -> str:
        return self._identity

    # --- the episodic-segment protocol ----------------------------------------
    def enter_segment(self, env, spec, executor: Any = None) -> None:
        """Arm ``spec.task``'s stage driver on the live env; ``executor`` (an arm's
        policy driver) takes the actions over, the stage keeps done() and the cap."""
        task = getattr(spec, "task", None)
        if task not in self._stages:
            raise ValueError(
                f"{self._identity} has no stage driver for sub-goal task {task!r}; "
                f"SEGMENT_SPECS must re-task each segment to one of "
                f"{sorted(self._stages)}")
        factory, cap = self._stages[task]
        self._env = env
        self._stage = factory()
        self._cap = D.tunables()["segment_cap"] or cap
        self.k = 0
        self._executor = executor
        self._native = executor is not None and executor.handshake()["transport"] == "inproc"
        if executor is not None:
            executor.reset()  # a chunk computed for another situation is stale
            self._prompt = str(env.get_ep_meta().get("lang") or task)
        if self._native:  # the stage's target (make_recovery's seam) is the executor's too
            executor.bind(env, target=getattr(self._stage, "obj_name", None))

    def segment_success(self, env) -> bool:
        return bool(self._stage.done(env))

    def segment_diagnostics(self, env) -> dict[str, Any]:
        """Stage-owned terminal details (never used for control) plus the two
        keys every robocasa segment seals: ``failure_mode`` ("reach_stall" or
        None) and ``tunables_sha`` (the knobs this segment ran under)."""
        return D.stage_diagnostics(self._stage, env)


class CompositePolicies:
    """Layer 3 ``harness.contracts.PolicyFactory``: one mount, one composite
    driver armed with the owning mission's stage table."""

    def __init__(self, stages: dict[str, tuple[Any, int]], identity: str,
                 tunables: dict | None = None, tunable_hints: dict | None = None) -> None:
        self._stages = stages
        self._identity = identity
        D.mount_tunables(tunables)  # the card's [tunables] (+ an evolve trial's overlay)
        # tunable_hints rides the same mount (manifest.mount_params); only evolve's
        # proposer reads it -- accepted so the card's mount shape stays one dict.

    def make_driver(self, spec: Any) -> CompositeStageDriver:
        return CompositeStageDriver(self._stages, self._identity)


if __name__ == "__main__":
    # Base-importable self-check: the composite adapter's dispatch + cap floor on
    # a fake stage (the kitchen_driver self-check, against the factored class).
    class _FakeStage:
        def __init__(self, done_at=3):
            self.steps, self.done_at = 0, done_at

        def act(self, env, obs):
            self.steps += 1
            return np.zeros(D.ADIM)

        def done(self, env):
            return self.steps >= self.done_at

    stages = {"probe": (lambda: _FakeStage(), 250),
              "stuck": (lambda: _FakeStage(done_at=10 ** 9), 5)}
    drv = CompositePolicies(stages, "probe@v1").make_driver(object())
    assert drv.exhausted is True and drv.identity == "probe@v1"
    assert drv.observe_once({}).shape == (D.ADIM,)

    class _S:
        task = "probe"
    drv.enter_segment(object(), _S())
    assert drv.k == 0 and not drv.exhausted
    while not drv.exhausted:
        drv.act({})
    assert drv.k == 3 and drv.segment_success(object()) is True

    class _S2:
        task = "stuck"
    drv.enter_segment(object(), _S2())
    while not drv.exhausted:
        drv.act({})
    assert drv.k == 5 and drv.segment_success(object()) is False

    class _S3:
        task = "nope"
    try:
        drv.enter_segment(object(), _S3())
    except ValueError:
        pass
    else:
        raise AssertionError("unknown sub-goal task must fail loudly")

    # PointPlaceDriver phase chain shape on a fake drop point (no sim): the
    # subclass surface is the only robocasa-touching part.
    class _Probe(PointPlaceDriver):
        def _drop_point(self, env):
            return np.array([0.0, 0.0, 1.0])

        def done(self, env):
            return False

    print("plugins/embodiment_robocasa/stage_extras.py self-check OK")
