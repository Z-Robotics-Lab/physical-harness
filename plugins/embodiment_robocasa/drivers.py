"""Scripted, frozen, per-stage drivers for the kitchen_thaw mission (PandaOmron).

RoboCasa's mobile manipulator takes a 12-dim action (install report §3.4):
arm OSC delta (0:6), gripper (6), base vx/vy/wyaw (7:10), torso (10), and a
base_mode switch (11): +1 = base drives / arm follows, -1 = arm mode. These
drivers are the robosuite card's phased-scripted pattern ported to that action:
each is a closed-loop P controller reading LIVE privileged state (base pose, eef
site, object bodies, fixture geoms) -- privileged is fine here, these are the
oracle scripted policies the harness governs, not learned policies under test.

Axes are EMPIRICAL, not copied from any demo (the report warns robosuite-master
re-signed the mobile base's forward axis). Measured in this venv (seed 7) with
SMALL, drift-subtracted deltas -- a saturated 5-step probe reads a rotated/curved
frame off the wheeled base and lies; the clean in-scene reading is:
  * base, base_mode=+1: world velocity = Rz(psi) @ (vx, vy) -- at psi=0, +vx->+X,
    +vy->+Y. +wyaw = +yaw (CCW). Plain base frame, no offset.
  * arm, base_mode=-1: the OSC delta frame is the base frame, world_from_osc =
    Rz(psi): at psi=0 +ax->+X, +ay->+Y, +az->+Z.
  * both are therefore commanded from a world error by Rz(-psi) @ err (xy; z is
    shared). gripper: +1 closes, -1 opens.

Each driver exposes act(env, obs) -> (12,) action and done(env) -> bool. A stage
runs until done() or a step budget (run_stage). Everything is deterministic given
the env's seeded scene: no rng in the controllers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from functools import cache
from pathlib import Path

import numpy as np

# 12-dim action layout (install report §3.4): arm OSC 0:6, gripper 6,
# base vx/vy/wyaw 7:10, torso 10, base_mode 11.
GRIP = 6
TORSO = 10
MODE = 11
ADIM = 12

GRIP_CLOSE = 1.0
GRIP_OPEN = -1.0

# Navigate success tolerance == NavigateKitchen._check_success (kitchen_navigate.py).
NAV_POS_TOL = 0.20
NAV_ORI_COS = 0.98


# ---- stage tunables (the card manifest's [tunables] table) -------------------

_MANIFEST = Path(__file__).with_name("manifest.toml")


#: The ``tunables`` a provider of this card was mounted with (harness.manifest.
#: mount_params folds the manifest table + PH_MOUNT_PARAMS_OVERRIDE into every
#: provider's params; the provider hands it here). One process-wide overlay.
_MOUNTED: dict = {}


def mount_tunables(overlay: dict | None) -> None:
    """Install the mounted ``tunables`` overlay every stage driver reads."""
    _MOUNTED.clear()
    _MOUNTED.update(overlay or {})


@cache
def _tunables(*overlays: str) -> dict:
    base = tomllib.loads(_MANIFEST.read_text()).get("tunables", {})
    for overlay in overlays:
        extra = json.loads(overlay) if overlay else {}
        if unknown := set(extra) - set(base):
            raise KeyError(f"unknown tunables {sorted(unknown)}; known: {sorted(base)}")
        base = {**base, **{k: type(base[k])(v) for k, v in extra.items()}}
    return base


def tunables() -> dict:
    """Effective stage tunables: manifest ``[tunables]`` defaults, overlaid by the
    mounted ``tunables`` (``mount_tunables``: an evolve trial's perturbation), then
    by the ``PH_TUNABLES`` env var (a JSON object). Unknown keys refuse loudly."""
    # ponytail: process-wide overlays, not per-episode; thread through the
    # driver instance if two arms ever need different tunables in one process.
    return _tunables(json.dumps(_MOUNTED, sort_keys=True) if _MOUNTED else "",
                     os.environ.get("PH_TUNABLES", ""))


def tunables_sha() -> str:
    """Content id of the EFFECTIVE tunables -- sealed with every segment's
    diagnostics so a tuned run is distinguishable from the default one."""
    return hashlib.sha256(
        json.dumps(tunables(), sort_keys=True).encode()).hexdigest()[:16]


class StallDetector:
    """eef-to-target progress watchdog: ``update(dist)`` is True once the best
    distance has not improved by ``eps`` for ``k`` consecutive steps. A stage
    that stalls fails its segment early with failure_mode "reach_stall" instead
    of burning the whole step cap against furniture/reach limits."""

    def __init__(self, k: int, eps: float = 0.002):
        self.k, self.eps = int(k), eps
        self.best = float("inf")
        self.since = 0

    def update(self, dist: float) -> bool:
        if dist < self.best - self.eps:
            self.best, self.since = float(dist), 0
        else:
            self.since += 1
        return self.since >= self.k


# ---- live-state readers (privileged; scripted-oracle side) -------------------

def _base_pose(env):
    """(x, y) world position and yaw of the mobile base body."""
    import robosuite.utils.transform_utils as T

    bid = env.sim.model.body_name2id("mobilebase0_base")
    p = np.asarray(env.sim.data.body_xpos[bid])
    yaw = float(T.mat2euler(np.asarray(env.sim.data.body_xmat[bid]).reshape(3, 3))[2])
    return p[:2].copy(), yaw


def _fixture(env, name):
    """Resolve a fixture: prefer the task's registered ref (env.microwave /
    env.fridge) over env.get_fixture, whose fuzzy name match returns the
    HousingCabinet for 'microwave' instead of the Microwave itself."""
    fx = getattr(env, name, None)
    return fx if fx is not None else env.get_fixture(name)


def _rot_world_to_base(err_xy, psi):
    """Rz(-psi) @ err_xy: a world xy error expressed in the base/OSC frame."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([c * err_xy[0] + s * err_xy[1],
                     -s * err_xy[0] + c * err_xy[1]])


def _eef(env):
    """World position of the right-hand eef site."""
    return np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]).copy()


def _obj_pos(env, name):
    return np.asarray(env.sim.data.body_xpos[env.obj_body_id[name]]).copy()


def _geom_pos(env, geom):
    gid = env.sim.model.geom_name2id(geom)
    return np.asarray(env.sim.data.geom_xpos[gid]).copy()


def _zero():
    a = np.zeros(ADIM)
    return a


def _torso_q(env):
    """Current torso-lift joint position (slide, range 0..0.34 m)."""
    m = env.sim.model
    j = m.joint_name2id("mobilebase0_joint_torso_height")
    return float(env.sim.data.qpos[m.jnt_qposadr[j]])


def _torso_cmd(env, target):
    """Torso channel value servoing the lift joint to `target` (P on qpos).

    Measured (this venv, seed 200000): the JOINT_POSITION torso channel acts as
    a rate command (+1 held sweeps the full 0.34 m range in ~40 steps, ~8.5 mm/
    step), holds position at 0, and works identically in arm AND base mode --
    the arm OSC rides along instead of fighting it (eef z tracks the lift).
    """
    return float(np.clip((float(target) - _torso_q(env)) * 25.0, -1.0, 1.0))


# ---- shared primitives -------------------------------------------------------

def _arm_action(env, goal_world, grip, kp=10.0, dyaw=0.0):
    """base_mode=-1 arm action driving the eef toward goal_world (P control).

    World error is rotated into the OSC base frame (Rz(-psi)) so the
    command is axis-correct at any base yaw; scaled by kp and clipped to the
    controller's [-1, 1] (== +-0.05 m/step). dyaw drives the wrist-yaw rotation
    channel (a[5]; measured: positive == CCW about world z) -- the grasp stage
    uses it to align the finger-opening axis across the object's thin side.
    """
    err = np.asarray(goal_world, float) - _eef(env)
    _, psi = _base_pose(env)
    bxy = _rot_world_to_base(err[:2], psi)
    cmd = np.array([bxy[0], bxy[1], err[2]])
    a = _zero()
    a[0:3] = np.clip(cmd * kp, -1.0, 1.0)
    a[5] = float(np.clip(dyaw, -1.0, 1.0))
    a[GRIP] = grip
    a[MODE] = GRIP_OPEN  # -1 == arm mode
    return a


def _base_action(env, goal_xy, goal_yaw, grip=GRIP_OPEN, kp=2.5, kyaw=4.0):
    """base_mode=+1 velocity action driving the base toward (goal_xy, goal_yaw).

    World xy error is rotated into the base frame (Rz(-psi)) to command (vx, vy);
    wyaw closes the yaw error. Yaw is held tightly (kyaw > kp) so the velocity
    frame stays fixed while translating -- a wandering yaw curves the path. In
    base mode the arm follows the base, so a carried object stays put ONLY if the
    gripper is commanded closed -- pass grip=GRIP_CLOSE when navigating loaded.
    """
    xy, psi = _base_pose(env)
    vxy = _rot_world_to_base(np.asarray(goal_xy, float) - xy, psi)
    dyaw = (float(goal_yaw) - psi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi,pi]
    a = _zero()
    a[7] = np.clip(vxy[0] * kp, -1.0, 1.0)
    a[8] = np.clip(vxy[1] * kp, -1.0, 1.0)
    a[9] = np.clip(dyaw * kyaw, -1.0, 1.0)
    a[GRIP] = grip
    a[MODE] = GRIP_CLOSE  # +1 == base mode
    return a


# ---- stage drivers -----------------------------------------------------------

class NavigateDriver:
    """Drive the base to a fixture's docking pose (compute_robot_base_placement_pose).

    Success mirrors NavigateKitchen: base within NAV_POS_TOL (0.20 m) of the dock
    xy AND facing it (cos(dyaw) >= 0.98). No path planning -- straight velocity
    servo; if furniture blocks the line, that is the honest failure surface.

    carry=True is the loaded-transport variant (carry-probe, local-archive/
    robocasa-adapt/carry-probe.md): a saturated base command whips the extended
    arm and strips even a REAL grasp within ~10 steps, so the loaded leg (1)
    STOWS -- a gentle, command-capped arm retract toward a body-hugging carry
    pose -- then (2) ARC-drives with base velocity and yaw-rate capped while the
    arm channels actively counter-sweep the eef back to the carry pose each
    step, and (3) stops at a STANDOFF instead of docking.
    Mode-flip itself is safe: HybridMobileBase sets the arm OSC to
    goal_update_mode="desired" in base mode, so the arm HOLDS its last desired
    goal under translation (measured: 60 zero-velocity base steps drift the eef
    <2 cm) -- but that goal is world-anchored under ROTATION, hence the
    counter-sweep.

    Known ceiling, measured (14 real grasps, train split): with no path planner
    the drive is a straight velocity servo into whatever furniture lies on the
    line, and ~40% of legs jam short of the standoff with the base saturated and
    moving <2 mm/step. Raising VCAP does not touch that class; a planner does.
    """

    #: carry calibration knobs. These are ACTUATOR and DISTRIBUTION numbers, not
    #: scene numbers: VCAP/WARC come from the base joints' own measured
    #: command->motion curve and CARRY_Z from the demos' hand-off band, so
    #: re-measure them if the base model or the demo set changes, never per scene.
    CARRY_FWD = 0.40   # stow target this far in front of base centre
    CARRY_LAT = -0.15  # ... on the arm's mount side
    CARRY_Z = 1.25     # carry height: the arm cannot pull inside ~0.7 m at shelf
                       # height (stalls, meat leads the base into the target
                       # fixture and strips) but reaches ~0.36 m once lowered --
                       # above counter tops, below the shelf lips.
                       # 1.00 was chosen for that retraction authority alone and
                       # sat BELOW the whole band the demos hand `place` over in
                       # (meat_z 1.341 +- 0.100, range [1.120, 1.520], n=29,
                       # runs/pi05-campaign/gate2_diag/demo_place_windows.json): no correctly
                       # stowed episode could be in distribution on that axis.
                       # 1.25 is inside the band and retraction still converges
                       # (tuning block: meat_z in the demo range at hand-off
                       # 5/14 -> 9/14, held 9/14 unchanged).
    STOW_TOL = 0.06
    STOW_STEPS = 80    # stow is best-effort: converge or spend this, then drive
    ARM_CAP = 0.3      # per-step arm command cap while stowing (gentle)
    VCAP = 0.60        # base velocity cap while loaded. The base's three drive
                       # joints carry frictionloss=250 (omron_mobile_base.xml),
                       # so command->motion has a hard DEAD ZONE. Measured
                       # in-scene, 60 steps per point: cmd 0.12 -> 0.12 mm/step,
                       # 0.20 -> 0.20, 0.35 -> 3.5, 0.50 -> 12.2, 0.75 -> 24.4.
                       # 0.35 sat on the knee, and cost twice over: it bought
                       # only ~1.1 m of travel out of the 450-step budget (legs
                       # measured 0.7-2.9 m from the dock), AND stick-slip at
                       # the knee stripped the cargo on exactly the runs that
                       # did reach the dock -- all three baseline arrivals had
                       # the meat on the floor. Above the knee the leg is both
                       # faster and gentler. TRANSLATION is safe to raise this
                       # way; rotation is not -- see WARC.
    WARC = 0.12        # loaded yaw-rate cap: turn only as a slow ARC while
                       # translating. Unlike the drive joints the yaw hinge has
                       # NO dead zone (measured linear from cmd 0.05: 1.8 ->
                       # 4.3 -> 15.6 -> 27.1 mrad/step at 0.05/0.12/0.35/0.50),
                       # so this cap is pure cargo protection and raising it
                       # only strips: 0.12 -> 0.30 on the tuning block took the
                       # meat from held-at-hand-off 9/13 down to 5/13 while
                       # buying nothing the higher VCAP had not already bought.
                       # Yaw is the lever that sweeps the eef; translation is not.
    CARRY_STOP = 0.65  # loaded standoff from the dock: driving the last ~0.4 m
                       # rams the carried object into the target appliance's
                       # face (measured drop at dist~0.39); the place stage's
                       # ARM covers that final reach, the base need not.
                       # 0.50 measured too tight across random kitchens: loaded
                       # runs plateau 0.6-0.8 from the dock (counter/furniture
                       # edge), burning the cap just short of the old gate
    CARRY_NEAR = 0.85  # stall-arrival band: when progress has physically
                       # converged (blocked by the counter edge) within this of
                       # the dock, the leg is DONE -- grinding on strips the
                       # cargo (measured: near-gate drops at steps 252-416 on
                       # runs that had plateaued at 0.67-0.78 long before)

    def __init__(self, fixture_name, carry=False):
        self.fixture_name = fixture_name
        self.carry = carry          # hold the gripper closed to keep a grasped object
        self._goal = None
        self._stow_left = self.STOW_STEPS if carry else 0
        # progress watchdog for BOTH legs (_watch): the base-to-dock distance
        # not improving by 2 cm for stall_k steps. Near the dock that is a
        # stall-ARRIVAL; farther out it is a "nav_stall" -- the segment fails
        # early for a redock_retry instead of burning its cap wedged on furniture.
        self._stall = StallDetector(tunables()["stall_k"], eps=0.02)
        self._blocked = False
        self.failure_mode = None

    def _target(self, env):
        if self._goal is None:
            from robocasa.utils.env_utils import compute_robot_base_placement_pose

            fx = _fixture(env, self.fixture_name)
            pos, ori = compute_robot_base_placement_pose(env, fx)
            self._goal = (np.asarray(pos[:2], float), float(ori[2]))
        return self._goal

    def _stow_action(self, env):
        """One gentle arm step toward the carry pose; None once stowed/spent.

        TWO-STAGE, ALL WHILE STATIONARY: retract horizontally at the lifted
        height first, then lower to CARRY_Z (with the torso riding down) only
        once pulled back near the body. Both orderings of the alternatives were
        measured and lose: lowering while still over the shelf drags the slab
        into the shelf lip (strips in the first ~100 transport steps), and
        deferring the lowering into the DRIVE -- or skipping it and carrying at
        lift height -- strips mid-transport (drops at steps 133-378: moving
        while easing z, and the high extended carry itself, are both unstable).
        """
        if self._stow_left <= 0:
            return None
        xy, psi = _base_pose(env)
        c, s = np.cos(psi), np.sin(psi)
        txy = xy + np.array([c * self.CARRY_FWD - s * self.CARRY_LAT,
                             s * self.CARRY_FWD + c * self.CARRY_LAT])
        eef = _eef(env)
        retracted = np.linalg.norm(eef[:2] - txy) < 0.12
        tz = self.CARRY_Z if retracted else eef[2]     # stage 1: hold height
        if (np.linalg.norm(eef[:2] - txy) < self.STOW_TOL
                and abs(eef[2] - self.CARRY_Z) < self.STOW_TOL):
            self._stow_left = 0
            return None
        self._stow_left -= 1
        a = _arm_action(env, np.array([txy[0], txy[1], tz]),
                        GRIP_CLOSE, kp=6.0)
        a[0:3] = np.clip(a[0:3], -self.ARM_CAP, self.ARM_CAP)
        a[TORSO] = _torso_cmd(env, 0.0) if retracted else 0.0
        return a

    def _watch(self, d: float) -> None:
        """The one plateau rule every nav leg (loaded or not, subclassed or not)
        feeds its base-to-dock distance: a stall inside the arrival band is
        ``_blocked`` (the loaded done() accepts it; unloaded that band is the
        NAV_POS_TOL gate itself, only the yaw still settling), a stall farther
        out seals failure_mode "nav_stall" so the planner can redock."""
        if self._stall.update(d):
            if d <= (self.CARRY_NEAR if self.carry else NAV_POS_TOL):
                self._blocked = True
            else:
                self.failure_mode = "nav_stall"

    def act(self, env, obs):
        gxy, gyaw = self._target(env)
        if not self.carry:
            self._watch(float(np.linalg.norm(np.asarray(gxy, float) - _base_pose(env)[0])))
            return _base_action(env, gxy, gyaw, grip=GRIP_OPEN)

        # Loaded transport. Measured (carry-probe traces): in base mode the
        # arm's held goal is world-anchored under ROTATION -- translation
        # carries the eef along (rel-base pose steady), but yaw sweeps the eef
        # laterally on the ~0.7 m lever and levers the object out of the
        # fingers. That asymmetry is the whole recipe: drive HARD (VCAP, above
        # the joints' friction dead zone) and turn SOFT (WARC), as a slow arc,
        # with an ACTIVE arm counter-sweep each step. The base is holonomic --
        # three independent forward/side/yaw joints (omron_mobile_base.xml) --
        # so the arc is not a kinematic necessity, only a way to keep the eef
        # pointing where the place stage will need it.
        stow = self._stow_action(env)
        if stow is not None:
            return stow
        xy, psi = _base_pose(env)
        # A straight BACK-OUT off the fridge face used to run here (a[7]=-0.25
        # until the base had moved 0.45 m or 60 ticks were spent). Deleted: at
        # -0.25 the drive command is BELOW the joints' friction dead zone (see
        # VCAP), so it moved the base 10-15 mm and always exited on the tick
        # count -- 61 of the leg's 450 steps for a centimetre. It never cleared
        # anything; the drive below leaves the fridge face just as well.
        vec = np.asarray(gxy, float) - xy
        heading = float(np.arctan2(vec[1], vec[0]))
        a = _base_action(env, gxy, heading, grip=GRIP_CLOSE)
        a[7:9] = np.clip(a[7:9], -self.VCAP, self.VCAP)
        a[9] = float(np.clip(a[9], -self.WARC, self.WARC))
        self._watch(float(np.linalg.norm(vec)))
        # NO z easing while moving: all height changes happen in the
        # stationary stow -- easing the eef down mid-drive was measured to
        # strip the cargo (drops at steps 144-332 on seeds that survived
        # with the height held through the drive).
        # active counter-sweep: in base mode the arm channels still ADD deltas
        # to the held (base-frame) goal, so pull the swept eef back toward the
        # carry pose WHILE driving -- the poor-man's whole-body coordination.
        rel = _rot_world_to_base(_eef(env)[:2] - xy, psi)
        err = np.array([self.CARRY_FWD, self.CARRY_LAT]) - rel
        a[0:2] = np.clip(err * 4.0, -self.ARM_CAP, self.ARM_CAP)
        return a

    def done(self, env):
        (gxy, gyaw), (xy, psi) = self._target(env), _base_pose(env)
        d = np.linalg.norm(gxy - xy)
        if self.carry:
            # loaded: position-only at the STANDOFF (facing/final approach
            # would ram the cargo into the appliance -- see act/CARRY_STOP),
            # OR a stall-arrival: progress physically converged near the dock
            # (counter/furniture edge) -- grinding on only strips the cargo.
            return bool(d <= self.CARRY_STOP
                        or (self._blocked and d <= self.CARRY_NEAR))
        return bool(d <= NAV_POS_TOL and np.cos(gyaw - psi) >= NAV_ORI_COS)


class GraspDriver:
    """Base-align to the arm's reach sweet spot, then a standoff approach onto
    the object -> gated close -> in-place squeeze -> gentle lift, with a
    deterministic retry schedule when the lift proves the enclosure false.
    Done == check_obj_grasped AND the object actually risen off its entry z.

    The original fixed-standoff driver (FWD=0.65 tuned on a z~1.0 shelf) scored
    3% on 60 random scratch seeds (capability-r1.md). The measured failure
    taxonomy, and what each part below answers:

    * 57% stuck in reach: the meat sits on HIGH (z 1.3-1.5) or LOW (z 0.6-0.9)
      shelves, outside the arm envelope at the fixed park. Fix: the TORSO lift
      (slide joint, 0..0.34 m, position-holding -- an action channel the drivers
      never used) raises the whole envelope to the shelf, and the residual
      height error shrinks the standoff FWD along the reach sphere
      (fwd = sqrt(FWD^2 - dz^2)) for shelves the torso cannot reach.
    * 32% closed on AIR even with the eef centred: the meat is a flat slab,
      measured reg_bbox horizontal extents 0.066-0.086 m (minor axis) x
      0.094-0.124 m (major) against an 0.08 m total finger span -- enclosure is
      GEOMETRICALLY possible only across the minor axis, and the old driver
      never commanded wrist yaw, so success was the luck of the spawn
      orientation (the "object-level conditionality" of calibration-r1). Fix:
      the wrist-yaw channel (a[5], measured CCW-positive about z) servos the
      finger-opening axis (== eef site x axis, measured) onto the object's
      bbox minor axis before closing.
    * the horizontal reach-in also PLOWED the object across the shelf (meat
      displaced up to 0.4 m/episode, each retry chasing it deeper into the
      fridge). Fix: over-then-down -- hover above the object, align xy+yaw,
      descend with open fingers straddling the slab, close IN PLACE.
    * per-object geometry still defeats single attempts: a fixed retry schedule
      (closer park / other lateral side / extra torso, RETRY below) re-runs the
      whole approach after a failed lift -- deterministic, same seed same
      trajectory; RSI can later learn WHEN to switch, the driver just owns the
      mechanical sequence.
    """

    FWD = 0.65        # standoff at the tuned work height (arm fwd reach)
    LAT = -0.15       # arm-side lateral offset (right-arm mount)
    WORK_Z = 1.00     # meat height FWD was tuned at, torso down
    TORSO_MAX = 0.34  # torso slide range (omron_mobile_base.xml)
    FWD_MIN = 0.38    # never park closer than this (base/fridge collision)
    ALIGN_TOL = 0.04  # base-park tolerance (P-tail floors ~0.03 on this base)
    ALIGN_CAP = 110   # align bailout: park where we are and try (blocked park)
    HOVER = 0.08      # hover height above the object before the descent
    HOVER_XY = 0.03   # xy alignment gate at hover
    YAW_TOL = 0.30    # rad: finger axis vs bbox minor axis gate (mod pi)
    KYAW = 2.0        # wrist-yaw P gain; per-step channel cap below
    YAW_CAP = 0.4
    PHASE_CAP = 90    # hover/descend stall bailout -> next retry attempt
    CLOSE_XY = 0.035  # xy gate for the close (the arm's xy P-tail floors at
                      # ~0.03 at fridge extension; 0.03 measured gate-churn)
    CLOSE_DZ = 0.02   # descent complete when eef z within this of the aim z
    CLOSE_TICKS = 12   # chase-close ticks onto the object (original)
    SQUEEZE_TICKS = 35 # then squeeze IN PLACE (kp=0) -- the probe-proven settle
                       # that turns a touching latch into a real enclosure
    LIFT_DZ = 0.20    # how far to raise after closing
    LIFT_TICKS = 40   # lift budget; not secure by then -> recover + retry
    SECURE_DZ = 0.08  # the OBJECT must rise this far off its entry z to count
                      # (0.04 verified too low: the meat cleared the latch but not
                      # the shelf lip, and the stow drag stripped it -- carry-probe;
                      # measured achievable lift at full extension is ~+0.09)
    LIFT_CAP = 0.3    # per-step arm command cap in the lift (gentle raise)
    RECOVER_TICKS = 18
    #: retry schedule: (mode, d_fwd, d_lat, d_aim_z, d_torso) per attempt --
    #: deterministic, exhausted in order. Modes measured on the dev block:
    #: * "over"     -- hover above, descend (the only mode that does not plow
    #:   the slab across the shelf; a lateral reach-in variant was measured and
    #:   dropped: it shoved the meat up to 0.4 m and churned at its gate).
    #: * "over_end" -- same but aim at the slab's base-near END, where the mesh
    #:   narrows: some slabs' minor extent (up to 0.086 m) exceeds the 0.08 m
    #:   finger span, so a centre grasp is geometrically impossible.
    #: The closer-park/lower-aim tweaks are the measured single-seed winners.
    #: The 6th field is the wrist-yaw STYLE -- the three styles were measured
    #: on the dev block and win DISJOINT seed sets (union 22/60 vs 14-15 for
    #: any single style), so the schedule sweeps them:
    #: * "gated" -- yaw servo only once xy < 0.06 (servoing at full extension
    #:   fights the reach: bare push extends 0.664 m, under a yaw servo ~0.56)
    #: * "full"  -- yaw servo throughout (works where the park is close)
    #: * "pre"   -- pre-rotate while retracted, then extend orientation-free
    #: The 7th field rotates the whole approach ring (the base re-parks on an
    #: arc around the object). Measured: +-0.22 arcs LOSE (the rotated parks
    #: are blocked far more often -- align-stuck 18/60 vs 9/60), so the
    #: schedule keeps the frontal approach and sweeps the other axes.
    RETRY = (("over", 0.0, 0.0, 0.0, 0.0, "gated", 0.0),
             ("over", -0.07, 0.0, -0.015, 0.0, "full", 0.0),
             ("over_end", 0.0, 0.0, -0.005, 0.0, "gated", 0.0),
             ("over", 0.0, 0.0, 0.0, 0.0, "pre", 0.0),
             ("over", -0.10, 0.03, -0.02, 0.06, "gated", 0.0),
             ("over_end", -0.07, -0.05, -0.015, 0.0, "full", 0.0))

    def __init__(self, obj_name):
        t = tunables()
        self.HOVER, self.HOVER_XY, self.FWD = t["hover_dz"], t["reach_tol"], t["standoff"]
        self._stall_k = t["stall_k"]
        self.failure_mode = None  # "reach_stall" once the retry schedule is spent stalled
        self.obj_name = obj_name
        self.phase = "align"
        self.attempt = 0
        self._psi = None       # approach yaw, locked at entry (the dock yaw)
        self._ticks = 0
        self._lift_z = None
        self._obj_z0 = None    # object z at entry: the secure-lift reference
        self._z_hist: list = []  # descent stall detector window
        self._stall = StallDetector(self._stall_k)  # hover/descend reach watchdog

    # -- per-attempt geometry --------------------------------------------------
    def _tweak(self):
        return self.RETRY[min(self.attempt, len(self.RETRY) - 1)]

    def _torso_target(self, env):
        mode, d_fwd, d_lat, d_aim, d_torso, style, d_psi = self._tweak()
        mz = float(_obj_pos(env, self.obj_name)[2])
        return float(np.clip(mz - self.WORK_Z + d_torso, 0.0, self.TORSO_MAX))

    def _standoff(self, env):
        """(fwd, lat) for this attempt: the reach-sphere shrink + retry tweak."""
        mode, d_fwd, d_lat, d_aim, d_torso, style, d_psi = self._tweak()
        mz = float(_obj_pos(env, self.obj_name)[2])
        dz = mz - (self.WORK_Z + self._torso_target(env))  # residual height err
        fwd = float(np.sqrt(max(self.FWD ** 2 - dz ** 2, self.FWD_MIN ** 2)))
        return max(fwd + d_fwd, self.FWD_MIN), self.LAT + d_lat

    def _apsi(self):
        """This attempt's approach yaw: the locked dock yaw + the retry arc."""
        return self._psi + self._tweak()[6]

    def _base_target(self, env):
        m = _obj_pos(env, self.obj_name)
        if self._psi is None:
            self._psi = _base_pose(env)[1]
        fwd, lat = self._standoff(env)
        c, s = np.cos(self._apsi()), np.sin(self._apsi())
        return m[:2] - np.array([c * fwd - s * lat, s * fwd + c * lat])

    def _next_attempt(self, stalled=False):
        # a stall with no fresh geometry left to try is the segment's honest end
        if stalled and self.attempt + 1 >= len(self.RETRY):
            self.failure_mode = "reach_stall"
        self.attempt += 1
        self.phase = "align"
        self._ticks = 0
        self._lift_z = None
        self._z_hist = []
        self._stall = StallDetector(self._stall_k)

    def _minor_axis(self, env):
        """World-horizontal unit vector across the object's THINNEST bbox
        extent -- the only span the 0.08 m fingers can enclose. None when the
        object has no reg_bbox geom (fall back to no yaw servo)."""
        m = env.sim.model
        try:
            gid = m.geom_name2id(f"{self.obj_name}_reg_bbox")
        except Exception:
            return None
        R = np.asarray(env.sim.data.geom_xmat[gid]).reshape(3, 3)
        size = np.asarray(m.geom_size[gid])
        for k in np.argsort(size[:2]):        # thin in-plane axis first
            v = R[:2, k]
            n = float(np.linalg.norm(v))
            if n > 0.3:                       # skip a near-vertical axis
                return v / n
        return None

    def _yaw_err(self, env, axis):
        """Signed angle (mod pi -- the fingers are symmetric) from the finger-
        opening axis (eef site x, measured) to the target horizontal axis."""
        if axis is None:
            return 0.0
        R = np.asarray(
            env.sim.data.site_xmat[env.robots[0].eef_site_id["right"]]).reshape(3, 3)
        f = R[:2, 0]
        n = float(np.linalg.norm(f))
        if n < 0.2:
            return 0.0
        f = f / n
        ang = np.arctan2(axis[1], axis[0]) - np.arctan2(f[1], f[0])
        return float((ang + np.pi / 2) % np.pi - np.pi / 2)

    def _grasp_xy(self, env):
        """Grasp-point xy: the object centre, or for the *_end modes the point
        3/5 of the way out to the slab's base-near END along its major axis
        (deterministic; the mesh narrows toward the ends, and over-span slabs
        are only enclosable there)."""
        m = _obj_pos(env, self.obj_name)
        mode = self._tweak()[0]
        if not mode.endswith("_end"):
            return m[:2]
        mdl = env.sim.model
        try:
            gid = mdl.geom_name2id(f"{self.obj_name}_reg_bbox")
        except Exception:
            return m[:2]
        R = np.asarray(env.sim.data.geom_xmat[gid]).reshape(3, 3)
        size = np.asarray(mdl.geom_size[gid])
        k = int(np.argmax(size[:2]))          # major in-plane axis
        v = R[:2, k]
        n = float(np.linalg.norm(v))
        if n < 0.3:
            return m[:2]
        v = v / n
        off = v * float(size[k]) * 0.6
        # pick the end nearer the base (the reachable one), deterministically
        xy, _ = _base_pose(env)
        if np.linalg.norm(m[:2] + off - xy) > np.linalg.norm(m[:2] - off - xy):
            off = -off
        return m[:2] + off

    def act(self, env, obs):
        m = _obj_pos(env, self.obj_name)
        eef = _eef(env)
        torso = _torso_cmd(env, self._torso_target(env))
        mode, d_fwd, d_lat, d_aim, d_torso, style, d_psi = self._tweak()
        gxy = self._grasp_xy(env)

        if self.phase == "align":
            tgt = self._base_target(env)
            self._ticks += 1
            if (np.linalg.norm(_base_pose(env)[0] - tgt) < self.ALIGN_TOL
                    or self._ticks > self.ALIGN_CAP):
                self.phase = "preyaw" if style == "pre" else "hover"
                self._ticks = 0
            else:
                a = _base_action(env, tgt, self._apsi(), grip=GRIP_OPEN, kp=6.0)
                a[TORSO] = torso
                return a
        if self.phase == "preyaw":
            # rotate the wrist onto the object's thin axis WHILE RETRACTED --
            # near the body the orientation authority is full. Commanding yaw
            # during the extension instead FIGHTS the reach: the OSC trades
            # orientation for position when left alone (measured: a bare push
            # extends 0.664 m forward; the same push under a live yaw servo
            # stalls at ~0.56 and the orientation ends 57 degrees sideways).
            yerr = self._yaw_err(env, self._minor_axis(env))
            self._ticks += 1
            if abs(yerr) < 0.2 or self._ticks > 50:
                self.phase = "hover"
                self._ticks = 0
            a = _arm_action(env, eef, GRIP_OPEN, kp=0.0,
                            dyaw=np.clip(yerr * self.KYAW,
                                         -self.YAW_CAP, self.YAW_CAP))
            a[TORSO] = torso
            return a
        if self.phase == "hover":
            # over-then-down: centre above the grasp point and get the fingers
            # across the thin side, per this attempt's yaw STYLE (see RETRY)
            wp = np.array([gxy[0], gxy[1], m[2] + self.HOVER])
            err_xy = wp[:2] - eef[:2]
            yerr = self._yaw_err(env, self._minor_axis(env))
            self._ticks += 1
            if (np.linalg.norm(err_xy) < self.HOVER_XY
                    and abs(eef[2] - wp[2]) < 0.04
                    and (style == "pre" or abs(yerr) < self.YAW_TOL)):
                self.phase = "descend"
                self._ticks = 0
                self._stall = StallDetector(self._stall_k)
            elif self._ticks > self.PHASE_CAP or self._stall.update(np.linalg.norm(wp - eef)):
                self._next_attempt(stalled=True)
                return _arm_action(env, eef, GRIP_OPEN, kp=0.0)
            if self.phase == "hover":
                full = np.clip(yerr * self.KYAW, -self.YAW_CAP, self.YAW_CAP)
                dyaw = (full if style == "full"
                        else full if (style == "gated"
                                      and float(np.linalg.norm(err_xy)) < 0.06)
                        else 0.0)
                a = _arm_action(env, wp, GRIP_OPEN, dyaw=dyaw)
                a[TORSO] = torso
                return a
        # descent/close aim: slightly BELOW the grasp point. A friction stall
        # (fingers rubbing the slab sides) is PRESSED through with a saturated
        # down command -- the small P-tail near the aim is weaker than the
        # contact friction and used to freeze the descent 1.2-1.7 cm high,
        # leaving a top-edge graze that slipped out. Only a stall that survives
        # the press (truly wedged) closes early: deepest reachable pinch.
        aim = np.array([gxy[0], gxy[1], m[2] + d_aim - 0.005])
        if self.phase == "descend":
            yerr = self._yaw_err(env, self._minor_axis(env))
            self._ticks += 1
            self._z_hist.append(float(eef[2]))
            err = aim - eef
            centred = np.linalg.norm(err[:2]) < self.CLOSE_XY
            stalled = (len(self._z_hist) > 8
                       and self._z_hist[-9] - self._z_hist[-1] < 0.002)
            wedged = (len(self._z_hist) > 25
                      and self._z_hist[-26] - self._z_hist[-1] < 0.003)
            if centred and (-err[2] < self.CLOSE_DZ or wedged):
                self.phase = "close"
                self._ticks = 0
            elif self._ticks > self.PHASE_CAP or self._stall.update(np.linalg.norm(err)):
                self._next_attempt(stalled=True)
                return _arm_action(env, eef, GRIP_OPEN, kp=0.0)
            else:
                # descend yaw per style: "full" keeps the hard servo; the
                # others take only MILD corrections once nearly centred (a
                # hard yaw servo at extension destroys the orientation)
                dyaw = (np.clip(yerr * self.KYAW, -self.YAW_CAP, self.YAW_CAP)
                        if style == "full"
                        else (np.clip(yerr * self.KYAW, -0.15, 0.15)
                              if float(np.linalg.norm(err[:2])) < 0.06 else 0.0))
                a = _arm_action(env, aim, GRIP_OPEN, dyaw=dyaw)
                a[2] = -0.5 if stalled else float(np.clip(a[2], -0.5, 0.5))
                return a
        if self.phase == "close":
            # close IN PLACE: the fingers already straddle the slab; chasing
            # the centre at gain here is what used to shove it off the shelf
            self._ticks += 1
            if self._ticks > self.CLOSE_TICKS:
                self.phase = "squeeze"
                self._ticks = 0
            return _arm_action(env, eef, GRIP_CLOSE, kp=0.0)
        if self.phase == "squeeze":
            # hold position (kp=0), keep closing: chasing the object centre at
            # full gain while the fingers close shoves the object instead of
            # enclosing it (carry-probe: the in-place settle is what turned the
            # seed-11 touching latch into a real, finger-holding enclosure).
            self._ticks += 1
            if self._ticks > self.SQUEEZE_TICKS:
                self.phase = "lift"
                self._ticks = 0
                self._lift_z = _eef(env)[2] + self.LIFT_DZ
            return _arm_action(env, eef, GRIP_CLOSE, kp=0.0)
        if self.phase == "lift":
            # GENTLY (carry-probe: a saturated 0.05 m/step lift accelerates the
            # just-enclosed object out of the fingers; capped 0.015 m/step keeps
            # the enclosure through the whole raise). done() fires mid-lift the
            # moment the object provably rides up; a spent budget without that
            # proof means the fingers closed on air -> recover and retry.
            self._ticks += 1
            if self._ticks > self.LIFT_TICKS:
                self.phase = "recover"
                self._ticks = 0
            a = _arm_action(env, np.array([eef[0], eef[1], self._lift_z]), GRIP_CLOSE)
            a[0:3] = np.clip(a[0:3], -self.LIFT_CAP, self.LIFT_CAP)
            return a
        # recover: open and rise straight up off the slab, then re-run the
        # approach with the next attempt's geometry (deterministic retry, not RSI)
        self._ticks += 1
        if self._ticks > self.RECOVER_TICKS:
            self._next_attempt()
        return _arm_action(env, np.array([eef[0], eef[1], eef[2] + 0.08]),
                           GRIP_OPEN)

    def done(self, env):
        """Grasped AND the object has actually risen off its entry pose.

        check_obj_grasped alone is a FALSE-POSITIVE latch (carry-probe diag,
        local-archive/robocasa-adapt/carry-probe.md): finger_joint2 is
        mirror-negative so its <0.035 test always passes, and joint1 passes with
        the gripper wide OPEN merely touching the object -- on seeds 4/5/8 the
        latch fired while the fingers then closed onto AIR and the object never
        left the shelf, sealing a fake segment success that doomed every later
        node. Requiring the object's own z to rise SECURE_DZ above its entry
        value is the relational proof of a real enclosure (the object moves with
        the hand); a false pinch can never satisfy it, so the segment honestly
        burns its cap and fails at grasp -- the node that is actually broken.
        """
        import robocasa.utils.object_utils as OU

        if self._obj_z0 is None:
            self._obj_z0 = float(_obj_pos(env, self.obj_name)[2])
        return bool(OU.check_obj_grasped(env, self.obj_name)
                    and float(_obj_pos(env, self.obj_name)[2])
                    > self._obj_z0 + self.SECURE_DZ)


class PlaceDriver:
    """Carry the held object over a fixture's interior and release it inside.

    Target is the fixture's interior-site centroid (get_int_sites) so the drop
    lands in the cavity, not on the door. Phases: over -> lower -> release ->
    retreat. Done == OU.obj_inside_of AND gripper released (OU.gripper_obj_far).
    """

    OVER_DZ = 0.10
    RELEASE_TICKS = 6

    def __init__(self, obj_name, fixture_name):
        self.obj_name = obj_name
        self.fixture_name = fixture_name
        self.phase = "over"
        self._ticks = 0
        self._target = None
        self._stall = StallDetector(tunables()["stall_k"])
        self.failure_mode = None

    def _interior(self, env):
        if self._target is None:
            fx = _fixture(env, self.fixture_name)
            regions = fx.get_int_sites(relative=False)
            pts = np.array([p0 for (p0, px, py, pz) in regions.values()])
            self._target = pts.mean(0)
        return self._target

    #: the recovery seam's live target (RobocasaRecoveryActor.for_stage): a place
    #: stage's repair aims at the cavity, not at the object riding in the gripper
    _drop_point = _interior

    def act(self, env, obs):
        c = self._interior(env)
        eef = _eef(env)
        if self.phase == "over":
            goal = np.array([c[0], c[1], c[2] + self.OVER_DZ])
            if np.linalg.norm((eef - goal)[:2]) < 0.03:
                self.phase = "lower"
            elif self._stall.update(np.linalg.norm(goal - eef)):
                self.failure_mode = "reach_stall"
            return _arm_action(env, goal, GRIP_CLOSE)
        if self.phase == "lower":
            goal = np.array([c[0], c[1], c[2]])
            if eef[2] - c[2] < 0.03:
                self.phase = "release"
            elif self._stall.update(np.linalg.norm(goal - eef)):
                self.failure_mode = "reach_stall"
            return _arm_action(env, goal, GRIP_CLOSE)
        if self.phase == "release":
            self._ticks += 1
            if self._ticks > self.RELEASE_TICKS:
                self.phase = "retreat"
            return _arm_action(env, np.array([c[0], c[1], c[2] + 0.02]), GRIP_OPEN)
        # retreat: back the eef up and out, gripper open
        goal = np.array([eef[0], eef[1], c[2] + 0.25])
        return _arm_action(env, goal, GRIP_OPEN)

    def done(self, env):
        import robocasa.utils.object_utils as OU

        return bool(OU.obj_inside_of(env, self.obj_name, _fixture(env, self.fixture_name))
                    and OU.gripper_obj_far(env, obj_name=self.obj_name))


class CloseDoorDriver:
    """Push a hinged fixture door shut by pressing on its handle from outside.

    Drives the (open) gripper to the door handle geom and keeps pushing toward
    the closed-door hinge side. Done == fixture.is_closed. Best-effort scripted
    push; if the OSC cannot generate the needed lateral force it is honest failure.
    """

    def __init__(self, fixture_name):
        self.fixture_name = fixture_name

    def act(self, env, obs):
        fx = _fixture(env, self.fixture_name)
        handle = _geom_pos(env, fx.handle_name)   # {prefix}door_handle_main
        body = np.asarray(fx.pos, float)          # fixture centre (hinge closes inward)
        push = handle + 0.15 * (body - handle) / (np.linalg.norm(body - handle) + 1e-9)
        return _arm_action(env, push, GRIP_OPEN, kp=6.0)

    def done(self, env):
        return bool(_fixture(env, self.fixture_name).is_closed(env))


class PressStartDriver:
    """Touch the microwave start-button geom with the gripper to turn it on.

    Microwave.update_state turns _turned_on True while the gripper contacts
    {prefix}start_button and the door is closed. So this holds the (closed)
    gripper against the button geom. Done == microwave turned_on. Requires the
    door already closed (CloseDoorDriver first) -- an open door forces state off.
    """

    def __init__(self, fixture_name="microwave"):
        self.fixture_name = fixture_name

    def act(self, env, obs):
        fx = _fixture(env, self.fixture_name)
        btn = _geom_pos(env, f"{fx.naming_prefix}start_button")
        return _arm_action(env, btn, GRIP_CLOSE, kp=8.0)

    def done(self, env):
        return bool(_fixture(env, self.fixture_name).get_state()["turned_on"])


def run_stage(env, driver, budget, obs=None):
    """Step `driver` until driver.done(env) or `budget` control steps elapse.

    Returns (done, steps, obs). The stage reads live env state, so obs is passed
    through only for drivers that want it; success is judged by done(env).
    """
    if obs is None:
        obs = env._get_observations() if hasattr(env, "_get_observations") else None
    for i in range(budget):
        if driver.done(env):
            return True, i, obs
        if getattr(driver, "failure_mode", None):   # e.g. "reach_stall": fail early
            return False, i, obs
        action = driver.act(env, obs)
        obs, _, _, _ = env.step(action)
    return driver.done(env), budget, obs
