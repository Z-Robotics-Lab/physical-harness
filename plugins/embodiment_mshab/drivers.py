"""The mshab rollout driver: a frozen random-action policy.

The heterogeneous episodic protocol (workload._governed_segment): the episode
driver binds each sub-goal itself via ``enter_segment(env, seg_spec)`` and
reports its own truth via ``segment_success(env)``. This driver's ONE sub-goal
is "drive a parameterized rollout" -- actions are seeded-uniform samples from
the env's action space (MshabEnv.sample_action, rng seeded per episode), so a
same-seed brief replays the same stream. Its segment truth is "the rollout
drove at least one step"; the env's own (unaudited) success flag rides
segment_diagnostics, never a gate. Swap the ref in mission_mshab_rollout's
binding for a checkpoint-backed provider when a trained policy lands.
"""

from __future__ import annotations

from typing import Any


class RolloutDriver:
    """Minimal PolicyDriver: observe_once/act/exhausted + the segment seams."""

    def __init__(self, spec: Any):
        self._spec = spec
        self._env: Any = None
        self._steps = 0

    @property
    def identity(self) -> str:
        return "mshab_random_rollout@v1"

    @property
    def exhausted(self) -> bool:
        # Never self-exhausts: the segment's step_budget (EPISODE horizon) is
        # the one clock that ends the rollout.
        return False

    def observe_once(self, obs) -> None:
        return None

    def enter_segment(self, env, seg_spec, executor: Any = None) -> None:
        del seg_spec, executor
        self._env = env
        self._steps = 0

    def act(self, obs):
        del obs
        self._steps += 1
        return self._env.sample_action()

    def segment_success(self, env) -> bool:
        del env
        return self._steps > 0

    def segment_diagnostics(self, env) -> dict:
        return {"steps_driven": self._steps,
                "env_success": bool(getattr(env, "env_success", False))}


class RolloutPolicies:
    """policy.driver provider: one fresh RolloutDriver per episode."""

    def make_driver(self, spec: Any) -> RolloutDriver:
        return RolloutDriver(spec)


def provider(**params: Any) -> RolloutPolicies:
    return RolloutPolicies(**params)


#: Per-skill step caps on the shared chain horizon (the teammate's
#: build_skill_chain SUBTASK_HORIZONS values): a stuck segment ends at its own
#: cap instead of eating the whole episode budget.
#: pick/place ride the env's widened 300-step budget (task_cfgs horizon in the
#: chain env): a behaviorally-successful place landed at ~205 driver steps.
_SEGMENT_CAPS = {"navigate": 500, "pick": 280, "place": 280, "open": 200, "close": 200}


class ChainDriver:
    """Heterogeneous episodic driver over the official MS-HAB RL checkpoints.

    ``enter_segment`` parses the sub-goal's re-tasked spec (``chain-<skill>.
    <target>``), refuses honestly when the env's own subtask pointer disagrees
    with the node (the OFFICIAL chain plan is the grounding authority -- a
    VLM graph in the wrong order fails here and folds back into replan), and
    resolves the checkpoint through the teammate's SkillLibrary (exact-target
    backend first, generic ``all`` fallback). ``act`` runs the loaded PPO
    policy on the adapter's raw pipeline obs; ``segment_success`` is the env's
    own subtask advance -- the one machine oracle this chain has.
    """

    def __init__(self, spec: Any):
        self._spec = spec
        self._env: Any = None
        self._steps = 0
        self._entry = 0
        self._cap = 200
        self._skill = ""
        self._mismatch: str | None = None
        self._policies: dict[str, Any] = {}
        self._library = None
        self._spawn_cache: dict[str, Any] = {}

    @property
    def identity(self) -> str:
        return "mshab_rl_chain@v1"

    def observe_once(self, obs) -> None:
        return None

    # -- checkpoint resolution -------------------------------------------------

    def _backend(self, skill_type: str, target: str):
        from mani_skill import ASSET_DIR

        from mshab.skills import SkillLibrary, SkillType

        if self._library is None:
            self._library = SkillLibrary.from_checkpoint_root(
                ASSET_DIR / "mshab_checkpoints")
        stype = SkillType(skill_type)
        hits = (self._library.find(task="set_table", skill_type=stype,
                                   target=target, ready=True)
                or self._library.find(task="set_table", skill_type=stype,
                                      target="all", ready=True))
        if not hits:
            raise ValueError(f"no ready rl checkpoint for {skill_type}/{target}")
        return hits[0].backend("rl")

    def _act_fn(self, skill_type: str, target: str):
        key = f"{skill_type}.{target}"
        if key not in self._policies:
            import torch
            from gymnasium import spaces

            from mshab.utils.config import parse_cfg

            backend = self._backend(skill_type, target)
            algo = parse_cfg(default_cfg_path=backend.config_path).algo
            device = torch.device("cuda")
            state = torch.load(backend.checkpoint_path, map_location=device)["agent"]
            act_shape = self._env.uenv.single_action_space.shape
            if algo.name == "ppo":
                from mshab.agents.ppo import Agent as PPOAgent

                policy = PPOAgent(self._env.pipeline_obs, act_shape)
                policy.eval(); policy.load_state_dict(state); policy.to(device)

                def act(obs, policy=policy):
                    with torch.no_grad():
                        return policy.get_action(obs, deterministic=True)
            elif algo.name == "sac":
                # evaluate.py's SAC branch verbatim: per-camera 4D frame-stacked
                # spaces flattened into the model's channel-stacked Boxes.
                from mshab.agents.sac import Agent as SACAgent

                obs_space = self._env.single_observation_space
                pixels_space: spaces.Dict = obs_space["pixels"]
                model_pixel_obs_space = dict()
                for k, space in pixels_space.items():
                    shape, low, high, dtype = (space.shape, space.low,
                                               space.high, space.dtype)
                    if len(shape) == 4:
                        shape = (shape[0] * shape[1], shape[-2], shape[-1])
                        low = low.reshape((-1, *low.shape[-2:]))
                        high = high.reshape((-1, *high.shape[-2:]))
                    model_pixel_obs_space[k] = spaces.Box(low, high, shape, dtype)
                policy = SACAgent(
                    spaces.Dict(model_pixel_obs_space),
                    obs_space["state"].shape, act_shape,
                    actor_hidden_dims=list(algo.actor_hidden_dims),
                    critic_hidden_dims=list(algo.critic_hidden_dims),
                    critic_layer_norm=algo.critic_layer_norm,
                    critic_dropout=algo.critic_dropout,
                    encoder_pixels_feature_dim=algo.encoder_pixels_feature_dim,
                    encoder_state_feature_dim=algo.encoder_state_feature_dim,
                    cnn_features=list(algo.cnn_features),
                    cnn_filters=list(algo.cnn_filters),
                    cnn_strides=list(algo.cnn_strides),
                    cnn_padding=algo.cnn_padding,
                    log_std_min=algo.actor_log_std_min,
                    log_std_max=algo.actor_log_std_max,
                    device=device)
                policy.eval(); policy.load_state_dict(state); policy.to(device)

                from mshab.utils.array import to_tensor

                def act(obs, policy=policy):
                    with torch.no_grad():
                        obs = to_tensor(obs, device=device, dtype="float")
                        return policy.actor(obs["pixels"], obs["state"],
                                            compute_pi=False,
                                            compute_log_pi=False)[0]
            else:
                raise ValueError(f"unsupported algo {algo.name!r} for {key}")
            self._policies[key] = act
        return self._policies[key]

    def _nav_act(self, obs):
        """Scripted differential-drive navigate (the robocasa NavigateDriver
        precedent): privileged goal off the env's own subtask goal marker,
        rotate-then-drive on the base dims (11 forward in heading frame, 12
        yaw rate -- probed 2026-09-12). The RL navigate checkpoint measured
        1/3 in its OWN env and ~0 in chain context (official evaluate long-
        horizon defaults to TELEPORT nav for a reason); straight-line drive
        with no planner -- an obstacle in the way is an honest failure."""
        import math

        import torch

        del obs
        robot = self._env.uenv.agent.robot
        q = robot.get_qpos()[0]
        x, y, yaw = float(q[0]), float(q[1]), float(q[2])
        goal = self._env.uenv.subtask_goals[self._entry]
        gx, gy = float(goal.pose.p[0, 0]), float(goal.pose.p[0, 1])
        dist = math.hypot(gx - x, gy - y)
        a = torch.zeros(1, 13)
        if dist > 0.25:
            err = (math.atan2(gy - y, gx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
            a[0, 12] = max(-1.0, min(1.0, 2.0 * err))
            if abs(err) < 0.6:
                a[0, 11] = max(-1.0, min(1.0, dist))
        return a

    # -- the segment protocol --------------------------------------------------

    def enter_segment(self, env, seg_spec, executor: Any = None) -> None:
        del executor
        self._env = env
        self._steps = 0
        self._mismatch = None
        task = str(seg_spec.task)
        name = task[len("chain-"):] if task.startswith("chain-") else task
        self._skill, _, self._target = name.partition(".")
        self._cap = _SEGMENT_CAPS.get(self._skill, 200)
        pointer = int(env.uenv.subtask_pointer[0])
        self._entry = pointer
        plan = env.uenv.task_plan
        if pointer >= len(plan):
            self._mismatch = "chain already finished; nothing left to drive"
            return
        current = plan[pointer].type
        if current != self._skill:
            self._mismatch = (f"env subtask {pointer} is {current!r}, node asked "
                              f"{self._skill!r} -- the official chain order is the "
                              "grounding authority")
            return
        if self._skill == "navigate":
            # TELEPORT nav, the official MS-HAB long-horizon evaluation mode
            # (the RL navigate checkpoint measured 1/3 in its OWN env, ~0 in
            # chain context, and straight-line scripted drive dies on the
            # first wall -- no path planner exists in this benchmark). The
            # base is SET to the env's own subtask goal pose; the settle
            # steps then let the env's navigate check pass on its own terms.
            import math

            import torch

            uenv = env.uenv
            goal = uenv.subtask_goals[pointer]
            gp, gq = goal.pose.p, goal.pose.q
            gx, gy = float(gp[0, 0]), float(gp[0, 1])
            w, xq, yq, zq = (float(gq[0, i]) for i in range(4))

            def _rot(vx, vy, vz):
                # rotate v by the goal quaternion (wxyz), world-frame result
                return (
                    (1 - 2 * (yq * yq + zq * zq)) * vx + 2 * (xq * yq - w * zq) * vy + 2 * (xq * zq + w * yq) * vz,
                    2 * (xq * yq + w * zq) * vx + (1 - 2 * (xq * xq + zq * zq)) * vy + 2 * (yq * zq - w * xq) * vz,
                )

            robot = uenv.agent.robot
            scene = uenv.scene
            # While CARRYING, every settle/wait action must keep the gripper
            # closed: dim 7 at 0 drifts the fingers toward the middle and
            # drops the object (probed: -1 holds a grasp 20 steps, +1 opens).
            held = False
            plan_all = uenv.task_plan
            if pointer + 1 < len(plan_all):
                nxt_obj = uenv.subtask_objs[pointer + 1]
                if nxt_obj is not None:
                    try:
                        held = bool(uenv.agent.is_grasping(nxt_obj)[0])
                    except Exception:  # noqa: BLE001 -- partial-env objs; hold is best-effort
                        held = False
            hold = torch.zeros(1, 13)
            hold[0, 7] = -1.0 if held else 0.0

            def _flush():
                # GPU sim: push + refresh BEFORE the controller re-anchor, or
                # reset() reads the stale pre-teleport qpos as its PD target
                # and drags the base back to origin (probed: within 5 steps).
                if hasattr(scene, "_gpu_apply_all"):
                    scene._gpu_apply_all()
                if hasattr(scene.px, "gpu_update_articulation_kinematics"):
                    scene.px.gpu_update_articulation_kinematics()
                if hasattr(scene, "_gpu_fetch_all"):
                    scene._gpu_fetch_all()

            # Snapshot for restore: a FAILED docking candidate must leave no
            # side effect -- an abandoned attempt once teleported the held
            # apple to a tcp pose and then walked away without it (obj-goal
            # dist 8m at place start).
            snap_q = robot.get_qpos().clone()
            snap_obj = None
            snap_rel = None
            if pointer + 1 < len(plan_all):
                _o = uenv.subtask_objs[pointer + 1]
                if _o is not None:
                    snap_obj = (_o, _o.pose.p.clone(), _o.pose.q.clone())
                    if held:
                        # the held object's pose RELATIVE to the tcp: a base
                        # teleport moves only the robot, and the ring path was
                        # losing the grasp at every candidate (probed: force 0
                        # close/orient pass, grasp 0 -- the apple stayed at
                        # the fridge). Re-attach it at each landing.
                        from mani_skill.utils.structs.pose import Pose as _P

                        snap_rel = uenv.agent.tcp.pose.inv() * _P.create_from_pq(
                            p=snap_obj[1], q=snap_obj[2])

            def _restore():
                q = robot.get_qpos()
                q[0, :] = snap_q[0]
                robot.set_qpos(q)
                if snap_obj is not None:
                    from mani_skill.utils.structs.pose import Pose

                    o, p0, q0 = snap_obj
                    o.set_pose(Pose.create_from_pq(p=p0, q=q0))
                    for setter in ("set_linear_velocity", "set_angular_velocity"):
                        fn = getattr(o, setter, None)
                        if callable(fn):
                            fn(torch.zeros(1, 3, device=snap_q.device))
                _flush()
                uenv.agent.controller.reset()

            def _probe_clean():
                # 3 wiggle steps (arm dim 0 alternating +-0.3 keeps is_static
                # False -> the env cannot seal navigate mid-probe), then read
                # force + grasp. True = pose admitted for a real settle.
                for i in range(3):
                    a = hold.clone()
                    # amplitude matters: 0.3 flung a freshly-teleported grasp
                    # out of the gripper; 0.08 still defeats is_static.
                    a[0, 0] = 0.08 if i % 2 == 0 else -0.08
                    self._env.step(a)
                ev = uenv.evaluate()
                # A held object IS a contact: the gripper squeeze alone reads
                # thousands of N (4751 at a clean pick-end pose), so <5N is
                # physically impossible while carrying. Penetration reads in
                # the MILLIONS -- the tiers are orders of magnitude apart.
                limit = 8000.0 if held else 5.0
                if float(ev.get("robot_force", [0.0])[0]) >= limit:
                    return False
                if held and snap_obj is not None and not bool(
                        uenv.agent.is_grasping(snap_obj[0])[0]):
                    return False
                return True

            def _try(px, py):
                # Land base_link AT (px, py) facing the goal by ITERATION: the
                # qpos gantry origin sits a yaw-dependent vector away from
                # base_link (probed: 1.0m); set, measure the residual, correct
                # -- twice converges with no frame convention trusted.
                yaw = math.atan2(gy - py, gx - px)
                for _ in range(3):
                    q = robot.get_qpos()
                    bl = uenv.agent.base_link.pose.p
                    q[0, 0] = float(q[0, 0]) + (px - float(bl[0, 0]))
                    q[0, 1] = float(q[0, 1]) + (py - float(bl[0, 1]))
                    q[0, 2] = yaw
                    robot.set_qpos(q)
                    _flush()
                if held and snap_rel is not None and snap_obj is not None:
                    # the carried object rides the teleport at its recorded
                    # tcp-relative pose; a free body would stay behind.
                    snap_obj[0].set_pose(uenv.agent.tcp.pose * snap_rel)
                    for setter in ("set_linear_velocity", "set_angular_velocity"):
                        fn = getattr(snap_obj[0], setter, None)
                        if callable(fn):
                            fn(torch.zeros(1, 3, device=snap_q.device))
                    _flush()
                uenv.agent.controller.reset()
                already = int(uenv.subtask_pointer[0]) > pointer
                ev = uenv.evaluate()
                if not already and not (
                        bool(ev.get("navigated_close", [False])[0])
                        and bool(ev.get("oriented_correctly", [False])[0])):
                    return False
                # COLLISION leg with the JIGGLE TRICK: navigate success
                # requires is_static, so probing with a small arm wiggle keeps
                # the env from SEALING the subtask on a candidate we have not
                # admitted yet (a penetrating pose once advanced the pointer
                # irreversibly at 4.1e6 N). Only after force and grasp read
                # clean do we hold still and let the env pass it for real.
                if not _probe_clean():
                    return False
                if already or int(uenv.subtask_pointer[0]) > pointer:
                    return True
                for _ in range(4):
                    self._env.step(hold)
                ev = uenv.evaluate()
                return (int(uenv.subtask_pointer[0]) > pointer
                        or (bool(ev.get("navigated_close", [False])[0])
                            and bool(ev.get("oriented_correctly", [False])[0])
                            and float(ev.get("robot_force", [0.0])[0]) < 5.0))

            def _spawn_dock():
                # Dock FROM THE NEXT SUBTASK'S OWN SPAWN DISTRIBUTION: the
                # official spawn_data rows carry the full 15-dof qpos (base +
                # torso + arm) TUNED to the target object's pose. Rank rows by
                # distance between their episode's object pose and OUR live
                # object, try the closest few. Pick-only for now: a full-qpos
                # teleport is safe with an empty gripper; a held object would
                # be left behind.
                ptr = self._entry
                plan = uenv.task_plan
                if ptr + 1 >= len(plan) or plan[ptr + 1].type not in ("pick", "place"):
                    return False
                nxt = plan[ptr + 1].type
                obj = uenv.subtask_objs[ptr + 1]
                if obj is None:
                    return False
                from mani_skill import ASSET_DIR

                sd = self._spawn_cache.get(nxt)
                if sd is None:
                    sd = torch.load(
                        ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
                        / f"spawn_data/set_table/{nxt}/train/spawn_data.pt",
                        map_location="cpu")
                    self._spawn_cache[nxt] = sd
                if nxt == "pick":
                    # rows whose EPISODE object sat closest to OUR live object
                    anchor = obj.pose.p[0, :2].cpu()
                    field = "obj_raw_pose"
                else:
                    # place: rows whose robot stood closest to OUR goal spot
                    anchor = uenv.subtask_goals[ptr + 1].pose.p[0, :2].cpu()
                    field = "robot_pos"
                ranked = []
                for key, entry in sd.items():
                    dmin = torch.norm(entry[field][:, :2] - anchor, dim=1).min(0)
                    ranked.append((float(dmin.values), key, int(dmin.indices)))
                ranked.sort()
                for _, key, row in ranked[:24]:
                    q_row = sd[key]["robot_qpos"][row]
                    q = robot.get_qpos()
                    q[0, :] = q_row.to(q.device)
                    robot.set_qpos(q)
                    _flush()
                    if nxt == "place":
                        # the held object teleports WITH the hand: the spawn
                        # row stores its pose RELATIVE TO the tcp; compose with
                        # the freshly-set tcp pose -- the joint robot+object
                        # state place trained on.
                        from mani_skill.utils.structs.pose import Pose

                        rel = sd[key]["obj_raw_pose_wrt_tcp"][row]
                        rel_pose = Pose.create_from_pq(
                            p=rel[None, :3].to(q.device),
                            q=rel[None, 3:7].to(q.device))
                        obj.set_pose(uenv.agent.tcp.pose * rel_pose)
                        for setter in ("set_linear_velocity", "set_angular_velocity"):
                            fn = getattr(obj, setter, None)
                            if callable(fn):
                                fn(torch.zeros(1, 3, device=q.device))
                        _flush()
                    uenv.agent.controller.reset()
                    # jiggle-probe first (see _probe_clean: the env cannot
                    # seal navigate while we are not static), then a real
                    # settle only for an admitted pose.
                    grasp_needed = (nxt == "place")
                    if not _probe_clean() or (
                            grasp_needed
                            and not bool(uenv.agent.is_grasping(obj)[0])):
                        _restore()
                        continue
                    for _ in range(4):
                        self._env.step(hold)
                    ev = uenv.evaluate()
                    if (int(uenv.subtask_pointer[0]) > ptr
                            or (bool(ev.get("navigated_close", [False])[0])
                                and bool(ev.get("oriented_correctly", [False])[0]))):
                        return True
                    _restore()
                return False

            # Candidate docking poses, the ENV ITSELF as the oracle: an
            # articulation goal (fridge) wants the base inside a docking box
            # in ITS local frame (x 0.93..1.83, lateral +-0.6); a plain marker
            # wants near + facing -- and NOT inside the furniture holding it,
            # hence the ring out to 1.4m with the collision leg above.
            if not _spawn_dock():
                candidates = [_rot(1.383, 0, 0), _rot(-1.383, 0, 0),
                              _rot(0, 0, 1.383), _rot(0, 0, -1.383),
                              _rot(0.7, 0, 0), _rot(-0.7, 0, 0)]
                for dist in (0.9, 1.2, 1.4):
                    for k in range(8):
                        b = k * math.pi / 4
                        candidates.append((dist * math.cos(b), dist * math.sin(b)))
                docked = False
                for dx, dy in candidates:
                    if _try(gx + dx, gy + dy):
                        docked = True
                        break
                    _restore()
                if not docked:
                    _restore()   # honest stay-put: better than a random pose
            # Zero the env's per-subtask force ledger after docking: rejected
            # penetrating candidates racked up FICTITIOUS billions of N (all
            # restored, never a real trajectory), and the leftover balance
            # made place's 7500N limit unpassable forever. Same semantics as
            # the env's own reset at a subtask transition -- the next segment
            # is billed only from its true start.
            uenv.robot_cumulative_force[:] = 0
            self._act = lambda obs: hold
        else:
            self._act = self._act_fn(self._skill, self._target or "all")

    def act(self, obs):
        del obs
        self._steps += 1
        return self._act(self._env.pipeline_obs)

    @property
    def exhausted(self) -> bool:
        if self._env is None or self._mismatch is not None:
            return True
        if int(self._env.uenv.subtask_pointer[0]) > self._entry:
            return True   # sub-goal done: stop before driving the NEXT subtask
        return self._steps >= self._cap

    def segment_success(self, env) -> bool:
        if self._mismatch is not None:
            return False
        pointer = int(env.uenv.subtask_pointer[0])
        return pointer > self._entry or pointer >= len(env.uenv.task_plan)

    def segment_diagnostics(self, env) -> dict:
        return {"skill": self._skill, "target": self._target,
                "steps_driven": self._steps, "entry_subtask": self._entry,
                "subtask_pointer": int(env.uenv.subtask_pointer[0]),
                "mismatch": self._mismatch,
                "env_success": bool(getattr(env, "env_success", False))}


class ChainPolicies:
    """policy.driver provider for the RL checkpoint chain."""

    def make_driver(self, spec: Any) -> ChainDriver:
        return ChainDriver(spec)


def chain_provider(**params: Any) -> ChainPolicies:
    return ChainPolicies(**params)
