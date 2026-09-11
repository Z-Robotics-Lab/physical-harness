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
_SEGMENT_CAPS = {"navigate": 500, "pick": 200, "place": 200, "open": 200, "close": 200}


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
                uenv.agent.controller.reset()
                ev = uenv.evaluate()
                return bool(ev["navigated_close"][0]) and bool(ev["oriented_correctly"][0])

            # Candidate docking poses, the ENV ITSELF as the oracle: an
            # articulation goal (fridge) wants the base inside a docking box
            # in ITS local frame (x 0.93..1.83, lateral +-0.6); a plain marker
            # wants near + facing. Try local +-x / +-z at docking range, then
            # the short plain-marker offset; first pose evaluate() admits wins.
            candidates = [_rot(1.383, 0, 0), _rot(-1.383, 0, 0),
                          _rot(0, 0, 1.383), _rot(0, 0, -1.383),
                          _rot(0.7, 0, 0), _rot(-0.7, 0, 0)]
            for dx, dy in candidates:
                if _try(gx + dx, gy + dy):
                    break
            self._act = lambda obs: torch.zeros(1, 13)
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
