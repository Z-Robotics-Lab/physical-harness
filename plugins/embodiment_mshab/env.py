"""ManiSkill-HAB embodiment: task wiring + env factory + the old-gym adapter.

Mirrors the libero card's env.py shape: a TASKS table and a make_env that
imports the simulator LAZILY so this module stays base-clean (test_boundaries
parses it; the base lane never drags mani_skill in).

Runs ONLY under the mshab checkout's venv (~/Desktop/maniskill-agentic-library/
.venv: py3.12, torch + sapien + mani_skill 3.0.0b18-mshab + numpy 2.x). Known
traps, all hit during the 2026-09-11 install (docs/project-documentation.md §5.7):

* setuptools >= 81 removed pkg_resources, which sapien still imports at top
  level -- the venv pins setuptools < 81.
* num_envs=1 refuses task plans spanning several ReplicaCAD build configs
  ("cover 2 build configs, but received 1 envs"), so make_env slices ONE plan.
* GPU sim + render coexists with a ~19GB resident model server at num_envs=1 /
  state obs; the README's 252-env recipe needs ~10GB and does not.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from harness.spec import EpisodeSpec

#: Per-task wiring: the ManiSkill-HAB long-horizon task and the subtask whose
#: [Name]SubtaskTrain-v0 env one rollout drives. One binding name per subtask
#: is the whole operator-facing tuning surface (brief = task name + seed).
TASKS: dict[str, dict] = {
    "mshab_pick": {"hab_task": "tidy_house", "subtask": "pick"},
    "mshab_place": {"hab_task": "tidy_house", "subtask": "place"},
    "mshab_open": {"hab_task": "tidy_house", "subtask": "open"},
    "mshab_close": {"hab_task": "tidy_house", "subtask": "close"},
}


def task_config(spec: EpisodeSpec) -> dict:
    if spec.task not in TASKS:
        raise KeyError(f"unknown task {spec.task!r}; known: {sorted(TASKS)}")
    return TASKS[spec.task]


def object_key(spec: EpisodeSpec) -> str:
    """mshab state obs is one flat vector; no single target-object key exists.
    Raising KeyError is the honest contract answer -- callers that need a
    reference height (workload's terminal_start_z probe) catch exactly this."""
    raise KeyError(f"mshab task {spec.task!r} has no per-object obs key "
                   "(flat state vector)")


def success(obs: Any, spec: EpisodeSpec, start_z: float) -> bool:
    """The env's OWN success flag, threaded onto the adapter obs. UNAUDITED --
    diagnostics only, never a gate, until its discrimination is measured."""
    del spec, start_z
    return bool(obs.get("success", 0.0)) if hasattr(obs, "get") else False


def _scalar(value: Any) -> float:
    """One float out of a torch tensor / numpy array / python scalar."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


class MshabEnv:
    """Old-gym adapter over ONE mshab SubtaskTrain env (num_envs=1).

    reset() -> obs dict, step() -> (obs, reward, done, info): the 4-tuple API
    the governed segment loop drives. obs is ``{"state": float32[D],
    "success": float32}`` -- a Mapping so scene.snapshot and predicates read it
    like every other embodiment's obs (absent keys are omitted, never faked).
    """

    def __init__(self, env: Any, seed: int):
        self._env = env
        self._seed = int(seed)
        # Driver-facing action sampling rng: seeded per episode so a same-seed
        # rollout replays the same action stream.
        self._rng = np.random.default_rng(self._seed)
        space = getattr(env, "single_action_space", None) or env.action_space
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        # pd_joint_delta_pos is [-1, 1]; clip guards an unbounded space.
        self._low = np.clip(np.nan_to_num(low, neginf=-1.0), -1.0, 1.0)
        self._high = np.clip(np.nan_to_num(high, posinf=1.0), -1.0, 1.0)
        self._success = False
        self._finger_idx: list[int] | None = None

    @property
    def env_success(self) -> bool:
        """The env's own (unaudited) success flag after the last step."""
        return self._success

    def sample_action(self) -> np.ndarray:
        return self._rng.uniform(self._low, self._high).astype(np.float32)

    def reset(self):
        obs, _info = self._env.reset(seed=self._seed)
        self._success = False
        return self._obs(obs)

    def step(self, action):
        import torch

        a = np.asarray(action, dtype=np.float32)
        if a.shape != tuple(self._env.action_space.shape):
            a = a[None]
        obs, reward, terminated, truncated, info = self._env.step(torch.from_numpy(a))
        flag = info.get("success") if hasattr(info, "get") else None
        if flag is not None:
            self._success = bool(_scalar(flag))
        done = bool(_scalar(terminated)) or bool(_scalar(truncated))
        return self._obs(obs), _scalar(reward), done, {"success": self._success}

    def render_frame(self) -> np.ndarray | None:
        """One HWC uint8 RGB frame for scripts/frame_dump (the 取景窗 probe)."""
        img = self._env.render()
        if img is None:
            return None
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 4:
            img = img[0]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        return img

    def close(self):
        return self._env.close()

    def _obs(self, obs) -> dict:
        if hasattr(obs, "detach"):
            obs = obs.detach().cpu().numpy()
        state = np.asarray(obs, dtype=np.float32).reshape(-1)
        return {"state": state, "success": np.float32(self._success),
                **self._proprio()}

    def _proprio(self) -> dict:
        """The base feature catalog's OBSERVABLE keys (robot0_*), extracted from
        the live Fetch: real proprioception a real Fetch reports, so the
        zero-privilege critic view projects without a KeyError. Same robosuite
        naming the catalog extractors read -- vocabulary, not a robosuite dep."""
        agent = self._env.unwrapped.agent
        qpos = self._flat(agent.robot.get_qpos())
        qvel = self._flat(agent.robot.get_qvel())
        if self._finger_idx is None:
            names = [j.name for j in agent.robot.get_active_joints()]
            idx = [i for i, n in enumerate(names)
                   if "gripper" in n or "finger" in n]
            # _finger_gap reads q[0]-q[1]: guarantee two entries (a one-finger
            # embodiment degrades to a constant 0 gap, never a crash).
            self._finger_idx = (idx + idx)[:2] if idx else [0, 0]
        return {
            "robot0_eef_pos": self._flat(agent.tcp.pose.p),
            "robot0_gripper_qpos": qpos[self._finger_idx],
            "robot0_joint_vel": qvel,
            "robot0_gripper_qvel": qvel[self._finger_idx],
        }

    @staticmethod
    def _flat(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)


def make_env(spec: EpisodeSpec) -> MshabEnv:
    """Build one seeded SubtaskTrain env for `spec` and wrap it old-gym."""
    import gymnasium as gym

    from mani_skill import ASSET_DIR

    import mshab.envs  # noqa: F401  registers [Name]SubtaskTrain-v0
    from mshab.envs.planner import plan_data_from_file

    cfg = task_config(spec)
    rearrange = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plans = plan_data_from_file(
        rearrange / "task_plans" / cfg["hab_task"] / cfg["subtask"] / "train" / "all.json")
    spawn_fp = (rearrange / "spawn_data" / cfg["hab_task"] / cfg["subtask"]
                / "train" / "spawn_data.pt")
    env = gym.make(
        f"{cfg['subtask'].capitalize()}SubtaskTrain-v0",
        num_envs=1,
        obs_mode="state",
        sim_backend="gpu",
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        render_mode="rgb_array",
        shader_dir="minimal",
        max_episode_steps=int(spec.horizon),
        # ponytail: ONE task plan -> one ReplicaCAD scene; num_envs=1 refuses
        # plans spanning several build configs. Filter plans by build config if
        # scene variety ever matters more than a single-scene rollout.
        task_plans=plans.plans[:1],
        scene_builder_cls=plans.dataset,
        spawn_data_fp=spawn_fp,
    )
    return MshabEnv(env, seed=spec.seed)
