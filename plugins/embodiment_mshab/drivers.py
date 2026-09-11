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

            from mshab.agents.ppo import Agent as PPOAgent

            backend = self._backend(skill_type, target)
            device = torch.device("cuda")
            policy = PPOAgent(self._env.pipeline_obs,
                              self._env.uenv.single_action_space.shape)
            policy.eval()
            policy.load_state_dict(
                torch.load(backend.checkpoint_path, map_location=device)["agent"])
            policy.to(device)
            self._policies[key] = policy
        policy = self._policies[key]

        def act(obs):
            import torch

            with torch.no_grad():
                return policy.get_action(obs, deterministic=True)

        return act

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
