"""ManiSkill-HAB embodiment: the EnvProvider for capability `embodiment.env`.

Thin adapter over env.py, mirroring the libero card's __init__ so a fourth
simulator satisfies the same contract (harness.contracts.EnvProvider). env is
imported at module top, but env.make_env imports mani_skill/mshab lazily, so
mounting this provider on a card-absent machine never drags the simulator in.

``success`` reads the env's OWN per-step success flag (mshab SubtaskTrain
info["success"], threaded onto the adapter's obs dict). Per this repo's
discipline that flag is UNAUDITED for discrimination -- the rollout mission
gates nothing on it (segment truth is "the rollout ran"; the flag rides
diagnostics). Audit it before any gate ever consumes it.
"""

from __future__ import annotations

from typing import Any

import plugins.embodiment_mshab.env as _env


class MshabEmbodiment:
    """Layer 3 `harness.contracts.EnvProvider`, backed by env.py verbatim."""

    def make_env(self, spec: Any) -> Any:
        return _env.make_env(spec)

    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted(_env.TASKS))

    def object_key(self, spec: Any) -> str:
        return _env.object_key(spec)

    def success(self, obs: Any, spec: Any, start_z: float) -> bool:
        return _env.success(obs, spec, start_z)


def provider() -> MshabEmbodiment:
    return MshabEmbodiment()
