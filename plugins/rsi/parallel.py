"""Process-pool rollout. On this machine 10 workers reach ~212 episodes/min,
which makes the rollout budget a non-issue for gating; see local-archive/docs/retired-from-public/verified-environment.md."""
from __future__ import annotations


def default_executor():
    """The exec.rollouts provider used when no executor is injected.

    L1 rung 2: every rollout fan-out in governor goes through an executor with
    the harness RolloutExecutor contract. Call sites take `executor=None` and
    fall back to this local pool, which maps exactly like the Pool.map blocks
    it replaced; the kernel-mounted path injects the resolved provider instead.
    Executors run in the PARENT process (they create the pools), so unlike
    provider refs this needs no spawn story.
    """
    from harness.executor import LocalPoolExecutor

    return LocalPoolExecutor()
