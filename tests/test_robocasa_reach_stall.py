"""robocasa lane: the reach-stall watchdog on a real kitchen. A ClusterDropDriver
aimed at an unreachable drop point (3 m past the counter) must fail its stage
with failure_mode "reach_stall" well inside the 300-step drop cap instead of
burning it, and the recovery actor built for that stage must run to completion."""

from __future__ import annotations

import json

import numpy as np
import pytest

from harness.spec import EpisodeSpec
from plugins.embodiment_robocasa import drivers as D
from plugins.embodiment_robocasa import provider
from plugins.embodiment_robocasa.recovery import RobocasaRecoveryActor, run_recovery
from plugins.embodiment_robocasa.recycle_driver import ClusterDropDriver
from plugins.rsi import repertoire


@pytest.mark.robocasa
def test_unreachable_drop_stalls_early_and_recovers(monkeypatch):
    env = provider().make_env(EpisodeSpec(seed=4243, task="recycle_cans"))
    try:
        env.reset()
        drv = ClusterDropDriver("can1", 0)
        real = drv._drop_point(env)
        far = real + np.array([3.0, 0.0, 0.0])
        drv._point = far  # bypass the lazy stove/counter lookup with an unreachable aim
        done, steps, obs = D.run_stage(env, drv, 300)
        assert not done and drv.failure_mode == "reach_stall"
        assert D.tunables()["stall_k"] <= steps < 300, steps
        diag = drv.diagnostics(env)
        assert diag["failure_mode"] == "reach_stall"
        tr = diag["trace"]   # the numeric stall geometry a proposer reads
        print("trace", tr)
        assert set(tr) == {"start", "stall", "end"} and 0 < tr["stall"]["step"] <= steps
        assert tr["stall"]["d_eef_target"] > 2.5 and tr["stall"]["target"][0] == pytest.approx(far[0], abs=1e-3)
        assert len(tr["stall"]["base"]) == 3 and len(tr["stall"]["eef"]) == 3
        # the reach repair built for this stage aims at its drop point and runs out
        drv._point = real
        d0 = float(np.linalg.norm(D._eef(env) - real))
        act = RobocasaRecoveryActor.for_stage(env, drv, repertoire.strategy("reapproach"))
        n, obs = run_recovery(env, act, obs)
        assert act.done and n == repertoire.strategy("reapproach").length
        # it moved toward the LIVE target (arrival is not asked: the spawn is out of
        # arm reach of the stove-side counter; that is base_nudge/the carry leg's job)
        assert float(np.linalg.norm(D._eef(env) - real)) < d0 - 0.05, (d0, steps)
        # nudge_max (tunable) bounds base_nudge's travel: 0.4 drives the base past
        # the default 0.15 hand-span toward the drop point
        monkeypatch.setenv("PH_TUNABLES", json.dumps({"nudge_max": 0.4}))
        xy0 = D._base_pose(env)[0].copy()
        act = RobocasaRecoveryActor.for_stage(env, drv, repertoire.strategy("base_nudge"))
        n, obs = run_recovery(env, act, obs)
        moved = float(np.linalg.norm(D._base_pose(env)[0] - xy0))
        print("base_nudge travel", moved, act.diagnostics())
        assert act.done and 0.15 < moved <= 0.45 and act.diagnostics()["base_travel"] == pytest.approx(moved, abs=1e-3)
    finally:
        env.close()
