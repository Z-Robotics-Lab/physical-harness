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
        # the point's PROVENANCE, read off the live stove: bbox + the card's knobs
        prov = drv.provenance()
        t0 = D.tunables()
        assert prov["stove_half_extent"] > 0 and prov["stove_top_z"] > 0.5, prov
        assert prov["edge_margin"] == t0["drop_edge_margin"] and prov["slot"] == 0
        assert np.linalg.norm(prov["toward_counter"]) == pytest.approx(1.0, abs=1e-2)  # rounded to mm
        far = real + np.array([3.0, 0.0, 0.0])
        drv._point = far  # bypass the lazy stove/counter lookup with an unreachable aim
        done, steps, obs = D.run_stage(env, drv, 300)
        assert not done and drv.failure_mode == "reach_stall"
        assert D.tunables()["stall_k"] <= steps < 300, steps
        diag = drv.diagnostics(env)
        assert diag["failure_mode"] == "reach_stall"
        tr = diag["trace"]   # the numeric stall geometry a proposer reads
        print("trace", tr)
        assert set(tr) == {"start", "stall", "end", "series", "groups", "sampling"} and 0 < tr["stall"]["step"] <= steps
        assert tr["stall"]["d_eef_target"] > 2.5 and tr["stall"]["target"][0] == pytest.approx(far[0], abs=1e-3)
        assert len(tr["stall"]["base"]) == 3 and len(tr["stall"]["eef"]) == 3
        # the per-step series: a place stage commands the ARM only, so the base pose it
        # inherited never changes -- an out-of-reach point is not closable in this segment
        ser = tr["series"]
        assert 0 < len(ser) <= D.SERIES_MAX and 0 < ser[-1]["step"] <= steps
        assert all(r["cmd"]["mode"] == "arm" for r in ser), ser[:2]
        assert not any(c in r["cmd"]["nonzero"] for r in ser for c in ("vx", "vy", "wyaw"))
        b0 = ser[0]["base"]   # < 2 cm of physical jitter while the gap is metres:
        assert max(abs(r["base"][i] - b0[i]) for r in ser for i in (0, 1)) < 0.02
        g = diag["geometry"]
        assert g["reach_max"] == D.REACH_MAX and g["d_base_point"] > g["reach_max"], g
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
