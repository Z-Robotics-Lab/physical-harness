"""Base-lane checks for the robocasa card's reach-robustness batch (overnight
item 2): the three reach recoveries are declared and executable, the [tunables]
table parses and overlays, and a hover/descend stage that makes no progress
fails its segment early with failure_mode "reach_stall". No simulator: the
drivers' live-state readers are monkeypatched, as recovery.py's self-check does."""

from __future__ import annotations

import json

import numpy as np
import pytest

from plugins.embodiment_robocasa import drivers as D
from plugins.embodiment_robocasa import stage_extras as X
from plugins.embodiment_robocasa.recovery import RobocasaRecoveryActor
from plugins.rsi import repertoire

REACH = ("reapproach", "base_nudge", "release_reset")


def test_reach_recoveries_declared_by_the_card():
    for n in REACH:
        assert n in repertoire.strategies_for("embodiment_robocasa")
        assert repertoire.strategy(n).length > 0


def test_tunables_defaults_overlay_and_sha(monkeypatch):
    t = D.tunables()
    assert set(t) >= {"hover_dz", "reach_tol", "standoff", "segment_cap", "stall_k"}
    assert t["reach_tol"] == 0.03 and t["stall_k"] == 40
    sha0 = D.tunables_sha()
    monkeypatch.setenv("PH_TUNABLES", json.dumps({"stall_k": 12}))
    assert D.tunables()["stall_k"] == 12 and D.tunables_sha() != sha0
    monkeypatch.setenv("PH_TUNABLES", json.dumps({"bogus": 1}))
    with pytest.raises(KeyError, match="bogus"):
        D.tunables()


def test_stall_detector_fires_only_without_progress():
    s = D.StallDetector(5)
    assert not any(s.update(1.0 - 0.01 * i) for i in range(50))   # steady progress
    s = D.StallDetector(5)
    assert [s.update(0.5) for _ in range(6)] == [False] * 5 + [True]  # 1 baseline + K


def _fake_world(monkeypatch, eef=(1.0, 2.0, 1.0), obj=(1.3, 2.1, 0.9), base=(0.0, 0.0)):
    state = {"eef": np.array(eef, float), "obj": np.array(obj, float),
             "base": np.array(base, float)}
    monkeypatch.setattr(D, "_eef", lambda env: state["eef"].copy())
    monkeypatch.setattr(D, "_obj_pos", lambda env, n: state["obj"].copy())
    monkeypatch.setattr(D, "_base_pose", lambda env: (state["base"].copy(), 0.0))
    return state


def test_reach_recoveries_execute_in_the_right_modes(monkeypatch):
    state = _fake_world(monkeypatch)
    for name in REACH:
        act = RobocasaRecoveryActor(object(), name, repertoire.strategy(name).steps,
                                    obj_name="can1")
        modes = []
        while not act.done:
            a = act.act({})
            assert a.shape == (D.ADIM,)
            modes.append(a[D.MODE])
        if name == "base_nudge":
            assert modes[0] == D.GRIP_CLOSE and modes[-1] == D.GRIP_OPEN  # base, then arm
        else:
            assert all(m == D.GRIP_OPEN for m in modes), (name, "arm mode throughout")
    # base_nudge never commands more than 0.15 m of base travel toward the target
    state["obj"] = np.array([0.5, 0.0, 0.9])   # straight +x of the base
    act = RobocasaRecoveryActor(object(), "base_nudge", (("nudge", 3, 0.0, 0.0),),
                                obj_name="can1")
    a = act.act({})
    assert a[7] > 0 and abs(a[8]) < 1e-9          # straight toward +x target
    state["base"] = np.array([0.2, 0.0])          # "moved" past the bound
    a = act.act({})
    assert abs(a[7]) < 1e-9 and abs(a[8]) < 1e-9  # holds still: bound spent
    assert act.diagnostics() == {"base_travel": 0.2}
    # the bound is the card's tunable nudge_max: 0.4 keeps driving from 0.2 out
    monkeypatch.setenv("PH_TUNABLES", json.dumps({"nudge_max": 0.4}))
    state["base"] = np.array([0.0, 0.0])
    act = RobocasaRecoveryActor(object(), "base_nudge", (("nudge", 3, 0.0, 0.0),),
                                obj_name="can1")
    act.act({})
    state["base"] = np.array([0.2, 0.0])
    assert act.act({})[7] > 0
    # release_reset opens first
    act = RobocasaRecoveryActor(object(), "release_reset",
                                repertoire.strategy("release_reset").steps, obj_name="can1")
    assert act.act({})[D.GRIP] == D.GRIP_OPEN


def test_for_stage_targets_a_place_stage_drop_point(monkeypatch):
    _fake_world(monkeypatch)

    class _Drop(X.PointPlaceDriver):
        def _drop_point(self, env):
            return np.array([5.0, 5.0, 1.0])

    act = RobocasaRecoveryActor.for_stage(object(), _Drop("can1"),
                                          repertoire.strategy("reapproach"))
    assert np.allclose(act._obj_xyz(), [5.0, 5.0, 1.0])
    assert act._grip() == D.GRIP_CLOSE  # a place stage is carrying


def test_stalled_place_stage_fails_segment_early_with_reach_stall(monkeypatch):
    _fake_world(monkeypatch)  # eef frozen: every step, zero progress
    monkeypatch.setenv("PH_TUNABLES", json.dumps({"stall_k": 10}))

    class _Drop(X.PointPlaceDriver):
        def _drop_point(self, env):
            return np.array([3.0, 3.0, 1.0])

        def done(self, env):
            return False

    drv = X.CompositeStageDriver({"drop": (lambda: _Drop("can1"), 300)}, "t")

    class _S:
        task = "drop"
    drv.enter_segment(object(), _S())
    steps = 0
    while not drv.exhausted:
        drv.act({})
        steps += 1
    assert steps < 300 and drv._stage.failure_mode == "reach_stall"
    diag = drv.segment_diagnostics(object())
    assert diag["failure_mode"] == "reach_stall" and diag["tunables_sha"] == D.tunables_sha()
    assert drv.segment_success(object()) is False
    # the stall geometry rides the diagnostics: eef / drop point / base + distances
    tr = diag["trace"]
    assert set(tr) == {"start", "stall", "end"} and tr["start"]["step"] == 1
    assert tr["stall"]["step"] == steps == tr["end"]["step"]
    assert tr["stall"]["eef"] == [1.0, 2.0, 1.0] and tr["stall"]["base"] == [0.0, 0.0, 0.0]
    assert tr["stall"]["target"] == [3.0, 3.0, 1.0 + D.tunables()["drop_over_dz"]]
    assert tr["stall"]["d_eef_target"] > D.tunables()["reach_tol"]
    assert tr["stall"]["d_base_target"] == pytest.approx(np.hypot(3.0, 3.0), abs=1e-3)


def test_segment_cap_tunable_overrides_the_stage_table(monkeypatch):
    monkeypatch.setenv("PH_TUNABLES", json.dumps({"segment_cap": 7}))
    drv = X.CompositeStageDriver({"x": (lambda: None, 300)}, "t")

    class _S:
        task = "x"
    drv.enter_segment(object(), _S())
    assert drv._cap == 7
