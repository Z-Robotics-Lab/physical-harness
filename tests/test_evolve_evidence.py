"""Base-lane checks for the CAUSAL evidence a first death carries: the per-step
``trace.series`` (what was commanded each step, not three snapshots), the drop
target's provenance, the ``upstream`` segment that parked the base, and the
campaign's successful-reference index. No simulator: the robocasa drivers' live
state readers are monkeypatched (as tests/test_robocasa_reach_robustness.py does)
and the evolve helpers run on synthetic trails.

The measured case these serve: recycle_cans seed 4243 dies at drop-can1 with the
SAME base pose at start/stall/end and the drop point 1.03 m away -- a segment whose
driver only ever commands the arm cannot close that gap, so the fix is upstream.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from plugins.embodiment_robocasa import drivers as D
from plugins.embodiment_robocasa import stage_extras as X
from plugins.embodiment_robocasa.recycle_driver import ClusterDropDriver
from scripts import evolve


def _fake_world(monkeypatch, eef=(0.88, -1.0, 0.88), base=(1.45, -1.73)):
    """A frozen kitchen: the eef and the base never move, whatever is commanded."""
    monkeypatch.setattr(D, "_eef", lambda env: np.array(eef, float))
    monkeypatch.setattr(D, "_base_pose", lambda env: (np.array(base, float), -0.47))


def _run_drop(point=(0.49, -1.36, 1.09)):
    class _Drop(X.PointPlaceDriver):
        def _drop_point(self, env):
            return np.array(point, float)

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
    return drv, steps


@pytest.mark.parametrize("stall_k", [10, 200])
def test_first_death_segment_carries_a_per_step_series(monkeypatch, stall_k):
    """Every row says what the driver COMMANDED: mode arm, no base channel, and the
    base pose never changes -- the segment could not have driven anywhere."""
    _fake_world(monkeypatch)
    monkeypatch.setenv("PH_TUNABLES", json.dumps({"stall_k": stall_k}))
    drv, steps = _run_drop()
    series = drv.segment_diagnostics(object())["trace"]["series"]
    assert 0 < len(series) <= D.SERIES_MAX and steps > stall_k
    assert {*series[0]} == {"step", "phase", "eef", "target", "base",
                            "d_eef", "d_base", "grip", "cmd"}
    assert all(r["cmd"]["mode"] == "arm" for r in series), series[:2]
    assert not any(c in r["cmd"]["nonzero"] for r in series for c in ("vx", "vy", "wyaw"))
    assert {tuple(r["base"]) for r in series} == {(1.45, -1.73, -0.47)}
    assert series[-1]["step"] == steps and series[0]["step"] == 1
    assert series[-1]["d_base"] == pytest.approx(np.hypot(1.45 - 0.49, -1.73 + 1.36), abs=1e-3)
    # the reach gap is mechanical: the drop point is farther from the base than the
    # arm can extend, and the base was never commanded
    assert series[-1]["d_base"] > D.REACH_MAX


def test_the_drop_point_carries_the_geometry_it_was_built_from(monkeypatch):
    """diagnostics.geometry names the fixture bbox, the knobs and the resulting point
    -- the drop point is EDGE_MARGIN past the stove edge, which is why it lands out of
    reach of a base parked at CARRY_STOP from the dock."""
    _fake_world(monkeypatch)
    corners = np.array([[x, y, z] for x in (-0.4, 0.4) for y in (-0.3, 0.3) for z in (0.9, 1.0)])

    class _Stove:
        def get_ext_sites(self, relative=False):
            return [corners]

    class _Counter:
        pos = (1.5, 0.0, 0.9)

    monkeypatch.setattr(D, "_fixture",
                        lambda env, name: _Stove() if name == "stove" else _Counter())
    drv = ClusterDropDriver("can1", 0)
    g = drv.diagnostics(object())["geometry"]
    t = D.tunables()
    assert g["stove_half_extent"] == pytest.approx(0.4) and g["stove_top_z"] == pytest.approx(1.0)
    assert g["toward_counter"] == [1.0, 0.0] and g["slot"] == 0
    assert (g["edge_margin"], g["spread"], g["drop_dz"]) == (
        t["drop_edge_margin"], t["drop_spread"], t["drop_dz"])
    assert g["point"][0] == pytest.approx(0.4 + t["drop_edge_margin"], abs=1e-3)
    assert g["point"][2] == pytest.approx(1.0 + t["drop_dz"], abs=1e-3)
    assert g["reach_max"] == D.REACH_MAX and g["d_base_point"] > D.REACH_MAX


def _trail():
    return [{"id": "nav-can1", "kind": "segment", "ok": True, "steps": 136,
             "trace_end": {"d_eef_target": 0.31, "d_base_target": 0.13}},
            {"id": "at-can1", "kind": "verify", "ok": True, "steps": None},
            {"id": "carry-can1", "kind": "segment", "ok": True, "steps": 174,
             "trace_end": {"base": [1.45, -1.728, -0.473], "d_base_target": 0.62}},
            {"id": "recover-drop-can1", "kind": "recovery", "ok": True, "steps": 45},
            {"id": "drop-can1", "kind": "segment", "ok": False, "steps": 65}]


def test_the_first_death_row_names_the_segment_that_parked_the_base():
    """The upstream row skips the verify and the recovery node: the carry leg is what
    left the base 1.03 m from the drop point."""
    trail = _trail()
    evolve._link_upstream(trail, "drop-can1", {"carry-can1": "carry", "nav-can1": "nav"})
    assert trail[-1]["upstream"] == {
        "node": "carry-can1", "skill": "carry", "steps": 174,
        "trace_end": {"base": [1.45, -1.728, -0.473], "d_base_target": 0.62}}
    assert all("upstream" not in n for n in trail[:-1])
    evolve._link_upstream(trail := _trail(), None, {})          # nothing died: no row
    assert all("upstream" not in n for n in trail)


def test_reference_keeps_the_last_successful_pass_of_every_segment():
    """Successes accrue across rounds; a node that has only ever died is listed as
    null; a later round's pass replaces the earlier one."""
    doc: dict = {}
    kept = {"seeds": {"4243": {"trail": _trail()}}}
    ref = evolve.update_reference(doc, kept, 1)
    assert ref["nav-can1"] == {"node": "nav-can1", "seed": 4243, "steps": 136,
                               "d_eef": 0.31, "d_base": 0.13, "round": 1}
    assert ref["drop-can1"] is None and "at-can1" not in ref and "recover-drop-can1" not in ref
    later = _trail()
    later[0]["ok"] = False        # nav-can1 dies this round: the round-1 pass stands
    later[-1].update(ok=True, trace_end={"d_eef_target": 0.02, "d_base_target": 0.9})
    evolve.update_reference(doc, {"seeds": {"4244": {"trail": later}}}, 2)
    assert doc["reference"]["nav-can1"]["round"] == 1
    assert doc["reference"]["drop-can1"] == {"node": "drop-can1", "seed": 4244, "steps": 65,
                                             "d_eef": 0.02, "d_base": 0.9, "round": 2}
