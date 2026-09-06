"""Diagnostic claims must follow sampled excitation/response, not task labels."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from plugins.embodiment_robocasa import drivers as D
from plugins.rsi.diagnosis import analyze_trace

GROUP = {"translation": {"state": "position", "axes": [0, 1], "target": "goal",
                         "commands": ["u", "v"], "noise": 0.003}}


def trace(points, *, goal=(1.0, 1.0), active=True):
    return {"groups": copy.deepcopy(GROUP), "series": [
        {"step": i, "phase": "move", "position": list(p), "goal": list(goal),
         "cmd": {"nonzero": ["u"] if active else []}}
        for i, p in enumerate(points)],
        "sampling": {"kind": "downsampled", "stride": 5}}


def finding(data):
    return analyze_trace(data)["findings"][0]


def test_unexcited_and_unresponsive_are_distinct_observations():
    fixed = [(0, 0)] * 5
    no_command = analyze_trace(trace(fixed, active=False))
    assert no_command["findings"][0]["kind"] == "unexcited_in_samples"
    assert no_command["coverage"]["kind"] == "downsampled"
    issued = finding(trace(fixed))
    assert issued["kind"] == "commanded_without_observed_motion"
    assert issued["evidence"]["commanded_intervals"] == 4
    assert "controllability" in no_command["limits"]


def test_response_direction_comes_from_motion_in_declared_coordinates():
    toward = finding(trace([(i / 10, i / 10) for i in range(5)]))
    assert toward["kind"] == "progressing"
    away = finding(trace([(-i / 10, -i / 10) for i in range(5)]))
    assert away["kind"] == "moved_away" and away["evidence"]["progress"] < 0
    # Oscillation in x cannot explain any observed y response. No reach limit,
    # robot name, stage label or physics-specific rule enters this deduction.
    orthogonal = finding(trace([(0, 0), (0.2, 0), (0, 0), (0.2, 0), (0, 0)], goal=(0, 1)))
    assert orthogonal["kind"] == "residual_outside_observed_span"
    assert orthogonal["evidence"]["response_rank"] == 1
    assert orthogonal["evidence"]["residual_outside_response_span"] == pytest.approx(1)


def test_phase_and_target_changes_cannot_manufacture_progress():
    data = trace([(0, 0)] * 6)
    # Bringing the target closer halfway through is not robot progress. Only
    # the final fixed-target window is examined, with its own first distance.
    for row in data["series"][3:]:
        row["goal"] = [0.1, 0.1]
        row["phase"] = "new-goal"
    result = finding(data)
    assert result["kind"] == "commanded_without_observed_motion"
    assert result["evidence"]["first_step"] == 3 and result["evidence"]["progress"] == 0


def test_a_smoother_scalar_cannot_hide_an_unobserved_response_direction():
    result = finding(trace([(i / 10, 0) for i in range(5)]))
    assert result["evidence"]["progress"] > 0
    assert result["kind"] == "residual_outside_observed_span"
    assert result["evidence"]["residual_outside_response_span"] == pytest.approx(1)


def test_a_slowly_moving_target_does_not_count_as_motion_and_missing_mode_is_unknown():
    data = trace([(0, 0)] * 10)
    for i, row in enumerate(data["series"]):
        row["goal"] = [1 - i * 0.002, 1]
    assert finding(data)["kind"] == "unknown"  # only two samples of a fixed target
    data = trace([(0, 0)] * 4)
    data["groups"]["translation"]["mode"] = "move"
    assert finding(data)["kind"] == "unknown"


@pytest.mark.parametrize("missing", ["cmd", "position", "goal"])
def test_missing_evidence_is_unknown(missing):
    data = trace([(0, 0)] * 4)
    del data["series"][-2][missing]
    assert finding(data)["kind"] == "unknown"
    assert analyze_trace({})["status"] == "unknown"
    assert analyze_trace({"series": data["series"]})["fingerprint"] == []


def test_nonfinite_and_missing_group_declarations_are_not_inferred():
    data = trace([(0, 0)] * 4)
    data["series"][-1]["position"] = [float("nan"), 0]
    assert finding(data)["kind"] == "unknown"
    del data["groups"]["translation"]["noise"]
    assert finding(data)["kind"] == "unknown"


def test_driver_trace_declares_sampling_and_keeps_signed_commands(monkeypatch):
    monkeypatch.setattr(D, "_eef", lambda env: np.array([0.0, 0.0, 0.0]))
    monkeypatch.setattr(D, "_base_pose", lambda env: (np.array([0.0, 0.0]), 0.0))
    trace = D.Trace()
    trace.at("start", object(), [1.0, 1.0, 1.0])
    action = D._zero()
    action[0], action[1], action[D.MODE] = -0.5, 0.25, -1
    for i in range(100):
        trace.step = i
        trace.sample(object(), "move", action)
    recorded = trace.dump(object())
    assert recorded["sampling"]["raw_samples"] == 100
    assert recorded["sampling"]["kind"] == "downsampled"
    assert recorded["series"][0]["cmd"]["values"]["dx"] == -0.5
    found = {f["channel"]: f["kind"] for f in analyze_trace(recorded)["findings"]}
    assert found == {"base_translation": "unexcited_in_samples",
                     "eef_translation": "commanded_without_observed_motion"}


def test_recorded_robot_traces_distinguish_missing_excitation_and_reversed_motion():
    data = json.loads((Path(__file__).parent / "fixtures" /
                       "response_recycle_round588.json").read_text())
    left, right = [analyze_trace(row["trace"], D.TRACE_GROUPS) for row in data["records"]]
    # The same data-only algorithm separates two observed failures; no reach
    # radius, task identifier, or driver failure_mode is provided to it.
    base, arm = left["findings"]
    assert base["kind"] == "unexcited_in_samples"
    assert base["evidence"]["movement"] == 0
    assert arm["evidence"]["progress"] > 0  # scalar distance alone looks promising
    assert arm["kind"] == "residual_outside_observed_span"
    assert right["findings"][0]["kind"] == "moved_away"
    assert right["findings"][0]["evidence"]["progress"] < -0.3
    assert left["coverage"]["kind"] == "unspecified"  # legacy coverage is not invented
