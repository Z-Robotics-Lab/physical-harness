"""Opt-in real RoboCasa rollouts driven by explicit fake model replies.

This tests the runtime, paired simulator trials, and station evidence path.
The fixed replies make no claim about model quality or cross-task scaling.
Run with the RoboCasa venv and MUJOCO_GL=egl; never in the default base lane.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from board import store as bs
from harness import media

REPO = Path(__file__).resolve().parent.parent
RUNTIME = REPO / "scripts" / "harness_runtime.py"


def _run(tmp_path, task, seeds, replies, *, rounds, timeout):
    runs = tmp_path / "runs"
    session = runs / "session-main"
    response = tmp_path / "model-replies.json"
    response.write_text(json.dumps(replies))
    submitted = bs.submit_brief(runs, json.dumps({"kind": "evolve", "task": task,
        "seeds": seeds, "rounds": rounds, "arm": "scripted", "confirm_seeds": 0}))
    campaign = session / "campaigns" / f"evolve-{task}" / "campaign.json"
    messages, frames, trails = set(), set(), []
    output = tmp_path / "runtime.log"
    with output.open("w") as log:
        proc = subprocess.Popen([sys.executable, str(RUNTIME), "--session-dir", str(session),
            "--drain", "--mode", "evolution"], cwd=REPO, stdout=log, stderr=log,
            env={**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(REPO),
                 "PH_MODEL_ENDPOINT_FAKE": str(response)})
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    live = json.loads(campaign.read_text())["live"]
                    messages.add(live["message"])
                    nodes = live.get("nodes")
                    if nodes and nodes != (trails[-1] if trails else None):
                        trails.append(nodes)
                    frames.add((session / "frame.jpg").stat().st_mtime)
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    pass
                time.sleep(0.2)
            if proc.poll() is None:
                proc.terminate()
                pytest.fail(f"simulator exceeded {timeout} seconds; log: {output}")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=60)
    assert proc.returncode == 0, output.read_text()[-4000:]
    assert (session / "done" / submitted["submitted"]).exists(), output.read_text()[-4000:]
    assert not [r for r in bs.chain_rows(session) if r["kind"] == "runtime.task_error"]
    assert any(messages) and len(frames) >= 2
    doc = json.loads(campaign.read_text())
    assert doc["status"] == doc["live"]["phase"] == "done"
    return session, doc, trails


@pytest.mark.robocasa
def test_real_kitchen_baseline_and_model_abstention_preserve_station_media(tmp_path):
    task = "kitchen_thaw"
    session, doc, _ = _run(tmp_path, task, [429002, 429003],
        [{"kind": "none", "payload": {}, "summary": "Retain the measured baseline.",
          "rationale": "This fixture requests no intervention."}], rounds=1, timeout=1800)
    row, = doc["rounds"]
    assert row["llm"]["status"] == "abstained" and row["tried"]["kind"] == "none"
    assert row["after"] is None and row["trial"] is None and row["after_seeds"] == []
    assert 0 <= row["before"] <= 2 and row["published"] is False
    assert len(bs.rsi_series(session, task)) == 1
    steps = [r["data"] for r in bs.chain_rows(session) if r["kind"] == "rsi_step"]
    assert len(steps) == 1 and steps[0]["task"] == task
    paths = bs.rsi_frames(session, task, 1)["media"]
    assert paths == row["media"]
    for rel in paths:
        assert "/baseline/" in rel
        assert 0 < (session / rel).stat().st_size <= media.MAX_BYTES


@pytest.mark.robocasa
def test_real_recycle_trials_use_the_models_explicit_upstream_node_and_values(tmp_path):
    from harness.manifest import mount_params

    ref = "plugins.embodiment_robocasa.recycle_driver:provider"
    original = mount_params(ref)["tunables"]["carry_stop"]
    values = [round(original * .81, 6), round(original * .93, 6)]
    answers = [{"kind": "tunables", "payload": {"node": "carry-can1", "ref": ref,
                "path": ["tunables", "carry_stop"], "to": value},
                "summary": "Measure this declared upstream action parameter.",
                "rationale": "Test the proposed setting against paired task-world outcomes."}
               for value in values]
    answers = [item for answer in answers for item in (
        {"op": "inspect", "args": {"view": "parameter", "node": "carry-can1", "parameter": "carry_stop"}}, answer)]
    session, doc, trails = _run(tmp_path, "recycle_cans", [4243], answers, rounds=2, timeout=5400)
    assert len(doc["rounds"]) == 2 and trails
    for row, value in zip(doc["rounds"], values):
        assert row["proposer"] == "llm" and row["llm"]["status"] == "proposed"
        assert row["tried"]["node"] == "carry-can1" and row["tried"]["kind"] == "tunables"
        assert row["tried"]["detail"]["to"] == value
        assert row["trial"] is not None and row["after"] is not None and len(row["after_seeds"]) == 1
        assert row["experiments"]["before"] != row["experiments"]["after"]
        assert row["evaluation"]["after"] is not None and row["published"] is False
        baseline = row["per_seed"][0]
        upstream = next(n for n in baseline["nodes"] if n["id"] == "carry-can1")
        assert upstream["steps"] > 0
        for rel in bs.rsi_frames(session, "recycle_cans", row["round"])["media"]:
            assert 0 < (session / rel).stat().st_size <= media.MAX_BYTES
