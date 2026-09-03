"""The lightweight evolve loop on the REAL robocasa kitchen: kitchen_thaw x 2 seeds,
arm scripted, ONE round, submitted to a tmp evolution-mode runtime drained headless
(MUJOCO_GL=egl) -- the same shape as tests/test_suite_robocasa_e2e.py. Asserts the
campaign lands (rsi_series one row, one rsi_step row), that a seed that
succeeded left a <1 MB media clip, and that while it ran the ``live`` block carried
a message and the 取景窗 frame file advanced. Nothing here touches runs/.

Run: cd <repo> && MUJOCO_GL=egl <robocasa-venv>/bin/python -m pytest -m robocasa tests/test_evolve_robocasa_e2e.py
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
TASK = "kitchen_thaw"


@pytest.mark.robocasa
def test_one_evolve_round_on_the_real_kitchen(tmp_path):
    runs = tmp_path / "runs"
    session = runs / "session-main"
    seeds = [429002, 429003]
    name = bs.submit_brief(runs, json.dumps(
        {"kind": "evolve", "task": TASK, "seeds": seeds, "rounds": 1, "arm": "scripted",
         "max_actuations": 24, "max_replans": 1, "proposer": "rules"}))["submitted"]
    proc = subprocess.Popen(
        [sys.executable, str(RUNTIME), "--session-dir", str(session), "--drain",
         "--mode", "evolution"],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(REPO)})
    campaign = session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    messages, frame_ts = set(), set()   # what the board would show while it runs
    deadline = time.monotonic() + 1800
    while proc.poll() is None and time.monotonic() < deadline:
        try:
            messages.add(json.loads(campaign.read_text())["live"]["message"])
            frame_ts.add((session / "frame.jpg").stat().st_mtime)
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    _, err = proc.communicate(timeout=60)
    proc.stderr = err
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert any(messages), messages
    assert len(frame_ts) >= 2, frame_ts   # egl runtime -> frames on -> evolve mirrors episodes
    assert bs.rsi_run(session, TASK)["live"]["phase"] == "done"
    assert (session / "done" / name).exists(), proc.stderr[-4000:]
    rows = bs.chain_rows(session)
    assert not [r for r in rows if r["kind"] == "runtime.task_error"], proc.stderr[-4000:]
    series = bs.rsi_series(session, TASK)
    assert len(series) == 1 and series[0]["round"] == 1
    assert 0 <= series[0]["before"] <= 2 and series[0]["best"] >= series[0]["before"]
    steps = [r["data"] for r in rows if r["kind"] == "rsi_step"]
    assert [s["round"] for s in steps] == [1] and steps[0]["task"] == TASK
    run = bs.rsi_run(session, TASK)
    assert run["status"] == "done" and run["cursor"] == 1 and run["latest"]["round"] == 1
    assert run["latest"]["tried"]["kind"] in ("executor", "tunables", "none")
    # media: every kept clip is a verified segment, under 1 MB, listed by rsi_frames
    frames = bs.rsi_frames(session, TASK, 1)
    assert frames == run["latest"]["media"]
    for rel in frames:
        f = session / rel
        assert f.is_file() and 0 < f.stat().st_size <= media.MAX_BYTES, rel
    kept = {s: media.index_of(session / "media", TASK, s) for s in seeds}
    if run["latest"]["best"] > 0:   # a successful seed verified every segment: clips exist
        assert any(kept.values()), kept
    assert all(f"media/{TASK}/{s}/{v['file']}" in frames
               for s, idx in kept.items() for v in idx.values())


@pytest.mark.robocasa
def test_evolve_recycle_cans_4243_perturbs_the_hinted_drop_knob(tmp_path):
    """Two rounds on the seed that dies at drop-can1 with reach_stall: round 1
    tries the card's first hinted knob (drop_edge_margin, -30%), round 2 the other
    direction, and the trial's suite ran under a different tunables_sha than the
    baseline. Success is not required -- the per-seed rows are the finding."""
    task = "recycle_cans"
    runs = tmp_path / "runs"
    session = runs / "session-main"
    name = bs.submit_brief(runs, json.dumps(
        {"kind": "evolve", "task": task, "seeds": [4243, 4243], "rounds": 2, "arm": "scripted",
         "proposer": "rules"}))["submitted"]
    proc = subprocess.Popen(
        [sys.executable, str(RUNTIME), "--session-dir", str(session), "--drain", "--mode", "evolution"],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(REPO)})
    campaign = session / "campaigns" / f"evolve-{task}" / "campaign.json"
    trails, deadline = [], time.monotonic() + 5400   # every distinct live node trail the page would show
    while proc.poll() is None and time.monotonic() < deadline:
        try:
            nodes = json.loads(campaign.read_text())["live"]["nodes"]
            if nodes and nodes != (trails[-1] if trails else None):
                trails.append(nodes)
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    _, err = proc.communicate(timeout=60)
    proc.stderr = err
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert (session / "done" / name).exists(), proc.stderr[-4000:]
    assert not [r for r in bs.chain_rows(session) if r["kind"] == "runtime.task_error"], proc.stderr[-4000:]
    doc = json.loads((session / "campaigns" / f"evolve-{task}" / "campaign.json").read_text())
    r1, r2 = doc["rounds"]
    print("per_seed", [(r["round"], r["tried"]["kind"], r["tried"]["detail"].get("path"),
                        r["tried"]["detail"].get("to"), r["before"], r["after"], r["per_seed"],
                        r["after_seeds"]) for r in doc["rounds"]])
    base = r1["per_seed"][0] if not r1["published"] else None
    if base:   # the trail carries steps on the actuated nodes (perceive/decide nodes have none)
        dead = next(n for n in base["nodes"] if n["id"] == base["first_death"])
        assert dead["ok"] is False and isinstance(dead["steps"], int) and base["elapsed_s"] > 0, base
        assert sum(isinstance(n["steps"], int) for n in base["nodes"] if n["ok"] is True) >= 1, base
    if base and base["failure_mode"] == "reach_stall":   # the acceptance fact this round answers
        # while it ran, the live trail showed nodes verified ok before the drop node died
        # (the drop node's ok=False is transient: the replan resets it, so only ok=True is asserted)
        seen = [t for t in trails if base["first_death"] in [n["id"] for n in t]]
        assert seen and any(any(n["ok"] is True for n in t[:[n["id"] for n in t].index(base["first_death"])])
                            for t in seen), trails[-1:]
        assert r1["tried"]["kind"] == "tunables", r1["tried"]
        d = r1["tried"]["detail"]
        assert d["path"][-1] == "drop_edge_margin" and d["hint"] == "reach_stall"
        assert d["to"] == pytest.approx(0.07) and d["from"] == pytest.approx(0.10)
        assert r2["tried"]["detail"]["path"][-1] == "drop_edge_margin"
        assert r2["tried"]["detail"]["to"] == pytest.approx(0.13)
        # the overlay reached the driver: the trial's dying node sealed another tunables_sha
        trial = r1["after_seeds"][0]
        assert len(trial["tunables_sha"] or "") == 16 and trial["tunables_sha"] != base["tunables_sha"]
    elif r1["tried"]["kind"] == "tunables":
        assert r1["tried"]["detail"]["hint"] in (None, base and base["failure_mode"])
