"""Item 2 on the REAL kitchen: execution robustness end to end through the real
scripts/harness_runtime.py subprocess headless (MUJOCO_GL=egl), tmp session only.

recycle_cans seed 4243 (the "four as-is retries at drop-can1" case): the replan
progress rule allows ONE as-is re-run per (graph, node, fault signature) and
refuses the next as no_progress; the stalled drop segment fails early with
failure_mode "reach_stall" (< 300 steps) instead of burning its cap; and at least
one recovery (reapproach / base_nudge / release_reset) is executed. kitchen_thaw
seed 41 runs on the card's max_actuations (20) as the default budget (its full-chain
pass is a strict xfail: the place/close capability walls are still open).

Run: cd <repo> && MUJOCO_GL=egl <robocasa-venv>/bin/python -m pytest -m robocasa \\
    tests/test_robocasa_recovery_e2e.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from board import store as bs
from harness.manifest import discover
from plugins.embodiment_robocasa import drivers as D
from plugins.rsi.repertoire import strategies_for

REPO = Path(__file__).resolve().parent.parent
RUNTIME = REPO / "scripts" / "harness_runtime.py"
RECOVERIES = ("reapproach", "base_nudge", "release_reset")


def drain_task(tmp_path: Path, brief: dict, timeout: float):
    """submit one task brief; drain it in a real runtime; return chain + op events."""
    runs = tmp_path / "runs"
    session = runs / "session-main"
    name = bs.submit_brief(runs, json.dumps(brief))["submitted"]
    proc = subprocess.run(
        [sys.executable, str(RUNTIME), "--session-dir", str(session), "--drain"],
        cwd=str(REPO), capture_output=True, text=True, timeout=timeout, check=False,
        env={**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(REPO)})
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert (session / "done" / name).exists(), proc.stderr[-4000:]
    rows = bs.chain_rows(session)
    assert not [r for r in rows if r["kind"] == "runtime.task_error"], proc.stderr[-4000:]
    return rows, bs.read_runtime_events(session)["events"]


def _kinds(rows, kind):
    return [r["data"] for r in rows if r["kind"] == kind]


def _executed_faults(rows) -> list[tuple]:
    """(graph_sha, node, signature) per executed graph that faulted, chain order."""
    out, sha = [], None
    for r in rows:
        if r["kind"] == "task.plan" and r["data"]["legal"]:
            sha = r["data"]["graph_sha"]
        elif r["kind"] == "task.fault":
            out.append((sha, r["data"]["node"], r["data"]["signature"]))
    return out


@pytest.mark.robocasa
def test_repertoire_lists_the_three_reach_recoveries():
    assert set(RECOVERIES) <= set(strategies_for("embodiment_robocasa"))


@pytest.mark.robocasa
def test_recycle_cans_4243_never_retries_as_is_twice_stalls_early_and_recovers(tmp_path):
    rows, events = drain_task(tmp_path, {
        "kind": "task", "task": "recycle_cans", "seed": 4243, "arm": "scripted",
        "max_replans": 4}, timeout=2400)
    end = _kinds(rows, "task.plan_complete")
    assert len(end) == 1
    end = end[0]
    # 1. no two consecutive as-is retries: an identical (graph, node, signature)
    #    may be re-run once; a third identical execution never happens.
    seq = _executed_faults(rows)
    triples = [seq[i:i + 3] for i in range(len(seq) - 2) if seq[i] == seq[i + 1] == seq[i + 2]]
    assert not triples, triples
    rejected = _kinds(rows, "task.replan_rejected")
    if not end["success"]:
        # an honest end: the refusal is on the chain, as is the folded fault
        assert any(r["reason"] == "no_progress" for r in rejected), (rejected, end["faults"])
        assert any(f["kind"] == "no_progress" for f in end["faults"]), end["faults"]
        assert all(f["kind"] != "budget" for f in end["faults"]), end["faults"]
    # 2. the stalled reach fails the segment early with reach_stall, never at the cap
    stalled = [e for e in events if e.get("kind") == "actuation_end"
               and (e.get("diagnostics") or {}).get("failure_mode") == "reach_stall"]
    assert stalled, [e for e in events if e.get("kind") == "actuation_end"][-3:]
    assert all(0 < e["steps"] < 300 for e in stalled), [(e["node"], e["steps"]) for e in stalled]
    assert all(len((e.get("diagnostics") or {}).get("tunables_sha", "")) == 16 for e in stalled)
    # 2b. the stall geometry is sealed with it: where the eef, the target and the
    #     base were at start / stall / end (the drop point was never reached)
    for e in stalled:
        tr = e["diagnostics"]["trace"]
        print("trace", e["node"], tr)
        assert set(tr) == {"start", "stall", "end"}, (e["node"], tr)
        assert tr["stall"]["d_eef_target"] > D.tunables()["reach_tol"], (e["node"], tr["stall"])
        assert 0 < tr["stall"]["step"] <= tr["end"]["step"], (e["node"], tr)
    nudged = [e for e in events if e.get("kind") == "actuation_end"
              and (e.get("diagnostics") or {}).get("strategy") == "base_nudge"]
    for e in nudged:   # a base_nudge repair seals how far the base actually went
        assert 0 < e["diagnostics"]["base_travel"] <= D.tunables()["nudge_max"] + 0.05, e["diagnostics"]
    # 3. at least one recovery node was executed on the live world
    executed = [v["node"] for v in _kinds(rows, "task.verify")]
    plans = _kinds(rows, "task.plan")
    rec_nodes = {n["id"] for p in plans if p["legal"] for n in p["graph"]["nodes"]
                 if n.get("skill") in RECOVERIES or n.get("kind") == "recovery"}
    recovered = [n for n in executed if n in rec_nodes] + _kinds(rows, "task.recovery")
    assert recovered, {"rejected": rejected, "faults": [
        (f["kind"], f.get("node")) for f in end["faults"]], "plans": len(plans)}


KITCHEN_THAW_DECLARED = 20


@pytest.fixture(scope="module")
def kitchen_thaw_41(tmp_path_factory):
    """ONE kitchen_thaw drain, seed 41, no max_actuations on the brief (the card's)."""
    rows, _ = drain_task(tmp_path_factory.mktemp("kt"), {
        "kind": "task", "task": "kitchen_thaw", "seed": 41, "arm": "scripted",
        "max_replans": 1}, timeout=900)
    return _kinds(rows, "task.plan_complete")[0]


@pytest.mark.robocasa
def test_kitchen_thaw_runs_on_the_card_max_actuations_as_default(kitchen_thaw_41):
    card = tomllib.loads((REPO / "plugins/mission_kitchen_thaw/manifest.toml").read_text())
    declared = card["task_bindings"]["kitchen_thaw"]["max_actuations"]
    assert discover().task_bindings["kitchen_thaw"]["max_actuations"] == declared \
        == KITCHEN_THAW_DECLARED
    end = kitchen_thaw_41
    # the task default (3) would have budget-faulted at node 3; the card's 20 carries
    # the 14-node chain (plus replan headroom) and is never exceeded
    assert 3 < end["actuations"] <= declared, (end["actuations"], end["faults"])
    assert not [f for f in end["faults"] if f["kind"] == "budget"], end["faults"]


@pytest.mark.robocasa
@pytest.mark.xfail(strict=True, reason=(
    "kitchen_thaw scripted has never completed the chain: thawed 0/150 in all three "
    "calibration rounds (STATUS.md 2026-08-26, place 0/22 and the microwave close/press "
    "walls are open capability gaps, tests/test_robocasa_drivers.py xfails). The stall "
    "detector only ends the same deaths sooner. Remove this marker when a seed passes."))
def test_kitchen_thaw_seed_41_completes_the_chain(kitchen_thaw_41):
    assert kitchen_thaw_41["success"] is True, kitchen_thaw_41["faults"]
