"""Part B of the 2026-09-02 acceptance: tunables reach every robocasa driver
provider through ``manifest.mount_params``; a loaded nav leg that stops
approaching its dock fails early with failure_mode "nav_stall"; the pack_lunch /
kitchen_thaw planners answer a no_progress fault with the card's mapped recovery
primitive (base stall -> redock_retry, grasp -> regrasp_kitchen, placement ->
reapproach), re-inserting done repairs byte-identically. Base lane except the
robocasa-marked drain at the end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from harness import protocol
from harness.manifest import discover, mount_params
from harness.registry import load_provider
from plugins.embodiment_robocasa import drivers as D
from plugins.embodiment_robocasa import recovery as R

REPO = Path(__file__).resolve().parent.parent
CARD = "plugins.embodiment_robocasa"
PROVIDERS = [f"{CARD}.{m}:provider" for m in ("recycle_driver", "lunch_driver", "kitchen_driver")]
MANIFEST_TUNABLES = tomllib.loads(
    (REPO / "plugins/embodiment_robocasa/manifest.toml").read_text())["tunables"]


@pytest.mark.parametrize("ref", PROVIDERS)
def test_card_tunables_reach_every_driver_provider_through_mount_params(ref, monkeypatch):
    monkeypatch.delenv("PH_MOUNT_PARAMS_OVERRIDE", raising=False)
    assert mount_params(ref)["tunables"] == MANIFEST_TUNABLES
    # an evolve trial's overlay merges into the same table and lands in the one
    # shared read every stage driver uses (drivers.tunables / tunables_sha)
    monkeypatch.setenv("PH_MOUNT_PARAMS_OVERRIDE",
                       json.dumps({ref: {"tunables": {"stall_k": 48}}}))
    params = mount_params(ref)
    assert params["tunables"]["stall_k"] == 48 and params["tunables"]["hover_dz"] == 0.08
    load_provider(ref, params)
    try:
        assert D.tunables()["stall_k"] == 48 and D.tunables()["hover_dz"] == 0.08
        assert D.tunables_sha() != D._tunables("")  # a tuned run is distinguishable
    finally:
        D.mount_tunables(None)
    assert D.tunables() == MANIFEST_TUNABLES


def test_mounted_tunables_refuse_unknown_keys():
    D.mount_tunables({"bogus": 1})
    try:
        with pytest.raises(KeyError):
            D.tunables()
    finally:
        D.mount_tunables(None)


def test_card_tunable_hints_reach_the_proposer_through_mount_params():
    hints = tomllib.loads(
        (REPO / "plugins/embodiment_robocasa/manifest.toml").read_text())["tunable_hints"]
    params = mount_params("plugins.embodiment_robocasa.recycle_driver:provider")
    assert params["tunable_hints"] == hints
    assert hints["reach_stall"][:2] == ["drop_edge_margin", "drop_over_dz"]
    assert set(sum(hints.values(), [])) <= set(MANIFEST_TUNABLES)
    for k in ("drop_over_dz", "drop_edge_margin", "drop_spread", "drop_dz"):
        assert isinstance(MANIFEST_TUNABLES[k], float), k


@pytest.mark.parametrize("task, node, mode, strategy", [
    ("recycle_cans", "drop-can1", "reach_stall", "base_nudge"),
    ("recycle_cans", "drop-can1", None, "reapproach"),
    ("recycle_cans", "carry-can1", "reach_stall", "redock_retry"),
    ("pack_lunch", "pack-hot0", "reach_stall", "base_nudge"),
    ("kitchen_thaw", "place", "reach_stall", "base_nudge"),
])
def test_planner_overrides_the_recovery_by_failure_mode(task, node, mode, strategy):
    planner = load_provider(discover().task_bindings[task]["planner"])
    plan = planner.plan({"task": task, "fault": {
        "kind": "no_progress", "node": node, "failure_mode": mode}})
    rec = next(n for n in plan["nodes"] if n["id"] == f"recover-{node}")
    assert rec["skill"] == strategy and rec["kind"] == "recovery"
    # a later fault re-inserts the repair that RAN (fault.recoveries_done), not
    # the table's default -- replan_monotone keeps the done node byte-identical
    again = planner.plan({"task": task, "fault": {
        "kind": "node_failure", "node": "report", "nodes_done": [f"recover-{node}"],
        "recoveries_done": {node: strategy}}})
    assert next(n for n in again["nodes"] if n["id"] == f"recover-{node}") == rec


def test_mount_params_leaves_other_cards_untouched():
    assert "tunables" not in mount_params("plugins.embodiment_robosuite:provider")
    assert mount_params("nowhere:provider") == {}


def _fake_kitchen(monkeypatch, base_xy):
    """A carry leg on a stubbed world: the dock at the origin, the base wherever
    ``base_xy`` says (a mutable list the test moves), the arm already stowed."""
    monkeypatch.setattr(D, "_base_pose", lambda env: (np.asarray(base_xy, float), 0.0))
    monkeypatch.setattr(D, "_eef", lambda env: np.array([base_xy[0] + 0.40, base_xy[1] - 0.15, 1.25]))
    monkeypatch.setattr(D, "_base_action", lambda env, gxy, yaw, grip: np.zeros(D.ADIM))
    nav = D.NavigateDriver("stove", carry=True)
    nav._goal = (np.zeros(2), 0.0)
    return nav


def test_loaded_leg_that_stops_approaching_fails_with_nav_stall(monkeypatch):
    base = [3.0, 0.0]
    nav = _fake_kitchen(monkeypatch, base)
    k = D.tunables()["stall_k"]
    for _ in range(3):           # approaching: no stall
        nav.act(None, None)
        base[0] -= 0.05
    for _ in range(k + 1):       # wedged far from the dock
        nav.act(None, None)
    assert nav.failure_mode == "nav_stall" and not nav.done(None)
    # the same plateau NEAR the dock is a stall-arrival: done, no failure
    base[:] = [0.75, 0.0]
    nav = _fake_kitchen(monkeypatch, base)
    for _ in range(k + 1):
        nav.act(None, None)
    assert nav.done(None) and nav.failure_mode is None


def test_unloaded_leg_that_stops_approaching_fails_with_nav_stall(monkeypatch):
    """Seed 4244's death: nav-can1 (NavToObjectDriver, carry=False) plateaued far
    from its dock and sealed failure_mode None -- the watchdog only ran loaded."""
    from plugins.embodiment_robocasa import stage_extras as X

    k = D.tunables()["stall_k"]
    for make in (lambda: D.NavigateDriver("stove"), lambda: X.NavToObjectDriver("stove", "can1")):
        base = [3.0, 0.0]
        monkeypatch.setattr(D, "_base_pose", lambda env: (np.asarray(base, float), 0.0))
        monkeypatch.setattr(D, "_base_action", lambda env, gxy, yaw, grip: np.zeros(D.ADIM))
        nav = make()
        nav._goal = (np.zeros(2), 0.0)
        for _ in range(3):
            nav.act(None, None)
            base[0] -= 0.05
        for _ in range(k + 1):       # wedged far out (the local reverse included)
            nav.act(None, None)
        assert nav.failure_mode == "nav_stall" and not nav.done(None)
        base[:] = [0.15, 0.0]        # inside NAV_POS_TOL, only the yaw settling
        nav = make()
        nav._goal = (np.zeros(2), 0.0)
        for _ in range(k + 1):
            nav.act(None, None)
        assert nav.failure_mode is None and nav.done(None)


def test_redock_retry_keeps_the_grip_on_a_loaded_leg(monkeypatch):
    monkeypatch.setattr(D, "_base_pose", lambda env: (np.zeros(2), 0.0))
    nav = D.NavigateDriver("stove", carry=True)
    act = R.RobocasaRecoveryActor.for_stage(object(), nav, discover_strategy("redock_retry"))
    assert act.carry
    a = act.act({})
    assert a[D.GRIP] == D.GRIP_CLOSE and a[D.MODE] == D.GRIP_CLOSE and a[7] < 0
    empty = R.RobocasaRecoveryActor.for_stage(object(), D.NavigateDriver("fridge"),
                                              discover_strategy("redock_retry"))
    assert empty.act({})[D.GRIP] == D.GRIP_OPEN


def discover_strategy(name):
    import importlib

    module, _, attr = discover().recoveries[name][1].partition(":")
    return getattr(importlib.import_module(module), attr)


@pytest.mark.parametrize("task, node, strategy", [
    ("pack_lunch", "carry-hot1", "redock_retry"),
    ("pack_lunch", "grasp-cold0", "regrasp_kitchen"),
    ("pack_lunch", "pack-hot0", "reapproach"),
    ("kitchen_thaw", "grasp", "regrasp_kitchen"),
    ("kitchen_thaw", "nav-micro", "redock_retry"),
    ("kitchen_thaw", "place", "reapproach"),
])
def test_planner_answers_no_progress_with_the_mapped_recovery(task, node, strategy):
    planner = load_provider(discover().task_bindings[task]["planner"])
    base = planner.plan({"task": task})
    plan = planner.plan({"task": task, "fault": {"kind": "no_progress", "node": node,
                                                 "nodes_done": []}})
    rec = next(n for n in plan["nodes"] if n["id"] == f"recover-{node}")
    assert rec["skill"] == strategy and rec["kind"] == "recovery"
    assert f"recover-{node}" in next(n for n in plan["nodes"] if n["id"] == node)["after"]
    assert protocol.content_id(plan) != protocol.content_id(base)
    assert set(discover().recoveries) >= {strategy}
    # a later fault keeps the done repair byte-identically (replan_monotone)
    again = planner.plan({"task": task, "fault": {
        "kind": "node_failure", "node": "report",
        "nodes_done": [node, f"recover-{node}"]}})
    assert next(n for n in again["nodes"] if n["id"] == f"recover-{node}") == rec
    assert all(n in {m["id"]: m for m in again["nodes"]} for n in (node, f"recover-{node}"))


def test_planner_leaves_an_unmapped_stage_alone():
    planner = load_provider(discover().task_bindings["kitchen_thaw"]["planner"])
    plan = planner.plan({"task": "kitchen_thaw", "fault": {"kind": "no_progress", "node": "press"}})
    assert plan == planner.plan({"task": "kitchen_thaw"})


# ── robocasa lane: the real kitchen ─────────────────────────────────────────────

@pytest.mark.robocasa
def test_pack_lunch_seed_1_carry_stalls_early_and_a_recovery_runs(tmp_path):
    from board import store as bs

    runs = tmp_path / "runs"
    session = runs / "session-main"
    name = bs.submit_brief(runs, json.dumps({
        "kind": "task", "task": "pack_lunch", "seed": 1, "arm": "scripted",
        "max_replans": 3}))["submitted"]
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/harness_runtime.py"),
         "--session-dir", str(session), "--drain"],
        cwd=str(REPO), capture_output=True, text=True, timeout=2400, check=False,
        env={**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(REPO)})
    assert proc.returncode == 0 and (session / "done" / name).exists(), proc.stderr[-4000:]
    rows = bs.chain_rows(session)
    assert not [r for r in rows if r["kind"] == "runtime.task_error"], proc.stderr[-4000:]
    events = bs.read_runtime_events(session)["events"]
    ends = [e for e in events if e.get("kind") == "actuation_end" and e["node"].startswith("carry-")]
    assert ends, [e.get("node") for e in events if e.get("kind") == "actuation_end"]
    stalled = [e for e in ends if (e.get("diagnostics") or {}).get("failure_mode") == "nav_stall"]
    assert stalled, [(e["node"], e["steps"], e.get("diagnostics")) for e in ends]
    assert all(0 < e["steps"] < 700 for e in stalled), [(e["node"], e["steps"]) for e in stalled]
    executed = [r["data"]["node"] for r in rows if r["kind"] == "task.verify"]
    assert any(n.startswith("recover-") for n in executed), executed
