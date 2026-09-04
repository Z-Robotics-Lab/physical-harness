"""The gradient and the accumulating accepted state (scripts/evolve.py): the lexicographic
``score`` tuple, node-focused trials that stop before the full suite, and a partial win that
joins ``accepted_stack`` and becomes the next round's baseline. Stdlib fakes, fake endpoint.

The fixture is the production shape: whole-task success is 0 on EVERY candidate (``grab``
never passes), so the only gradient is how far the seeds get. ``reach`` passes for the seeds
at or below the overlay knob ``reach_gate`` (baseline 1: seed 1 alone gets past it).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from test_evolve_e2e import _OBS, _Env, _Planner, CATALOGUE, ORACLES  # noqa: F401

from board import store as bs
from harness import protocol
from harness.fakes import _FakeEnvHandle
from harness.skill_library import segment_specs
from scripts import evolve

TASK = "e2e_score"
EMB = "test_evolve_score:env_provider"
POLICY = "test_evolve_score:policy_provider"
RECORDS = {"reach": {"id": "reach", "name": "reach", "kind": "segment", "args": {},
                     "bindings": {EMB: {"task": "reach"}}},
           "grab": {"id": "grab", "name": "grab", "kind": "segment", "args": {},
                    "bindings": {EMB: {"task": "grab"}}}}
SEGMENT_SPECS = segment_specs(
    {k: protocol.SkillRecordV0.from_dict(v) for k, v in RECORDS.items()}, EMB)
EPISODE = {"task": "reach", "horizon": 40}
#: (seed, reach_gate) of every segment episode -- what a focused trial did NOT spend.
RUNS: list[tuple[int, int]] = []


def _gate() -> int:
    """The knob an evolve tunables trial writes, read off the same overlay env var
    ``harness.manifest.mount_params`` merges (the tmp card is outside ``plugins/``)."""
    over = json.loads(os.environ.get(evolve.OVERRIDE_ENV) or "{}")
    return int((over.get(POLICY) or {}).get("reach_gate", 1))


class _SeedHandle(_FakeEnvHandle):
    def __init__(self, seed):
        self.seed = seed

    def reset(self):
        super().reset()
        return dict(_OBS)

    def step(self, action):
        self.t += 1
        return dict(_OBS), 0.0, False, {}


class _ScoreEnv(_Env):
    def make_env(self, spec):
        return _SeedHandle(spec.seed)


class _Driver:
    STEPS = 4
    n = 0

    @property
    def exhausted(self):
        return self.n >= self.STEPS

    def observe_once(self, obs):
        pass

    def on_handback(self):
        pass

    def act(self, obs):
        self.n += 1
        return (0.0,)

    def enter_segment(self, env, spec, executor=None):
        self.n = 0
        RUNS.append((env.seed, _gate()))
        self.ok = spec.task == "reach" and env.seed <= _gate()

    def segment_success(self, env):
        return self.ok


class _Policy:
    def make_driver(self, spec):
        return _Driver()


def env_provider():
    return _ScoreEnv()


def policy_provider(**params):
    return _Policy()


_CARD = f"""
[task_bindings.{TASK}]
env = "{EMB}"
policy = "{POLICY}"
planner = "test_evolve_e2e:planner_provider"
catalogue = "test_evolve_e2e:CATALOGUE"
records = "test_evolve_score:RECORDS"
oracles = "test_evolve_e2e:ORACLES"
episodic = true
episode = "test_evolve_score:EPISODE"
segment_specs = "test_evolve_score:SEGMENT_SPECS"
max_replans = 1
"""


def _knob(to, node, summary="调闸"):
    return {"kind": "tunables", "summary": summary, "rationale": "-",
            "payload": {"ref": POLICY, "path": ["reach_gate"], "to": to, "node": node}}


def _run(tmp_path, monkeypatch, canned, rounds=1) -> tuple[Path, dict]:
    root = tmp_path / "s"
    (tmp_path / "plugins" / "score").mkdir(parents=True)
    (tmp_path / "plugins" / "score" / "manifest.toml").write_text(_CARD)
    monkeypatch.setenv("PH_PLUGINS_EXTRA", str(tmp_path / "plugins"))
    monkeypatch.setattr(evolve, "_BASE_EXTRA", str(tmp_path / "plugins"))
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    (tmp_path / "canned.json").write_text(json.dumps(canned))
    monkeypatch.setenv("PH_MODEL_ENDPOINT_FAKE", str(tmp_path / "canned.json"))
    RUNS.clear()
    assert evolve.main(["--mode", "evolution", "--task", TASK, "--session", str(root),
                        "--skills-root", str(root / "skills"), "--seeds", "1", "3",
                        "--rounds", str(rounds), "--confirm-seeds", "0"]) == 0
    return root, json.loads((root / "campaigns" / f"evolve-{TASK}" / "campaign.json").read_text())


# ── the score tuple, in the small ─────────────────────────────────────────────────

def _suite(rows: dict) -> dict:
    """A run_suite-shaped result from ``{seed: [(node, ok), ...]}``."""
    done = lambda ns: all(ok for _, ok in ns)
    return {"count": sum(done(ns) for ns in rows.values()), "sha": "-",
            "seeds": {str(s): {"success": done(ns), "nodes": {},
                               "first_death": next((n for n, ok in ns if not ok), None),
                               "trail": [{"id": n, "ok": ok} for n, ok in ns]}
                      for s, ns in rows.items()}}


def test_score_orders_by_successes_then_milestones_then_the_target_node():
    reached = _suite({1: [("nav", True), ("drop", False)]})
    earlier = _suite({1: [("nav", False), ("drop", False)]})
    won = _suite({1: [("nav", True), ("drop", True)]})
    assert evolve.score(reached, "drop") == (0, 1, 0)
    # the death moved EARLIER: worse, though the success count is 0 -> 0 either way
    assert evolve.score(earlier, "drop") == (0, 0, 0) < evolve.score(reached, "drop")
    assert evolve.score(won, "drop") == (1, 2, 1) > evolve.score(reached, "drop")
    assert evolve.regressions(reached, earlier) == [{"seed": 1, "node": "nav",
                                                     "was_ok_now_not": True}]
    assert evolve.regressions(reached, won) == []


def test_focus_seeds_is_the_target_nodes_cluster_inside_the_dev_range():
    before = _suite({1: [("nav", True), ("drop", False)],
                     2: [("nav", False), ("drop", False)],
                     3: [("nav", False), ("drop", False)]})
    assert evolve.focus_seeds([], before, "nav", [1, 3]) == [2, 3]
    assert evolve.focus_seeds([], before, "drop", [1, 3]) == [1]
    assert evolve.focus_seeds([], before, None, [1, 3]) == []   # nothing to narrow to


# ── the loop, end to end on the fake endpoint ────────────────────────────────────

def test_a_partial_win_is_accepted_and_becomes_the_next_rounds_baseline(tmp_path, monkeypatch):
    session, doc = _run(tmp_path, monkeypatch, [_knob(3, "reach-0"), _knob(4, "grab-0")],
                        rounds=2)
    r1, r2 = doc["rounds"]
    # whole-task success never moves (0/3 both ways) -- the milestones do, so the round
    # is ACCEPTED without being PUBLISHED
    assert (r1["before"], r1["after"], r1["published"]) == (0, 0, False)
    assert (r1["before_score"], r1["after_score"]) == ([0, 1, 1], [0, 3, 3])
    assert (r1["outcome"], r1["accepted"]) == ("improved", True)
    assert "no node regressed" in r1["accepted_reason"]
    assert r1["trial"] == {"scope": "full", "seeds": [1, 2, 3], "target_pass": 3}
    assert doc["accepted_stack"] == [
        {"round": 1, "kind": "tunables", "score": [0, 3, 3],
         "detail": {"node": "reach-0", "skill": "reach", "ref": POLICY,
                    "path": ["reach_gate"], "from": None, "to": 3}}]
    assert doc["applied"]["tunables"] == {POLICY: {"reach_gate": 3}}
    # the NEXT round starts from it: every seed now dies at grab-0, not at reach-0
    assert [s["first_death"] for s in r2["per_seed"]] == ["grab-0"] * 3
    assert r2["before_score"] == [0, 3, 0] and r2["parent"] == 1
    assert bs.rsi_campaigns(session)[0] | {"accepted_rounds": [1], "published_rounds": []} \
        == bs.rsi_campaigns(session)[0]


def test_a_focused_trial_that_moves_a_death_earlier_is_worse_and_spends_no_full_suite(
        tmp_path, monkeypatch):
    session, doc = _run(tmp_path, monkeypatch, [_knob(0, "grab-0")])
    r = doc["rounds"][0]
    # grab-0's cluster is seed 1 alone: the trial runs there, does not make grab-0 pass,
    # and the other two seeds are never spent
    assert r["trial"] == {"scope": "focused", "seeds": [1], "target_pass": 0}
    assert {seed for seed, gate in RUNS if gate == 0} == {1}
    assert r["usage"]["sim_s_saved"] > 0
    # seed 1 lost reach-0: worse, not accepted, nothing joins the stack
    assert (r["before_score"], r["after_score"]) == ([0, 1, 0], [0, 0, 0])
    assert (r["outcome"], r["accepted"], r["published"]) == ("worse", False, False)
    assert doc.get("accepted_stack") is None and doc["applied"]["tunables"] == {}
    assert doc["last_outcome"]["regressions"] == [{"seed": 1, "node": "reach-0",
                                                   "was_ok_now_not": True}]
    assert doc["last_outcome"]["summary"] == "调闸" and doc["last_outcome"]["outcome"] == "worse"
    assert bs.rsi_campaigns(session)[0]["accepted_rounds"] == []
