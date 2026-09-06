"""Fixed world-verification vectors and accumulating development candidates.

The integration fixture keeps task success at zero while an independent world
predicate improves. Historical campaign fixtures remain diagnostic evidence;
without their original frozen oracle contract they cannot qualify as new wins.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_evolve_e2e import _OBS, CATALOGUE, ORACLES, _Env, _Planner  # noqa: F401

from board import store as bs
from harness import protocol
from harness.fakes import _FakeEnvHandle
from harness.skill_library import segment_specs
from plugins.rsi import evaluation
from scripts import evolve

TASK = "e2e_score"
EMB = "test_evolve_score:env_provider"
POLICY = "test_evolve_score:policy_provider"
RECORDS = {"reach": {"id": "reach", "name": "reach", "kind": "segment", "args": {},
                     "bindings": {EMB: {"task": "reach"}}},
           "grab": {"id": "grab", "name": "grab", "kind": "segment", "args": {},
                    "bindings": {EMB: {"task": "grab"}}},
           "reached": {"id": "reached", "name": "reached", "kind": "verify", "args": {},
                       "bindings": {EMB: {}}}}
CATALOGUE = {**CATALOGUE, "reached": {}}
PREDICATES = {"reached": "test_evolve_score:reached"}
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
        self.achieved = set()
        return dict(_OBS)

    def step(self, action):
        self.t += 1
        if action[0] == 1.0:
            self.achieved.add("reach")
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
        return (self.command,)

    def enter_segment(self, env, spec, executor=None):
        self.n = 0
        RUNS.append((env.seed, _gate()))
        self.task = spec.task
        self.command = 1.0 if spec.task == "reach" and env.seed <= _gate() else 0.0

    def segment_success(self, env):
        return self.task in env.achieved


class _Policy:
    def make_driver(self, spec):
        return _Driver()


def env_provider():
    return _ScoreEnv()


def policy_provider(**params):
    return _Policy()


def reached():
    return lambda node, ctx: {"success": "reach" in ctx.episode.env.achieved}


class _ScorePlanner(_Planner):
    def plan(self, brief):
        plan = super().plan(brief)
        plan["nodes"].insert(1, {"id": "verify-reach", "skill": "reached", "kind": "verify",
                                 "args": {}, "after": ["reach-0"]})
        plan["nodes"][-1]["after"] = ["verify-reach"]
        return plan


def planner_provider():
    return _ScorePlanner()


_CARD = f"""
[task_bindings.{TASK}]
env = "{EMB}"
policy = "{POLICY}"
planner = "test_evolve_score:planner_provider"
catalogue = "test_evolve_score:CATALOGUE"
predicates = "test_evolve_score:PREDICATES"
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
    params_card = tmp_path / "plugins" / "score_params"
    params_card.mkdir()
    (params_card / "manifest.toml").write_text(
        f'enabled = false\n[mounts."policy.driver"]\nref = "{POLICY}"\n'
        'params = {reach_gate = 1}\n')
    monkeypatch.setenv("PH_PLUGINS_EXTRA", str(tmp_path / "plugins"))
    monkeypatch.setattr(evolve, "_BASE_EXTRA", str(tmp_path / "plugins"))
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    answers = []
    for answer in canned:
        answers.extend([{"op": "inspect", "args": {"view": "parameter",
                        "node": answer["payload"]["node"], "parameter": "reach_gate"}}, answer])
    (tmp_path / "canned.json").write_text(json.dumps(answers))
    monkeypatch.setenv("PH_MODEL_ENDPOINT_FAKE", str(tmp_path / "canned.json"))
    RUNS.clear()
    assert evolve.main(["--mode", "evolution", "--task", TASK, "--session", str(root),
                        "--skills-root", str(root / "skills"), "--seeds", "1", "3",
                        "--rounds", str(rounds), "--confirm-seeds", "0"]) == 0
    return root, json.loads((root / "campaigns" / f"evolve-{TASK}" / "campaign.json").read_text())


# The measured predicate has one fixed semantic identity. Scheduler trails below
# may vary independently; only these oracle observations are reward evidence.
_CHECK = {"id": "verified-reach", "kind": "verify", "skill": "reached", "args": {}}
_CONTRACT = evaluation.compile_contract({"nodes": [_CHECK]}, task=TASK,
                                        predicates=PREDICATES, terminal_ref=EMB)


def _verified(rows, *, trails=None, contract=None):
    contract = contract or _CONTRACT
    return {"count": 999, "sha": "display-count-is-untrusted", "seeds": {
        str(seed): {"success": True, "trail": (trails or {}).get(seed, []),
                    "evaluation": evaluation.evaluate(contract, [
                        {"node": _CHECK, "authority": "predicate", "source": PREDICATES["reached"],
                         "evidence_policy": "world-dependencies-v1", "blocked_reads": [],
                         "success": checkpoint}],
                        {"authority": "embodiment.terminal_success", "source": EMB,
                         "success": terminal})}
        for seed, (checkpoint, terminal) in rows.items()}}


def test_only_a_measured_condition_creates_partial_progress():
    before = _verified({1: (False, False)})
    after = _verified({1: (True, False)})
    result = evaluation.compare(before, after, _CONTRACT)
    assert result["accepted"] and len(result["gains"]) == 1
    assert result["before"]["progress"] == 0.0
    assert result["after"]["progress"] == 0.5
    assert result["after"]["successes"] == 0


def test_scheduler_success_and_extra_nodes_never_create_a_gain():
    before = _verified({1: (True, False)})
    after = _verified({1: (True, False)}, trails={1: [
        {"id": f"extra-{i}", "kind": "segment", "ok": True} for i in range(500)]})
    result = evaluation.compare(before, after, _CONTRACT)
    assert not result["accepted"] and result["gains"] == []
    assert result["before"] == result["after"]


# ── the loop, end to end on the fake endpoint ────────────────────────────────────

def test_a_partial_win_is_accepted_and_becomes_the_next_rounds_baseline(tmp_path, monkeypatch):
    session, doc = _run(tmp_path, monkeypatch, [_knob(3, "reach-0"), _knob(4, "grab-0")],
                        rounds=2)
    r1, r2 = doc["rounds"]
    # whole-task success never moves (0/3 both ways) -- the milestones do, so the round
    # is ACCEPTED without being PUBLISHED
    assert (r1["before"], r1["after"], r1["published"]) == (0, 0, False)
    assert (r1["before_score"], r1["after_score"]) == ([0, 1/6], [0, 0.5])
    assert (r1["outcome"], r1["accepted"]) == ("improved", True)
    assert "without regressions" in r1["accepted_reason"]
    assert r1["trial"] == {"scope": "full", "seeds": [1, 2, 3], "target_pass": 3}
    assert len(doc["accepted_stack"]) == 1
    assert doc["accepted_stack"][0]["detail"] | {"node": "reach-0", "to": 3} == doc["accepted_stack"][0]["detail"]
    assert doc["applied"]["tunables"] == {POLICY: {"reach_gate": 3}}
    # the NEXT round starts from it: every seed now dies at grab-0, not at reach-0
    assert [s["first_death"] for s in r2["per_seed"]] == ["grab-0"] * 3
    assert r2["before_score"] == [0, 0.5] and r2["parent"] == 1
    assert bs.rsi_campaigns(session)[0] | {"accepted_rounds": [1], "published_rounds": []} \
        == bs.rsi_campaigns(session)[0]


def test_a_full_paired_trial_that_loses_a_verified_condition_is_rejected(
        tmp_path, monkeypatch):
    session, doc = _run(tmp_path, monkeypatch, [_knob(0, "grab-0")])
    r = doc["rounds"][0]
    assert r["trial"] == {"scope": "full", "seeds": [1, 2, 3], "target_pass": 0}
    assert {seed for seed, gate in RUNS if gate == 0} == {1, 2, 3}
    assert r["usage"]["sim_s_saved"] == 0
    # seed 1 lost reach-0: worse, not accepted, nothing joins the stack
    assert (r["before_score"], r["after_score"]) == ([0, 1/6], [0, 0.0])
    assert (r["outcome"], r["accepted"], r["published"]) == ("worse", False, False)
    assert doc.get("accepted_stack") == [] and doc["applied"]["tunables"] == {}
    lost = doc["last_outcome"]["regressions"]
    assert len(lost) == 1 and lost[0]["seed"] == "1" and lost[0]["before"] is True and lost[0]["after"] is None
    assert doc["last_outcome"]["outcome"] == "worse"
    assert bs.rsi_campaigns(session)[0]["accepted_rounds"] == []


@pytest.mark.parametrize("repair", [[], [{"id": "recover-drop", "kind": "recovery", "ok": True}],
                                      [{"id": "fix-drop", "kind": "recovery", "ok": False}]])
def test_recovery_presence_names_and_self_reported_status_do_not_change_reward(repair):
    before = _verified({1: (True, False)}, trails={1: [
        {"id": "recover-drop", "kind": "recovery", "ok": True}]})
    after = _verified({1: (True, False)}, trails={1: repair})
    result = evaluation.compare(before, after, _CONTRACT)
    assert not result["accepted"] and result["regressions"] == result["gains"] == []
    assert result["before"] == result["after"]


def test_an_unobserved_previously_true_condition_is_a_regression_even_when_terminal_improves():
    before = _verified({1: (True, False)})
    after = _verified({1: (None, True)}, trails={1: [{"id": "drop", "ok": True}]})
    result = evaluation.compare(before, after, _CONTRACT)
    assert not result["accepted"] and len(result["gains"]) == len(result["regressions"]) == 1
    assert result["regressions"][0]["after"] is None


def test_a_candidate_cannot_skip_an_unfavourable_seed():
    before = _verified({1: (True, False), 2: (True, True)})
    after = _verified({1: (True, True)})
    result = evaluation.compare(before, after, _CONTRACT)
    assert not result["accepted"] and "same nonempty seed set" in result["reason"]


def test_duplicate_or_renamed_checkpoint_nodes_do_not_inflate_the_contract():
    duplicate = evaluation.compile_contract({"nodes": [{**_CHECK, "id": f"check-{i}"}
                                                        for i in range(100)]},
                                             task=TASK, predicates=PREDICATES, terminal_ref=EMB)
    assert duplicate["sha"] == _CONTRACT["sha"] and len(duplicate["obligations"]) == 2
    result = evaluation.compare(_verified({1: (True, False)}),
                                _verified({1: (True, False)}, contract=duplicate), _CONTRACT)
    assert not result["accepted"] and result["after"]["progress"] == 0.5


def test_missing_terminal_is_never_reconstructed_from_a_successful_report_node():
    after = _verified({1: (True, None)}, trails={1: [
        {"id": "report", "kind": "decide", "ok": True}]})
    result = evaluation.summary(after, _CONTRACT)
    assert result["successes"] == 0 and result["observed"] == 1


def _rounds() -> dict:
    return json.loads((Path(__file__).parent / "fixtures"
                       / "evolve_recycle_rounds.json").read_text())


def _historical_suite(rows):
    return {"seeds": {str(row["seed"]): {"success": row["success"], "trail": row["nodes"]}
                      for row in rows}}


@pytest.mark.parametrize("round_no", ["131", "389", "556"])
def test_historical_node_trails_are_preserved_but_cannot_be_promoted_under_a_new_ruler(round_no):
    row = _rounds()[round_no]
    expected_reason = ("focused trial: nav-can1 passed on no seed of its cluster" if round_no == "556"
                       else "regressed: 4243/recover-drop-can1")
    assert row["was"] == {"accepted": False, "reason": expected_reason}
    assert row["per_seed"] and row["after_seeds"]
    assert all("nodes" in seed and "evaluation" not in seed for seed in row["after_seeds"])
    result = evaluation.compare(_historical_suite(row["per_seed"]),
                                _historical_suite(row["after_seeds"]), _CONTRACT)
    assert not result["accepted"] and "lacks observations" in result["reason"]


def test_a_historical_focused_trial_is_not_relabelled_as_full_paired_evidence():
    row = _rounds()["556"]
    assert row["trial"]["scope"] == "focused" and row["trial"]["seeds"] == [4244]
    result = evaluation.compare(_verified({4243: (True, False), 4244: (False, False)}),
                                _verified({4244: (True, False)}), _CONTRACT)
    assert not result["accepted"] and "same nonempty seed set" in result["reason"]
