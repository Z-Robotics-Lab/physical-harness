"""Cross-task memory stays evidence-backed; transfer counts retain censoring."""

import copy
import json

import pytest

from plugins.rsi.experience import (
    development_report,
    intervention_strategy,
    read_experiences,
    record_experience,
    retrieve_experiences,
    transfer_report,
)

DIAGNOSIS = {"fingerprint": ["translation:commanded_without_observed_motion"]}


def test_rule_parameter_strategy_keeps_relative_change_and_record_reference(tmp_path):
    reference = "session-robocasa-rsi/recycle_cans/round-589"
    strategy = intervention_strategy({"kind": "tunables", "detail": {
        "path": ["tunables", "drop_standoff"], "from": 0.65, "to": 0.455}},
        summary="set the old task's value to 0.455", reference=reference)
    assert strategy["kind"] == "tunables" and strategy["scope"] == "parameter"
    assert "tunables.drop_standoff" in strategy["summary"] and "-30%" in strategy["summary"]
    assert "0.65" not in strategy["summary"] and "0.455" not in strategy["summary"]
    assert "Reported rationale: set the old task's value to <task-specific value>" in strategy["summary"]
    assert strategy["reference"] == reference
    path = tmp_path / "experience.json"
    assert record_experience(path, task="old-task", diagnosis=DIAGNOSIS,
                             intervention=strategy, accepted=False,
                             evidence={"round": 589}) is None
    assert not path.exists()  # Describing a strategy is insufficient to record it.
    record_experience(path, task="old-task", diagnosis=DIAGNOSIS,
                      intervention=strategy, accepted=False,
                      evidence={"round": 589, "before_sha": "before", "after_sha": "after"})
    retrieved = retrieve_experiences(path, task="new-task", diagnosis=DIAGNOSIS)
    assert retrieved[0]["intervention"] == strategy and retrieved[0]["accepted"] is False


@pytest.mark.parametrize(("before", "after", "description"), [
    (0, 2, "increased from zero"), (0, -2, "decreased from zero"),
    (-2, -1, "+50%"), (None, 2, "relative change unavailable"),
    (float("nan"), 2, "relative change unavailable"), (2, 2, "unchanged"),
])
def test_parameter_strategy_does_not_invent_a_relative_scale(before, after, description):
    strategy = intervention_strategy({"kind": "tunables", "detail": {
        "path": ["gain"], "from": before, "to": after}})
    assert description in strategy["summary"]


def test_executor_and_card_strategies_explain_the_intervention_without_copying_source():
    executor = intervention_strategy({"kind": "executor", "detail": {"to": "old-task-policy"}})
    assert "another installed executor" in executor["summary"]
    assert "old-task-policy" not in executor["summary"]
    tried = {"kind": "card", "detail": {"layer": "recovery", "source": "secret source"}}
    card = intervention_strategy(tried, summary="Probe another response direction", reference="audit.json")
    assert card == {"kind": "card", "scope": "recovery",
                    "summary": "Probe another response direction", "reference": "audit.json"}
    assert "recovery" in intervention_strategy(tried)["summary"]
    with pytest.raises(TypeError, match="reference must be a string"):
        intervention_strategy(tried, reference={"round": 589})


def test_plan_strategy_summarizes_only_inserted_call_types_and_counts():
    original = {"id": "place", "kind": "segment", "skill": "place", "args": {"target": "private"}}
    added = {"id": "approach", "kind": "segment", "skill": "move", "args": {"target": "private"}}
    tried = {"kind": "plan", "detail": {"graph": {"nodes": [added, original]},
                                        "reference_graph": {"nodes": [original]}}}
    strategy = intervention_strategy(tried)
    assert set(strategy) == {"kind", "scope", "summary", "reference"}
    assert "Insert 1 installed action calls" in strategy["summary"]
    assert "segment/move x1" in strategy["summary"]
    assert "place" not in strategy["summary"] and "private" not in strategy["summary"]
    del tried["detail"]["reference_graph"]
    assert "unknown" in intervention_strategy(tried)["summary"]


def remember(path, task="training-a", accepted=True, n=1, **extra):
    return record_experience(path, task=task, diagnosis=DIAGNOSIS,
                             intervention={"kind": "patch", "scope": "upstream", "summary": "change",
                                           "source": "do not copy arbitrary source code"},
                             accepted=accepted,
                             evidence={"round": n, "before_sha": f"before-{n}",
                                       "after_sha": f"after-{n}"}, **extra)


def test_memory_uses_structural_evidence_excludes_same_task_and_never_copies_code(tmp_path):
    path = tmp_path / "experience.json"
    good = remember(path)
    remember(path, task="training-b", accepted=False, n=2)
    remember(path, task="unseen", n=3)
    retrieved = retrieve_experiences(path, task="unseen", diagnosis=DIAGNOSIS)
    assert [r["task"] for r in retrieved] == ["training-a", "training-b"]
    assert [r["accepted"] for r in retrieved] == [True, False]
    assert retrieved[0]["evidence"]["before_sha"] == "before-1"
    assert "source" not in retrieved[0]["intervention"]
    assert remember(path) == good and len(read_experiences(path)) == 3
    assert retrieve_experiences(path, task="unseen", diagnosis={"fingerprint": []}) == []


def test_training_prefix_and_bounded_store_prevent_future_task_leakage(tmp_path):
    path = tmp_path / "experience.json"
    for n in range(1, 5):
        remember(path, task=f"task-{n}", n=n, max_records=3)
    assert len(read_experiences(path)) == 3
    found = retrieve_experiences(path, task="held-out", diagnosis=DIAGNOSIS, before_sequence=3)
    assert [r["task"] for r in found] == ["task-2"]


def test_protocol_filters_exclude_legacy_evidence_without_changing_the_memory_prefix(tmp_path):
    path = tmp_path / "experience.json"
    current = {"protocol_id": "fixed-verification-v2", "evidence_policy": "world-dependencies-v1"}
    identities = {
        "legacy": {},
        "v1": {**current, "protocol_id": "fixed-verification-v1"},
        "v2-missing-policy": {"protocol_id": "fixed-verification-v2"},
        "v2-wrong-policy": {**current, "evidence_policy": "node-only"},
        "v2-current": current,
        "v2-after-prefix": current,
    }
    for n, (task, identity) in enumerate(identities.items(), 1):
        recorded = record_experience(path, task=task, diagnosis=DIAGNOSIS,
                                     intervention={"kind": "patch", "scope": "controller"},
                                     accepted=task != "v2-after-prefix",
                                     evidence={"round": n, "before_sha": f"before-{n}",
                                               "after_sha": f"after-{n}", **identity})
        assert all(recorded["evidence"][key] == value for key, value in identity.items())
    before = path.read_bytes()

    def tasks(**filters):
        return {r["task"] for r in retrieve_experiences(
            path, task="held-out", diagnosis=DIAGNOSIS, limit=20, **filters)}

    assert tasks() == set(identities)  # Unfiltered API remains compatible.
    assert tasks(protocol_id=current["protocol_id"]) == {
        "v2-missing-policy", "v2-wrong-policy", "v2-current", "v2-after-prefix"}
    assert tasks(evidence_policy=current["evidence_policy"]) == {"v1", "v2-current", "v2-after-prefix"}
    assert tasks(**current) == {"v2-current", "v2-after-prefix"}
    assert tasks(**current, before_sequence=6) == {"v2-current"}
    assert tasks(**current, before_sequence=5) == set()
    assert retrieve_experiences(path, task="v2-current", diagnosis=DIAGNOSIS,
                                before_sequence=6, **current) == []
    rows = read_experiences(path)
    assert [r["sequence"] for r in rows] == list(range(1, 7))
    assert "protocol_id" not in rows[0]["evidence"] and "evidence_policy" not in rows[0]["evidence"]
    assert path.read_bytes() == before  # Retrieval never relabels historical records.


def test_notes_or_unmeasured_outcomes_cannot_become_experience(tmp_path):
    path = tmp_path / "experience.json"
    for evidence in ({"round": 1}, {"round": 1, "before_sha": "same", "after_sha": "same"}):
        assert record_experience(path, task="a", diagnosis=DIAGNOSIS,
                                 intervention={"kind": "patch"}, accepted=True, evidence=evidence) is None
    assert not path.exists()


def test_transfer_evaluator_measures_unseen_tasks_with_cold_and_warm_search(tmp_path):
    """Synthetic protocol validation, NOT empirical evidence of robot scaling.

    One training task discovers the upstream intervention. A new task has the
    same diagnostic structure: warm orders that strategy first; cold needs 3
    evaluations. A second unseen task never improves and remains censored.
    """
    path = tmp_path / "experience.json"
    remember(path)
    warm = retrieve_experiences(path, task="held-out-a", diagnosis=DIAGNOSIS,
                                before_sequence=2)[0]["intervention"]["scope"]
    rows = []
    for task in ("held-out-a", "held-out-b"):
        for condition, strategies in (("cold", ["parameter", "local", "upstream"]),
                                      ("warm", [warm, "parameter", "local"])):
            for n, strategy in enumerate(strategies, 1):
                accepted = task == "held-out-a" and strategy == "upstream"
                rows.append({"task": task, "condition": condition, "trial": n,
                             "accepted": accepted, "training_tasks": 1})
                if accepted:
                    break
    report = transfer_report(rows)
    assert report["paired_tasks"] == 2 and report["censored_runs"] == 2
    measured, censored = report["pairs"]
    assert measured["cold"]["first_accepted_trial"] == 3
    assert measured["warm"]["first_accepted_trial"] == 1 and measured["trials_saved"] == 2
    assert censored["trials_saved"] is None and censored["warm"]["censored"]


def test_transfer_report_rejects_missing_or_duplicate_trials():
    row = {"task": "a", "condition": "cold", "trial": 1, "accepted": False}
    for rows in ([row, row], [{**row, "trial": 2}], [{**row, "accepted": None}]):
        with pytest.raises(ValueError):
            transfer_report(rows)


def _development_row(round_no, *, accepted=False, multiplier=1):
    return {"round": round_no, "evaluation": {"acceptance": {"accepted": accepted}},
            "llm": {"usage_complete": True},
            "usage": {"episode_attempts": 3 * multiplier, "model_calls": 2 * multiplier,
                      "input_bytes": 100 * multiplier, "sim_s": 1.5 * multiplier,
                      "wall_s": 4.5 * multiplier,
                      "llm_tokens": {"prompt": 10 * multiplier, "completion": 4 * multiplier}},
            "run_budget": {"used": {"model_calls": 99999, "input_bytes": 99999999}}}


def _expected_cost(multiplier):
    return {"episode_attempts": 3 * multiplier, "model_calls": 2 * multiplier,
            "input_bytes": 100 * multiplier, "sim_s": 1.5 * multiplier,
            "wall_s": 4.5 * multiplier, "llm_tokens": 14 * multiplier}


def test_development_report_counts_two_acceptances_and_the_complete_cost_of_each_interval():
    rows = [_development_row(10), _development_row(11, accepted=True, multiplier=2),
            _development_row(12, multiplier=3), _development_row(13, accepted=True, multiplier=4)]
    rows[2]["outcome"] = "error"  # Failed work still consumed its recorded costs.
    before = copy.deepcopy(rows)
    report = development_report(list(reversed(rows)), epoch_start=10)
    assert report == {"first_accepted_round": 2, "total_trials": 4, "censored": False,
                      "accepted_updates": 2, "cost": {"total": _expected_cost(10),
                      "first_accepted": _expected_cost(3), "since_previous_acceptance": _expected_cost(7)}}
    assert rows == before
    resumed = json.loads(json.dumps(rows)) + [_development_row(14, multiplier=5)]
    report = development_report(resumed, epoch_start=10)
    assert report["accepted_updates"] == 2 and report["total_trials"] == 5
    assert report["cost"]["total"] == _expected_cost(15)
    assert report["cost"]["first_accepted"] == _expected_cost(3)
    assert report["cost"]["since_previous_acceptance"] == _expected_cost(5)
    assert development_report(rows[:2], epoch_start=10)["cost"]["since_previous_acceptance"] == _expected_cost(3)


def test_development_report_filters_epochs_and_reports_censoring_without_inventing_costs():
    legacy = {"round": 8, "accepted": True, "published": True}
    rows = [legacy, _development_row(9), _development_row(10)]
    report = development_report(rows, epoch_start=9)
    assert report == {"first_accepted_round": None, "total_trials": 2, "censored": True,
                      "accepted_updates": 0, "cost": {"total": _expected_cost(2),
                      "first_accepted": None, "since_previous_acceptance": _expected_cost(2)}}
    empty = development_report(rows, epoch_start=11)
    assert empty == {"first_accepted_round": None, "total_trials": 0, "censored": True,
                     "accepted_updates": 0, "cost": {"total": _expected_cost(0),
                     "first_accepted": None, "since_previous_acceptance": _expected_cost(0)}}
    rows.append(_development_row(12, accepted=True))
    new = development_report(rows, epoch_start=12)
    assert new["first_accepted_round"] == 1 and new["accepted_updates"] == new["total_trials"] == 1
    assert new["cost"]["total"] == new["cost"]["first_accepted"] == _expected_cost(1)


def test_development_report_uses_strict_evaluation_acceptance_with_legacy_fallback():
    rows = [_development_row(1), _development_row(2), _development_row(3)]
    rows[0].update(accepted=True, published=True)  # Modern false overrides both.
    rows[1]["evaluation"]["acceptance"]["accepted"] = "true"
    rows[1]["accepted"] = True  # A malformed modern value is not acceptance.
    rows[2].pop("evaluation")
    rows[2]["published"] = True  # Publication alone supplies no acceptance.
    assert development_report(rows, epoch_start=1)["censored"] is True
    rows.append({"round": 4, "accepted": True})
    report = development_report(rows, epoch_start=1)
    assert report["first_accepted_round"] == 4 and report["accepted_updates"] == 1
    assert all(value is None for value in report["cost"]["total"].values())
    assert all(value is None for value in report["cost"]["first_accepted"].values())


@pytest.mark.parametrize("bad", [None, -1, float("inf"), float("nan"), True, "3"])
def test_development_report_preserves_unknown_costs_per_field(bad):
    first, second = _development_row(1, accepted=True), _development_row(2)
    second["usage"]["sim_s"] = bad
    second["usage"]["llm_tokens"]["completion"] = bad
    second["usage"].pop("input_bytes")
    report = development_report([first, second], epoch_start=1)
    expected = {**_expected_cost(2), "sim_s": None, "input_bytes": None, "llm_tokens": None}
    assert report["cost"]["total"] == expected
    assert report["cost"]["first_accepted"] == _expected_cost(1)
    assert report["cost"]["since_previous_acceptance"] == {
        **_expected_cost(1), "sim_s": None, "input_bytes": None, "llm_tokens": None}


def test_development_report_does_not_recover_tokens_from_partial_or_cumulative_usage():
    row = _development_row(1)
    row["llm"]["usage_complete"] = False
    assert development_report([row], epoch_start=1)["cost"]["total"]["llm_tokens"] is None
    row.pop("usage")
    assert all(value is None for value in development_report([row], epoch_start=1)["cost"]["total"].values())


def test_development_report_overflowing_cost_sum_is_unknown_and_zero_is_measured():
    rows = [_development_row(1, multiplier=0), _development_row(2, multiplier=0)]
    assert development_report(rows, epoch_start=1)["cost"]["total"] == _expected_cost(0)
    for row in rows:
        row["usage"]["wall_s"] = 1e308
    assert development_report(rows, epoch_start=1)["cost"]["total"]["wall_s"] is None
