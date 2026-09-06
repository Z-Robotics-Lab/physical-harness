"""Model failures stop RSI explicitly; they never select a substitute experiment."""

from __future__ import annotations

import json

import pytest
from test_evolve_e2e import _CARD, LLM_NONE, TASK
from test_mission_e2e import _kinds, _Runtime

from scripts import evolve


@pytest.mark.parametrize("failure", ["load", "chat"])
def test_endpoint_failure_seals_an_error_without_trial_fallback_or_memory(tmp_path, failure):
    # A missing fake endpoint tests loading; an empty reply sequence fails in chat.
    env = {"PH_MODEL_ENDPOINT_FAKE": str(tmp_path / "missing.json")} if failure == "load" else {}
    rt = _Runtime(tmp_path, card=_CARD, canned=[], env=env, mode="evolution")
    try:
        _, events = rt.run({"kind": "evolve", "task": TASK, "seeds": [1, 2],
                            "rounds": 1, "confirm_seeds": 0}, expect="failed")
        campaign = json.loads((rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json").read_text())
        assert campaign["status"] == campaign["live"]["phase"] == "failed"
        assert len(campaign["rounds"]) == 1
        row = campaign["rounds"][0]
        assert row["proposer"] == "llm" and row["outcome"] == "error"
        assert row["llm"]["status"] == "error" and row["llm"]["reason"]
        assert set(row["llm"]["error"]) >= {"type", "message", "stage"}
        assert row["needs"] == ["model_endpoint"] and row["tried"]["kind"] == "none"
        assert row["trial"] is None and row["after_seeds"] == []
        assert all(row[key] is None for key in ("after", "after_score", "suite_sha"))
        assert row["experiments"]["after"] is None and row["evaluation"]["after"] is None
        assert row["accepted"] is False and row["published"] is False
        assert campaign["applied"] == {"executors": {}, "tunables": {}}
        assert row["experience"]["recorded"] is None
        assert not (tmp_path / "rsi-experience.json").exists()
        assert all("/baseline/" in path for path in row["media"])
        assert len(_kinds(events, "rsi_step")) == 1
        assert _kinds(events, "runtime.task_error")
    finally:
        rt.stop()


def test_invalid_model_answers_are_rejected_without_becoming_rule_trials(tmp_path):
    rt = _Runtime(tmp_path, card=_CARD, canned="not a JSON response", mode="evolution")
    try:
        rt.run({"kind": "evolve", "task": TASK, "seeds": [1, 2],
                "rounds": 1, "confirm_seeds": 0})
        campaign = json.loads((rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json").read_text())
        assert campaign["status"] == "done"
        assert len(campaign["rounds"]) == 1
        for row in campaign["rounds"]:
            assert row["llm"]["status"] == "abstained" and row["proposer"] == "llm"
            assert row["llm"]["stop_reason"] == "budget_exhausted"
            assert row["outcome"] == "none" and row["trial"] is None
            assert row["after"] is None and row["after_seeds"] == []
            assert row["experience"]["recorded"] is None
        assert campaign["applied"] == {"executors": {}, "tunables": {}}
    finally:
        rt.stop()


def test_rules_is_refused_at_cli_and_runtime_entry_points(tmp_path):
    with pytest.raises(SystemExit) as error:
        evolve.main(["--proposer", "rules"])
    assert error.value.code == 2
    rt = _Runtime(tmp_path, card=_CARD, canned=LLM_NONE, mode="evolution")
    try:
        _, events = rt.run({"kind": "evolve", "task": TASK, "proposer": "rules"}, expect="failed")
        errors = _kinds(events, "runtime.task_error")
        assert errors and "proposer" in errors[-1]["error"]
        assert not _kinds(events, "rsi_step")
        assert not (rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json").exists()
    finally:
        rt.stop()


def test_parameter_preflight_feedback_repairs_nonfinite_and_wrong_prefix_before_a_real_trial(tmp_path):
    from test_evolve_llm_e2e import _params_card

    _params_card(tmp_path)
    def answer(path, value):
        return {"kind": "tunables", "payload": {"node": "grab-0",
                "ref": "test_evolve_e2e:policy_provider", "path": path, "to": value},
                "summary": "Test the declared parameter.", "rationale": "Compare paired outcomes."}

    replies = [{"op": "inspect", "args": {"view": "parameter", "node": "grab-0", "parameter": "stall_k"}},
               answer(["tunables", "stall_k"], float("inf")),
               answer(["wrong_prefix", "stall_k"], 28),
               answer(["tunables", "stall_k"], 27)]
    rt = _Runtime(tmp_path, card=_CARD, canned=replies, mode="evolution")
    try:
        rt.run({"kind": "evolve", "task": TASK, "seeds": [1, 2],
                "rounds": 1, "confirm_seeds": 0})
        root = rt.session / "campaigns" / f"evolve-{TASK}"
        row, = json.loads((root / "campaign.json").read_text())["rounds"]
        audit = json.loads((root / "llm" / "round-1.json").read_text())
        assert row["llm"]["status"] == "proposed" and row["proposer"] == "llm"
        assert row["tried"]["detail"]["to"] == 27
        assert row["tried"]["detail"]["path"] == ["tunables", "stall_k"]
        assert row["after"] == 0 and len(row["after_seeds"]) == 2
        assert row["trial"] is not None and row["suite_sha"]
        assert audit["calls"] == 4 and len(audit["attempts"]) == 2
        for attempt in audit["attempts"]:
            assert "finite installed driver parameter" in attempt["reason"]
            assert any(attempt["reason"] in json.dumps(request["messages"], ensure_ascii=False)
                       for request in audit["requests"])
    finally:
        rt.stop()
