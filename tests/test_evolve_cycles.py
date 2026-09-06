"""Continuous scheduling renews cycle allowances, never cumulative brief costs."""

import copy
import json
from types import SimpleNamespace

import pytest

from scripts import evolve
from test_evolve_agent_runtime import _Model, _run


@pytest.fixture(autouse=True)
def no_scheduler_sleep(monkeypatch):
    # Scope the clock to the scheduler; fake environment resets keep their own timing.
    clock = SimpleNamespace(**vars(evolve.time))
    clock.sleep = lambda seconds: pytest.fail("cycles must not sleep")
    monkeypatch.setattr(evolve, "time", clock)


class CycleModel(_Model):
    def __init__(self, actions):
        super().__init__()
        self.actions = iter(actions)

    def chat(self, messages, **options):
        self.requests.append(copy.deepcopy(messages))
        return json.dumps(next(self.actions))


def stop(memo=""):
    return {"op": "stop", "args": {"reason": "No additional candidate this cycle."}, "memo": memo}


def test_continuous_abstention_retains_context_then_samples_without_rerunning_baseline(tmp_path, monkeypatch):
    model = CycleModel([stop("Consider the recorded neutral baseline."), _Model._trial("first"), stop()])
    _, doc = _run(tmp_path, monkeypatch, model, rounds=2, max_calls=2, continuous=True)
    first, second = doc["rounds"]
    assert first["cycle_outcome"] == "abstained" and second["cycle_outcome"] == "no_update"
    assert first["usage"]["episode_attempts"] == 2
    assert second["usage"]["episode_attempts"] == 1
    assert first["policy"]["active_id"] == second["policy"]["before_id"]
    assert [r["cycle_budget"]["used"]["model_calls"] for r in doc["rounds"]] == [1, 2]
    assert [r["run_budget"]["used"]["model_calls"] for r in doc["rounds"]] == [1, 3]
    assert doc["run_budget"]["limits"]["model_calls"] is None
    assert doc["run_budget"]["limits"]["input_bytes"] is None
    assert doc["run_budget"]["limits"]["probe_episodes"] is None
    assert all(r["cycle_budget"]["limits"]["model_calls"] == 2 for r in doc["rounds"])
    context = json.loads(model.requests[1][1]["content"])["state"]["cycle_context"]
    assert context["previous_round"] == first["round"]
    assert context["stop_reason"] == "model_stop"
    assert context["cycle_outcome"] == "abstained"
    assert first["memo"] == "Consider the recorded neutral baseline."
    assert doc["continuous"] is True and doc["stop_reason"] == "round_limit"


def test_each_continuous_cycle_gets_probe_allowance_and_total_never_resets(tmp_path, monkeypatch):
    model = CycleModel([_Model._trial("first"), stop(), _Model._trial("second"), stop()])
    _, doc = _run(tmp_path, monkeypatch, model, continuous=True, rounds=2,
                  max_calls=2, max_probe_episodes=1)
    assert [len(r["learning"]["probes"]) for r in doc["rounds"]] == [1, 1]
    assert [r["cycle_budget"]["used"]["probe_episodes"] for r in doc["rounds"]] == [1, 1]
    assert [r["run_budget"]["used"]["probe_episodes"] for r in doc["rounds"]] == [1, 2]
    assert doc["run_budget"]["used"]["input_bytes"] == sum(r["usage"]["input_bytes"] for r in doc["rounds"])
    assert len(model.requests) == 4


def test_finite_model_stop_continues_but_only_with_remaining_shared_budget(tmp_path, monkeypatch):
    model = CycleModel([stop(), stop()])
    _, doc = _run(tmp_path, monkeypatch, model, rounds=5, max_calls=2)
    assert len(doc["rounds"]) == len(model.requests) == 2
    assert [r["cycle_budget"]["limits"]["model_calls"] for r in doc["rounds"]] == [2, 1]
    assert doc["run_budget"]["used"]["model_calls"] == doc["run_budget"]["limits"]["model_calls"] == 2
    assert doc["stop_reason"] == "budget_exhausted"


def test_neutral_branch_survives_cycle_and_composes_before_full_acceptance(tmp_path, monkeypatch):
    class ComposeAcrossCycles(_Model):
        def chat(self, messages, **options):
            state = json.loads(messages[1]['content'])['state']
            index = len(self.requests)
            self.requests.append(copy.deepcopy(messages))
            if index == 0:
                answer = self._trial('first')
            elif index == 1:
                answer = stop('First alone is neutral; next combine second on the measured parent.')
            elif index == 2:
                parents = [p for p in state['working_policies'] if p['scope'] == 'probe']
                assert len(parents) == 1
                assert parents[0]['tried']['detail']['to'] == 1.0
                assert parents[0]['measurements'][0]['gains'] == []
                answer = self._trial('second', parents[0]['policy_id'])
            elif index == 3:
                answer = {'op': 'choose', 'args': {'policy_id': state['policy_id']}}
            else:
                assert all(p['scope'] == 'incumbent' for p in state['working_policies'])
                answer = stop()
            return json.dumps(answer)
    model = ComposeAcrossCycles()
    _, doc = _run(tmp_path, monkeypatch, model, continuous=True, rounds=3,
                  max_calls=2, max_probe_episodes=1)
    first, second, third = doc['rounds']
    assert first['accepted'] is False and first['evaluation']['after'] is None
    assert second['accepted'] is True and second['evaluation']['after']['successes'] == 2
    assert second['learning']['probes'][0]['parent_id'] == first['learning']['probes'][0]['policy_id']
    assert [r['cycle_budget']['used']['probe_episodes'] for r in doc['rounds']] == [1, 1, 0]
    assert [r['cycle_budget']['used']['full_evaluations'] for r in doc['rounds']] == [0, 1, 0]
    assert len(doc['accepted_stack'][0]['ancestry']) == 2
    assert third['policy']['before_id'] == second['policy']['active_id']
    replay = doc['learning_replay'][0]
    assert replay['tried']['detail']['to'] == 1.0 and replay['parent_id']
    assert replay['sample_count'] == 1


def test_cached_cross_cycle_probe_has_no_new_sampling_charge(tmp_path, monkeypatch):
    model = CycleModel([_Model._trial('first'), stop(), _Model._trial('first'), stop()])
    _, doc = _run(tmp_path, monkeypatch, model, continuous=True, rounds=2, max_calls=2)
    assert [r['cycle_budget']['used']['probe_episodes'] for r in doc['rounds']] == [1, 0]
    assert doc['rounds'][1]['learning']['cached_probes'] == 1
    assert doc['rounds'][1]['usage']['episode_attempts'] == 0
    assert doc['run_budget']['used']['probe_episodes'] == 1
    assert len(doc['learning_replay']) == 1
    assert doc['cycle_context']['cycles_without_sample'] == 1


def test_continuous_empty_cycles_advance_immediately_and_count_toward_round_limit(tmp_path, monkeypatch):
    model = CycleModel([stop()] * 10)
    _, doc = _run(tmp_path, monkeypatch, model, continuous=True, rounds=10, max_calls=1)
    assert [r["cycle_budget"]["cycle"] for r in doc["rounds"]] == list(range(1, 11))
    assert doc["run_budget"]["used"]["model_calls"] == 10
    assert doc["stop_reason"] == "round_limit"


@pytest.mark.parametrize("budget", [{"max_calls": 0}, {"max_input_bytes": 0}, {"max_probe_episodes": 0}])
def test_continuous_zero_budget_does_not_open_baseline_or_spin(tmp_path, monkeypatch, budget):
    monkeypatch.setattr(evolve, "run_suite", lambda *a, **k: pytest.fail("unusable budget must not sample"))
    model = CycleModel([])
    _, doc = _run(tmp_path, monkeypatch, model, continuous=True, rounds=0, **budget)
    assert doc["rounds"] == [] and model.requests == []
    assert doc["status"] == "done" and doc["stop_reason"] == "budget_exhausted"


def test_continuous_irreducible_request_budget_stops_without_empty_cycle_retries(tmp_path, monkeypatch):
    model = CycleModel([])
    _, doc = _run(tmp_path, monkeypatch, model, continuous=True, rounds=0, max_input_bytes=16)
    assert len(doc["rounds"]) == 1 and model.requests == []
    assert doc["stop_reason"] == "budget_exhausted"


def test_cancel_at_continuous_cycle_boundary_preserves_completed_cycle(tmp_path, monkeypatch):
    marker = tmp_path / "cancel"
    class CancellingModel(CycleModel):
        def chat(self, messages, **options):
            reply = super().chat(messages, **options)
            marker.touch()
            return reply
    model = CancellingModel([stop()])
    _, doc = _run(tmp_path, monkeypatch, model, continuous=True, rounds=0,
                  cancel_marker=marker, expected_code=3)
    assert len(doc["rounds"]) == len(model.requests) == 1
    assert doc["status"] == doc["stop_reason"] == doc["live"]["phase"] == "cancelled"
    assert doc["run_budget"]["used"]["model_calls"] == 1


def test_failed_candidate_measurement_continues_without_delay(tmp_path, monkeypatch):
    original = evolve.run_suite
    failures = []
    def fail_first_probe(*args, **kwargs):
        if "probe-" in kwargs.get("media_prefix", "") and not failures:
            failures.append(True)
            kwargs["progress"](seed_started_at=1.0, seed=11, seed_index=0)
            raise RuntimeError("candidate failed while sampling")
        return original(*args, **kwargs)
    monkeypatch.setattr(evolve, "run_suite", fail_first_probe)
    model = CycleModel([_Model._trial("first"), stop(), _Model._trial("second"), stop()])
    _, doc = _run(tmp_path, monkeypatch, model, continuous=True, rounds=2, max_calls=2)
    first, second = doc["rounds"]
    assert first["learning"]["probes"][0]["error"]["type"] == "RuntimeError"
    assert second["learning"]["probes"][0]["measurement_sha"]
    assert [r["usage"]["episode_attempts"] for r in doc["rounds"]] == [3, 1]
    assert doc["status"] == "done" and doc["stop_reason"] == "round_limit"


def test_endpoint_failure_ends_continuous_job_with_charged_failed_call(tmp_path, monkeypatch):
    class BrokenModel(_Model):
        def chat(self, messages, **options):
            self.requests.append(messages)
            raise ConnectionError("endpoint unavailable")
    model = BrokenModel()
    _, doc = _run(tmp_path, monkeypatch, model, continuous=True, rounds=0, expected_code=5)
    assert len(doc["rounds"]) == len(model.requests) == 1
    assert doc["rounds"][0]["cycle_outcome"] == "error"
    assert doc["status"] == "failed" and doc["stop_reason"] == "model_error"
    assert doc["run_budget"]["used"]["model_calls"] == 1
