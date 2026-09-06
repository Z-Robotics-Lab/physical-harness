"""The real evolve entry point selects composed policies and obeys brief budgets.

Only the model endpoint and robot embodiment are fixtures. The environment action
changes the independently verified world; the real planner, runner, learner,
evaluation, campaign persistence and board readers remain on the execution path.
"""

from __future__ import annotations

import copy
import json

from test_evolve_e2e import _CARD, TASK, _Driver

from board import store as bs
from harness.manifest import mount_params
from scripts import evolve, evolve_llm

POLICY = "test_evolve_agent_runtime:policy_provider"


class _ConjunctionDriver(_Driver):
    def __init__(self, tunables):
        self.tunables = dict(tunables)

    def act(self, obs):
        if self.task == "grab" and self.tunables["first"] and self.tunables["second"]:
            self.n += 1
            return (2.0,)
        return super().act(obs)


class _Policy:
    def __init__(self, tunables):
        self.tunables = dict(tunables)

    def make_driver(self, spec):
        return _ConjunctionDriver(self.tunables)


def policy_provider(**params):
    return _Policy(mount_params(POLICY)["tunables"])


class _Model:
    identity = "fake(explicit-online-conjunction-experiment)"
    images = False

    def __init__(self, *, stop_after_probe=False, repeat_after_round=False):
        self.last_usage = {"prompt": 10, "completion": 5}
        self.requests = []
        self.first_id = None
        self.stop_after_probe = stop_after_probe
        self.repeat_after_round = repeat_after_round

    @staticmethod
    def _trial(field, parent=None):
        return {"op": "trial", "args": {"kind": "tunables", "parent_policy_id": parent,
                "payload": {"node": "grab-0", "ref": POLICY,
                            "path": ["tunables", field], "to": 1.0}}}

    def chat(self, messages, **options):
        index = len(self.requests)
        body = json.loads(messages[1]["content"])
        self.requests.append((copy.deepcopy(messages), dict(options)))
        if index == 0:
            answer = {"op": "inspect", "args": {"view": "parameter", "node": "grab-0", "parameter": "first"}}
        elif index == 1:
            answer = self._trial("first")
        elif index == 2:
            self.first_id = body["state"]["policy_id"]
            if self.stop_after_probe:
                answer = {"op": "stop", "args": {"reason": "The measured probe was neutral; retain the incumbent."}}
            else:
                answer = self._trial("second")
        elif index == 3:
            answer = {"op": "inspect", "args": {"view": "parameter", "node": "grab-0",
                       "parameter": "second", "policy_id": self.first_id}}
        elif index == 4:
            answer = self._trial("second", self.first_id)
        elif index == 5:
            answer = {"op": "choose", "args": {"policy_id": body["state"]["policy_id"]}}
        else:
            assert self.repeat_after_round, "The model was called after its declared decision budget."
            answer = {"op": "stop", "args": {"reason": "No new experiment proposed."}}
        return json.dumps(answer)


def _run(tmp_path, monkeypatch, model, *, rounds=1, max_calls=8, max_input_bytes=96000,
         max_probe_episodes=3, continuous=False, cancel_marker=None, expected_code=0):
    card = tmp_path / "plugins" / "experiment"
    card.mkdir(parents=True)
    (card / "manifest.toml").write_text(_CARD.replace("test_evolve_e2e:policy_provider", POLICY))
    params = tmp_path / "plugins" / "parameter_declaration"
    params.mkdir()
    (params / "manifest.toml").write_text(
        f'enabled = false\n[mounts."policy.driver"]\nref = "{POLICY}"\n'
        '[mounts."policy.driver".params.tunables]\nfirst = 0.0\nsecond = 0.0\n')
    monkeypatch.setenv("PH_PLUGINS_EXTRA", str(tmp_path / "plugins"))
    monkeypatch.setattr(evolve, "_BASE_EXTRA", str(tmp_path / "plugins"))
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr(evolve_llm, "endpoint", lambda: model)
    session = tmp_path / "session"
    options = (["--continuous"] if continuous else []) + (
        ["--cancel-marker", str(cancel_marker)] if cancel_marker is not None else [])
    assert evolve.main(["--mode", "evolution", "--task", TASK, "--session", str(session),
                        "--skills-root", str(session / "skills"), "--seeds", "11", "12",
                        "--rounds", str(rounds), "--confirm-seeds", "0", "--max-model-calls", str(max_calls),
                        "--max-input-bytes", str(max_input_bytes), "--max-probe-episodes", str(max_probe_episodes),
                        *options]) == expected_code
    path = session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    return session, json.loads(path.read_text())


def test_main_accepts_the_selected_composed_overlay_after_real_paired_world_success(tmp_path, monkeypatch):
    model = _Model()
    session, doc = _run(tmp_path, monkeypatch, model)
    row, = doc["rounds"]
    probes = row["learning"]["probes"]
    assert len(probes) == 3
    assert probes[0]["comparison"]["gains"] == probes[1]["comparison"]["gains"] == []
    assert probes[2]["comparison"]["gains"] and all(p["accepted"] is False for p in probes)
    assert probes[2]["parent_id"] == probes[0]["policy_id"]
    assert (row["before"], row["after"], row["accepted"], row["published"]) == (0, 2, True, False)
    assert doc["applied"]["tunables"][POLICY]["tunables"] == {"first": 1.0, "second": 1.0}
    assert row["policy"]["active_id"] == row["policy"]["candidate_id"] == probes[2]["policy_id"]
    assert row["policy"]["updated"] is True
    ancestry = doc["accepted_stack"][0]["ancestry"]
    assert {item["policy_id"] for item in ancestry} == {probes[0]["policy_id"], probes[2]["policy_id"]}
    assert row["trial"] == {"scope": "full", "seeds": [11, 12], "target_pass": 2}
    assert row["learning"]["full_evaluations"] == 1
    assert row["run_budget"]["used"]["model_calls"] == len(model.requests) == 6
    assert row["run_budget"]["used"]["probe_episodes"] == 3
    assert row["usage"]["llm_tokens"] == {"prompt": 60, "completion": 30}
    assert not list((session / "skills").glob("**/record.json"))
    persisted = bs.rsi_run(session, TASK, 1)["rounds"][0]
    for key in ("learning", "policy", "run_budget", "evaluation", "transfer", "usage"):
        assert persisted[key] == row[key]


def test_main_stop_after_a_real_probe_keeps_after_and_experience_unmeasured(tmp_path, monkeypatch):
    _, doc = _run(tmp_path, monkeypatch, _Model(stop_after_probe=True))
    row, = doc["rounds"]
    assert len(row["learning"]["probes"]) == 1
    assert row["learning"]["full_evaluations"] == 0
    assert row["trial"] is row["after"] is row["after_score"] is row["suite_sha"] is None
    assert row["experiments"]["after"] is row["evaluation"]["after"] is None
    assert row["after_seeds"] == [] and row["experience"]["recorded"] is None
    assert row["policy"]["updated"] is False and doc["applied"]["tunables"] == {}
    assert not doc["accepted_stack"]
    assert len(doc["learning_replay"]) == 1  # Measured negative evidence can be inspected later.


def test_submitted_brief_call_budget_is_shared_across_rounds_and_cannot_reset_after_acceptance(tmp_path, monkeypatch):
    model = _Model()
    _, doc = _run(tmp_path, monkeypatch, model, rounds=2, max_calls=6)
    assert len(model.requests) == doc["run_budget"]["used"]["model_calls"] == 6
    selection = json.loads(model.requests[-1][0][1]['content'])
    assert selection['phase'] == 'selection'
    assert selection['state']['evaluation_budget']['candidate_episodes'] == 2
    assert selection['state']['development_cost']['current_round']['episode_attempts'] == 5
    assert doc["rounds"][0]["accepted"] is True
    assert len(doc["rounds"]) == 1  # No phantom zero-call cycle after the shared budget is spent.
    assert doc["stop_reason"] == "budget_exhausted"
    assert doc["status"] == "done"


def test_round_cost_counts_sampling_once_and_carries_costs_to_model_and_board(tmp_path, monkeypatch):
    model = _Model(repeat_after_round=True)
    # Leave input space for the second round; the independent shared-budget test remains bounded.
    session, doc = _run(tmp_path, monkeypatch, model, rounds=2, max_input_bytes=192000,
                        max_probe_episodes=4)
    improved, stopped = doc['rounds']
    assert improved['usage']['episode_attempts'] == 7  # baseline 2 + probes 3 + full 2
    assert stopped['usage']['episode_attempts'] == 0  # reused baseline, model chose stop
    assert [r['usage']['model_calls'] for r in doc['rounds']] == [6, 1]
    costs = stopped['transfer']['cost']
    assert costs['total']['episode_attempts'] == costs['first_accepted']['episode_attempts'] == 7
    assert costs['total']['model_calls'] == 7 and costs['first_accepted']['model_calls'] == 6
    assert costs['since_previous_acceptance']['episode_attempts'] == 0
    assert costs['since_previous_acceptance']['model_calls'] == 1
    assert stopped['transfer']['accepted_updates'] == 1
    selection = json.loads(model.requests[5][0][1]['content'])
    assert selection['state']['evaluation_budget']['used'] == 0
    assert selection['state']['development_cost']['current_round']['episode_attempts'] == 5
    assert selection['state']['working_policies'][-1]['measurements'][0]['cost']['episodes'] == 1
    final_request = json.loads(model.requests[-1][0][1]['content'])['state']
    assert final_request['development_cost']['accepted_updates'] == 1
    assert final_request['development_cost']['previous_rounds']['total']['episode_attempts'] == 7
    assert final_request['development_cost']['current_round']['episode_attempts'] == 0
    assert final_request['evaluation_budget']['candidate_episodes'] == 2
    assert final_request['evaluation_budget']['baseline_reused'] is True
    persisted = bs.rsi_run(session, TASK, 2)['rounds'][0]
    assert persisted['transfer']['cost'] == costs
    # Sharding and restart keep per-round costs, never the brief's cumulative counters.
    assert evolve.index_row(improved)['usage'] == improved['usage']
    resumed = evolve.EvolveStore(session, TASK).load()
    from plugins.rsi.experience import development_report
    assert development_report(resumed['rounds'], epoch_start=doc['epoch_start'])['cost'] == costs


def test_failed_probe_still_charges_started_sampling_and_cannot_claim_first_improvement(tmp_path, monkeypatch):
    original = evolve.run_suite

    def fail_probe(*args, **kwargs):
        if 'probe-' not in kwargs.get('media_prefix', ''):
            return original(*args, **kwargs)
        kwargs['progress'](seed_started_at=1.0, seed=11, seed_index=0)
        raise RuntimeError('fixture: sampling failed after starting an episode')

    monkeypatch.setattr(evolve, 'run_suite', fail_probe)
    class FailedProbeModel(_Model):
        def chat(self, messages, **options):
            self.requests.append((copy.deepcopy(messages), options))
            return json.dumps(self._trial('first') if len(self.requests) == 1 else {
                'op': 'stop', 'args': {'reason': 'Sampling is unavailable.'}})
    _, doc = _run(tmp_path, monkeypatch, FailedProbeModel())
    row, = doc['rounds']
    assert row['usage']['episode_attempts'] == 3  # baseline + started failed probe
    assert row['usage']['model_calls'] == 2
    assert row['usage']['sim_s'] >= 0 and row['usage']['wall_s'] >= row['usage']['sim_s'] - .01
    assert row['transfer']['censored'] is True
    assert row['transfer']['cost']['first_accepted'] is None
    assert row['transfer']['cost']['total']['episode_attempts'] == 3
    assert row['after'] is None and row['learning']['probes'][0]['error'] is not None
