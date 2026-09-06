"""Exploration is measured work on a branch, never development acceptance.

The tiny world needs two independent changes together. Neither one alone earns
reward, so a strict hill-climber cannot cross its plateau. These are protocol
regressions, not empirical evidence that transfer scales on robot benchmarks.
"""

from __future__ import annotations

import copy

import pytest

from plugins.rsi import evaluation
from plugins.rsi.learner import ProgramLearner


def _candidate(name):
    return {"kind": "tunables", "node": "move", "detail": {
        "ref": "test:controller", "path": ["tunables", name], "from": False, "to": True}}


class _World:
    def __init__(self, *, max_probes=3, sampled=True):
        self.contract = evaluation.compile_contract(
            {"nodes": []}, task="conjunction", terminal_ref="test:world")
        self.applied = {"tunables": {"first": 0.0, "second": 0.0}}
        self.calls, self.validated = [], []
        self.sampled = sampled
        self.baseline = self.measure([11, 12], self.applied)
        self.learner = ProgramLearner(
            applied=self.applied, baseline=self.baseline, contract=self.contract,
            seeds=[11, 12], project=self.project, validate=self.validate,
            apply=self.apply, run=self.run, max_probes=max_probes)

    def measure(self, seeds, overlay):
        done = all(overlay["tunables"].values())
        trace = {"series": [{"step": 1, "position": int(done)}]} if self.sampled else {}
        return {"experiment_id": "source-epoch-one", "seeds": {
            str(seed): {"evaluation": evaluation.evaluate(self.contract, [], {
                "authority": "embodiment.terminal_success", "source": "test:world", "success": done}),
                "nodes": {"move": {"steps": 1, "ok": done, "trace": copy.deepcopy(trace),
                                   "skill": "move", "executor": "scripted"}}}
            for seed in range(seeds[0], seeds[1] + 1)}}

    def project(self, overlay, suite):
        return {"applied": copy.deepcopy(overlay), "experiment_id": suite["experiment_id"]}

    def validate(self, tried, projection):
        assert tried["detail"]["path"][1] in projection["applied"]["tunables"]
        self.validated.append(copy.deepcopy(projection["applied"]))

    @staticmethod
    def apply(tried, parent):
        result = copy.deepcopy(parent)
        result["tunables"][tried["detail"]["path"][1]] = tried["detail"]["to"]
        return result

    def run(self, seeds, overlay, scope):
        self.calls.append((list(seeds), copy.deepcopy(overlay), scope))
        return self.measure(seeds, overlay)


def test_two_neutral_interventions_can_combine_but_only_full_paired_measurement_is_eligible():
    world = _World()
    learner, incumbent = world.learner, copy.deepcopy(world.applied)
    a = learner.trial(_candidate("first"))
    b = learner.trial(_candidate("second"))
    assert a["comparison"]["gains"] == b["comparison"]["gains"] == []
    assert a["behavior"][0]["sampled_trace_equal"] is True
    assert a["evaluation"] == a["baseline_evaluation"]

    combined = learner.trial(_candidate("second"), parent_policy_id=a["policy_id"])
    assert combined["parent_id"] == a["policy_id"]
    assert combined["comparison_to"] == learner.initial_id
    assert combined["comparison"]["gains"] and combined["comparison"]["accepted"]
    assert combined["accepted"] is False and combined["acceptance_scope"] == "not_evaluated"
    assert combined["scope"] == "probe" and combined["seeds"] == [11]
    assert learner.selected_id is learner.selected_suite is learner.selected_overlay is None
    assert world.applied == learner.policies[learner.initial_id]["overlay"] == incumbent

    selected = learner.choose(combined["policy_id"])
    assert selected == _candidate("second")
    assert learner.selected_overlay == {"tunables": {"first": True, "second": True}}
    assert world.calls[-1] == ([11, 12], learner.selected_overlay, "full")
    assert [scope for _, _, scope in world.calls] == ["probe"] * 3 + ["full"]
    assert evaluation.compare(world.baseline, learner.selected_suite, world.contract)["accepted"]
    assert world.applied == incumbent
    assert learner.report()["full_evaluations"] == 1
    with pytest.raises(ValueError, match="budget"):
        learner.choose(combined["policy_id"])


def test_stopping_after_a_probe_has_no_full_trial_or_selected_overlay():
    world = _World()
    world.learner.trial(_candidate("first"))
    assert world.learner.selected_suite is None
    assert world.learner.report()["selected_policy_id"] is None
    assert world.learner.report()["full_evaluations"] == 0
    assert [scope for _, _, scope in world.calls] == ["probe"]


def test_probe_feedback_omits_unobserved_graph_tail_but_seals_it_and_keeps_zero_measurements():
    world = _World()
    measure = world.measure

    def with_unexecuted_tail(seeds, overlay):
        suite = measure(seeds, overlay)
        for row in suite["seeds"].values():
            row["nodes"].update({f"later-{i}": {} for i in range(30)})
            row["nodes"]["stopped"] = {"steps": 0, "ok": False}
        return suite

    world.measure = with_unexecuted_tail
    receipt = world.learner.trial(_candidate("first"))
    assert receipt["unobserved_behavior_rows"] == 30
    assert {row["node"] for row in receipt["behavior"]} == {"move", "stopped"}
    stopped = next(row for row in receipt["behavior"] if row["node"] == "stopped")
    assert stopped["after_steps"] == 0 and stopped["after_ok"] is False
    assert stopped["before_steps"] is None and stopped["before_ok"] is None
    full = world.learner.report()["probes"][0]
    assert len(full["behavior"]) == 32
    assert full["evaluation"] == receipt["evaluation"]
    assert full["measurement_sha"] == receipt["measurement_sha"]
    world.learner.begin_cycle(max_probes=1)
    cached = world.learner.trial(_candidate('first'))
    assert cached['cached'] is True and cached['unobserved_behavior_rows'] == 30


def test_zero_probe_budget_prevents_environment_execution():
    world = _World(max_probes=0)
    with pytest.raises(ValueError, match="budget"):
        world.learner.trial(_candidate("first"))
    assert world.calls == world.learner.probes == []


@pytest.mark.parametrize("method", ["projection", "choose", "trial"])
def test_unknown_policy_cannot_be_inspected_selected_or_used_as_parent(method):
    world = _World()
    with pytest.raises(ValueError, match="unknown policy_id"):
        if method == "trial":
            world.learner.trial(_candidate("first"), parent_policy_id="invented")
        else:
            getattr(world.learner, method)("invented")
    assert world.calls == []


@pytest.mark.parametrize("seed", [True, 10, 13, "11"])
def test_probe_cannot_reach_an_undeclared_or_heldout_seed(seed):
    world = _World()
    with pytest.raises(ValueError, match="development seed"):
        world.learner.trial(_candidate("first"), seed=seed)
    assert world.calls == []


def test_failed_environment_call_spends_probe_budget_and_cannot_be_selected():
    world = _World(max_probes=1)

    def fail(*args):
        world.calls.append(args)
        raise RuntimeError("simulator failed before a measurement")

    world.learner._run = fail
    with pytest.raises(RuntimeError, match="before a measurement"):
        world.learner.trial(_candidate("first"))
    receipt = world.learner.probes[0]
    assert receipt["error"]["type"] == "RuntimeError"
    assert receipt["accepted"] is False
    assert receipt["policy_id"] not in world.learner.policies
    with pytest.raises(ValueError, match="budget"):
        world.learner.trial(_candidate("second"))
    assert len(world.calls) == 1


def test_equal_steps_without_actual_samples_do_not_claim_equal_trajectories():
    world = _World(sampled=False)
    receipt = world.learner.trial(_candidate("first"))
    behavior = receipt["behavior"][0]
    assert behavior["before_steps"] == behavior["after_steps"] == 1
    assert behavior["sampled_trace_equal"] is None


def test_incomplete_full_suite_cannot_be_selected_or_silently_reuse_a_probe():
    world = _World()
    receipt = world.learner.trial(_candidate("first"))
    world.learner._run = lambda seeds, overlay, scope: world.measure([11, 11], overlay)
    with pytest.raises(ValueError, match="entire paired seed set"):
        world.learner.choose(receipt["policy_id"])
    assert world.learner.selected_id is world.learner.selected_suite is None
    assert world.learner.report()["full_evaluations"] == 1


@pytest.mark.parametrize("returned_seeds", [{}, {"12": {}}, {"11": {}, "12": {}}, {11: {}, "11": {}}])
def test_invalid_probe_seed_batch_spends_budget_without_creating_a_working_policy(returned_seeds):
    world = _World(max_probes=1)
    world.learner._run = lambda *_: {"seeds": returned_seeds}
    with pytest.raises(ValueError, match="paired seed set"):
        world.learner.trial(_candidate("first"), seed=11)
    assert list(world.learner.policies) == [world.learner.initial_id]
    report = world.learner.report()
    assert len(report["probes"]) == 1
    assert report["probes"][0]["error"]["type"] == "ValueError"
    assert report["selected_policy_id"] is None
    with pytest.raises(ValueError, match="budget"):
        world.learner.trial(_candidate("second"))


def test_actual_runner_trail_samples_and_node_metadata_are_both_compared():
    world = _World()
    original = world.measure

    def actual_shape(seeds, overlay):
        suite = original(seeds, overlay)
        for row in suite["seeds"].values():
            metrics = row["nodes"]["move"]
            row["trail"] = [{"id": "move", **{k: metrics.pop(k) for k in ("steps", "ok", "trace")}}]
        return suite

    world.measure = actual_shape
    world.learner.baseline = actual_shape([11, 12], world.applied)
    result = world.learner.trial(_candidate("first"))
    assert result["behavior"] == [{"seed": 11, "node": "move", "before_steps": 1,
        "after_steps": 1, "before_ok": False, "after_ok": False, "sampled_trace_equal": True}]


def test_reverting_a_branch_to_a_known_policy_keeps_immutable_ancestry_and_incumbent():
    world = _World(max_probes=4)
    learner = world.learner
    baseline = copy.deepcopy(learner.policies[learner.initial_id])
    a = learner.trial(_candidate("first"))
    original_a = copy.deepcopy(learner.policies[a["policy_id"]])
    b = learner.trial(_candidate("second"), a["policy_id"])
    undo_second = _candidate("second")
    undo_second["detail"]["to"] = 0.0
    back_to_a = learner.trial(undo_second, b["policy_id"])
    assert back_to_a["policy_id"] == a["policy_id"]
    assert learner.policies[a["policy_id"]] == original_a
    undo_first = _candidate("first")
    undo_first["detail"]["to"] = 0.0
    back_to_initial = learner.trial(undo_first, a["policy_id"])
    assert back_to_initial["policy_id"] == learner.initial_id
    assert learner.policies[learner.initial_id] == baseline
    with pytest.raises(ValueError, match="incumbent"):
        learner.choose(learner.initial_id)
    assert learner.policies[a["policy_id"]]["parent_id"] == learner.initial_id


def test_directory_preserves_earlier_candidate_gains_after_a_later_neutral_probe():
    learner = _World().learner
    first = learner.trial(_candidate("first"))
    improved = learner.trial(_candidate("second"), first["policy_id"])
    later = learner.trial(_candidate("second"))
    projected = learner.projection(later["policy_id"])
    assert projected["feedback"]["policy_id"] == later["policy_id"]
    directory = {row["policy_id"]: row for row in projected["working_policies"]}
    earlier = directory[improved["policy_id"]]
    assert earlier["scope"] == "probe" and earlier["acceptance_scope"] == "not_evaluated"
    assert earlier["comparison_to"] == learner.initial_id
    assert earlier["measurements"] == [{
        "seed": 11, "comparable": True, "gains": [{"obligation": learner.contract["obligations"][0]["id"],
            "before": False, "after": True}], "regressions": [],
        "reason": "fixed verification vector improved without regressions",
        "cost": {"episodes": 1, "sim_s": None}}]
    assert directory[later["policy_id"]]["measurements"][0]["gains"] == []
    assert list(directory) == list(learner.policies)  # insertion order, no ranking
    assert learner.selected_id is None and learner.full_calls == 0
    assert not {"observations", "behavior", "overlay", "tried_detail"} & earlier.keys()


def test_directory_deduplicates_policy_seed_measurements_and_preserves_other_seeds():
    learner = _World().learner
    first = learner.trial(_candidate("first"), seed=11)
    directory = copy.deepcopy(learner.projection()["working_policies"])
    assert learner.trial(_candidate("first"), seed=11)["cached"] is True
    assert learner.projection()["working_policies"] == directory
    again = learner.trial(_candidate("first"), seed=12)
    assert again["policy_id"] == first["policy_id"]
    policies = learner.projection()["working_policies"]
    assert len(policies) == 2 and len(learner.probes) == 2
    assert [row["seed"] for row in policies[1]["measurements"]] == [11, 12]


def test_directory_preserves_unknown_comparison_and_excludes_failed_measurements():
    world = _World()
    learner = world.learner
    unknown = evaluation.evaluate(world.contract, [], None)
    for row in learner.baseline["seeds"].values():
        row["evaluation"] = copy.deepcopy(unknown)
    measure = world.measure

    def without_oracle(seeds, overlay, scope):
        suite = measure(seeds, overlay)
        for row in suite["seeds"].values():
            row["evaluation"] = copy.deepcopy(unknown)
        return suite

    learner._run = without_oracle
    measured = learner.trial(_candidate("first"))
    entry = learner.projection()["working_policies"][1]["measurements"][0]
    assert entry["comparable"] is False
    assert entry["gains"] is entry["regressions"] is None
    assert entry["reason"] == "evaluation unavailable: no independent observation"

    def fail(*_):
        raise RuntimeError("no measurement")

    learner._run = fail
    with pytest.raises(RuntimeError, match="no measurement"):
        learner.trial(_candidate("second"))
    projected = learner.projection()
    assert [row["policy_id"] for row in projected["working_policies"]] == [learner.initial_id, measured["policy_id"]]
    assert projected["working_policies"][1]["measurements"] == [entry]


@pytest.mark.parametrize("before_value,after_value,direction", [
    (None, True, "gains"), (True, None, "regressions")])
def test_directory_retains_component_identity_and_unknown_transition(before_value, after_value, direction):
    world = _World()

    def reading(value):
        return evaluation.evaluate(world.contract, [], {
            "authority": "embodiment.terminal_success", "source": "test:world", "success": value})

    for row in world.learner.baseline["seeds"].values():
        row["evaluation"] = reading(before_value)
    measure = world.measure

    def measured(seeds, overlay, scope):
        suite = measure(seeds, overlay)
        for row in suite["seeds"].values():
            row["evaluation"] = reading(after_value)
        return suite

    world.learner._run = measured
    receipt = world.learner.trial(_candidate("first"))
    entry = world.learner.projection()["working_policies"][1]["measurements"][0]
    assert entry["comparable"] is True
    assert entry[direction] == [{"obligation": world.contract["obligations"][0]["id"],
                                 "before": before_value, "after": after_value}]
    assert entry["reason"] == receipt["comparison"]["reason"]
    assert world.learner.selected_id is None


def test_model_sees_full_suite_cost_and_each_measured_probe_cost_without_a_ranking():
    world = _World()
    learner = world.learner
    assert learner.projection()['evaluation_budget'] == {
        'limit': 1, 'used': 0, 'candidate_episodes': 2, 'baseline_reused': True}
    a = learner.trial(_candidate('first'))
    entry = learner.projection()['working_policies'][1]
    assert entry['measurements'][0]['cost'] == a['cost'] == {'episodes': 1, 'sim_s': None}
    learner.choose(a['policy_id'])
    assert learner.projection()['evaluation_budget']['used'] == 1
    assert learner.report()['selection']['cost'] == {'episodes': 2, 'sim_s': None}


def test_cycle_budget_renewal_preserves_neutral_branches_for_later_composition():
    world = _World(max_probes=1)
    learner = world.learner
    first = learner.trial(_candidate('first'))
    assert first['comparison']['gains'] == []
    first_report = learner.report()

    learner.begin_cycle(max_probes=1)
    assert learner.report()['probes'] == []
    assert learner.projection()['probe_budget'] == {'limit': 1, 'used': 0}
    retained = next(p for p in learner.projection()['working_policies']
                    if p['policy_id'] == first['policy_id'])
    assert retained['measurements'][0]['seed'] == 11
    combined = learner.trial(_candidate('second'), first['policy_id'])
    assert combined['comparison']['gains']
    assert learner.selected_id is None
    assert learner.policies[learner.initial_id]['overlay'] == world.applied

    learner.choose(combined['policy_id'])
    assert evaluation.compare(world.baseline, learner.selected_suite, world.contract)['accepted']
    assert [scope for _, _, scope in world.calls] == ['probe', 'probe', 'full']
    assert len(learner.report()['probes']) == len(first_report['probes']) == 1
    assert learner.report()['probes'][0]['policy_id'] == combined['policy_id']
    assert first_report['probes'][0]['policy_id'] == first['policy_id']


def test_begin_cycle_resets_selection_without_promoting_the_selected_branch():
    world = _World()
    learner = world.learner
    branch = learner.trial(_candidate('first'))
    learner.choose(branch['policy_id'])
    learner.begin_cycle(max_probes=2)
    report = learner.report()
    assert report['probes'] == [] and report['full_evaluations'] == 0
    assert report['selection'] is report['selected_policy_id'] is None
    assert learner.selected_suite is learner.selected_overlay is None
    assert learner.projection()['incumbent_policy_id'] == learner.initial_id
    assert learner.policies[learner.initial_id]['overlay'] == world.applied
    # A new cycle may select this measured branch, but still owes a full suite.
    learner.choose(branch['policy_id'])
    assert [scope for _, _, scope in world.calls] == ['probe', 'full', 'full']


def test_repeated_policy_seed_across_cycles_is_cached_without_charging_a_new_probe():
    world = _World(max_probes=1)
    learner = world.learner
    measured = learner.trial(_candidate('first'), seed=11)
    learner.begin_cycle(max_probes=1)
    cached = learner.trial(_candidate('first'), seed=11)
    assert cached['cached'] is True and cached['policy_id'] == measured['policy_id']
    assert cached['comparison'] == measured['comparison']
    assert len(world.calls) == 1 and learner.report()['probes'] == []
    assert learner.projection()['probe_budget']['used'] == 0
    learner.trial(_candidate('second'))
    assert len(world.calls) == 2 and len(learner.report()['probes']) == 1


def test_new_seed_refreshes_working_observation_but_keeps_ancestry_and_all_measurements():
    world = _World()
    learner = world.learner
    incumbent = copy.deepcopy(learner.policies[learner.initial_id])
    first = learner.trial(_candidate('first'), seed=11)
    original = copy.deepcopy(learner.policies[first['policy_id']])
    learner.begin_cycle(max_probes=1)
    again = learner.trial(_candidate('first'), seed=12)
    assert again['policy_id'] == first['policy_id']
    assert set(learner.observations(first['policy_id'])['seeds']) == {'12'}
    current = learner.policies[first['policy_id']]
    assert current['tried'] == original['tried']
    assert current['parent_id'] == original['parent_id'] == learner.initial_id
    assert learner.policies[learner.initial_id] == incumbent
    measured = next(p for p in learner.projection()['working_policies']
                    if p['policy_id'] == first['policy_id'])
    assert {m['seed'] for m in measured['measurements']} == {11, 12}
    cached = learner.trial(_candidate('first'), seed=11)
    assert cached['cached'] is True
    assert set(learner.observations(first['policy_id'])['seeds']) == {'11'}
    assert learner.policies[learner.initial_id] == incumbent
    assert len(world.calls) == 2 and len(learner.report()['probes']) == 1


def test_full_selection_revalidates_changed_ancestor_content_before_execution():
    world = _World()
    learner = world.learner
    material = {'first': 'version-1', 'second': 'version-1'}
    validated = []

    def validate_material(tried, projection):
        world.validate(tried, projection)
        name = tried['detail']['path'][-1]
        validated.append(name)
        if tried['detail']['artifact_sha'] != material[name]:
            raise ValueError('candidate content changed after measurement')

    learner._validate = validate_material
    first, second = _candidate('first'), _candidate('second')
    first['detail']['artifact_sha'] = second['detail']['artifact_sha'] = 'version-1'
    parent = learner.trial(first)
    combined = learner.trial(second, parent['policy_id'])
    learner.begin_cycle(max_probes=1)
    material['first'] = 'version-2'
    validated.clear()
    prior_calls = len(world.calls)
    with pytest.raises(ValueError, match='content changed'):
        learner.choose(combined['policy_id'])
    assert 'first' in validated
    assert len(world.calls) == prior_calls
    assert learner.selected_id is learner.selected_suite is learner.selected_overlay is None


@pytest.mark.parametrize('chain', [False, True])
def test_cycle_workspace_pruning_is_bounded_and_retains_complete_ancestry(chain):
    from plugins.rsi.learner import WORKSPACE_LIMIT

    world = _World(max_probes=WORKSPACE_LIMIT + 4)
    learner = world.learner
    parent = learner.initial_id
    for index in range(WORKSPACE_LIMIT + 4):
        edit = _candidate('first')
        edit['detail']['to'] = index + 1
        receipt = learner.trial(edit, parent if chain else learner.initial_id)
        assert receipt['comparison']['gains'] == []
        parent = receipt['policy_id']
    learner.begin_cycle(max_probes=1)
    assert 1 < len(learner.policies) <= WORKSPACE_LIMIT + 1
    for policy_id in learner.policies:
        seen = set()
        while policy_id != learner.initial_id:
            assert policy_id not in seen
            seen.add(policy_id)
            policy_id = learner.policies[policy_id]['parent_id']
            assert policy_id in learner.policies
    assert learner.report()['probes'] == []
