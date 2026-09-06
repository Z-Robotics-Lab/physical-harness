"""Old measured branches must not consume a new cycle's opportunity to sample.

These use the real learner and command loop with a deterministic fake world and
model. No benchmark, network endpoint, or experiment logs are involved.
"""

from test_evolve_agent_tools import _read_all
from test_evolve_observation_loop import STOP, _loop, _trial, _world
from test_rsi_program_learner import _candidate

from scripts import evolve_llm
from scripts.evolve_evidence import encoded


def _retained_neutral_world(*, count=1, large_summaries=False):
    world = _world()
    learner = world.learner
    learner.max_probes = count
    policies = []
    for index in range(count):
        candidate = _candidate("first")
        candidate["detail"]["to"] = float(index + 1)
        receipt = learner.trial(candidate)
        policies.append(receipt["policy_id"])
        assert receipt["comparison"]["gains"] == []
        assert receipt["comparison"]["regressions"] == []
        if large_summaries:
            # Detailed per-condition explanations belong in addressable history,
            # not repeated in every decision as the workspace grows.
            learner.probes[-1]["comparison"]["reason"] = "; ".join(
                f"condition {clause}: fixed verification unchanged across the paired observation; "
                "no newly satisfied condition, no measured regression"
                for clause in range(24)
            )
    learner.begin_cycle(max_probes=3)
    assert learner.probes == []
    assert len(learner.policies) == count + 1
    world.calls.clear()
    return world, policies


def test_historical_candidate_does_not_force_selection_before_the_first_new_trial(tmp_path):
    world, policies = _retained_neutral_world()

    _, row, bodies, _ = _loop(tmp_path, world, [_trial("second")], budget={"max_calls": 1})

    assert bodies[0].get("phase") != "selection"
    assert policies[0] in {p["policy_id"] for p in bodies[0]["state"]["working_policies"]}
    assert row["calls"] == row["trial_calls"] == 1
    assert len(world.learner.probes) == 1
    assert [scope for _, _, scope in world.calls] == ["probe"]
    assert world.learner.full_calls == 0


def test_large_retained_workspace_still_fits_an_executable_first_sampling_decision(tmp_path):
    world, policies = _retained_neutral_world(count=9, large_summaries=True)
    unabridged = evolve_llm.compact_brief(world.learner.projection())
    assert len(encoded(unabridged)) > evolve_llm.AGENT_BUDGET["max_request_bytes"]

    _, row, bodies, endpoint = _loop(tmp_path, world, [_trial("second"), STOP],
                                   budget={"max_calls": 2})

    assert bodies[0].get("phase") != "selection"
    assert "move" in bodies[0]["state"]["driver_index"]
    compact = {p["policy_id"]: p for p in bodies[0]["state"]["working_policies"]}
    assert set(compact) == {world.learner.initial_id, *policies}
    for policy_id in policies:
        policy = compact[policy_id]
        assert policy["parent_id"] == world.learner.initial_id
        assert policy["scope"] == "probe"
        references = [value for key, value in policy.items()
                      if key.endswith("_ref") and isinstance(value, dict)
                      and value.get("view") == "history"]
        assert references, "Compacted measured outcomes must remain addressable."
        reference = references[0]
        assert reference["policy_id"] == policy_id
        full, _ = _read_all(world.learner.projection(policy_id), reference)
        assert full["working_policy"]["policy_id"] == policy_id
        assert full["working_policy"]["measurements"][0]["comparable"] is True
        assert "condition 23" in full["working_policy"]["measurements"][0]["reason"]
    assert row["trial_calls"] == 1
    assert len(world.learner.probes) == 1
    assert all(len(encoded(messages)) <= evolve_llm.AGENT_BUDGET["max_request_bytes"]
               for messages, _ in endpoint.requests)
    assert row["budget"]["used"]["input_bytes"] <= row["budget"]["limits"]["max_input_bytes"]
    assert [scope for _, _, scope in world.calls] == ["probe"]


def test_cached_old_trial_does_not_reserve_last_call_away_from_new_sampling(tmp_path):
    world, _ = _retained_neutral_world()

    _, row, bodies, _ = _loop(tmp_path, world, [_trial("first"), _trial("second")],
                             budget={"max_calls": 2})

    assert all(body.get("phase") != "selection" for body in bodies)
    assert bodies[1]["last_tool_result"]["data"]["cached"] is True
    assert bodies[1]["state"]["probe_budget"]["used"] == 0
    assert row["trial_calls"] == 2
    assert world.learner.cache_hits == 1
    assert len(world.learner.probes) == 1
    assert [scope for _, _, scope in world.calls] == ["probe"]
    assert world.learner.full_calls == 0
