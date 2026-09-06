"""Compact requests retain an executable protocol for stateless model endpoints."""

import copy

from test_evolve_agent_tools import _read_all
from test_evolve_llm_e2e import recycle_cans_projection
from test_evolve_observation_loop import STOP, _loop, _trial, _world

from harness.protocol import content_id
from plugins.rsi import evaluation
from scripts import evolve_llm
from scripts.evolve_evidence import encoded, request_messages


def test_two_probes_and_choice_fit_without_extra_model_calls_or_losing_capabilities(tmp_path):
    world = _world(sampled_motion=True)
    original = world.learner._project
    catalog, _ = recycle_cans_projection()

    def project(overlay, suite):
        state = original(overlay, suite)
        state["drivers"].update(copy.deepcopy(catalog["drivers"]))
        return state

    world.learner._project = project
    first = world.learner._identity({"tunables": {"first": 1.0, "second": 0.0}})
    combined = world.learner._identity({"tunables": {"first": 1.0, "second": 1.0}})
    _, row, bodies, endpoint = _loop(tmp_path, world, [_trial("first"), _trial("second", first),
        {"op": "choose", "args": {"policy_id": combined}}], budget={"max_input_bytes": 30000})
    assert row["stop_reason"] == "chosen" and row["calls"] == 3 and row["trial_calls"] == 2
    assert row["evidence_reads"] == 0  # Compacting the catalog must not require an extra lookup.
    assert row["budget"]["used"]["input_bytes"] == sum(len(encoded(m)) for m, _ in endpoint.requests) < 30000
    assert [scope for _, _, scope in world.calls] == ["probe", "probe", "full"]
    assert evaluation.compare(world.baseline, world.learner.selected_suite, world.contract)["accepted"]
    for body, (messages, _) in zip(bodies, endpoint.requests, strict=True):
        state = body["state"]
        projection = world.learner.projection(state["policy_id"])
        full = evolve_llm.compact_brief(projection)
        assert state["evaluation"] == full["evaluation"]
        assert state["working_policies"] == full["working_policies"][:len(state["working_policies"])]
        if body.get("phase") == "selection":
            assert body["allowed_ops"] == ["choose", "stop"]
            assert world.learner.selected_id in {p["policy_id"] for p in state["working_policies"]}
            continue
        assert set(state["driver_index"]) == set(full["driver_index"])
        assert state["observations"] == full["observations"]
        assert "source_index" not in state and all("modules" not in d for d in state["driver_index"].values())
        for node, driver in state["driver_index"].items():
            capability = state["capabilities"][driver["capability"]]
            assert capability["tunables"] == full["driver_index"][node]["tunables"]
            assert capability["executors"] == full["driver_index"][node]["executors"]
        retrieved, _ = _read_all(projection, state["catalog_ref"])
        assert content_id(retrieved) == state["catalog_ref"]["sha"]
        objective, _ = _read_all(projection, state["evaluation"]["objectives_ref"])
        assert objective["contract"] == world.contract
        assert messages[0]["content"] == evolve_llm._AGENT_RULES
        assert "tunables={node,parameter,to:number}" in messages[0]["content"]
    assert bodies[1]["last_tool_result"]["data"]["accepted"] is False
    candidate = next(p for p in bodies[2]["state"]["working_policies"] if p["policy_id"] == combined)
    assert candidate["measurements"][0]["gains"]


def test_capability_deduplication_preserves_every_nodes_zero_values_and_write_schema():
    base = {"skill": "move", "executor": "scripted", "modules": ["installed.driver"],
            "bound_classes": ["installed.driver:Driver"], "tunables": {"speed": 0.0}, "executors": ["scripted", "alt"]}
    state = {"policy_id": "current", "driver_index": {f"move-{i}": copy.deepcopy(base) for i in range(40)},
             "source_index": {}, "observations": []}
    body = {"state": state, "last_tool_result": None, "retained_evidence": []}
    unchanged = copy.deepcopy(body)
    compact, messages = request_messages(body, "protocol", 24000)
    assert body == unchanged and set(compact["state"]["driver_index"]) == set(state["driver_index"])
    assert list(compact["state"]["capabilities"].values()) == [{"tunables": {"speed": 0.0}, "executors": ["scripted", "alt"], "modules": ["installed.driver"]}]
    assert len({d["capability"] for d in compact["state"]["driver_index"].values()}) == 1
    again, repeated = request_messages(compact, "protocol", 24000)
    assert again == compact and repeated == messages
    assert len(encoded(messages)) < len(encoded([{"role": "system", "content": "protocol"},
                                               {"role": "user", "content": encoded(body).decode()}]))


def test_large_retained_measurements_keep_exact_edits_and_addressable_verification():
    gains = [{'obligation': f'condition-{i}', 'before': False, 'after': True} for i in range(200)]
    policies = [{'policy_id': f'policy-{i}', 'tried': {'kind': 'tunables', 'detail': {'to': i}},
                 'measurements': [{'seed': 11, 'comparable': True, 'gains': gains, 'regressions': []},
                                  {'seed': 12, 'comparable': False, 'gains': None, 'regressions': None}]}
                for i in range(12)]
    body = {'state': {'policy_id': 'current', 'working_policies': policies}}
    compact, messages = request_messages(body, 'protocol', 24000)
    assert len(encoded(messages)) < 24000
    assert body['state']['working_policies'][0]['measurements'][0]['gains'] == gains
    for i, policy in enumerate(compact['state']['working_policies']):
        assert policy['tried']['detail']['to'] == i
        assert policy['measurements_ref'] == {'view': 'history', 'policy_id': f'policy-{i}'}
        assert policy['measurements'][0]['gains_count'] == 200
        assert policy['measurements'][0]['regressions_count'] == 0
        assert policy['measurements'][1]['gains_count'] is None
        projection = {'policy_id': policy['policy_id'], 'working_policies': policies}
        full, _ = _read_all(projection, policy['measurements_ref'])
        assert full['working_policy']['measurements'][0]['gains'] == gains
    again, repeated = request_messages(compact, "protocol", 24000)
    assert again == compact and repeated == messages


def test_latest_page_is_sent_once_without_losing_error_or_mutating_the_evidence_cache():
    latest = {"view": "error", "policy_id": "probe", "sha": "error-sha", "cursor": 0, "next": None,
              "data": {"error": {"type": "ValueError", "message": "invalid selected policy"}}}
    previous = {"view": "trial", "policy_id": "probe", "sha": "measurement", "cursor": 0, "next": None,
                "data": {"accepted": False, "scope": "probe", "comparison": {"gains": ["frozen-condition"]}}}
    body = {"state": {"policy_id": "probe", "observations": []}, "last_tool_result": latest,
            "retained_evidence": [previous, copy.deepcopy(latest)]}
    original = copy.deepcopy(body)
    compact, _ = request_messages(body, "protocol", 24000)
    assert body == original
    assert compact["last_tool_result"] == latest and compact["retained_evidence"] == [previous]


def test_cycle_context_carries_the_last_memo_and_stop_reason_without_raw_history(tmp_path):
    world = _world()
    original = world.learner._project
    memo = "earlier" * 180 + "next experiment"
    context = {"previous_round": 4, "cycle_outcome": "none", "stop_reason": "budget_exhausted",
               "reason": "Input allowance ended after a neutral probe.", "memo": memo, "continuous": True, "cycle": 5,
               "cycle_budget": {"max_model_calls": 8}, "run_budget": {"model_calls": 32},
               "raw": "RAW_HISTORY_MUST_NOT_ENTER"}

    def project(overlay, suite):
        return {**original(overlay, suite), "cycle_context": context}

    world.learner._project = project
    _, row, bodies, _ = _loop(tmp_path, world, [
        {"op": "choose", "args": {"policy_id": "unknown"}, "memo": "Retain measured facts; repair this choice."}, STOP])
    assert bodies[0]["memo"] == memo[-1000:]
    assert bodies[1]["memo"] == row["memo"] == "Retain measured facts; repair this choice."
    for body in bodies:
        carried = body["state"]["cycle_context"]
        assert carried == {k: v for k, v in context.items() if k not in ("memo", "raw", "cycle_budget", "run_budget")}
        assert "RAW_HISTORY_MUST_NOT_ENTER" not in encoded(body).decode()
    assert bodies[1]["last_tool_result"]["data"]["error"]["message"].startswith("unknown policy_id")
    assert world.calls == []
