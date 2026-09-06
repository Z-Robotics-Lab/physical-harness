"""Bounded model decisions retain honest access to the complete experiment.

The model below is a deterministic endpoint fixture, never a proposal fallback.
All probes and source reads are local; these tests spend no model/simulator budget.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest
from test_evolve_llm_e2e import _repeat_proj, recycle_cans_projection

from harness.protocol import content_id
from scripts import evolve_llm
from scripts.evolve_evidence import encoded


def _projection():
    proj, before = _repeat_proj([])
    proj.update(task="fixture", round=1, policy_id="incumbent")
    proj["drivers"]["grab-0"].update(
        source_ref="test_evolve_e2e:policy_provider", modules=["test_evolve_e2e"])
    return proj, before


class _Endpoint:
    identity = "fake(explicit-agent-tool-sequence)"
    images = True

    def __init__(self, replies, usage=None):
        self.replies, self.requests, self.last_usage = list(replies), [], usage

    def chat(self, messages, **options):
        self.requests.append((copy.deepcopy(messages), dict(options)))
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer)


def _tools(proj):
    calls = []
    measured = {}

    def trial(tried, parent_policy_id=None):
        calls.append(("trial", copy.deepcopy(tried), parent_policy_id))
        policy = f"probe-{len(calls)}"
        measured[policy] = {**copy.deepcopy(proj), "policy_id": policy, "incumbent_policy_id": proj["policy_id"]}
        return {"policy_id": policy, "scope": "probe", "accepted": False}

    def choose(policy_id):
        calls.append(("choose", policy_id))
        return {"kind": "executor", "node": "grab-0", "detail": {"to": "alt"}}

    def projection(policy_id=None):
        if policy_id in measured:
            return copy.deepcopy(measured[policy_id])
        if policy_id not in (None, proj["policy_id"]):
            raise ValueError("unknown policy_id")
        return copy.deepcopy(proj)

    return {"trial": trial, "choose": choose, "projection": projection}, calls


def _propose(tmp_path, endpoint, *, budget=None, proj=None, tools=None):
    default, before = _projection()
    proj = proj or default
    tools = tools or _tools(proj)[0]
    return evolve_llm.llm_propose(endpoint, proj, before, 1, tmp_path,
                                   agent_tools=tools, budget=budget)


def test_compact_keeps_every_executed_node_and_excludes_unrequested_large_evidence():
    proj, _ = recycle_cans_projection(trace={"series": [{"position": "TRACE_SENTINEL" * 1000}]})
    for i in range(200):
        proj["drivers"][f"passed-upstream-{i}"] = copy.deepcopy(proj["drivers"]["carry-can1"])
    proj["module_sources"] = {"source": "SOURCE_SENTINEL" * 10000}
    proj["history"] = [{"trial_evidence": "HISTORY_SENTINEL" * 10000}]
    proj["plan_space"] = {"graph": "GRAPH_SENTINEL" * 10000}
    compact = evolve_llm.compact_brief(proj)
    assert set(compact["driver_index"]) == set(proj["drivers"])
    assert "carry-can1" in compact["driver_index"]  # It passed, but may be causally relevant.
    text = encoded(compact).decode()
    for marker in ("TRACE_SENTINEL", "SOURCE_SENTINEL", "HISTORY_SENTINEL", "GRAPH_SENTINEL"):
        assert marker not in text
    assert all(not row["success"] for row in compact["observations"])


def _read_all(proj, args, *, cap=900):
    pages, fragments, code_parts, cursor = [], [], [], 0
    while True:
        item = evolve_llm.inspect_evidence(proj, {**args, "cursor": cursor}, max_bytes=cap)
        pages.append(item)
        assert len(encoded(item)) <= cap
        assert item["cursor"] == cursor
        if isinstance(item["data"], dict) and item["data"].get("code_read"):
            code_parts.append(item["data"]["code"])
            if item["next"] is None:
                assert item["complete"]
                assert len({p["sha"] for p in pages}) == 1
                return {**item["data"], "code": "".join(code_parts)}, pages
            assert item["next"] > cursor and not item["complete"]
            cursor = item["next"]
            continue
        fragment = isinstance(item["data"], dict) and item["data"].get("encoding") == "json_fragment"
        if not fragment:
            assert len(pages) == 1 and item["complete"]
            return item["data"], pages
        fragments.append(item["data"]["text"])
        if item["next"] is None:
            assert item["complete"]
            break
        assert item["next"] > cursor and not item["complete"]
        cursor = item["next"]
    assert len({p["sha"] for p in pages}) == 1
    value = json.loads("".join(fragments))
    assert content_id(value) == pages[0]["sha"]
    return value, pages


def test_unicode_paging_is_byte_bounded_lossless_and_keeps_evidence_identity():
    proj, _ = _projection()
    proj["history"] = [{"reason": "机器人证据🙂" * 1000, "after": None}]
    data, pages = _read_all(proj, {"view": "history"})
    assert data["history"] == proj["history"] and len(pages) > 1
    assert all(p["policy_id"] == "incumbent" for p in pages)


def test_trace_inspection_selects_the_requested_node_and_seed_without_erasing_unknowns():
    proj, _ = _projection()
    proj["this_round"]["per_seed"] = [
        {"seed": seed, "trail": [{"id": "grab-0", "trace": {"series": [{"step": seed, "distance": None}]}},
                                 {"id": "other", "trace": {"secret": "other node"}}]}
        for seed in (11, 12)]
    value = evolve_llm.inspect_evidence(proj, {"view": "trace", "node": "grab-0", "seed": 12})
    assert value["data"] == [{"seed": 12, "nodes": [proj["this_round"]["per_seed"][1]["trail"][0]]}]
    with pytest.raises(ValueError, match="executed driver"):
        evolve_llm.inspect_evidence(proj, {"view": "trace", "node": "not-executed"})


def test_parameter_view_exposes_real_assignment_and_consumption_not_an_invented_tuning_direction():
    module = "plugins.embodiment_robocasa.drivers"
    source = Path("plugins/embodiment_robocasa/drivers.py").read_text()
    proj = {"policy_id": "installed", "drivers": {"carry": {"modules": [module],
            "tunables": {"values": {"carry_stop": 0.65}}}}}
    value, _ = _read_all(proj, {"view": "parameter", "node": "carry", "parameter": "carry_stop"})
    assert value["effective"] == {"carry_stop": 0.65}
    assert value["complete_dependency_analysis"] is False
    assert any('self.CARRY_STOP = tunables()["carry_stop"]' in r["code"] for r in value["references"])
    assert any("d <= self.CARRY_STOP" in r["code"] for r in value["references"])
    for ref in value["references"]:
        assert ref["source_sha"] == content_id(source)
        assert ref["code"] == "".join(source.splitlines(keepends=True)[ref["start"] - 1:ref["end"]])
    body, _ = _read_all(proj, {"view": "source", "node": "carry", "module": module, "symbol": "_base_action"})
    assert "kp" in body["code"] and "CARRY_STOP" not in body["code"]
    unknown, _ = _read_all(proj, {"view": "parameter", "node": "carry", "parameter": "imaginary"})
    assert unknown["declared_tunable"] is False and unknown["effective"] is None
    assert unknown["unresolved"] is True and unknown["references"] == []
    with pytest.raises(ValueError, match="module must"):
        evolve_llm.inspect_evidence(proj, {"view": "source", "node": "carry", "module": "os"})


def test_named_parameter_trial_derives_the_installed_address_and_passes_real_validation(tmp_path):
    from scripts import evolve

    proj, before = recycle_cans_projection()
    proj["policy_id"] = "installed"
    tools, calls = _tools(proj)
    original_trial = tools["trial"]

    def trial(candidate, parent_policy_id=None):
        evolve._validate_try(candidate, proj, {}, {})
        return original_trial(candidate, parent_policy_id)

    tools["trial"] = trial
    ep = _Endpoint([
        {"op": "inspect", "args": {"view": "parameter", "node": "nav-can1", "parameter": "stall_k"}},
        {"op": "trial", "args": {"kind": "tunables", "payload": {
            "node": "nav-can1", "parameter": "stall_k", "to": 20}}, "reason": "measure a changed stall window"},
        {"op": "stop", "args": {"reason": "test complete"}},
    ])
    evolve_llm.llm_propose(ep, proj, before, 1, tmp_path, agent_tools=tools)
    assert len(calls) == 1 and calls[0][0] == "trial"
    detail = calls[0][1]["detail"]
    assert detail["ref"] == proj["drivers"]["nav-can1"]["tunables"]["ref"]
    assert detail["path"] == ["tunables", "stall_k"] and detail["to"] == 20


@pytest.mark.parametrize("extra", [{"parameter": "VCAP"}, {"parameter": None},
                                    {"parameter": "stall_k", "ref": "another.provider"}])
def test_named_parameter_does_not_expand_edit_permission(tmp_path, extra):
    proj, before = recycle_cans_projection()
    proj["policy_id"] = "installed"
    tools, calls = _tools(proj)
    ep = _Endpoint([
        {"op": "inspect", "args": {"view": "parameter", "node": "nav-can1", "parameter": "stall_k"}},
        {"op": "trial", "args": {"kind": "tunables", "payload": {"node": "nav-can1", "to": 20, **extra}}},
        {"op": "stop", "args": {}},
    ])
    _, row = evolve_llm.llm_propose(ep, proj, before, 1, tmp_path, agent_tools=tools)
    assert not calls and row["trial_calls"] == 0
    assert "declared parameters" in ep.requests[-1][0][1]["content"]


def test_parameter_directory_does_not_page_through_every_source_consumer():
    proj, _ = recycle_cans_projection()
    page = evolve_llm.inspect_evidence(proj, {"view": "parameter", "node": "nav-can1"})
    assert page["complete"] and page["next"] is None
    assert page["data"]["effective"] == proj["drivers"]["nav-can1"]["tunables"]["values"]
    assert not page["data"]["references"] and len(encoded(page)) < 2500


@pytest.mark.parametrize("parameter", ["VCAP", "NavigateDriver.VCAP", "self.VCAP"])
def test_installed_class_constant_is_readable_without_becoming_a_writable_tunable(parameter):
    from scripts.evolve import _validate_try

    proj, _ = recycle_cans_projection()
    node = "carry-can1"
    data, _ = _read_all(proj, {"view": "parameter", "node": node, "parameter": parameter})
    assert data["declared_tunable"] is False and data["effective"] is None
    assert data["unresolved"] is False and data["complete_dependency_analysis"] is False
    assert any("VCAP =" in ref["code"] for ref in data["references"])
    assert any("self.VCAP" in ref["code"] for ref in data["references"])
    for ref in data["references"]:
        module = importlib.import_module(ref["module"])
        source = Path(module.__file__).read_text()
        assert ref["source_sha"] == content_id(source)
        assert ref["code"] == "".join(source.splitlines(keepends=True)[ref["start"] - 1:ref["end"]])
    knobs = proj["drivers"][node]["tunables"]
    assert parameter not in knobs["values"]
    tried = {"kind": "tunables", "node": node,
             "detail": {"ref": knobs["ref"], "path": [*knobs["path"], parameter], "to": 0.7}}
    with pytest.raises(ValueError, match="finite installed driver parameter"):
        _validate_try(tried, proj, {}, {})


def test_static_class_attribute_references_do_not_claim_a_runtime_effective_value(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    module = "runtime_attribute_fixture"
    (tmp_path / f"{module}.py").write_text(
        "class Driver:\n"
        "    LIMIT = 3.0\n"
        "    def __init__(self, runtime_value):\n"
        "        self.LIMIT = runtime_value\n"
        "    def act(self):\n"
        "        return self.LIMIT\n")
    proj = {"drivers": {"move": {"modules": [module]}}}
    assert importlib.import_module(module).Driver(9.0).act() == 9.0
    data, _ = _read_all(proj, {"view": "parameter", "node": "move", "parameter": "LIMIT"})
    assert data["declared_tunable"] is False and data["effective"] is None
    assert data["complete_dependency_analysis"] is False
    assert any("LIMIT = 3.0" in ref["code"] for ref in data["references"])
    assert any("self.LIMIT = runtime_value" in ref["code"] for ref in data["references"])
    definitions = data["static_definitions"]
    assert any(item.get("literal_known") and item.get("literal_value") == 3.0 for item in definitions)
    assert any(item["expression"] == "runtime_value" and not item.get("literal_known") for item in definitions)


@pytest.mark.parametrize(("suffix", "expression"), [
    ("set", "{1, 2}"), ("bytes", "b'abc'"), ("complex", "1 + 2j"),
    ("tuple_key", "{(1, 2): 'value'}"),
])
def test_python_literals_without_a_json_value_remain_readable_as_static_source(tmp_path, monkeypatch, suffix, expression):
    monkeypatch.syspath_prepend(str(tmp_path))
    module = f"static_literal_{suffix}_fixture"
    (tmp_path / f"{module}.py").write_text(f"class Driver:\n    VALUE = {expression}\n")
    proj = {"drivers": {"move": {"modules": [module]}}}
    data, _ = _read_all(proj, {"view": "parameter", "node": "move", "parameter": "VALUE"})
    assert data["effective"] is None and data["declared_tunable"] is False
    assert data["unresolved"] is False
    definition, = data["static_definitions"]
    assert definition["expression"] == expression
    assert definition["literal_known"] is False and "literal_value" not in definition
    assert f"VALUE = {expression}" in definition["code"]


def test_parameter_reference_search_does_not_expand_the_nodes_source_authority(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    allowed, unrelated = "allowed_parameter_fixture", "unrelated_parameter_fixture"
    (tmp_path / f"{allowed}.py").write_text("def act():\n    return 0\n")
    (tmp_path / f"{unrelated}.py").write_text("SECRET_LIMIT = 27\n")
    importlib.import_module(unrelated)
    proj = {"drivers": {"move": {"modules": [allowed]}, "other": {"modules": [unrelated]}}}
    data, _ = _read_all(proj, {"view": "parameter", "node": "move", "parameter": "SECRET_LIMIT"})
    assert data["declared_tunable"] is False and data["effective"] is None
    assert data["unresolved"] is True and data["references"] == []
    assert unrelated not in encoded(data).decode()
    with pytest.raises(ValueError, match="module must"):
        evolve_llm.inspect_evidence(proj, {"view": "source", "node": "move", "module": unrelated})


def test_source_pages_keep_exact_text_and_change_identity_when_source_changes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    path = tmp_path / "inspect_source_fixture.py"
    path.write_text("def step():\n    return 1\n")
    proj = {"drivers": {"move": {"modules": ["inspect_source_fixture"]}}}
    args = {"view": "source", "node": "move", "module": "inspect_source_fixture", "symbol": "step"}
    before = evolve_llm.inspect_evidence(proj, args)
    assert before["data"]["code"] == path.read_text()
    path.write_text("def step():\n    return 2\n")
    after = evolve_llm.inspect_evidence(proj, args)
    assert after["data"]["code"] == path.read_text()
    assert before["sha"] != after["sha"]
    assert before["data"]["source_sha"] != after["data"]["source_sha"]


@pytest.mark.parametrize("budget", [{"max_request_bytes": 100}, {"max_input_bytes": 100}])
def test_input_budget_prevents_the_first_api_call_without_silently_dropping_nodes(tmp_path, budget):
    ep = _Endpoint([])
    tried, row = _propose(tmp_path, ep, budget=budget)
    assert tried["kind"] == "none" and tried["detail"]["needs"] == ["budget"]
    assert row["status"] == "abstained" and row["stop_reason"] == "budget_exhausted"
    assert row["calls"] == row["budget"]["used"]["calls"] == 0
    assert ep.requests == []


def test_invalid_answers_spend_calls_and_requests_do_not_accumulate_the_transcript(tmp_path):
    ep = _Endpoint(["invalid JSON"] * 3, usage={"prompt": 120, "completion": 7})
    tried, row = _propose(tmp_path, ep, budget={"max_calls": 3, "max_output_tokens": 37})
    assert tried["kind"] == "none" and row["stop_reason"] == "budget_exhausted"
    assert len(ep.requests) == row["calls"] == 3
    assert row["usage"] == {"prompt": 360, "completion": 21}
    assert row["budget"]["used"]["output_tokens"] == 21
    assert row["budget"]["used"]["input_bytes"] == sum(len(encoded(m)) for m, _ in ep.requests)
    for messages, options in ep.requests:
        assert [m["role"] for m in messages] == ["system", "user"]
        assert options["max_tokens"] == 37
        assert len(encoded(messages)) <= row["budget"]["limits"]["max_request_bytes"]
        assert isinstance(messages[1]["content"], str)  # No automatic image expansion.


def test_chat_failure_is_charged_and_keeps_unknown_provider_usage_unknown(tmp_path):
    ep = _Endpoint([TimeoutError("test transport timeout")])
    tried, row = _propose(tmp_path, ep)
    assert tried["detail"]["needs"] == ["model_endpoint"]
    assert row["status"] == "error" and row["error"]["stage"] == "chat"
    assert row["calls"] == row["budget"]["used"]["calls"] == 1
    assert row["budget"]["used"]["input_bytes"] > 0
    assert row["usage"] is None and row["budget"]["used"]["output_tokens"] is None
    assert not row["budget"]["usage_complete"]


def test_installed_executor_trial_can_use_the_catalog_without_an_extra_read(tmp_path):
    proj, _ = _projection()
    tools, calls = _tools(proj)
    ep = _Endpoint([
        {"op": "trial", "args": {"kind": "executor", "payload": {"node": "grab-0", "to": "alt"}}},
        {"op": "stop", "args": {"reason": "Measured the catalog-selected executor."}},
    ])
    tried, row = _propose(tmp_path, ep, proj=proj, tools=tools)
    assert len(calls) == row["trial_calls"] == 1 and row["evidence_reads"] == 0
    second = json.loads(ep.requests[1][0][1]["content"])
    assert second["last_tool_result"]["data"]["scope"] == "probe"
    assert second["last_tool_result"]["data"]["accepted"] is False
    assert row["status"] == "abstained" and row["stop_reason"] == "model_stop"
    assert tried["kind"] == "none"


def test_same_model_receives_neutral_feedback_and_can_compose_the_same_atom_under_a_new_parent(tmp_path):
    from test_rsi_program_learner import _World

    from plugins.rsi import evaluation

    world = _World()
    learner = world.learner

    def project(overlay, suite):
        return {"task": "conjunction", "round": 1, "applied": copy.deepcopy(overlay),
                "evaluation_contract": world.contract, "experiment_id": suite["experiment_id"],
                "drivers": {"move": {"node": "move", "skill": "move", "executor": "scripted",
                    "modules": ["test_rsi_program_learner"], "tunables": {
                        "ref": "test:controller", "path": ["tunables"], "values": dict(overlay["tunables"])}}},
                "this_round": {"per_seed": [], "count": 0, "seeds_total": len(suite["seeds"])}}

    learner._project = project
    first_id = learner._identity({"tunables": {"first": 1.0, "second": 0.0}})
    both_id = learner._identity({"tunables": {"first": 1.0, "second": 1.0}})

    def trial(name, parent=None):
        return {"op": "trial", "args": {"kind": "tunables", "parent_policy_id": parent,
                "payload": {"node": "move", "ref": "test:controller", "path": ["tunables", name], "to": 1.0}}}

    ep = _Endpoint([
        {"op": "inspect", "args": {"view": "parameter", "node": "move", "parameter": "first"}},
        trial("first"),
        trial("second"),
        {"op": "inspect", "args": {"view": "parameter", "node": "move", "parameter": "second", "policy_id": first_id}},
        trial("second", first_id),
        {"op": "choose", "args": {"policy_id": both_id}},
    ])
    tools = {"trial": learner.trial, "choose": learner.choose, "projection": learner.projection,
             "baseline": lambda policy=None: learner.policies[policy or learner.initial_id]["suite"]}
    tried, row = evolve_llm.llm_propose(ep, learner.projection(), world.baseline, 1, tmp_path,
                                       agent_tools=tools)
    assert row["status"] == "proposed" and row["stop_reason"] == "chosen"
    assert row["calls"] == 6 and row["trial_calls"] == 3
    assert tried["kind"] == "tunables" and tried["detail"]["path"][-1] == "second"
    assert learner.selected_id == both_id
    assert evaluation.compare(world.baseline, learner.selected_suite, world.contract)["accepted"]
    for request in (ep.requests[2], ep.requests[3]):
        feedback = json.loads(request[0][1]["content"])["last_tool_result"]["data"]
        assert feedback["comparison"]["gains"] == []
        assert feedback["behavior"][0]["sampled_trace_equal"] is True
        assert feedback["evaluation"] == feedback["baseline_evaluation"]
        assert feedback["accepted"] is False
    assert [scope for _, _, scope in world.calls] == ["probe", "probe", "probe", "full"]
    audit = json.loads((tmp_path / "round-1.json").read_text())
    assert audit["attempts"] == []
    assert len(audit["requests"]) == 6
    assert row["budget"]["used"]["output_tokens"] is None  # Fixture gives no invented API usage.


def test_inspection_permission_is_scoped_to_the_selected_parent_policy(tmp_path):
    proj, _ = _projection()
    tools, calls = _tools(proj)
    parent = {**proj, "policy_id": "working-parent"}
    tools["projection"] = lambda policy=None: copy.deepcopy(parent if policy == "working-parent" else proj)
    ep = _Endpoint([
        {"op": "inspect", "args": {"view": "source", "symbol": "test_evolve_e2e:_Driver.act"}},
        {"op": "trial", "args": {"kind": "patch", "parent_policy_id": "working-parent",
                                   "payload": {"node": "grab-0", "module": "test_evolve_e2e", "edits": []}}},
        {"op": "stop", "args": {"reason": "Need the changed parent's source first."}},
    ])
    _, row = _propose(tmp_path, ep, proj=proj, tools=tools)
    assert calls == [] and row["trial_calls"] == 0
    assert "requires_inspection" in json.dumps(json.loads(ep.requests[2][0][1]["content"])["last_tool_result"])


def test_cumulative_budget_prevents_the_next_request_before_it_is_sent(tmp_path):
    ep = _Endpoint(["invalid JSON", {"op": "stop", "args": {"reason": "done"}}])
    _, row = _propose(tmp_path / "measured", ep)
    first_bytes = len(encoded(ep.requests[0][0]))
    ep_limited = _Endpoint(["invalid JSON"])
    tried, limited = _propose(tmp_path / "limited", ep_limited,
                              budget={"max_input_bytes": first_bytes + 100})
    assert len(ep_limited.requests) == 1
    assert limited["budget"]["used"]["input_bytes"] <= first_bytes + 100
    assert limited["stop_reason"] == "budget_exhausted" and tried["kind"] == "none"
    assert row["calls"] == 2


def test_full_frozen_evaluator_and_unknown_evidence_remain_available_on_demand():
    from test_rsi_evaluation import contract

    proj, _ = _projection()
    proj["evaluation_contract"] = contract()
    proj["diagnosis"] = {"findings": [{"kind": "unknown", "node": "grab-0", "evidence": {
        "predicate": None, "blocked_reads": ["ctx.episode.driver"], "terminal": False}}]}
    compact = evolve_llm.compact_brief(proj)
    assert compact["evaluation"]["contract_sha"] == proj["evaluation_contract"]["sha"]
    value, _ = _read_all(proj, {"view": "evaluation"})
    assert value["contract"] == proj["evaluation_contract"]
    assert value["diagnosis"] == proj["diagnosis"]


def test_missing_usage_on_one_call_is_not_filled_in_from_later_partial_usage(tmp_path):
    class MixedUsage(_Endpoint):
        def chat(self, messages, **options):
            self.last_usage = None if not self.requests else {"prompt": 12, "completion": 4}
            return super().chat(messages, **options)

    ep = MixedUsage(["invalid JSON", {"op": "stop", "args": {"reason": "insufficient evidence"}}])
    _, row = _propose(tmp_path, ep)
    assert row["calls"] == 2
    assert row["usage"] is None and row["budget"]["used"]["output_tokens"] is None
    assert row["budget"]["usage_complete"] is False


def test_initial_decision_does_not_automatically_expand_images_or_full_source(tmp_path):
    proj, _ = _projection()
    proj["keyframes"] = ["a-keyframe.png"]
    proj["module_sources"] = {"huge": "UNREQUESTED_MODULE" * 20000}
    ep = _Endpoint([{"op": "stop", "args": {"reason": "No warranted intervention."}}])
    ep.images = True
    _, row = _propose(tmp_path, ep, proj=proj)
    assert row["status"] == "abstained" and row["calls"] == 1
    assert "UNREQUESTED_MODULE" not in json.dumps(ep.requests)
    assert all(isinstance(message["content"], str) for message in ep.requests[0][0])
    assert "a-keyframe.png" not in json.dumps(ep.requests)


@pytest.mark.parametrize("output_limit", [123, 4096])
def test_sampling_decisions_disable_extra_thinking_and_keep_the_output_budget(tmp_path, output_limit):
    ep = _Endpoint([{"op": "stop", "args": {"reason": "No supported intervention."}}])
    _, row = _propose(tmp_path, ep, budget={"max_output_tokens": output_limit})
    _, options = ep.requests[0]
    assert options["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in options
    assert "temperature" not in options
    assert options["max_tokens"] == row["budget"]["limits"]["max_output_tokens"] == output_limit


def test_unknown_dynamic_parameter_dependencies_are_reported_as_unresolved(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "dynamic_parameter_fixture.py").write_text(
        "def consume(options, field):\n    return options[field]\n")
    proj = {"drivers": {"move": {"modules": ["dynamic_parameter_fixture"],
                                "tunables": {"values": {"speed": 0.2}}}}}
    item = evolve_llm.inspect_evidence(proj, {"view": "parameter", "node": "move", "parameter": "speed"})
    assert item["data"]["unresolved"] is True
    assert item["data"]["complete_dependency_analysis"] is False
    assert item["data"]["references"] == []


def test_next_model_request_sees_the_new_working_policy_and_spent_probe_budget(tmp_path):
    proj, _ = _projection()
    state = copy.deepcopy(proj)
    state["probe_budget"] = {"limit": 3, "used": 0}
    state["working_policies"] = [{"policy_id": "incumbent", "scope": "incumbent", "parent_id": None}]

    def trial(tried, parent_policy_id=None):
        state["probe_budget"]["used"] += 1
        state["working_policies"].append({"policy_id": "measured-policy", "scope": "probe", "parent_id": "incumbent"})
        return {"policy_id": "measured-policy", "scope": "probe", "accepted": False}

    ep = _Endpoint([
        {"op": "inspect", "args": {"view": "parameter", "node": "grab-0", "parameter": "stall_k"}},
        {"op": "trial", "args": {"kind": "executor", "payload": {"node": "grab-0", "to": "alt"}}},
        {"op": "stop", "args": {"reason": "Keep this neutral measurement for later inspection."}},
    ])
    tools = {"trial": trial, "projection": lambda policy=None: copy.deepcopy(state),
             "choose": lambda policy: pytest.fail("A stop must not select a policy.")}
    _, row = _propose(tmp_path, ep, proj=proj, tools=tools)
    final_state = json.loads(ep.requests[2][0][1]["content"])["state"]
    assert final_state["probe_budget"] == {"limit": 3, "used": 1}
    assert any(p["policy_id"] == "measured-policy" for p in final_state["working_policies"])
    assert row["trial_calls"] == 1 and row["stop_reason"] == "model_stop"


@pytest.mark.parametrize("view", ["node", "source", "parameter", "trace"])
def test_every_node_scoped_inspection_explicitly_reports_the_missing_required_argument(view):
    proj, _ = _projection()
    with pytest.raises(ValueError) as error:
        evolve_llm.inspect_evidence(proj, {"view": view})
    message = str(error.value)
    assert "node" in message and "require" in message
    assert "driver_index" in message


def test_missing_inspection_node_feedback_allows_the_same_model_to_repair_read_and_trial(tmp_path):
    proj, _ = _projection()
    tools, calls = _tools(proj)
    args = {"view": "parameter", "parameter": "stall_k"}
    omitted = {"op": "inspect", "args": args}
    ep = _Endpoint([
        omitted,
        {"op": "inspect", "args": {**args, "node": "grab-0"}},
        {"op": "trial", "args": {"kind": "executor", "payload": {"node": "grab-0", "to": "alt"}}},
        {"op": "stop", "args": {"reason": "The probe is measured; no full evaluation requested."}},
    ])
    _, row = _propose(tmp_path, ep, proj=proj, tools=tools)
    repair_input = json.loads(ep.requests[1][0][1]["content"])
    feedback = repair_input["last_tool_result"]["data"]
    assert "required" in feedback["error"]["message"] and "node" in feedback["error"]["message"]
    assert repair_input["previous_command"] == omitted
    # The corrected command really reads the selected source/parameter page;
    # it is not a textual suggestion accepted without exercising the validator.
    page = json.loads(ep.requests[2][0][1]["content"])["last_tool_result"]
    assert page["node"] == "grab-0" and page["view"] == "parameter"
    assert row["evidence_reads"] == 1 and row["trial_calls"] == 1
    assert len(calls) == 1 and calls[0][0] == "trial"
    assert row["calls"] == 4 and row["stop_reason"] == "model_stop"
    audit = json.loads((tmp_path / "round-1.json").read_text())
    assert len(audit["attempts"]) == 1


def test_inspect_examples_for_every_driver_are_executable_and_never_choose_a_default_target():
    # Keep the model-facing requirement aligned with the validator. A previous
    # prompt called node optional and spent all eight calls on rejected reads.
    assert "Node/trace/parameter require node" in evolve_llm._AGENT_RULES
    assert "<class ID>.<method>" in evolve_llm._AGENT_RULES
    proj, _ = _projection()
    proj["drivers"]["passed-upstream"] = copy.deepcopy(proj["drivers"]["grab-0"])
    compact = evolve_llm.compact_brief(proj)
    for node in compact["driver_index"]:
        example = {"op": "inspect", "args": {"view": "source", "node": node}}
        response = evolve_llm.inspect_evidence(proj, example["args"])
        assert response["node"] == node and response["view"] == "source"
        assert response["data"]["modules"]
    assert "default_node" not in compact and "target" not in compact


def test_repeating_the_same_invalid_tool_arguments_stops_before_wasting_the_remaining_budget(tmp_path):
    proj, _ = _projection()
    tools, calls = _tools(proj)
    ep = _Endpoint([{"op": "inspect", "args": {"view": "parameter", "parameter": "stall_k"},
                     "reason": f"Different explanation {i}", "memo": f"Changing memo {i}"}
                    for i in range(8)])
    tried, row = _propose(tmp_path, ep, proj=proj, tools=tools)
    assert row["calls"] == len(ep.requests) == 3
    assert row["status"] == "abstained" and row["stop_reason"] == "repeated_invalid_command"
    assert tried["kind"] == "none" and tried["detail"]["needs"] == ["tool_protocol"]
    assert calls == [] and row["trial_calls"] == row["evidence_reads"] == 0
    assert row["budget"]["used"]["calls"] == 3


def test_module_only_source_read_reports_all_owners_but_trial_still_requires_the_model_to_select_a_node(tmp_path):
    proj, _ = _projection()
    proj["drivers"]["passed-upstream"] = copy.deepcopy(proj["drivers"]["grab-0"])
    tools, calls = _tools(proj)
    ep = _Endpoint([
        {"op": "inspect", "args": {"view": "source", "module": "test_evolve_e2e", "symbol": "_Driver"}},
        {"op": "trial", "args": {"kind": "executor", "payload": {"to": "alt"}}},
        {"op": "trial", "args": {"kind": "executor", "payload": {"node": "grab-0", "to": "alt"}}},
        {"op": "stop", "args": {"reason": "The explicit probe is sufficient for this fixture."}},
    ])
    _, row = _propose(tmp_path, ep, proj=proj, tools=tools)
    source = json.loads(ep.requests[1][0][1]["content"])["last_tool_result"]
    assert source["node"] is None
    assert set(source["data"]["owner_nodes"]) == {"grab-0", "passed-upstream"}
    assert "class _Driver:" in source["data"]["code"]
    assert row["evidence_reads"] == row["trial_calls"] == 1
    assert len(calls) == 1 and calls[0][0] == "trial" and calls[0][1]["node"] == "grab-0"
    audit = json.loads((tmp_path / "round-1.json").read_text())
    assert len(audit["attempts"]) == 1
    assert row["stop_reason"] == "model_stop"


def test_module_only_source_cannot_read_a_module_outside_current_policy_authority():
    proj, _ = _projection()
    with pytest.raises(ValueError) as error:
        evolve_llm.inspect_evidence(proj, {"view": "source", "module": "os"})
    assert "module" in str(error.value)
    assert "test_evolve_e2e" in str(error.value)


def test_source_directory_is_distinct_from_actual_code_and_cannot_authorize_a_patch(tmp_path):
    proj, _ = _projection()
    tools, calls = _tools(proj)
    ep = _Endpoint([
        {"op": "inspect", "args": {"view": "source", "module": "test_evolve_e2e"}},
        {"op": "trial", "args": {"kind": "patch", "payload": {"node": "grab-0", "module": "test_evolve_e2e", "edits": []}}},
        {"op": "inspect", "args": {"view": "source", "symbol": "test_evolve_e2e:_Driver.act"}},
        {"op": "stop", "args": {"reason": "Actual source is now available."}},
    ])
    _, row = _propose(tmp_path, ep, proj=proj, tools=tools)
    metadata = json.loads(ep.requests[1][0][1]["content"])["last_tool_result"]
    assert metadata["code_read"] is False and metadata["data"]["code_read"] is False
    assert "code" not in metadata["data"] and metadata["data"]["symbols"]
    rejection = json.loads(ep.requests[2][0][1]["content"])["last_tool_result"]
    assert "requires_inspection" in rejection["data"]["error"]["message"]
    method = json.loads(ep.requests[3][0][1]["content"])["last_tool_result"]
    assert method["code_read"] is True and "def act(self, obs):" in method["data"]["code"]
    assert method["data"]["symbol"] == "test_evolve_e2e:_Driver.act"
    assert not calls and row["trial_calls"] == 0 and row["evidence_reads"] == 2
    assert row["stop_reason"] == "model_stop"


def test_source_code_pages_preserve_utf8_verbatim_without_nested_json_fragments(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    path = tmp_path / "unicode_source_fixture.py"
    code = "def act():\n" + "    # 机器人🙂\\n quote=\"保持原文\"\n" * 200 + "    return 1\n"
    path.write_text(code)
    proj = {"drivers": {"move": {"source_ref": "unicode_source_fixture:act",
                                "modules": ["unicode_source_fixture"]}}}
    result, pages = _read_all(proj, {"view": "source", "symbol": "unicode_source_fixture:act"}, cap=1200)
    assert len(pages) > 1 and result["code"] == code
    assert result["source_sha"] == content_id(code)
    assert all(p["data"].get("encoding") != "json_fragment" for p in pages)
    assert all(p["data"]["code_read"] and p["code_read"] for p in pages)
    assert all(p["data"]["source_sha"] == content_id(code) for p in pages)


def test_source_index_comes_from_actual_stage_bindings_and_each_method_is_readable():
    import tomllib

    from test_evolve_patch_e2e import _CARD, EMB, INSTALLED, MODULE, TASK, _two_death_proj

    from scripts import harness_runtime as hr

    binding = tomllib.loads(_CARD)["task_bindings"][TASK]
    _, before = _two_death_proj()
    for episode in before["seeds"].values():
        episode["trail"] = [{"id": node, "kind": "segment", "ok": False, "steps": 1}
                            for node in episode["nodes"]]
    proj = evolve_llm.rsi_projection(
        {"task": TASK, "seeds": [1, 2], "cursor": 0, "rounds": [], "applied": {}}, before,
        hr._binding_records(binding), EMB, "scripted", binding, [])
    compact = evolve_llm.compact_brief(proj)
    assert compact["driver_index"]["reach-0"]["bound_classes"] == [f"{MODULE}:ReachStage"]
    assert compact["driver_index"]["grab-0"]["bound_classes"] == [f"{MODULE}:GrabStage"]
    assert set(compact["source_index"]) == {f"{MODULE}:ReachStage", f"{MODULE}:GrabStage"}
    source_lines = INSTALLED.read_text().splitlines(keepends=True)
    for cls in compact["source_index"].values():
        for method in cls["methods"]:
            exact, _ = _read_all(proj, {"view": "source", "symbol": method["id"]}, cap=1600)
            expected = "".join(source_lines[method["start"] - 1:method["end"]])
            assert exact["code"] == expected and exact["symbol"] == method["id"]
            ranged, _ = _read_all(proj, {"view": "source", "module": MODULE,
                                       "start": method["start"], "end": method["end"]}, cap=1600)
            assert ranged["code"] == expected


def _visible_evidence(body):
    latest = body.get("last_tool_result")
    return [*body["retained_evidence"], *([latest] if latest is not None else [])]


def _page_key(page):
    return tuple(page.get(key) for key in ("policy_id", "view", "node", "sha", "cursor", "next"))


def _working_set_bytes(body):
    return len(encoded({key: body[key] for key in ("last_tool_result", "retained_evidence")}))


def test_related_methods_remain_together_and_rereading_a_page_does_not_grow_the_working_set(tmp_path):
    proj, _ = _projection()
    methods = ["test_evolve_e2e:_Driver.act", "test_evolve_e2e:_Driver.segment_success"]
    reads = [{"op": "inspect", "args": {"view": "source", "symbol": symbol}} for symbol in methods]
    ep = _Endpoint([*reads, reads[0], {"op": "stop", "args": {"reason": "Both related methods are now visible."}}])
    # This tests cache identity under rereads, with an explicit larger read budget.
    _, row = _propose(tmp_path, ep, proj=proj, budget={"max_read_calls": 3})
    bodies = [json.loads(messages[1]["content"]) for messages, _ in ep.requests]
    before_repeat, after_repeat = (_visible_evidence(bodies[index]) for index in (2, 3))
    assert len(before_repeat) == len(after_repeat) == 2
    assert {_page_key(p) for p in before_repeat} == {_page_key(p) for p in after_repeat}
    assert len(encoded(before_repeat)) == len(encoded(after_repeat))
    assert {p["data"]["symbol"] for p in after_repeat} == set(methods)
    expected = {symbol: evolve_llm.inspect_evidence(proj, {"view": "source", "symbol": symbol})
                for symbol in methods}
    for page in after_repeat:
        assert page == expected[page["data"]["symbol"]]
    assert row["evidence_reads"] == 3 and len(row["evidence_refs"]) == 2
    for body, (messages, _) in zip(bodies, ep.requests, strict=True):
        assert _working_set_bytes(body) <= row["budget"]["limits"]["max_working_set_bytes"]
        assert len(encoded(messages)) <= row["budget"]["limits"]["max_request_bytes"]


def test_latest_probe_measurement_survives_a_full_cache_and_further_source_reads(tmp_path):
    proj, _ = _projection()
    tools, calls = _tools(proj)
    receipt = {"policy_id": "measured-probe", "scope": "probe", "accepted": False,
               "measurement_sha": content_id({"seed": 1, "terminal": False}),
               "evaluation": {"successes": 0, "episodes": 1, "progress": 0.0},
               "comparison": {"accepted": False, "gains": [], "regressions": []}}

    def measured_trial(tried, parent_policy_id=None):
        calls.append(("trial", tried, parent_policy_id))
        return copy.deepcopy(receipt)

    tools["trial"] = measured_trial
    original_projection = tools["projection"]
    tools["projection"] = lambda policy=None: (
        {**copy.deepcopy(proj), "policy_id": policy} if policy == receipt["policy_id"]
        else original_projection(policy))
    read = lambda method: {"op": "inspect", "args": {"view": "source", "symbol": f"test_evolve_e2e:_Driver.{method}"}}
    ep = _Endpoint([
        read("act"), read("segment_success"),
        {"op": "trial", "args": {"kind": "executor", "payload": {"node": "grab-0", "to": "alt"}}},
        read("exhausted"), read("on_handback"),
        {"op": "stop", "args": {"reason": "The neutral measured feedback remains available."}},
    ])
    _, row = _propose(tmp_path, ep, proj=proj, tools=tools,
                       budget={"max_working_set_bytes": 1800})
    assert row["calls"] == 6 and row["trial_calls"] == 1 and len(calls) == 1
    bodies = [json.loads(messages[1]["content"]) for messages, _ in ep.requests]
    assert bodies[3]["last_tool_result"]["view"] == "trial"
    for body in bodies[3:]:
        pages = _visible_evidence(body)
        trial_page, = [page for page in pages if page["view"] == "trial"]
        assert trial_page["data"] == receipt
        assert trial_page["sha"] == content_id(receipt)
        assert _working_set_bytes(body) <= 1800
    final_pages = _visible_evidence(bodies[-1])
    assert not any(page.get("data", {}).get("symbol") == "test_evolve_e2e:_Driver.act" for page in final_pages)
    for messages, _ in ep.requests:
        assert len(encoded(messages)) <= row["budget"]["limits"]["max_request_bytes"]


def test_source_catalog_is_addressable_and_refreshes_when_the_binding_index_changes(tmp_path):
    from test_evolve_e2e import _AltExecutor, _Driver

    from scripts.evolve_evidence import bound_class_index

    proj, _ = _projection()
    proj["drivers"]["grab-0"]["bound_classes"] = bound_class_index([_Driver])
    changed = copy.deepcopy(proj)
    changed["policy_id"] = "changed-binding"
    changed["drivers"]["grab-0"]["bound_classes"] = bound_class_index([_AltExecutor])
    tools, _ = _tools(proj)
    tools["projection"] = lambda policy=None: copy.deepcopy(changed if policy == "changed-binding" else proj)
    ep = _Endpoint([
        {"op": "inspect", "args": {"view": "source", "symbol": "test_evolve_e2e:_Driver.act"}},
        {"op": "inspect", "args": {"view": "source", "symbol": "test_evolve_e2e:_AltExecutor.act", "policy_id": "changed-binding"}},
        {"op": "stop", "args": {"reason": "The changed binding has its own delivered source index."}},
    ])
    _, row = _propose(tmp_path, ep, proj=proj, tools=tools)
    states = [json.loads(messages[1]["content"])["state"] for messages, _ in ep.requests]
    first, slim, fresh = states
    assert first["source_index_sha"] == slim["source_index_sha"]
    assert first["catalog_ref"] == slim["catalog_ref"]
    assert fresh["source_index_sha"] != first["source_index_sha"]
    assert fresh["catalog_ref"]["sha"] != first["catalog_ref"]["sha"]
    for state, projection in ((first, proj), (fresh, changed)):
        assert "source_index" not in state
        catalog, _ = _read_all(projection, state["catalog_ref"])
        assert content_id(catalog) == state["catalog_ref"]["sha"]
        for entry in catalog["source_index"].values():
            for method in entry["methods"]:
                page = evolve_llm.inspect_evidence(projection, {"view": "source", "symbol": method["id"]})
                assert page["code_read"] is True
    assert row["evidence_reads"] == 2 and row["stop_reason"] == "model_stop"


@pytest.mark.parametrize("trial", [False, True])
def test_rejected_evidence_page_preserves_working_set_and_latest_reward(trial):
    from scripts.evolve_evidence import EvidenceWorkingSet

    working = EvidenceWorkingSet(800)
    reward = {"view": "trial", "sha": "reward", "data": {"gain": 1}}
    source = {"view": "source", "sha": "source", "data": "exact source"}
    working.add(reward, trial=True)
    working.add(source)
    before = working.snapshot()
    with pytest.raises(ValueError, match="working-set budget"):
        working.add({"view": "trial" if trial else "source", "sha": "oversize",
                     "data": "x" * 800}, trial=trial)
    assert working.snapshot() == before
    assert working.pages[working.trial] == reward
    working.add({"view": "history", "sha": "retry", "data": "small retry"})
    assert reward in _visible_evidence(working.snapshot())
    assert _working_set_bytes(working.snapshot()) <= 800


def test_equal_source_in_distinct_policies_keeps_its_policy_identity_in_the_working_set():
    from scripts.evolve_evidence import EvidenceWorkingSet

    proj, _ = _projection()
    other = {**proj, "policy_id": "another-policy"}
    args = {"view": "source", "symbol": "test_evolve_e2e:_Driver.act"}
    first = evolve_llm.inspect_evidence(proj, args)
    second = evolve_llm.inspect_evidence(other, args)
    assert first["sha"] == second["sha"] and first["data"] == second["data"]
    working = EvidenceWorkingSet(8000)
    for page in (first, second, first):
        working.add(page)
    visible = _visible_evidence(working.snapshot())
    assert len(visible) == 2
    assert {page["policy_id"] for page in visible} == {proj["policy_id"], other["policy_id"]}
    assert working.snapshot()["last_tool_result"] == first
    assert _working_set_bytes(working.snapshot()) <= 8000


@pytest.mark.parametrize("outcome", ["measured", "cached", "error"])
def test_read_budget_bounds_acquisition_and_only_new_measurements_replenish_it(tmp_path, outcome):
    proj, _ = _projection()
    tools, calls = _tools(proj)
    original_trial = tools["trial"]

    def trial(candidate, parent_policy_id=None):
        if outcome == "error":
            raise RuntimeError("candidate evaluation failed")
        receipt = original_trial(candidate, parent_policy_id)
        return {**receipt, "cached": outcome == "cached"}

    tools["trial"] = trial
    ep = _Endpoint([
        {"op": "inspect", "args": {"requests": [
            {"view": "source", "symbol": "test_evolve_e2e:_Driver.act"},
            {"view": "source", "symbol": "test_evolve_e2e:_Driver.segment_success"}]}},
        {"op": "inspect", "args": {"view": "parameter", "node": "grab-0"}},
        {"op": "inspect", "args": {"view": "trace", "node": "grab-0"}},
        {"op": "trial", "args": {"kind": "executor", "payload": {"node": "grab-0", "to": "alt"}}},
        {"op": "stop", "args": {"reason": "No further evidence requested."}},
    ])
    _, row = _propose(tmp_path, ep, proj=proj, tools=tools)
    bodies = [json.loads(messages[1]["content"]) for messages, _ in ep.requests]
    assert [body["read_calls_left"] for body in bodies] == [2, 1, 0, 0, 2 if outcome == "measured" else 0]
    assert "read budget exhausted" in bodies[3]["last_tool_result"]["data"]["error"]["message"]
    assert row["evidence_reads"] == 3  # Two batched pages + one directory, no third paid read.
    assert row["trial_calls"] == 1 and len(calls) == (0 if outcome == "error" else 1)
    assert row["calls"] == 5 and row["stop_reason"] == "model_stop"


def test_symbol_continuation_does_not_silently_ignore_a_mistaken_line_argument():
    proj, _ = recycle_cans_projection()
    read = {"view": "source", "symbol": "plugins.embodiment_robocasa.drivers:NavigateDriver._act"}
    first = evolve_llm.inspect_evidence(proj, read, max_bytes=1200)
    assert first["next"] is not None and first["next"] > 0
    with pytest.raises(ValueError, match="cursor=next"):
        evolve_llm.inspect_evidence(proj, {**read, "start": first["next"]})
    second = evolve_llm.inspect_evidence(proj, {**read, "cursor": first["next"]}, max_bytes=1200)
    whole = evolve_llm.inspect_evidence(proj, read)
    assert first["sha"] == second["sha"] == whole["sha"]
    joined = first["data"]["code"] + second["data"]["code"]
    assert joined == whole["data"]["code"][:len(joined)]
