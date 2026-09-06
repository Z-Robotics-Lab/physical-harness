"""Model-visible observations stay scoped and measured across the full tool loop.

The explicit endpoint replies below are fixtures, never a proposal fallback.
ProgramLearner, frozen evaluation, evidence paging, and command handling are real.
"""

from __future__ import annotations

import copy
import importlib
import json

import pytest
from test_evolve_agent_tools import _Endpoint, _read_all, _visible_evidence, _working_set_bytes
from test_rsi_diagnosis import trace
from test_rsi_program_learner import _candidate, _World

from harness.protocol import content_id
from plugins.rsi import evaluation
from plugins.rsi.diagnosis import analyze_trace
from plugins.rsi.learner import ProgramLearner
from scripts import evolve_llm
from scripts.evolve_evidence import bound_class_index, encoded


def _world(*, epoch="current-source", task="conjunction", replay=None, sampled_motion=False):
    world = _World()
    world.contract = evaluation.compile_contract({"nodes": []}, task=task, terminal_ref="test:world")
    original = world.measure

    def measure(seeds, overlay):
        suite = original(seeds, overlay)
        if sampled_motion:
            distance = overlay["tunables"]["first"] + 2 * overlay["tunables"]["second"]
            for row in suite["seeds"].values():
                row["nodes"]["move"]["trace"] = trace([(i * 0.03 * distance, 0) for i in range(5)])
        suite.update(experiment_id=epoch, sha=content_id(suite["seeds"]))
        return suite

    world.measure = measure
    world.baseline = measure([11, 12], world.applied)

    def project(overlay, suite):
        observed = [{"seed": int(seed), "success": row["evaluation"]["complete"],
                     "evaluation": copy.deepcopy(row["evaluation"]),
                     "trail": [{"id": node, **copy.deepcopy(value)} for node, value in row["nodes"].items()]}
                    for seed, row in suite["seeds"].items()]
        projection = {"task": task, "round": 1, "experiment_id": suite["experiment_id"],
                "evaluation_contract": world.contract, "applied": copy.deepcopy(overlay),
                "development_seeds": [11, 12], "learning_replay": copy.deepcopy(replay or []),
                "drivers": {"move": {"node": "move", "skill": "move", "executor": "scripted",
                    "source_ref": "test_rsi_program_learner:_World", "modules": ["test_rsi_program_learner"],
                    "tunables": {"ref": "test:controller", "path": ["tunables"],
                                 "values": dict(overlay["tunables"])}}},
                "this_round": {"per_seed": observed, "count": sum(r["success"] for r in observed),
                               "seeds_total": len(observed)}}
        if sampled_motion:
            projection["diagnosis"] = {"traces": [{"seed": seed, "node": "move",
                **analyze_trace(row["nodes"]["move"]["trace"])} for seed, row in suite["seeds"].items()]}
        return projection

    world.learner = ProgramLearner(applied=world.applied, baseline=world.baseline,
        contract=world.contract, seeds=[11, 12], project=project, validate=world.validate,
        apply=world.apply, run=world.run)
    return world


def _loop(tmp_path, world, replies, *, budget=None):
    learner = world.learner
    endpoint = _Endpoint(replies, usage={"prompt": 10, "completion": 5})
    tried, row = evolve_llm.llm_propose(endpoint, learner.projection(), world.baseline, 1, tmp_path,
        agent_tools={"trial": learner.trial, "choose": learner.choose, "projection": learner.projection,
                     "baseline": learner.observations}, budget=budget)
    bodies = [json.loads(messages[1]["content"]) for messages, _ in endpoint.requests]
    return tried, row, bodies, endpoint


def _trial(parameter, parent=None):
    return {"op": "trial", "args": {"kind": "tunables", "parent_policy_id": parent,
            "payload": {"node": "move", "parameter": parameter, "to": 1.0}}}


STOP = {"op": "stop", "args": {"reason": "The test decision is complete."}}


@pytest.mark.parametrize("pressure", ["last_call", "oversized_catalog", "cumulative_bytes"])
def test_budget_closure_lets_model_select_a_measured_combination_instead_of_losing_it(tmp_path, monkeypatch, pressure):
    world = _world()
    learner = world.learner
    first = learner._identity({"tunables": {"first": 1.0, "second": 0.0}})
    combined = learner._identity({"tunables": {"first": 1.0, "second": 1.0}})
    if pressure != "last_call":
        compact = evolve_llm.compact_brief

        def enlarged_catalog(proj):
            state = compact(proj)
            if proj.get("policy_id") == combined:
                state["driver_index"]["node-" * 7600] = {"skill": "move", "executor": "scripted"}
            return state

        monkeypatch.setattr(evolve_llm, "compact_brief", enlarged_catalog)

    class Endpoint(_Endpoint):
        def chat(self, messages, **options):
            body = json.loads(messages[1]["content"])
            if body.get("phase") == "selection":
                self.replies = [{"op": "choose", "args": {"policy_id": combined}}]
            return super().chat(messages, **options)

    # The endpoint keeps investigating unless the remaining decision is explicit.
    # Its policy choice is a fixture, never a framework intervention or fallback.
    endpoint = Endpoint([_trial("first"), _trial("second", first),
                         {"op": "inspect", "args": {"view": "history"}}])
    budget = ({"max_calls": 3} if pressure == "last_call" else
              {"max_input_bytes": 60000, "max_request_bytes": 60000} if pressure == "cumulative_bytes" else {})
    tried, row = evolve_llm.llm_propose(endpoint, learner.projection(), world.baseline, 1, tmp_path,
        agent_tools={"trial": learner.trial, "choose": learner.choose, "projection": learner.projection,
                     "baseline": learner.observations}, budget=budget)
    assert row["stop_reason"] == "chosen" and tried["kind"] == "tunables"
    assert evaluation.compare(world.baseline, learner.selected_suite, world.contract)["accepted"]
    assert learner.selected_id == combined and learner.full_calls == 1
    assert len(endpoint.requests) == row["calls"] == 3
    final = json.loads(endpoint.requests[-1][0][1]["content"])
    assert final["allowed_ops"] == ["choose", "stop"]
    assert "driver_index" not in final["state"] and "observations" not in final["state"]
    assert row["budget"]["used"]["input_bytes"] <= row["budget"]["limits"]["max_input_bytes"]


@pytest.mark.parametrize("reply", [STOP, {"op": "trial", "args": {"kind": "tunables",
                                   "payload": {"node": "move", "parameter": "second", "to": 1}}}])
def test_budget_closure_never_automatically_selects_or_spends_an_extra_probe(tmp_path, reply):
    world = _world()
    tried, row, bodies, _ = _loop(tmp_path, world, [_trial("first"), reply], budget={"max_calls": 2})
    assert bodies[-1]["phase"] == "selection"
    assert tried["kind"] == "none" and world.learner.full_calls == 0
    assert len(world.learner.probes) == 1
    assert row["calls"] == 2


def test_selection_preserves_invalid_choice_feedback_for_model_repair(tmp_path, monkeypatch):
    world = _world()
    learner = world.learner
    policy = learner._identity({"tunables": {"first": 1.0, "second": 0.0}})
    compact = evolve_llm.compact_brief

    def large_catalog(proj):
        state = compact(proj)
        if proj["policy_id"] == policy:
            state["driver_index"]["node-" * 7600] = {"skill": "move", "executor": "scripted"}
        return state

    monkeypatch.setattr(evolve_llm, "compact_brief", large_catalog)
    _, row, bodies, _ = _loop(tmp_path, world, [_trial("first"),
        {"op": "choose", "args": {"policy_id": "invented"}},
        {"op": "choose", "args": {"policy_id": policy}}])
    assert bodies[1]["phase"] == bodies[2]["phase"] == "selection"
    error = bodies[2]["last_tool_result"]["data"]
    assert "unknown policy_id" in error["error"]["message"]
    assert error["previous_command"]["args"]["policy_id"] == "invented"
    assert row["stop_reason"] == "chosen" and learner.full_calls == 1


def test_proposer_requires_measured_execution_callbacks_before_loading_a_model():
    with pytest.raises(ValueError, match="agent_tools is required"):
        evolve_llm.llm_propose()


def test_large_sampled_motion_keeps_reward_and_allows_explicit_full_selection(tmp_path, monkeypatch):
    world = _world()
    learner = world.learner
    candidate = _candidate("first")
    candidate["detail"]["to"] = 1.0
    policy = learner._identity(world.apply(candidate, world.applied))
    original = evolve_llm.compact_brief

    def large_projection(proj):
        body = original(proj)
        if proj["policy_id"] != learner.initial_id:
            body["observations"][0]["nodes"][0]["motion"] = {"samples": [0.125] * 20000}
        return body

    monkeypatch.setattr(evolve_llm, "compact_brief", large_projection)
    tried, row, bodies, endpoint = _loop(tmp_path, world, [
        _trial("first"), {"op": "choose", "args": {"policy_id": policy}}])
    assert tried["kind"] == "tunables" and row["stop_reason"] == "chosen"
    assert learner.full_calls == 1 and learner.selected_id == policy
    assert bodies[1]["state"]["policy_id"] == policy
    assert bodies[1]["state"]["probe_budget"] == {"limit": 3, "used": 1}
    assert bodies[1]["last_tool_result"]["data"]["comparison"] == learner.probes[0]["comparison"]
    assert bodies[1]["state"]["observations"][0]["nodes"][0]["motion_ref"]["policy_id"] == policy
    assert all(len(encoded(messages)) <= 24000 for messages, _ in endpoint.requests)


def test_request_compaction_preserves_cache_and_never_hides_irreducible_overflow():
    from scripts.evolve_evidence import request_messages

    body = {"state": {"policy_id": "probe", "observations": []},
            "last_tool_result": {"view": "trial", "data": {"comparison": {"gains": ["fixed"]}}},
            "retained_evidence": [{"view": "source", "policy_id": "base", "sha": "original",
                "cursor": 0, "data": {"symbol": "fixture:Driver.act", "code": "x" * 30000}}]}
    unchanged = copy.deepcopy(body)
    compact, messages = request_messages(body, "protocol", 2000)
    assert len(encoded(messages)) <= 2000 and body == unchanged
    assert compact["last_tool_result"] == body["last_tool_result"]
    assert compact["retained_evidence"][0]["data"]["read"]["symbol"] == "fixture:Driver.act"
    assert compact["retained_evidence"][0]["sha"] == "original"
    _, messages = request_messages(body, "protocol", 1)
    assert len(encoded(messages)) > 1  # The caller must reject; never truncate valid JSON/reward.


def test_choose_incumbent_after_probe_stops_without_full_evaluation_or_promotion(tmp_path):
    world = _world()
    incumbent = world.learner.initial_id
    tried, row, bodies, _ = _loop(tmp_path, world, [
        _trial("first"),
        {"op": "choose", "args": {"policy_id": incumbent}, "reason": "Retain the measured baseline."},
    ])
    assert bodies[-1]["state"]["policy_id"] != incumbent
    assert tried["kind"] == "none" and row["stop_reason"] == "model_stop"
    assert row["calls"] == 2 and row["reason"] == "model retained the incumbent"
    assert len(world.learner.probes) == 1 and world.learner.full_calls == 0
    assert world.learner.selected_id is None and world.learner.selected_overlay is None
    audit = json.loads((tmp_path / "round-1.json").read_text())
    assert audit["attempts"] == []
    assert audit["events"][-1]["command"]["op"] == "choose"


def test_unknown_choice_is_not_treated_as_retaining_incumbent(tmp_path):
    world = _world()
    _, row, bodies, _ = _loop(tmp_path, world, [
        {"op": "choose", "args": {"policy_id": "not-a-measured-policy"}}, STOP])
    error = bodies[-1]["last_tool_result"]["data"]["error"]
    assert "unknown policy_id" in error["message"]
    assert row["calls"] == 2 and world.calls == []
    assert world.learner.selected_id is None


def test_simulator_failure_refreshes_spent_probe_budget_without_resetting_reads(tmp_path):
    world = _world()

    def fail(*args):
        raise RuntimeError("simulator failed after the probe was reserved")

    world.learner._run = fail
    _, row, bodies, _ = _loop(tmp_path, world, [
        {"op": "inspect", "args": {"view": "parameter", "node": "move"}},
        _trial("first"), STOP])
    assert bodies[1]["state"]["probe_budget"] == {"limit": 3, "used": 0}
    assert bodies[2]["state"]["probe_budget"] == {"limit": 3, "used": 1}
    assert bodies[2]["read_calls_left"] == bodies[1]["read_calls_left"] == 1
    assert bodies[2]["last_tool_result"]["data"]["error"]["type"] == "RuntimeError"
    assert row["trial_calls"] == 1 and len(world.learner.probes) == 1
    assert world.learner.full_calls == 0 and world.learner.selected_id is None


def test_readonly_batch_shares_one_byte_budget_and_keeps_each_pages_cursor_and_identity(tmp_path):
    world = _world()
    world.baseline["seeds"]["11"]["nodes"]["move"]["trace"] = {"series": ["证据🙂" * 4000]}
    requests = [{"view": "source", "symbol": "test_rsi_program_learner:_World.measure", "cursor": 17},
                {"view": "trace", "node": "move", "seed": 11, "cursor": 7}]
    budget = {"max_tool_bytes": 2800, "max_working_set_bytes": 3000,
              "max_request_bytes": 14000, "max_input_bytes": 28000}
    _, row, bodies, endpoint = _loop(tmp_path, world,
        [{"op": "inspect", "args": {"requests": requests}}, STOP], budget=budget)
    pages = _visible_evidence(bodies[1])
    assert row["calls"] == 2 and row["evidence_reads"] == 2 and row["trial_calls"] == 0
    assert len(pages) == 2 and {page["cursor"] for page in pages} == {7, 17}
    assert sum(len(encoded(page)) for page in pages) == row["budget"]["used"]["tool_bytes"] <= 2800
    for request in requests:
        expected = evolve_llm.inspect_evidence(world.learner.projection(), request)
        actual, = [page for page in pages if page["view"] == request["view"]]
        assert actual["sha"] == expected["sha"] and actual["policy_id"] == world.learner.initial_id
    for body, (messages, _) in zip(bodies, endpoint.requests, strict=True):
        assert _working_set_bytes(body) <= 3000
        assert len(encoded(messages)) <= 14000
    assert row["budget"]["used"]["input_bytes"] == sum(len(encoded(m)) for m, _ in endpoint.requests) <= 28000
    assert world.calls == []


@pytest.mark.parametrize("invalid", [
    {"view": "source", "module": "os", "start": 1},
    {"view": "trial", "kind": "tunables", "payload": {"node": "move", "parameter": "first", "to": 1}},
    {"view": "choose", "policy_id": "invented"},
])
def test_readonly_batch_cannot_expand_source_authority_or_execute_a_mutation(tmp_path, invalid):
    world = _world()
    before = copy.deepcopy(world.learner.policies)
    _, row, bodies, _ = _loop(tmp_path, world, [{"op": "inspect", "args": {"requests": [
        {"view": "parameter", "node": "move"}, invalid]}}, STOP])
    assert bodies[1]["last_tool_result"]["view"] == "error"
    assert row["evidence_reads"] == 1 and row["trial_calls"] == 0
    assert world.calls == [] and world.learner.policies == before
    assert world.learner.selected_id is None


def test_batch_rejects_more_than_four_reads_before_resolving_any_evidence(tmp_path):
    world = _world()
    reads = [{"view": "parameter", "node": "move"}] * 5
    _, row, bodies, _ = _loop(tmp_path, world, [{"op": "inspect", "args": {"requests": reads}}, STOP])
    assert row["evidence_reads"] == row["trial_calls"] == 0
    assert "1 to 4" in bodies[1]["last_tool_result"]["data"]["error"]["message"]
    assert world.calls == []


@pytest.mark.parametrize("evaluator", ["same", "different", "missing"])
def test_historical_probe_is_readable_but_cannot_select_or_parent_a_current_policy(tmp_path, evaluator):
    old = _world(epoch="old-source")
    probe = old.learner.trial(_candidate("first"))
    replay = [{**copy.deepcopy(probe), "round": n, "task": "conjunction",
               "contract_sha": None if evaluator == "missing" else old.contract["sha"]}
              for n in range(1, 9)]
    current = _world(epoch="new-source", task="changed-evaluator" if evaluator == "different" else "conjunction",
                     replay=replay)
    _, row, bodies, _ = _loop(tmp_path, current, [
        {"op": "inspect", "args": {"view": "history"}},
        {"op": "choose", "args": {"policy_id": probe["policy_id"]}},
        _trial("second", probe["policy_id"]), STOP])
    history = bodies[0]["state"]["historical_probe_replay"]
    assert [record["round"] for record in history] == list(range(3, 9))
    assert all(record["transfer_only"] is (evaluator != "same") for record in history)
    assert all(record["measurement_sha"] == probe["measurement_sha"] for record in history)
    assert all(record["policy_id"] == probe["policy_id"] for record in history)
    assert all(record["seeds"] == [11] for record in history)
    for body in bodies:
        assert probe["policy_id"] not in {p["policy_id"] for p in body["state"]["working_policies"]}
    for body in bodies[2:]:
        assert "unknown policy_id" in body["last_tool_result"]["data"]["error"]["message"]
    assert row["trial_calls"] == 0 and current.calls == []
    assert current.learner.selected_id is None
    reread, _ = _read_all(current.learner.projection(), {"view": "history"})
    assert reread["learning_replay"] == replay


def test_batch_read_then_neutral_probes_can_compose_using_the_observed_parent(tmp_path):
    world = _world()
    learner = world.learner
    first_id = learner._identity({"tunables": {"first": 1.0, "second": 0.0}})
    both_id = learner._identity({"tunables": {"first": 1.0, "second": 1.0}})
    reads = [{"view": "source", "symbol": f"test_rsi_program_learner:_World.{method}"}
             for method in ("apply", "measure")]
    _, row, bodies, _ = _loop(tmp_path, world, [{"op": "inspect", "args": {"requests": reads}},
        _trial("first"), _trial("second"), _trial("second", first_id),
        {"op": "choose", "args": {"policy_id": both_id}}])
    assert row["calls"] == 5 and row["evidence_reads"] == 2 and row["trial_calls"] == 3
    assert {page["data"]["symbol"] for page in _visible_evidence(bodies[1])} == {read["symbol"] for read in reads}
    for body in bodies[2:4]:
        feedback = body["last_tool_result"]["data"]
        assert feedback["scope"] == "probe" and feedback["accepted"] is False
        assert feedback["evaluation"] == feedback["baseline_evaluation"]
        assert feedback["comparison"]["gains"] == [] and feedback["seeds"] == [11]
        assert feedback["measurement_sha"]
        assert feedback["comparison_to"] == learner.initial_id
    for body, used in zip(bodies[2:], (1, 2, 3), strict=True):
        assert body["state"]["probe_budget"] == {"limit": 3, "used": used}
    assert first_id in {p["policy_id"] for p in bodies[3]["state"]["working_policies"]}
    combined = bodies[4]["last_tool_result"]["data"]
    assert combined["parent_id"] == first_id and combined["accepted"] is False
    assert combined["comparison"]["gains"]
    assert learner.selected_id == both_id
    assert learner.selected_overlay == {"tunables": {"first": 1.0, "second": 1.0}}
    assert [scope for _, _, scope in world.calls] == ["probe", "probe", "probe", "full"]
    assert world.calls[-1][0] == [11, 12]
    assert evaluation.compare(world.baseline, learner.selected_suite, world.contract)["accepted"]
    assert world.applied == {"tunables": {"first": 0.0, "second": 0.0}}


def test_trial_refreshes_homepage_to_the_measured_candidate_without_changing_the_implicit_parent(tmp_path):
    world = _world(sampled_motion=True)
    learner = world.learner
    first_id = learner._identity({"tunables": {"first": 1.0, "second": 0.0}})
    _, row, bodies, _ = _loop(tmp_path, world,
        [_trial("first"), _trial("second", first_id), _trial("second"), STOP])
    initial = bodies[0]["state"]
    assert initial["policy_id"] == initial["incumbent_policy_id"] == learner.initial_id
    assert initial["evaluation"]["successes"] == 0 and initial["evaluation"]["episodes"] == 2
    for body in bodies[1:]:
        receipt, state = body["last_tool_result"]["data"], body["state"]
        assert state["policy_id"] == receipt["policy_id"]
        assert state["incumbent_policy_id"] == learner.initial_id
        expected = evolve_llm.compact_brief(learner.projection(receipt["policy_id"]))
        assert state["observations"] == expected["observations"]
        assert state["evaluation"] == expected["evaluation"]
        assert state["evaluation"]["episodes"] == 1
        assert state["development_seeds"] == [11, 12]
        assert state["observations"][0]["nodes"][0]["motion"] != initial["observations"][0]["nodes"][0]["motion"]
    combined = bodies[2]["state"]
    assert combined["evaluation"]["successes"] == 1
    assert combined["observations"][0]["success"] is True
    assert combined["capabilities"][combined["driver_index"]["move"]["capability"]]["tunables"] == {"first": 1.0, "second": 1.0}
    standalone = bodies[3]["last_tool_result"]["data"]
    assert standalone["parent_id"] == learner.initial_id
    assert bodies[3]["state"]["evaluation"]["successes"] == 0
    assert bodies[3]["state"]["capabilities"][bodies[3]["state"]["driver_index"]["move"]["capability"]]["tunables"] == {"first": 0.0, "second": 1.0}
    assert world.calls[-1][1] == {"tunables": {"first": 0.0, "second": 1.0}}
    assert row["trial_calls"] == 3 and [scope for _, _, scope in world.calls] == ["probe"] * 3
    assert learner.selected_id is None and world.applied == {"tunables": {"first": 0.0, "second": 0.0}}


def test_first_request_preserves_unknown_zero_false_and_measured_motion_without_diagnosis_labels(tmp_path):
    world = _world()
    for row in world.baseline["seeds"].values():
        row["nodes"]["zero"] = {"ok": False, "steps": 0}
        row["nodes"]["unknown"] = {"ok": None, "steps": 0}
        row["nodes"]["never-observed"] = {}
    original = world.learner._project
    measured = analyze_trace(trace([(0, 0)] * 5, active=True))
    unknown = analyze_trace({})

    def project(overlay, suite):
        proj = original(overlay, suite)
        proj["diagnosis"] = {"traces": [{"seed": "11", "node": "move", **measured},
                                         {"seed": 12, "node": "move", **unknown}]}
        return proj

    world.learner._project = project
    _, _, bodies, _ = _loop(tmp_path, world, [STOP])
    state = bodies[0]["state"]
    first, second = state["observations"]
    nodes = {row["node"]: row for row in first["nodes"]}
    assert nodes["zero"] == {"node": "zero", "ok": False, "steps": 0, "motion": None}
    assert nodes["unknown"] == {"node": "unknown", "ok": None, "steps": 0, "motion": None}
    assert "never-observed" not in nodes
    expected = {"coverage": measured["coverage"], "channels": [
        {"channel": row["channel"], "evidence": row["evidence"]} for row in measured["findings"]]}
    assert nodes["move"]["motion"] == expected
    assert expected["channels"][0]["evidence"]["progress"] == 0
    assert nodes["move"]["motion"]["coverage"]["kind"] == "downsampled"
    other = next(row for row in second["nodes"] if row["node"] == "move")
    assert other["motion"]["channels"] == [{"channel": row.get("channel"), "evidence": row.get("evidence")}
                                               for row in unknown["findings"]]
    assert state["capabilities"][state["driver_index"]["move"]["capability"]]["tunables"] == {"first": 0.0, "second": 0.0}
    assert first["success"] is False and state["evaluation"]["successes"] == 0
    for finding in measured["findings"]:
        assert finding["kind"] not in encoded(state).decode()
    assert "fingerprint" not in encoded(state).decode()


def test_parameter_reference_merging_preserves_every_hit_and_exact_source(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    module = "overlapping_parameter_fixture"
    source = ("class Driver:\n    LIMIT = 2\n    def step(self):\n"
              "        before = self.LIMIT\n        after = self.LIMIT + 1\n        return before + after\n")
    path = tmp_path / f"{module}.py"
    path.write_text(source)
    proj = {"drivers": {"move": {"modules": [module]}}}
    data, _ = _read_all(proj, {"view": "parameter", "node": "move", "parameter": "LIMIT"})
    reference, = data["references"]
    assert reference["lines"] == [2, 4, 5] and reference["line"] == 2
    assert reference["code"] == "".join(source.splitlines(keepends=True)[reference["start"] - 1:reference["end"]])
    assert reference["source_sha"] == content_id(source) and path.read_text() == source
    assert data["effective"] is None and data["declared_tunable"] is False


@pytest.mark.parametrize("binding", ["A", "missing"])
def test_parameter_scope_uses_the_bound_class_and_keeps_module_helpers_without_guessing(tmp_path, monkeypatch, binding):
    monkeypatch.syspath_prepend(str(tmp_path))
    module_name = f"bound_parameter_{binding}_fixture"
    sections = ["def helper(driver):\n    return driver._count\n",
                "class A:\n    def read(self):\n        self._count = 1\n        return self._count\n",
                "class B:\n    def read(self):\n        self._count = 999\n        return self._count\n"]
    source = ("\n" * 8).join(sections)
    path = tmp_path / f"{module_name}.py"
    path.write_text(source)
    module = importlib.import_module(module_name)
    classes = bound_class_index([module.A]) if binding == "A" else [{"module": module_name, "class": "Missing"}]
    proj = {"drivers": {"move": {"modules": [module_name], "bound_classes": classes}}}
    data, _ = _read_all(proj, {"view": "parameter", "node": "move", "parameter": "_count"})
    assert any("return driver._count" in ref["code"] for ref in data["references"])
    assert all("999" not in ref["code"] for ref in data["references"])
    assert all("999" not in definition["code"] for definition in data["static_definitions"])
    if binding == "A":
        assert any("self._count = 1" in ref["code"] for ref in data["references"])
        assert len(data["static_definitions"]) == 1
    else:
        assert data["static_definitions"] == []
        assert all("self._count" not in ref["code"] for ref in data["references"])
    assert data["effective"] is None and path.read_text() == source
