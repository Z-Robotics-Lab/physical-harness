"""Operational predicates cannot turn controller outputs into world evidence."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from harness.events import SessionLog
from plugins.task import workload
from test_task_seam import HET_CATALOGUE, _FixedPlanner, _RolloutFake, _task_kernel


NODE = {"id": "check", "kind": "verify", "skill": "check", "args": {}}


def _context():
    return workload.NodeCtx(
        seed=7, env_ref="fixture:world", policy_ref="fixture:policy", skills=(),
        arm="scripted", segment_specs={"grasp": {"task": "lift"}},
        episode=SimpleNamespace(env=SimpleNamespace(grounded=True), obs={"height": 1.0},
                                spec=SimpleNamespace(task="lift", seed=7, policy_provider="fixture:policy"),
                                driver=SimpleNamespace(done=lambda: True), factories={"fixture": object()}),
        predicates={"check": "fixture:check"},
        nodes_out={"survey": {"success": True, "facts": {"height": 1.0}},
                   "actor": {"success": True, "facts": {"height": 1.0}},
                   "decision": {"success": True, "decision": {"done": True}}},
        _provenance={"survey": {"kind": "perceive", "clean": True, "blocked_reads": []},
                     # Even a payload with facts and a clean marker does not
                     # change the kind actually dispatched by the workload.
                     "actor": {"kind": "segment", "clean": True, "blocked_reads": []},
                     "decision": {"kind": "decide", "clean": True, "blocked_reads": []}})


def test_world_and_clean_perception_facts_are_measured_with_one_predicate_call(monkeypatch):
    ctx = _context()
    calls = []

    def predicate(node, seen):
        calls.append(node)
        assert seen.episode.env is ctx.episode.env
        assert seen.episode.obs is ctx.episode.obs
        assert (seen.seed, seen.env_ref, seen.episode.spec.task, seen.episode.spec.seed) == (
            7, "fixture:world", "lift", 7)
        facts = (seen.nodes_out.get("survey") or {}).get("facts") or {}
        return {"success": seen.episode.env.grounded and facts["height"] == seen.episode.obs["height"]}

    monkeypatch.setattr(workload, "_predicate", lambda node, context: predicate)
    result = workload._verify(NODE, ctx)
    assert result["success"] is result["verification_success"] is True
    assert result["evidence_policy"] == "world-dependencies-v1"
    assert result["blocked_reads"] == [] and calls == [NODE]


@pytest.mark.parametrize("read,path", [
    (lambda ctx: ctx.nodes_out["actor"]["success"], "ctx.nodes_out['actor']"),
    (lambda ctx: ctx.nodes_out["actor"]["facts"]["height"], "ctx.nodes_out['actor']"),
    (lambda ctx: ctx.nodes_out["survey"]["success"], "ctx.nodes_out['survey']['success']"),
    (lambda ctx: ctx.nodes_out["decision"]["decision"], "ctx.nodes_out['decision']"),
    (lambda ctx: ctx.nodes_out.get("missing", {}), "ctx.nodes_out['missing']"),
    (lambda ctx: "actor" in ctx.nodes_out, "ctx.nodes_out['actor']"),
    (lambda ctx: len(ctx.nodes_out), "ctx.nodes_out[*]"),
    (lambda ctx: list(ctx.nodes_out.values()), "ctx.nodes_out[*]"),
    (lambda ctx: ctx.episode.driver.done(), "ctx.episode.driver"),
    (lambda ctx: ctx.episode.factories, "ctx.episode.factories"),
    (lambda ctx: ctx.episode.spec.policy_provider, "ctx.episode.spec.policy_provider"),
    (lambda ctx: ctx.policy_ref, "ctx.policy_ref"),
    (lambda ctx: ctx.segment_specs, "ctx.segment_specs"),
    (lambda ctx: ctx.arm, "ctx.arm"),
    (lambda ctx: ctx.skills, "ctx.skills"),
])
def test_non_world_reads_keep_their_operational_value_but_withhold_evaluation(monkeypatch, read, path):
    ctx = _context()
    expected = read(ctx)
    calls = []

    def predicate(node, seen):
        calls.append(node)
        assert read(seen) == expected
        # Eagerly reading a dependency is conservatively tainted even if this
        # particular boolean branch then returns the independent world value.
        return {"success": seen.episode.env.grounded}

    monkeypatch.setattr(workload, "_predicate", lambda node, context: predicate)
    result = workload._verify(NODE, ctx)
    assert result["success"] is True and result["verification_success"] is None
    assert path in result["blocked_reads"] and calls == [NODE]


def test_unclassified_facts_and_upstream_taint_are_never_silently_trusted(monkeypatch):
    ctx = _context()
    monkeypatch.setattr(workload, "_predicate", lambda node, context: lambda node, seen: {
        "success": seen.nodes_out["survey"]["facts"]["height"] > 0})
    ctx._provenance.clear()
    result = workload._verify(NODE, ctx)
    assert result["success"] is True and result["verification_success"] is None
    ctx._provenance["survey"] = {"kind": "perceive", "clean": False,
                                 "blocked_reads": ["ctx.episode.driver"]}
    result = workload._verify(NODE, ctx)
    assert result["verification_success"] is None
    assert result["blocked_reads"] == ["ctx.episode.driver", "ctx.nodes_out['survey']"]


@pytest.mark.parametrize("prefix", ["", "part."])
@pytest.mark.parametrize("tainted", [False, True])
def test_workload_propagates_actual_dispatch_provenance_through_facts_and_task_aliases(
        monkeypatch, prefix, tainted):
    calls = Counter()
    task_label = {"task": "part"} if prefix else {}
    plan = {"goal": "audit evidence dependencies", "nodes": [
        {"id": prefix + "grasp-0", "kind": "manipulate", "skill": "grasp",
         "args": {"object": "cube"}, "after": [], **task_label},
        {"id": prefix + "survey", "kind": "perceive", "skill": "survey",
         "args": {}, "after": [prefix + "grasp-0"], **task_label},
        {"id": prefix + "check", "kind": "verify", "skill": "check",
         "args": {}, "after": [prefix + "survey"], **task_label}],
        "verify": [{"after": prefix + "grasp-0", "predicate": "lifted"}]}
    if prefix:
        plan["tasks"] = [{"id": "part", "goal": []}]

    def survey(node, ctx):
        calls["survey"] += 1
        value = ctx.nodes_out["grasp-0"]["success"] if tainted else ctx.seed == 7
        return {"success": True, "facts": {"grounded": value}}

    def check(node, ctx):
        calls["check"] += 1
        return {"success": (ctx.nodes_out.get("survey") or {})["facts"]["grounded"]}

    predicates = {"survey": survey, "check": check}
    monkeypatch.setattr(workload, "_predicate", lambda node, context: predicates[node["skill"]])
    monkeypatch.setattr(workload, "_governed_rollout", _RolloutFake([True]))
    log = SessionLog()
    kernel = _task_kernel(_FixedPlanner(plan), log=log)
    out = workload.run({"task": "inventory", "catalogue": HET_CATALOGUE, "oracles": ["lifted"],
                        "predicates": {key: f"fixture:{key}" for key in predicates}},
                       kernel, seed=7, max_actuations=4)

    assert out["success"] is True and out["replans"] == 0, out["faults"]
    assert calls == {"survey": 1, "check": 1}
    observation, = out["verification_observations"]
    assert observation["success"] is (None if tainted else True)
    assert observation["evidence_policy"] == "world-dependencies-v1"
    assert bool(observation["blocked_reads"]) is tainted
    if tainted:
        assert any("grasp-0" in path for path in observation["blocked_reads"])
        assert any("survey" in path for path in observation["blocked_reads"])
    survey_event = next(row["data"] for row in log.rows()
                        if row["kind"] == "task.verify" and row["data"]["node"] == prefix + "survey")
    assert bool(survey_event["dependency_audit"]["blocked_reads"]) is tainted
    sealed = next(row["data"]["observation"] for row in log.rows()
                  if row["kind"] == "task.verify" and row["data"]["node"] == prefix + "check")
    assert sealed == observation and log.verify()
