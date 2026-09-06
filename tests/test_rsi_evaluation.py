"""Adversarial contracts: the graph and controller cannot author the ruler."""

from __future__ import annotations

import copy

import pytest

from plugins.rsi.evaluation import EvaluationMonitor, compare, compile_contract, evaluate, summary


VERIFY = {"id": "placed", "kind": "verify", "skill": "object_inside",
          "args": {"object": "cup", "target": "cabinet"}, "after": ["place"]}
PLAN = {"nodes": [
    {"id": "place", "kind": "segment", "skill": "place", "args": {}, "after": []},
    VERIFY,
    {"id": "report", "kind": "decide", "skill": "report", "args": {}, "after": ["placed"]}],
    "verify": [{"after": "place", "predicate": "done"}]}
PREDICATES = {"object_inside": "test:inside", "report": "test:report"}


def contract(**kwargs):
    return compile_contract(PLAN, task="put_away", predicates=PREDICATES,
                            terminal_ref="test:embodiment", **kwargs)


def observation(ok, **kwargs):
    return {"node": copy.deepcopy(VERIFY), "success": ok,
            "authority": "predicate", "source": "test:inside",
            "evidence_policy": "world-dependencies-v1", "blocked_reads": [], **kwargs}


def terminal(ok):
    return {"authority": "embodiment.terminal_success", "source": "test:embodiment", "success": ok}


def suite(c, entries):
    return {"seeds": {str(seed): {"evaluation": evaluate(c, [observation(ok)], terminal(done)),
                                  "success": True, "trail": []}
                      for seed, (ok, done) in entries.items()}}


def test_only_independent_predicate_and_terminal_are_obligations():
    c = contract()
    assert len(c["obligations"]) == 2  # segment done and report-readable are excluded
    reading = evaluate(c, [observation(True)], terminal(False))
    assert reading["passed"] == 1 and reading["complete"] is False
    assert evaluate(c, [observation(False)], terminal(True))["complete"] is True


def test_graph_renames_repetition_and_easy_nodes_cannot_add_credit():
    c = contract()
    renamed = observation(True)
    renamed["node"]["id"] = "another-name"
    inserted = observation(True)
    inserted["node"]["skill"] = "easy"
    m = EvaluationMonitor(c)
    for row in [renamed] * 50 + [inserted] * 50:
        m.observe(row["node"], row)
    assert m.snapshot()["passed"] == 1
    # Even the server plan's display aliases/repetition do not define weights.
    p = copy.deepcopy(PLAN)
    p["nodes"][1]["id"] = "alias"
    p["nodes"].append({**VERIFY, "id": "duplicate"})
    assert compile_contract(p, task="put_away", predicates=PREDICATES,
                            terminal_ref="test:embodiment")["sha"] == c["sha"]


def test_changed_arguments_wrong_authority_and_diagnostics_leave_unknown():
    c = contract()
    moved_target = observation(True)
    moved_target["node"]["args"]["target"] = "under_gripper"
    forged = observation(True, source="candidate:inside", diagnostics={"reward": 1e20})
    result = evaluate(c, [moved_target, forged], {"success": True})
    assert result["observed"] == 0 and result["passed"] == 0 and not result["complete"]
    m = EvaluationMonitor(c)
    m.observe(VERIFY, observation(1))
    assert m.snapshot()["observed"] == 0  # numeric truthiness is not evidence


def test_checkpoint_recheck_replaces_previous_truth_without_accumulating():
    c = contract()
    r = evaluate(c, [observation(True), observation(False)])
    assert r["observed"] == 1 and r["passed"] == 0


@pytest.mark.parametrize("audit", [
    {},  # Legacy observations have no dependency audit; do not relabel them.
    {"evidence_policy": "world-dependencies-v1"},
    {"evidence_policy": "legacy", "blocked_reads": []},
    {"evidence_policy": "world-dependencies-v1", "blocked_reads": None},
    {"evidence_policy": "world-dependencies-v1", "blocked_reads": ["ctx.episode.driver"]},
])
def test_legacy_or_blocked_predicate_observations_cannot_create_a_gain(audit):
    c = contract()
    raw = {"node": copy.deepcopy(VERIFY), "success": True,
           "authority": "predicate", "source": "test:inside", **audit}
    original = copy.deepcopy(raw)
    reading = evaluate(c, [raw], terminal(False))
    assert reading["passed"] == 0 and reading["observed"] == 1
    after = {"seeds": {"1": {"evaluation": reading}}}
    verdict = compare(suite(c, {1: (False, False)}), after, c)
    assert not verdict["accepted"] and verdict["gains"] == []
    assert raw == original  # Reading history never invents the missing audit.


def test_a_blocked_recheck_replaces_previous_predicate_truth_with_unknown():
    c = contract()
    result = evaluate(c, [observation(True), observation(True, blocked_reads=["ctx.results"])],
                      terminal(False))
    assert result["passed"] == 0 and result["observed"] == 1
    verdict = compare(suite(c, {1: (True, False)}), {"seeds": {"1": {"evaluation": result}}}, c)
    assert not verdict["accepted"] and len(verdict["regressions"]) == 1
    assert verdict["regressions"][0]["after"] is None


def test_paired_vector_refuses_tradeoff_missing_seed_and_changed_ruler():
    c = contract()
    before = suite(c, {1: (True, False), 2: (False, False)})
    gain = suite(c, {1: (True, False), 2: (True, False)})
    assert compare(before, gain, c)["accepted"] is True
    assert compare(before, gain, c)["comparable"] is True
    lost = suite(c, {1: (None, True), 2: (True, True)})
    verdict = compare(before, lost, c)
    assert not verdict["accepted"] and len(verdict["regressions"]) == 1
    assert not compare(before, suite(c, {1: (True, True)}), c)["accepted"]
    assert compare(before, suite(c, {1: (True, True)}), c)["comparable"] is False
    other = contract(identity={"predicate_source_sha": "changed"})
    assert not compare(before, suite(other, {1: (True, True), 2: (True, True)}), c)["accepted"]
    assert compare(before, suite(other, {1: (True, True), 2: (True, True)}), c)["comparable"] is False


def test_summary_ignores_candidate_success_and_normalizes_fixed_denominator():
    c = contract()
    s = suite(c, {1: (True, False), 2: (False, False)})
    assert summary(s, c) == {"successes": 0, "episodes": 2, "passed": 1,
                             "progress": 0.25, "obligations": 2, "observed": 4,
                             "contract_sha": c["sha"]}


def test_no_verifier_is_unavailable_and_tampering_the_contract_fails():
    c = compile_contract(PLAN)
    assert not c["available"]
    assert not compare({"seeds": {"1": {}}}, {"seeds": {"1": {}}}, c)["accepted"]
    declared = contract()
    unreadable = suite(declared, {1: (None, None)})
    assert not unreadable["seeds"]["1"]["evaluation"]["available"]
    assert "unavailable" in compare(unreadable, unreadable, declared)["reason"]
    assert compare(unreadable, unreadable, declared)["comparable"] is False
    c = contract()
    c["obligations"][0]["source"] = "candidate:oracle"
    with pytest.raises(ValueError, match="identity"):
        EvaluationMonitor(c)


def test_symbolic_ensures_are_grounded_metadata_not_claimed_truth():
    c = compile_contract(PLAN, {"object_inside": {"ensures": ["inside(object,target)"]}},
                         task="put_away", predicates=PREDICATES)
    assert c["obligations"][0]["ensures"] == ["inside(cup,cabinet)"]
    assert EvaluationMonitor(c).snapshot()["passed"] == 0


def test_workload_emits_world_observations_before_close_and_not_graph_success(monkeypatch):
    import test_persistent_mission as fixture
    from plugins.rsi import governed
    from plugins.task import workload

    fixture._fresh()
    monkeypatch.setattr(governed, "governed_segment", fixture._fake_drive)
    calls = []

    def task_terminal(obs, spec, start_z, env):
        calls.append((env.closes, spec.task, start_z))
        return False  # graph completion and a readable report do not prove task success

    monkeypatch.setattr(fixture.EMB, "terminal_success", task_terminal, raising=False)
    out = workload.run(fixture._brief(), fixture._kernel(), seed=3, max_actuations=10)
    c = compile_contract(fixture._EpisodicPlanner().plan({}), task="clearall",
                         predicates=fixture._PREDICATES,
                         terminal_ref="test_persistent_mission:epi_embodiment")
    result = evaluate(c, out["verification_observations"], out["terminal_observation"])
    assert out["success"] is True and result["complete"] is False
    assert result["passed"] == 2 and result["observed"] == 3
    assert calls == [(0, "clear_a", 0.9)] and fixture.WORLD.closes == 1


def test_missing_terminal_and_unknown_verifier_are_never_inferred_from_segment(monkeypatch):
    import test_persistent_mission as fixture
    from plugins.rsi import governed
    from plugins.task import workload

    fixture._fresh()
    monkeypatch.setattr(governed, "governed_segment", fixture._fake_drive)
    monkeypatch.setattr(fixture, "inbin_a", lambda: lambda node, ctx: {"success": None})
    out = workload.run(fixture._brief(), fixture._kernel(), seed=3,
                       max_actuations=10, max_replans=0)
    assert out["verification_observations"][0]["success"] is None
    assert out["terminal_observation"]["success"] is None
    assert "no terminal_success" in out["terminal_observation"]["error"]
    assert fixture.WORLD.closes == 1


def test_frames_wrapper_preserves_the_original_terminal_authority(monkeypatch):
    import test_persistent_mission as fixture
    from harness import Kernel
    from harness.definitions import CAPABILITIES
    from plugins.rsi import governed
    from plugins.task import workload
    from scripts import frame_dump

    fixture._fresh()
    original = "test_persistent_mission:epi_embodiment"
    mounted = "scripts.frame_dump:frames_provider"
    monkeypatch.setattr(governed, "governed_segment", fixture._fake_drive)
    monkeypatch.setattr(frame_dump, "_BASE_REF", original)
    monkeypatch.setattr(frame_dump, "_PATH", None)
    monkeypatch.setattr(fixture.EMB, "terminal_success",
                        lambda obs, spec, start_z, env: {"a", "b"} <= env.placed, raising=False)
    kernel = Kernel(CAPABILITIES)
    for name, (provider, ref, params) in fixture._kernel()._providers.items():
        kernel.provide(name, frame_dump.frames_provider() if name == "embodiment.env" else provider,
                       ref=mounted if name == "embodiment.env" else ref, params=params)
    out = workload.run(fixture._brief(embodiment=original), kernel, seed=3, max_actuations=10)
    c = compile_contract(fixture._EpisodicPlanner().plan({}), task="clearall",
                         predicates=fixture._PREDICATES, terminal_ref=original)
    assert out["terminal_observation"]["source"] == original
    assert out["terminal_observation"]["mounted_ref"] == mounted
    assert evaluate(c, out["verification_observations"], out["terminal_observation"])["complete"]
