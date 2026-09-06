"""Round 83: the task.planner seam — contract, validation, and the workload loop.

Rung 1: a planner is untrusted regardless of author (governor/proposer.py's
stance), so validate_plan's rejection surface is the load-bearing thing: one
test per refusal class, each asserting the message names the offender so it
can be folded straight back into the next brief. The AST boundary tests cover
the new plugins/task package automatically (harness+stdlib imports only).

Rung 2: the workload's replan loop, tested with a monkeypatched rollout (no
simulation anywhere in this file, the test_rsi_workload fake手法): one-pass,
invalid-plan-refolded-then-repaired, stage-failure-to-exhaustion, the
max_actuations floor, and the ledger note.
"""

from __future__ import annotations

import json

import pytest

from harness import Kernel
from harness.contracts import TaskPlanner
from harness.definitions import CAPABILITIES
from harness.events import SessionLog
from harness.registry import load_provider
from plugins.graphs import InMemorySkillGraph
from plugins.task import workload
from plugins.task.planner_stack import CATALOGUE, ORACLES, StackPlanner
from plugins.task.validate import validate_plan

PLANNER_REF = "plugins.task.planner_stack:provider"
BRIEF = {"task": "stack", "scene": {}, "catalogue": CATALOGUE}


def _plan() -> dict:
    return StackPlanner().plan(BRIEF)


def test_stack_plan_passes_validation():
    ok, msg = validate_plan(_plan(), CATALOGUE, ORACLES)
    assert ok and msg == ""


def test_unknown_skill_is_refused_by_name():
    plan = _plan()
    plan["nodes"][0]["skill"] = "teleport"
    ok, msg = validate_plan(plan, CATALOGUE, ORACLES)
    assert not ok and "teleport" in msg and "catalogue" in msg


def test_bad_args_are_refused_by_name():
    unknown = _plan()
    unknown["nodes"][0]["args"]["speed"] = 2.0
    ok, msg = validate_plan(unknown, CATALOGUE, ORACLES)
    assert not ok and "speed" in msg

    mistyped = _plan()
    mistyped["nodes"][0]["args"]["object"] = 7
    ok, msg = validate_plan(mistyped, CATALOGUE, ORACLES)
    assert not ok and "'object'" in msg and "str" in msg


def test_hallucinated_predicate_is_refused():
    plan = _plan()
    plan["verify"][0]["predicate"] = "hope_it_worked"
    ok, msg = validate_plan(plan, CATALOGUE, ORACLES)
    assert not ok and "hope_it_worked" in msg and "oracles" in msg


def test_empty_graph_is_refused():
    for hollow in ({**_plan(), "nodes": []}, {**_plan(), "verify": []}):
        ok, _msg = validate_plan(hollow, CATALOGUE, ORACLES)
        assert not ok, hollow


def test_same_brief_yields_byte_identical_json():
    a = json.dumps(StackPlanner().plan(BRIEF), sort_keys=True)
    b = json.dumps(StackPlanner().plan(BRIEF), sort_keys=True)
    assert a == b


def test_planner_satisfies_the_contract_through_the_kernel():
    p = load_provider(PLANNER_REF)
    assert isinstance(p, TaskPlanner)
    assert p.identity == "stack_planner@v1"
    # Definition.__post_init__ already vetted task.planner's Protocol at import;
    # provide() runs the structural check, resolve() records the accounting.
    k = Kernel(CAPABILITIES)
    k.provide("task.planner", p, ref=PLANNER_REF)
    assert k.resolve("task.planner", consumer="task") is p
    assert [r.consumer for r in k.resolutions()] == ["task"]


def test_non_stack_task_fails_loudly():
    with pytest.raises(ValueError):
        StackPlanner().plan({"task": "lift"})


# =====================================================================
# rung 2: plugins.task.workload.run — the replan loop, rollout faked
# =====================================================================

WBRIEF = {"task": "stack", "catalogue": CATALOGUE, "oracles": ORACLES}


class _FakeEnv:
    def make_env(self, spec):
        raise AssertionError("no env in these tests: the rollout is monkeypatched")

    def tasks(self):
        return ("stack",)

    def object_key(self, spec):
        return "cubeA_pos"

    def success(self, obs, spec, start_z):
        return False


class _FakePolicy:
    def make_driver(self, spec):
        raise AssertionError("no driver in these tests")


class _FakeExecutor:
    def map(self, fn, items, *, workers):
        return [fn(item) for item in items]


class _FakeScene:
    def snapshot(self, obs):
        return {"frame": "world", "t": 0.0, "nodes": [], "relations": []}


class _CountingPlanner:
    """The real StackPlanner's plan, with every brief it saw captured."""

    def __init__(self):
        self.briefs: list[dict] = []

    def plan(self, brief):
        self.briefs.append(dict(brief))
        return StackPlanner().plan(brief)


class _FlakyPlanner(_CountingPlanner):
    """Hallucinates a skill first; repairs itself once the fault comes back."""

    def plan(self, brief):
        self.briefs.append(dict(brief))
        good = StackPlanner().plan(brief)
        if brief.get("fault") is None:
            bad = json.loads(json.dumps(good))
            bad["nodes"][0]["skill"] = "teleport"
            return bad
        return good


def _rollout_result(ok: bool) -> dict:
    """governed_rollout's stages-on shape (governed.py result assembly)."""
    return {"success": ok, "steps": 177, "stages": [
        {"name": "grasp", "entered_step": 0, "exited_step": 66, "success": True,
         "reached": True, "privilege_used": 1},
        {"name": "place", "entered_step": 66, "exited_step": None, "success": ok,
         "reached": True, "privilege_used": 1},
    ]}


class _RolloutFake:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.specs = []
        self.bundles = []

    def __call__(self, spec, bundle=None):
        self.specs.append(spec)
        self.bundles.append(bundle)
        return _rollout_result(self.outcomes.pop(0))


def _task_kernel(planner, log=None) -> Kernel:
    k = Kernel(CAPABILITIES, log=log)
    k.provide("task.planner", planner, ref="tests.fakes:planner")
    k.provide("graph.scene", _FakeScene(), ref="tests.fakes:scene")
    k.provide("graph.skill", InMemorySkillGraph(), ref="plugins.graphs:skill_graph_provider")
    k.provide("embodiment.env", _FakeEnv(), ref="tests.fakes:env")
    k.provide("policy.driver", _FakePolicy(), ref="tests.fakes:policy")
    k.provide("exec.rollouts", _FakeExecutor(), ref="tests.fakes:executor")
    return k


def test_workload_one_pass(monkeypatch):
    kernel = _task_kernel(StackPlanner())
    fake = _RolloutFake([True])
    monkeypatch.setattr(workload, "_governed_rollout", fake)

    out = workload.run(dict(WBRIEF), kernel, seed=123)

    assert out["success"] is True
    assert out["replans"] == 0 and out["actuations"] == 1
    assert [s["name"] for s in out["nodes"]["stack-0"]["stages"]] == ["grasp", "place"]
    # the dispatched spec carries the kernel-resolved refs and the REAL
    # stack_stages() chain, loaded by ref string (plugins never import siblings)
    from plugins.embodiment_robosuite.env import stack_stages

    spec = fake.specs[0]
    assert spec.task == "stack" and spec.seed == 123 and spec.terminal_label
    assert spec.env_provider == "tests.fakes:env"
    assert spec.policy_provider == "tests.fakes:policy"
    assert spec.stages == stack_stages()
    resolved = {r.capability: r.consumer for r in kernel.resolutions()}
    assert resolved == {"task.planner": "task", "graph.scene": "task",
                        "graph.skill": "task", "embodiment.env": "task",
                        "policy.driver": "task", "exec.rollouts": "task"}
    assert not any(r.privileged for r in kernel.resolutions())


def test_invalid_plan_is_folded_back_and_repaired(monkeypatch):
    planner = _FlakyPlanner()
    kernel = _task_kernel(planner)
    fake = _RolloutFake([True])
    monkeypatch.setattr(workload, "_governed_rollout", fake)

    out = workload.run(dict(WBRIEF), kernel, seed=1)

    assert out["success"] is True and out["replans"] == 1
    assert len(planner.briefs) == 2
    fault = planner.briefs[1]["fault"]
    assert fault["kind"] == "invalid_plan"
    assert "teleport" in fault["msg"], "the validator's own words must reach the planner"
    assert out["faults"][0]["kind"] == "invalid_plan"
    assert len(fake.specs) == 1, "an invalid plan must never actuate"


def test_stage_failure_replans_until_max_replans_exhausted(monkeypatch):
    planner = _CountingPlanner()
    kernel = _task_kernel(planner)
    fake = _RolloutFake([False, False, False])
    monkeypatch.setattr(workload, "_governed_rollout", fake)

    out = workload.run(dict(WBRIEF), kernel, seed=1, max_replans=2, max_actuations=10)

    assert out["success"] is False
    assert len(planner.briefs) == 3, "initial plan + exactly max_replans replans"
    # the deterministic planner answers the same graph: one as-is retry actuates,
    # the second identical answer is refused (protocol.replan_progress), never run
    assert out["replans"] == 2 and out["actuations"] == 2
    assert [f["kind"] for f in out["faults"]] == ["node_failure", "node_failure", "no_progress"]
    # the Fault-shaped brief preserves failed/done/left, attributably
    fault = planner.briefs[1]["fault"]
    assert fault["kind"] == "node_failure" and fault["node"] == "stack-0"
    assert fault["done"] == ["grasp"] and fault["left"] == ["place"]
    assert "stack_success" in fault["failed"], "the failed verify predicate is named too"


def test_max_actuations_is_a_model_independent_floor(monkeypatch):
    planner = _CountingPlanner()
    kernel = _task_kernel(planner)
    fake = _RolloutFake([False, False, False, False])
    monkeypatch.setattr(workload, "_governed_rollout", fake)

    out = workload.run(dict(WBRIEF), kernel, seed=1, max_replans=5, max_actuations=1)

    assert out["success"] is False
    assert len(fake.specs) == 1, "the floor is enforced BEFORE dispatch"
    assert out["actuations"] == 1
    assert out["faults"][-1]["kind"] == "budget"
    assert len(planner.briefs) == 2, "budget exhaustion stops the loop, replans unspent"


def test_plan_complete_enters_the_session_chain(monkeypatch):
    log = SessionLog()
    kernel = _task_kernel(StackPlanner(), log=log)
    monkeypatch.setattr(workload, "_governed_rollout", _RolloutFake([True]))

    workload.run(dict(WBRIEF), kernel, seed=7)

    rows = [r for r in log.rows() if r["kind"] == "task.plan_complete"]
    assert len(rows) == 1
    data = rows[0]["data"]
    assert data["success"] is True and data["actuations"] == 1 and data["replans"] == 0
    assert data["nodes"]["stack-0"]["stages"] == [
        {"name": "grasp", "success": True}, {"name": "place", "success": True}]
    kinds = [r["kind"] for r in log.rows()]
    assert kinds.index("capability.resolve") < kinds.index("task.plan_complete"), \
        "the note must sit downstream of the resolutions in the same chain"
    assert log.verify(), "the combined ledger no longer verifies"


# =====================================================================
# round 86: clear_table — the first true multi-node graph, rollout faked
# =====================================================================

CT_BRIEF = {"task": "clear_table", "catalogue": CATALOGUE, "oracles": ORACLES}


def test_pick_catalogue_and_oracle_validate():
    plan = StackPlanner().plan({**CT_BRIEF, "scene": {}})
    ok, msg = validate_plan(plan, CATALOGUE, ORACLES)
    assert ok and msg == ""
    assert [n["id"] for n in plan["nodes"]] == ["pick-can", "pick-milk"]
    assert plan["nodes"][1]["after"] == ["pick-can"]
    assert json.dumps(plan, sort_keys=True) == \
        json.dumps(StackPlanner().plan({**CT_BRIEF, "scene": {}}), sort_keys=True)
    # the catalogue only TYPES the object arg; a str with no scene binding
    # fails loudly at dispatch, before any provider ref is even loaded
    node = {"id": "x", "skill": "pick", "args": {"object": "bottle"}, "after": []}
    with pytest.raises(ValueError, match="bottle"):
        workload._dispatch(node, seed=1, env_ref="bogus", policy_ref="bogus", skills=())


def test_grasp_node_dispatches_the_lift_geometric_binding(monkeypatch):
    """The geometric card's grasp node routes through the generic loop: SKILL_SPECS
    gives it an execution binding (round 97 follow-up), so _dispatch builds a lift
    spec with the pick grasp-stage chain instead of raising 'no execution binding'."""
    from plugins.embodiment_robosuite.env import pick_stages

    fake = _RolloutFake([True])
    monkeypatch.setattr(workload, "_governed_rollout", fake)
    node = {"id": "grasp-0", "skill": "grasp", "args": {"object": "cube"}, "after": []}

    result = workload._dispatch(node, seed=0, env_ref="tests.fakes:env",
                               policy_ref="tests.fakes:policy", skills=())

    assert result["success"] is True
    (spec,) = fake.specs
    assert spec.task == "lift" and spec.seed == 0 and spec.terminal_label
    assert spec.percept_noise == 0.012          # the card [claim] baseline
    assert spec.stages == pick_stages()          # real lift criteria, not a fake gate
    assert spec.env_provider == "tests.fakes:env"
    assert spec.policy_provider == "tests.fakes:policy"


def test_pick_stages_shape():
    from harness.spec import NOMINAL_SCHEDULE
    from plugins.embodiment_robosuite.env import pick_stages

    (grasp,) = pick_stages()
    assert grasp.name == "grasp"
    assert grasp.budget == sum(d for _, d in NOMINAL_SCHEDULE)
    (clause,) = grasp.success
    assert (clause.feature, clause.op, clause.threshold) == \
        ("observable.finger_gap", "gt", 0.01)


def test_clear_table_two_node_closed_loop(monkeypatch):
    kernel = _task_kernel(StackPlanner())
    fake = _RolloutFake([True, True])
    monkeypatch.setattr(workload, "_governed_rollout", fake)

    out = workload.run(dict(CT_BRIEF), kernel, seed=42, max_actuations=4)

    assert out["success"] is True
    assert out["replans"] == 0 and out["actuations"] == 2
    from plugins.embodiment_robosuite.env import pick_stages

    assert [s.task for s in fake.specs] == ["pickcan", "pickmilk"]
    assert all(s.stages == pick_stages() for s in fake.specs)
    # round 86: the probe ran at 0.012 and the first real closed loop silently
    # dispatched at the 0.020 EpisodeSpec default -- pin the operating point.
    assert all(s.percept_noise == 0.012 for s in fake.specs)
    assert list(out["nodes"]) == ["pick-can", "pick-milk"]
    assert all(n["success"] for n in out["nodes"].values())


def test_clear_table_replan_skips_done_node(monkeypatch):
    planner = _CountingPlanner()
    kernel = _task_kernel(planner)
    fake = _RolloutFake([True, False, True])
    monkeypatch.setattr(workload, "_governed_rollout", fake)

    out = workload.run(dict(CT_BRIEF), kernel, seed=1, max_actuations=4)

    assert out["success"] is True and out["replans"] == 1
    assert [s.task for s in fake.specs] == ["pickcan", "pickmilk", "pickmilk"], \
        "finished pick-can is skipped on replan, never re-dispatched"
    assert out["actuations"] == 3
    fault = planner.briefs[1]["fault"]
    assert fault["kind"] == "node_failure" and fault["node"] == "pick-milk"
    assert fault["nodes_done"] == ["pick-can"]
    assert fault["nodes_left"] == ["pick-milk"]
    assert fault["done"] == ["grasp"], "stage-level attribution is preserved"


def test_mounted_params_on_actuating_capabilities_are_refused():
    kernel = Kernel(CAPABILITIES)
    kernel.provide("task.planner", StackPlanner(), ref="tests.fakes:planner")
    kernel.provide("graph.scene", _FakeScene(), ref="tests.fakes:scene")
    kernel.provide("graph.skill", InMemorySkillGraph(), ref="plugins.graphs:skill_graph_provider")
    kernel.provide("embodiment.env", _FakeEnv(), ref="tests.fakes:env",
                   params={"variant": "viewer"})
    kernel.provide("policy.driver", _FakePolicy(), ref="tests.fakes:policy")
    kernel.provide("exec.rollouts", _FakeExecutor(), ref="tests.fakes:executor")
    with pytest.raises(ValueError, match="parameter-free"):
        workload.run(dict(WBRIEF), kernel, seed=1)


# =====================================================================
# generic node kinds (m6-mission-design §2): perceive / decide / verify
# over the SAME loop that runs manipulate, oracles machine-checked, replan
# routing free across heterogeneous nodes. No simulation: predicates are
# pure fakes resolved by ref (the same load_provider crossing as `stages`).
# =====================================================================

# Card-authored predicate FACTORIES (load_provider calls each, returns the
# callable predicate(node, ctx)). Referenced by ref from the tables below and
# from the plugin_doctor tests -- exactly the cross-plugin crossing the base uses.

def survey_pred():
    """perceive: reads the seed-deterministic scene, declares the privilege it
    read (one privileged pose + one observable), seals facts for a later decide."""
    def _run(node, ctx):
        return {"success": True,
                "facts": {"objects": ["cube", "can"], "seed": ctx.seed},
                "privilege": ["privileged.object_z", "observable.finger_gap"]}
    return _run


def decide_order_pred():
    """decide: PURE fn of ctx.nodes_out (the survey node's sealed facts)."""
    def _run(node, ctx):
        objs = ctx.nodes_out["survey"]["facts"]["objects"]
        return {"success": True, "decision": {"order": sorted(objs)}}
    return _run


def verify_survey_pred():
    """verify: a machine predicate over a prior node's sealed success."""
    def _run(node, ctx):
        return {"success": bool(ctx.nodes_out["survey"]["success"])}
    return _run


def flaky_gate_pred():
    """verify that fails its FIRST attempt then passes -- keyed off its OWN sealed
    prior entry (the loop seals a failed node before the fault), so the cross-kind
    replan is deterministic with no global state."""
    def _run(node, ctx):
        return {"success": node["id"] in ctx.nodes_out}
    return _run


def bad_priv_pred():
    """perceive declaring a feature the harness catalog does not know -- the base's
    privilege_cost must refuse it (a card cannot invent privileged reads)."""
    def _run(node, ctx):
        return {"success": True, "privilege": ["privileged.made_up"]}
    return _run


HET_PREDICATES = {
    "survey": "tests.test_task_seam:survey_pred",
    "check": "tests.test_task_seam:verify_survey_pred",
    "order": "tests.test_task_seam:decide_order_pred",
    "gate": "tests.test_task_seam:flaky_gate_pred",
}
#: a table with a dead value -- the doctor must redden on it.
HET_PREDICATES_DEAD = {"survey": "tests.test_task_seam:does_not_exist"}

HET_CATALOGUE = {"survey": {}, "check": {}, "order": {}, "gate": {},
                 "grasp": {"object": str}}
HET_ORACLES = ["lifted"]


class _FixedPlanner:
    """Returns a fixed heterogeneous graph (deep-copied each call so the loop's
    mutations never leak), capturing every brief for replan assertions."""

    def __init__(self, plan):
        self._plan = plan
        self.briefs: list[dict] = []

    def plan(self, brief):
        self.briefs.append(dict(brief))
        return json.loads(json.dumps(self._plan))


# survey(perceive) -> check(verify) -> order(decide) -> grasp(manipulate)
HET_PLAN = {
    "goal": "inventory build",
    "nodes": [
        {"id": "survey", "kind": "perceive", "skill": "survey", "args": {}, "after": []},
        {"id": "check-survey", "kind": "verify", "skill": "check", "args": {},
         "after": ["survey"]},
        {"id": "order", "kind": "decide", "skill": "order", "args": {},
         "after": ["check-survey"]},
        {"id": "grasp-0", "skill": "grasp", "args": {"object": "cube"},
         "after": ["order"]},  # no kind -> defaults to manipulate
    ],
    "verify": [{"after": "grasp-0", "predicate": "lifted"}],
}

# survey(perceive) -> gate(verify, fails once) -> grasp(manipulate)
REPLAN_PLAN = {
    "goal": "gated build",
    "nodes": [
        {"id": "survey", "kind": "perceive", "skill": "survey", "args": {}, "after": []},
        {"id": "gate", "kind": "verify", "skill": "gate", "args": {}, "after": ["survey"]},
        {"id": "grasp-0", "skill": "grasp", "args": {"object": "cube"}, "after": ["gate"]},
    ],
    "verify": [{"after": "grasp-0", "predicate": "lifted"}],
}

HBRIEF = {"task": "inventory", "catalogue": HET_CATALOGUE, "oracles": HET_ORACLES,
          "predicates": HET_PREDICATES}


def test_each_node_kind_dispatches_on_its_own_honest_oracle(monkeypatch):
    kernel = _task_kernel(_FixedPlanner(HET_PLAN))
    fake = _RolloutFake([True])
    monkeypatch.setattr(workload, "_governed_rollout", fake)

    out = workload.run({**HBRIEF}, kernel, seed=7, max_actuations=6)

    assert out["success"] is True and out["replans"] == 0 and out["actuations"] == 4
    # perceive: facts sealed, and the BASE metered the privilege it declared
    assert out["nodes"]["survey"]["facts"]["objects"] == ["cube", "can"]
    assert out["nodes"]["survey"]["governance"] == {
        "privilege_features": ("privileged.object_z", "observable.finger_gap"),
        "privilege_cost": 1}  # object_z is privileged (1), finger_gap observable (0)
    # decide: a pure route over the survey node's sealed facts
    assert out["nodes"]["order"]["decision"] == {"order": ["can", "cube"]}
    assert out["nodes"]["order"]["governance"]["privilege_cost"] == 0
    # verify: machine predicate over the prior sealed success
    assert out["nodes"]["check-survey"]["success"] is True
    # manipulate: still routed through _dispatch to a real EpisodeSpec (grasp=lift)
    (spec,) = fake.specs
    assert spec.task == "lift" and spec.seed == 7


def test_replan_routes_across_heterogeneous_kinds(monkeypatch):
    planner = _FixedPlanner(REPLAN_PLAN)
    kernel = _task_kernel(planner)
    fake = _RolloutFake([True])  # the manipulate node only runs after the gate passes
    monkeypatch.setattr(workload, "_governed_rollout", fake)

    out = workload.run({**HBRIEF}, kernel, seed=3, max_actuations=6)

    assert out["success"] is True and out["replans"] == 1
    # the verify NODE's own False folded a node_failure that drove the replan --
    # no new routing code, the existing fault->replan loop carried a non-manipulate
    # node's failure exactly as it carries a manipulate node's.
    fault = planner.briefs[1]["fault"]
    assert fault["kind"] == "node_failure" and fault["node"] == "gate"
    assert fault["failed"] == ["gate"], "a kindful node names itself when it has no stages"
    # survey (perceive) succeeded first pass -> skipped on replan; the manipulate
    # node ran exactly once, after the gate passed.
    assert len(fake.specs) == 1
    assert out["nodes"]["survey"]["success"] and out["nodes"]["grasp-0"]["success"]


def test_kind_handler_table_covers_every_validator_kind():
    from plugins.task.validate import NODE_KINDS
    assert set(workload._KIND_HANDLERS) == set(NODE_KINDS)


def test_perceive_privilege_is_metered_by_the_base_not_the_card():
    ctx = workload.NodeCtx(seed=1, env_ref="e", policy_ref="p", skills=(),
                           nodes_out={}, predicates=HET_PREDICATES)
    node = {"id": "survey", "kind": "perceive", "skill": "survey", "args": {}, "after": []}
    res = workload._perceive(node, ctx)
    assert res["governance"]["privilege_cost"] == 1


def test_perceive_refuses_an_undeclared_privileged_feature():
    ctx = workload.NodeCtx(seed=1, env_ref="e", policy_ref="p", skills=(),
                           nodes_out={},
                           predicates={"x": "tests.test_task_seam:bad_priv_pred"})
    node = {"id": "x", "kind": "perceive", "skill": "x", "args": {}, "after": []}
    with pytest.raises(KeyError, match="made_up"):
        workload._perceive(node, ctx)


def test_a_kindful_node_with_no_predicate_ref_fails_loudly():
    ctx = workload.NodeCtx(seed=1, env_ref="e", policy_ref="p", skills=(),
                           nodes_out={}, predicates={})
    node = {"id": "d", "kind": "decide", "skill": "unbound", "args": {}, "after": []}
    with pytest.raises(ValueError, match="no ref in the card's PREDICATES"):
        workload._decide(node, ctx)


def test_validate_admits_the_optional_kind_and_refuses_an_unknown_one():
    plan = json.loads(json.dumps(HET_PLAN))
    ok, msg = validate_plan(plan, HET_CATALOGUE, HET_ORACLES)
    assert ok and msg == ""
    plan["nodes"][0]["kind"] = "telepathy"
    ok, msg = validate_plan(plan, HET_CATALOGUE, HET_ORACLES)
    assert not ok and "telepathy" in msg and "known kinds" in msg


# =====================================================================
# untrusted-planner hardening (vlm-graph-paper-plan §1): verify coverage
# + replan stability, each refused with a message that folds back.
# =====================================================================


def _ct_plan() -> dict:
    return StackPlanner().plan({**CT_BRIEF, "scene": {}})


def test_an_uncovered_manipulate_node_is_refused():
    plan = _ct_plan()
    plan["verify"] = [v for v in plan["verify"] if v["after"] != "pick-milk"]
    ok, msg = validate_plan(plan, CATALOGUE, ORACLES)
    assert not ok and "pick-milk" in msg and "verify" in msg


def test_a_verify_kind_successor_counts_as_coverage():
    plan = {
        "goal": "grasp gated by a verify NODE, no verify-list edge of its own",
        "nodes": [
            {"id": "survey", "kind": "perceive", "skill": "survey", "args": {},
             "after": []},
            {"id": "grasp-0", "skill": "grasp", "args": {"object": "cube"},
             "after": ["survey"]},
            {"id": "check-grasp", "kind": "verify", "skill": "check", "args": {},
             "after": ["grasp-0"]},
        ],
        "verify": [{"after": "check-grasp", "predicate": "lifted"}],
    }
    ok, msg = validate_plan(plan, HET_CATALOGUE, HET_ORACLES)
    assert ok, msg
    # remove the successor: the same manipulate node is now uncovered
    plan["nodes"] = plan["nodes"][:2]
    plan["verify"] = [{"after": "grasp-0", "predicate": "lifted"}]
    ok, _ = validate_plan(plan, HET_CATALOGUE, HET_ORACLES)
    assert ok  # a verify-list edge covers it again
    plan["verify"] = [{"after": "survey", "predicate": "lifted"}]
    ok, msg = validate_plan(plan, HET_CATALOGUE, HET_ORACLES)
    assert not ok and "grasp-0" in msg


def test_a_replan_preserving_done_nodes_passes():
    done = [{"id": "pick-can", "skill": "pick", "args": {"object": "can"}}]
    ok, msg = validate_plan(_ct_plan(), CATALOGUE, ORACLES, done=done)
    assert ok and msg == ""


def test_a_replan_dropping_a_done_node_is_refused():
    plan = _ct_plan()
    plan["nodes"] = [{"id": "pick-milk", "skill": "pick",
                      "args": {"object": "milk"}, "after": []}]
    plan["verify"] = [{"after": "pick-milk", "predicate": "pick_success"}]
    done = [{"id": "pick-can", "skill": "pick", "args": {"object": "can"}}]
    ok, msg = validate_plan(plan, CATALOGUE, ORACLES, done=done)
    assert not ok and "pick-can" in msg and "dropped" in msg


def test_a_replan_rewriting_a_done_node_is_refused():
    plan = _ct_plan()
    plan["nodes"][0]["args"] = {"object": "milk"}
    done = [{"id": "pick-can", "skill": "pick", "args": {"object": "can"}}]
    ok, msg = validate_plan(plan, CATALOGUE, ORACLES, done=done)
    assert not ok and "pick-can" in msg and "rewrote" in msg


class _AmnesiacPlanner(_CountingPlanner):
    """Drops the finished pick-can on its first replan (a model forgetting done
    work); repairs itself once the invalid_plan fault names the dropped node."""

    def plan(self, brief):
        self.briefs.append(dict(brief))
        good = StackPlanner().plan(brief)
        fault = brief.get("fault")
        if fault is not None and fault["kind"] == "node_failure":
            bad = json.loads(json.dumps(good))
            bad["nodes"] = [{**n, "after": []} for n in bad["nodes"]
                            if n["id"] != "pick-can"]
            bad["verify"] = [v for v in bad["verify"] if v["after"] != "pick-can"]
            return bad
        return good


def test_replan_instability_burns_a_replan_and_folds_back(monkeypatch):
    planner = _AmnesiacPlanner()
    kernel = _task_kernel(planner)
    fake = _RolloutFake([True, False, True])
    monkeypatch.setattr(workload, "_governed_rollout", fake)

    out = workload.run(dict(CT_BRIEF), kernel, seed=1, max_actuations=6)

    assert out["success"] is True and out["replans"] == 2
    assert out["faults"][1]["kind"] == "invalid_plan"
    assert "pick-can" in out["faults"][1]["msg"], "the refusal names the dropped node"
    # the validator's own words reached the planner on the next brief
    assert "pick-can" in planner.briefs[2]["fault"]["msg"]
    # finished work never re-dispatched, the unstable graph never actuated
    assert [s.task for s in fake.specs] == ["pickcan", "pickmilk", "pickmilk"]


def test_a_manipulate_only_plan_needs_no_predicates_table():
    # stack/clear_table carry no `kind` and no `predicates`: the machinery is
    # inert for them -- byte-identical to before node kinds existed.
    plan = StackPlanner().plan({**BRIEF, "scene": {}})
    assert all("kind" not in n for n in plan["nodes"])
    ok, _ = validate_plan(plan, CATALOGUE, ORACLES)
    assert ok


def test_an_ungrounded_arg_folds_back_instead_of_crashing(monkeypatch):
    """A dispatch-time grounding refusal (a pick object with no task binding --
    the first thing a real VLM planner fabricated) is the PLANNER's fault: it
    must burn a replan and fold the refusal (naming the known bindings) back
    into the next brief, never crash the loop."""
    class _Ungrounded(_CountingPlanner):
        def plan(self, brief):
            self.briefs.append(dict(brief))
            if brief.get("fault") is None:
                return json.loads(json.dumps({
                    "goal": "pick something",
                    "nodes": [{"id": "pick-0", "skill": "pick",
                               "args": {"object": "vlm"}, "after": []}],
                    "verify": [{"after": "pick-0", "predicate": "pick_success"}],
                }, sort_keys=True))
            return StackPlanner().plan({**brief, "task": "stack"})

    planner = _Ungrounded()
    kernel = _task_kernel(planner)
    fake = _RolloutFake([True])
    monkeypatch.setattr(workload, "_governed_rollout", fake)

    out = workload.run(dict(WBRIEF), kernel, seed=1)

    assert out["success"] is True and out["replans"] == 1
    fault = planner.briefs[1]["fault"]
    assert fault["kind"] == "node_failure" and fault["node"] == "pick-0"
    assert "refused at dispatch" in fault["msg"] and "can" in fault["msg"], \
        "the refusal must name the known bindings so a replan can ground itself"
    assert len(fake.specs) == 1, "the ungrounded node never actuated"


def test_protocol_gate_admits_a_recovery_node_as_an_argless_record():
    """A planner's ``insert_recovery`` answer passes ``_graph_problems`` exactly as
    the plan it grew from: the strategy name is no catalogue skill, so the gate
    folds it as an arg-less, contract-free record (the loop resolves the name)."""
    from harness import protocol
    g = {"goal": "x", "nodes": [
        {"id": "a", "skill": "grasp", "args": {"object": "apple"}, "after": []},
        {"id": "b", "skill": "place", "args": {"object": "apple"}, "after": ["a"]}],
        "verify": [{"after": "b", "predicate": "placed"}]}
    cat = {"grasp": {"object": str}, "place": {"object": str}}
    brief = {"facts": [], "objects": ["apple"]}
    g2 = protocol.insert_recovery(g, "b", "reapproach")
    assert (workload._graph_problems(g2, g, ["a"], brief, cat, 0)
            == workload._graph_problems(g, g, ["a"], brief, cat, 0))


def test_recycle_cans_planner_keeps_a_done_recovery_node_across_faults():
    """A stateless planner must re-emit a recovery node that already RAN (it is in
    done_specs): without it every later replan dies as 'dropped done node'."""
    from plugins.mission_recycle_cans import planner as P
    pl = P.RecycleCansPlanner()
    cat, brief = P.CATALOGUE, {"facts": [], "objects": []}
    g1 = pl.plan({"task": "recycle_cans", "fault": {"kind": "no_progress", "node": "drop-can1"}})
    recovered = next(i for i, node in enumerate(g1["nodes"]) if node["id"] == "recover-drop-can1")
    done = [node["id"] for node in g1["nodes"][:recovered + 1]]
    g2 = pl.plan({"task": "recycle_cans", "fault": {
        "kind": "node_failure", "node": "grasp-can2", "nodes_done": done}})
    assert "recover-drop-can1" in {n["id"] for n in g2["nodes"]}
    assert workload._graph_problems(g2, g1, done, brief, cat, 0) == []
