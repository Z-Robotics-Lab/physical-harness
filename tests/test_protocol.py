"""One test per Legal(G) rule plus the protocol helpers (harness/protocol.py)."""

from __future__ import annotations

import pytest

from harness import protocol as P

GRASP = P.SkillRecordV0(id="r1", name="grasp", args={"object": "entity"},
                        requires=("reachable(object)", "gripper_free()"),
                        ensures=("holding(object)",), clobbers=("gripper_free()",))
PLACE = P.SkillRecordV0(id="r2", name="place", args={"object": "entity", "target": "str"},
                        requires=("holding(object)",),
                        ensures=("in(object,target)", "gripper_free()"),
                        clobbers=("holding(object)",))
LOOK = P.SkillRecordV0(id="r3", name="look", ensures=("visible(pear)",))
RECS = {r.name: r for r in (GRASP, PLACE, LOOK)}
FACTS = {"reachable(apple)", "gripper_free()"}
OBJS = {"apple", "microwave"}


def graph(nodes, goal=("in(apple,microwave)",)):
    return {"mission": "thaw", "seed": 1, "tasks": [{"id": "t", "goal": list(goal)}],
            "nodes": nodes}


GOOD = graph([
    {"id": "g", "task": "t", "skill": "grasp", "args": {"object": "apple"}, "after": []},
    {"id": "p", "task": "t", "skill": "place",
     "args": {"object": "apple", "target": "microwave"}, "after": ["g"]},
])


def test_legal_graph():
    assert P.validate_graph(GOOD, RECS, FACTS, OBJS) == (True, [])


def test_typed():
    bad = graph([{"id": "g", "task": "t", "skill": "grasp",
                  "args": {"object": 3, "extra": 1}, "after": []}])
    ok, problems = P.validate_graph(bad, RECS, FACTS, OBJS)
    assert not ok and any(p.startswith("typed:") for p in problems)


def test_bound():
    two = P.SkillRecordV0.from_dict({**P.to_plain(PLACE), "bindings": {
        "sim": {"policies": {"scripted": {"task": "place"}, "pi05": {"ref": "x:provider"}}}}})
    recs = {**RECS, "place": two}
    ok, problems = P.validate_graph(graph([
        {**GOOD["nodes"][0], "executor": "pi05"}, {**GOOD["nodes"][1], "executor": "pi05"}]),
        recs, FACTS, OBJS)
    assert not ok and len(problems) == 1 and problems[0].startswith("bound: node 'g'")
    assert P.validate_graph(graph([
        {**GOOD["nodes"][0], "executor": "scripted"}, {**GOOD["nodes"][1], "executor": "pi05"}]),
        recs, FACTS, OBJS) == (True, [])


def test_projection_executors():
    ev = {"sim": {"n": 4, "k": 4, "by_executor": {"pi05": {"n": 20, "k": 18}}}}
    two = P.SkillRecordV0.from_dict({**P.to_plain(PLACE), "evidence": ev, "bindings": {"sim": {
        "policies": {"scripted": {"task": "place"},
                     "pi05": {"ref": "x:provider", "checkpoint_sha": "ab" * 32}}}}})
    card = P.vlm_projection({"place": two, "grasp": GRASP}, (), (), (), None,
                            show_evidence=True)["skills"]
    assert card[0]["executors"] == {"scripted": {"evidence": None}}
    ex = card[1]["executors"]
    assert ex["scripted"] == {"evidence": None}            # whole-record row never lent
    assert ex["pi05"]["checkpoint_sha"] == "ab" * 32 and 0.68 < ex["pi05"]["evidence"][0] < 0.9
    hidden = P.vlm_projection({"place": two}, (), (), (), None)["skills"][0]["executors"]
    assert hidden["pi05"]["evidence"] is None
    assert "executor" in P.VLM_OUTPUT_SCHEMA["nodes"][0]


def test_grounded():
    bad = graph([{"id": "g", "task": "t", "skill": "grasp", "args": {"object": "pear"},
                  "after": []}], goal=("holding(pear)",))
    ok, problems = P.validate_graph(bad, RECS, FACTS | {"reachable(pear)"}, OBJS)
    assert not ok and problems == [
        "grounded: node 'g' arg object='pear' is not in sigma0.objects nor produced by a predecessor"]
    # produced by a predecessor -> grounded
    good = graph([{"id": "l", "task": "t", "skill": "look", "args": {}, "after": []},
                  {"id": "g", "task": "t", "skill": "grasp", "args": {"object": "pear"},
                   "after": ["l"]}], goal=("holding(pear)",))
    assert P.validate_graph(good, RECS, FACTS | {"reachable(pear)"}, OBJS)[0]


def test_supported():
    ok, problems = P.validate_graph(GOOD, RECS, {"gripper_free()"}, OBJS)
    assert not ok and problems == [
        "supported: node 'g' requires reachable(apple) which nothing provides"]


def test_threat_incomparable_node():
    # a second grasp unordered w.r.t. place clobbers gripper_free() -> threat to g
    bad = graph([
        {"id": "g", "task": "t", "skill": "grasp", "args": {"object": "apple"}, "after": []},
        {"id": "g2", "task": "t", "skill": "grasp", "args": {"object": "apple"}, "after": []},
    ], goal=("holding(apple)",))
    ok, problems = P.validate_graph(bad, RECS, FACTS, OBJS)
    assert not ok and any(p.startswith("supported:") and "threatened" in p for p in problems)
    # same clobber ordered strictly after the consumer is no threat
    ordered = graph([
        {"id": "g", "task": "t", "skill": "grasp", "args": {"object": "apple"}, "after": []},
        {"id": "p", "task": "t", "skill": "place",
         "args": {"object": "apple", "target": "microwave"}, "after": ["g"]},
        {"id": "g2", "task": "t", "skill": "grasp", "args": {"object": "apple"}, "after": ["p"]},
    ], goal=("holding(apple)",))
    assert P.validate_graph(ordered, RECS, FACTS, OBJS)[0]


def test_covered():
    ok, problems = P.validate_graph(graph(GOOD["nodes"], goal=("visible(pear)",)),
                                    RECS, FACTS, OBJS)
    assert problems == ["covered: task 't' goal visible(pear) is ensured by none of its nodes"]
    # goal holding(apple) is clobbered by place before task end
    ok, problems = P.validate_graph(graph(GOOD["nodes"], goal=("holding(apple)",)),
                                    RECS, FACTS, OBJS)
    assert problems == ["covered: task 't' goal holding(apple) is clobbered before task end"]


def test_replan_monotone():
    new = graph([dict(GOOD["nodes"][0], args={"object": "microwave"})])
    ok, problems = P.replan_monotone(GOOD, new, ["g", "p"])
    assert not ok and len(problems) == 2
    assert P.replan_monotone(GOOD, GOOD, ["g"]) == (True, [])


def test_replan_cannot_credit_completed_restore_after_a_future_clobber():
    records = {
        "restore": P.SkillRecordV0(id="restore", name="restore", ensures=("ready()",)),
        "clear": P.SkillRecordV0(id="clear", name="clear", clobbers=("ready()",)),
        "consume": P.SkillRecordV0(id="consume", name="consume", requires=("ready()",)),
    }
    restore = {"id": "a", "task": "t", "skill": "restore", "after": []}
    clear = {"id": "b", "task": "t", "skill": "clear", "after": ["a"]}
    old = graph([restore, clear], goal=())
    new = graph([{**clear, "after": []}, {**restore, "after": ["b"]},
                 {"id": "c", "task": "t", "skill": "consume", "after": ["a"]}], goal=())
    assert P.validate_graph(old, records, (), ()) == (True, [])
    assert P.validate_graph(new, records, (), ()) == (True, [])
    # a already succeeded; replay would skip it and run clear directly before consume.
    ok, problems = P.replan_monotone(old, new, {"a"})
    assert not ok and any("ordered prefix" in p for p in problems)


@pytest.mark.parametrize("order", [("b", "a", "c"), ("c", "a", "b")])
def test_replan_preserves_completed_order_even_without_dependency_edges(order):
    nodes = {nid: {"id": nid, "task": "t", "skill": "s", "after": []}
             for nid in ("a", "b", "c")}
    old = graph(list(nodes.values()), goal=())
    new = graph([nodes[nid] for nid in order], goal=())
    assert P.validate_graph(new, {"s": P.SkillRecordV0(id="s", name="s")}, (), ()) == (True, [])
    ok, problems = P.replan_monotone(old, new, {"a", "b"})
    assert not ok and any("ordered prefix" in p for p in problems)


def test_replan_completed_prefix_cannot_depend_on_later_unfinished_work():
    old = graph([{"id": nid, "task": "t", "skill": "s", "after": []}
                 for nid in ("a", "b")], goal=())
    new = graph([{**old["nodes"][0], "after": ["b"]}, old["nodes"][1]], goal=())
    ok, problems = P.replan_monotone(old, new, {"a"})
    assert not ok and any("depends on unfinished" in p for p in problems)


@pytest.mark.parametrize("field,value", [("kind", "verify"), ("task", "other"),
                                          ("executor", "replacement")])
def test_replan_freezes_completed_execution_identity(field, value):
    old = graph([{"id": "a", "task": "t", "skill": "s", "kind": "segment",
                  "executor": "original", "after": []}], goal=())
    new = graph([{**old["nodes"][0], field: value}], goal=())
    # Both raw mappings and converted graphs must retain the dispatch kind.
    for before, after in ((old, new), (P.ExecutionGraph.from_dict(old),
                                      P.ExecutionGraph.from_dict(new))):
        ok, problems = P.replan_monotone(before, after, {"a"})
        assert not ok and any("execution identity" in p for p in problems)


def test_replan_can_replace_unfinished_suffix_after_completed_prefix():
    old = graph([{"id": nid, "task": "t", "skill": "s", "after": []}
                 for nid in ("a", "b", "c")], goal=())
    new = graph([*old["nodes"][:2], {"id": "repair", "task": "t", "skill": "repair",
                                   "kind": "recovery", "after": ["b"]},
                 {**old["nodes"][2], "after": ["repair"]}], goal=())
    assert P.replan_monotone(old, new, {"b", "a"}) == (True, [])


def test_default_node_serialization_preserves_existing_graph_identity():
    default = P.Node(id="n", task="t", skill="s")
    previous = {"id": "n", "task": "t", "skill": "s", "args": {}, "after": [],
                "on_fail": {}, "executor": None}
    assert P.to_plain(default) == previous
    assert P.content_id(default) == P.content_id(previous)
    segment = P.Node(id="n", task="t", skill="s", kind="segment")
    assert P.to_plain(segment) == {**previous, "kind": "segment"}
    assert P.content_id(segment) != P.content_id(default)


def test_content_id_stable():
    g = P.ExecutionGraph.from_dict(GOOD)
    assert P.content_id(g) == P.content_id(P.ExecutionGraph.from_dict(GOOD))
    assert P.content_id(g) == P.content_id(P.to_plain(g))
    assert P.content_id(g) != P.content_id(P.ExecutionGraph.from_dict(dict(GOOD, seed=2)))
    t = P.Trajectory(x={"mission": "thaw"}, y={"graph": "abc"}, o={"legal": True})
    assert t.id == P.content_id({"x": t.x, "y": t.y})


def test_three_valued_eval():
    pred = P.PredicateRecord(id="p", name="holding", args=("apple",), reads=("grasped",))
    assert P.eval_predicate(pred, {}, lambda s: s["grasped"]) is None
    assert P.eval_predicate(pred, {"grasped": 1}, lambda s: s["grasped"]) is True
    assert P.all3([True, None]) is None and P.all3([None, False]) is False and P.all3([]) is True
    assert P.fault_from_verify(P.VerifyEvent("g", {"a()": True})) is None
    assert P.fault_from_verify(P.VerifyEvent("g", {"a()": True, "b()": None})).failed == ("b()",)


def test_pred_ref_str():
    assert P.pred_ref_str(" holding( apple , x )") == "holding(apple,x)"
    assert P.pred_ref_str({"name": "free"}) == P.pred_ref_str(("free",)) == "free()"
    assert P.instantiate("in(object,target)", {"object": "apple"}) == "in(apple,target)"
    with pytest.raises(ValueError):
        P.parse_pred_ref("1bad(")


def test_replan_progress_allows_one_as_is_retry_then_no_progress():
    g = {"goal": "x", "nodes": [{"id": "a", "skill": "grasp", "args": {"object": "apple"},
                                 "after": [], "executor": "scripted"}], "verify": []}
    row = {"graph_sha": P.graph_sha(g), "node": "a", "signature": "node_failure"}
    fault = {"kind": "node_failure", "node": "a"}
    assert P.replan_progress([], g, fault) == (True, "fresh")
    assert P.replan_progress([row], g, fault) == (True, "retry")
    assert P.replan_progress([row, row], g, fault) == (False, "no_progress")
    # a hinted no_progress fault carries the original signature forward
    assert P.replan_progress([row, row], g, {**fault, "kind": "no_progress",
                                             "signature": "node_failure"})[1] == "no_progress"
    # another executor / args = a different graph; another node or signature = fresh
    alt = {**g, "nodes": [{**g["nodes"][0], "executor": "pi05"}]}
    assert P.replan_progress([row, row], alt, fault) == (True, "fresh")
    assert P.replan_progress([row, row], g, {"kind": "node_failure", "node": "b"}) == (True, "fresh")
    assert P.replan_progress([row, row], g, {"kind": "invalid_plan"}) == (True, "fresh")


def test_insert_recovery_is_the_progress_a_no_progress_fault_asks_for():
    from plugins.task.validate import plan_to_graph, validate_plan
    g = {"goal": "x", "nodes": [
        {"id": "a", "skill": "grasp", "args": {"object": "apple"}, "after": []},
        {"id": "b", "skill": "place", "args": {"object": "apple"}, "after": ["a"]}],
        "verify": [{"after": "a", "predicate": "held"}, {"after": "b", "predicate": "placed"}]}
    cat = {"grasp": {"object": str}, "place": {"object": str}}
    oracles = {"held": {}, "placed": {}}
    fault = {"kind": "no_progress", "node": "b", "signature": "node_failure"}
    rows = [{"graph_sha": P.graph_sha(g), "node": "b", "signature": "node_failure"}] * 2
    assert P.replan_progress(rows, g, fault) == (False, "no_progress")
    g2 = P.insert_recovery(g, "b", "reapproach")
    assert [n["id"] for n in g2["nodes"]] == ["a", "recover-b", "b"]
    assert g2["nodes"][1] == {"id": "recover-b", "skill": "reapproach", "args": {},
                              "kind": "recovery", "after": ["a"]}
    assert g2["nodes"][2]["after"] == ["a", "recover-b"]
    assert P.replan_progress(rows, g2, fault) == (True, "fresh")
    assert P.replan_monotone(plan_to_graph(g), plan_to_graph(g2), ["a"]) == (True, [])
    # the strategy name is not a catalogue skill, and a recovery node takes no args
    assert validate_plan(g2, cat, oracles) == (True, "")
    assert not validate_plan({**g2, "nodes": [{**g2["nodes"][1], "args": {"x": 1}}, *g2["nodes"][::2]]},
                             cat, oracles)[0]
    assert P.insert_recovery(g2, "b", "reapproach") == g2     # idempotent
    with pytest.raises(KeyError):
        P.insert_recovery(g, "zz", "reapproach")


@pytest.mark.parametrize("name,kind,want", [
    ("grasp_can1", "segment", "grasp"), ("nav_fridge", "segment", "nav"),
    ("place_meat", "segment", "place"), ("drop_can1", "segment", "drop"),
    ("pack_hot0", "segment", "pack"), ("carry", "segment", "carry"),
    ("v_at_can1", "verify", "verify"), ("plan", "decide", "decide"), ("survey", "perceive", "perceive"),
])
def test_skill_class_derivation(name, kind, want):
    assert P.skill_class(P.SkillRecordV0(id=name, name=name, kind=kind)) == want
    declared = P.SkillRecordV0.from_dict({"id": name, "name": name, "kind": kind, "class": "x"})
    assert P.skill_class(declared) == "x" and P.to_plain(declared)["class"] == "x"
    assert "class_" not in P.to_plain(declared)


PLAN = P.PlanRecord(id="p1", task="thaw", goal=("in(apple,microwave)",), graph=GOOD,
                    embodiment="e", arm="scripted", evidence={}, rule={})


def test_skill_dependencies_causal_and_uses():
    deps = P.skill_dependencies([GRASP, PLACE, LOOK, PLAN])
    assert ("place", "grasp", "causal") in deps          # place requires holding, grasp ensures it
    assert ("grasp", "place", "causal") not in deps      # gripper_free() is zero-arity: a resource, no edge
    assert ("look", "grasp", "causal") not in deps and not any(s == d for s, d, _ in deps)
    assert [e for e in deps if e[2] == "uses"] == [("p1", "grasp", "uses"), ("p1", "place", "uses")]
    assert P.skill_dependencies({"g": GRASP, "p": P.to_plain(PLAN)}) == [("p1", "grasp", "uses"),
                                                                          ("p1", "place", "uses")]


def test_skill_dependencies_position_wise_only():
    def rec(name, req=(), ens=()):
        return P.SkillRecordV0(id=name, name=name, requires=req, ensures=ens)
    zero = [rec("a", req=("gripper_free()",)), rec("b", ens=("gripper_free()",))]
    assert P.skill_dependencies(zero) == []                                   # zero-arity ignored
    ground = [rec("carry_can1", req=("holding(can1)",)), rec("grasp_can1", ens=("holding(can1)",)),
              rec("grasp_can2", ens=("holding(can2)",))]
    assert P.skill_dependencies(ground) == [("carry_can1", "grasp_can1", "causal")]   # ground match
    var = [rec("carry", req=("holding(object)",)), rec("grasp", ens=("holding(object)",)),
           rec("lift", ens=("holding(item)",))]
    assert P.skill_dependencies(var) == [("carry", "grasp", "causal")]       # same variable name


def test_skill_instances_name_prefix_within_class():
    def rec(name, cls=""):
        return P.SkillRecordV0(id=name, name=name, class_=cls)
    recs = [rec("grasp"), rec("grasp_can"), rec("grasp_can_left"), rec("grasp_can1"),
            rec("nav_fridge"), rec("navigate"), rec("place_meat", cls="grasp")]
    assert P.skill_instances(recs) == [("grasp_can", "grasp"), ("grasp_can_left", "grasp_can"),
                                       ("grasp_can1", "grasp")]   # longest generic wins; class must match
    assert P.skill_instances({r.name: P.to_plain(r) for r in recs[:2]}) == [("grasp_can", "grasp")]


def test_skill_benchmarks_membership():
    bound = {n: P.SkillRecordV0(id=n, name=n, bindings={"e": {}}) for n in ("grasp", "place", "look")}
    cards = {"b_tasks": {"embodiment": "e", "tasks": ["thaw"]},
             "b_all": {"embodiment": "e", "tasks": []},
             "b_other": {"embodiment": "other", "tasks": ["thaw"]}}
    out = P.skill_benchmarks([*bound.values(), PLAN], cards)
    assert out == {"grasp": ["b_all", "b_tasks"], "place": ["b_all", "b_tasks"], "look": ["b_all"]}
    assert P.skill_benchmarks([GRASP, PLAN], cards) == {"grasp": []}     # unbound: nowhere


def test_skill_helpers_take_raw_record_dicts():
    raw = {n: P.to_plain(r) for n, r in RECS.items()}
    assert P.skill_class(raw["grasp"]) == "grasp"
    assert ("place", "grasp", "causal") in P.skill_dependencies(raw)
    assert P.skill_benchmarks(raw, {"b": {"embodiment": "e", "tasks": []}}) == {n: [] for n in raw}
