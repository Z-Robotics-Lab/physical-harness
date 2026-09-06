"""Every offered graph edit is executable without gaining evaluation authority."""

from copy import deepcopy

import pytest

from plugins.rsi.evaluation import compile_contract, evaluate
from plugins.rsi.interventions import plan_space, validate_candidate

BRIEF = {"catalogue": {"move": {"target": str}, "place": {"object": str, "target": str},
                        "positioned": {"target": str}, "placed": {"object": str, "target": str}},
         "oracles": ["positioned", "placed"]}
REFERENCE = {"goal": "Put the object on its assigned shelf", "nodes": [
    {"id": "dock", "kind": "segment", "skill": "move", "args": {"target": "shelf"}, "after": []},
    {"id": "place", "kind": "segment", "skill": "place",
     "args": {"object": "item", "target": "shelf"}, "after": ["dock"]},
    {"id": "check", "kind": "verify", "skill": "placed",
     "args": {"object": "item", "target": "shelf"}, "after": ["place"]}],
    "verify": [{"after": "dock", "predicate": "positioned"},
               {"after": "place", "predicate": "placed"}]}


def inserted():
    graph = deepcopy(REFERENCE)
    graph["nodes"].insert(1, {"id": "approach", "kind": "segment", "skill": "move",
                               "args": {"target": "shelf"}, "after": ["dock"]})
    graph["nodes"][2]["after"] = ["approach"]
    graph["verify"].append({"after": "approach", "predicate": "positioned"})
    return graph


def test_derived_space_lists_only_installed_capabilities_and_validates_the_offered_insertion():
    space = plan_space(REFERENCE, BRIEF)
    assert space["graph"] == REFERENCE and set(space["skills"]) == set(BRIEF["catalogue"])
    assert space["skills"]["move"] == {"target": "str"}
    validate_candidate(inserted(), REFERENCE, BRIEF)
    space["graph"]["nodes"][0]["args"]["target"] = "changed"
    assert REFERENCE["nodes"][0]["args"]["target"] == "shelf"  # no aliasing server authority


@pytest.mark.parametrize("edit", [
    lambda graph: graph.update(goal="Stand still"),
    lambda graph: graph["nodes"][2]["args"].update(target="under-the-robot"),
    lambda graph: graph["nodes"][3]["args"].update(target="under-the-robot"),
    lambda graph: graph["verify"].__setitem__(1, {"after": "place", "predicate": "positioned"}),
    lambda graph: graph["nodes"].pop(2),
])
def test_retargeting_dropping_work_and_replacing_a_verifier_are_refused(edit):
    graph = inserted()
    edit(graph)
    with pytest.raises(ValueError):
        validate_candidate(graph, REFERENCE, BRIEF)


def test_insertions_still_require_installed_skills_machine_checks_and_legal_dependencies():
    for mutation in (
        lambda graph: graph["nodes"][1].update(skill="invented"),
        lambda graph: graph["verify"].pop(),
        lambda graph: graph["nodes"][1].update(after=["check"]),
        lambda graph: graph["nodes"][1].update(kind="verify", skill="positioned"),
    ):
        graph = inserted()
        mutation(graph)
        with pytest.raises(ValueError):
            validate_candidate(graph, REFERENCE, BRIEF)


def test_adding_a_valid_action_does_not_change_the_frozen_objective_or_its_reward():
    graph = inserted()
    validate_candidate(graph, REFERENCE, BRIEF)
    ruler = compile_contract(REFERENCE, predicates={"placed": "task_oracles:placed"})
    same_ruler = compile_contract(graph, predicates={"placed": "task_oracles:placed"})
    assert same_ruler["sha"] == ruler["sha"]
    observation = {"node": REFERENCE["nodes"][-1], "authority": "predicate",
                   "source": "task_oracles:placed", "success": False,
                   "evidence_policy": "world-dependencies-v1", "blocked_reads": []}
    before = evaluate(ruler, [observation])
    extra = {"node": graph["nodes"][1], "authority": "predicate",
             "source": "task_oracles:placed", "success": True,
             "evidence_policy": "world-dependencies-v1", "blocked_reads": []}
    after = evaluate(ruler, [observation, extra] * 20)
    assert after == before and after["passed"] == 0


def test_a_candidate_cannot_reorder_existing_checkpoints():
    graph = deepcopy(REFERENCE)
    graph["nodes"] = [graph["nodes"][1], graph["nodes"][0], graph["nodes"][2]]
    graph["nodes"][0]["after"] = []
    graph["nodes"][1]["after"] = ["place"]
    graph["nodes"][2]["after"] = ["dock"]
    with pytest.raises(ValueError):
        validate_candidate(graph, REFERENCE, BRIEF)
