"""The unified RoboCasa skill graph adapter (harness/unified_skill_graph.py).

Read-only over the generated document: loader refusals (schema, endpoints,
duplicates, IS_A / DECOMPOSES_TO cycles), deterministic retrieval that returns a
compact subtree rather than the whole graph, and server-side expansion that
keeps HAS_STAGE order, follows DECOMPOSES_TO recursively, and refuses unknown
skills. The real graph file is used where it exists; the synthetic fixtures
below never touch the datasets.
"""

from __future__ import annotations

import copy
import json

import pytest

from harness import unified_skill_graph as usg
from harness.unified_skill_graph import SkillGraphError, UnifiedSkillGraph

# --- a tiny graph in the real schema ------------------------------------------


def _node(id_, kind, **extra):
    return {"id": id_, "name": id_.split(":", 1)[1].replace(":", "."), "kind": kind, **extra}


def _graph() -> dict:
    return {
        "schema_version": "1.0",
        "root": "concept:RobotSkill",
        "relation_semantics": {},
        "nodes": [
            _node("concept:RobotSkill", "root"),
            _node("concept:CompositeSkill", "category"),
            _node("concept:Manipulation", "category"),
            _node("concept:ObjectManipulation", "category"),
            _node("concept:DeviceInteraction", "category"),
            _node("concept:Pick", "canonical_skill"),
            _node("concept:Place", "canonical_skill"),
            _node("concept:Transport", "canonical_skill"),
            _node("concept:PressButton", "canonical_skill"),
            _node("concept:TransferObject", "canonical_skill"),
            {"id": "skill:CoffeeSetupMug", "name": "CoffeeSetupMug", "kind": "observed_skill",
             "executable": True, "taxonomy_parent": "TransferObject",
             "evidence": {"datasets": ["PrepareCoffee"],
                          "instructions": {"pick up the mug from the cabinet": 1,
                                           "place the mug under the coffee machine dispenser": 1}}},
            {"id": "skill:StartCoffeeMachine", "name": "StartCoffeeMachine",
             "kind": "observed_skill", "executable": True, "taxonomy_parent": "CompositeSkill",
             "evidence": {"datasets": ["PrepareCoffee"],
                          "instructions": {"press the start button on the coffee machine": 1}}},
            {"id": "skill:OpenFridge", "name": "OpenFridge", "kind": "observed_skill",
             "executable": True, "taxonomy_parent": "Manipulation",
             "evidence": {"datasets": ["Thaw"], "instructions": {"open the fridge door": 1}}},
            {"id": "stage:CoffeeSetupMug:pick", "name": "CoffeeSetupMug.pick",
             "kind": "observed_stage", "skill": "CoffeeSetupMug", "stage": "pick"},
            {"id": "stage:CoffeeSetupMug:place", "name": "CoffeeSetupMug.place",
             "kind": "observed_stage", "skill": "CoffeeSetupMug", "stage": "place"},
            {"id": "stage:StartCoffeeMachine:execute", "name": "StartCoffeeMachine.execute",
             "kind": "observed_stage", "skill": "StartCoffeeMachine", "stage": "execute"},
            {"id": "stage:Pick:pick", "name": "Pick.pick", "kind": "observed_stage",
             "skill": "Pick", "stage": "pick"},
            {"id": "terminal:done", "name": "done", "kind": "terminal"},
        ],
        "edges": [
            {"source": "concept:CompositeSkill", "target": "concept:RobotSkill", "relation": "IS_A"},
            {"source": "concept:Manipulation", "target": "concept:RobotSkill", "relation": "IS_A"},
            {"source": "concept:ObjectManipulation", "target": "concept:Manipulation", "relation": "IS_A"},
            {"source": "concept:DeviceInteraction", "target": "concept:Manipulation", "relation": "IS_A"},
            {"source": "concept:Pick", "target": "concept:ObjectManipulation", "relation": "IS_A"},
            {"source": "concept:Place", "target": "concept:ObjectManipulation", "relation": "IS_A"},
            {"source": "concept:Transport", "target": "concept:ObjectManipulation", "relation": "IS_A"},
            {"source": "concept:PressButton", "target": "concept:DeviceInteraction", "relation": "IS_A"},
            {"source": "concept:TransferObject", "target": "concept:CompositeSkill", "relation": "IS_A"},
            {"source": "skill:CoffeeSetupMug", "target": "concept:TransferObject", "relation": "IS_A"},
            {"source": "skill:StartCoffeeMachine", "target": "concept:CompositeSkill", "relation": "IS_A"},
            {"source": "skill:OpenFridge", "target": "concept:Manipulation", "relation": "IS_A"},
            {"source": "concept:TransferObject", "target": "concept:Pick", "relation": "DECOMPOSES_TO", "order": 0},
            {"source": "concept:TransferObject", "target": "concept:Transport", "relation": "DECOMPOSES_TO", "order": 1},
            {"source": "concept:TransferObject", "target": "concept:Place", "relation": "DECOMPOSES_TO", "order": 2},
            {"source": "skill:CoffeeSetupMug", "target": "concept:Pick", "relation": "DECOMPOSES_TO", "order": 0},
            {"source": "skill:CoffeeSetupMug", "target": "concept:Place", "relation": "DECOMPOSES_TO", "order": 1},
            {"source": "skill:StartCoffeeMachine", "target": "concept:PressButton", "relation": "DECOMPOSES_TO", "order": 0},
            # HAS_STAGE deliberately listed out of order: the adapter sorts by `order`.
            {"source": "skill:CoffeeSetupMug", "target": "stage:CoffeeSetupMug:place", "relation": "HAS_STAGE", "order": 1},
            {"source": "skill:CoffeeSetupMug", "target": "stage:CoffeeSetupMug:pick", "relation": "HAS_STAGE", "order": 0},
            {"source": "skill:StartCoffeeMachine", "target": "stage:StartCoffeeMachine:execute", "relation": "HAS_STAGE", "order": 0},
            {"source": "concept:Pick", "target": "stage:Pick:pick", "relation": "HAS_STAGE", "order": 0},
            {"source": "stage:CoffeeSetupMug:pick", "target": "concept:Pick", "relation": "REALIZES"},
            {"source": "stage:CoffeeSetupMug:place", "target": "concept:Place", "relation": "REALIZES"},
            {"source": "stage:Pick:pick", "target": "concept:Pick", "relation": "REALIZES"},
            {"source": "stage:CoffeeSetupMug:pick", "target": "stage:CoffeeSetupMug:place",
             "relation": "OBSERVED_TRANSITION", "count": 514},
            {"source": "stage:CoffeeSetupMug:place", "target": "stage:StartCoffeeMachine:execute",
             "relation": "OBSERVED_TRANSITION", "count": 514},
            {"source": "stage:StartCoffeeMachine:execute", "target": "terminal:done",
             "relation": "OBSERVED_TRANSITION", "count": 514},
        ],
    }


# --- loader -------------------------------------------------------------------


def test_valid_graph_loads_and_reports_provenance(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps(_graph()))
    g = UnifiedSkillGraph.load(path)
    prov = g.provenance()
    assert prov["schema_version"] == "1.0" and prov["nodes"] == 18
    assert prov["relations"] == {"DECOMPOSES_TO": 6, "HAS_STAGE": 4, "IS_A": 12,
                                 "OBSERVED_TRANSITION": 3, "REALIZES": 3}
    assert len(prov["sha256"]) == 64 and prov["source"] == str(path)
    assert [n.name for n in g.skills()] == [
        "CoffeeSetupMug", "OpenFridge", "Pick", "Place", "PressButton",
        "StartCoffeeMachine", "TransferObject", "Transport"]


def test_missing_edge_endpoint_is_rejected():
    data = _graph()
    data["edges"].append({"source": "skill:CoffeeSetupMug", "target": "concept:Ghost",
                          "relation": "IS_A"})
    with pytest.raises(SkillGraphError, match="target 'concept:Ghost' is not a node"):
        UnifiedSkillGraph(data)


def test_duplicate_node_is_rejected():
    data = _graph()
    data["nodes"].append(copy.deepcopy(data["nodes"][5]))  # concept:Pick again
    with pytest.raises(SkillGraphError, match="duplicate node id 'concept:Pick'"):
        UnifiedSkillGraph(data)


def test_is_a_cycle_is_rejected():
    data = _graph()
    data["edges"].append({"source": "concept:RobotSkill", "target": "concept:Pick",
                          "relation": "IS_A"})
    with pytest.raises(SkillGraphError, match="IS_A cycle"):
        UnifiedSkillGraph(data)


def test_decomposes_to_cycle_is_rejected():
    data = _graph()
    data["edges"].append({"source": "concept:Pick", "target": "concept:TransferObject",
                          "relation": "DECOMPOSES_TO", "order": 0})
    with pytest.raises(SkillGraphError, match="DECOMPOSES_TO cycle"):
        UnifiedSkillGraph(data)


def test_unknown_relation_schema_and_missing_file_are_loud(tmp_path):
    data = _graph()
    data["edges"][0]["relation"] = "CAUSES"
    with pytest.raises(SkillGraphError, match="relation 'CAUSES'"):
        UnifiedSkillGraph(data)
    data = _graph()
    data["schema_version"] = "2.0"
    with pytest.raises(SkillGraphError, match="unsupported schema_version"):
        UnifiedSkillGraph(data)
    with pytest.raises(SkillGraphError, match="not readable"):
        UnifiedSkillGraph.load(tmp_path / "absent.json")


def test_graph_path_is_configurable_not_hardcoded(monkeypatch, tmp_path):
    monkeypatch.delenv(usg.GRAPH_PATH_ENV, raising=False)
    assert usg.default_graph_path() == usg.DEFAULT_GRAPH_PATH
    assert not usg.DEFAULT_GRAPH_PATH.is_absolute() or str(usg.DEFAULT_GRAPH_PATH).startswith(
        str(usg._REPO.parent))
    override = tmp_path / "elsewhere.json"
    monkeypatch.setenv(usg.GRAPH_PATH_ENV, str(override))
    assert usg.default_graph_path() == override


# --- queries ------------------------------------------------------------------


def test_taxonomy_parent_and_path_follow_is_a_only():
    g = UnifiedSkillGraph(_graph())
    assert g.canonical_parent("CoffeeSetupMug") == "TransferObject"
    assert g.taxonomy_path("CoffeeSetupMug") == (
        "RobotSkill", "CompositeSkill", "TransferObject", "CoffeeSetupMug")
    assert g.taxonomy_path("PressButton") == (
        "RobotSkill", "Manipulation", "DeviceInteraction", "PressButton")
    assert g.canonical_parent("RobotSkill") is None


def test_stages_are_ordered_and_realize_canonicals():
    g = UnifiedSkillGraph(_graph())
    stages = g.stages("CoffeeSetupMug")
    assert [s["stage"] for s in stages] == ["pick", "place"]        # order, not file order
    assert [s["realizes"] for s in stages] == ["Pick", "Place"]
    assert g.decomposition("TransferObject") == ("Pick", "Transport", "Place")
    # observed transitions are descriptive adjacency, kept apart from the recipe
    nxt = g.observed_transitions("stage:CoffeeSetupMug:place")
    assert nxt == ({"target": "StartCoffeeMachine.execute",
                    "target_id": "stage:StartCoffeeMachine:execute",
                    "count": 514, "episode_support": None},)


def test_library_snapshot_is_a_tree_with_embedded_stages_and_separate_recipes():
    snap = UnifiedSkillGraph(_graph()).library_snapshot()
    assert snap["root"] == "concept:RobotSkill"
    assert len(snap["nodes"]) == 13  # stages and terminal are not tree nodes
    by_name = {node["name"]: node for node in snap["nodes"]}
    coffee = by_name["CoffeeSetupMug"]
    assert coffee["parent"] == "concept:TransferObject"
    assert [stage["stage"] for stage in coffee["stages"]] == ["pick", "place"]
    assert coffee["decomposition"] == ["Pick", "Place"]
    assert snap["recipes"] == [
        {"skill": "CoffeeSetupMug", "steps": ["Pick", "Place"]},
        {"skill": "StartCoffeeMachine", "steps": ["PressButton"]},
        {"skill": "TransferObject", "steps": ["Pick", "Transport", "Place"]},
    ]


# --- retrieval ----------------------------------------------------------------


def test_prepare_coffee_retrieves_the_coffee_skills_first():
    g = UnifiedSkillGraph(_graph())
    ranked = g.retrieve("Prepare a cup of coffee.")
    assert [r["name"] for r in ranked][:2] == ["CoffeeSetupMug", "StartCoffeeMachine"]
    assert ranked[0]["matched"] == ["coffee", "prepare"]
    assert "OpenFridge" not in {r["name"] for r in ranked}


def test_retrieval_returns_a_compact_subtree_not_the_whole_graph():
    g = UnifiedSkillGraph(_graph())
    sub = g.retrieve_subtree("Prepare a cup of coffee.")
    names = [r["name"] for r in sub["skills"]]
    assert names[:2] == ["CoffeeSetupMug", "StartCoffeeMachine"]
    # the closure explains the seeds: canonical components, not every skill
    assert {"Pick", "Place", "PressButton", "TransferObject"} <= set(names)
    assert "OpenFridge" not in names
    assert len(names) < sub["total_skills"] == 8
    assert sub["categories"] == ["CompositeSkill", "RobotSkill"]


def test_retrieval_is_stable_across_calls_and_node_order():
    a = UnifiedSkillGraph(_graph())
    shuffled = _graph()
    shuffled["nodes"].reverse()
    shuffled["edges"].reverse()
    b = UnifiedSkillGraph(shuffled)
    q = "Prepare a cup of coffee."
    assert a.retrieve_subtree(q) == b.retrieve_subtree(q) == a.retrieve_subtree(q)


# --- expansion ----------------------------------------------------------------


def test_coffee_setup_mug_expands_to_pick_then_place():
    g = UnifiedSkillGraph(_graph())
    leaves = g.expand("CoffeeSetupMug")
    assert [leaf.label for leaf in leaves] == ["CoffeeSetupMug.pick", "CoffeeSetupMug.place"]
    assert [leaf.canonical for leaf in leaves] == ["Pick", "Place"]
    assert {leaf.via for leaf in leaves} == {"HAS_STAGE+REALIZES"}


def test_start_coffee_machine_expands_to_press_button():
    g = UnifiedSkillGraph(_graph())
    (leaf,) = g.expand("StartCoffeeMachine")
    assert leaf.label == "StartCoffeeMachine.execute"
    # no REALIZES edge: the ontology recipe's same-position component stands in,
    # and the leaf says so
    assert leaf.canonical == "PressButton" and leaf.via == "HAS_STAGE+DECOMPOSES_TO[order]"
    assert g.decomposition("StartCoffeeMachine") == ("PressButton",)


def test_composite_canonical_expands_recursively_in_order():
    g = UnifiedSkillGraph(_graph())
    labels = [leaf.label for leaf in g.expand("TransferObject")]
    # Pick has its own annotated stage; Transport and Place stand as canonical leaves
    assert labels == ["Pick.pick", "Transport", "Place"]
    assert [leaf.label for leaf in usg.leaf_chain(g, ["CoffeeSetupMug", "StartCoffeeMachine"])] == [
        "CoffeeSetupMug.pick", "CoffeeSetupMug.place", "StartCoffeeMachine.execute"]


def test_expand_refuses_unknown_skills_categories_and_cycles():
    g = UnifiedSkillGraph(_graph())
    with pytest.raises(SkillGraphError, match="unknown skill graph node 'MakeEspresso'"):
        g.expand("MakeEspresso")
    with pytest.raises(SkillGraphError, match="is a category, not a selectable skill"):
        g.expand("ObjectManipulation")
    # a cycle smuggled past construction (monkeypatched adjacency) is still caught
    data = _graph()
    cyclic = UnifiedSkillGraph(data)
    cyclic._out["concept:Pick"]["DECOMPOSES_TO"] = [
        {"source": "concept:Pick", "target": "concept:TransferObject",
         "relation": "DECOMPOSES_TO", "order": 0}]
    cyclic._out["concept:Pick"].pop("HAS_STAGE")
    with pytest.raises(SkillGraphError, match="DECOMPOSES_TO cycle"):
        cyclic.expand("TransferObject")


# --- the real generated graph, when present -----------------------------------


@pytest.mark.skipif(not usg.DEFAULT_GRAPH_PATH.exists(),
                    reason="generated unified_skill_graph.json not present in this checkout")
def test_real_graph_loads_and_expands_the_coffee_chain():
    g = usg.load_graph()
    assert g.provenance()["relations"].keys() == usg.RELATIONS
    chain = usg.leaf_chain(g, ["CoffeeSetupMug", "StartCoffeeMachine"])
    assert [leaf.label for leaf in chain] == [
        "CoffeeSetupMug.pick", "CoffeeSetupMug.place", "StartCoffeeMachine.execute"]
    assert [leaf.canonical for leaf in chain] == ["Pick", "Place", "PressButton"]
    seeds = [r["name"] for r in g.retrieve("Prepare a cup of coffee.")]
    assert seeds[:2] == ["CoffeeSetupMug", "StartCoffeeMachine"]
    # the loader cache hands back the same object until the file changes
    assert usg.load_graph() is g
