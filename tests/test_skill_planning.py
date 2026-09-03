"""Natural language -> skill plan (plugins/task/skill_planning.py).

Every model reply here comes from a canned OpenAI-shaped HTTP server (the
test_planner_vlm pattern); the real DeepSeek API is never called and no key is
read. Properties pinned:

- routing: "prepare coffee" plans over the RoboCasa graph vocabulary; a pack-all
  instruction routes to pack_all_robocasa; an unmatched instruction never
  reaches the model (no_match);
- the model sees a COMPACT catalogue with taxonomy + binding availability, never
  the whole graph;
- the planner card's behaviour is reused unchanged: one re-ask on bad JSON,
  then a plan the validator refuses -> status rejected;
- the validator is the runtime's gate: invented skill / bad arg / bad after are
  rejected, never repaired;
- the execution boundary: an unbound leaf makes the plan planning_only with the
  gaps listed; every leaf bound (pack_all_robocasa) makes it executable; a
  planning_only record can never be turned into a brief.
"""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer

import pytest
from test_planner_vlm import _Server

import plugins.planner_vlm as pv
from harness.manifest import discover
from harness.unified_skill_graph import DEFAULT_GRAPH_PATH, load_graph
from plugins.task import skill_planning as sp

pytestmark = pytest.mark.skipif(
    not DEFAULT_GRAPH_PATH.exists(),
    reason="generated unified_skill_graph.json not present in this checkout")

COFFEE_TEXT = "Prepare a cup of coffee."
PACK_TEXT = "Pack every food item into its assigned tupperware."

#: A legal graph-channel chain: two composites, verify-covered, symbolic oracle.
COFFEE_PLAN = {
    "goal": "Prepare a cup of coffee",
    "nodes": [
        {"id": "setup-mug", "skill": "CoffeeSetupMug", "kind": "segment", "args": {},
         "after": []},
        {"id": "start-machine", "skill": "StartCoffeeMachine", "kind": "segment", "args": {},
         "after": ["setup-mug"]},
    ],
    "verify": [{"after": "setup-mug", "predicate": "annotation_complete"},
               {"after": "start-machine", "predicate": "annotation_complete"}],
}


def pack_plan() -> dict:
    """The canonical pack_all_robocasa graph: the card's own per-object skill
    order, verify-covered, targets per planning_context. SKILLS is imported
    rather than spelled out, so a rename in the skill library moves this test
    with it instead of leaving it asserting a vocabulary nobody serves."""
    from plugins.mission_pack_all import CATALOGUE, ITEMS, SKILLS, TARGET_BY_OBJECT
    nodes, verify, prev = [], [], None
    for obj in ITEMS:
        for skill in SKILLS:
            nid = f"{skill}-{obj}"
            args = {"object": obj}
            if "target" in CATALOGUE[skill]:
                args["target"] = TARGET_BY_OBJECT[obj]
            nodes.append({"id": nid, "skill": skill, "kind": "segment", "args": args,
                          "after": [prev] if prev else []})
            verify.append({"after": nid, "predicate": "segment_success"})
            prev = nid
    return {"goal": "pack every item", "nodes": nodes, "verify": verify}


@pytest.fixture()
def endpoint():
    """A fresh canned server per test: its port enters planner_vlm's frozen-graph
    key, so tests never replay one another's plans."""
    _Server.replies, _Server.requests = [], []
    server = HTTPServer(("127.0.0.1", 0), _Server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield {"endpoint_params": {"base_url": f"http://127.0.0.1:{server.server_address[1]}/v1"}}
    server.shutdown()
    thread.join()


def _payload(request: dict) -> dict:
    content = request["messages"][1]["content"]
    return json.loads(content[content.find("{"):content.rfind("}") + 1])


def _plan(text, params, **kw):
    return sp.plan_skill_task(text, planner_params=params, **kw)


# --- routing + disclosure -----------------------------------------------------


def test_prepare_coffee_routes_to_the_graph_with_a_compact_catalogue(endpoint):
    _Server.replies = [json.dumps(COFFEE_PLAN)]
    out = _plan(COFFEE_TEXT, endpoint, seed=1)
    assert out["channel"]["id"] == "robocasa_skill_graph" and out["channel"]["kind"] == "graph"
    assert out["channel"]["matched"] == ["coffee", "cup", "prepare"]
    names = [s["name"] for s in out["selected_catalogue"]["skills"]]
    assert names[:2] == ["CoffeeSetupMug", "StartCoffeeMachine"]
    assert {"Pick", "Place", "PressButton"} <= set(names)
    assert out["selected_catalogue"]["size"] < out["selected_catalogue"]["graph_total_skills"]
    # what the model actually received: the compact catalogue, taxonomy, bindings
    sent = _payload(_Server.requests[0])
    cards = {c["name"]: c for c in sent["skills"]}
    assert set(cards) == set(names)
    assert cards["CoffeeSetupMug"]["args"] == {} and cards["Pick"]["args"] == {"object": "str"}
    assert sent["oracles"] == ["annotation_complete"]
    assert {s["name"] for s in sent["taxonomy"]["skills"]} == set(names)
    assert sent["binding_availability"]["CoffeeSetupMug"] == {"bound": False, "tasks": []}
    assert "taxonomy" in _Server.requests[0]["messages"][0]["content"]  # the added rules
    assert out["graph_provenance"]["schema_version"] == "1.0"
    assert out["retrieval"]["graph_total_skills"] == 56


def test_pack_instruction_routes_to_the_executable_task_channel(endpoint):
    _Server.replies = [json.dumps(pack_plan())]
    out = _plan(PACK_TEXT, endpoint, seed=2)
    assert out["channel"] == {"id": "pack_all_robocasa", "kind": "task",
                              "task": "pack_all_robocasa",
                              "planner": "plugins.planner_vlm:provider",
                              "score": out["channel"]["score"],
                              "matched": out["channel"]["matched"]}
    assert out["channel"]["score"] >= 5
    sent = _payload(_Server.requests[0])
    from plugins.mission_pack_all import SKILLS
    assert {c["name"] for c in sent["skills"]} == set(SKILLS)
    assert sent["planning_context"]["required_per_object_order"] == list(SKILLS)
    assert all(v["bound"] for v in sent["binding_availability"].values())


def test_unmatched_instruction_never_reaches_the_model(endpoint):
    out = _plan("fly me to the moon", endpoint, seed=3)
    assert out["status"] == sp.STATUS_NO_MATCH and out["executable"] is False
    assert out["composite_plan"] is None and _Server.requests == []
    assert "nothing was sent to the model" in out["validation"]["message"]


def test_channel_can_be_pinned_and_unknown_pins_are_refused(endpoint):
    canonical_only = {
        "goal": "pick then place", "nodes": [
            {"id": "p", "skill": "Pick", "kind": "segment", "args": {"object": "rock"},
             "after": []},
            {"id": "q", "skill": "Place", "kind": "segment",
             "args": {"object": "rock", "target": "moon"}, "after": ["p"]}],
        "verify": [{"after": "p", "predicate": "annotation_complete"},
                   {"after": "q", "predicate": "annotation_complete"}]}
    _Server.replies = [json.dumps(canonical_only)]
    out = _plan("fly me to the moon", endpoint, seed=4, channel="robocasa_skill_graph")
    assert out["channel"]["id"] == "robocasa_skill_graph" and out["retrieval"]["pinned"]
    # no retrieval hit: the compact fallback is the canonical interfaces, not the graph
    names = {s["name"] for s in out["selected_catalogue"]["skills"]}
    assert names == {n.name for n in load_graph().skills() if n.kind == "canonical_skill"}
    assert "CoffeeSetupMug" not in names and len(names) < 56
    assert out["status"] == sp.STATUS_PLANNING_ONLY
    assert [r["label"] for r in out["expanded_plan"]["chain"]] == ["Pick.pick", "Place.place"]
    with pytest.raises(sp.PlanningError, match="unknown channel 'mars'"):
        _plan(COFFEE_TEXT, endpoint, channel="mars")
    with pytest.raises(sp.PlanningError, match="non-empty"):
        _plan("   ", endpoint)


# --- the planner card, reused: strict JSON with one re-ask ---------------------


def test_malformed_json_is_reasked_once_then_planned(endpoint):
    _Server.replies = ["not json", json.dumps(COFFEE_PLAN)]
    out = _plan(COFFEE_TEXT, endpoint, seed=5)
    assert len(_Server.requests) == 2
    assert "failed strict JSON parsing" in _Server.requests[1]["messages"][-1]["content"]
    assert out["status"] == sp.STATUS_PLANNING_ONLY and out["validation"]["ok"]


def test_two_malformed_replies_yield_a_rejected_plan_not_an_invented_one(endpoint):
    _Server.replies = ["garbage", "still garbage"]
    out = _plan(COFFEE_TEXT, endpoint, seed=6)
    assert len(_Server.requests) == 2                      # exactly one retry
    assert out["status"] == sp.STATUS_REJECTED and out["executable"] is False
    assert out["validation"]["ok"] is False and out["validation"]["unparseable"] is True
    assert "unparseable" in out["goal"]
    assert out["composite_plan"]["plan_id"] is None and out["expanded_plan"] is None


# --- the validator is the runtime's gate, never a repair shop ------------------


def _rejected(endpoint, plan, seed):
    _Server.replies = [json.dumps(plan)]
    out = _plan(COFFEE_TEXT, endpoint, seed=seed)
    assert out["status"] == sp.STATUS_REJECTED and out["executable"] is False
    assert out["expanded_plan"] is None
    return out["validation"]["message"]


def test_invented_skill_is_rejected(endpoint):
    plan = json.loads(json.dumps(COFFEE_PLAN))
    plan["nodes"][1]["skill"] = "MakeEspresso"
    assert "unknown skill 'MakeEspresso'" in _rejected(endpoint, plan, 7)


def test_invented_or_mistyped_args_are_rejected(endpoint):
    plan = json.loads(json.dumps(COFFEE_PLAN))
    plan["nodes"][0]["args"] = {"object": "mug"}          # observed skills take no args
    assert "unknown ['object']" in _rejected(endpoint, plan, 8)
    plan = json.loads(json.dumps(COFFEE_PLAN))
    plan["nodes"].append({"id": "pick-mug", "skill": "Pick", "kind": "segment",
                          "args": {"object": 7}, "after": ["start-machine"]})
    plan["verify"].append({"after": "pick-mug", "predicate": "annotation_complete"})
    assert "object" in _rejected(endpoint, plan, 9)


def test_forward_after_edges_duplicate_ids_and_invented_predicates_are_rejected(endpoint):
    plan = json.loads(json.dumps(COFFEE_PLAN))
    plan["nodes"][0]["after"] = ["start-machine"]          # not an EARLIER id
    assert "must list ids of earlier nodes" in _rejected(endpoint, plan, 10)
    plan = json.loads(json.dumps(COFFEE_PLAN))
    plan["nodes"][1]["id"] = "setup-mug"
    assert "duplicate node id" in _rejected(endpoint, plan, 11)
    plan = json.loads(json.dumps(COFFEE_PLAN))
    plan["verify"][0]["predicate"] = "coffee_is_hot"
    assert "unknown predicate 'coffee_is_hot'" in _rejected(endpoint, plan, 12)
    plan = json.loads(json.dumps(COFFEE_PLAN))
    plan["nodes"].append({"id": "stage-leak", "skill": "CoffeeSetupMug.pick", "kind": "segment",
                          "args": {}, "after": []})
    assert "unknown skill 'CoffeeSetupMug.pick'" in _rejected(endpoint, plan, 13)


# --- the execution boundary ----------------------------------------------------


def test_coffee_chain_expands_server_side_and_is_planning_only(endpoint):
    _Server.replies = [json.dumps(COFFEE_PLAN)]
    out = _plan(COFFEE_TEXT, endpoint, seed=14)
    assert out["status"] == sp.STATUS_PLANNING_ONLY and out["executable"] is False
    chain = out["expanded_plan"]["chain"]
    assert [r["label"] for r in chain] == [
        "CoffeeSetupMug.pick", "CoffeeSetupMug.place", "StartCoffeeMachine.execute"]
    assert [r["canonical"] for r in chain] == ["Pick", "Place", "PressButton"]
    assert [r["after"] for r in chain] == [[], ["CoffeeSetupMug.pick"], ["CoffeeSetupMug.place"]]
    assert out["expanded_plan"]["terminal"] == "done"
    nodes = out["expanded_plan"]["nodes"]
    assert [d["skill"] for d in nodes[0]["decomposition"]] == ["Pick", "Place"]
    assert [d["skill"] for d in nodes[1]["decomposition"]] == ["PressButton"]
    assert nodes[0]["taxonomy_path"] == ["RobotSkill", "CompositeSkill", "TransferObject",
                                         "CoffeeSetupMug"]
    assert [m["label"] for m in out["missing_bindings"]] == [r["label"] for r in chain]
    assert all("RoboCasa annotation" in m["reason"] for m in out["missing_bindings"])
    assert out["unbound_oracles"] == ["annotation_complete"]
    plan_out = out["composite_plan"]["plan"]
    assert {k: plan_out[k] for k in COFFEE_PLAN} == COFFEE_PLAN
    # upstream stamps WHICH endpoint saw WHICH prompt bytes onto every plan
    assert set(plan_out["planner"]) == {"provider", "endpoint", "prompt_sha"}
    assert len(out["composite_plan"]["plan_id"]) == 64
    # the graph's own `executable: true` flag is NOT a binding
    assert load_graph().graph_executable_flag("CoffeeSetupMug") is True


def test_expand_false_keeps_node_level_chain(endpoint):
    _Server.replies = [json.dumps(COFFEE_PLAN)]
    out = _plan(COFFEE_TEXT, endpoint, seed=15, expand=False)
    assert [r["label"] for r in out["expanded_plan"]["chain"]] == [
        "CoffeeSetupMug", "StartCoffeeMachine"]
    assert out["status"] == sp.STATUS_PLANNING_ONLY


def test_every_leaf_bound_makes_the_pack_plan_executable(endpoint):
    _Server.replies = [json.dumps(pack_plan())]
    out = _plan(PACK_TEXT, endpoint, seed=16)
    assert out["status"] == sp.STATUS_EXECUTABLE and out["executable"] is True
    assert out["missing_bindings"] == [] and out["unbound_oracles"] == []
    chain = out["expanded_plan"]["chain"]
    assert len(chain) == 16 and all(r["bound"] for r in chain)
    first = chain[0]
    assert first["label"] == "navigate.nav_hot1" and first["stage"] == "nav_hot1"
    assert first["binding"]["policy"] == "plugins.embodiment_robocasa.lunch_driver:provider"
    assert first["binding"]["task"] == "pack_all_robocasa"
    assert first["taxonomy_path"][-1] == "NavigateToObject"          # display alias only
    assert chain[1]["after"] == ["navigate.nav_hot1"]


def test_pack_plan_that_breaks_the_task_grounding_is_rejected(endpoint):
    plan = pack_plan()
    plan["nodes"][3]["args"]["target"] = "tupperware1"       # hot1 belongs in tupperware0
    _Server.replies = [json.dumps(plan)]
    out = _plan(PACK_TEXT, endpoint, seed=17)
    assert out["status"] == sp.STATUS_REJECTED
    assert "targets 'tupperware1' for object 'hot1'" in out["validation"]["message"]


def test_verify_plan_record_refuses_everything_but_an_executable_task_record(endpoint):
    _Server.replies = [json.dumps(COFFEE_PLAN), json.dumps(pack_plan())]
    coffee = _plan(COFFEE_TEXT, endpoint, seed=18)["composite_plan"]
    pack = _plan(PACK_TEXT, endpoint, seed=19)["composite_plan"]
    v = sp.verify_plan_record(coffee)
    assert v["ok"] is False and v["status"] == sp.STATUS_PLANNING_ONLY and "brief" not in v
    v = sp.verify_plan_record({**pack, "channel": "ghost_task"})
    assert v["ok"] is False and v["status"] == sp.STATUS_REJECTED
    tampered = json.loads(json.dumps(pack))
    tampered["plan"]["nodes"][0]["skill"] = "teleport"
    v = sp.verify_plan_record(tampered)
    assert v["ok"] is False and "unknown skill 'teleport'" in v["error"]
    assert sp.verify_plan_record("not a record")["ok"] is False
    v = sp.verify_plan_record(pack, seed=424243, max_actuations=24)
    assert v["ok"] is True and v["status"] == sp.STATUS_EXECUTABLE
    # the brief is selector+budgets only: exactly the keys the runtime admits
    assert v["brief"] == {"kind": "task", "task": "pack_all_robocasa",
                          "instruction": PACK_TEXT, "seed": 424243, "max_actuations": 24}
    assert v["plan_id"] == pack["plan_id"]


def test_binding_index_names_only_manifest_catalogue_skills():
    index = sp.binding_index(discover())
    assert "pack_all_robocasa" in index["grasp"] and "basket_smoke_vlm" in index["grasp"]
    assert "Pick" not in index and "CoffeeSetupMug" not in index
    channels = {c.id: c for c in sp.task_channels(discover())}
    assert "pack_all_robocasa" in channels and "kitchen_thaw" not in channels  # table planner


def test_same_instruction_replays_the_frozen_plan(endpoint):
    _Server.replies = [json.dumps(COFFEE_PLAN)]
    a = _plan(COFFEE_TEXT, endpoint, seed=20)
    calls = len(_Server.requests)
    b = _plan(COFFEE_TEXT, endpoint, seed=20)
    assert len(_Server.requests) == calls and a == b      # planner_vlm's freeze, reused
    assert isinstance(pv._FROZEN, dict)
