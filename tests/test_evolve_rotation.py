"""Node rotation, the stuck escalation and the exhausted-knobs line -- the three
things that let one node (drop-can1, 66 rounds, best 0) eat a live campaign while
seed 4244 died at nav-can1 and never got a round. No simulator: the real
recycle_cans binding + records, a synthetic round, the model_endpoint fake."""

from __future__ import annotations

import json

import pytest

from harness.manifest import discover, mount_params
from scripts import evolve, evolve_llm
from scripts import harness_runtime as hr

REF = "plugins.embodiment_robocasa.recycle_driver:provider"
PLANNER = "plugins.mission_recycle_cans.planner"
NODES = ("nav-can1", "grasp-can1", "carry-can1", "drop-can1")


def _seed(dead: str) -> dict:
    return {"success": False, "first_death": dead, "failure_mode": "reach_stall", "keyframes": [],
            "trail": [{"id": n, "ok": n != dead, "steps": 100, "failure_mode": None} for n in NODES],
            "nodes": {n: {"skill": n.replace("-", "_"), "success": n != dead, "executor": "scripted"}
                      for n in NODES}}


def _proj(deaths: dict[str, str], rounds: list) -> dict:
    """The real recycle_cans brief on a synthetic round: seed -> where it died."""
    binding = discover().task_bindings["recycle_cans"]
    before = {"count": 0, "seeds": {s: _seed(d) for s, d in deaths.items()}}
    doc = {"task": "recycle_cans", "seeds": [4243, 4244], "cursor": len(rounds),
           "rounds": rounds, "applied": {}}
    return evolve_llm.rsi_projection(doc, before, hr._binding_records(binding), "robocasa",
                                     "scripted", binding, []), before


def _round(no: int, node: str, knob: str = "hover_dz", up: bool = True) -> dict:
    return {"round": no, "published": False, "outcome": "same", "before": 0, "after": 0,
            "tried": {"kind": "tunables", "node": node,
                      "detail": {"ref": REF, "path": ["tunables", knob],
                                 "from": 1.0, "to": 2.0 if up else 0.5}}}


def _fake(tmp_path, canned, name="canned.json"):
    (tmp_path / name).write_text(json.dumps(canned))   # the fake's reply cursor is keyed by path
    return evolve_llm.load_provider(evolve_llm.FAKE_REF, {"path": str(tmp_path / name)})


def test_consecutive_rounds_rotate_over_the_distinct_first_death_nodes(tmp_path):
    """4243 dies at drop-can1, 4244 at nav-can1: round 1 takes one, round 2 the other
    (least recently targeted first), and an answer that names no node lands on it."""
    deaths = {"4243": "drop-can1", "4244": "nav-can1"}
    targets, rounds = [], []
    for r, knob in ((1, "standoff"), (2, "hover_dz"), (3, "reach_tol")):
        proj, before = _proj(deaths, list(rounds))
        assert [d["node"] for d in proj["death_nodes"]][0] == proj["target"]["node"]
        assert {d["node"]: d["seeds"] for d in proj["death_nodes"]} == {"drop-can1": [4243], "nav-can1": [4244]}
        ep = _fake(tmp_path, [{"kind": "tunables", "summary": "试试。", "rationale": "-",
                               "payload": {"ref": REF, "path": ["tunables", knob], "to": 0.9}}],
                   name=f"round{r}.json")
        tried, _ = evolve_llm.llm_propose(ep, proj, before, r, tmp_path / f"llm{r}")
        assert tried["kind"] == "tunables"
        assert tried["node"] == proj["target"]["node"]   # no node in the payload: the round's target
        targets.append(tried["node"])
        rounds.append({**_round(r, tried["node"]), "tried": tried})
    assert targets == ["drop-can1", "nav-can1", "drop-can1"]
    assert [d["rounds_targeted"] for d in _proj(deaths, rounds)[0]["death_nodes"]] == [1, 2]


def test_a_node_stuck_for_stuck_rounds_widens_the_patchable_modules(monkeypatch):
    """Both seeds die at drop-can1: after STUCK_ROUNDS rounds on it with no improvement
    the brief says so and names the whole stage pipeline as patchable."""
    deaths = {"4243": "drop-can1", "4244": "drop-can1"}
    hist = [_round(i, "drop-can1", knob=f"k{i}") for i in range(1, evolve.STUCK_ROUNDS)]
    proj, _ = _proj(deaths, hist)
    assert "stuck" not in proj and proj["stuck_rounds"] == evolve.STUCK_ROUNDS
    hist.append(_round(evolve.STUCK_ROUNDS, "drop-can1", knob="last"))
    proj, _ = _proj(deaths, hist)
    st = proj["stuck"]
    assert (st["node"], st["rounds"]) == ("drop-can1", evolve.STUCK_ROUNDS)
    assert "parameter tweaks on it are exhausted" in st["note"] and "another node" in st["note"]
    assert "tunables last up" in st["tried"] and len(st["tried"]) == evolve.STUCK_ROUNDS
    for m in (REF.partition(":")[0], "plugins.embodiment_robocasa.drivers", PLANNER):
        assert m in st["modules"] and m in proj["first_death"]["modules"]   # a patch may name them now
    assert evolve_llm.brief(proj)["stuck"] == st
    # an improvement on the node breaks the streak
    assert evolve.stuck_on("drop-can1", [*hist[:-1], {**hist[-1], "published": True}]) is None
    monkeypatch.setattr(evolve, "STUCK_ROUNDS", 2)
    assert evolve.stuck_on("drop-can1", hist[:2]) == {"node": "drop-can1", "rounds": 2}


def test_the_other_death_nodes_material_follows_it_and_is_in_the_brief():
    """``target.judged`` invites the model to answer on the OTHER death node, and 86 of the
    live campaign's 568 shard rounds took the invitation (replayed off campaign.json's
    rotation). Only ``fd`` used to follow: the state_init the prompt's patch checklist points
    at, the modules write_patch accepts, the source an exact ``old`` is copied out of and the
    "still untried on X" refusal all described the rotation head -- a DIFFERENT class in a
    different module (drop-can1 is ClusterDropDriver/PointPlaceDriver, nav-can1 is
    NavigateDriver/NavToObjectDriver), which is how 4 live rounds died on SelfCheckError."""
    proj, _ = _proj({"4243": "drop-can1", "4244": "nav-can1"}, [])
    head = proj["target"]["node"]
    other = next(n for n in proj["drivers"] if n != head)
    a, b = proj["drivers"][head], proj["drivers"][other]
    assert a["state_init"] != b["state_init"] and a["modules"] != b["modules"]
    assert proj["first_death"] is a and REF.partition(":")[0] in b["modules"] + a["modules"]
    # the brief carries them per node (``drivers`` is material, not brief): the model picks a
    # node off death_nodes, so that row has to say what answering there means
    rows = {r["node"]: r for r in evolve_llm.brief(proj)["death_nodes"]}
    assert rows[other]["state_init"] == b["state_init"] and rows[other]["modules"] == b["modules"]
    assert rows[other]["stage_modules"] == b["stage_modules"]
    # ...and the source material covers BOTH nodes' modules, so `old` is a copy either way
    assert set(proj["module_sources"]) == set(a["modules"]) | set(b["modules"])
    # (whole-text vs class/function extract is the shared MODULE_CHARS budget's call; what
    # matters is that the other node's modules are THERE, numbered, and reported per node)
    assert set(b["modules_full"]) <= set(b["modules"])
    assert all(proj["module_sources"][m].startswith("# file: ") for m in b["modules"])
    # what is left to try is that node's list too (both nodes bind ONE provider here, so the
    # knobs are shared and the patch targets are not)
    left = evolve_llm._untried(proj, evolve_llm._tried_pairs(proj, b), b)
    assert [u for u in left if u.startswith("patch ")] == [f"patch {m}" for m in b["modules"]]
    assert evolve_llm.brief(proj)["untried"] != left        # the brief's is the head's
    # a stuck round widens EVERY node's table, not the head's alone: write_patch checks the
    # modules of payload.node, so widening one node takes the promise back on the other
    hist = [_round(i, head, knob=f"k{i}") for i in range(evolve.STUCK_ROUNDS)]
    hist.append(_round(len(hist), other))   # ...so the rotation hands the stuck node back
    wide, _ = _proj({"4243": "drop-can1", "4244": "nav-can1"}, hist)
    assert wide["stuck"]["node"] == head == wide["target"]["node"]
    assert PLANNER in wide["drivers"][head]["modules"] and PLANNER in wide["drivers"][other]["modules"]


def test_the_brief_says_tunables_are_exhausted_once_every_pair_is_tried():
    knobs = sorted(mount_params(REF)["tunables"])
    hist = [_round(i, "drop-can1", knob=k, up=up)
            for i, (k, up) in enumerate((k, up) for k in knobs for up in (True, False))]
    proj, _ = _proj({"4243": "drop-can1"}, hist)
    b = evolve_llm.brief(proj)
    assert b["exhausted"].startswith("tunables exhausted for drop-can1")
    assert not [u for u in b["untried"] if u.startswith("tunables ")]
    assert "exhausted" not in evolve_llm.brief(_proj({"4243": "drop-can1"}, hist[:-1])[0])
