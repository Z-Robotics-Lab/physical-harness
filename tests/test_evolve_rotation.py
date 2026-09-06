"""Observed action capabilities belong to the model to select, without target rotation."""

from __future__ import annotations

import json

import pytest

from harness.manifest import discover
from scripts import evolve_llm
from scripts.evolve_evidence import compact_brief, inspect_evidence
from scripts import harness_runtime as hr

REF = "plugins.embodiment_robocasa.recycle_driver:provider"
PLANNER = "plugins.mission_recycle_cans.planner"
NODES = ("nav-can1", "grasp-can1", "carry-can1", "drop-can1")


def _seed(dead: str) -> dict:
    return {"success": False, "first_death": dead, "failure_mode": "reach_stall", "keyframes": [],
            "trail": [{"id": n, "ok": n != dead, "steps": 100, "kind": "segment", "failure_mode": None} for n in NODES],
            "nodes": {n: {"skill": n.replace("-", "_"), "success": n != dead, "executor": "scripted"}
                      for n in NODES}}


def _proj(deaths: dict[str, str], rounds: list, epoch_start: int = 0) -> dict:
    """The real recycle_cans brief on a synthetic round: seed -> where it died."""
    binding = discover().task_bindings["recycle_cans"]
    before = {"count": 0, "seeds": {s: _seed(d) for s, d in deaths.items()}}
    doc = {"task": "recycle_cans", "seeds": [4243, 4244], "cursor": len(rounds),
           "rounds": rounds, "applied": {}, "epoch_start": epoch_start}
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


def test_capabilities_include_successful_actions_and_history_does_not_select_a_target():
    deaths = {"4243": "drop-can1", "4244": "nav-can1"}
    plain, before = _proj(deaths, [])
    history = [_round(i, "drop-can1", knob=f"k{i}") for i in range(1, 15)]
    later, _ = _proj(deaths, history)
    assert list(plain["drivers"]) == list(later["drivers"]) == sorted(NODES)
    assert {d["node"]: d["seeds"] for d in plain["death_nodes"]} == {
        "drop-can1": [4243], "nav-can1": [4244]}
    assert plain["death_nodes"] == later["death_nodes"]
    assert "carry-can1" in plain["drivers"] and before["seeds"]["4243"]["nodes"]["carry-can1"]["success"]
    for proj in (plain, later, compact_brief(later)):
        assert not {"target", "stuck", "stuck_rounds", "untried", "exhausted"} & set(proj)
    assert compact_brief(later)["driver_index"]


def test_source_inspection_follows_the_selected_action_without_expanding_patch_authority():
    history = [_round(i, "drop-can1", knob=f"k{i}") for i in range(1, 15)]
    proj, _ = _proj({"4243": "drop-can1", "4244": "nav-can1"}, history)
    methods = {"nav-can1": "plugins.embodiment_robocasa.stage_extras:NavToObjectDriver.__init__",
               "drop-can1": "plugins.embodiment_robocasa.recycle_driver:ClusterDropDriver._drop_point"}
    pages = []
    for node, symbol in methods.items():
        driver = proj["drivers"][node]
        assert compact_brief(proj)["driver_index"][node]["modules"] == driver["modules"]
        assert PLANNER not in driver["modules"]
        page = inspect_evidence(proj, {"view": "source", "node": node, "symbol": symbol})
        assert page["complete"] and page["code_read"]
        assert page["data"]["module"] in driver["modules"]
        assert node in page["data"]["owner_nodes"]
        with pytest.raises(ValueError, match="module must be one of"):
            inspect_evidence(proj, {"view": "source", "node": node, "module": PLANNER})
        pages.append(page)
    assert pages[0]["data"]["code"] != pages[1]["data"]["code"]
