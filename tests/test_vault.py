"""The Skill Vault fold (board/vault.py) is a deterministic read over sealed data.

Four properties, all over the LIVE repo tree (the fold's only honest fixture is
the sealed evidence itself -- like the plugin_doctor --verify-claim tests, these
skip when a fresh clone lacks runs/):

- fold over real runs/ yields the known nodes + edges (place descends from stack,
  skill_place claims place, the privileged/observable REQUIRES split, SUPERSEDES);
- determinism: two folds are byte-identical (json.dumps sort_keys);
- face byte-equivalence: build_graph == storecli stdout == mcp tool, all 3 fns;
- vault_doctor red/green: a reserved key / unknown node / dangling see_also each
  fails loud, a valid additive annotation attaches and overwrites no derived field.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from board import mcp_server as ms
from board import storecli
from board import vault as bv

REPO = Path(bv.__file__).resolve().parent.parent
RUNS = REPO / "runs"

STACK = "57162e40d2bd4a0d59973d8c51d19f7267b682ba582c7b5c84568b334f02d41d"
ADC = "adc5578932681b6607737cdee40164c472e1bde277b0637a3b2c02623a3c4440"
EB = "eb46481a88b93cf9db9e774734fdde063725557d83f1abffe3033cd33a45a40f"

# the sealed evidence is not in git; a fresh clone legitimately skips (base-gate
# fresh-clone variance), never fails.
_sealed = pytest.mark.skipif(
    not (RUNS / "stack-g1" / "skills" / f"{STACK}.json").exists(),
    reason="sealed runs/ evidence not in this checkout")


@_sealed
def test_fold_over_real_runs():
    g = bv.build_graph(RUNS)
    by_id = {n["id"]: n for n in g["nodes"]}
    edges = {(e["rel"], e["src"], e["dst"]) for e in g["edges"]}

    # the three mounted skills + the 27 cards + the 10 capabilities are all present
    # (12th card: the M7 clear_workspace persistent-episode mission; 13th: the
    # inactive embodiment_robocasa second-simulator card, listed like every card;
    # 14th-17th: the M7 robocasa persistent-episode missions -- kitchen_thaw,
    # recycle_cans, pack_lunch, steam_prep; 18th: the inactive model_endpoint
    # chat-transport card, whose model.endpoint seam is the 10th capability;
    # 19th: the inactive policy_vla_remote websocket transport card; 20th: the
    # inactive embodiment_libero third-simulator card; 21st: the planner_vlm
    # VLM-planner card, task_bindings-only like skill_toy; 22nd-24th: the PR #2
    # static-skill-library missions -- basket_smoke, pack_all, stack_blocks;
    # 25th: the pure-data benchmark_robocasa suite card; 26th: the planner_library
    # PlanRecord-first planner card, consumed by ref, no mounts; 27th: the
    # executor_mcp_segment MCP segment-executor card, provides executor
    # mcp_segment, no mounts; 28th: skill_graph_robocasa, the PR #5 unified
    # skill-graph card).
    assert {STACK, ADC, EB} <= set(by_id)
    assert sum(n["kind"] == "package" for n in g["nodes"]) == 28
    assert sum(n["kind"] == "capability" for n in g["nodes"]) == 10

    stack = by_id[STACK]
    assert stack["kind"] == "skill" and stack["task"] == "stack"
    assert stack["status"] == "promoted" and stack["privilege"] == 0
    assert stack["evidence"]["heldout"]["governed_rate"] == 0.65  # verbatim
    assert stack["evidenced_by"] == "stack-g1"

    assert by_id[EB]["privilege"] == 1 and by_id[EB]["status"] == "promoted"
    # gen-2 place descends from gen-1 place (child_sha chain) AND from stack (prereg)
    assert ("DESCENDS_FROM", EB, ADC) in edges
    assert ("DESCENDS_FROM", ADC, STACK) in edges
    assert ("DESCENDS_FROM", EB, STACK) in edges

    # package <-> skill references (the operator's 互相包含引用)
    assert ("CLAIMS", "plugins/skill_place", ADC) in edges
    assert ("CLAIMS", "plugins/skill_place", EB) in edges
    assert ("CLAIMS", "plugins/task", STACK) in edges
    assert ("BINDS", "plugins/task", "stack") in edges
    assert ("PROVIDES", "plugins/embodiment_robosuite", "embodiment.env") in edges

    # the transfer story: privileged trigger REQUIRES ground_truth, observable percept.
    assert ("REQUIRES", EB, "embodiment.ground_truth") in edges
    assert ("REQUIRES", STACK, "percept.model") in edges
    # the packaging duplicate-seam: enabled reasoner over disabled model_qwen.
    assert ("SUPERSEDES", "plugins/reasoner", "plugins/model_qwen") in edges

    # every capability the skills/packages point at resolves to a real node.
    assert by_id["embodiment.ground_truth"]["privileged"] is True


@_sealed
def test_node_page_has_both_directions():
    g = bv.build_graph(RUNS)
    page = bv.node(g, STACK)
    out = {(e["rel"], e["dst"]) for e in page["out"]}
    back = {(e["rel"], e["src"]) for e in page["backlinks"]}
    assert ("REQUIRES", "percept.model") in out           # skill -> capability
    assert ("EVIDENCED_BY", "stack-g1") in out            # skill -> store
    assert ("CLAIMS", "plugins/task") in back             # package -> this skill
    assert ("DESCENDS_FROM", ADC) in back                 # descendant -> this skill


@_sealed
def test_unknown_node_is_an_error():
    g = bv.build_graph(RUNS)
    assert bv.node(g, "nope") == {"error": "unknown node"}
    assert bv.neighbors(g, "nope") == {"error": "unknown node"}


@_sealed
def test_determinism_byte_identical():
    a = json.dumps(bv.build_graph(RUNS), sort_keys=True)
    b = json.dumps(bv.build_graph(RUNS), sort_keys=True)
    assert a == b


@_sealed
def test_faces_byte_equivalent():
    """All three faces are the SAME function (round-95 discipline)."""
    ms.configure(RUNS)
    g = bv.build_graph(RUNS)

    def cli(*argv):
        r = subprocess.run([sys.executable, "-m", "board.storecli", *argv,
                            "--runs", str(RUNS)], capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stderr
        return r.stdout.rstrip("\n")

    assert cli("vault") == json.dumps(g) == json.dumps(ms.vault())
    assert cli("vault_node", EB) == json.dumps(bv.node(g, EB)) == json.dumps(ms.vault_node(EB))
    assert (cli("vault_neighbors", EB, "--relation", "DESCENDS_FROM")
            == json.dumps(bv.neighbors(g, EB, "DESCENDS_FROM"))
            == json.dumps(ms.vault_neighbors(EB, "DESCENDS_FROM")))


# --- vault_doctor: additive annotations can add context, never contradict -----


def _graph_with(tmp_path, ann):
    """A tiny graph over a fake node id + a sidecar dir carrying one annotation."""
    graph = {"nodes": [{"kind": "skill", "id": "n1"}, {"kind": "skill", "id": "n2"}],
             "edges": []}
    ann_dir = tmp_path / "ann"
    ann_dir.mkdir()
    (ann_dir / "n1.json").write_text(json.dumps(ann))
    return graph, ann_dir


def test_vault_doctor_green_valid_annotation_attaches(tmp_path):
    graph, ann_dir = _graph_with(tmp_path, {"note": "hand link", "see_also": ["n2"]})
    assert bv.vault_doctor(graph, ann_dir) == []
    # the loader attaches under node.annotations and touches no derived field.
    nodes = [{"kind": "skill", "id": "n1", "status": "promoted"}]
    bv._attach_annotations(nodes, ann_dir)
    assert nodes[0]["annotations"] == {"note": "hand link", "see_also": ["n2"]}
    assert nodes[0]["status"] == "promoted"  # derived field untouched


def test_vault_doctor_red_reserved_key(tmp_path):
    graph, ann_dir = _graph_with(tmp_path, {"status": "retired"})  # derived key
    errs = bv.vault_doctor(graph, ann_dir)
    assert errs and "additive set" in errs[0]
    # and a reserved key is NOT loaded over the node's derived field.
    nodes = [{"kind": "skill", "id": "n1", "status": "promoted", "annotations": None}]
    bv._attach_annotations(nodes, ann_dir)
    assert nodes[0]["annotations"] is None and nodes[0]["status"] == "promoted"


def test_vault_doctor_red_unknown_node(tmp_path):
    graph = {"nodes": [{"kind": "skill", "id": "n2"}], "edges": []}
    ann_dir = tmp_path / "ann"
    ann_dir.mkdir()
    (ann_dir / "ghost.json").write_text(json.dumps({"note": "x"}))
    errs = bv.vault_doctor(graph, ann_dir)
    assert errs and "unknown node" in errs[0]


def test_vault_doctor_red_dangling_see_also(tmp_path):
    graph, ann_dir = _graph_with(tmp_path, {"see_also": ["does_not_exist"]})
    errs = bv.vault_doctor(graph, ann_dir)
    assert errs and "see_also target" in errs[0]


# --- the one skill graph: library records + classes + benchmark cards -------


def _library_tree(tmp_path):
    """Two library records (grasp ensures what carry requires), one benchmark
    card, one package the carry binding refs -- all under tmp so the fold is
    exercised on a known tree, not the live library."""
    lib = tmp_path / "records"
    lib.mkdir()
    (lib / "grasp_x.json").write_text(json.dumps({
        "id": "grasp_x", "name": "grasp_x", "kind": "segment", "class": "grasp",
        "args": {"object": "str"}, "requires": ["present(object)"],
        "ensures": ["holding(object)"], "clobbers": [], "limits": {}, "failure_modes": [],
        "bindings": {"robocasa": {"backend": "scripted"}}, "evidence": {}}))
    (lib / "carry_x.json").write_text(json.dumps({
        "id": "carry_x", "name": "carry_x", "kind": "segment", "class": "carry",
        "args": {"object": "str"}, "requires": ["holding(object)"],
        "ensures": ["reachable(object)"], "clobbers": [], "limits": {}, "failure_modes": [],
        "bindings": {"robocasa": {"policies": {
            "scripted": {"transport": "inproc"},
            "pi05": {"transport": "ssp", "ref": "plugins.policy_fake:provider",
                     "checkpoint_sha": "ab" * 32}}}},
        "evidence": {"robocasa": {"n": 10, "k": 7, "by_executor": {"pi05": {"n": 4, "k": 3}}}}}))
    (lib / "carry_x_v2.json").write_text(json.dumps({
        "id": "carry_x_v2", "name": "carry_x_v2", "kind": "segment", "class": "carry",
        "args": {}, "requires": ["gripper_free()"], "ensures": ["at(bin)"], "clobbers": [],
        "limits": {}, "failure_modes": [], "bindings": {}, "evidence": {}}))
    (lib / "free_x.json").write_text(json.dumps({
        "id": "free_x", "name": "free_x", "kind": "segment", "class": "free",
        "args": {}, "requires": [], "ensures": ["gripper_free()"], "clobbers": [],
        "limits": {}, "failure_modes": [], "bindings": {}, "evidence": {}}))
    (lib / "plan_t1.json").write_text(json.dumps({
        "kind": "plan", "id": "p1", "task": "t1", "goal": [], "embodiment": "robocasa",
        "arm": "scripted", "evidence": {}, "rule": {},
        "graph": {"nodes": [{"id": "n1", "skill": "carry_x"}]}}))
    plugins = tmp_path / "plugins"
    (plugins / "benchmark_fake").mkdir(parents=True)
    (plugins / "benchmark_fake" / "manifest.toml").write_text(
        '[benchmarks.fake_v0]\nembodiment = "robocasa"\ntasks = ["t1"]\narms = ["scripted", "pi05"]\n')
    (plugins / "policy_fake").mkdir()
    (plugins / "policy_fake" / "manifest.toml").write_text('name = "policy_fake"\n')
    # a mission card binding t1 on robocasa: its skills line is the USES source and
    # the second EVIDENCED_ON route (grasp_x has no plan, only this card)
    (plugins / "mission_fake").mkdir()
    (plugins / "mission_fake" / "manifest.toml").write_text(
        '[task_bindings.t1]\nenv = "plugins.embodiment_robocasa:provider"\n'
        'skills = ["carry_x", "grasp_x"]\n')
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs, plugins, lib


def _fold(tmp_path):
    runs, plugins, lib = _library_tree(tmp_path)
    return bv.build_graph(runs, plugins, annotations=None, library=lib)


def test_library_record_folds_to_skill_class_nodes(tmp_path):
    g = _fold(tmp_path)
    by_id = {n["id"]: n for n in g["nodes"]}
    edges = {(e["rel"], e["src"], e["dst"]) for e in g["edges"]}
    carry = by_id["skill:carry_x"]
    assert carry["kind"] == "skill" and carry["status"] == "library" and carry["class"] == "carry"
    assert carry["requires"] == ["holding(object)"] and carry["ensures"] == ["reachable(object)"]
    assert carry["bindings"]["robocasa"]["pi05"] == {
        "transport": "ssp", "ref": "plugins.policy_fake:provider", "checkpoint_sha": "ab" * 32}
    assert carry["evidence"]["robocasa"] == {"n": 10, "k": 7, "by_executor": {"pi05": {"n": 4, "k": 3}}}
    assert by_id["class:carry"] == {"kind": "class", "id": "class:carry", "name": "carry",
                                    "skills": 2, "annotations": None}
    assert ("IN_CLASS", "skill:carry_x", "class:carry") in edges
    assert ("IN_CLASS", "skill:grasp_x", "class:grasp") in edges
    assert ("BOUND_TO", "skill:carry_x", "plugins/policy_fake") in edges
    # nothing in the fold is nameless: every edge endpoint is a node (BINDS points
    # at a task NAME, never a node -- same as the live graph)
    named = [e for e in g["edges"] if e["rel"] != "BINDS"]
    assert {e["src"] for e in named} | {e["dst"] for e in named} <= set(by_id)
    assert bv.vault_doctor(g, tmp_path / "no-annotations") == []


def test_requires_ensures_pair_yields_depends_on(tmp_path):
    g = _fold(tmp_path)
    dep = [e for e in g["edges"] if e["rel"] == "DEPENDS_ON"]
    assert [(e["src"], e["dst"]) for e in dep] == [("skill:carry_x", "skill:grasp_x")]
    assert dep[0]["rule"] == "requires∩ensures" and dep[0]["via"] == "skill-library/records/carry_x.json"
    # carry_x_v2 requires gripper_free() which free_x ensures: zero-arity, no DEPENDS_ON


def test_name_prefix_within_class_yields_instance_of(tmp_path):
    g = _fold(tmp_path)
    inst = [e for e in g["edges"] if e["rel"] == "INSTANCE_OF"]
    assert [(e["src"], e["dst"], e["rule"], e["via"]) for e in inst] == [
        ("skill:carry_x_v2", "skill:carry_x", "name prefix within class",
         "skill-library/records/carry_x_v2.json")]
    by_id = {n["id"]: n for n in g["nodes"]}
    assert by_id["skill:carry_x"]["instances"] == 1 and "instances" not in by_id["skill:carry_x_v2"]


def test_benchmark_card_yields_evidenced_on(tmp_path):
    g = _fold(tmp_path)
    by_id = {n["id"]: n for n in g["nodes"]}
    assert by_id["benchmark:fake_v0"] == {
        "kind": "benchmark", "id": "benchmark:fake_v0", "name": "fake_v0", "embodiment": "robocasa",
        "tasks": ["t1"], "arms": ["scripted", "pi05"], "card": "plugins/benchmark_fake",
        "missions": ["plugins/mission_fake"], "annotations": None}
    ev = {e["src"]: e for e in g["edges"] if e["rel"] == "EVIDENCED_ON"}
    # carry_x via the plan record; grasp_x via the covered mission's skills line
    assert set(ev) == {"skill:carry_x", "skill:grasp_x"}
    assert ev["skill:carry_x"]["dst"] == "benchmark:fake_v0"
    assert (ev["skill:carry_x"]["n"], ev["skill:carry_x"]["k"]) == (10, 7)  # verbatim
    assert "n" not in ev["skill:grasp_x"]  # no evidence -> no invented number


def test_mission_card_yields_covers_and_uses(tmp_path):
    g = _fold(tmp_path)
    by_id = {n["id"]: n for n in g["nodes"]}
    assert by_id["plugins/mission_fake"]["tasks"] == ["t1"]
    assert by_id["plugins/mission_fake"]["skills"] == 2
    assert "tasks" not in by_id["plugins/policy_fake"]  # not a mission card
    covers = [e for e in g["edges"] if e["rel"] == "COVERS"]
    assert [(e["src"], e["dst"], e["rule"], e["via"]) for e in covers] == [
        ("benchmark:fake_v0", "plugins/mission_fake", "benchmark tasks ∩ task_bindings",
         "plugins/benchmark_fake/manifest.toml + plugins/mission_fake/manifest.toml")]
    uses = [e for e in g["edges"] if e["rel"] == "USES"]
    assert [(e["src"], e["dst"], e["rule"], e["via"]) for e in uses] == [
        ("plugins/mission_fake", "skill:carry_x", "manifest task_bindings.skills",
         "plugins/mission_fake/manifest.toml"),
        ("plugins/mission_fake", "skill:grasp_x", "manifest task_bindings.skills",
         "plugins/mission_fake/manifest.toml")]
    # a benchmark on another embodiment covers nothing, whatever the task names
    (tmp_path / "plugins" / "benchmark_other").mkdir()
    (tmp_path / "plugins" / "benchmark_other" / "manifest.toml").write_text(
        '[benchmarks.other_v0]\nembodiment = "libero"\ntasks = ["t1"]\n')
    g2 = bv.build_graph(tmp_path / "runs", tmp_path / "plugins", annotations=None,
                        library=tmp_path / "records")
    assert [e["src"] for e in g2["edges"] if e["rel"] == "COVERS"] == ["benchmark:fake_v0"]


def test_library_fold_byte_deterministic(tmp_path):
    runs, plugins, lib = _library_tree(tmp_path)
    a = json.dumps(bv.build_graph(runs, plugins, annotations=None, library=lib), sort_keys=True)
    b = json.dumps(bv.build_graph(runs, plugins, annotations=None, library=lib), sort_keys=True)
    assert a == b
    g = json.loads(a)
    assert [(n["kind"], n["id"]) for n in g["nodes"]] == sorted((n["kind"], n["id"]) for n in g["nodes"])
    assert [(e["rel"], e["src"], e["dst"]) for e in g["edges"]] == sorted(
        (e["rel"], e["src"], e["dst"]) for e in g["edges"])


def test_live_library_is_in_the_graph():
    """The checked-in library and the benchmark card fold without runs/."""
    g = bv.build_graph(RUNS)
    ids = {n["id"] for n in g["nodes"]}
    assert "skill:carry" in ids and "benchmark:robocasa_v0" in ids
    assert any(n["kind"] == "class" for n in g["nodes"])
