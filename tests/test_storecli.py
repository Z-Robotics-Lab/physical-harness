"""The CLI face (board/storecli.py) is a byte-thin passthrough.

Round-95 discipline: prove all three call-faces are the SAME function. Each fn's
stdout must equal ``json.dumps(board.store.<fn>(...))`` exactly (so the panel
renders the byte-identical dict the LLM gets from board/mcp_server.py, which
test_mcp_server.py separately pins to board.store), and the shared
board.store.safe_child guard must reject a ``../`` name.

Fixtures reuse the sibling test modules' builders (real CampaignStore + a real
SessionLog chain) so the passthrough is exercised against production on-disk
shapes, not a hand-mocked one.
"""

from __future__ import annotations

import json
from pathlib import Path

from test_read_session import _session
from test_store import _campaign, _mkstore, _paired

from board import mcp_server as ms
from board import store as bs
from board import storecli


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    runs = tmp_path / "runs"
    runs.mkdir()
    _campaign(runs / "stack-g1")
    rescore = {"block": 42200, "n": 200, "judgement": True,
               "paired": _paired(0.60, 0.70, 19, 0, 1e-8, n=200),
               "vs_blind": _paired(0.70, 0.78, 40, 5, 1e-6), "stage_attribution": None}
    _mkstore(runs / "stack-g1-rescore-42200", [("heldout_rescore", rescore)])
    _session(runs, "session-main")
    status = tmp_path / "STATUS.md"
    status.write_text(
        "**PHASE 3 区块预算:** stack-g1 dev 41000-41580;\n"
        "**已烧:** held-out 42000-42199。\n")
    progress = tmp_path / "progress.md"
    progress.write_text("## Round 95 - 2026-08-23 - the cockpit\nbody one\n")
    return runs, status, progress


def _run(capsys, *argv) -> tuple[int, str]:
    code = storecli.main(list(argv))
    return code, capsys.readouterr().out.rstrip("\n")


def test_every_fn_is_byte_identical_to_board_store(tmp_path, capsys):
    runs, status, progress = _fixture(tmp_path)
    base = ["--runs", str(runs), "--status", str(status), "--progress", str(progress)]
    cases = [
        (["list_stores"], bs.list_stores(runs)),
        (["store", "stack-g1"], bs.store_detail(runs / "stack-g1")),
        (["heldout", "stack-g1"], bs.heldout_blocks(runs, "stack-g1")),
        (["sessions"], bs.discover_sessions(runs)),
        (["session", "session-main"], bs.read_session(runs / "session-main")),
        # runtime_status: fixture session has no runtime_status.json, so this
        # case is both the face-equivalence proof AND null-when-absent (both
        # sides serialize to "null").
        (["runtime_status", "session-main"], bs.read_runtime_status(runs / "session-main")),
        (["ledger"], bs.burned_blocks(runs)),
        (["plan_index", "session-main"], bs.plan_index(runs / "session-main")),
        (["skill_evidence", "session-main"], bs.skill_evidence(runs / "session-main")),
        (["skills", "session-main"], bs.skills(runs / "session-main")),
        (["rounds"], bs.parse_rounds(progress.read_text())),
    ]
    for argv, expected in cases:
        code, out = _run(capsys, *argv, *base)
        assert code == 0, argv
        assert out == json.dumps(expected), argv
    # non-trivial fixtures, so identity is not identity-of-empty
    assert bs.list_stores(runs) and bs.heldout_blocks(runs, "stack-g1")["blocks"]
    assert bs.burned_blocks(runs) and bs.parse_rounds(progress.read_text())


def test_submit_brief_two_faces_share_one_drop(tmp_path, capsys):
    """The ONE write fn: the CLI face is a passthrough into board.store.
    submit_brief (the same function the MCP tool delegates to), so both faces
    drop the raw brief bytes VERBATIM into the same inbox and answer the same
    {"submitted", "inbox"} shape -- zero validation on this side (the resident
    runtime's _BRIEF_KEYS on claim stays the sole authority)."""
    runs, status, progress = _fixture(tmp_path)
    base = ["--runs", str(runs), "--status", str(status), "--progress", str(progress)]
    raw = '{"kind":"rsi","task":"kitchen_thaw"}'
    code, out = _run(capsys, "submit_brief", "--brief", raw, "--session", "session-main", *base)
    cli = json.loads(out)
    direct = bs.submit_brief(runs, raw, "session-main")
    inbox = runs / "session-main" / "inbox"
    assert code == 0
    assert cli["inbox"] == direct["inbox"] == str(inbox)
    for res in (cli, direct):  # both faces landed the SAME bytes, unparsed
        assert (inbox / res["submitted"]).read_text() == raw


def test_submit_brief_unknown_session_drops_nothing(tmp_path, capsys):
    runs, status, progress = _fixture(tmp_path)
    code, out = _run(capsys, "submit_brief", "--brief", "{}", "--session", "../oops",
                     "--runs", str(runs), "--status", str(status), "--progress", str(progress))
    assert code == 0 and json.loads(out) == {"error": "unknown session '../oops'"}
    assert not list(runs.rglob("brief-*.json"))


_CAMPAIGN = {
    "task": "kitchen_thaw", "session": "session-main", "seeds": [1, 2], "arm": "auto",
    "rounds": [
        {"round": 1, "tried": {"kind": "executor", "node": "grasp", "detail": "scripted->geometric"},
         "before": 0, "after": 1, "best": 1, "suite_sha": "a" * 64, "published": True,
         "media": ["media/kitchen_thaw/1/grasp.gif"], "ts": 1.0,
         "per_seed": [{"seed": 1, "success": True, "first_death": None, "failure_mode": None,
                       "tunables_sha": None, "elapsed_s": 0.4, "nodes": [
                           {"id": "reach", "ok": True, "steps": 9, "failure_mode": None, "after": [],
                            "kind": "segment", "task": "reach"},
                           {"id": "grasp", "ok": True, "steps": 12, "failure_mode": None, "after": ["reach"],
                            "kind": "segment", "task": "grasp"}]},
                      {"seed": 2, "success": False, "first_death": "grasp", "failure_mode": "slip",
                       "tunables_sha": None, "elapsed_s": 0.3, "nodes": [
                           {"id": "reach", "ok": True, "steps": 9, "failure_mode": None, "after": [],
                            "kind": "segment", "task": "reach"},
                           {"id": "grasp", "ok": False, "steps": 12, "failure_mode": "slip", "after": ["reach"],
                            "kind": "segment", "task": "grasp"}]}],
         "needs": []},
        {"round": 2, "tried": {"kind": "tunables", "node": "grasp", "detail": "hover_z*1.2"},
         "before": 1, "after": 1, "best": 1, "suite_sha": "b" * 64, "published": False,
         "media": [], "ts": 2.0},
    ],
    "best": 1, "cursor": 2, "status": "running",
    "live": {"phase": "baseline", "round": 3, "seeds_total": 2, "seed_index": 1, "seed": 2,
             "node": "grasp", "started_at": 0.5, "round_started_at": 2.5, "phase_started_at": 2.5,
             "last_round_s": 1.0, "seed_started_at": 3.0, "per_seed_partial": [
                 {"seed": 1, "success": True, "first_death": None, "failure_mode": None,
                  "elapsed_s": 0.4, "nodes": [{"id": "reach", "ok": True, "steps": 9, "failure_mode": None,
                                               "after": [], "kind": "segment", "task": "reach"},
                                              {"id": "grasp", "ok": True, "steps": 12, "failure_mode": None,
                                               "after": ["reach"], "kind": "segment", "task": "grasp"}]}],
             "nodes": [{"id": "reach", "skill": "reach", "ok": True, "steps": None, "failure_mode": None,
                        "after": [], "kind": "segment", "task": "reach"},
                       {"id": "grasp", "skill": "grasp", "ok": None, "steps": None, "failure_mode": None,
                        "after": ["reach"], "kind": "segment", "task": "grasp"}],
             "tried": None, "message": "第 3 轮 基线评测：种子 2 运行中 (grasp) 节点 1/2，2/2",
             "messages": [{"ts": 3.0, "text": "第 3 轮 基线评测：种子 2 运行中 (grasp) 节点 1/2，2/2"}]},
}


def test_rsi_faces_are_byte_identical(tmp_path, capsys):
    """rsi_run / rsi_series / rsi_frames: the same object on the CLI, MCP and
    library faces (fixture campaign.json in the spec's shape); absent campaign
    -> null / [] / []; the task rides the shared safe_child guard."""
    runs, status, progress = _fixture(tmp_path)
    camp = runs / "session-main" / "campaigns" / "evolve-kitchen_thaw"
    camp.mkdir(parents=True)
    (camp / "campaign.json").write_text(json.dumps(_CAMPAIGN))
    sd = runs / "session-main"
    base = ["--runs", str(runs), "--session", "session-main"]
    ms.configure(runs)
    cases = [
        (["rsi_run", "kitchen_thaw"], bs.rsi_run(sd, "kitchen_thaw"),
         ms.rsi_run("kitchen_thaw")),
        (["rsi_series", "kitchen_thaw"], bs.rsi_series(sd, "kitchen_thaw"),
         ms.rsi_series("kitchen_thaw")),
        (["rsi_frames", "kitchen_thaw", "--round", "1"], bs.rsi_frames(sd, "kitchen_thaw", 1),
         ms.rsi_frames("kitchen_thaw", 1)),
        (["rsi_frames", "kitchen_thaw", "--round", "9"], bs.rsi_frames(sd, "kitchen_thaw", 9),
         ms.rsi_frames("kitchen_thaw", 9)),
        (["rsi_run", "nope"], bs.rsi_run(sd, "nope"), ms.rsi_run("nope")),
        (["rsi_series", "nope"], bs.rsi_series(sd, "nope"), ms.rsi_series("nope")),
    ]
    for argv, lib, mcp in cases:
        code, out = _run(capsys, *argv, *base)
        assert code == 0, argv
        assert out == json.dumps(lib) == json.dumps(mcp), argv
    # non-trivial: the fixture rounds actually flow through each face
    run = bs.rsi_run(sd, "kitchen_thaw")
    assert run["status"] == "running" and run["cursor"] == 2 and run["latest"]["round"] == 2
    assert run["live"] == _CAMPAIGN["live"] and run["live"]["message"]
    assert run["open_brief"] is None
    (camp / "campaign.json").write_text(json.dumps({k: v for k, v in _CAMPAIGN.items() if k != "live"}))
    assert bs.rsi_run(sd, "kitchen_thaw")["live"] is None   # pre-live campaign reads as null
    (camp / "campaign.json").write_text(json.dumps(_CAMPAIGN))
    r1 = _CAMPAIGN["rounds"][0]
    # node_rate: seed 1 2/2, seed 2 1/2 -> 0.75 (per_seed = before; no after_seeds -> null);
    # by_task: reach passes both seeds, grasp only seed 1; round 2 carries no trail -> nulls,
    # best carries the running max
    assert bs.rsi_series(sd, "kitchen_thaw") == [   # pre-per_seed rounds read as null
        {"round": 1, "before": 0, "after": 1, "best": 1, "per_seed": r1["per_seed"], "needs": [],
         "proposer": None, "llm": None,   # pre-LLM rounds read as null
         "node_rate": {"before": 0.75, "after": None, "best": 0.75},
         "by_task": {"grasp": {"before": 0.5, "after": None}, "reach": {"before": 1.0, "after": None}}},
        {"round": 2, "before": 1, "after": 1, "best": 1, "per_seed": None, "needs": None,
         "proposer": None, "llm": None, "node_rate": {"before": None, "after": None, "best": 0.75}, "by_task": {}}]
    assert bs.rsi_series(sd, "kitchen_thaw") == ms.rsi_series("kitchen_thaw")
    assert bs.rsi_frames(sd, "kitchen_thaw", 1) == ["media/kitchen_thaw/1/grasp.gif"]
    assert bs.rsi_frames(sd, "kitchen_thaw", 9) == [] and bs.rsi_run(sd, "nope") is None
    # traversal: a ../ task never leaves the session; a ../ session is refused
    assert bs.rsi_run(sd, "../../session-main") is None
    assert ms.rsi_run("kitchen_thaw", "../session-main") == {"error": "unknown session"}
    code, out = _run(capsys, "rsi_run", "kitchen_thaw", "--runs", str(runs), "--session", "../x")
    assert code == 3 and json.loads(out) == {"error": "unknown session"}
    code, out = _run(capsys, "rsi_run", "--runs", str(runs))
    assert code == 3 and "task" in json.loads(out)["error"]


def test_traversal_name_rejected_by_shared_guard(tmp_path, capsys):
    runs, status, progress = _fixture(tmp_path)
    base = ["--runs", str(runs), "--status", str(status), "--progress", str(progress)]
    for name in ("../etc", "..", "../STATUS.md"):
        code, out = _run(capsys, "store", name, *base)
        assert code == 3 and json.loads(out) == {"error": "unknown store"}
    code, out = _run(capsys, "session", "../session-main", *base)
    assert code == 3 and json.loads(out) == {"error": "unknown session"}
    code, out = _run(capsys, "runtime_status", "../session-main", *base)
    assert code == 3 and json.loads(out) == {"error": "unknown session"}


def test_unknown_fn_is_a_nonzero_error(tmp_path, capsys):
    runs, status, progress = _fixture(tmp_path)
    code, out = _run(capsys, "nope", "--runs", str(runs))
    assert code == 2 and json.loads(out)["error"].startswith("unknown fn")


def test_skills_face_is_byte_identical(tmp_path, capsys):
    """skills: the records overview on the CLI, MCP and library faces; a copy
    the session published (evolution write path) overlays the library record
    of the same name, and the row carries by_executor counts per embodiment."""
    runs, status, progress = _fixture(tmp_path)
    sd = runs / "session-main"
    lib = bs.skills(sd)
    assert lib and all(r["source"] == "library" for r in lib)
    name = lib[0]["name"]
    (sd / "skills").mkdir()
    (sd / "skills" / "deadbeef.json").write_text(json.dumps({
        "name": name, "kind": "segment", "limits": {"reach_m": 0.6},
        "failure_modes": ["reach_stall"],
        "bindings": {"fake": {"task": name, "policies": {"pi05": {"transport": "ssp"}}}},
        "evidence": {"fake": {"n": 4, "k": 3, "by_executor": {"pi05": {"n": 4, "k": 3}}}}}))
    (sd / "skills" / "cap.json").write_text(json.dumps({"kind": "capability", "ref": "x:y"}))
    ms.configure(runs)
    direct = bs.skills(sd)
    assert direct == ms.skills("session-main")
    code, out = _run(capsys, "skills", "session-main", "--runs", str(runs))
    assert code == 0 and out == json.dumps(direct)
    row = next(r for r in direct if r["name"] == name)
    assert row["source"] == "session" and row["bindings"] == {"fake": ["pi05", "scripted"]}
    assert row["evidence"] == {"fake": {"n": 4, "k": 3, "by_executor": {"pi05": {"n": 4, "k": 3}}}}
    assert row["limits"] == {"reach_m": 0.6} and row["failure_modes"] == ["reach_stall"]
    assert len(direct) == len(lib)  # overlay, not a duplicate; the capability row is skipped
    assert ms.skills("../x") == {"error": "unknown session"}


def test_rsi_campaigns_faces_are_byte_identical(tmp_path, capsys):
    """rsi_campaigns: the on-disk campaign list (what survives a restart) is the
    same object on the CLI, MCP and library faces; running first, then newest;
    open_brief resolves the intake evolve brief per task from the intake dirs."""
    runs, status, progress = _fixture(tmp_path)
    sd = runs / "session-main"
    done = {**_CAMPAIGN, "task": "stack", "status": "done", "best": 2, "cursor": 3, "live": None}
    for doc in (done, _CAMPAIGN):
        d = sd / "campaigns" / f"evolve-{doc['task']}"
        d.mkdir(parents=True)
        (d / "campaign.json").write_text(json.dumps(doc))
    (sd / "processing").mkdir()
    (sd / "processing" / "brief-aaaa.json").write_text(
        json.dumps({"kind": "evolve", "task": "kitchen_thaw", "rounds": 3}))
    (sd / "processing" / "brief-bbbb.json").write_text(json.dumps({"kind": "task", "task": "stack"}))
    # the done campaign was touched last: status must still sort running first
    import os
    os.utime(sd / "campaigns" / "evolve-stack" / "campaign.json", (9e9, 9e9))
    ms.configure(runs)
    lib = bs.rsi_campaigns(sd)
    code, out = _run(capsys, "rsi_campaigns", "session-main", "--runs", str(runs))
    assert code == 0 and out == json.dumps(lib) == json.dumps(ms.rsi_campaigns("session-main"))
    assert [c["task"] for c in lib] == ["kitchen_thaw", "stack"]
    running, finished = lib
    assert running == {"task": "kitchen_thaw", "status": "running", "cursor": 2, "rounds": 2,
                       "best": 1, "seeds": [1, 2], "arm": "auto", "node_rate_best": 0.75,
                       "updated": running["updated"],
                       "live": {"phase": "baseline", "message": _CAMPAIGN["live"]["message"],
                                "nodes_done": "1/2"},
                       "open_brief": "brief-aaaa.json"}
    assert finished["status"] == "done" and finished["live"] is None \
        and finished["open_brief"] is None and finished["updated"] == 9e9
    assert bs.rsi_run(sd, "kitchen_thaw")["open_brief"] == "brief-aaaa.json"
    assert bs.rsi_campaigns(runs / "session-b") == []
    assert ms.rsi_campaigns("../x") == {"error": "unknown session"}
    code, out = _run(capsys, "rsi_campaigns", "../x", "--runs", str(runs))
    assert code == 3 and json.loads(out) == {"error": "unknown session"}
