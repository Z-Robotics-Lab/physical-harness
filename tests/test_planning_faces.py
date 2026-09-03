"""plan_skill_task / submit_skill_plan on both call faces (board/mcp_server.py,
board/storecli.py) are the SAME board.planning function.

- plan_skill_task returns one stable JSON shape and writes NOTHING under runs/;
- the two faces answer byte-identically;
- submit_skill_plan refuses a planning_only / rejected / tampered record with
  no drop, and drops exactly the runtime's task brief for an executable one,
  answering with the brief_status handle (queued behind a live runtime, stalled
  without one); cancel_brief then stops it;
- one mocked end-to-end pass: UI-shaped request -> MCP face -> fake DeepSeek ->
  graph retrieval -> expansion -> validation -> the result the panel renders.
The model is always the canned server; DeepSeek is never called.
"""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest
from test_planner_vlm import _Server
from test_skill_planning import COFFEE_PLAN, COFFEE_TEXT, PACK_TEXT, pack_plan

from board import mcp_server as ms
from board import planning as bp
from board import storecli
from harness.events import SessionLog
from harness.unified_skill_graph import DEFAULT_GRAPH_PATH
from plugins.task import skill_planning as sp

pytestmark = pytest.mark.skipif(
    not DEFAULT_GRAPH_PATH.exists(),
    reason="generated unified_skill_graph.json not present in this checkout")

#: The response keys the panel and the agent rely on. Adding one is fine;
#: renaming or dropping one breaks the UI, so the set is pinned.
SHAPE = {"status", "instruction", "session", "seed", "goal", "channel", "retrieval",
         "selected_catalogue", "composite_plan", "expanded_plan", "executable",
         "missing_bindings", "unbound_oracles", "validation", "graph_provenance",
         "planner"}


@pytest.fixture()
def faces(tmp_path, monkeypatch):
    """A runs/ tree with a session-robocasa inbox, the MCP face configured on it,
    and the storecli face pointed at the same canned endpoint through the
    documented env override (what an operator would set)."""
    _Server.replies, _Server.requests = [], []
    server = HTTPServer(("127.0.0.1", 0), _Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    runs = tmp_path / "runs"
    session = runs / "session-robocasa"
    (session / "inbox").mkdir(parents=True)
    SessionLog(session / "session-log").append("runtime.boot", {"mode": "execution"})
    # Both faces derive the planner endpoint from the SAME documented env
    # override, so a plan's frozen-graph key (endpoint params included) is one.
    ms.configure(runs)
    monkeypatch.setenv(bp.ENV_BASE_URL, url)
    yield {"runs": runs, "session": session, "url": url}
    server.shutdown()


def _snapshot(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def test_plan_skill_task_has_a_stable_shape_and_writes_nothing(faces):
    before = _snapshot(faces["runs"])
    _Server.replies = [json.dumps(COFFEE_PLAN)]
    out = ms.plan_skill_task(COFFEE_TEXT, session="session-robocasa", seed=101)
    assert set(out) == SHAPE
    assert out["status"] == "planning_only" and out["executable"] is False
    assert out["session"] == "session-robocasa"
    json.dumps(out)                                        # wire-safe
    assert _snapshot(faces["runs"]) == before              # a preview is not evidence


def test_both_faces_answer_byte_identically(faces, capsys):
    _Server.replies = [json.dumps(COFFEE_PLAN)]
    mcp_out = ms.plan_skill_task(COFFEE_TEXT, seed=102)
    code = storecli.main(["plan_skill_task", "--instruction", COFFEE_TEXT, "--seed", "102",
                          "--runs", str(faces["runs"])])
    cli_out = capsys.readouterr().out.rstrip("\n")
    assert code == 0
    assert cli_out == json.dumps(mcp_out)
    # the CLI face refuses a call with nothing to plan
    assert storecli.main(["plan_skill_task", "--runs", str(faces["runs"])]) == 3
    assert "needs --instruction" in capsys.readouterr().out


def test_skill_library_unions_graph_and_runtime_bindings_on_both_faces(faces, capsys):
    before = _snapshot(faces["runs"])
    mcp_out = ms.skill_library()
    assert mcp_out["summary"]["graph_skills"] == 56
    assert mcp_out["summary"]["runtime_skills"] > 0
    nodes = {node["name"]: node for node in mcp_out["graph"]["nodes"]}
    assert nodes["CoffeeSetupMug"]["decomposition"] == ["Pick", "Place"]
    assert nodes["CoffeeSetupMug"]["bound"] is False
    assert "pick" in nodes["Pick"]["implementation_candidates"]
    runtime = {row["name"]: row for row in mcp_out["runtime_skills"]}
    assert runtime["pick"]["canonical"] == "Pick"
    assert runtime["pick"]["bindings"]
    code = storecli.main(["skill_library", "--runs", str(faces["runs"])])
    assert code == 0
    assert json.loads(capsys.readouterr().out) == mcp_out
    assert _snapshot(faces["runs"]) == before


def test_planning_only_record_cannot_be_submitted(faces):
    _Server.replies = [json.dumps(COFFEE_PLAN)]
    record = ms.plan_skill_task(COFFEE_TEXT, seed=103)["composite_plan"]
    before = _snapshot(faces["runs"])
    res = ms.submit_skill_plan(record, session="session-robocasa")
    assert res["submitted"] is False and res["status"] == "planning_only"
    assert "planning-only" in res["error"]
    assert _snapshot(faces["runs"]) == before              # nothing dropped
    # a rejected plan has no record worth submitting either
    _Server.replies = ["garbage", "garbage"]
    rejected = ms.plan_skill_task(COFFEE_TEXT, seed=104)
    assert rejected["status"] == "rejected"
    res = ms.submit_skill_plan(rejected["composite_plan"])
    assert res["submitted"] is False and res["status"] == "planning_only"  # graph channel
    res = ms.submit_skill_plan({"channel": "pack_all_robocasa", "task": "pack_all_robocasa",
                                "instruction": PACK_TEXT, "plan": rejected["composite_plan"]["plan"]})
    assert res["submitted"] is False and res["status"] == "rejected"
    assert _snapshot(faces["runs"]) == before


def test_executable_record_drops_the_runtime_brief_and_hands_back_a_handle(faces, live_runtime):
    live_runtime(faces["session"])                          # else honestly `stalled`
    _Server.replies = [json.dumps(pack_plan())]
    out = ms.plan_skill_task(PACK_TEXT, seed=105)
    assert out["status"] == "executable"
    res = ms.submit_skill_plan(out["composite_plan"], session="session-robocasa",
                               seed=424244, max_actuations=24)
    assert res["submitted"] is True and res["state"] == "queued"
    assert res["plan_id"] == out["composite_plan"]["plan_id"]
    assert res["brief_id"].startswith("brief-") and res["queue_position"] == 1
    dropped = json.loads((faces["session"] / "inbox" / res["brief_id"]).read_text())
    # selector + budgets + the bounded instruction: exactly what _BRIEF_KEYS admits
    assert dropped == {"kind": "task", "task": "pack_all_robocasa", "instruction": PACK_TEXT,
                       "seed": 424244, "max_actuations": 24}
    assert "runtime re-plans" in res["execution_note"]
    # the lifecycle faces are the existing ones
    status = ms.brief_status(res["brief_id"], session="session-robocasa")
    assert status["state"] == "queued" and status["task"] == "pack_all_robocasa"
    assert ms.cancel_brief(res["brief_id"], session="session-robocasa")["requested"] is True


def test_executable_record_without_a_live_runtime_is_stalled_not_queued(faces):
    _Server.replies = [json.dumps(pack_plan())]
    record = ms.plan_skill_task(PACK_TEXT, seed=106)["composite_plan"]
    res = ms.submit_skill_plan(record, session="session-robocasa")
    assert res["submitted"] is True and res["state"] == "stalled"


def test_cli_submit_face_is_the_same_function(faces, capsys):
    _Server.replies = [json.dumps(pack_plan())]
    record = ms.plan_skill_task(PACK_TEXT, seed=107)["composite_plan"]
    code = storecli.main(["submit_skill_plan", "--plan", json.dumps(record), "--seed", "424245",
                          "--session", "session-robocasa", "--runs", str(faces["runs"])])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["submitted"] is True and out["task"] == "pack_all_robocasa"
    dropped = json.loads((faces["session"] / "inbox" / out["brief_id"]).read_text())
    assert dropped["seed"] == 424245 and dropped["task"] == "pack_all_robocasa"
    # bad JSON is a refusal, not a drop
    assert storecli.main(["submit_skill_plan", "--plan", "{not json",
                          "--runs", str(faces["runs"])]) == 3
    assert "not JSON" in capsys.readouterr().out


def test_unreachable_endpoint_is_an_honest_error_never_an_invented_plan(faces):
    ms.configure(faces["runs"], planner_params={"endpoint_params": {"base_url": "http://127.0.0.1:9/v1"}})
    out = ms.plan_skill_task(COFFEE_TEXT, seed=108)
    assert out["status"] == "rejected" and "unreachable" in out["error"]
    assert out["executable"] is False


def test_planner_params_from_env_never_reads_a_secret():
    assert bp.planner_params_from_env({}) is None
    p = bp.planner_params_from_env({"PH_PLANNER_BASE_URL": "http://h:1/v1",
                                    "PH_PLANNER_MODEL": "m", "DEEPSEEK_API_KEY": "sk-secret"})
    assert p == {"endpoint_params": {"base_url": "http://h:1/v1",
                                     "api_key_env": "DEEPSEEK_API_KEY", "model": "m"}}
    assert "sk-secret" not in json.dumps(p)


def test_mocked_end_to_end_ui_request_to_render_model(faces):
    """UI request -> MCP face -> fake DeepSeek -> graph retrieval -> expansion
    -> validation -> the model the panel renders (no UI code, the wire dict)."""
    _Server.replies = [json.dumps(COFFEE_PLAN)]
    ui_request = {"instruction": COFFEE_TEXT, "session": "session-robocasa",
                  "channel": "auto", "expand": True}
    out = ms.plan_skill_task(ui_request["instruction"], ui_request["session"],
                             expand=ui_request["expand"], channel=ui_request["channel"],
                             seed=109)
    # the fake model saw the compact catalogue (not 56 skills) with taxonomy
    sent = json.loads(_Server.requests[0]["messages"][1]["content"].split("\n", 1)[1]
                      .rsplit("\n\n", 1)[0])
    assert len(sent["skills"]) < 56 and "taxonomy" in sent
    # what the panel shows
    assert out["status"] == "planning_only"
    labels = [row["label"] for row in out["expanded_plan"]["chain"]] + [out["expanded_plan"]["terminal"]]
    assert labels == ["CoffeeSetupMug.pick", "CoffeeSetupMug.place",
                      "StartCoffeeMachine.execute", "done"]
    by_node = {n["id"]: n for n in out["expanded_plan"]["nodes"]}
    assert [d["skill"] for d in by_node["setup-mug"]["decomposition"]] == ["Pick", "Place"]
    assert [d["skill"] for d in by_node["start-machine"]["decomposition"]] == ["PressButton"]
    assert out["executable"] is False                       # Execute stays disabled
    assert len(out["missing_bindings"]) == 3                # shown, never hidden
    assert out["validation"]["ok"] is True
    assert sp.STATUS_PLANNING_ONLY == out["status"]
