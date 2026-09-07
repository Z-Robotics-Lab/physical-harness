"""The proposals inbox: ``submit_proposal`` / ``proposals`` byte-equal across the
three faces (library, CLI, MCP) and the shape gate at the trust boundary. The evolve
loop's consumption of a pending proposal rides tests/test_evolve_e2e.py."""

from __future__ import annotations

import json

import pytest
from test_mission_e2e import SESSION

from board import mcp_server as ms
from board import store as bs
from board import storecli


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    sd = runs / SESSION
    (sd / "session-log").mkdir(parents=True)          # what board.store.is_session checks
    (sd / "session-log" / "rows.jsonl").write_text("")
    return runs, sd


def _raw(task, kind, payload, note="") -> str:
    return json.dumps({"task": task, "kind": kind, "payload": payload, "note": note})


def test_three_faces_agree_on_submit_and_list(session, capsys):
    runs, sd = session
    ms.configure(runs)
    base = ["--runs", str(runs), "--session", SESSION]
    raw = _raw("other_task", "executor", {"to": "alt", "node": "grab-0"}, "faces")
    lib = bs.submit_proposal(sd, raw)
    code = storecli.main(["submit_proposal", "--brief", raw] + base)
    cli = json.loads(capsys.readouterr().out)
    mcp = ms.submit_proposal(json.loads(raw))
    assert code == 0 and all(r["inbox"] == str(sd / "proposals") for r in (lib, cli, mcp))
    ids = [lib["submitted"], cli["submitted"], mcp["submitted"]]
    assert all((sd / "proposals" / f"{i}.json").is_file() for i in ids)
    code = storecli.main(["proposals"] + base)
    out = capsys.readouterr().out.rstrip("\n")
    assert code == 0 and out == json.dumps(bs.proposals(sd)) == json.dumps(ms.proposals())
    rows = bs.proposals(sd)
    assert [r["id"] for r in rows] == ids   # submission order
    assert rows[0] == {"id": ids[0], "task": "other_task", "kind": "executor",
                       "payload": {"to": "alt", "node": "grab-0"}, "note": "faces", "applied": None}
    # the shape gate, same answer on every face
    for bad in ('{"task":"t","kind":"magic","payload":{}}', '{"task":"t","kind":"card"}',
                '{"task":"t","kind":"card","payload":{},"extra":1}', "not json"):
        assert "error" in bs.submit_proposal(sd, bad), bad
    assert ms.submit_proposal({"task": "t", "kind": "card"})["error"] == "proposal needs payload: object"
    assert ms.proposals("../x") == {"error": "unknown session"}
    assert storecli.main(["submit_proposal"] + base) == 3   # needs --brief
    capsys.readouterr()
    assert len(bs.proposals(sd)) == 3
