"""The evolve loop end to end: an ``evolve`` brief through the REAL scripts/harness_runtime.py
(evolution mode) spawning the REAL scripts/evolve.py, whose suites run in REAL child
processes mapped onto the agent's working copy. No simulator: the fake card package
``tests/fakes/evolve_card`` (reach then grab; grab only lands once ``Driver.STOP`` drops
below 0.5). A canned model reads the code, is refused on the frozen file, edits the
copy, probes one seed, finishes: 0/2 -> 2/2, confirmed on fresh seeds, accepted as the
incumbent. Round 2 gives up. Then cancel mid-run and resume from the cursor.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
from pathlib import Path

import pytest
from test_mission_e2e import SESSION, _kinds, _Runtime, _wait

from board import mcp_server as ms
from board import store as bs
from board import storecli
from harness import media
from harness.manifest import mount_params
from scripts import evolve

CARD = "fakes.evolve_card"
TASK = "e2e_evolve"
_CARD = f"""
[task_bindings.{TASK}]
env = "{CARD}.stage:env_provider"
policy = "{CARD}.stage:policy_provider"
planner = "{CARD}.stage:planner_provider"
catalogue = "{CARD}.stage:CATALOGUE"
records = "{CARD}.stage:RECORDS"
oracles = "{CARD}.stage:ORACLES"
episodic = true
episode = "{CARD}.stage:EPISODE"
segment_specs = "{CARD}.stage:SEGMENT_SPECS"
max_replans = 1
"""

DIAGNOSE = {"action": "diagnose", "contrast": "both seeds die at grab-0 with the base 0.65 m out",
            "hypothesis": "the loaded standoff is too large", "plan": "lower STOP"}
READ = {"thought": "grab dies; read the driver", "action": "read", "path": "stage.py", "start": 60, "end": 100}
EDIT_FROZEN = {"action": "edit", "path": "predicates.py", "old": "FROZEN = True", "new": "FROZEN = False"}
EDIT = {"thought": "the standoff is too large", "action": "edit", "path": "stage.py",
        "old": "STOP = 0.65", "new": "STOP = 0.3"}
RUN = {"action": "run", "seed": 1}
FINISH = {"action": "finish", "summary": "lower the loaded standoff so the grab lands"}
GIVE_UP = {"action": "give_up", "reason": "every seed already succeeds"}
NOTE = {"action": "note", "text": "STOP lives in stage.py; 0.3 lands the grab", "keep": False}


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    rt = _Runtime(tmp_path_factory.mktemp("runs"), card=_CARD,
                  canned=[READ, DIAGNOSE, EDIT_FROZEN, EDIT, RUN, FINISH, NOTE, GIVE_UP, NOTE], mode="evolution")
    rt.campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    yield rt
    rt.stop()


def _doc(runtime) -> dict:
    return json.loads(runtime.campaign.read_text())


_LIVE: list[dict] = []


def _poll(stop: threading.Event, path: Path, seen: list) -> None:
    while not stop.is_set():
        try:
            live = json.loads(path.read_text()).get("live")
            if live and live != (seen[-1] if seen else None):
                seen.append(live)
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(0.005)


@pytest.fixture(scope="module")
def two_rounds(runtime):
    # a pending operator proposal rides into round 1's brief
    bs.submit_proposal(runtime.session, json.dumps({"task": TASK, "kind": "tunables",
                                                    "payload": {"node": "grab-0", "stop": 0.3}, "note": "try 0.3"}))
    stop = threading.Event()
    t = threading.Thread(target=_poll, args=(stop, runtime.campaign, _LIVE), daemon=True)
    t.start()
    try:
        return runtime.run({"kind": "evolve", "task": TASK, "seeds": [1, 2], "rounds": 2, "arm": "auto"})
    finally:
        stop.set()
        t.join()


def test_live_block_shows_every_phase_and_done_at_the_end(runtime, two_rounds):
    phases = [l["phase"] for l in _LIVE]
    assert {"baseline", "probe", "retest", "confirm", "done"} <= set(phases), sorted(set(phases))
    # the rolling message log keeps every phase transition, even ones too quick to poll
    texts = {m["text"] for l in _LIVE for m in l["messages"]}
    assert any("LLM 分析" in t for t in texts) and any("单种子试跑" in t for t in texts), texts
    base = [l for l in _LIVE if l["phase"] == "baseline" and l["round"] == 1]
    # seeds run as parallel children: a seed too quick for the poller still has its own live row
    seen = {l["seed"] for l in base} | {int(k) for l in base for k in (l.get("seeds_live") or {})}
    assert seen >= {1, 2} and {"reach-0", "grab-0"} & ({l["node"] for l in base}
                                                        | {r.get("node") for l in base for r in (l.get("seeds_live") or {}).values()})
    assert all(l["seeds_total"] == 2 for l in base if l["seed_index"] is not None)
    probe = [l for l in _LIVE if l["phase"] == "probe" and l["seed_index"] is not None]
    assert probe and all(l["seeds_total"] == 1 and l["seed"] == 1 for l in probe)
    assert any("运行中" in l["message"] for l in base)   # whichever parallel child the poller caught
    trails = [tuple(n["ok"] for n in l["nodes"]) for l in _LIVE if l["nodes"]]
    assert trails and all(t in ((None, None), (True, None), (True, True), (True, False)) for t in trails), trails
    live = _doc(runtime)["live"]
    assert live["phase"] == "done" and live["round"] == 2 and live["message"] == "已完成 2 轮"
    assert bs.rsi_run(runtime.session, TASK)["live"] == live


def test_round_one_edits_the_copy_and_is_accepted_round_two_gives_up(runtime, two_rounds):
    _, rows = two_rounds
    doc = _doc(runtime)
    assert doc["status"] == "done" and doc["cursor"] == 2 and doc["best"] == 4
    assert doc["seeds"] == [1, 4] and doc["card"] == CARD and doc["arm"] == "auto"
    r1, r2 = doc["rounds"]
    assert (r1["before"], r1["after"], r1["accepted"], r1["published"], r1["outcome"]) == (0, 2, True, False, "improved")
    assert r1["tried"]["kind"] == "edit" and r1["tried"]["node"] == "grab-0"
    assert r1["tried"]["detail"]["files"] == ["stage.py"] and "+    STOP = 0.3" in r1["tried"]["detail"]["diff"]
    assert r1["before_score"] == [0, 0.3333] and r1["after_score"] == [2, 1.0]
    assert r1["confirm"] == {"seeds": [3, 4], "before": 0, "after": 2, "regressions": []}
    assert r1["accepted_reason"].startswith("gained ") and "1:grab-0" in r1["accepted_reason"]
    assert r1["proposal"]["kind"] == "tunables" and r1["parent"] == 0
    # the read before the diagnosis is refused (a call, not a step); the diagnosis is kept on the row
    assert r1["llm"]["status"] == "finished" and r1["llm"]["actions"] == {"read": 1, "diagnose": 1, "edit": 2, "run": 1, "finish": 1, "note": 1}
    assert any("frozen" in e for e in r1["llm"]["errors"]) and any("diagnose first" in e for e in r1["llm"]["errors"])
    assert r1["diagnosis"] == [DIAGNOSE | {}] or r1["diagnosis"][0]["hypothesis"] == DIAGNOSE["hypothesis"]
    assert [p["seed"] for p in r1["probes"]] == [1] and r1["probes"][0]["success"] is True
    assert r1["probes"][0]["compare"]["gains"] == ["1:grab-0", "1:task"]
    assert r1["usage"]["model_calls"] == 7 and r1["usage"]["episode_attempts"] == 2 + 1 + 2 + 2 + 2
    # the closing note call wrote the model's notes; they ride into the next brief
    assert "STOP lives in stage.py" in (runtime.session / f"campaigns/evolve-{TASK}" / "notes.md").read_text()
    # the incumbent IS the edited copy; the next round starts from it
    assert doc["incumbent"] == {"workspace": f"campaigns/evolve-{TASK}/work/r1e1", "round": 1, "tunables": {}}
    assert [e["accepted"] for e in r1["evaluations"]] == [True] and r1["evaluations"][0]["gains"] == ["1:grab-0", "1:task", "2:grab-0", "2:task"]
    ws = runtime.session / doc["incumbent"]["workspace"]
    assert "STOP = 0.3" in (ws / "stage.py").read_text() and (ws / "predicates.py").read_text().count("FROZEN = True")
    assert "STOP = 0.3" in (runtime.session / f"campaigns/evolve-{TASK}/work/r2" / "stage.py").read_text()
    assert (r2["before"], r2["after"], r2["accepted"], r2["outcome"], r2["parent"]) == (4, None, False, "none", 1)
    assert r2["tried"]["kind"] == "none" and r2["llm"]["status"] == "gave_up" and r2["needs"] == ["edit"]
    assert r2["usage"]["episode_attempts"] == 4 and r2["after_seeds"] == []
    # the notebook is the memory: verdicts, the diff, the probe
    nb = (runtime.session / f"campaigns/evolve-{TASK}" / "notebook.md").read_text()
    assert "## Round 1 — ACCEPTED" in nb and "+    STOP = 0.3" in nb and "probe seed 1: success" in nb
    assert "## Round 2 — NO EDIT" in nb
    audit = json.loads((runtime.session / f"campaigns/evolve-{TASK}/llm/round-1.json").read_text())
    assert audit["status"] == "finished" and audit["calls"] == 7 and audit["diagnoses"][0]["plan"] == "lower STOP"
    assert "Operator proposal pending" in audit["messages"][1]["content"]
    assert audit["messages"][2]["content"] == json.dumps(READ)
    # media: verified segments kept as clips per phase, never overwritten across phases
    assert {p.rsplit(f"/{TASK}/", 1)[-1] for p in r1["media"] if "/retest/" in p} == {
        f"{s}/{n}.gif" for s in (1, 2) for n in ("reach-0", "grab-0")}
    assert {p.rsplit(f"/{TASK}/", 1)[-1] for p in r2["media"]} == {
        f"{s}/{n}.gif" for s in (1, 2, 3, 4) for n in ("reach-0", "grab-0")}
    assert set(r1["media"]).isdisjoint(r2["media"])
    for rel in r1["media"]:
        f = runtime.session / rel
        assert f.is_file() and 0 < f.stat().st_size <= media.MAX_BYTES, rel
    # development acceptance installs nothing
    assert not list((runtime.session / "skills").glob("*.json"))
    steps = _kinds(rows, "rsi_step")
    assert [(s["round"], s["before"], s["after"], s["accepted"]) for s in steps] == [(1, 0, 2, True), (2, 4, None, False)]
    assert [s["per_seed"] for s in steps] == [r1["per_seed"], r2["per_seed"]]
    assert all(row["success"] for row in r1["after_seeds"]) and all(not row["success"] for row in r1["per_seed"])
    assert [n["ok"] for n in r1["per_seed"][0]["nodes"]] == [True, False]
    assert not _kinds(rows, "runtime.task_error")


def test_three_faces_agree_on_the_real_campaign(runtime, two_rounds, capsys):
    sd = runtime.session
    ms.configure(runtime.runs)
    base = ["--runs", str(runtime.runs), "--session", SESSION]
    cases = [
        (["rsi_run", TASK], bs.rsi_run(sd, TASK), ms.rsi_run(TASK)),
        (["rsi_run", TASK, "--round", "1"], bs.rsi_run(sd, TASK, 1), ms.rsi_run(TASK, round=1)),
        (["rsi_series", TASK], bs.rsi_series(sd, TASK), ms.rsi_series(TASK)),
        (["rsi_frames", TASK, "--round", "1"], bs.rsi_frames(sd, TASK, 1), ms.rsi_frames(TASK, 1)),
    ]
    for argv, lib, mcp in cases:
        code = storecli.main(argv + base)
        out = capsys.readouterr().out.rstrip("\n")
        assert code == 0 and out == json.dumps(lib) == json.dumps(mcp), argv
    doc = _doc(runtime)
    series = bs.rsi_series(sd, TASK)
    assert bs.rsi_run(sd, TASK) == {**doc, "rounds": series[-bs.RUN_TAIL:], "latest": series[-1], "open_brief": None}
    assert all("per_seed" not in row for row in series)
    assert bs.rsi_run(sd, TASK, 1)["rounds"] == [doc["rounds"][0]] and bs.rsi_run(sd, TASK, 999)["rounds"] == []
    assert [(s["node_rate"], s["by_task"]) for s in series] == [
        ({"before": 0.5, "after": 1.0, "best": 1.0}, {"grab": {"before": 0.0, "after": 1.0}, "reach": {"before": 1.0, "after": 1.0}}),
        ({"before": 1.0, "after": None, "best": 1.0}, {"grab": {"before": 1.0, "after": None}, "reach": {"before": 1.0, "after": None}})]
    camp = bs.rsi_campaigns(runtime.runs / SESSION)[0]
    assert camp["node_rate_best"] == 1.0 and camp["accepted_rounds"] == [1] and camp["usage"]["llm_tokens"] is not None
    assert bs.rsi_command_summary(sd, TASK, 1)["actions"] == doc["rounds"][0]["llm"]["actions"]
    assert bs.rsi_frames(sd, TASK, 1) == {"media": doc["rounds"][0]["media"], "dropped": doc["rounds"][0]["media_dropped"]}


def test_cancel_lands_and_resubmit_resumes_from_cursor(runtime, two_rounds):
    (runtime.runs / "canned.json").write_text(json.dumps(GIVE_UP))   # every call: give up
    before = len(bs.chain_rows(runtime.session))
    name = bs.submit_brief(runtime.runs, json.dumps({"kind": "evolve", "task": TASK, "rounds": 4}), session=SESSION)["submitted"]
    _wait(lambda: (runtime.session / "processing" / name).exists(), 30, "claim")
    assert bs.cancel_brief(runtime.session, name)["requested"] is True
    _wait(lambda: (runtime.session / "cancelled" / name).exists(), 60, "cancelled filing")
    rows = bs.chain_rows(runtime.session)[before:]
    assert _kinds(rows, "runtime.task_cancelled")[0]["brief"] == name
    doc = _doc(runtime)
    assert doc["status"] == "cancelled" and doc["cursor"] == 2 and len(doc["rounds"]) == 2
    _, rows = runtime.run({"kind": "evolve", "task": TASK, "rounds": 1})
    doc = _doc(runtime)
    assert doc["status"] == "done" and doc["cursor"] == 3 and [r["round"] for r in doc["rounds"]] == [1, 2, 3]
    assert doc["rounds"][2]["tried"]["kind"] == "none" and doc["rounds"][2]["best"] == 4
    assert doc["incumbent"]["round"] == 1   # a rejected round never moves the incumbent
    assert [s["round"] for s in _kinds(rows, "rsi_step")] == [3]
    # continuous = until cancelled: the flag alone, no per-cycle budgets
    name = bs.submit_brief(runtime.runs, json.dumps({"kind": "evolve", "task": TASK, "continuous": True}),
                           session=SESSION)["submitted"]
    _wait(lambda: _doc(runtime).get("cursor", 0) >= 5, 90, "two more rounds under continuous")
    assert bs.cancel_brief(runtime.session, name)["requested"] is True
    _wait(lambda: (runtime.session / "cancelled" / name).exists(), 60, "cancel continuous")
    for bad in ({"continuous": "false"}, {"max_model_calls": 8}, {"proposer": "rules"}):
        _, rows = runtime.run({"kind": "evolve", "task": TASK, **bad}, expect="failed")
        assert not _kinds(rows, "rsi_step")


def test_evolve_is_refused_outside_evolution_mode(tmp_path_factory):
    rt = _Runtime(tmp_path_factory.mktemp("runs"), card=_CARD)
    try:
        _, rows = rt.run({"kind": "evolve", "task": TASK, "rounds": 1}, expect="failed")
        assert "evolution mode" in _kinds(rows, "runtime.task_error")[0]["error"]
    finally:
        rt.stop()


# ── the pure parts ──────────────────────────────────────────────────────────────

def _row(ok: list, success: bool, kinds=None) -> dict:
    kinds = kinds or ["segment", "verify", "segment", "verify"]
    return {"success": success, "trail": [{"id": f"n{i}", "kind": k, "ok": o} for i, (k, o) in enumerate(zip(kinds, ok))]}


def test_compare_accepts_a_net_milestone_gain_but_never_a_lost_success():
    # verify nodes (n1, n3) are the milestones; segment nodes' self-reports are not
    before = {"seeds": {"1": _row([True, True, True, False], False), "2": _row([True, True, None, None], False)}}
    assert evolve.milestones(before["seeds"]["1"]) == {"n1": True, "n3": False, "task": False}
    gain = {"seeds": {"1": _row([True, True, True, True], True), "2": _row([True, True, None, None], False)}}
    assert evolve.compare(before, gain) == {"gains": ["1:n3", "1:task"], "regressions": [], "lost_success": [],
                                            "successes": [0, 1], "accepted": True}
    # a trade that gains more than it loses is progress
    swap = {"seeds": {"1": _row([True, True, True, True], True), "2": _row([True, False, None, None], False)}}
    assert evolve.compare(before, swap) == {"gains": ["1:n3", "1:task"], "regressions": ["2:n1"], "lost_success": [],
                                            "successes": [0, 1], "accepted": True}
    # an even trade is not
    even = {"seeds": {"1": _row([True, True, True, True], False), "2": _row([True, False, None, None], False)}}
    assert evolve.compare(before, even)["accepted"] is False
    # the number of seeds finishing the task may not drop: one finished seed swapped for
    # another (same count, net milestone gain) passes; a finished seed lost for partial
    # milestones elsewhere does not, whatever the net
    done = {"seeds": {"1": _row([True, True, True, True], True), "2": _row([True, True, None, None], False)}}
    swapped = {"seeds": {"1": _row([True, True, True, True], False), "2": _row([True, True, True, True], True)}}
    c = evolve.compare(done, swapped)
    assert c["lost_success"] == ["1"] and c["successes"] == [1, 1] and c["accepted"] is True
    partial = {"seeds": {"1": _row([True, True, True, False], False), "2": _row([True, True, True, True], False),
                         "3": _row([True, True, True, True], False)}}
    done3 = {"seeds": {**done["seeds"], "3": _row([True, False, None, None], False)}}
    c = evolve.compare(done3, partial)
    assert c["successes"] == [1, 0] and len(c["gains"]) > len(c["regressions"]) and c["accepted"] is False
    assert evolve.compare(before, before)["accepted"] is False
    assert evolve.compare(before, {"seeds": {}})["regressions"] == ["1:n1", "2:n1"]   # unmeasured = lost
    # no verify kind: every node's ok is the oracle's verify row
    seg = {"seeds": {"1": _row([True, False], False, ["segment", "segment"])}}
    assert evolve.milestones(seg["seeds"]["1"]) == {"n0": True, "n1": False, "task": False}
    assert evolve.score(before) == [0, 0.3333]


def test_workspace_edits_compile_and_the_frozen_files_stay_frozen(tmp_path):
    stock = Path(__file__).parent / "fakes" / "evolve_card"
    ws = evolve.Workspace.create(tmp_path / "r1", stock, stock)
    assert not ws.changed() and ws.diff() == ""
    with pytest.raises(ValueError, match="frozen"):
        ws.edit("predicates.py", "FROZEN = True", "FROZEN = False")
    with pytest.raises(ValueError, match="exactly once"):
        ws.edit("stage.py", "return", "pass")
    with pytest.raises(ValueError, match="would not compile"):
        ws.edit("stage.py", "STOP = 0.65", "STOP = = 0.3")
    with pytest.raises(ValueError, match="inside the card copy"):
        ws.read("../outside.py")
    assert "written" in ws.edit("stage.py", "STOP = 0.65", "STOP = 0.3")
    assert ws.changed() and "+    STOP = 0.3" in ws.diff() and ws.protected_ok() is None
    assert "helper.py written" in ws.write("helper.py", "X = 1\n")
    assert "  71| " in ws.read("stage.py", 71, 71) and "no match" == ws.grep("zzz_nothing")
    assert ws.grep("STOP =").startswith("stage.py:")
    (ws.path / "predicates.py").write_text("FROZEN = False\n")   # tampered outside the tools
    assert "frozen" in ws.protected_ok()


def test_overlay_maps_a_package_onto_a_directory_and_tunables_reach_every_hosted_provider(tmp_path, monkeypatch):
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "__init__.py").write_text("")
    (copy / "mod.py").write_text("VALUE = 'from the copy'\n")
    evolve.install_overlay({"evolve_overlay_probe": str(copy)})
    assert importlib.import_module("evolve_overlay_probe.mod").VALUE == "from the copy"
    ref = "plugins.embodiment_robocasa.recycle_driver:provider"
    stock = mount_params(ref)["tunables"]["carry_stop"]
    monkeypatch.setenv(evolve.OVERRIDE_ENV, json.dumps({"plugins.embodiment_robocasa": {"tunables": {"carry_stop": 0.2}}}))
    assert mount_params(ref)["tunables"]["carry_stop"] == 0.2 != stock
    assert mount_params("plugins.embodiment_robocasa.drivers:provider")["tunables"]["carry_stop"] == 0.2
    assert mount_params(ref)["tunables"]["stall_k"] == 40   # untouched keys survive the merge
    monkeypatch.setenv(evolve.OVERRIDE_ENV, json.dumps({ref: {"tunables": {"carry_stop": 0.1}}}))
    assert mount_params(ref)["tunables"]["carry_stop"] == 0.1   # an exact ref still wins
    monkeypatch.delenv(evolve.OVERRIDE_ENV)
    assert mount_params(ref)["tunables"]["carry_stop"] == stock
    assert os.environ.get(evolve.OVERRIDE_ENV) is None
