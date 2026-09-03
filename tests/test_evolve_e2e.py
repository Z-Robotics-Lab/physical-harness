"""The lightweight evolve loop end to end: an ``evolve`` brief through the REAL
scripts/harness_runtime.py (evolution mode) spawning the REAL scripts/evolve.py.
No simulator: a tmp task card (PH_PLUGINS_EXTRA) whose ``grab`` record binds
scripted + an ``alt`` executor; the scripted driver fails ``grab`` deterministically,
``alt`` succeeds, and the record's ``by_executor`` evidence says so -- so round 1
switches grab-0's executor (published: 0/2 -> 2/2), round 2 has nothing to try.
Then cancel mid-run and resubmit: the loop resumes from cursor (round 3).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from test_mission_e2e import SESSION, _Runtime, _kinds, _wait

from board import mcp_server as ms
from board import store as bs
from board import storecli
from harness import fakes, media, protocol
from harness.skill_library import segment_specs

EMB = "test_evolve_e2e:env_provider"
ALT = "test_evolve_e2e:alt_provider"

RECORDS = {
    "reach": {"id": "reach", "name": "reach", "kind": "segment", "args": {},
              "bindings": {EMB: {"task": "reach"}}},
    "grab": {"id": "grab", "name": "grab", "kind": "segment", "args": {},
             "bindings": {EMB: {"policies": {"scripted": {"task": "grab"},
                                             "alt": {"ref": ALT}}}},
             "evidence": {EMB: {"n": 4, "k": 0, "by_executor": {"alt": {"n": 4, "k": 4}}}}},
}
CATALOGUE = {"reach": {}, "grab": {}}
ORACLES = ("seg_ok",)
EPISODE = {"task": "reach", "horizon": 20}
SEGMENT_SPECS = segment_specs(
    {k: protocol.SkillRecordV0.from_dict(v) for k, v in RECORDS.items()}, EMB)


_OBS = {"robot0_gripper_qpos": [0.03, -0.03], "robot0_gripper_qvel": [0.0, 0.0],
        "robot0_joint_vel": [0.0] * 7, "robot0_eef_pos": [0.0, 0.0, 1.0],
        "cubeA_pos": [0.0, 0.0, 0.1]}   # what the governed loop's step features read


class _Handle(fakes._FakeEnvHandle):
    """The stdlib fake env (synthetic 128px ``frame()`` for the media recorder),
    slowed so a cancel lands mid-campaign."""

    def reset(self):
        time.sleep(0.2)
        super().reset()
        return dict(_OBS)

    def step(self, action):
        self.t += 1
        return dict(_OBS), 0.0, False, {}


class _Env:
    def make_env(self, spec):
        return _Handle()

    def tasks(self):
        return ("reach", "grab")

    def object_key(self, spec):
        raise AssertionError("heterogeneous segment path never reads object_key")

    def success(self, obs, spec, start_z):
        return True

    def terminal_success(self, obs, spec, start_z, env=None):
        return True


class _Driver:
    """Scripted episode driver: grab succeeds only under a non-scripted executor.
    Each segment drives STEPS env steps (so the media recorder sees frames)."""
    STEPS = 8
    n = 0

    @property
    def exhausted(self):
        return self.n >= self.STEPS

    def observe_once(self, obs):
        pass

    def on_handback(self):
        pass

    def act(self, obs):
        self.n += 1
        return (0.0,)

    def enter_segment(self, env, spec, executor=None):
        self.n = 0
        self.ok = spec.task != "grab" or executor is not None

    def segment_success(self, env):
        return self.ok


class _Policy:
    def make_driver(self, spec):
        return _Driver()


class _Alt:
    def make_driver(self, spec):
        return object()


class _Planner:
    identity = "evolve_e2e:fixed"

    def plan(self, brief):
        return {"goal": "reach then grab",
                "nodes": [{"id": "reach-0", "skill": "reach", "kind": "segment", "args": {}, "after": []},
                          {"id": "grab-0", "skill": "grab", "kind": "segment", "args": {},
                           "after": ["reach-0"]}],
                "verify": [{"after": "reach-0", "predicate": "seg_ok"},
                           {"after": "grab-0", "predicate": "seg_ok"}],
                "rationale": "fixed"}


def env_provider():
    return _Env()


def policy_provider():
    return _Policy()


def alt_provider(**params):
    return _Alt()


def planner_provider():
    return _Planner()


_CARD = f"""
[task_bindings.e2e_evolve]
env = "{EMB}"
policy = "test_evolve_e2e:policy_provider"
planner = "test_evolve_e2e:planner_provider"
catalogue = "test_evolve_e2e:CATALOGUE"
records = "test_evolve_e2e:RECORDS"
oracles = "test_evolve_e2e:ORACLES"
episodic = true
episode = "test_evolve_e2e:EPISODE"
segment_specs = "test_evolve_e2e:SEGMENT_SPECS"
max_replans = 1
"""

TASK = "e2e_evolve"


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    rt = _Runtime(tmp_path_factory.mktemp("runs"), card=_CARD, mode="evolution")
    rt.campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    yield rt
    rt.stop()


def _doc(runtime) -> dict:
    return json.loads(runtime.campaign.read_text())


_LIVE: list[dict] = []   # every distinct ``live`` block the poller saw during two_rounds


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
    stop = threading.Event()
    t = threading.Thread(target=_poll, args=(stop, runtime.campaign, _LIVE), daemon=True)
    t.start()
    try:
        return runtime.run({"kind": "evolve", "task": TASK, "seeds": [1, 2], "rounds": 2,
                            "arm": "auto"})
    finally:
        stop.set()
        t.join()


def test_live_block_shows_progress_during_the_run_and_done_at_the_end(runtime, two_rounds):
    """The operator's 「看不到进度」: campaign.json's ``live`` advances per phase, seed
    and node while the loop runs (polled by a thread), and reads ``done`` at the end."""
    base = [l for l in _LIVE if l["phase"] == "baseline" and l["round"] == 1]
    assert [l["seed_index"] for l in base if l["seed_index"] is not None][:1] == [0]
    assert {l["seed_index"] for l in base} >= {0, 1}, base
    assert {l["seed"] for l in base} >= {1, 2} and {"reach-0", "grab-0"} & {l["node"] for l in base}
    partial = [l["per_seed_partial"] for l in base if len(l["per_seed_partial"]) == 1]
    assert partial and all(p[0]["seed"] == 1 and p[0]["first_death"] == "grab-0" for p in partial), partial
    assert all(l["seeds_total"] == 2 and l["message"] and l["started_at"] for l in _LIVE)
    assert any(l["phase"] == "retest" and l["tried"]["kind"] == "executor" for l in _LIVE)
    assert "种子 1 运行中" in next(l["message"] for l in base if l["seed"] == 1)
    # the node trail of seed 1 (a plain list in plan order): ok flips None -> True in
    # plan order as verify rows land, and the rolling message log accumulates
    trails = [tuple(n["ok"] for n in l["nodes"]) for l in _LIVE if l["nodes"]]
    assert trails[0] == (None, None) and any(t[0] is True for t in trails), trails
    assert all(t in ((None, None), (True, None), (True, True), (True, False)) for t in trails), trails
    assert all(l["nodes"][0]["id"] == "reach-0" and l["nodes"][1]["skill"] == "grab" for l in _LIVE if l["nodes"])
    assert all(l["seed_started_at"] for l in base if l["seed"] is not None)
    assert [m["text"] for m in _LIVE[-1]["messages"]][-1] == "已完成 2 轮"
    assert len(_LIVE[-1]["messages"]) == 20 and all(m["ts"] for m in _LIVE[-1]["messages"])
    live = _doc(runtime)["live"]
    assert live["phase"] == "done" and live["round"] == 2 and live["last_round_s"] is not None
    assert live["message"] == "已完成 2 轮" and bs.rsi_run(runtime.session, TASK)["live"] == live


def test_two_rounds_land_in_campaign_json_and_the_chain(runtime, two_rounds):
    name, rows = two_rounds
    doc = _doc(runtime)
    assert doc["status"] == "done" and doc["cursor"] == 2 and doc["best"] == 2
    assert doc["task"] == TASK and doc["seeds"] == [1, 2] and doc["arm"] == "auto"
    r1, r2 = doc["rounds"]
    assert r1["tried"]["kind"] == "executor" and r1["tried"]["node"] == "grab-0"
    assert r1["tried"]["detail"]["from"] == "scripted" and r1["tried"]["detail"]["to"] == "alt"
    assert (r1["before"], r1["after"], r1["published"], r1["best"]) == (0, 2, True, 2)
    assert r2["tried"]["kind"] == "none" and (r2["before"], r2["after"], r2["published"]) == (2, 2, False)
    assert all(len(r["suite_sha"]) == 64 for r in (r1, r2))
    # media: only verified segments were kept (both nodes, both seeds after the switch),
    # synthetic frames encoded under 1 MB, session-relative paths as rsi_frames returns them
    assert set(r1["media"]) == set(r2["media"]) == {
        f"media/{TASK}/{seed}/{node}.gif" for seed in (1, 2) for node in ("reach-0", "grab-0")}
    for rel in r1["media"]:
        f = runtime.session / rel
        assert f.is_file() and 0 < f.stat().st_size <= media.MAX_BYTES, rel
    for seed in (1, 2):
        idx = media.index_of(runtime.session / "media", TASK, seed)
        assert set(idx) == {"reach-0", "grab-0"} and all(v["frames"] > 0 for v in idx.values())
    # published = the record with its measured by_executor row, through the skills-root door
    rec = json.loads((runtime.session / "skills" / f"{r1['tried']['detail']['digest']}.json").read_text())
    assert rec["name"] == "grab" and rec["evidence"][EMB]["by_executor"]["alt"] == {"n": 6, "k": 6}
    steps = _kinds(rows, "rsi_step")
    assert [(s["round"], s["before"], s["after"], s["published"]) for s in steps] == \
        [(1, 0, 2, True), (2, 2, 2, False)]
    # per-seed detail of the kept suite rides both the round and its rsi_step row
    n_steps = r1["per_seed"][0]["nodes"][0]["steps"]
    assert isinstance(n_steps, int) and n_steps > 0
    kept = [{"seed": s, "success": True, "first_death": None, "failure_mode": None,
             "tunables_sha": None, "elapsed_s": pytest.approx(1, abs=30),
             "nodes": [{"id": n, "ok": True, "steps": n_steps, "failure_mode": None}
                       for n in ("reach-0", "grab-0")]} for s in (1, 2)]
    assert [s["per_seed"] for s in steps] == [r1["per_seed"], r2["per_seed"]] == [kept, kept]
    # the baseline (0/2) rows carry the trail with the first-death node ok=False
    base_rows = [p for l in _LIVE for p in l["per_seed_partial"] if l["phase"] == "baseline"]
    assert base_rows and all(r["first_death"] == "grab-0" and r["elapsed_s"] > 0 and
                             [n["ok"] for n in r["nodes"]] == [True, False] for r in base_rows), base_rows
    assert (r1["needs"], r2["needs"]) == ([], []) and steps[1]["needs"] == []   # all seeds pass
    assert all(s["brief"] == name and s["task"] == TASK and s["suite_sha"] for s in steps)
    assert not _kinds(rows, "runtime.task_error")


def test_three_faces_agree_on_the_real_campaign(runtime, two_rounds, capsys):
    """rsi_run / rsi_series / rsi_frames byte-equal across library, CLI and MCP on
    the campaign.json the real run wrote (not a fixture)."""
    sd = runtime.session
    ms.configure(runtime.runs)
    base = ["--runs", str(runtime.runs), "--session", SESSION]
    cases = [
        (["rsi_run", TASK], bs.rsi_run(sd, TASK), ms.rsi_run(TASK)),
        (["rsi_series", TASK], bs.rsi_series(sd, TASK), ms.rsi_series(TASK)),
        (["rsi_frames", TASK, "--round", "1"], bs.rsi_frames(sd, TASK, 1), ms.rsi_frames(TASK, 1)),
    ]
    for argv, lib, mcp in cases:
        code = storecli.main(argv + base)
        out = capsys.readouterr().out.rstrip("\n")
        assert code == 0 and out == json.dumps(lib) == json.dumps(mcp), argv
    doc = _doc(runtime)
    assert bs.rsi_run(sd, TASK) == {**doc, "latest": doc["rounds"][-1], "open_brief": None}
    assert [s["after"] for s in bs.rsi_series(sd, TASK)] == [2, 2]
    assert [(s["per_seed"], s["needs"]) for s in bs.rsi_series(sd, TASK)] == \
        [(r["per_seed"], r["needs"]) for r in doc["rounds"]]
    assert bs.rsi_frames(sd, TASK, 1) == doc["rounds"][0]["media"]


def test_cancel_lands_and_resubmit_resumes_from_cursor(runtime, two_rounds):
    before = len(bs.chain_rows(runtime.session))
    name = bs.submit_brief(runtime.runs, json.dumps(
        {"kind": "evolve", "task": TASK, "seeds": [1, 2], "rounds": 4}), session=SESSION)["submitted"]
    _wait(lambda: (runtime.session / "processing" / name).exists(), 30, "claim")
    assert bs.cancel_brief(runtime.session, name)["requested"] is True
    _wait(lambda: (runtime.session / "cancelled" / name).exists(), 60, "cancelled filing")
    rows = bs.chain_rows(runtime.session)[before:]
    assert _kinds(rows, "runtime.task_cancelled")[0]["brief"] == name
    doc = _doc(runtime)
    assert doc["status"] == "cancelled" and doc["cursor"] == 2 and len(doc["rounds"]) == 2
    # resume: same task again -> continues at round 3
    _, rows = runtime.run({"kind": "evolve", "task": TASK, "rounds": 3})
    doc = _doc(runtime)
    assert doc["status"] == "done" and doc["cursor"] == 3
    assert [r["round"] for r in doc["rounds"]] == [1, 2, 3]
    assert doc["rounds"][2]["tried"]["kind"] == "none" and doc["rounds"][2]["best"] == 2
    assert [s["round"] for s in _kinds(rows, "rsi_step")] == [3]


def test_evolve_is_refused_outside_evolution_mode(tmp_path_factory):
    rt = _Runtime(tmp_path_factory.mktemp("runs"), card=_CARD)
    try:
        _, rows = rt.run({"kind": "evolve", "task": TASK, "rounds": 1}, expect="failed")
        assert "evolution mode" in _kinds(rows, "runtime.task_error")[0]["error"]
    finally:
        rt.stop()


def test_proposer_tries_an_unproven_executor_once_then_says_what_it_needs(monkeypatch):
    """No evidence favours ``alt`` -> still one honest switch; once tried on that
    node the round is ``none`` with ``needs`` naming what would unblock it."""
    from scripts import evolve
    recs = {k: protocol.SkillRecordV0.from_dict({**v, "evidence": {}}) for k, v in RECORDS.items()}
    binding = {"policy": "test_evolve_e2e:policy_provider"}
    dead = {"success": False, "first_death": "grab-0", "failure_mode": "reach_stall",
            "nodes": {"reach-0": {"skill": "reach", "success": True, "executor": "scripted"},
                      "grab-0": {"skill": "grab", "success": False, "executor": "scripted"}}}
    before = {"count": 0, "seeds": {"1": dead, "2": dead}, "sha": "x"}
    assert evolve.per_seed(before) == [
        {"seed": s, "success": False, "first_death": "grab-0", "failure_mode": "reach_stall",
         "tunables_sha": None, "elapsed_s": None, "nodes": []} for s in (1, 2)]
    first = evolve.propose(before, recs, EMB, "auto", binding, 1, {"executors": {}, "tunables": {}}, [])
    assert (first["kind"], first["node"], first["detail"]["to"]) == ("executor", "grab-0", "alt")
    again = evolve.propose(before, recs, EMB, "auto", binding, 2, {"executors": {}, "tunables": {}},
                           [{"round": 1, "tried": first}])
    assert again["kind"] == "none" and again["detail"]["needs"] == [
        "tunables on test_evolve_e2e:policy_provider", "evidence for another executor", "proposal"]
    # a tunable a +-30% step leaves where it is (segment_cap 0 -> 0) is no trial:
    # the next knob is proposed; when none moves, the same honest none
    hist = [{"round": 1, "tried": first}]
    monkeypatch.setattr(evolve, "mount_params",
                        lambda ref: {"tunables": {"segment_cap": 0, "stall_k": 40}})
    knob = evolve.propose(before, recs, EMB, "auto", binding, 2, {"executors": {}, "tunables": {}}, hist)
    assert (knob["kind"], knob["detail"]["path"], knob["detail"]["to"], knob["detail"]["hint"]) == (
        "tunables", ["tunables", "stall_k"], 28, None)
    monkeypatch.setattr(evolve, "mount_params", lambda ref: {"tunables": {"segment_cap": 0}})
    none = evolve.propose(before, recs, EMB, "auto", binding, 2, {"executors": {}, "tunables": {}}, hist)
    assert none["kind"] == "none" and none["detail"]["needs"] == again["detail"]["needs"]


def test_proposer_follows_the_card_hints_for_the_failure_mode_in_both_directions(monkeypatch):
    """A reach_stall first death: the card's [tunable_hints] order wins over name
    order, each knob goes -30% then +30% (a (knob, direction) in history is not
    retried), then the next hinted knob; ``detail.hint`` names the mode."""
    from scripts import evolve
    recs = {k: protocol.SkillRecordV0.from_dict({**v, "evidence": {}}) for k, v in RECORDS.items()}
    binding = {"policy": "test_evolve_e2e:policy_provider"}
    dead = {"success": False, "first_death": "grab-0", "failure_mode": "reach_stall",
            "nodes": {"grab-0": {"skill": "grab", "success": False, "executor": "scripted"}}}
    before = {"count": 0, "seeds": {"1": dead}, "sha": "x"}
    monkeypatch.setattr(evolve, "mount_params", lambda ref: {
        "tunables": {"drop_edge_margin": 0.10, "drop_over_dz": 0.12, "reach_tol": 0.03, "stall_k": 40},
        "tunable_hints": {"reach_stall": ["drop_edge_margin", "drop_over_dz", "reach_tol"]}})
    hist = [{"round": 1, "tried": {"kind": "executor", "node": "grab-0", "detail": {"to": "alt"}}}]
    steps = []
    for r in (2, 3, 4, 5):
        t = evolve.propose(before, recs, EMB, "auto", binding, r, {"executors": {}, "tunables": {}}, hist)
        assert t["kind"] == "tunables" and t["node"] == "grab-0"
        steps.append((t["detail"]["path"][-1], t["detail"]["from"], round(t["detail"]["to"], 3), t["detail"]["hint"]))
        hist.append({"round": r, "tried": t})
    assert steps == [("drop_edge_margin", 0.10, 0.07, "reach_stall"),
                     ("drop_edge_margin", 0.10, 0.13, "reach_stall"),
                     ("drop_over_dz", 0.12, 0.084, "reach_stall"),
                     ("drop_over_dz", 0.12, 0.156, "reach_stall")]
    # no hint for the mode: name order, unhinted
    monkeypatch.setattr(evolve, "mount_params", lambda ref: {"tunables": {"stall_k": 40, "hover_dz": 0.08}})
    t = evolve.propose(before, recs, EMB, "auto", binding, 6, {"executors": {}, "tunables": {}}, hist[:1])
    assert (t["detail"]["path"][-1], round(t["detail"]["to"], 3), t["detail"]["hint"]) == ("hover_dz", 0.056, None)
