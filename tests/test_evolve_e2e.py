"""The lightweight evolve loop end to end: an ``evolve`` brief through the REAL
scripts/harness_runtime.py (evolution mode) spawning the REAL scripts/evolve.py.
No simulator: a tmp task card whose independent world oracle observes state
changed by actions. The alternate executor grasps successfully; paired
development accepts 0/2 -> 2/2 without installing an unverified skill.
Then cancel mid-run and resubmit: the loop resumes from cursor (round 3).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from test_mission_e2e import SESSION, _kinds, _Runtime, _wait

from board import mcp_server as ms
from board import store as bs
from board import storecli
from harness import fakes, media, protocol
from harness.skill_executor import InprocExecutor
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
        self.achieved = set()
        return dict(_OBS)

    def step(self, action):
        assert len(action) == 1, f"expected a 1-dim action, got {action!r}"   # the shape gate a real env has
        self.t += 1
        if action[0] == 1.0:
            self.achieved.add("reach")
        elif action[0] == 2.0 and "reach" in self.achieved:
            self.achieved.add("grab")
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
        return {"reach", "grab"} <= env.achieved


class _Driver:
    """Scripted episode driver: grab succeeds only under a non-scripted executor, whose
    ``act`` then drives (the kitchen seam). Each segment drives STEPS env steps (so the
    media recorder sees frames)."""
    STEPS = 8
    n = 0
    _ex = None

    @property
    def exhausted(self):
        return self.n >= self.STEPS

    def observe_once(self, obs):
        pass

    def on_handback(self):
        pass

    def act(self, obs):
        self.n += 1
        return self._ex.act(obs) if self._ex is not None else (1.0 if self.task == "reach" else 0.0,)

    def enter_segment(self, env, spec, executor=None):
        self.n, self._ex = 0, executor
        self.task = spec.task

    def segment_success(self, env):
        return self.task in env.achieved


class _Policy:
    def make_driver(self, spec):
        return _Driver()


class _AltExecutor(InprocExecutor):
    def act(self, obs):
        return (2.0,)


class _Alt:
    def make_driver(self, spec):
        return _AltExecutor()


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


def policy_provider(**params):
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
LLM_ALT = {"kind": "executor", "payload": {"node": "grab-0", "to": "alt"},
           "summary": "Try the other installed grab executor.", "rationale": "Compare paired world outcomes."}
LLM_NONE = {"kind": "none", "payload": {}, "summary": "No further supported intervention.",
            "rationale": "The independent task objective is already satisfied."}
LLM_READ_GRAB = {"op": "inspect", "args": {"view": "source", "node": "grab-0",
                 "module": "test_evolve_e2e", "symbol": "_Driver"}}


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    rt = _Runtime(tmp_path_factory.mktemp("runs"), card=_CARD,
                  canned=[LLM_READ_GRAB, LLM_ALT, LLM_NONE], mode="evolution")
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
    assert all(l["seeds_total"] in (1, 2, 4) and l["message"] and l["started_at"] for l in _LIVE)
    active_probes = [l for l in _LIVE if l.get("experiment_scope") == "probe" and l["seed_index"] is not None]
    assert active_probes and all(l["seeds_total"] == 1 for l in active_probes)
    assert all(l["seeds_total"] == 2 for l in base if l["seed_index"] is not None)
    assert _LIVE[-1]["seeds_total"] == 4
    assert any(l["phase"] == "retest" and l["tried"]["kind"] == "executor" for l in _LIVE)
    assert "种子 1 运行中" in next(l["message"] for l in base if l["seed"] == 1)
    # the node trail of seed 1 (a plain list in plan order): ok flips None -> True in
    # plan order as verify rows land, and the rolling message log accumulates
    trails = [tuple(n["ok"] for n in l["nodes"]) for l in _LIVE if l["nodes"]]
    assert trails[0] == (None, None) and any(t[0] is True for t in trails), trails
    assert all(t in ((None, None), (True, None), (True, True), (True, False)) for t in trails), trails
    assert all(l["nodes"][0]["id"] == "reach-0" and l["nodes"][1]["skill"] == "grab" for l in _LIVE if l["nodes"])
    # every trail node carries the plan graph's edges and kind: grab-0 comes after reach-0
    assert all([(n["after"], n["kind"]) for n in l["nodes"]] == [([], "segment"), (["reach-0"], "segment")]
               for l in _LIVE if l["nodes"])
    assert all(l["seed_started_at"] for l in base if l["seed"] is not None)
    assert [m["text"] for m in _LIVE[-1]["messages"]][-1] == "已完成 2 轮"
    assert 1 <= len(_LIVE[-1]["messages"]) <= 20 and all(m["ts"] for m in _LIVE[-1]["messages"])
    live = _doc(runtime)["live"]
    assert live["phase"] == "done" and live["round"] == 2 and live["last_round_s"] is not None
    assert live["message"] == "已完成 2 轮" and bs.rsi_run(runtime.session, TASK)["live"] == live


def test_two_rounds_land_in_campaign_json_and_the_chain(runtime, two_rounds):
    name, rows = two_rounds
    doc = _doc(runtime)
    assert doc["status"] == "done" and doc["cursor"] == 2 and doc["best"] == 4
    assert doc["task"] == TASK and doc["seeds"] == [1, 4] and doc["arm"] == "auto"
    r1, r2 = doc["rounds"]
    assert [r["proposer"] for r in (r1, r2)] == ["llm", "llm"]
    assert [r["llm"]["status"] for r in (r1, r2)] == ["proposed", "abstained"]
    assert r1["tried"]["kind"] == "executor" and r1["tried"]["node"] == "grab-0"
    assert r1["tried"]["detail"]["from"] == "scripted" and r1["tried"]["detail"]["to"] == "alt"
    assert (r1["before"], r1["after"], r1["published"], r1["accepted"], r1["best"]) == (0, 2, False, True, 2)
    assert r2["tried"]["kind"] == "none" and (r2["before"], r2["after"], r2["published"]) == (4, None, False)
    assert len(r1["suite_sha"]) == 64 and r2["suite_sha"] is None
    assert r2["trial"] is None and r2["after_seeds"] == [] and r2["after_score"] is None
    assert r2["experiments"]["after"] is None and r2["evaluation"]["after"] is None
    # media: only verified segments were kept (both nodes, both seeds after the switch),
    # synthetic frames encoded under 1 MB, session-relative paths as rsi_frames returns them
    assert {p.rsplit(f"/{TASK}/", 1)[-1] for p in r1["media"] if "/retest/" in p} == {
        f"{seed}/{node}.gif" for seed in (1, 2) for node in ("reach-0", "grab-0")}
    assert {p.rsplit(f"/{TASK}/", 1)[-1] for p in r2["media"]} == {
        f"{seed}/{node}.gif" for seed in (1, 2, 3, 4) for node in ("reach-0", "grab-0")}
    assert set(r1["media"]).isdisjoint(r2["media"])  # later phases cannot overwrite earlier evidence
    for rel in r1["media"]:
        f = runtime.session / rel
        assert f.is_file() and 0 < f.stat().st_size <= media.MAX_BYTES, rel
    for seed in (1, 2):
        root = runtime.session / "media" / "rsi" / TASK / "epoch-1" / "round-1" / "retest"
        idx = media.index_of(root, TASK, seed)
        assert set(idx) == {"reach-0", "grab-0"} and all(v["frames"] > 0 for v in idx.values())
    # Development evidence cannot install a skill without the full GOAL battery.
    assert not list((runtime.session / "skills").glob("*.json"))
    assert r1["evaluation"]["installation"]["status"] == "not_evaluated"
    steps = _kinds(rows, "rsi_step")
    assert [(s["round"], s["before"], s["after"], s["published"]) for s in steps] == \
        [(1, 0, 2, False), (2, 4, None, False)]
    # per-seed detail of the kept suite rides both the round and its rsi_step row
    n_steps = r1["per_seed"][0]["nodes"][0]["steps"]
    assert isinstance(n_steps, int) and n_steps > 0
    kept = [{"seed": s, "success": True, "first_death": None, "failure_mode": None,
             "tunables_sha": None, "elapsed_s": pytest.approx(1, abs=30),
             "verification_observations": [],
             "terminal_observation": {"authority": "embodiment.terminal_success",
                                      "source": EMB, "mounted_ref": EMB, "success": True},
             # no per-node failure_mode: the fake stage seals none, and a node nobody
             # measured carries NO key (a None there reads as "no stall" -- see
             # D.merge_executor_diagnostics). The seed-level one above is a plain default.
             "nodes": [{"id": n, "ok": True, "steps": n_steps,
                        "after": after, "kind": "segment", "task": n.split("-")[0]}
                       for n, after in (("reach-0", []), ("grab-0", ["reach-0"]))]} for s in (1, 2)]
    assert [s["per_seed"] for s in steps] == [r1["per_seed"], r2["per_seed"]]
    assert [{k: v for k, v in row.items() if k != "evaluation"} for row in r1["after_seeds"]] == kept
    assert all(row["evaluation"]["complete"] for row in r1["after_seeds"])
    assert all(not row["success"] for row in r1["per_seed"])
    assert all(row["success"] for row in r2["per_seed"])
    # the baseline (0/2) rows carry the trail with the first-death node ok=False
    base_rows = [p for l in _LIVE for p in l["per_seed_partial"] if l["phase"] == "baseline"]
    assert base_rows and any(r["first_death"] == "grab-0" and r["elapsed_s"] > 0 and
                             [n["ok"] for n in r["nodes"]] == [True, False] for r in base_rows), base_rows
    assert r1["needs"] == [] and r2["needs"] == steps[1]["needs"]
    assert all(s["brief"] == name and s["task"] == TASK for s in steps)
    assert not _kinds(rows, "runtime.task_error")


def test_three_faces_agree_on_the_real_campaign(runtime, two_rounds, capsys):
    """rsi_run / rsi_series / rsi_frames byte-equal across library, CLI and MCP on
    the campaign.json the real run wrote (not a fixture) -- rsi_run with and
    without the --round argument."""
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
    run = bs.rsi_run(sd, TASK)
    # bounded by construction: the header, compact rows, and NO per-seed trail
    assert run == {**doc, "rounds": series[-bs.RUN_TAIL:], "latest": series[-1], "open_brief": None}
    assert all("per_seed" not in row and "after_seeds" not in row for row in series)
    # ... and --round hands back that ONE round in full, trails and all
    one = bs.rsi_run(sd, TASK, 1)["rounds"]
    assert one == [doc["rounds"][0]] and one[0]["per_seed"]
    assert bs.rsi_run(sd, TASK, 999)["rounds"] == []
    assert [s["after"] for s in series] == [2, None]
    # sub-task rates from the trails: the kept suite (= the published trial) verified every
    # node of every seed, so node_rate and both stage groups (reach, grab) read 1.0
    assert [(s["node_rate"], s["by_task"]) for s in bs.rsi_series(sd, TASK)] == [
        ({"before": 0.5, "after": 1.0, "best": 1.0},
         {"grab": {"before": 0.0, "after": 1.0}, "reach": {"before": 1.0, "after": 1.0}}),
        ({"before": 1.0, "after": None, "best": 1.0},
         {"grab": {"before": 1.0, "after": None}, "reach": {"before": 1.0, "after": None}})]
    assert bs.rsi_campaigns(runtime.runs / SESSION)[0]["node_rate_best"] == 1.0
    assert bs.rsi_frames(sd, TASK, 1) == {"media": doc["rounds"][0]["media"], "dropped": doc["rounds"][0]["media_dropped"]}


def test_cancel_lands_and_resubmit_resumes_from_cursor(runtime, two_rounds):
    # Each resumed evolve subprocess starts a new fake cursor; its model now abstains.
    (runtime.runs / "canned.json").write_text(json.dumps(LLM_NONE))
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
    _, rows = runtime.run({"kind": "evolve", "task": TASK, "rounds": 1})
    doc = _doc(runtime)
    assert doc["status"] == "done" and doc["cursor"] == 3
    assert [r["round"] for r in doc["rounds"]] == [1, 2, 3]
    assert doc["rounds"][2]["tried"]["kind"] == "none" and doc["rounds"][2]["best"] == 4
    # 开始/继续 on a finished campaign sends a task-only brief (rounds = default):
    # that means "N more rounds", never an empty loop that finishes in a blink.
    runtime.run({"kind": "evolve", "task": TASK, "rounds": 1})
    doc = json.loads((runtime.runs / SESSION / "campaigns" / f"evolve-{TASK}" / "campaign.json").read_text())
    assert doc["status"] == "done" and doc["cursor"] == 4 and [r["round"] for r in doc["rounds"]] == [1, 2, 3, 4]
    assert [s["round"] for s in _kinds(rows, "rsi_step")] == [3]


def test_continuous_brief_renews_cycle_budget_and_cancels_without_delay(tmp_path):
    """Real inbox, worker and subprocess; synthetic world and model only."""
    rt = _Runtime(tmp_path, card=_CARD, canned=LLM_NONE, mode="evolution")
    rt.campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    try:
        name = bs.submit_brief(rt.runs, json.dumps({
            "kind": "evolve", "task": TASK, "seeds": [1, 2], "rounds": 0,
            "continuous": True, "max_model_calls": 1, "confirm_seeds": 0,
        }), session=SESSION)["submitted"]
        _wait(lambda: rt.campaign.exists() and _doc(rt).get("cursor", 0) >= 2,
              60, "second completed learning cycle despite model abstention")
        doc = _doc(rt)
        assert doc["status"] == "running" and doc["continuous"] is True
        assert doc["stop_reason"] is None
        first, second = doc["rounds"][:2]
        assert first["llm"]["stop_reason"] == second["llm"]["stop_reason"] == "model_stop"
        assert first["after"] is second["after"] is None
        assert first["usage"]["episode_attempts"] == 2
        assert second["usage"]["episode_attempts"] == 0  # unchanged measured baseline is reusable
        assert second["cycle_budget"]["used"]["model_calls"] == 1
        assert second["run_budget"]["used"]["model_calls"] == 2
        assert second["run_budget"]["limits"]["model_calls"] is None
        public = bs.rsi_campaigns(rt.session)[0]
        assert public["continuous"] is True and public["stop_reason"] is None
        assert public["run_budget"]["used"]["model_calls"] >= 2
        assert bs.rsi_series(rt.session, TASK)[1]["cycle_outcome"] == second["cycle_outcome"]
        assert bs.cancel_brief(rt.session, name)["requested"] is True
        _wait(lambda: (rt.session / "cancelled" / name).exists(), 20, "cancel continuous execution")
        events = bs.chain_rows(rt.session)
        sealed = [s for s in _kinds(events, "rsi_step") if s["brief"] == name]
        assert [s["round"] for s in sealed][:2] == [1, 2]
        assert sealed[1]["cycle_budget"] == second["cycle_budget"]
        assert sealed[1]["cycle_outcome"] == second["cycle_outcome"]
        assert any(s["brief"] == name for s in _kinds(events, "runtime.task_cancelled"))
        assert not _kinds(events, "runtime.task_error")
    finally:
        rt.stop()


def test_continuous_requires_explicit_boolean_before_starting_worker(tmp_path):
    rt = _Runtime(tmp_path, card=_CARD, canned=LLM_NONE, mode="evolution")
    try:
        for value in ("false", 1, None):
            _, rows = rt.run({"kind": "evolve", "task": TASK, "continuous": value}, expect="failed")
            assert "continuous must be a boolean" in _kinds(rows, "runtime.task_error")[-1]["error"]
            assert not _kinds(rows, "rsi_step")
        assert not (rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json").exists()
    finally:
        rt.stop()


def test_evolve_is_refused_outside_evolution_mode(tmp_path_factory):
    rt = _Runtime(tmp_path_factory.mktemp("runs"), card=_CARD)
    try:
        _, rows = rt.run({"kind": "evolve", "task": TASK, "rounds": 1}, expect="failed")
        assert "evolution mode" in _kinds(rows, "runtime.task_error")[0]["error"]
    finally:
        rt.stop()



def test_unmeasured_seed_summary_has_no_invented_oracle_evidence():
    from scripts import evolve
    dead = {"success": False, "first_death": "grab-0", "failure_mode": "reach_stall",
            "nodes": {"grab-0": {"skill": "grab", "success": False, "executor": "scripted"}}}
    before = {"count": 0, "seeds": {"1": dead, "2": dead}, "sha": "x"}
    assert evolve.per_seed(before) == [
        {"seed": seed, "success": False, "first_death": "grab-0", "failure_mode": "reach_stall",
         "tunables_sha": None, "elapsed_s": None, "nodes": [], "evaluation": None,
         "verification_observations": [], "terminal_observation": None} for seed in (1, 2)]
