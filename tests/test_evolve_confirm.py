"""ENPIRE/ASPIRE additions to the lightweight evolve loop, in-process on a stdlib fake:
failure keyframes kept on drop (media + rsi_frames + the LLM brief), the hypothesis
tree fields (parent / outcome), additional paired development seeds, usage
(tokens / sim seconds), and a proposal's numeric tunables ``from``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_evolve_e2e import ALT, CATALOGUE, ORACLES, RECORDS, _Env, _Handle, _Planner  # noqa: F401

from board import store as bs
from harness import media
from harness.fakes import _FakeEnvHandle
from scripts import evolve

EMB = "test_evolve_confirm:env_provider"
TASK = "e2e_confirm"
RECORDS = {**RECORDS, "grab": {**RECORDS["grab"], "bindings": {EMB: RECORDS["grab"]["bindings"][
    "test_evolve_e2e:env_provider"]}, "evidence": {EMB: RECORDS["grab"]["evidence"]["test_evolve_e2e:env_provider"]}},
           "reach": {**RECORDS["reach"], "bindings": {EMB: {"task": "reach"}}}}
from harness import protocol
from harness.skill_library import segment_specs

SEGMENT_SPECS = segment_specs({k: protocol.SkillRecordV0.from_dict(v) for k, v in RECORDS.items()}, EMB)
EPISODE = {"task": "reach", "horizon": 60}   # room for both segments' 16 steps -> 4 frames each


class _SeedHandle(_FakeEnvHandle):
    def __init__(self, seed):
        self.seed = seed

    def reset(self):
        super().reset()
        self.achieved = set()
        return dict(_Handle.reset.__globals__["_OBS"])

    def step(self, action):
        self.t += 1
        if action[0] == 1.0:
            self.achieved.add("reach")
        elif action[0] == 2.0 and "reach" in self.achieved:
            self.achieved.add("grab")
        return dict(_Handle.reset.__globals__["_OBS"]), 0.0, False, {}


class _ConfEnv(_Env):
    def make_env(self, spec):
        return _SeedHandle(spec.seed)


class _Driver:
    """grab: scripted passes only on seeds >= 3 (the confirm range); ``alt`` passes on
    the debug seeds always and on the confirm seeds per PH_TEST_CONFIRM_MODE
    (hold: passes, regress: fails, never: fails everywhere)."""
    STEPS = 16
    n = 0
    command = 0.0
    last_progress_step = 9   # -> keyframe 1 is frame 1 of 4, not the middle

    @property
    def exhausted(self):
        return self.n >= self.STEPS

    def observe_once(self, obs):
        pass

    def on_handback(self):
        pass

    def act(self, obs):
        self.n += 1
        return (self.command,)

    def enter_segment(self, env, spec, executor=None):
        self.n = 0
        self.task = spec.task
        mode = os.environ.get("PH_TEST_CONFIRM_MODE", "hold")
        if spec.task != "grab":
            self.command = 1.0
        elif executor is None:
            self.command = 2.0 if env.seed >= 3 else 0.0
        else:
            self.command = 2.0 if mode != "never" and (env.seed < 3 or mode == "hold") else 0.0

    def segment_success(self, env):
        return self.task in env.achieved


class _Policy:
    def make_driver(self, spec):
        return _Driver()


def env_provider():
    return _ConfEnv()


def policy_provider():
    return _Policy()


_CARD = f"""
[task_bindings.{TASK}]
env = "{EMB}"
policy = "test_evolve_confirm:policy_provider"
planner = "test_evolve_e2e:planner_provider"
catalogue = "test_evolve_e2e:CATALOGUE"
records = "test_evolve_confirm:RECORDS"
oracles = "test_evolve_e2e:ORACLES"
episodic = true
episode = "test_evolve_confirm:EPISODE"
segment_specs = "test_evolve_confirm:SEGMENT_SPECS"
max_replans = 1
"""


def _campaign(tmp_path, monkeypatch, mode: str, canned=None, confirm=2, seeds=("1", "2"),
              pre=None) -> tuple[Path, dict]:
    root = tmp_path / mode
    (root / "plugins" / "conf").mkdir(parents=True)
    (root / "plugins" / "conf" / "manifest.toml").write_text(_CARD)
    monkeypatch.setenv("PH_PLUGINS_EXTRA", str(root / "plugins"))
    monkeypatch.setattr(evolve, "_BASE_EXTRA", str(root / "plugins"))
    monkeypatch.setenv("PH_TEST_CONFIRM_MODE", mode)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    argv = ["--mode", "evolution", "--task", TASK, "--session", str(root / "s"),
            "--skills-root", str(root / "s" / "skills"), "--seeds", *seeds, "--rounds", "1",
            "--confirm-seeds", str(confirm)]
    if pre is not None:   # an earlier campaign on disk (the baseline the origin cluster comes from)
        (root / "s" / "campaigns" / f"evolve-{TASK}").mkdir(parents=True)
        (root / "s" / "campaigns" / f"evolve-{TASK}" / "campaign.json").write_text(json.dumps(pre))
    if canned is None:
        from test_evolve_e2e import LLM_ALT
        canned = [LLM_ALT]
    canned = [{"op": "inspect", "args": {"view": "source", "node": "grab-0",
                "module": "test_evolve_confirm", "symbol": "_Driver"}}, *canned]
    (root / "canned.json").write_text(json.dumps(canned))
    monkeypatch.setenv("PH_MODEL_ENDPOINT_FAKE", str(root / "canned.json"))
    assert evolve.main(argv) == 0
    return root / "s", json.loads((root / "s" / "campaigns" / f"evolve-{TASK}" / "campaign.json").read_text())


def test_win_that_holds_on_additional_dev_seeds_is_accepted_without_installation(tmp_path, monkeypatch):
    canned = [{"kind": "executor", "payload": {"node": "grab-0", "to": "alt"}, "summary": "grab 死", "rationale": "换 alt"}]
    session, doc = _campaign(tmp_path, monkeypatch, "hold", canned=canned)
    r = doc["rounds"][0]
    assert (r["before"], r["after"], r["published"], r["accepted"], r["outcome"], r["parent"]) == (0, 2, False, True, "improved", 0)
    confirm = r['confirm']
    assert {k: confirm[k] for k in ('seeds', 'before', 'after')} == {'seeds': [3, 4], 'before': 2, 'after': 2}
    assert confirm['evaluation']['objective_id'] == r['evaluation']['objective_id']
    assert confirm['experiments']['before'] != confirm['experiments']['after']
    for side in ('before', 'after'):
        assert [s['seed'] for s in confirm['evaluation'][side]] == [3, 4]
        assert all(s['evaluation']['terminal'] is True for s in confirm['evaluation'][side])
    # The original paired suite is scored before the additional development seeds.
    assert r["regression"] == {"lost": []}
    assert r["burned"] == [] and r["layer"] is None
    assert doc["confirm_base"] is None  # every comparison reruns the current predecessor
    assert doc["seeds"] == [1, 4] and not list((session / "skills").glob("*.json"))
    assert r["usage"]["sim_s"] > 0 and r["usage"]["llm_tokens"] is None
    # This campaign has one round; its saved usage is rounded to milliseconds.
    assert doc["live"]["sim_s"] == pytest.approx(r["usage"]["sim_s"], abs=0.0005)
    assert any("新种子确认" in m["text"] for m in doc["live"]["messages"])
    # Media remains available to station without expanding every model request.
    audit = json.loads((session / "campaigns" / f"evolve-{TASK}" / "llm" / "round-1.json").read_text())
    text = audit["messages"][1]["content"]
    assert "keyframes" not in text
    assert all(any(str(path).endswith(f"baseline/{TASK}/{seed}/grab-0.fail-{i}.jpg") for path in session.rglob("*.jpg"))
               for seed in (1, 2) for i in range(3))
    # confirm is round detail, not a series row: it rides the ONE round rsi_run(round=1) serves
    assert bs.rsi_run(session, TASK, 1)["rounds"][0]["confirm"] == r["confirm"]
    c = bs.rsi_campaigns(session)[0]
    assert c["published_rounds"] == [] and c["accepted_rounds"] == [1]
    assert c["usage"] == {"llm_tokens": None, "sim_s": r["usage"]["sim_s"]}


def test_win_that_regresses_on_confirm_seeds_is_not_published(tmp_path, monkeypatch):
    session, doc = _campaign(tmp_path, monkeypatch, "regress")
    r = doc["rounds"][0]
    assert (r["before"], r["after"], r["published"], r["outcome"]) == (0, 2, False, "improved")
    assert {k: r['confirm'][k] for k in ('seeds', 'before', 'after')} == {'seeds': [3, 4], 'before': 2, 'after': 0}
    assert all(s['evaluation']['terminal'] is True for s in r['confirm']['evaluation']['before'])
    assert all(s['evaluation']['terminal'] is False for s in r['confirm']['evaluation']['after'])
    assert r['accepted'] is False
    assert doc["applied"]["executors"] == {} and doc["best"] == 0
    assert bs.rsi_campaigns(session)[0]["published_rounds"] == []


def test_failure_keyframes_are_kept_on_drop_and_listed_by_rsi_frames(tmp_path, monkeypatch):
    session, doc = _campaign(tmp_path, monkeypatch, "never", confirm=0)
    r = doc["rounds"][0]
    assert (r["published"], r["outcome"], r["confirm"]) == (False, "same", None)
    frames = bs.rsi_frames(session, TASK, 1)
    for seed in (1, 2):
        d = frames["dropped"][f"after/{seed}/grab-0"]
        assert d["reason"] == "verify_failed"
        n = len(d["keyframes"])   # the last suite on these seeds ran ``alt`` (one governed frame)
        assert 1 <= n <= 3
        assert [Path(p).name for p in d["keyframes"]] == [f"grab-0.fail-{i}.jpg" for i in range(n)]
        assert all(f"/retest/{TASK}/{seed}/" in p for p in d["keyframes"])
        before = frames["dropped"][f"before/{seed}/grab-0"]["keyframes"]
        assert set(before).isdisjoint(d["keyframes"])
        for rel in d["keyframes"]:
            assert 0 < (session / rel).stat().st_size <= 25_000, rel
        root = session / "media" / "rsi" / TASK / "epoch-1" / "round-1" / "retest"
        assert media.dropped_of(root, TASK, seed)["grab-0"]["keyframes"] == \
            [f"grab-0.fail-{i}.jpg" for i in range(n)]
    assert frames["media"] == r["media"] and any(p.endswith(f"/{TASK}/1/reach-0.gif") for p in frames["media"])


def test_stall_keyframe_follows_the_drivers_last_progress_step(tmp_path):
    rec = media.SegmentRecorder(tmp_path, "t", 1, every=1)
    env, driver = _FakeEnvHandle(), _Driver()
    driver.last_progress_step = 3
    rec.start(env, driver)
    for _ in range(10):
        env.step(None)
        driver.act(None)
    assert rec.drop("n") == ["n.fail-0.jpg", "n.fail-1.jpg", "n.fail-2.jpg"]
    from PIL import Image
    imgs = [Image.open(tmp_path / "t" / "1" / f"n.fail-{i}.jpg") for i in range(3)]
    assert all(i.size == (128, 128) for i in imgs)
    # frames differ (the fake's stripe walks with the step): stall = frame index 2, not the middle (4)
    assert list(imgs[1].getdata()) != list(imgs[0].getdata()) != list(imgs[2].getdata())


class _ImageEndpoint:
    identity, images = "img", True

    def __init__(self):
        self.seen, self.last_usage = None, None

    def chat(self, messages, **opts):
        self.seen = messages
        self.last_usage = {"prompt": 12, "completion": 3}
        return json.dumps({"kind": "none", "payload": {}, "summary": "看过了", "rationale": "-"})


def test_proposal_tunables_from_is_the_knobs_current_value(monkeypatch):
    monkeypatch.setattr(evolve, "mount_params", lambda ref: {"tunables": {"stall_k": 40}})
    before = {"seeds": {"1": {"first_death": "g", "nodes": {"g": {"skill": "grab", "executor": "scripted"}}}}}
    p = {"id": "x", "kind": "tunables", "note": "", "payload": {"node": "g", "ref": "r", "path": ["tunables", "stall_k"], "to": 28}}
    assert evolve.from_proposal(p, before)["detail"]["from"] == 40
    p["payload"]["path"] = ["tunables", "nope"]
    assert evolve.from_proposal(p, before)["detail"]["from"] is None


def _baseline(rng, seeds, milestone="grab-0") -> dict:
    """A campaign (dev range ``rng``) whose round 1 baseline put every dev seed in one
    failure cluster -- the origin cluster a later round's fix must not regress."""
    return {"task": TASK, "session": "s", "seeds": list(rng), "arm": "auto", "best": 0, "cursor": 1,
            "status": "running", "applied": {"executors": {}, "tunables": {}},
            "rounds": [{"round": 1, "tried": {"kind": "none", "node": milestone, "detail": {}},
                        "before": 0, "after": 0, "best": 0, "published": False, "outcome": "none",
                        "per_seed": [{"seed": s, "success": False, "first_death": milestone,
                                      "nodes": [{"id": milestone, "ok": False}]} for s in seeds]}]}


def test_a_trial_that_loses_a_paired_world_success_is_not_accepted(tmp_path, monkeypatch):
    """Zetta's historical regression: seed 3 was in the grab-0 cluster and the accepted state
    already wins it; the trial wins 1 and 2 but breaks 3 -- a net gain that is still a
    regression, so the publish is blocked before the fresh-seed confirm ever runs."""
    _, doc = _campaign(tmp_path, monkeypatch, "regress", seeds=("1", "3"),
                       pre=_baseline([1, 3], [1, 2, 3]))
    r = doc["rounds"][-1]
    assert (r["round"], r["before"], r["after"], r["outcome"]) == (2, 1, 2, "worse")
    lost = r["regression"]["lost"]
    assert len(lost) == 1 and lost[0]["seed"] == "3"
    assert lost[0]["before"] is True and lost[0]["after"] is False
    assert r["published"] is False and r["confirm"] is None   # the cluster comes first
    assert doc["applied"]["executors"] == {} and doc["best"] == 1


def test_additional_seeds_join_development_without_claiming_a_heldout_burn(tmp_path, monkeypatch):
    """All extra comparisons are development; this path never claims held-out evidence."""
    _, doc = _campaign(tmp_path, monkeypatch, "regress")
    r = doc["rounds"][0]
    assert {k: r['confirm'][k] for k in ('seeds', 'before', 'after')} == {'seeds': [3, 4], 'before': 2, 'after': 0}
    assert r['published'] is False
    assert r["burned"] == [] and doc["seeds"] == [1, 4]
    assert doc["live"]["seeds_total"] == 4
