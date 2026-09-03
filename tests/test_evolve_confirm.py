"""ENPIRE/ASPIRE additions to the lightweight evolve loop, in-process on a stdlib fake:
failure keyframes kept on drop (media + rsi_frames + the LLM brief), the hypothesis
tree fields (parent / outcome), confirm-before-publish on fresh scratch seeds, usage
(tokens / sim seconds), and a proposal's numeric tunables ``from``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_evolve_e2e import _Env, _Handle, _Planner, ALT, CATALOGUE, EPISODE, ORACLES, SEGMENT_SPECS, RECORDS  # noqa: F401

from board import store as bs
from harness import media
from harness.fakes import _FakeEnvHandle
from scripts import evolve, evolve_llm

EMB = "test_evolve_confirm:env_provider"
TASK = "e2e_confirm"
RECORDS = {**RECORDS, "grab": {**RECORDS["grab"], "bindings": {EMB: RECORDS["grab"]["bindings"][
    "test_evolve_e2e:env_provider"]}, "evidence": {EMB: RECORDS["grab"]["evidence"]["test_evolve_e2e:env_provider"]}},
           "reach": {**RECORDS["reach"], "bindings": {EMB: {"task": "reach"}}}}
from harness import protocol  # noqa: E402
from harness.skill_library import segment_specs  # noqa: E402
SEGMENT_SPECS = segment_specs({k: protocol.SkillRecordV0.from_dict(v) for k, v in RECORDS.items()}, EMB)
EPISODE = {"task": "reach", "horizon": 60}   # room for both segments' 16 steps -> 4 frames each


class _SeedHandle(_FakeEnvHandle):
    def __init__(self, seed):
        self.seed = seed

    def reset(self):
        super().reset()
        return dict(_Handle.reset.__globals__["_OBS"])

    def step(self, action):
        self.t += 1
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
        return (0.0,)

    def enter_segment(self, env, spec, executor=None):
        self.n = 0
        mode = os.environ.get("PH_TEST_CONFIRM_MODE", "hold")
        if spec.task != "grab":
            self.ok = True
        elif executor is None:
            self.ok = env.seed >= 3
        else:
            self.ok = mode != "never" and (env.seed < 3 or mode == "hold")

    def segment_success(self, env):
        return self.ok


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


def _campaign(tmp_path, monkeypatch, mode: str, canned=None, confirm=2) -> tuple[Path, dict]:
    root = tmp_path / mode
    (root / "plugins" / "conf").mkdir(parents=True)
    (root / "plugins" / "conf" / "manifest.toml").write_text(_CARD)
    monkeypatch.setenv("PH_PLUGINS_EXTRA", str(root / "plugins"))
    monkeypatch.setattr(evolve, "_BASE_EXTRA", str(root / "plugins"))
    monkeypatch.setenv("PH_TEST_CONFIRM_MODE", mode)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    argv = ["--mode", "evolution", "--task", TASK, "--session", str(root / "s"),
            "--skills-root", str(root / "s" / "skills"), "--seeds", "1", "2", "--rounds", "1",
            "--confirm-seeds", str(confirm)]
    if canned is not None:
        (root / "canned.json").write_text(json.dumps(canned))
        monkeypatch.setenv("PH_MODEL_ENDPOINT_FAKE", str(root / "canned.json"))
    else:
        monkeypatch.delenv("PH_MODEL_ENDPOINT_FAKE", raising=False)
        argv += ["--proposer", "rules"]
    assert evolve.main(argv) == 0
    return root / "s", json.loads((root / "s" / "campaigns" / f"evolve-{TASK}" / "campaign.json").read_text())


def test_win_on_debug_seeds_that_holds_on_confirm_seeds_is_published(tmp_path, monkeypatch):
    canned = [{"kind": "executor", "payload": {"to": "alt"}, "summary": "grab 死", "rationale": "换 alt"}]
    session, doc = _campaign(tmp_path, monkeypatch, "hold", canned=canned)
    r = doc["rounds"][0]
    assert (r["before"], r["after"], r["published"], r["outcome"], r["parent"]) == (0, 2, True, "improved", 0)
    assert r["confirm"] == {"seeds": [3, 4], "before": 2, "after": 2}
    assert doc["confirm_base"] == {"seeds": [3, 4], "count": 2}   # cached: the next confirm skips the baseline
    assert r["usage"]["sim_s"] > 0 and r["usage"]["llm_tokens"] is None and doc["live"]["sim_s"] >= r["usage"]["sim_s"]
    assert any("新种子确认" in m["text"] for m in doc["live"]["messages"])
    # the LLM brief listed the baseline's failure keyframes (paths; the fake takes no images)
    audit = json.loads((session / "campaigns" / f"evolve-{TASK}" / "llm" / "round-1.json").read_text())
    text = audit["messages"][1]["content"]
    assert all(f"media/{TASK}/{seed}/grab-0.fail-{i}.jpg" in text for seed in (1, 2) for i in range(3))
    assert "keyframes" in text and "first frame" in audit["messages"][0]["content"]
    assert bs.rsi_series(session, TASK)[0]["confirm"] == r["confirm"]
    c = bs.rsi_campaigns(session)[0]
    assert c["published_rounds"] == [1] and c["usage"] == {"llm_tokens": None, "sim_s": r["usage"]["sim_s"]}


def test_win_that_regresses_on_confirm_seeds_is_not_published(tmp_path, monkeypatch):
    session, doc = _campaign(tmp_path, monkeypatch, "regress")
    r = doc["rounds"][0]
    assert (r["before"], r["after"], r["published"], r["outcome"]) == (0, 2, False, "improved")
    assert r["confirm"] == {"seeds": [3, 4], "before": 2, "after": 0}
    assert doc["applied"]["executors"] == {} and doc["best"] == 0
    assert bs.rsi_campaigns(session)[0]["published_rounds"] == []


def test_failure_keyframes_are_kept_on_drop_and_listed_by_rsi_frames(tmp_path, monkeypatch):
    session, doc = _campaign(tmp_path, monkeypatch, "never", confirm=0)
    r = doc["rounds"][0]
    assert (r["published"], r["outcome"], r["confirm"]) == (False, "same", None)
    frames = bs.rsi_frames(session, TASK, 1)
    for seed in (1, 2):
        d = frames["dropped"][f"{seed}/grab-0"]
        assert d["reason"] == "verify_failed"
        n = len(d["keyframes"])   # the last suite on these seeds ran ``alt`` (one governed frame)
        assert 1 <= n <= 3 and d["keyframes"] == [f"media/{TASK}/{seed}/grab-0.fail-{i}.jpg" for i in range(n)]
        for rel in d["keyframes"]:
            assert 0 < (session / rel).stat().st_size <= 25_000, rel
        assert media.dropped_of(session / "media", TASK, seed)["grab-0"]["keyframes"] == \
            [f"grab-0.fail-{i}.jpg" for i in range(n)]
    assert frames["media"] == r["media"] and f"media/{TASK}/1/reach-0.gif" in frames["media"]


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


def test_keyframes_ride_as_image_parts_when_the_endpoint_accepts_images(tmp_path):
    from PIL import Image
    (tmp_path / "media" / "t" / "1").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "media" / "t" / "1" / "g.fail-0.jpg")
    proj = {"first_death": {"node": "g"}, "this_round": {"per_seed": [
        {"seed": 1, "keyframes": ["media/t/1/g.fail-0.jpg"]}, {"seed": 2, "keyframes": []}]}}
    ep = _ImageEndpoint()
    tried, row = evolve_llm.llm_propose(ep, proj, {"seeds": {}}, 1, tmp_path / "llm", session=tmp_path)
    parts = ep.seen[1]["content"]
    assert [p["type"] for p in parts] == ["text", "text", "image_url"]
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,") and "seed 1 keyframe 0" in parts[1]["text"]
    assert row["usage"] == {"prompt": 12, "completion": 3} and tried["kind"] == "none"
    audit = (tmp_path / "llm" / "round-1.json").read_text()   # paths only, never bytes
    assert "base64" not in audit and "media/t/1/g.fail-0.jpg" in audit


def test_proposal_tunables_from_is_the_knobs_current_value(monkeypatch):
    monkeypatch.setattr(evolve, "mount_params", lambda ref: {"tunables": {"stall_k": 40}})
    before = {"seeds": {"1": {"first_death": "g", "nodes": {"g": {"skill": "grab", "executor": "scripted"}}}}}
    p = {"id": "x", "kind": "tunables", "note": "", "payload": {"ref": "r", "path": ["tunables", "stall_k"], "to": 28}}
    assert evolve.from_proposal(p, before)["detail"]["from"] == 40
    p["payload"]["path"] = ["tunables", "nope"]
    assert evolve.from_proposal(p, before)["detail"]["from"] is None
