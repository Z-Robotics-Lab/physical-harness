"""harness.media: segment clips kept on verify success, dropped on failure, <1MB.

Frames come from the fake embodiment's synthetic ``frame()`` (harness.fakes);
the recorder taps ``driver.act`` while a segment drives and encodes only when
the workload's ``_segment`` hook reports success. No simulator, no chain rows.
"""

from __future__ import annotations

import json
from dataclasses import replace

from harness import media
from harness.fakes import _FakeEnvHandle
from harness.spec import EpisodeSpec
from plugins.task import workload

MB = 1_000_000


class _Driver:
    """Heterogeneous episodic driver stand-in: steps the env on act, done by flag."""

    def __init__(self, ok: bool) -> None:
        self.env, self.ok, self.k = None, ok, 0

    def enter_segment(self, env, spec):
        self.env, self.k = env, 0

    def act(self, obs):
        self.k += 1
        return self.env.step((0.0,) * 7)[0]

    def segment_success(self, env) -> bool:
        return self.ok


def _drive(env, driver, steps: int) -> None:
    obs = env.reset()
    driver.enter_segment(env, None)
    for _ in range(steps):
        obs = driver.act(obs)


def test_keep_writes_one_small_clip_and_indexes_it(tmp_path):
    root = tmp_path / "media"
    rec = media.SegmentRecorder(root, "thaw", 7, every=1)
    env, driver = _FakeEnvHandle(), _Driver(True)
    rec.start(env, driver)
    _drive(env, driver, 600)          # 600 raw frames: over 1MB unencoded
    assert len(rec.frames) == 600
    path = rec.keep("thaw.pick")
    assert path is not None and path.parent == root / "thaw" / "7"
    assert path.suffix in (".mp4", ".gif") and 0 < path.stat().st_size <= MB
    assert "act" not in driver.__dict__          # tap removed
    files = media.index_of(root, "thaw", 7)
    assert set(files) == {"thaw.pick"} and files["thaw.pick"]["file"] == path.name
    assert files["thaw.pick"]["bytes"] == path.stat().st_size
    assert not list(path.parent.glob("*.tmp*"))  # atomic: no leftovers
    assert not rec.frames


def test_oversize_clip_is_subsampled_until_under_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "MAX_BYTES", 8_000)
    rec = media.SegmentRecorder(tmp_path, "thaw", 7, every=1)
    env, driver = _FakeEnvHandle(), _Driver(True)
    rec.start(env, driver)
    _drive(env, driver, 300)
    path = rec.keep("n")
    entry = media.index_of(tmp_path, "thaw", 7)["n"]
    assert path.stat().st_size <= 8_000
    assert entry["frames"] < 300 and entry["fps"] < media.FPS


def test_drop_leaves_nothing_on_disk(tmp_path):
    root = tmp_path / "media"
    rec = media.SegmentRecorder(root, "thaw", 7)
    env, driver = _FakeEnvHandle(), _Driver(False)
    rec.start(env, driver)
    _drive(env, driver, 40)
    assert rec.frames
    rec.drop()
    assert not rec.frames and not root.exists() and "act" not in driver.__dict__


def test_no_frame_source_records_nothing(tmp_path):
    class _Blind:
        def reset(self):
            return {}

        def step(self, a):
            return {}, 0.0, False, {}

    rec = media.SegmentRecorder(tmp_path, "t", 1, every=1)
    driver = _Driver(True)
    rec.start(_Blind(), driver)
    assert rec.keep("n") is None and "act" not in driver.__dict__


def test_workload_segment_hook_keeps_on_success_drops_on_failure(tmp_path, monkeypatch):
    """The `_segment` handler brackets the drive with start/finish: the clip
    lands iff the segment verified, under the brief's media_dir, by node id."""
    root = tmp_path / "media"
    spec = EpisodeSpec(seed=3, task="fake", horizon=500,
                       env_provider="harness.fakes:env_provider",
                       policy_provider="harness.fakes:policy_provider")

    def fake_governed(ep, seg_spec, bundle, *, step_budget, executor=None):
        ep.driver.enter_segment(ep.env, seg_spec)
        obs = ep.obs
        for _ in range(60):
            obs = ep.driver.act(obs)
        ep.obs, ep.cursor = obs, ep.cursor + 60
        return {"obs": obs, "steps": 60, "stages": [],
                "success": ep.driver.segment_success(ep.env)}

    monkeypatch.setattr(workload, "_governed_segment", fake_governed)
    brief = {"task": "thaw", "media_dir": str(root)}
    rec = media.recorder_for(brief, 3)
    assert isinstance(rec, media.SegmentRecorder)
    assert media.recorder_for({"task": "thaw"}, 3) is None   # off without media_dir

    def run_node(node_id: str, ok: bool) -> dict:
        env = _FakeEnvHandle()
        ep = workload.EpisodeContext(None, env, _Driver(ok), spec, env.reset())
        ctx = workload.NodeCtx(seed=3, env_ref=spec.env_provider,
                               policy_ref=spec.policy_provider, skills=(),
                               nodes_out={}, predicates={}, episode=ep, media=rec)
        return workload._segment({"id": node_id, "skill": "s", "args": {}}, ctx)

    assert run_node("good", True)["success"] is True
    assert run_node("bad", False)["success"] is False
    files = media.index_of(root, "thaw", 3)
    assert set(files) == {"good"}
    kept = root / "thaw" / "3" / files["good"]["file"]
    assert kept.exists() and kept.stat().st_size <= MB
    assert not [p for p in kept.parent.iterdir() if p.stem == "bad"]
    assert json.loads((kept.parent / "index.json").read_text())["files"].keys() == {"good"}


def test_ctx_default_has_no_recorder():
    ctx = workload.NodeCtx(seed=0, env_ref="e", policy_ref="p", skills=(),
                           nodes_out={}, predicates={})
    assert ctx.media is None and replace(ctx, media=None).media is None


# -- never silent: the reason a node left no clip ----------------------------------

def test_finish_seals_no_frame_source_when_nothing_can_frame(tmp_path):
    class _Blind:
        def reset(self):
            return {}

        def step(self, a):
            return {}, 0.0, False, {}

    rec = media.SegmentRecorder(tmp_path, "t", 1, every=1)
    driver = _Driver(True)
    rec.start(_Blind(), driver)
    _drive(_Blind(), driver, 8)
    out = rec.finish("n", True)
    assert out == {"kept": False, "reason": "no_frame_source"}
    assert media.dropped_of(tmp_path, "t", 1) == {"n": {"reason": "no_frame_source", "keyframes": []}}
    assert media.index_of(tmp_path, "t", 1) == {}


def test_finish_seals_no_frames_when_the_source_yields_none_and_names_the_error(tmp_path):
    class _Emb:
        def frame(self, obs):
            raise RuntimeError("camera off")

    rec = media.SegmentRecorder(tmp_path, "t", 2, every=1)
    env, driver = _FakeEnvHandle(), _Driver(True)
    rec.start(env, driver, _Emb())          # embodiment wins over env.frame
    _drive(env, driver, 8)
    out = rec.finish("n", True)
    assert out["kept"] is False and out["reason"] == "no_frames"
    assert "camera off" in out["error"]
    assert media.dropped_of(tmp_path, "t", 2) == {"n": {"reason": "no_frames", "keyframes": []}}

    class _Silent:
        def frame(self, obs):
            return None

    rec.start(env, driver, _Silent())
    _drive(env, driver, 8)
    assert rec.finish("m", True) == {"kept": False, "reason": "no_frames"}


def test_embodiment_frame_reads_the_obs_and_a_kept_node_leaves_dropped(tmp_path):
    seen = []

    class _Emb:
        def frame(self, obs):
            seen.append(obs)
            return b"\x10\x20\x30" * (media.SIZE * media.SIZE)

    rec = media.SegmentRecorder(tmp_path, "t", 3, every=1)
    env, driver = _FakeEnvHandle(), _Driver(True)
    rec.start(env, driver, _Emb())
    _drive(env, driver, 6)
    assert len(seen) == 6 and all(isinstance(o, dict) for o in seen)
    assert rec.finish("n", False) == {"kept": False, "reason": "verify_failed"}
    assert media.dropped_of(tmp_path, "t", 3)["n"]["reason"] == "verify_failed"
    rec.start(env, driver, _Emb())
    _drive(env, driver, 6)
    out = rec.finish("n", True)
    assert out["kept"] is True and out["file"].startswith("t/3/n.")
    assert media.dropped_of(tmp_path, "t", 3) == {} and set(media.index_of(tmp_path, "t", 3)) == {"n"}


def test_robocasa_embodiment_frame_is_the_obs_camera_image_flipped():
    """The embodiment-level source every robocasa driver shares: no renderer,
    the obs image (opengl convention) flipped upright; None on an image-less obs."""
    import numpy as np

    from plugins.embodiment_robocasa import env as renv
    from plugins.embodiment_robocasa import provider

    img = np.zeros((128, 128, 3), np.uint8)
    img[0, :, 0] = 255                    # top row red in obs -> bottom row in the frame
    out = provider().frame({"robot0_agentview_left_image": img, "robot0_eef_pos": [0, 0, 0]})
    assert out.shape == (128, 128, 3) and out[-1, 0, 0] == 255 and out[0, 0, 0] == 0
    assert media._to_image(out).size == (media.SIZE, media.SIZE)
    assert renv.frame({"robot0_eef_pos": [0, 0, 0]}) is None and renv.frame(None) is None
