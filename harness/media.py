"""Segment media recorder: 128px frames in memory, on disk only after verify.

One ``SegmentRecorder`` per (session media root, task, seed). ``start(env,
driver, embodiment)`` taps ``driver.act`` so every EVERY-th driver step grabs one
frame from the first duck-typed source present: ``embodiment.frame(obs)`` (the
camera image already in the obs -- every driver of that embodiment gets it for
free, no renderer needed), else ``driver.frame()``, else ``env.frame()``.
``keep(node)`` encodes ``media/<task>/<seed>/<node>.mp4`` (imageio+ffmpeg
importable) else ``.gif`` (PIL), re-encoding at a lower fps until under
MAX_BYTES, and updates the seed's ``index.json``; ``drop()`` discards. Frames
are live state like scripts/frame_dump: they never enter the session-log chain
(only the index/paths reach the board's rsi_frames face). A lost clip never
fails a task, but it is never silent either: ``finish`` returns
``{kept, reason|file}`` for the node's diagnostics and writes the reason under
``index.json["dropped"]`` (no_frame_source / no_frames / verify_failed /
encode_failed) with up to 3 failure keyframes (``<node>.fail-{0,1,2}.jpg``: first,
stall/last-progress (``driver.last_progress_step`` when exposed, else middle), last).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

SIZE = 128
FPS = 10
#: capture every Nth driver step; a 300-step robocasa segment -> ~75 frames
EVERY = 4
MAX_BYTES = 1_000_000
KEYFRAME_QUALITY = 60   # 128px JPEG at q60: a few KB, well under the ~25 KB budget
#: The whole-episode video (``episode=True``): every captured frame of every segment,
#: passed or failed, streamed to ``<seed>/episode.mp4`` at this size -- the operator's
#: "watch the best round's rollout" view, never evidence. mp4 only (imageio+ffmpeg).
EPISODE_SIZE = 256
#: Failure keyframes are what a vision model reasons over: larger than the clips.
KEYFRAME_SIZE = 256
#: Driver-independent motion trace: base/eef pose read off the simulator every
#: MOTION_EVERY driver steps (``read_pose``), so a segment's evidence never depends on
#: what the stage driver chose to report.
MOTION_EVERY = 2


class SegmentRecorder:
    def __init__(self, root: str | os.PathLike, task: str, seed: int, *,
                 every: int = EVERY, episode: bool = False) -> None:
        self.root = Path(root)
        self.task = str(task)
        self.seed = int(seed)
        self.every = max(1, int(every))
        self.episode = bool(episode)
        self._ep = None            # the streaming episode writer, opened on the first frame
        self._ep_frames = 0
        self.frames: list[Any] = []   # PIL RGB images, SIZE x SIZE
        self.big: list[Any] = []      # the same frames at KEYFRAME_SIZE, for the failure keyframes
        self.motion: list[dict] = []  # [{step, base:[x,y,yaw], eef:[x,y,z]}] of the running segment
        self.scene: dict | None = None
        self.contacts_start: dict | None = None
        self._env = None
        self._src = None
        self._driver = None
        self._untap = None
        self._n = 0
        self.error: str | None = None   # last capture/encode failure, for the reason

    # -- recording -------------------------------------------------------------
    def start(self, env: Any, driver: Any, embodiment: Any = None) -> None:
        self.stop()
        self.frames, self.big, self.motion, self._n, self.error = [], [], [], 0, None
        self._driver, self._env = driver, env
        self.scene = read_scene(env)   # the layout at segment start (objects move; fixtures do not)
        self.contacts_start = read_contacts(env)   # what is in the hand / on the floor as the segment begins
        emb = getattr(embodiment, "frame", None)
        src = getattr(driver, "frame", None) or getattr(env, "frame", None)
        # one callable(obs): the embodiment reads the obs, the legacy sources ignore it
        self._src = emb if emb is not None else (src and (lambda obs: src()))
        if self._src is None:
            return
        orig = driver.act

        def act(obs):
            self.capture(obs)
            return orig(obs)

        driver.act = act   # instance attr shadows the class method; stop() removes it
        self._untap = lambda: driver.__dict__.pop("act", None)

    def capture(self, obs: Any = None) -> None:
        self._n += 1
        if self._n % MOTION_EVERY == 0 and self._env is not None:
            pose = read_pose(self._env)
            if pose:
                hand = (read_contacts(self._env) or {}).get("hand")   # what the gripper holds, per sample
                self.motion.append({"step": self._n, **pose, **({"hand": hand} if hand is not None else {})})
        if self._n % self.every or self._src is None:
            return
        try:
            raw = self._src(obs)
            img = _to_image(raw)
            if img is not None:
                self.frames.append(img)
                self.big.append(_to_image(raw, KEYFRAME_SIZE))
                if self.episode:
                    self._episode_frame(_to_image(raw, EPISODE_SIZE))
        except Exception as exc:  # noqa: BLE001 -- a lost frame never touches the task
            self.error = repr(exc)

    def _episode_frame(self, img) -> None:
        import numpy as np
        if self._ep is None:
            try:
                import imageio.v2 as imageio
                import imageio_ffmpeg  # noqa: F401
            except ImportError:      # gif would be huge: no episode video without ffmpeg
                self.episode = False
                return
            self.seed_dir.mkdir(parents=True, exist_ok=True)
            self._ep = imageio.get_writer(str(self.seed_dir / "episode.tmp.mp4"), fps=FPS, format="FFMPEG",
                                          codec="libx264", macro_block_size=None)
        self._ep.append_data(np.asarray(img))
        self._ep_frames += 1

    def close_episode(self) -> dict | None:
        """End of the episode: finalise ``episode.mp4`` and index it as the pseudo-node
        ``episode`` (so the board's media list carries it). None when nothing streamed."""
        if self._ep is None:
            return None
        writer, self._ep = self._ep, None
        try:
            writer.close()
            path = self.seed_dir / "episode.mp4"
            os.replace(self.seed_dir / "episode.tmp.mp4", path)
            entry = {"file": path.name, "bytes": path.stat().st_size, "frames": self._ep_frames,
                     "fps": FPS, "ts": time.time()}
            _index(self.seed_dir, "episode", entry)
            return entry
        except Exception as exc:  # noqa: BLE001 -- a lost video never touches the task
            self.error = repr(exc)
            return None

    def stop(self) -> None:
        if self._untap is not None:
            self._untap()
            self._untap = None
        self._src = None

    # -- outcome ---------------------------------------------------------------
    def drop(self, node: str | None = None) -> list[str]:
        """Discard the clip; with ``node``, first save up to 3 failure keyframes
        (``<node>.fail-<i>.jpg``, SIZE px JPEG) and return their file names."""
        self.stop()
        frames, self.frames = (self.big or self.frames), []
        self.big = []
        if node is None or not frames:
            return []
        stall = getattr(self._driver, "last_progress_step", None)
        mid = (len(frames) - 1) // 2 if stall is None else min(len(frames) - 1, max(0, int(stall) // self.every - 1))
        picks = sorted({0, mid, len(frames) - 1})
        names = []
        try:
            self.seed_dir.mkdir(parents=True, exist_ok=True)
            for i, k in enumerate(picks):
                name = f"{node}.fail-{i}.jpg"
                frames[k].save(self.seed_dir / name, "JPEG", quality=KEYFRAME_QUALITY)
                names.append(name)
        except Exception as exc:  # noqa: BLE001 -- a lost keyframe never touches the task
            self.error = repr(exc)
        return names

    def keep(self, node: str) -> Path | None:
        """Encode the segment's frames to ``<root>/<task>/<seed>/<node>.(mp4|gif)``
        and index it. None when nothing was captured or every encode failed."""
        self.stop()
        frames, self.frames, self.big = self.frames, [], []
        if not frames:
            return None
        try:
            self.seed_dir.mkdir(parents=True, exist_ok=True)
            path, fps, n = _encode(frames, self.seed_dir / str(node))
            _index(self.seed_dir, node, {"file": path.name, "bytes": path.stat().st_size,
                                         "frames": n, "fps": fps, "ts": time.time()})
            return path
        except Exception as exc:  # noqa: BLE001 -- a lost clip never touches the task
            self.error = repr(exc)
            return None

    @property
    def seed_dir(self) -> Path:
        return self.root / self.task / str(self.seed)

    def finish(self, node: str, ok: bool) -> dict:
        """The workload's one call: keep on verify success, drop otherwise. Returns
        the node's ``diagnostics.media``: ``{"kept": True, "file": "<task>/<seed>/
        <node>.mp4"}`` or ``{"kept": False, "reason": ...[, "error": ...]}`` -- the
        same reason is indexed under ``index.json["dropped"]`` so a run with no
        clip at all still leaves a readable trace under media/."""
        had_src, had_frames = self._src is not None, bool(self.frames)
        motion, self.motion = self.motion, []
        scene, self.scene = self.scene, None
        # where the task objects ended up (a released object may have rolled or fallen)
        objects = {k: v["pos"] for k, v in (read_scene(self._env) or {}).items() if k.startswith("obj:")} \
            if self._env is not None else {}
        contacts_end = read_contacts(self._env) if self._env is not None else None
        contacts, self.contacts_start = ({"start": self.contacts_start, "end": contacts_end}
                                         if contacts_end or self.contacts_start else None), None
        extra = {**({"motion": motion} if motion else {}), **({"scene": scene} if scene else {}),
                 **({"objects_end": objects} if objects else {}), **({"contacts": contacts} if contacts else {})}
        path = self.keep(node) if ok else None
        if path is not None:
            return {"kept": True, "file": str(path.relative_to(self.root)), **extra}
        keyframes = self.drop(node)
        reason = ("verify_failed" if not ok else "no_frame_source" if not had_src
                  else "no_frames" if not had_frames else "encode_failed")
        out = {"kept": False, "reason": reason, **extra}
        if self.error:
            out["error"] = self.error
        try:
            self.seed_dir.mkdir(parents=True, exist_ok=True)
            _index(self.seed_dir, node, None, {"reason": reason, "keyframes": keyframes})
        except OSError:
            pass
        return out


def recorder_for(brief: Any, seed: int) -> SegmentRecorder | None:
    """A recorder when the brief names a ``media_dir`` (the runtime sets it for
    evolve/suite briefs and for a task brief with ``media: true``); else None.
    ``media_episode: true`` adds the whole-episode video."""
    root = brief.get("media_dir")
    return SegmentRecorder(root, brief.get("task", "task"), seed,
                           episode=bool(brief.get("media_episode"))) if root else None


# -- helpers -------------------------------------------------------------------

def read_scene(env: Any) -> dict | None:
    """The kitchen as the simulator lays it out for this seed: every fixture's ``pos``
    (and ``size`` when it has one) plus every task object's body position -- duck-typed
    off ``env.fixtures`` / ``env.obj_body_id`` (robocasa), import-free. None elsewhere."""
    import contextlib
    out: dict = {}
    fixtures = getattr(env, "fixtures", None)
    if isinstance(fixtures, dict):
        for name, fx in fixtures.items():
            with contextlib.suppress(Exception):
                pos = getattr(fx, "pos", None)
                if pos is None:
                    continue
                entry = {"pos": [round(float(v), 3) for v in list(pos)[:3]]}
                size = getattr(fx, "size", None)
                if size is not None:
                    entry["size"] = [round(float(v), 3) for v in list(size)[:3]]
                out[str(name)] = entry
    objs, sim = getattr(env, "obj_body_id", None), getattr(env, "sim", None)
    if isinstance(objs, dict) and sim is not None:
        for name, bid in objs.items():
            with contextlib.suppress(Exception):
                out[f"obj:{name}"] = {"pos": [round(float(v), 3) for v in sim.data.body_xpos[bid][:3]]}
    return out or None


def read_contacts(env: Any) -> dict | None:
    """Who touches whom, off the MuJoCo contact list: task objects in the gripper
    (``hand``), on the floor (``floor``), fixtures the mobile base is pressed against
    (``base``), and every task object's contact partners (``objects``). Duck-typed off
    ``env.sim`` / ``env.obj_body_id`` (robocasa); None elsewhere. Bodies are named by
    their MuJoCo body name minus the ``_main`` suffix."""
    import contextlib
    sim, objs = getattr(env, "sim", None), getattr(env, "obj_body_id", None)
    if sim is None or not isinstance(objs, dict):
        return None
    out: dict = {"hand": [], "floor": [], "base": [], "objects": {}}
    with contextlib.suppress(Exception):
        m, d = sim.model, sim.data
        by_body = {int(bid): name for name, bid in objs.items()}
        short = lambda n: (n or "?").removesuffix("_main")
        for i in range(int(d.ncon)):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            n1, n2 = m.body_id2name(b1) or "", m.body_id2name(b2) or ""
            for obj_bid, other_bid, other in ((b1, b2, n2), (b2, b1, n1)):
                if obj_bid in by_body:
                    name = by_body[obj_bid]
                    partner = by_body.get(other_bid, short(other))
                    out["objects"].setdefault(name, [])
                    if partner not in out["objects"][name]:
                        out["objects"][name].append(partner)
                    if other.startswith("gripper0") and name not in out["hand"]:
                        out["hand"].append(name)
                    if "floor" in other and name not in out["floor"]:
                        out["floor"].append(name)
            for base_name, other in ((n1, n2), (n2, n1)):
                if base_name.startswith("mobilebase0") and not other.startswith(("robot0", "mobilebase0", "gripper0")):
                    fx = by_body.get(int(m.geom_bodyid[c.geom2 if base_name == n1 else c.geom1]), short(other))
                    if fx not in out["base"] and "floor" not in fx:
                        out["base"].append(fx)
    return out


def read_pose(env: Any) -> dict | None:
    """Base ``[x, y, yaw]`` and end-effector ``[x, y, z]`` read straight off a
    robosuite-style MuJoCo env (``env.sim``, ``env.robots``), duck-typed and
    import-free; whichever half is absent is simply omitted. None when the env
    exposes neither (the stdlib fakes)."""
    sim, robots = getattr(env, "sim", None), getattr(env, "robots", None)
    if sim is None:
        return None
    import contextlib
    import math
    out: dict = {}
    with contextlib.suppress(Exception):   # fixed-base robots have no mobile base body
        bid = sim.model.body_name2id("mobilebase0_base")
        p, r = sim.data.body_xpos[bid], sim.data.body_xmat[bid]
        out["base"] = [round(float(p[0]), 3), round(float(p[1]), 3), round(math.atan2(float(r[3]), float(r[0])), 3)]
    with contextlib.suppress(Exception):   # no arm site: nothing to read
        site = robots[0].eef_site_id
        site = site["right"] if isinstance(site, dict) else site
        out["eef"] = [round(float(v), 3) for v in sim.data.site_xpos[site][:3]]
    return out or None


def _to_image(raw: Any, size: int = SIZE):
    """A frame from any source shape -> size x size PIL RGB image. ``bytes`` is
    a packed SIZE*SIZE*3 RGB buffer (the stdlib-only fake); anything else is an
    HxWx3 uint8 array."""
    if raw is None:
        return None
    from PIL import Image

    if isinstance(raw, (bytes, bytearray)):
        img = Image.frombytes("RGB", (SIZE, SIZE), bytes(raw))
    else:
        import numpy as np

        img = Image.fromarray(np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size))
    return img


def _writer():
    try:
        import imageio.v2 as imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
        return ".mp4", _write_mp4
    except ImportError:
        return ".gif", _write_gif


def _write_mp4(frames, path: str, fps: int) -> None:
    import imageio.v2 as imageio
    import numpy as np

    imageio.mimwrite(path, [np.asarray(f) for f in frames], fps=fps,
                     format="FFMPEG", codec="libx264", macro_block_size=None)


def _write_gif(frames, path: str, fps: int) -> None:
    frames[0].save(path, "GIF", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)


def _encode(frames: list, stem: Path) -> tuple[Path, int, int]:
    """Write ``stem + ext`` atomically; halve the frame rate (subsample) until
    the file is under MAX_BYTES or a single frame remains."""
    ext, write = _writer()
    path = Path(str(stem) + ext)
    tmp = Path(str(stem) + ".tmp" + ext)
    stride = 1
    while True:
        sub = frames[::stride]
        fps = max(1, FPS // stride)
        write(sub, str(tmp), fps)
        if tmp.stat().st_size <= MAX_BYTES or len(sub) <= 1:
            break
        stride *= 2
    os.replace(tmp, path)
    return path, fps, len(sub)


def _index(seed_dir: Path, node: str, entry: dict | None, reason: dict | None = None) -> None:
    """Atomically move ``node`` to ``files`` (kept, ``entry``) or ``dropped``
    (``{reason, keyframes}``) in the seed's index.json -- a node is in exactly one of them."""
    idx = seed_dir / "index.json"
    try:
        data = json.loads(idx.read_text())
    except (OSError, ValueError):
        data = {}
    files, dropped = data.setdefault("files", {}), data.setdefault("dropped", {})
    if entry is not None:
        files[str(node)] = entry
        dropped.pop(str(node), None)
    else:
        dropped[str(node)] = reason
        files.pop(str(node), None)
    tmp = idx.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=1))
    os.replace(tmp, idx)


def index_of(root: str | os.PathLike, task: str, seed: int) -> dict:
    """The kept files for one (task, seed): ``{node: {file, bytes, frames, fps, ts}}``
    -- what the board's rsi_frames face lists. Empty when nothing was kept."""
    return _read_index(root, task, seed, "files")


def dropped_of(root: str | os.PathLike, task: str, seed: int) -> dict:
    """``{node: {reason, keyframes: [file names]}}`` of the segments that left no clip
    (verify_failed / no_frame_source / no_frames / encode_failed; an index older than
    keyframes reads ``keyframes: []``). Empty when nothing was dropped."""
    return {n: v if isinstance(v, dict) else {"reason": v, "keyframes": []}
            for n, v in _read_index(root, task, seed, "dropped").items()}


def _read_index(root, task, seed, key: str) -> dict:
    idx = Path(root) / str(task) / str(seed) / "index.json"
    try:
        return dict(json.loads(idx.read_text()).get(key) or {})
    except (OSError, ValueError):
        return {}
