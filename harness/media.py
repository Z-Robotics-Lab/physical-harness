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


class SegmentRecorder:
    def __init__(self, root: str | os.PathLike, task: str, seed: int, *,
                 every: int = EVERY) -> None:
        self.root = Path(root)
        self.task = str(task)
        self.seed = int(seed)
        self.every = max(1, int(every))
        self.frames: list[Any] = []   # PIL RGB images, SIZE x SIZE
        self._src = None
        self._driver = None
        self._untap = None
        self._n = 0
        self.error: str | None = None   # last capture/encode failure, for the reason

    # -- recording -------------------------------------------------------------
    def start(self, env: Any, driver: Any, embodiment: Any = None) -> None:
        self.stop()
        self.frames, self._n, self.error, self._driver = [], 0, None, driver
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
        if self._n % self.every or self._src is None:
            return
        try:
            img = _to_image(self._src(obs))
            if img is not None:
                self.frames.append(img)
        except Exception as exc:  # noqa: BLE001 -- a lost frame never touches the task
            self.error = repr(exc)

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
        frames, self.frames = self.frames, []
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
        frames, self.frames = self.frames, []
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
        path = self.keep(node) if ok else None
        if path is not None:
            return {"kept": True, "file": str(path.relative_to(self.root))}
        keyframes = self.drop(node)
        reason = ("verify_failed" if not ok else "no_frame_source" if not had_src
                  else "no_frames" if not had_frames else "encode_failed")
        out = {"kept": False, "reason": reason}
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
    evolve/suite briefs and for a task brief with ``media: true``); else None."""
    root = brief.get("media_dir")
    return SegmentRecorder(root, brief.get("task", "task"), seed) if root else None


# -- helpers -------------------------------------------------------------------

def _to_image(raw: Any):
    """A frame from any source shape -> SIZE x SIZE PIL RGB image. ``bytes`` is
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
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE))
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
