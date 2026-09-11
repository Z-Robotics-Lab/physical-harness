"""The frames overlay: offscreen viewport frames on ``runs/<session>/frame.jpg``.

watch_stack's sibling for a REMOTE browser: the operator watches the sim through
the ph-station 取景窗 panel, which polls the board's ``runtime_frame`` face --
so this dumps a small JPEG per step interval instead of opening an X window.
Same delegating-wrapper design as scripts/watch_stack.py: a generic
embodiment.env provider wraps ANY base ref (robosuite card, robocasa card, or
the viewer overlay itself) and only ADDS a render -- semantics delegate
verbatim, so the wrapper can never move an episode's outcome.

Live state, not evidence (the runtime_status.json family): the frame file is
overwritten in place (temp + os.replace, the brief_drop atomicity), never enters
the session-log chain, and every failure in the dump path is swallowed -- a
broken GL stack loses the picture, never the task. The offscreen render reads
mjData and consumes no rng (mjv_updateScene/mjr_render are read-only; the one
``sim.forward()`` in lazy context creation recomputes derived quantities from
the existing state), so frames cannot perturb episode determinism.

Frame cadence is a STEP interval, not a wall clock: time-based throttling would
make the dump sequence depend on host load. Every step (~30fps on a paced
rollout); size/quality tunables sit below with the measured costs.

KEYFRAMES ride the same overlay: an ``harness.opstream.on_emit`` listener keeps
one extra still per interesting event at
``runs/<session>/keyframes/<seq:06d>-<kind>.jpg``. Registration is inverted
(harness imports nothing; this module registers itself at import), the still
reuses the offscreen context the live dump already built -- no second GL
context, no extra VRAM -- and the whole directory is cleared by ``opstream.arm``
on the feed's truncate-per-boot lifecycle. Anchored to a feed seq and never to
a chain row, so a keyframe is live state like frame.jpg: deleting the directory
loses zero evidence.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from harness import opstream

#: Armed destination (str path) or None. Module-level singleton like
#: harness.opstream: one resident runtime per process, armed at boot only.
_PATH: str | None = None

#: The env of the most recent reset/step, or None between episodes (cleared on
#: close: rendering a torn-down sim is not a swallowable failure). The keyframe
#: listener has no env argument -- it fires from the event stream, so the
#: wrapper leaves the live world here for it, the watch_stack module-global
#: channel again.
_LAST_ENV = None

#: Event kinds worth a still, as DATA: the listener is one set membership test,
#: so widening the set is a constant edit, never a new branch. A kind emitted
#: when no world is open captures nothing (task_done fires after close; node_*
#: only has a live env on the persistent-episode path) -- an absent still is a
#: missing thumbnail, and the index face only lists what exists.
KEYFRAME_KINDS = frozenset({"plan_built", "node_start", "stage_transition",
                            "node_verified", "node_failed", "task_done"})

#: Per-boot capture ceiling (~2000 x ~30KB = ~60MB worst case). A runaway
#: replan loop stops capturing instead of filling the disk; silently, because
#: raising here would break the task the stills only watch.
MAX_KEYFRAMES = 2000
_CAPTURED = 0

#: Latest-rollout video capture. Frames are operational state beside frame.jpg,
#: never chain evidence. One resident runtime processes one brief at a time.
VIDEO_FPS = 20
MAX_VIDEO_FRAMES = 6000
_VIDEO_ACTIVE = False
_VIDEO_SEQ = 0

#: The base embodiment.env ref whose SEMANTICS the frames overlay delegates to.
#: harness_runtime._mount_plan sets it per task (the sim override or the viewer
#: ref when --render is also on). Read at CALL time -- in-process only, module
#: globals do not survive spawn, which is fine: campaigns run headless subprocesses
#: that never mount this overlay.
_BASE_REF = "plugins.embodiment_robosuite:provider"

#: Dump every Nth env step. A step interval, not a time throttle, so the dump
#: sequence is deterministic per episode. 1 = every step: measured on the 4090
#: EGL stack a 640x480 offscreen render is ~0.2ms and the JPEG encode ~2ms
#: against a ~30ms paced step, so per-step dumping costs the sim <10% and the
#: reader (the board's long-poll), not the writer, is the viewport fps ceiling.
EVERY = 1


def _size(raw: str) -> tuple[int, int]:
    """Parse a WxH spec (e.g. ``640x480``); anything malformed falls back to
    the 640x480 default -- a bad env var degrades resolution, never the frames."""
    try:
        w, h = raw.lower().split("x")
        return int(w), int(h)
    except ValueError:
        return 640, 480


#: Frame size, overridable per boot via PH_FRAMES_SIZE=WxH (default 640x480:
#: fills a dashboard grid cell; ~15-50KB as a q70 JPEG depending on the scene).
WIDTH, HEIGHT = _size(os.environ.get("PH_FRAMES_SIZE", "640x480"))

#: JPEG quality: 70 keeps the 640x480 frame's b64-over-RPC hop cheap (~2x the
#: old 400x300 q80 bytes for 2.6x the pixels).
QUALITY = 70

#: Offscreen camera preference, first present in the model wins: the robocasa
#: kitchen head cam, then the robosuite tabletop views. None (free camera) as
#: the last resort keeps the dump alive on an unknown model.
CAMERAS = ("robot0_agentview_left", "agentview", "frontview")


def arm(path) -> None:
    """Direct subsequent dumps at ``path`` (str or Path), or disarm with None.
    Runtime-boot only; unarmed, the wrapper is a pure pass-through."""
    global _PATH, _LAST_ENV, _CAPTURED, _VIDEO_ACTIVE, _VIDEO_SEQ
    _PATH = str(path) if path else None
    _LAST_ENV = None
    _CAPTURED = 0
    _VIDEO_ACTIVE = False
    _VIDEO_SEQ = 0
    if _PATH is not None:
        shutil.rmtree(os.path.join(os.path.dirname(_PATH), "rollout-frames"),
                      ignore_errors=True)


def _record_video_frame(image) -> None:
    """Append one already-rendered image to the active rollout staging area."""
    global _VIDEO_SEQ
    if not _VIDEO_ACTIVE or _PATH is None or _VIDEO_SEQ >= MAX_VIDEO_FRAMES:
        return
    try:
        directory = os.path.join(os.path.dirname(_PATH), "rollout-frames")
        os.makedirs(directory, exist_ok=True)
        _VIDEO_SEQ += 1
        image.save(os.path.join(directory, f"{_VIDEO_SEQ:06d}.jpg"),
                   "JPEG", quality=QUALITY)
    except Exception:  # noqa: BLE001, S110 -- rollout capture cannot affect the task
        pass


def _video_event(seq: int, kind: str) -> None:
    """Start/finalize latest-rollout capture from task lifecycle events."""
    del seq
    global _VIDEO_ACTIVE, _VIDEO_SEQ
    if _PATH is None:
        return
    root = os.path.dirname(_PATH)
    staging = os.path.join(root, "rollout-frames")
    output = os.path.join(root, "rollout.mp4")
    if kind == "task_claimed":
        _VIDEO_ACTIVE = True
        _VIDEO_SEQ = 0
        shutil.rmtree(staging, ignore_errors=True)
        try:
            os.remove(output)
        except OSError:
            pass
        return
    if kind not in {"task_done", "task_failed", "task_cancelled"}:
        return
    _VIDEO_ACTIVE = False
    if _VIDEO_SEQ == 0:
        shutil.rmtree(staging, ignore_errors=True)
        return
    temporary = os.path.join(root, "rollout.tmp.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(VIDEO_FPS),
             "-i", os.path.join(staging, "%06d.jpg"), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", temporary],
            check=True, timeout=120, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        os.replace(temporary, output)
    except Exception:  # noqa: BLE001 -- every encoder failure leaves no rollout
        try:
            os.remove(temporary)
        except OSError:
            pass
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def dump(env, path: str | None = None) -> None:
    """Render one offscreen frame of ``env`` and atomically publish it to
    ``path`` (default: the armed live-viewport file).

    NEVER raises (the opstream.emit contract): a lost frame is a stale viewport,
    a raised one would kill the task. An env exposing ``render_frame()`` (a
    non-mujoco adapter) renders its own frame; otherwise the mujoco path
    creates the sim's offscreen render context lazily -- the production envs
    are built windowless AND cameraless, so none exists yet -- and flips the
    image vertically (mjr_readPixels is bottom-up).

    The armed check gates BOTH destinations, so keyframes follow ``--frames``:
    frames off, nothing renders anywhere.
    """
    if _PATH is None:
        return
    dest = path or _PATH
    try:
        render_frame = getattr(env, "render_frame", None)
        if callable(render_frame):
            # A non-mujoco embodiment adapter (SAPIEN/ManiSkill: no mujoco
            # `.sim` handle) publishes its own HWC uint8 RGB frame through
            # this probe -- already top-down, so no vertical flip.
            px = render_frame()
            if px is None:
                return
        else:
            sim = env.sim
            if sim._render_context_offscreen is None:
                from robosuite.utils.binding_utils import MjRenderContextOffscreen
                MjRenderContextOffscreen(sim, device_id=-1)
            names = tuple(getattr(sim.model, "camera_names", ()) or ())
            camera = next((c for c in CAMERAS if c in names), None)
            # mjr_readPixels is bottom-up: flip here, in the mujoco branch only.
            px = sim.render(width=WIDTH, height=HEIGHT, camera_name=camera)[::-1]
        from PIL import Image

        tmp = dest + ".tmp"
        image = Image.fromarray(px)
        image.save(tmp, "JPEG", quality=QUALITY)
        os.replace(tmp, dest)
        if path is None:
            _record_video_frame(image)
    except Exception:  # noqa: BLE001, S110 -- viewport capture cannot affect the task
        pass


def keyframe(seq: int, kind: str) -> None:
    """opstream on_emit listener: pin one still to event ``seq``.

    Reuses the live dump's render path (same offscreen context, same camera
    pick, same atomic publish), so a keyframe costs one render + one encode and
    zero extra GPU memory. Silent on every skip -- unarmed, uninteresting kind,
    no open world, or the per-boot ceiling reached.
    """
    global _CAPTURED
    d = opstream.keyframe_dir()
    if (_PATH is None or _LAST_ENV is None or d is None
            or kind not in KEYFRAME_KINDS or _CAPTURED >= MAX_KEYFRAMES):
        return
    _CAPTURED += 1
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return
    dump(_LAST_ENV, os.path.join(d, f"{seq:06d}-{kind}.jpg"))


class _FrameEnv:
    """Delegating wrapper: same env, plus a frame dump every EVERY steps.

    Only renders -- reset/step return the base env's values verbatim and the
    dump consumes no rng, so wrapping changes no episode semantics. It also
    publishes the live world to ``_LAST_ENV`` for the keyframe listener, and
    retracts it on close so no listener ever renders a torn-down sim.
    """

    def __init__(self, env):
        self._env = env
        self._steps = 0

    def reset(self):
        global _LAST_ENV
        obs = self._env.reset()
        self._steps = 0
        _LAST_ENV = self._env
        dump(self._env)
        return obs

    def step(self, action):
        global _LAST_ENV
        out = self._env.step(action)
        self._steps += 1
        _LAST_ENV = self._env
        # An env exposing frames_suppressed=True is mid-bookkeeping (teleport
        # dock probing): those frames are strobing pose-candidates, not
        # execution, and rendering is not evidence either way.
        if self._steps % EVERY == 0 and not getattr(
                self._env, "frames_suppressed", False):
            dump(self._env)
        return out

    def close(self):
        # Explicit (not __getattr__): a closed sim is the one render input that
        # is not merely broken but unsafe to touch, so the listener's handle
        # goes at the same instant the world does.
        global _LAST_ENV
        if _LAST_ENV is self._env:
            _LAST_ENV = None
        return self._env.close()

    def __getattr__(self, name):  # sim, _check_success, get_ep_meta, ...
        return getattr(self._env, name)


class FrameEmbodiment:
    """A manifest embodiment.env provider overlaying the frame dump on ANY base.

    ViewerEmbodiment's pattern verbatim: the semantic contract methods delegate
    to the wrapped provider (explicit forwards for the STRUCTURAL EnvProvider
    members -- runtime_checkable's getattr_static does not see __getattr__);
    only make_env is specialised, and unlike the viewer it needs no construction
    change at all -- the offscreen context is created lazily at first dump.
    """

    def __init__(self, base_ref: str | None = None):
        from harness.registry import load_provider

        # Read the module global at CALL time, not def time (the binding trap).
        self._base = load_provider(base_ref or _BASE_REF)

    def make_env(self, spec):
        return _FrameEnv(self._base.make_env(spec))

    def tasks(self):
        return self._base.tasks()

    def object_key(self, spec):
        return self._base.object_key(spec)

    def success(self, obs, spec, start_z):
        return self._base.success(obs, spec, start_z)

    def __getattr__(self, name):  # terminal_success/close/...
        return getattr(self._base, name)


def frames_provider():
    return FrameEmbodiment()


# Inverted registration, once per process (module import is cached): harness/
# imports no scripts, so the capture layer wires ITSELF onto the event stream.
opstream.on_emit(keyframe)
opstream.on_emit(_video_event)
