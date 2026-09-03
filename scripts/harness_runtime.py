#!/usr/bin/env python3
"""The resident runtime: one boot, a persistent session chain, a fresh kernel
per task, and a watched inbox of JSON briefs.

    PYTHONPATH=. .venv/bin/python scripts/harness_runtime.py \
        --session-dir runs/session-main
    (add --drain to process whatever is pending and exit)

M4's system layer. The three durable things vary at three rates, so they split
three ways (round-94 design):

- ONE long-lived ``SessionLog`` under ``<session-dir>/session-log/``, ``load()``-
  or-fresh at boot -- never ``SessionLog(dir)`` over an existing ledger (that
  ``FileExistsError``s; events.py:49-55). Every task appends to this one chain,
  so "restart-resumable + verifiable" is just reopening the file.
- a FRESH ``Kernel`` per task on that shared log. The kernel is single-mount and
  its ``policy.driver`` is chosen BY TASK, so it cannot be reused across
  stack/clear_table; a fresh ``Kernel(CAPABILITIES, log=shared_log)`` per task is
  cheap (dicts) and every kernel's mounts/resolutions land in the one chain.
- a SHARED skills dir as the ``graph.skill`` root. RSI (rung 3) writes content-
  addressed records there; the next task sees them because its FRESH graph.skill
  mount re-globs the root at ``__init__`` -- fresh-kernel-per-task gives this for
  free.

Intake is a watched directory: an external writer drops ``inbox/<id>.json`` with
``os.replace`` (atomic, so the runtime never reads a half-written file); the
runtime claims by ``os.rename`` into ``processing/`` (the loser of a race gets
``FileNotFoundError`` and moves on), runs it, then ``os.replace``s the file into
``done/``, ``failed/`` or ``cancelled/``. On boot any ``processing/*.json`` left
by a crash is re-queued to ``inbox/`` (at-least-once) -- at most ``_MAX_REQUEUES``
times, after which the brief is poison and goes to ``failed/`` under its own
``runtime.task_error`` row, because a brief that kills the process re-queues into
the next boot and the only symptom is a runtime that will not stay up.

Cancellation is COOPERATIVE and one-directional: ``board.store.cancel_brief``
drops a marker in ``cancel/<brief-id>`` and stops. This runtime is the only
thing that acts on one -- at the claim, at each node boundary inside the
workload, and on a 2s probe while a campaign/rsi subprocess runs (killed by
process GROUP, so a worker pool leaves no orphans). It ends in ``cancelled/``
under a ``runtime.task_cancelled`` row, never ``runtime.task_error``: an
operator's stop and a system crash must stay separable in the evidence.

ONE runtime per session dir, enforced here: ``boot`` takes an exclusive flock on
``<session-dir>/runtime.lock`` and a second instance refuses to start, naming the
pid that holds it (``_claim_session``). The guard lives in the guarded thing, not
in whatever launched it -- scripts/cockpit's adopt-or-spawn scan keeps the normal
path off this rail, but the operator starting a runtime by hand deserves the same
protection.

Authority-laundering defense (non-negotiable): a brief names NO provider/mount
ref. It is a selector+budgets, plus an optional task-only natural-language
``instruction`` -- ``{"kind":"task","task":"stack","seed":90000,
"max_replans":3,"max_actuations":3}``. The runtime resolves the task STRING
to its policy/planner/catalogue/oracles through the UNION of installed plugin
manifests at boot (``harness.manifest.discover``) and stamps them server-side;
adding a task is installing a plugin dir (a filesystem act), never a brief. A
manifest declaring ``actuation:real`` is refused at boot -- a real-actuator
embodiment is a DIFFERENT authenticated runtime, never a brief (nor a card) away.

Crash-safety lives HERE, not in the well-tested workload: ``workload.run`` stays
loud-as-data for planning faults and raises for structural ones; the loop wraps
its call in try/except and writes its own ``runtime.task_error`` note on any
escape, then continues. A single task's failure never kills the system.

This wires plugins + profiles together, so it lives in scripts/ beside its
sibling closed-loop driver scripts/task_plan.py -- harness/ imports no plugin
(tests/test_kernel.py) and profiles/ stays declarative (tests/test_boundaries.py).
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from board.store import burned_blocks, cancelled_run
from harness import opstream
from harness.config import Mount, Patch, resolve_plan
from harness.definitions import CAPABILITIES
from harness.events import SessionLog
from harness.kernel import Kernel
from harness import predicates, protocol
from harness.manifest import discover
from harness.registry import load_provider
from harness.skill_record import skill_index
from plugins.graphs import InMemorySkillGraph
from plugins.task import workload
from profiles import base_profile
from scripts.brief_drop import drop

#: The render overlay: the generalised viewer wrapper (scripts/watch_stack.py),
#: mounted on ``embodiment.env`` per task when ``--render`` is set. Param-free on
#: purpose -- the workload refuses Mount params on embodiment.env (they cannot
#: ride an EpisodeSpec ref), so the base ref it wraps + pacing travel via the
#: watch_stack module globals _prepare_render arms (in-process only).
RENDER_ENV_REF = "scripts.watch_stack:viewer_provider"
#: watch_stack's default per-step pacing is 0.02s of extra sleep; 1/50 == 0.02.
DEFAULT_RENDER_FPS = 50.0

#: The frames overlay: the offscreen-viewport wrapper (scripts/frame_dump.py),
#: mounted OUTERMOST on ``embodiment.env`` per task when frames are on -- it
#: wraps the sim override and the --render viewer alike (the base ref it wraps
#: travels via frame_dump module globals, same in-process channel as watch_stack).
FRAMES_ENV_REF = "scripts.frame_dump:frames_provider"


def _load_attr(ref: str):
    """Import ``module:attr`` and return the ATTRIBUTE (not call it).

    A task binding names its catalogue/oracles by ref -- data authored on the
    skill side (``type`` objects, not JSON), so they cannot ride a brief. This is
    the read half of the same ``module:attr`` crossing ``load_provider`` uses for
    factories, minus the call: the attribute IS the value.
    """
    module_name, attr = ref.split(":", 1)
    return getattr(importlib.import_module(module_name), attr)


#: The two session modes. EXECUTION is the fail-safe default: a real task run
#: never triggers RSI. EVOLUTION is the only mode that may accept a campaign
#: brief and let a campaign write the shared skills root.
MODES = ("execution", "evolution")

#: The session claim. One runtime per session dir, enforced HERE rather than in
#: whatever launched it: the guard belongs inside the thing it guards, and the
#: operator starts runtimes by hand as often as scripts/cockpit does.
LOCKFILE = "runtime.lock"

#: Claims this process already holds, by resolved session dir. flock() is keyed
#: to the OPEN FILE DESCRIPTION, so a second open()+flock() of the same file in
#: one process contends with itself; re-booting a session we already hold (a
#: --drain then serve, the reboot tests) must be a no-op, not a self-refusal.
_CLAIMED: dict[Path, int] = {}


def _claim_session(session_dir: Path) -> None:
    """Take the exclusive, non-blocking flock on ``<session-dir>/runtime.lock``.

    Two runtimes on one session dir cannot corrupt the inbox (the claim is an
    atomic rename) but they double-run briefs, interleave two writers into one
    session chain and fight over runtime_status.json/frame.jpg. So the second
    one refuses to start and names the pid holding the claim.

    flock, deliberately, not "the lock file exists": the kernel drops the
    advisory lock when the fd closes, including on SIGKILL and on a crash, so
    there is no zombie claim to clear by hand -- the file itself is allowed to
    outlive its holder and means nothing on its own. The pid written inside is
    a courtesy for the error message, never the lock.
    """
    key = session_dir.resolve()
    if key in _CLAIMED:
        return
    fd = os.open(key / LOCKFILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = os.read(fd, 32).decode("utf-8", "replace").strip() or "unknown"
        os.close(fd)
        raise SystemExit(
            f"harness_runtime: session {session_dir} is already served by pid "
            f"{holder} (its flock on {key / LOCKFILE} is held). Refusing to "
            "start a second runtime on one session dir -- stop that one first, "
            "or point --session-dir somewhere else.") from None
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    _CLAIMED[key] = fd   # held for the life of the process; closing releases it


@dataclass(frozen=True)
class Runtime:
    """The booted state: the four intake dirs, the shared skills root, the log.

    ``mode`` is fixed at first boot (write-once MODE file) and immutable across
    restart. ``skills_manifest`` is the sorted content-digest stem set sealed in
    the ``runtime.boot`` row -- the baseline the per-execution-task immutability
    audit folds against, so an out-of-band skills-root mutation is machine-caught.
    """

    inbox: Path
    processing: Path
    done: Path
    failed: Path
    skills_root: Path
    log: SessionLog
    mode: str = "execution"
    #: Per-BOOT operator flag (never a brief, never MODE): overlay the rendering
    #: env provider on each task so the operator watches the MuJoCo window live.
    #: Composes with mode; campaigns spawn a headless subprocess regardless.
    render: bool = False
    #: Per-BOOT operator flag, render's remote-browser sibling: overlay the
    #: frame-dump env provider on each task so the ph-station 取景窗 shows the
    #: sim. Auto-on for a headless-GL CLI boot; composes with render and mode.
    frames: bool = False
    skills_manifest: tuple[str, ...] = ()
    #: task STRING -> {policy, planner, catalogue, oracles} and campaign name ->
    #: script, both folded from the installed manifests at boot -- the
    #: authority-laundering allowlists, no longer welded into this module.
    task_bindings: dict[str, dict] = field(default_factory=dict)
    campaigns: dict[str, str] = field(default_factory=dict)
    #: benchmark name -> its pure-data card ({tasks, arms, max_replans, ...}).
    benchmarks: dict[str, dict] = field(default_factory=dict)


def _skills_manifest(skills_root: Path) -> list[str]:
    """Sorted content-digest stems under the skills root. Filenames already ARE
    content digests, so this list IS the skill set -- set-equality on it is the
    immutability check, no rehash needed."""
    return sorted(f.stem for f in skills_root.glob("*.json"))


#: How many times a crash may re-queue ONE brief before it is filed as poison.
#: The crash-recovery re-queue is at-least-once and was unbounded: a brief that
#: takes the whole process down -- a segfaulting sim, an OOM kill, anything the
#: try/except in _process cannot catch because the interpreter never gets to run
#: it -- came back on every boot and killed the runtime again, forever, and the
#: only visible symptom was a session that would not stay up.
#: 2 re-queues = 3 attempts. ponytail: the ceiling is per-brief and lifetime, so
#: three OPERATOR restarts during one long brief also spend it; that is loud
#: (failed/ + a chain row naming the count) and re-submittable, which beats an
#: invisible boot loop. Make it a decaying window only if a real run trips it.
_MAX_REQUEUES = 2
#: Where the counts live. NOT in the brief: _BRIEF_KEYS rejects any key it does
#: not know, so stamping an attempt counter inside the JSON would make every
#: re-queued brief hard-fail as an injected key. Live state at the session root,
#: same family as runtime_status.json, never a chain row.
REQUEUE_FILE = "requeue.json"


def _requeue(processing: Path, inbox: Path, failed: Path, session_dir: Path,
             log: SessionLog) -> None:
    """Crash recovery: re-queue every stranded ``processing/*.json``, but only
    ``_MAX_REQUEUES`` times each; the next one is filed under ``failed/`` with a
    ``runtime.task_error`` row so ``brief_status`` reports it as an outcome
    instead of the brief silently reappearing at the head of the queue forever.

    The counter map is rebuilt from what is stranded RIGHT NOW, so a brief that
    crashed once and later finished drops out of it -- no reaper, no growth.
    """
    try:
        counts = json.loads((session_dir / REQUEUE_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        counts = {}
    fresh: dict[str, int] = {}
    for p in sorted(processing.glob("*.json")):
        _reap_orphan(_pgid_marker(processing, p.name), p.name, log)
        n = int(counts.get(p.name, 0)) + 1
        if n > _MAX_REQUEUES:
            log.append("runtime.task_error", {
                "brief": p.name, "task": None,
                "error": f"crash-loop: re-queued {_MAX_REQUEUES} times and the "
                         "runtime died holding it again each time; filed as "
                         "poison rather than re-queued once more"})
            os.replace(p, failed / p.name)
            continue
        fresh[p.name] = n
        os.replace(p, inbox / p.name)
    drop(session_dir, REQUEUE_FILE, json.dumps(fresh))


def _pgid_marker(processing: Path, brief_id: str) -> Path:
    """Live state next to the claimed brief: the pgid of its campaign/rsi group.
    Not ``*.json``, so the requeue glob never mistakes it for a brief."""
    return processing / (brief_id + ".pgid")


def _group_alive(pgid: int) -> bool:
    """Any non-zombie process whose pgrp (/proc/*/stat field 5) is ``pgid``."""
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            state, _, pgrp = stat.read_text().rsplit(")", 1)[1].split()[:3]
            if state != "Z" and int(pgrp) == pgid:
                return True
        except (OSError, ValueError):
            continue
    return False


def _stop_group(pgid: int, proc: subprocess.Popen | None = None) -> None:
    """SIGTERM a campaign/rsi group, SIGKILL whatever is still alive after
    ``_CANCEL_GRACE_S``. ``proc`` (when this runtime is the parent) is drained
    and reaped at the end so it never lingers as a zombie."""
    for sig, wait in ((signal.SIGTERM, _CANCEL_GRACE_S), (signal.SIGKILL, 1.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + wait
        while _group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if not _group_alive(pgid):
            break
    if proc is not None:
        proc.communicate()


def _reap_orphan(marker: Path, brief_id: str, log: SessionLog) -> None:
    """A campaign/rsi group is its OWN session (never in the runtime's group), so a
    runtime that died holding a brief (SIGKILL, crash, OOM) leaves it running and
    still writing campaigns/<stem>. Kill-then-requeue: the re-run starts clean
    instead of racing a ghost writer into the same store (2026-08-28 shape); an
    operator who would rather refuse the requeue can add that later."""
    if not marker.exists():
        return
    pgid = int(marker.read_text())
    if _group_alive(pgid):
        log.append("runtime.orphan_killed", {"brief": brief_id, "pgid": pgid})
        _stop_group(pgid)
    marker.unlink()


def _prepare_render(fps: float) -> None:
    """Sanity-check the GL env for a live MuJoCo window and arm the viewer overlay.

    round-80 lesson: ``MUJOCO_GL=egl`` is HEADLESS -- a window needs native GL, so
    an egl/osmesa setting is unset here (loud, never a silent headless fallback).
    No ``$DISPLAY`` = refuse the whole boot. Sets watch_stack's module globals (the
    base env ref to wrap + per-step pacing) -- the in-process channel the task
    plan loop reads; NOT spawn-safe, which is why campaigns stay headless.
    """
    from scripts import watch_stack

    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "--render needs an X DISPLAY for the MuJoCo window, but $DISPLAY is "
            "unset; run on the workstation display (e.g. DISPLAY=:1), not headless")
    gl = os.environ.get("MUJOCO_GL", "").lower()
    if gl in ("egl", "osmesa"):
        print(f"harness_runtime: --render unsetting MUJOCO_GL={gl!r} "
              "(headless GL; a live window needs native GL)", file=sys.stderr)
        os.environ.pop("MUJOCO_GL", None)
    watch_stack._DELAY = 1.0 / fps if fps > 0 else 0.0
    watch_stack._RENDER_BASE_REF = resolve_plan(base_profile()).ref("embodiment.env")


def boot(session_dir: str | Path, inbox: str | Path | None = None, *,
         mode: str = "execution", render: bool = False,
         render_fps: float = DEFAULT_RENDER_FPS, frames: bool = False) -> Runtime:
    """Load-or-fresh the session chain, make the intake dirs, re-queue crashes.

    The session's ``mode`` is written once to ``<session-dir>/MODE`` at first
    boot and asserted on every re-boot -- a mismatched ``--mode`` is refused, not
    overwritten (write-once). Boot also seals a ``runtime.boot`` row
    ``{mode, skills_manifest, mount_plan_sha, render}``: row 0 of a fresh chain, or -- for
    a session that predates MODE -- retrofitted at the tail with a ``migrated``
    marker (appending never rewrites the existing chain, so no sealed sha moves).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if render:
        _prepare_render(render_fps)  # refuses loudly with no $DISPLAY, before any dir work
    session_dir = Path(session_dir)
    log_dir = session_dir / "session-log"
    skills_root = session_dir / "skills"
    inbox = Path(inbox) if inbox is not None else session_dir / "inbox"
    # processing/done/failed are SIBLINGS of inbox on purpose: os.rename is only
    # atomic within one filesystem, and the claim is a rename inbox->processing.
    processing = inbox.parent / "processing"
    done = inbox.parent / "done"
    failed = inbox.parent / "failed"
    # cancelled/ is the fourth ending, a sibling for the same rename reason: an
    # operator stop must be distinguishable from a crash forever after.
    # cancel/ is its live-state inbox -- board.store.cancel_brief drops a marker
    # there and this runtime is the only thing that acts on one.
    cancelled = inbox.parent / "cancelled"
    cancel = inbox.parent / "cancel"
    for d in (log_dir, skills_root, inbox, processing, done, failed,
              cancelled, cancel):
        d.mkdir(parents=True, exist_ok=True)

    # Claim the session BEFORE the first mutation below (the crash re-queue
    # moves files), so a refused second runtime has touched nothing.
    _claim_session(session_dir)

    # write-once mode: fixed at first boot, immutable across restart.
    mode_file = session_dir / "MODE"
    if mode_file.exists():
        recorded = mode_file.read_text().strip()
        if recorded != mode:
            raise ValueError(
                f"{mode_file} is {recorded!r}; refusing to re-boot as {mode!r} "
                "(MODE is write-once)")
    else:
        mode_file.write_text(mode)

    log = (SessionLog.load(log_dir) if (log_dir / "rows.jsonl").exists()
           else SessionLog(log_dir))

    # restart-resume: anything claimed but not finished goes back to inbox,
    # BOUNDED (a brief that kills the runtime would otherwise re-queue forever).
    # ponytail: at-least-once -- a task that crashed after its plan_complete note
    # re-runs and appends a second note; dedup by brief-id only if soak shows it
    # hurts.
    _requeue(processing, inbox, failed, session_dir, log)

    # boot seal: the immutability baseline comes from the SEALED row (catches a
    # skills-root mutation that happened between boots), not a fresh recompute.
    # A fresh session seals row 0; a pre-MODE session gets it retrofitted at the
    # tail (marked migrated) since row 0 is already spoken for.
    boot_row = next((r for r in log.rows() if r["kind"] == "runtime.boot"), None)
    if boot_row is None:
        manifest = _skills_manifest(skills_root)
        seal = {"mode": mode, "skills_manifest": manifest,
                "mount_plan_sha": resolve_plan(base_profile()).sha(),
                "render": render}
        if log.rows():
            seal["migrated"] = True  # predates MODE; not row 0
        log.append("runtime.boot", seal)
        baseline = tuple(manifest)
    else:
        baseline = tuple(boot_row["data"]["skills_manifest"])

    # ONE read of the skills root, feeding both consumers below. Both modes now:
    # the index has to describe the live library in evolution too, and a corrupt
    # record raising at boot rather than at the first task's graph.skill mount is
    # the direction this file already prefers (loud and early).
    records = InMemorySkillGraph(root=str(skills_root)).skills()

    # Boot-time refusal of a record that cannot be assembled into the governance
    # bundle it would steer under. Corrupt JSON already raised in
    # InMemorySkillGraph(...) at mount; this second layer catches valid-JSON-but-
    # unassemblable records (missing task/preconditions/recovery, unknown feature,
    # non-invertible program) by DRY-RUNNING the exact task-time path
    # (workload.assemble_bundle -> rule_from_canonical + privilege_cost) over each
    # task present -- so "assemblable at boot" <=> "assemblable at task time" by
    # construction, and the execution-mode skills-root immutability audit keeps the
    # set fixed in between. Any exception here refuses the boot, never mid-episode.
    # Execution only: evolution's records are campaign-produced (well-formed by
    # construction) and inert until a later execution boot validates them.
    if mode == "execution":
        for task in sorted({r.get("task") for r in records if r.get("task") is not None}):
            workload.assemble_bundle(records, task)

    # The planner's one-read view of the skill library, DERIVED from the records
    # this boot just mounted -- same read that fed the manifest seal above, so it
    # cannot describe a library other than the one in force. Live-state family
    # (session root, overwritten every boot, never a chain row): a SEALED index
    # would be a second copy of the truth, free to drift while nothing notices.
    drop(session_dir, "skill_index.json",
         json.dumps(skill_index(records), sort_keys=True, indent=1))

    # Live operational status: overwritten on EVERY boot (fresh and resumed).
    # NOT a chain row -- render is live runtime state, not sealed evidence, so it
    # cannot ride the write-once runtime.boot seal (which is blank for sessions
    # that predate it). Sits at the session root, so chain verification (session-
    # log/rows.jsonl only) never sees it. Written atomically (the shared
    # brief_drop temp+replace) so a poll never reads a half write;
    # ``heartbeat_ts`` starts at boot and is re-stamped by the poll loop
    # (_heartbeat) so a reader can tell a live runtime from a dead one.
    now = time.time()
    drop(session_dir, "runtime_status.json", json.dumps({
        "pid": os.getpid(), "render": render, "frames": frames, "mode": mode,
        "boot_ts": now, "heartbeat_ts": now,
        "display": os.environ.get("DISPLAY")}))

    # Same live-state family: arm (or disarm) the frames overlay's destination.
    # runs/<session>/frame.jpg is overwritten in place, never a chain row; a
    # failing dump is swallowed inside frame_dump.dump, so a broken GL stack
    # loses the viewport, never a task.
    from scripts import frame_dump
    frame_dump.arm(session_dir / "frame.jpg" if frames else None)

    # Same live-state family: the operational event feed the execution-graph
    # panel animates. Truncated per boot (this boot IS the feed's horizon);
    # arm() failure leaves it unarmed and the runtime unharmed.
    opstream.arm(session_dir / "runtime_events.jsonl")
    opstream.emit("boot", pid=os.getpid(), mode=mode, render=render)

    # The authority allowlists are the manifest union at boot (discover() raises
    # on an actuation:real card, so the sim runtime refuses to boot with one).
    registry = discover()
    return Runtime(inbox, processing, done, failed, skills_root, log, mode,
                   render, frames, baseline, registry.task_bindings,
                   registry.campaigns, registry.benchmarks)


def _mount_plan(binding: dict, skills_root: Path, render: bool = False,
                frames: bool = False):
    """A fresh MountPlan for this task: base profile + the task's sim policy +
    planner + the shared skills root (so a fresh graph.skill mount re-globs RSI's
    output). policy and planner come from the task binding (a manifest), so
    swapping either is a card edit, never a base one.

    ``render`` overlays the viewer wrapper (RENDER_ENV_REF) on ``embodiment.env``
    around whatever the base plan resolved -- the same governed_rollout code path,
    plus a window. Param-free (the workload refuses env mount params); the base ref
    it wraps rides watch_stack's module globals _prepare_render armed at boot."""
    override = [
        # PlanRecords first (planner_library wraps the card's planner by ref):
        # a mounted plan for (task, embodiment, arm) replays without a model call.
        Mount("task.planner", _LIBRARY_PLANNER_REF, {"inner": binding["planner"]}),
        Mount("policy.driver", binding["policy"]),
        Mount("graph.skill", "plugins.graphs:skill_graph_provider",
              {"root": str(skills_root)}),
    ]
    # A mission riding a DIFFERENT simulator names its embodiment.env / percept.model
    # by ref in its binding: the robocasa card is enabled=false (kept OUT of the base
    # fold so its second embodiment.env claim never collides with robosuite's), so a
    # per-session override IS how that second sim mounts. Single-sim bindings omit
    # these keys -> the base-folded robosuite mounts stand, byte-identical.
    for cap, key in (("embodiment.env", "env"), ("percept.model", "percept")):
        if key in binding:
            override.append(Mount(cap, binding[key]))
    if render:
        # ponytail: render re-overrides embodiment.env to the viewer wrapper, so it
        # wins over a sim override -- render+second-sim is unsupported (E2E is egl
        # headless). Wrap the sim ref here only once a windowed second sim is needed.
        override.append(Mount("embodiment.env", RENDER_ENV_REF))
    if frames:
        # OUTERMOST overlay: the frame dump wraps whatever ref stands after the
        # sim/render overrides -- so the 取景窗 watches ANY task on ANY sim, with
        # or without a window. The wrapped ref rides frame_dump's module global
        # (in-process only, same channel as watch_stack's _RENDER_BASE_REF).
        from scripts import frame_dump
        frame_dump._BASE_REF = (
            RENDER_ENV_REF if render
            else binding.get("env",
                             resolve_plan(base_profile()).ref("embodiment.env")))
        override.append(Mount("embodiment.env", FRAMES_ENV_REF))
    return resolve_plan(base_profile(), patches=(
        Patch("runtime", override=tuple(override)),))


def task_brief(task: str, binding: dict) -> dict:
    """The ``plugins.task.workload`` brief for one task binding.

    The ONE place a manifest binding becomes a workload brief: catalogue/oracles
    always, PREDICATES for a heterogeneous mission, the episode block for a
    persistent one -- every value reached by ref, never carried in a JSON brief.
    Shared by the resident runtime and by the generic RSI probe
    (scripts/rsi_campaign.py), so a calibration set runs the byte-identical brief
    a live task run does; a second copy of this assembly is exactly the drift that
    would make a calibration measure a different mission than the one that ships.
    """
    wbrief = {"task": task, "catalogue": _load_attr(binding["catalogue"]),
              "oracles": _load_attr(binding["oracles"]),
              # The embodiment a task.plan row / PlanRecord is keyed by: the
              # binding's env ref (base-folded robosuite unless the card names
              # one), never a render/frames overlay that wraps it for a session.
              "embodiment": binding.get("env", resolve_plan(base_profile()).ref("embodiment.env"))}
    # Optional planner-only, server-authored context. The natural-language
    # instruction may be supplied by the task brief, but the skill semantics and
    # scene inventory remain manifest refs -- a caller cannot redefine either.
    for key in ("skill_docs", "planning_context", "default_instruction", "records",
                "initial_facts"):
        if key in binding:
            wbrief[key] = _load_attr(binding[key])
    # A heterogeneous mission (perceive/decide/verify nodes) declares a PREDICATES
    # table by ref beside catalogue/oracles; thread it so the loop can resolve each
    # kindful node's machine oracle. Manipulate-only bindings omit it.
    if "predicates" in binding:
        wbrief["predicates"] = _load_attr(binding["predicates"])
    # A persistent-episode mission (M7) opts in with episodic=true and names its
    # ONE-episode block + per-sub-goal segment_specs by ref beside the rest; thread
    # them so workload.run opens the single world and drives each segment in it.
    # Non-episodic bindings omit these -- the fresh-per-node path, byte-identical.
    if binding.get("episodic"):
        wbrief["episodic"] = True
        wbrief["episode"] = _load_attr(binding["episode"])
        wbrief["segment_specs"] = _load_attr(binding["segment_specs"])
    return wbrief


def _run_task(brief: dict, rt: Runtime, cancelled=None) -> dict:
    """Build a fresh kernel on the shared log and run one governed plan loop.

    The task STRING resolves to its binding through the installed manifests; an
    unknown task (no card declares it) is refused HERE, before any mount -- a
    brief cannot conjure a task no plugin provides. catalogue/oracles are
    skill-authored ``type`` objects imported by ref from the binding, never
    carried in the JSON brief.

    ``cancelled`` is the operator's stop probe, threaded into the workload's node
    loop: a zero-arg predicate the loop calls at each NODE BOUNDARY. The workload
    knows nothing about markers or directories -- the seam is one callable.
    """
    task = brief["task"]
    binding = rt.task_bindings.get(task)
    if binding is None:
        raise ValueError(f"no task binding for {task!r}; install a plugin that "
                         f"declares it (known: {sorted(rt.task_bindings)})")
    seed = int(brief.get("seed", 0))
    max_replans = int(brief.get("max_replans", 3))
    # Actuation floor: brief -> the card's task binding (every mission card
    # declares one to cover its node count) -> the one-node default.
    max_actuations = int(brief.get("max_actuations", binding.get("max_actuations", 3)))
    kernel = Kernel(CAPABILITIES, log=rt.log)
    kernel.mount(_mount_plan(binding, rt.skills_root, render=rt.render,
                             frames=rt.frames))
    wbrief = task_brief(task, binding)
    if "instruction" in brief:
        wbrief["instruction"] = brief["instruction"]
    # Executor arm (skill_library.ARMS; workload refuses an unknown one): pi05
    # runs each segment its record binds to the policy card, scripted elsewhere.
    wbrief["arm"] = brief.get("arm", "scripted")
    # Segment clips (harness.media): opt-in on a task brief, on for suite/evolve.
    if brief.get("media"):
        wbrief["media_dir"] = str(rt.inbox.parent / "media")
    return workload.run(wbrief, kernel, seed=seed,
                        max_replans=max_replans, max_actuations=max_actuations,
                        segment_retries=int(binding.get("segment_retries", 0)),
                        cancelled=cancelled)


#: The mission decomposer: the VLM card's ``decompose`` over the SAME model
#: seam its plan() uses. Server-side configuration, never a brief key; the ref
#: is sealed in every ``mission.decomposed`` row. PH_MISSION_DECOMPOSER lets a
#: GPU-less e2e route it to the fake endpoint the way test cards route planners.
_DECOMPOSER_REF = os.environ.get("PH_MISSION_DECOMPOSER", "plugins.planner_vlm:provider")
_LIBRARY_PLANNER_REF = "plugins.planner_library:provider"


def _binding_records(binding: dict) -> dict[str, protocol.SkillRecordV0]:
    recs = _load_attr(binding["records"]) if "records" in binding else {}
    return {k: v if isinstance(v, protocol.SkillRecordV0) else protocol.SkillRecordV0.from_dict(v)
            for k, v in recs.items()}


def _known_tasks(bindings: dict[str, dict]) -> list[dict]:
    """The decomposer's task menu: every task binding with its goal preds -- the
    card's declared ``goal`` ref when present, else the records' FINAL ensures
    (ensured by some skill, required by none: the effects a task is for)."""
    out = []
    for task, binding in sorted(bindings.items()):
        recs = _binding_records(binding)
        if "goal" in binding:
            goal = [protocol.pred_ref_str(g) for g in _load_attr(binding["goal"])]
        else:
            ens = {p for r in recs.values() for p in r.ensures}
            req = {p for r in recs.values() for p in r.requires}
            goal = sorted(ens - req)
        desc = _load_attr(binding["default_instruction"]) if "default_instruction" in binding else ""
        out.append({"task": task, "goal": goal, "description": str(desc)})
    return out


def _predicate_catalogue(bindings: dict[str, dict]) -> dict[str, tuple[str, ...]]:
    """name -> arg names: the registered predicate cards plus every predicate the
    bindings' records mention (its template args)."""
    cat = {name: tuple(rec.args) for name, rec in predicates.records().items()}
    for binding in bindings.values():
        for r in _binding_records(binding).values():
            for p in (*r.requires, *r.ensures, *r.clobbers):
                name, args = protocol.parse_pred_ref(p)
                cat.setdefault(name, tuple(args))
    return cat


def _compose(mission: str, parts: list[tuple[dict, Mapping]], planner: dict) -> dict:
    """ONE plan-dialect graph from per-task plans: node ids namespaced by task id,
    every node labelled with its task, verify entries carried along."""
    tasks, nodes, verify = [], [], []
    for t, plan in parts:
        tid = t["id"]
        tasks.append({"id": tid, "goal": list(t["goal"])})
        for n in plan.get("nodes") or ():
            nodes.append({**n, "id": f"{tid}.{n['id']}", "task": tid,
                          "after": [f"{tid}.{a}" for a in n["after"]]})
        verify += [{**v, "after": f"{tid}.{v['after']}"} for v in plan.get("verify") or ()]
    return {"goal": mission, "tasks": tasks, "nodes": nodes, "verify": verify,
            "rationale": "; ".join(f"{t['id']}: {p.get('rationale', '')}" for t, p in parts),
            "planner": planner}


def _run_mission(brief: dict, rt: Runtime, cancelled=None) -> dict:
    """Natural-language mission -> decompose (sealed) -> per-task plan (PlanRecord
    library first, the binding's planner otherwise) -> ONE composed graph ->
    the ordinary workload run path (Legal(G) with real goals gates dispatch).

    ponytail: every decomposed task must share one binding's env/policy (a
    composed graph runs in ONE kernel); split into per-task kernels when a
    mission first spans two simulators."""
    mission, seed = str(brief["mission"]), int(brief.get("seed", 0))
    arm = brief.get("arm", "scripted")
    known = _known_tasks(rt.task_bindings)
    catalogue = _predicate_catalogue(rt.task_bindings)
    objects: set[str] = set()
    for task, binding in rt.task_bindings.items():
        tb = task_brief(task, binding)
        objects |= set(workload._sigma0(tb, {}, {}, _binding_records(binding))[1])
    decomposer = load_provider(_DECOMPOSER_REF, {})
    try:
        dec = decomposer.decompose({"mission": mission, "known_tasks": known,
                                    "predicates": catalogue, "objects": sorted(objects)})
        for t in dec["tasks"]:
            if t.get("task") is None:
                raise ValueError(f"task {t['id']!r} names no known task binding; "
                                 "the runtime can only plan under a binding")
    except ValueError as exc:
        rt.log.append("mission.refused", {"mission": mission, "seed": seed,
                                          "decomposer": _DECOMPOSER_REF, "error": str(exc)})
        raise
    rt.log.append("mission.decomposed", {
        "mission": mission, "seed": seed, "decomposer": _DECOMPOSER_REF,
        "tasks": dec["tasks"], "prompt_sha": dec["prompt_sha"],
        "rationale": dec["rationale"]})
    bindings = [rt.task_bindings[t["task"]] for t in dec["tasks"]]
    first = bindings[0]
    for b in bindings[1:]:
        for key in ("env", "policy", "percept", "episodic"):
            if b.get(key) != first.get(key):
                raise ValueError(f"mission tasks disagree on binding {key!r}: "
                                 f"{first.get(key)!r} != {b.get(key)!r}")
    max_actuations = int(brief.get("max_actuations", sum(
        int(b.get("max_actuations", 3)) for b in bindings)))
    kernel = Kernel(CAPABILITIES, log=rt.log)
    kernel.mount(_mount_plan(first, rt.skills_root, render=rt.render, frames=rt.frames))
    plans = [r for r in kernel.resolve("graph.skill", consumer="mission").skills()
             if r.get("kind") == "plan"]
    wbrief = task_brief(dec["tasks"][0]["task"], first)
    parts, planners = [], {}
    for t, binding in zip(dec["tasks"], bindings):
        tb = {**task_brief(t["task"], binding), "arm": arm, "seed": seed, "scene": {},
              "budget": max_actuations, "plans": plans,
              "instruction": f"{mission} -- {t['task']}: achieve {', '.join(t['goal'])}"}
        recs = workload._records(tb, tb["catalogue"])
        tb["facts"], tb["objects"] = workload._sigma0(tb, {}, {}, recs)
        plan = load_provider(_LIBRARY_PLANNER_REF, {"inner": binding["planner"]}).plan(tb)
        parts.append((t, plan))
        planners[t["id"]] = plan.get("planner") or {"provider": binding["planner"]}
        for key in ("catalogue", "records", "skill_docs", "predicates"):
            if key in tb:
                wbrief[key] = {**(wbrief.get(key) or {}), **tb[key]}
        for key in ("oracles", "initial_facts"):
            if key in tb:
                wbrief[key] = tuple(dict.fromkeys((*(wbrief.get(key) or ()), *tb[key])))
    wbrief["arm"] = arm
    wbrief["graph"] = _compose(mission, parts, {
        "provider": "mission", "decomposer": _DECOMPOSER_REF,
        "prompt_sha": dec["prompt_sha"], "tasks": planners})
    return workload.run(wbrief, kernel, seed=seed,
                        max_replans=int(brief.get("max_replans", 3)),
                        max_actuations=max_actuations,
                        segment_retries=int(first.get("segment_retries", 0)),
                        cancelled=cancelled)


def _runs_root(rt: "Runtime") -> Path:
    """The runs/ tree the derived seed ledger is read from: ``runs/<session>/inbox``
    -> ``runs``. Sealed preregistrations anywhere below it burn their blocks."""
    return rt.inbox.parent.parent


def _declared_ranges(brief: dict) -> list[tuple[int, int]]:
    """dev∪heldout the brief declares it will burn, as inclusive [lo,hi] pairs."""
    ranges = []
    for key in ("dev", "heldout"):
        for pair in brief.get(key, ()):
            lo, hi = int(pair[0]), int(pair[1])
            ranges.append((min(lo, hi), max(lo, hi)))
    return ranges


def _assert_unburned(brief: dict, what: str, runs_root: Path,
                     allow_empty: bool = False) -> None:
    """The seed-ledger guard (non-negotiable invariant): the burned set DERIVED
    from every sealed preregistration (board.store.burned_blocks) is one enforced
    check at the scheduling boundary. Reject BEFORE spawning if the declared
    dev∪heldout intersects any burned range (inclusive intervals). No store at
    all -> refuse (an absent ledger is not an empty one); a brief that declares
    no dev/heldout (calibration only) never consults the ledger.

    Shared by the campaign and rsi paths. An rsi brief usually declares nothing
    (the chain allocates), so its scheduler fills dev/heldout in from the
    allocation and calls this with the SAME ``_declared_ranges`` reader -- the
    guard is never routed around, only fed.
    """
    declared = _declared_ranges(brief)
    if not declared:
        return
    try:
        burned = burned_blocks(runs_root)
    except ValueError:
        # allow_empty: the caller is about to seal the FIRST prereg under this
        # runs root (a suite on a fresh session), so an absent ledger is empty.
        if not allow_empty:
            raise
        burned = []
    for lo, hi in declared:
        for blo, bhi, role, sha in burned:
            if lo <= bhi and blo <= hi:
                raise ValueError(
                    f"seed-ledger overlap: {what} declares [{lo},{hi}] "
                    f"which hits burned {role} [{blo},{bhi}] (prereg {sha[:12]})")


def _copy_skills(src: Path, dst: Path) -> list[str]:
    """Fold a campaign's published skill records into the shared graph.skill root
    (idempotent -- the filename stem IS the content digest, so a record already
    in the root is skipped). Returns every digest now present from this run."""
    copied = []
    for f in (sorted(src.glob("*.json")) if src.is_dir() else ()):
        if not (dst / f.name).exists():
            shutil.copy2(f, dst / f.name)
        copied.append(f.stem)
    return copied


def _prereg_sha(out: Path) -> str | None:
    """The campaign's preregistration content hash, read back from its store
    index (run_campaign puts it as row 0). None if the store wrote none."""
    index = out / "index.jsonl"
    if not index.exists():
        return None
    for line in index.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "preregistration":
            return row.get("sha")
    return None


def _campaign_card_dir(name: str) -> Path | None:
    """The card dir whose manifest declares campaign ``name`` (its [claim] lives
    there). Re-scans the manifests rather than threading a fourth field through
    the Runtime -- the scan is already how discover() knows the owner, and a
    campaign spawn is not hot."""
    for mf in sorted((REPO_ROOT / "plugins").glob("*/manifest.toml")):
        if name in tomllib.loads(mf.read_text()).get("campaigns", {}):
            return mf.parent
    return None


def _campaign_cmd(name: str, script: Path, out: Path) -> list[str]:
    """The subprocess argv for a campaign script.

    Self-contained campaign scripts (stack_campaign.py) take only ``--out``; the
    parameterized ``acceptance_campaign.py`` also needs ``--claim <card dir>`` --
    it reads the [claim] table off that card's manifest. So a skill card's
    [campaigns] entry pointing at acceptance_campaign.py is runnable via
    submit_brief, not only by hand (GOAL v4.1 GUI 同源: 验货 IS an evolution-mode
    campaign brief through the runtime).
    ponytail: branch on the script basename; a per-script arg schema in the
    manifest would generalize only once a third campaign CLI shape appears.
    """
    if os.environ.get("PH_CAMPAIGN_ARGV"):
        # test seam: tests/test_runtime_killpg_e2e.py runs a long `sleep` as the
        # campaign so the group teardown is driven end to end without a sim.
        return json.loads(os.environ["PH_CAMPAIGN_ARGV"])
    cmd = [sys.executable, str(script), "--out", str(out)]
    if script.name == "acceptance_campaign.py":
        card = _campaign_card_dir(name)
        if card is None:
            raise ValueError(
                f"acceptance campaign {name!r} needs a --claim card dir, but no "
                "installed manifest declares this campaign name")
        cmd += ["--claim", str(card)]
    return cmd


def _cancel_marker(rt: Runtime, brief_id: str) -> Path:
    """The operator's cooperative stop flag for one brief, written by
    board.store.cancel_brief. Live state -- a plain file, never a chain row; the
    SEAL is the ``runtime.task_cancelled`` row this runtime writes when it acts."""
    return rt.inbox.parent / "cancel" / brief_id


def _cancel_requested(rt: Runtime, brief_id: str) -> bool:
    return _cancel_marker(rt, brief_id).exists()


#: Cancel-probe cadence while a campaign/rsi subprocess runs. A chain runs for
#: hours, so a 2s stat is free and bounds how long an operator waits.
_CANCEL_POLL_S = 2.0
#: Grace between SIGTERM and SIGKILL on a cancelled subprocess group.
_CANCEL_GRACE_S = 10.0


def _run_watched(cmd: list[str], env: dict, rt: Runtime, brief_id: str,
                 out: Path, what: str, tick=None) -> tuple[int, str]:
    """``subprocess.run`` for a campaign/rsi chain, plus operator cancellation.

    ``start_new_session=True`` puts the child in its OWN process group, which is
    the whole trick: a campaign is a worker POOL, and signalling only the parent
    leaves the workers as orphans still burning GPU (learned the hard way), so
    the stop is ``os.killpg`` over the group -- and because the group is NEW, the
    runtime's own process is never in the blast radius.

    The group never outlives the brief: cancel, SIGTERM (``cockpit --stop`` ->
    SystemExit), Ctrl-C or any crash of this frame kills it by pgid first, and
    the half-written store is marked ``CANCELLED`` before the raise continues: a
    truncated dev sweep must never be readable as a result (two-state law). A
    cancel lands in ``_process``'s crash-safety wrap, which sees the marker and
    seals ``runtime.task_cancelled`` rather than ``runtime.task_error``. The
    pgid is also dropped beside the claimed brief so a boot after a SIGKILL can
    reap the orphan (``_reap_orphan``) before re-queueing.

    ``tick`` (optional, zero-arg) runs on every poll while the child is alive --
    the evolve loop seals its finished rounds off campaign.json as they land.
    """
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    marker = _pgid_marker(rt.processing, brief_id)
    marker.write_text(str(proc.pid))  # pgid == pid (start_new_session)
    try:
        while True:
            try:
                # communicate() drains both pipes while it waits, so a chatty
                # campaign can never deadlock on a full stderr buffer; retrying
                # after a timeout loses no output (documented contract).
                _, err = proc.communicate(timeout=_CANCEL_POLL_S)
                return proc.returncode, err
            except subprocess.TimeoutExpired:
                if tick is not None:
                    tick()
                if _cancel_requested(rt, brief_id):
                    raise RuntimeError(f"{what} cancelled by the operator")
    except BaseException:
        if proc.poll() is None:
            _stop_group(proc.pid, proc)
        out.mkdir(parents=True, exist_ok=True)
        drop(out, "CANCELLED", json.dumps(
            {"brief": brief_id, "what": what, "ts": time.time()}))
        raise
    finally:
        marker.unlink(missing_ok=True)


def _run_campaign(brief: dict, rt: Runtime, brief_id: str) -> None:
    """Schedule one preregistered RSI campaign as an in-system task: guard the
    seed ledger, spawn the fixed campaign script as a SUBPROCESS, fold its
    published skills into the shared root, and note it in the chain.

    Cross-link is by the prereg content hash -- the only cross-link primitive
    that exists. A later task's FRESH graph.skill mount re-globs the shared root
    and sees the copied records for free (M4#4). Ledger-overlap and any nonzero
    exit raise, so the caller's crash-safety wrap files the brief under failed/
    with a runtime.task_error note -- one failure path, not two.
    """
    name = brief["campaign"]
    script = rt.campaigns.get(name)
    if script is None:
        raise ValueError(f"unknown campaign {name!r}")
    # A manifest carries the script path relative to the repo; REPO_ROOT / an
    # already-absolute path is that absolute path, so both forms resolve here.
    script = REPO_ROOT / script

    _assert_unburned(brief, name, _runs_root(rt))

    out = rt.inbox.parent / "campaigns" / Path(brief_id).stem
    code, err = _run_watched(_campaign_cmd(name, script, out),
                             {**os.environ, "MUJOCO_GL": "egl"},
                             rt, brief_id, out, f"campaign {name!r}")
    if code != 0:
        raise RuntimeError(f"campaign {name!r} exited {code}: {err.strip()[-500:]}")

    copied = _copy_skills(out / "skills", rt.skills_root)
    rt.log.append("runtime.campaign_scheduled",
                  {"brief": brief_id, "campaign": name, "out": str(out),
                   "prereg_sha": _prereg_sha(out), "skills": copied})


def _rsi_blocks(brief: dict, runs_root: Path) -> dict:
    """Claim this rsi brief's cal/dev/held-out blocks off the DERIVED ledger.

    Allocation happens HERE, server-side, for the same reason provider refs do:
    a brief that could name its own seed blocks could name a burned one. The
    operator may pin a block (``cal``/``dev``/``heldout`` as ``[lo,hi]``) -- pinning
    a calibration block is the normal case since calibration never gates -- but
    whatever comes out is fed straight back through ``_assert_unburned``.
    """
    from scripts.rsi_campaign import allocate

    def pin(key):
        v = brief.get(key)
        return (int(v[0]), int(v[1])) if v is not None else None

    return allocate(burned_blocks(runs_root),
                    floor=int(brief.get("floor", 0)),
                    cal=pin("cal"), dev=pin("dev"), heldout=pin("heldout"))


def _run_rsi(brief: dict, rt: Runtime, brief_id: str) -> None:
    """Schedule one GENERIC RSI chain: `{"kind":"rsi","task":"<task>"}` and the
    runtime walks allocate -> calibrate -> gate -> prereg -> dev -> held-out ->
    fold, for ANY installed task.

    Same shape as ``_run_campaign`` -- allocate, guard the ledger, spawn the
    script as a subprocess (two-state discipline: a campaign never runs inside
    the resident runtime's process), fold published skills into the shared root,
    note it in the chain. What differs is only that the seed blocks are computed
    rather than declared, and that the chain may stop honestly at the gate, in
    which case the store holds a verdict and no skills.
    """
    task = brief["task"]
    if task not in rt.task_bindings:
        raise ValueError(f"no task binding for {task!r}; install a plugin that "
                         f"declares it (known: {sorted(rt.task_bindings)})")
    blocks = _rsi_blocks(brief, _runs_root(rt))
    # the SAME guard the campaign path uses, fed the allocated blocks. cal is
    # deliberately absent: a calibration block never gates and stays re-measurable.
    _assert_unburned({"dev": [list(blocks["dev"])], "heldout": [list(blocks["heldout"])]},
                     f"rsi {task!r}", _runs_root(rt))

    out = rt.inbox.parent / "campaigns" / Path(brief_id).stem
    cmd = [sys.executable, str(REPO_ROOT / "scripts/rsi_campaign.py"),
           "--task", task, "--out", str(out),
           "--workers", str(int(brief.get("workers", 10)))]
    for key in ("cal", "dev", "heldout"):
        cmd += [f"--{key}", f"{blocks[key][0]}:{blocks[key][1]}"]
    if brief.get("node"):
        cmd += ["--node", str(brief["node"])]
    env = {**os.environ, "MUJOCO_GL": "egl"}
    if rt.frames:
        # The 取景窗 life sign for a chain: ONE pool worker (lockfile winner in
        # rsi_campaign._maybe_arm_frames) mirrors its episodes to the session's
        # frame.jpg. Live state, not evidence -- same file the task path dumps.
        env["PH_RSI_FRAMES"] = str(rt.inbox.parent / "frame.jpg")
    code, err = _run_watched(cmd, env, rt, brief_id, out, f"rsi chain for {task!r}")
    if code != 0:
        raise RuntimeError(
            f"rsi chain for {task!r} exited {code}: {err.strip()[-500:]}")

    report = json.loads((out / "rsi_report.json").read_text())
    copied = _copy_skills(out / "skills", rt.skills_root)
    rt.log.append("runtime.rsi_scheduled", {
        "brief": brief_id, "task": task, "out": str(out),
        "blocks": report.get("blocks"), "stage": report.get("stage"),
        "gate": {"proceed": report["gate"]["proceed"],
                 "failed": report["gate"]["failed"],
                 "target_node": report["gate"]["target_node"]},
        "prereg_sha": report.get("preregistration_sha") or _prereg_sha(out),
        "skills": copied,
        "ledger_entry": report.get("ledger_entry"),
    })


def _seal_rounds(rt: Runtime, brief_id: str, task: str, path: Path) -> None:
    """Seal one ``rsi_step`` row per campaign.json round not yet in the chain.
    Idempotent by (task, round): a resumed campaign seals only its new rounds."""
    if not path.exists():
        return
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError:  # mid-rename never happens (tmp+rename), but stay tolerant
        return
    sealed = {r["data"].get("round") for r in rt.log.rows()
              if r["kind"] == "rsi_step" and r["data"].get("task") == task}
    for rd in doc.get("rounds") or ():
        if rd["round"] not in sealed:
            if rd.get("proposal"):   # the inbox entry this round consumed, sealed first
                rt.log.append("rsi_proposal_applied", {"brief": brief_id, "task": task,
                                                       "round": rd["round"], **rd["proposal"]})
            rt.log.append("rsi_step", {"brief": brief_id, "task": task,
                                       **{k: rd.get(k) for k in ("round", "tried", "before", "after",
                                                                 "best", "published", "suite_sha",
                                                                 "per_seed", "needs", "proposer", "llm",
                                                                 "parent", "outcome", "confirm", "usage")}})


def _run_evolve(brief: dict, rt: Runtime, brief_id: str) -> None:
    """The LIGHTWEIGHT evolution loop: `{"kind":"evolve","task":"<task>","seeds":[lo,hi],
    "rounds":N,"arm":"auto"}` spawns scripts/evolve.py as a watched subprocess (same
    cancel/killpg as the campaign paths). Rounds land in
    ``campaigns/evolve-<task>/campaign.json``; every finished round is sealed here
    as an ``rsi_step`` row (on each poll and at exit, cancel included). A cancel
    the loop sees at a round boundary exits nonzero -> the marker makes it
    ``runtime.task_cancelled``. Resubmitting the same task resumes from cursor."""
    task = brief["task"]
    if task not in rt.task_bindings:
        raise ValueError(f"no task binding for {task!r}; install a plugin that "
                         f"declares it (known: {sorted(rt.task_bindings)})")
    out = rt.inbox.parent / "campaigns" / f"evolve-{task}"
    cmd = [sys.executable, str(REPO_ROOT / "scripts/evolve.py"), "--mode", "evolution",
           "--task", task, "--session", str(rt.inbox.parent),
           "--skills-root", str(rt.skills_root),
           "--rounds", str(int(brief.get("rounds", 0))),   # 0 = until 停止
           "--arm", str(brief.get("arm", "auto")),
           "--cancel-marker", str(_cancel_marker(rt, brief_id))]
    if brief.get("seeds"):
        cmd += ["--seeds", str(int(brief["seeds"][0])), str(int(brief["seeds"][1]))]
    if brief.get("proposer"):   # llm (default) | rules
        cmd += ["--proposer", str(brief["proposer"])]
    for k in ("max_replans", "max_actuations", "confirm_seeds"):   # budgets: brief > binding > workload default
        if brief.get(k) is not None:
            cmd += [f"--{k.replace('_', '-')}", str(int(brief[k]))]
    seal = lambda: _seal_rounds(rt, brief_id, task, out / "campaign.json")
    env = {**os.environ, "MUJOCO_GL": "egl"}
    if rt.frames:   # same 取景窗 mirror as the rsi chain: live state, never evidence
        env["PH_RSI_FRAMES"] = str(rt.inbox.parent / "frame.jpg")
    try:
        code, err = _run_watched(cmd, env, rt, brief_id, out, f"evolve {task!r}", tick=seal)
    finally:
        seal()
    if code != 0:
        raise RuntimeError(f"evolve {task!r} exited {code}: {err.strip()[-500:]}")


def _find_key(obj, key: str) -> str | None:
    """First value under ``key`` anywhere inside a nested row payload, or None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            if (hit := _find_key(v, key)) is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            if (hit := _find_key(v, key)) is not None:
                return hit
    return None


def _run_suite(brief: dict, rt: Runtime, brief_id: str, cancelled) -> bool:
    """Run one benchmark card as ordinary task briefs -- every (task, seed) goes
    through the SAME ``_run_task`` a task brief uses -- and seal one
    content-addressed suite artifact under ``<session>/suites/<sha>.json``.

    Before the first episode a preregistration burns ``[lo,hi]`` as a heldout
    block in the session's campaign-store layout (``campaigns/<brief>``), the
    layout ``board.store.burned_blocks`` already scans; an overlap with a burned
    block is refused first (the shared ``_assert_unburned`` guard, fed the
    suite's block). Returns True when the operator cancelled mid-suite.
    """
    from plugins.rsi.campaign import CampaignStore
    from harness.config import sha_json
    name = brief["suite"]
    card = rt.benchmarks.get(name)
    if card is None:
        raise ValueError(f"unknown suite {name!r} (known: {sorted(rt.benchmarks)})")
    arm = brief.get("arm", "scripted")
    if arm != "auto" and arm not in card.get("arms", ()):
        raise ValueError(f"suite {name!r} has no arm {arm!r} (arms: {card.get('arms')})")
    lo, hi = (int(x) for x in brief["seeds"])
    lo, hi = min(lo, hi), max(lo, hi)
    _assert_unburned({"heldout": [[lo, hi]]}, f"suite {name!r}", _runs_root(rt),
                     allow_empty=True)
    store = CampaignStore(rt.inbox.parent / "campaigns" / Path(brief_id).stem)
    prereg_sha = store.put("preregistration", {
        "kind": "preregistration", "suite": name, "arm": arm,
        "tasks": list(card["tasks"]), "blocks": {"heldout": [lo, hi]},
        "heldout": list(range(lo, hi + 1))})
    rt.log.append("runtime.suite_preregistered",
                  {"brief": brief_id, "suite": name, "arm": arm,
                   "seeds": [lo, hi], "prereg_sha": prereg_sha})
    max_replans = int(brief.get("max_replans", card.get("max_replans", 3)))
    # Actuation floor: brief -> card -> the task path's own default. kitchen_thaw's
    # 14-node chain dies at the task default (3) before its first segment.
    budget = brief.get("max_actuations", card.get("max_actuations"))
    budget = {} if budget is None else {"max_actuations": int(budget)}
    first_row = len(rt.log.rows())
    per_task = {}
    for task in card["tasks"]:
        n = k = 0
        lengths = []
        first_death = None
        for seed in range(lo, hi + 1):
            at = len(rt.log.rows())
            out = _run_task({"kind": "task", "task": task, "seed": seed, "media": True,
                             "max_replans": max_replans, "arm": arm, **budget},
                            rt, cancelled=cancelled)
            if cancelled_run(out):
                return True
            n += 1
            k += bool(out["success"])
            lengths.append(sum(1 for r in rt.log.rows()[at:] if r["kind"] == "task.verify"))
            if first_death is None and not out["success"]:
                first_death = seed
        per_task[task] = {"n": n, "k": k,
                          "L_mean": (sum(lengths) / len(lengths)) if lengths else None,
                          "first_death": first_death}
    artifact = {"kind": "suite_result", "suite": name, "arm": arm,
                "seeds": [lo, hi], "per_task": per_task, "prereg_sha": prereg_sha}
    ckpt = _find_key([r["data"] for r in rt.log.rows()[first_row:]], "checkpoint_sha")
    if ckpt is not None:
        artifact["checkpoint_sha"] = ckpt
    sha = sha_json(artifact)
    suites = rt.inbox.parent / "suites"
    suites.mkdir(exist_ok=True)
    (suites / f"{sha}.json").write_text(json.dumps(artifact, indent=1, sort_keys=True))
    rt.log.append("suite.sealed", {"brief": brief_id, "suite": name, "arm": arm,
                                   "seeds": [lo, hi], "sha": sha})
    return False


#: A brief is a selector plus budgets. A task may additionally carry one inert
#: natural-language ``instruction``; it selects no provider/skill/oracle and is
#: bounded below before reaching the planner. Providers are always chosen
#: server-side (the manifest union's task_bindings / campaigns); any other key
#: is rejected.
#:
#: ``rsi`` is the GENERIC evolution brief: it names a TASK, not a hand-written
#: campaign script, and the runtime walks the whole discipline chain for it. Its
#: optional keys are all overrides of things the chain would otherwise decide
#: from measurement -- ``node`` overrides the attribution's target (recorded in
#: the verdict), ``cal``/``dev``/``heldout`` pin a block instead of allocating.
_BRIEF_KEYS = {
    "task": {"kind", "task", "instruction", "seed", "max_replans",
             "max_actuations", "arm", "media"},
    "campaign": {"kind", "campaign", "dev", "heldout"},
    "suite": {"kind", "suite", "arm", "seeds", "max_replans", "max_actuations"},
    "rsi": {"kind", "task", "node", "cal", "dev", "heldout", "workers", "floor"},
    "mission": {"kind", "mission", "seed", "arm", "max_replans", "max_actuations"},
    "evolve": {"kind", "task", "seeds", "rounds", "arm", "max_replans", "max_actuations", "proposer",
               "confirm_seeds"},
}
_MAX_INSTRUCTION_CHARS = 4000


def _seal_cancelled(rt: Runtime, brief_id: str, brief: dict, claimed: Path,
                    stage: str) -> None:
    """File a cancelled brief into ``cancelled/`` under its OWN chain row.

    ``runtime.task_cancelled``, never ``runtime.task_error``: an operator
    stopping a run and a run crashing must stay separable in the evidence
    forever, or a later audit reads the human's decision as a capability the
    harness lacks. ``stage`` records which boundary caught it -- ``queued`` (the
    brief never started), ``running`` (stopped at a node/subprocess boundary) or
    ``runtime_stopped`` (the runtime itself was told to exit mid-brief).
    """
    rt.log.append("runtime.task_cancelled",
                  {"brief": brief_id, "task": brief.get("task") or brief.get("campaign"),
                   "stage": stage})
    opstream.emit("task_cancelled", brief=brief_id, task=brief.get("task"),
                  stage=stage)
    (rt.inbox.parent / "cancelled").mkdir(parents=True, exist_ok=True)
    os.replace(claimed, rt.inbox.parent / "cancelled" / brief_id)
    _cancel_marker(rt, brief_id).unlink(missing_ok=True)


def _process(rt: Runtime, path: Path) -> None:
    """Claim one brief, run it, file it under done/, failed/ or cancelled/."""
    brief_id = path.name
    claimed = rt.processing / brief_id
    try:
        os.rename(path, claimed)  # atomic claim; a racing worker gets FileNotFound
    except FileNotFoundError:
        return
    try:
        brief = json.loads(claimed.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        # malformed: nothing to attribute a note to (design: failed/ with no note)
        os.replace(claimed, rt.failed / brief_id)
        return
    opstream.emit("task_claimed", brief=brief_id, task=brief.get("task"),
                  campaign=brief.get("campaign"), seed=brief.get("seed"))
    # Checkpoint 1, AFTER the atomic claim: exactly one process owns this brief
    # now, so cancelling a queued brief has no race with claiming it -- the
    # winner of the rename is the one that reads the marker.
    if _cancel_requested(rt, brief_id):
        _seal_cancelled(rt, brief_id, brief, claimed, "queued")
        return
    stopped = False
    try:
        kind = brief.get("kind", "task")
        unknown = set(brief) - _BRIEF_KEYS.get(kind, set())
        if unknown:
            # Rejection, not neutralization: a brief is selector+budgets only, and
            # an injected key (a provider ref, say) must fail loudly rather than
            # ride along ignored until some future reader starts honoring it.
            raise ValueError(f"unknown brief keys {sorted(unknown)}")
        if kind == "task" and "instruction" in brief:
            instruction = brief["instruction"]
            if (not isinstance(instruction, str) or not instruction.strip()
                    or len(instruction) > _MAX_INSTRUCTION_CHARS):
                raise ValueError(
                    "task instruction must be a non-empty string of at most "
                    f"{_MAX_INSTRUCTION_CHARS} characters")
        if kind == "mission":
            mission = brief.get("mission")
            if (not isinstance(mission, str) or not mission.strip()
                    or len(mission) > _MAX_INSTRUCTION_CHARS):
                raise ValueError("mission must be a non-empty string of at most "
                                 f"{_MAX_INSTRUCTION_CHARS} characters")
        if kind in ("task", "mission"):
            # execution mounts a frozen skills root; a mid-session out-of-band
            # mutation (some record added or removed since the boot seal) must
            # fail loudly rather than let a task run against skills it wasn't
            # admitted with. Evolution's _copy_skills legitimately grows the root,
            # so the audit is execution-only.
            if rt.mode == "execution":
                current = _skills_manifest(rt.skills_root)
                if set(current) != set(rt.skills_manifest):
                    raise ValueError(
                        "skills-root mutated mid-session (execution mode): "
                        f"boot manifest {list(rt.skills_manifest)} != {current}")
            # Checkpoint 2 lives INSIDE the workload (the node boundary). It
            # reports back through the result, not through the marker: a cancel
            # that arrives after the last node did not stop anything, and filing
            # a mission that finished under cancelled/ would be a lie in the
            # other direction.
            stopped = cancelled_run(
                (_run_task if kind == "task" else _run_mission)(
                    brief, rt, cancelled=lambda: _cancel_requested(rt, brief_id)))
        elif kind == "suite":
            stopped = _run_suite(brief, rt, brief_id,
                                 cancelled=lambda: _cancel_requested(rt, brief_id))
        elif kind in ("campaign", "rsi", "evolve"):
            # v4.1 hard rule: an evolution brief is accepted ONLY in evolution
            # mode. Rejection, not neutralization -- same pattern as the
            # injected-key defense: a real task run provably triggers no RSI.
            if rt.mode != "evolution":
                raise ValueError(
                    f"{kind} briefs are accepted only in evolution mode "
                    f"(session mode is {rt.mode!r})")
            {"campaign": _run_campaign, "rsi": _run_rsi,
             "evolve": _run_evolve}[kind](brief, rt, brief_id)
        else:
            raise ValueError(f"unknown brief kind {kind!r}")
    except Exception as exc:  # noqa: BLE001 -- escape hatch: crash-safety lives here
        # Checkpoint 3: a campaign/rsi cancellation ARRIVES as a raise (the
        # killed group exits nonzero), so a set marker here means this ending is
        # the operator's, not a fault. It also covers a task that raised while a
        # stop was pending -- calling that a task_error would both misattribute
        # it and strand the marker.
        if _cancel_requested(rt, brief_id):
            _seal_cancelled(rt, brief_id, brief, claimed, "running")
            return
        rt.log.append("runtime.task_error",
                      {"brief": brief_id, "task": brief.get("task"),
                       "error": repr(exc)})
        opstream.emit("task_failed", brief=brief_id, task=brief.get("task"),
                      error=repr(exc))
        os.replace(claimed, rt.failed / brief_id)
        return
    except BaseException:
        # SIGTERM (cockpit --stop) / Ctrl-C mid-brief: the operator's stop, not
        # a fault, and never left in processing/ for the next boot to re-run.
        _seal_cancelled(rt, brief_id, brief, claimed, "runtime_stopped")
        raise
    if stopped:
        _seal_cancelled(rt, brief_id, brief, claimed, "running")
        return
    # The work finished. A marker that arrived too late cancelled nothing, and
    # must not outlive the brief it named.
    _cancel_marker(rt, brief_id).unlink(missing_ok=True)
    opstream.emit("task_done", brief=brief_id, task=brief.get("task"))
    try:
        os.replace(claimed, rt.done / brief_id)
    except OSError as exc:
        # A filesystem failure at the very last rename must not kill the loop;
        # the brief stays in processing/ and re-queues on the next boot.
        rt.log.append("runtime.task_error",
                      {"brief": brief_id, "task": brief.get("task"),
                       "error": f"done-rename failed: {exc!r}"})


def _pending(rt: Runtime) -> list[Path]:
    """Return a stable oldest-first snapshot of the inbox.

    Inbox ownership changes by atomic rename.  A submitter, cancellation path,
    or another runtime incarnation can therefore move a file after ``glob``
    found it but before ``stat`` reads its timestamp.  That is normal lifecycle
    churn, not a fatal runtime error: omit the vanished entry and pick it up from
    its new owner (or on the next poll) instead of killing the resident loop.
    """
    pending: list[tuple[float, Path]] = []
    for path in rt.inbox.glob("*.json"):
        try:
            pending.append((path.stat().st_mtime, path))
        except FileNotFoundError:
            continue
    return [path for _, path in sorted(pending, key=lambda item: item[0])]


#: Heartbeat cadence for runtime_status.json (the UI liveness signal).
HEARTBEAT_S = 10.0


def _status_stamp(session_dir: Path, **fields) -> None:
    """Merge ``fields`` into runtime_status.json: read-modify-write, so every
    boot-written field rides through verbatim, and atomic via the shared
    brief_drop temp+replace, so board.store.read_runtime_status never sees a
    half write. Never raises: a broken status file loses the badge, never the
    loop."""
    try:
        status = json.loads((session_dir / "runtime_status.json").read_text())
        drop(session_dir, "runtime_status.json", json.dumps({**status, **fields}))
    except (OSError, json.JSONDecodeError, TypeError):
        # TypeError: valid JSON that is not an object. Swallowed with the rest --
        # the beat runs in a thread, and an escape there would kill the beat
        # silently, which is precisely the failure this whole change is about.
        pass


def _heartbeat(session_dir: Path) -> None:
    """One beat. boot_ts alone cannot tell a live runtime from a dead one; the
    board passes ``heartbeat_ts`` through as an age, and the UI reads it."""
    _status_stamp(session_dir, heartbeat_ts=time.time())


def _beat_forever(session_dir: Path) -> threading.Thread:
    """Beat from a DAEMON THREAD, started at boot and never joined.

    It used to beat from the poll loop, which only comes back around BETWEEN
    briefs: a runtime an hour into an rsi chain looked exactly like a corpse,
    and "busy or dead" is the ambiguity that let three briefs rot in inboxes
    nobody was serving. Stamping at node boundaries inside ``_process`` would
    not have fixed it either -- campaigns and rsi chains run in SUBPROCESSES
    the parent only waits on, and there is no node boundary to hook in a wait.
    A thread beats through all of it: subprocess waits, sim steps, blocking IO.

    Daemon, so it never holds the process open; each beat is an atomic,
    exception-swallowing merge, so it can neither corrupt the file nor take the
    run down with it. It is the ONLY concurrent writer of that file (boot writes
    it before this starts), so the read-modify-write needs no lock.
    """
    def beat() -> None:
        while True:
            time.sleep(HEARTBEAT_S)   # sleep FIRST: boot just stamped
            _heartbeat(session_dir)
    thread = threading.Thread(target=beat, name="heartbeat", daemon=True)
    thread.start()
    return thread


def main(session_dir: str | Path, inbox: str | Path | None = None, *,
         drain: bool = False, poll_interval: float = 1.0,
         mode: str = "execution", render: bool = False,
         render_fps: float = DEFAULT_RENDER_FPS, frames: bool = False) -> Runtime:
    """Boot, then drain the inbox once (``drain``) or poll it forever."""
    rt = boot(session_dir, inbox, mode=mode, render=render, render_fps=render_fps,
              frames=frames)
    session_dir = Path(session_dir)
    _beat_forever(session_dir)
    # A clean exit says so, so the leftover status file cannot pass for a live
    # runtime. Belt-and-braces only: a kill -9 never gets here, which is why
    # read_runtime_status checks the pid against /proc rather than trusting this.
    try:
        if drain:
            while True:
                pending = _pending(rt)
                if not pending:
                    return rt
                for p in pending:
                    _process(rt, p)
        while True:
            for p in _pending(rt):
                _process(rt, p)
            time.sleep(poll_interval)
    finally:
        _status_stamp(session_dir, stopped_ts=time.time())


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--inbox", default=None,
                    help="brief inbox dir (default <session-dir>/inbox)")
    ap.add_argument("--mode", choices=MODES, default="execution",
                    help="session mode, fixed write-once at first boot "
                         "(default: execution -- campaigns rejected)")
    ap.add_argument("--drain", action="store_true",
                    help="process pending briefs then exit (default: poll forever)")
    ap.add_argument("--poll-interval", type=float, default=1.0)
    ap.add_argument("--render", action="store_true",
                    help="overlay a live MuJoCo window on each task so an operator "
                         "watches the sim (needs $DISPLAY; refuses headless, unsets "
                         "MUJOCO_GL=egl). A per-boot deployment choice like --mode, "
                         "never a brief; composes with execution/evolution. "
                         "Campaigns spawn a headless subprocess regardless.")
    ap.add_argument("--render-fps", type=float, default=DEFAULT_RENDER_FPS,
                    help=f"render pacing when --render is set: added per-step sleep "
                         f"= 1/fps (default {DEFAULT_RENDER_FPS:g}, watch_stack's 0.02s)")
    ap.add_argument("--frames", action="store_true",
                    help="overlay the offscreen frame dump on each task so the "
                         "ph-station 取景窗 shows the sim (runs/<session>/frame.jpg, "
                         "~4fps by step interval). Auto-ON when MUJOCO_GL is a "
                         "headless GL (egl/osmesa); pass explicitly for a windowed "
                         "--render boot. A per-boot deployment choice, never a brief.")
    args = ap.parse_args()
    # Headless-GL auto-enable (CLI only, so library callers and tests stay
    # explicit): an egl/osmesa runtime has no window, so the frame dump is the
    # only way an operator sees the sim at all.
    frames = args.frames or (
        os.environ.get("MUJOCO_GL", "").lower() in ("egl", "osmesa"))
    # SIGTERM's default action skips every finally, so `cockpit --stop` left the
    # status file claiming a live runtime. Turn it into an ordinary exit so
    # main()'s finally can mark it stopped. CLI only -- a library caller (tests,
    # soak.py) keeps whatever signal semantics its own process chose.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main(args.session_dir, args.inbox, drain=args.drain,
         poll_interval=args.poll_interval, mode=args.mode,
         render=args.render, render_fps=args.render_fps, frames=frames)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
