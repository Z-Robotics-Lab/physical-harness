"""Read-only parse layer over RSI campaign stores + STATUS/progress markdown.

Pure functions, no server, and two writes -- both into LIVE intake, never into
sealed evidence, both through the shared brief_drop atomic drop: ``submit_brief``
(a brief into a session inbox) and ``cancel_brief`` (a stop marker into the
session's ``cancel/``). Neither touches the hash chain: the resident runtime is
its one writer, and a cancel only becomes a ``runtime.task_cancelled`` row when
the runtime acts on the marker. The store
shape is the one `plugins.rsi.campaign.CampaignStore` writes: an
``index.jsonl`` of ``{seq, kind, sha, time}`` rows plus content-addressed
``artifacts/<sha>.json`` payloads. The *kind* lives in the index, never in the
payload, so the index is the discriminator (see CampaignStore.put).

Robust to partial/mid-write JSON: a campaign writing artifacts tonight can be
polled while a file is half-flushed. Unreadable index rows and unreadable
artifacts are skipped and counted, so a LIVE poll landing mid-write simply
retries on the next tick rather than crashing.
"""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from harness.events import SessionLog
from harness.protocol import Trajectory, to_plain
from scripts.brief_drop import drop

# --- store discovery / robust reads -----------------------------------------


def is_store(store_dir: str | Path) -> bool:
    """A directory is a campaign store iff it carries an index.jsonl."""
    return (Path(store_dir) / "index.jsonl").exists()


def safe_child(runs_dir: str | Path, name: str, is_kind) -> Path | None:
    """Resolve ``name`` to a direct child of ``runs_dir`` that passes ``is_kind``,
    rejecting path traversal. One audited guard for every name-addressed read
    (stores and sessions), shared by the board HTTP shell and the MCP server so
    neither can be walked outside runs_dir with a ``../`` name."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    runs_dir = Path(runs_dir)
    path = (runs_dir / name).resolve()
    if path.parent != runs_dir.resolve() or not is_kind(path):
        return None
    return path


def _index_rows(store_dir: Path) -> list[dict]:
    """Index rows in seq order, skipping any line that is not valid JSON.

    A campaign appends to index.jsonl line-by-line; a poll can catch a
    half-written trailing line. Skip it -- the next poll gets the whole row.
    """
    rows: list[dict] = []
    index_path = store_dir / "index.jsonl"
    if not index_path.exists():
        return rows
    for line in index_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "kind" in row and "sha" in row:
            rows.append(row)
    return rows


def read_store_artifacts(store_dir: str | Path) -> tuple[dict[str, list[dict]], int]:
    """Every readable artifact grouped by kind, in index order, plus a skip count.

    The skip count is artifacts referenced by the index whose JSON did not parse
    or whose file is missing -- i.e. mid-write. Callers surface it so the UI can
    show "1 artifact still being written" instead of pretending it is absent.
    """
    store_dir = Path(store_dir)
    by_kind: dict[str, list[dict]] = {}
    skipped = 0
    for row in _index_rows(store_dir):
        path = store_dir / "artifacts" / f"{row['sha']}.json"
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        by_kind.setdefault(row["kind"], []).append(payload)
    return by_kind, skipped


def store_mtime(store_dir: str | Path) -> float:
    """Newest mtime across index.jsonl and the artifacts dir -- the LIVE signal.

    A new generation both appends to the index and drops a new artifact file, so
    either mtime moving means "something changed"; the frontend polls this.
    """
    store_dir = Path(store_dir)
    times = [0.0]
    for p in (store_dir / "index.jsonl", store_dir / "artifacts"):
        try:
            times.append(p.stat().st_mtime)
        except OSError:
            pass
    return max(times)


# --- payload shaping --------------------------------------------------------


def _delta(paired: dict) -> float | None:
    """governed_rate - base_rate: the effect a paired gate measured. None if the
    dict is not a paired result (defensive against a mid-write partial)."""
    if not isinstance(paired, dict):
        return None
    g, b = paired.get("governed_rate"), paired.get("base_rate")
    if g is None or b is None:
        return None
    return g - b


def trigger_str(trig: dict) -> str:
    """A one-line human summary of a Trigger payload, e.g.
    ``observable.finger_gap value lt 0.001 @arm58``. Mirrors the fields the
    campaign's Trigger.describe() carries without importing the plugin."""
    if not isinstance(trig, dict):
        return str(trig)
    thr = trig.get("threshold")
    thr_s = f"{thr:.4g}" if isinstance(thr, (int, float)) else str(thr)
    return (f"{trig.get('feature')} {trig.get('reducer')} {trig.get('op')} "
            f"{thr_s} @arm{trig.get('arm_after')} dwell{trig.get('dwell')}")


def _rule_view(rule: dict) -> dict:
    """{trigger_str, recovery} from a rule payload ({trigger, recovery, rule_id})."""
    if not isinstance(rule, dict):
        return {"rule_id": None, "trigger_str": str(rule), "recovery": None}
    rec = rule.get("recovery") or {}
    return {"rule_id": rule.get("rule_id"),
            "trigger_str": trigger_str(rule.get("trigger", {})),
            "recovery": rec.get("name") or rec.get("strategy")}


def _ablation(entries: list) -> list[dict]:
    """[[sd, paired], ...] -> [{sd, governed_rate, base_rate, delta, privileged}]."""
    out = []
    for entry in entries or []:
        try:
            sd, paired = entry
        except (ValueError, TypeError):
            continue
        out.append({"sd": sd, "governed_rate": paired.get("governed_rate"),
                    "base_rate": paired.get("base_rate"), "delta": _delta(paired),
                    "declared_privilege": paired.get("declared_privilege")})
    return out


def store_detail(store_dir: str | Path) -> dict:
    """Full structured view of one store: whichever of the known artifact kinds
    are present. Unknown kinds are still surfaced (name + count) so a new probe
    type is visible before this parser learns to shape it."""
    store_dir = Path(store_dir)
    by_kind, skipped = read_store_artifacts(store_dir)
    detail: dict = {"name": store_dir.name, "skipped": skipped,
                    "kinds": {k: len(v) for k, v in by_kind.items()}}

    prereg = (by_kind.get("preregistration") or [None])[0]
    if prereg:
        heldout = prereg.get("heldout") or []
        dev = prereg.get("dev") or []
        detail["prereg"] = {
            "dev_n": len(dev), "heldout_n": len(heldout),
            "task": prereg.get("task"), "policy": prereg.get("policy"),
            "stages": bool(prereg.get("stages")),
            "terminal_label": prereg.get("terminal_label"),
            "alpha": prereg.get("alpha"), "min_fixed": prereg.get("min_fixed"),
            "percept_noise": prereg.get("percept_noise"),
            "heldout_block": [min(heldout), max(heldout) + 1] if heldout else None,
        }

    gens = []
    for g in by_kind.get("generation", []):
        dg, bg = g.get("dev_gate", {}), g.get("blind_gate", {})
        gens.append({
            "generation": g.get("generation"), "promoted": g.get("promoted"),
            "reason": g.get("reason"), "rule": _rule_view(g.get("rule", {})),
            "dev_gate": dg, "dev_delta": _delta(dg),
            "blind_gate": bg, "blind_delta": _delta(bg),
        })
    if gens:
        detail["generations"] = sorted(gens, key=lambda x: x["generation"] or 0)

    result = (by_kind.get("campaign_result") or [None])[0]
    if result:
        detail["result"] = {
            "generations": result.get("generations"),
            "promoted": result.get("promoted"), "rules": result.get("rules"),
            "heldout": result.get("heldout"),
            "heldout_delta": _delta(result.get("heldout", {})),
            "heldout_vs_blind": result.get("heldout_vs_blind"),
            "ablation": _ablation(result.get("ablation", [])),
        }

    rescores = []
    for r in by_kind.get("heldout_rescore", []):
        paired = r.get("paired", {})
        rescores.append({
            "block": r.get("block"), "n": r.get("n"),
            "paired": paired, "delta": _delta(paired),
            "vs_blind": r.get("vs_blind"), "judgement": r.get("judgement"),
            "stage_attribution": r.get("stage_attribution"),
        })
    if rescores:
        detail["rescores"] = rescores

    r25 = (by_kind.get("round25_rerun") or [None])[0]
    if r25:
        arms = {}
        for name, arm in (r25.get("arms") or {}).items():
            arms[name] = {"trigger_str": trigger_str((arm.get("rule") or {}).get("trigger", {})),
                          "heldout": arm.get("heldout")}
        detail["round25"] = {"dev_block": r25.get("dev_block"),
                             "heldout_block": r25.get("heldout_block"), "arms": arms}

    probe = (by_kind.get("arm_time_probe") or [None])[0]
    if probe:
        detail["probe"] = {"grade": probe.get("grade"), "p1": probe.get("p1"),
                          "p2": probe.get("p2")}

    fix = (by_kind.get("round88_fix") or [None])[0]
    if fix:
        detail["round88_fix"] = {"grade": fix.get("grade"), "dev_rate": fix.get("dev_rate"),
                                "top_score": fix.get("top_score"), "anchors": fix.get("anchors"),
                                "heldout": fix.get("heldout")}
    return detail


def store_summary(store_dir: str | Path) -> dict:
    """Cheap card for the store list: counts, task, promoted/total generations,
    mtime. Reads the artifacts once (stores are tiny), so the list is honest
    about mid-write skips too."""
    store_dir = Path(store_dir)
    by_kind, skipped = read_store_artifacts(store_dir)
    prereg = (by_kind.get("preregistration") or [None])[0]
    result = (by_kind.get("campaign_result") or [None])[0]
    gens = by_kind.get("generation", [])
    return {
        "name": store_dir.name,
        "mtime": store_mtime(store_dir),
        "kinds": {k: len(v) for k, v in by_kind.items()},
        "task": prereg.get("task") if prereg else None,
        "generations": len(gens),
        "promoted": sum(1 for g in gens if g.get("promoted")),
        "heldout": (result or {}).get("heldout", {}).get("fixed") if result else None,
        "skipped": skipped,
    }


def list_stores(runs_dir: str | Path) -> list[dict]:
    """Every store under runs_dir, newest first -- auto-discovery, TensorBoard-style."""
    runs_dir = Path(runs_dir)
    stores = [store_summary(p) for p in sorted(runs_dir.iterdir()) if p.is_dir() and is_store(p)]
    return sorted(stores, key=lambda s: s["mtime"], reverse=True)


#: A campaign heartbeat older than this reads as stale/finished, not running.
#: Episodes finish every few seconds under a worker pool; two minutes of silence
#: means the battery exited (or died) without reaching total.
_PROGRESS_RUNNING_S = 120


def campaign_progress(runs_dir: str | Path) -> list[dict]:
    """Every live campaign heartbeat under runs/ -- both a hand-run store
    (``runs/<store>/progress.json``) and a chain fired THROUGH the runtime, which
    lands two levels deeper at ``runs/<session>/campaigns/<brief>/progress.json``
    (harness_runtime writes campaign/rsi output under the session's inbox
    sibling). Written per finished episode by scripts/campaign_progress.py,
    newest first.

    Live state, not sealed evidence (same family as runtime_status): no chain
    verify, a mid-write or malformed file is skipped and the next poll recovers.
    Each row is the heartbeat's payload (done/total/started_ts/updated_ts/label
    + the rolling stats the writer folded) plus ``name`` (the store dir, or
    ``<session>/<brief>`` for a nested chain so the two never collide),
    ``fresh`` (updated within the freshness window) and ``running`` (fresh AND
    not yet at total). The
    rolling statistics arrive pre-folded from the python writer -- the TS panel
    only displays them (ETA is a pure display conversion of started_ts/done)."""
    runs_dir = Path(runs_dir)
    now = time.time()
    out = []
    flat = [(p, p.name) for p in sorted(runs_dir.iterdir())]
    nested = [(p, f"{p.parent.parent.name}/{p.name}")
              for p in sorted(runs_dir.glob("*/campaigns/*"))]
    for p, name in flat + nested:
        if not p.is_dir():
            continue
        try:
            row = json.loads((p / "progress.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not (isinstance(row, dict)
                and isinstance(row.get("done"), int)
                and isinstance(row.get("total"), int)):
            continue
        updated = row.get("updated_ts")
        fresh = isinstance(updated, (int, float)) and now - updated < _PROGRESS_RUNNING_S
        # ``fresh`` is forwarded on its own because ``running`` folds it away:
        # an rsi chain's LAST heartbeat carries the gate verdict AND sits at
        # done == total, so a reader keyed only on ``running`` would retire the
        # card at the exact moment the result appears. Both booleans are decided
        # here, python-side; the panel only filters on them.
        out.append({**row, "name": name, "fresh": fresh,
                    "running": fresh and row["done"] < row["total"]})
    out.sort(key=lambda r: r.get("updated_ts") or 0, reverse=True)
    return out


def heldout_blocks(runs_dir: str | Path, store_name: str) -> dict:
    """The multi-block held-out comparison for a campaign: its own scored block
    plus every sibling ``<name>-rescore-*`` store's rescored block. Each block
    carries the paired numbers and (when the campaign ran a stage overlay) the
    grasp/place stage attribution. This is view #2 -- stack-g1's three blocks."""
    runs_dir = Path(runs_dir)
    detail = store_detail(runs_dir / store_name)
    blocks = []
    result = detail.get("result")
    if result and result.get("heldout"):
        h = result["heldout"]
        prereg = detail.get("prereg", {})
        first = (prereg.get("heldout_block") or [None])[0]
        blocks.append({"source": store_name, "block": first, "paired": h,
                       "delta": result.get("heldout_delta"),
                       "vs_blind": result.get("heldout_vs_blind"),
                       "stage_attribution": None})
    for p in sorted(runs_dir.iterdir()):
        if not (p.is_dir() and is_store(p) and p.name.startswith(f"{store_name}-rescore")):
            continue
        for rs in store_detail(p).get("rescores", []):
            blocks.append({"source": p.name, "block": rs["block"], "paired": rs["paired"],
                           "delta": rs["delta"], "vs_blind": rs.get("vs_blind"),
                           "judgement": rs.get("judgement"),
                           "stage_attribution": rs.get("stage_attribution")})
    blocks.sort(key=lambda b: b["block"] if b["block"] is not None else 0)
    return {"store": store_name, "task": detail.get("prereg", {}).get("task"), "blocks": blocks}


# --- runtime session chain --------------------------------------------------
#
# The resident runtime (scripts/harness_runtime.py) writes ONE long-lived
# SessionLog under ``<session>/session-log/rows.jsonl`` -- a different on-disk
# shape from a campaign store (chained rows carrying their data inline, no
# content-addressed artifacts). Kept deliberately separate from is_store/
# list_stores so a runtime session never renders as an empty campaign.


def is_session(session_dir: str | Path) -> bool:
    """A directory is a runtime session iff it carries session-log/rows.jsonl."""
    return (Path(session_dir) / "session-log" / "rows.jsonl").exists()


def session_inbox(runs_dir: str | Path, session: str) -> Path | None:
    """Resolve a runtime session NAME to its inbox dir, or ``None`` when the name
    is not a real booted session under ``runs_dir`` (traversal rejected).

    The WRITE-side twin of the read guard: submit_brief/run_task route a brief
    into the returned inbox, exactly as the read faces resolve a session through
    ``safe_child(is_session)``. Same one audited guard, so a ``../`` name can
    never route a brief outside runs_dir and only a session with a resident
    runtime (its ``session-log/rows.jsonl`` exists) is a legal target.

    Three brief kinds land here -- ``task`` (run a mission once), ``campaign`` (a
    named hand-written campaign script) and ``rsi`` (the generic self-improvement
    chain, minimal form ``{"kind":"rsi","task":"<task>"}``; docs/project-documentation.md §4).
    This function routes all three identically: the resident runtime's
    ``_BRIEF_KEYS`` stays the sole authority on what a brief may say."""
    path = safe_child(runs_dir, session, is_session)
    return (path / "inbox") if path else None


def brief_inbox(runs_dir: str | Path, session: str,
                default_session: str = "session-main",
                default_inbox: str | Path | None = None) -> Path | None:
    """The inbox a brief routes into, or ``None`` for an unknown session.

    The default session resolves WITHOUT the is_session gate (a first submit may
    precede the runtime's first boot) to ``default_inbox`` when given (the MCP
    server's configure(inbox=) override) else ``<runs>/<session>/inbox``; any
    other session goes through session_inbox (real booted session, ``../``
    rejected). The ONE routing rule for every submit face."""
    if session == default_session:
        return (Path(default_inbox) if default_inbox
                else Path(runs_dir) / default_session / "inbox")
    return session_inbox(runs_dir, session)


def submit_brief(runs_dir: str | Path, raw: str, session: str = "session-main",
                 default_session: str = "session-main",
                 default_inbox: str | Path | None = None) -> dict:
    """Atomically drop ``raw`` as a brief into ``session``'s inbox.

    The ONE submit implementation both write faces (board/mcp_server.py's
    submit_brief tool, ``storecli submit_brief``) pass through, so their outputs
    are the same dict from the same code: ``{"submitted": <name>, "inbox":
    <dir>}``, or ``{"error": ...}`` for an unknown session. ZERO validation of
    ``raw`` -- not even a JSON parse: a brief rides through verbatim and the
    resident runtime's ``_BRIEF_KEYS`` re-validation on claim is the SOLE
    authority (a malformed or injected brief must hard-fail THERE, loudly,
    never be silently normalized by a producer). Atomicity is the shared
    brief_drop.drop (temp write + os.replace), so the runtime never claims a
    half-written file."""
    inbox = brief_inbox(runs_dir, session, default_session, default_inbox)
    if inbox is None:
        return {"error": f"unknown session {session!r}"}
    inbox.mkdir(parents=True, exist_ok=True)
    name = f"brief-{uuid.uuid4().hex}.json"
    drop(inbox, name, raw)
    return {"submitted": name, "inbox": str(inbox)}


# --- runtime liveness --------------------------------------------------------
#
# ``runtime_status.json`` is a FILE, and a file outlives the process that wrote
# it. Every one of the three intake incidents was that single fact: 2026-08-27 an
# RSI brief sat 21h in a session whose runtime had died; 2026-08-28 the web
# process was reaped and the runtimes lived on; 2026-08-29 session-robocasa's
# runtime was long dead, its leftover status file made every face answer
# "runtime alive", and a brief held queue position 1 forever.
#
# So liveness is asked of /proc, never of the status file. The file only names
# the pid; whether that pid is STILL a harness_runtime serving THIS session dir
# is the question, and the same pid-recycling guard _model_identity applies to
# the model server applies here.

#: How stale a heartbeat may get on an IDLE runtime before it reads as wedged.
#: The poll loop stamps every 10s (harness_runtime.HEARTBEAT_S) but never DURING
#: a brief -- "busy or dead" is the documented ambiguity, and an empty
#: processing/ is what breaks the tie, so this threshold only ever judges an idle
#: runtime (see health()).
_HEARTBEAT_STALE_S = 60.0


def _serves_session(pid: int, session_dir: Path) -> bool:
    """True iff ``pid`` is live AND is a harness_runtime whose ``--session-dir``
    IS this session. Both halves matter: a dead pid is the incident above, and a
    recycled one (this box runs three runtimes) would otherwise vouch for the
    wrong session. The session dir is compared RESOLVED, so a relative argv
    (``--session-dir runs/session-main``, the canonical cockpit spawn form) is
    joined against that process's own cwd -- never ours."""
    try:
        argv = [a.decode(errors="replace")
                for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
    except (OSError, ValueError):
        return False
    if not any("harness_runtime" in a for a in argv):
        return False
    try:
        served = Path(argv[argv.index("--session-dir") + 1])
    except (ValueError, IndexError):
        return False
    if not served.is_absolute():
        try:
            served = Path(os.readlink(f"/proc/{pid}/cwd")) / served
        except OSError:
            return False
    try:
        return served.resolve() == Path(session_dir).resolve()
    except OSError:
        return False


def runtime_liveness(session_dir: str | Path) -> dict:
    """Is a resident runtime SERVING this session right now?

    ``{alive, pid, mode, boot_ts, heartbeat_ts, heartbeat_age_s, reason}`` --
    ``reason`` names why not when ``alive`` is False, and is ``None`` when it is
    True. Same live-state family as read_runtime_status (no chain verify, never
    raises), but it JUDGES where that one deliberately does not: the pid is
    checked against /proc, because the whole class of intake incidents in this
    repo is a status file that outlived its writer.

    ``heartbeat_age_s`` is reported, never judged here: the poll loop does not
    beat while a brief runs, so a stale beat on a BUSY runtime is normal. Only
    health() (which can see processing/) turns age into a verdict.
    """
    session_dir = Path(session_dir)
    status = read_runtime_status(session_dir)
    if status is None:
        return {"alive": False, "pid": None, "mode": None, "boot_ts": None,
                "heartbeat_ts": None, "heartbeat_age_s": None,
                "reason": "no runtime_status.json -- this session has never booted"}
    beat = status.get("heartbeat_ts") or status.get("boot_ts")
    out = {"pid": status.get("pid"), "mode": status.get("mode"),
           "boot_ts": status.get("boot_ts"), "heartbeat_ts": beat,
           "heartbeat_age_s": (round(max(time.time() - float(beat), 0.0), 1)
                               if beat else None)}
    try:
        pid = int(status["pid"])
    except (KeyError, TypeError, ValueError):
        return {**out, "alive": False, "reason": "runtime_status.json names no pid"}
    if not _serves_session(pid, session_dir):
        return {**out, "alive": False,
                "reason": f"pid {pid} is not a harness_runtime serving "
                          f"{session_dir.name} (stale runtime_status.json -- the "
                          "runtime died and left the file behind)"}
    return {**out, "alive": True, "reason": None}


# --- brief lifecycle: status + cancel ----------------------------------------
#
# A brief lives in exactly one of five sibling directories under the session,
# and WHICH ONE holds it is its state -- the runtime's own intake protocol
# (one os.rename per transition) read back, not a second bookkeeping copy.
# ``cancelled/`` is the fifth: an operator stop must seal as its own ending, or
# a later audit reads a human pressing stop as a capability the harness lacks.

_BRIEF_DIRS = (("inbox", "queued"), ("processing", "running"),
               ("done", "done"), ("failed", "failed"),
               ("cancelled", "cancelled"))

#: States a brief never leaves. Nothing waits on one, and nothing cancels one.
_BRIEF_TERMINAL = ("done", "failed", "cancelled", "unknown")

#: Long-poll bounds for brief_status -- read_runtime_frame's pattern at TASK
#: cadence: a state change here is a node boundary (seconds to hours), not a
#: 30ms frame dump, so the tick is coarse and the cap sits inside a normal RPC
#: budget. A caller that waits out the cap gets the CURRENT state back, never a
#: timeout error: "still running" is an answer, and the whole point of this face
#: is that a long task never surfaces to its caller as a failure.
_BRIEF_WAIT_CAP_MS = 30000
_BRIEF_WAIT_TICK_S = 0.5

#: How many of a brief's own events ride back in a status reply: the TAIL, so
#: the answer is "where is it now", not a transcript. runtime_events serves the
#: whole feed when the transcript is what you want.
_BRIEF_EVENTS = 20

#: Chain rows that NAME their brief, newest-first-wins. ``task.plan_complete``
#: is deliberately absent -- it carries no brief id, so a task's outcome is
#: attributed through the feed's claim boundary instead (see _brief_events).
_BRIEF_CHAIN_KINDS = ("runtime.task_error", "runtime.task_cancelled",
                      "runtime.campaign_scheduled", "runtime.rsi_scheduled")

#: The events that CLOSE a brief's segment in the operational feed.
_BRIEF_END_KINDS = ("task_done", "task_failed", "task_cancelled")


def _brief_locate(session_dir: Path, brief_id: str) -> tuple[str, Path | None]:
    """``(state, path)`` for one brief -- which intake dir holds it -- or
    ``("unknown", None)``. Name-addressed, so it goes through the shared
    safe_child guard: a ``../`` brief id can never stat outside the session."""
    for sub, state in _BRIEF_DIRS:
        path = safe_child(session_dir / sub, brief_id, Path.is_file)
        if path is not None:
            return state, path
    return "unknown", None


def _brief_queue(session_dir: Path) -> list[str]:
    """Inbox names in the order the runtime will CLAIM them -- mtime order,
    mirroring harness_runtime._pending. Sorting differently here would report a
    position the runtime does not honor."""
    try:
        entries = sorted((session_dir / "inbox").glob("*.json"),
                         key=lambda p: p.stat().st_mtime)
    except OSError:
        return []
    return [p.name for p in entries]


def _brief_events(events: list[dict], brief_id: str) -> list[dict]:
    """One brief's slice of the operational feed.

    The runtime claims briefs SERIALLY (one resident process per session --
    harness.opstream's single-flight note), so a ``task_claimed`` names the
    owner of every event until its task_done/task_failed/task_cancelled: the
    claim boundary IS the attribution, and no per-event brief id is needed on
    the hot path. The feed truncates per boot, so a brief finished before the
    last boot has an empty segment -- honest, and its directory still answers
    the state."""
    seg: list[dict] = []
    mine = False
    for e in events:
        if e.get("kind") == "task_claimed":
            mine = e.get("brief") == brief_id
        if mine:
            seg.append(e)
            if e.get("kind") in _BRIEF_END_KINDS:
                mine = False
    return seg


def _ahead_running_s(session_dir: Path, events: list[dict]) -> float:
    """How long the brief the runtime is running RIGHT NOW has been running.

    Queue position 2 behind a chain three hours in and queue position 2 behind
    one that just started are different facts; this is the difference. The clock
    is the runtime's own ``task_claimed`` event -- ``processing/<id>.json`` keeps
    its DROP mtime (os.rename does not touch it), so the file is no clock at all.
    0.0 when the runtime is idle (or the feed is unarmed)."""
    for e in reversed(events):
        if e.get("kind") != "task_claimed":
            continue
        # the newest claim IS the current one; if it is no longer in
        # processing/, the runtime finished it and is idle.
        if _brief_locate(session_dir, str(e.get("brief") or ""))[0] != "running":
            return 0.0
        return round(max(time.time() - float(e.get("ts") or 0.0), 0.0), 1)
    return 0.0


def _brief_outcome(session_dir: Path, brief_id: str,
                   seg: list[dict]) -> dict | None:
    """What the brief DID, or ``None`` while it has done nothing yet.

    Chain first, and only rows that NAME the brief (exact, and they survive the
    per-boot feed truncation): a rejected key, an operator cancel, a scheduled
    campaign/rsi. A finished TASK seals ``task.plan_complete``, which carries no
    brief id, so its outcome is the ``plan_complete`` EVENT from this brief's own
    segment. Either way the fields are copied VERBATIM -- nothing here re-decides
    success -- and the event path is live state: ``session()`` is the sealed read.
    """
    row = next((r for r in reversed(chain_rows(session_dir))
                if r["kind"] in _BRIEF_CHAIN_KINDS
                and r["data"].get("brief") == brief_id), None)
    if row is not None:
        return {"chain_kind": row["kind"], "chain_seq": row["seq"], **row["data"]}
    ev = next((e for e in reversed(seg) if e.get("kind") == "plan_complete"), None)
    return None if ev is None else {k: v for k, v in ev.items() if k != "seq"}


def _brief_selector(path: Path | None) -> str | None:
    """The brief's selector (its task, or its campaign name) read off the brief
    file wherever it currently sits. None when unreadable -- a status reply must
    never fail because a brief is mid-replace."""
    if path is None:
        return None
    try:
        brief = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return brief.get("task") or brief.get("campaign") if isinstance(brief, dict) else None


def brief_status(session_dir: str | Path, brief_id: str,
                 wait_ms: int = 0) -> dict:
    """Where one brief is and what it did -- the ONE call that answers "is my
    long task still running, and did it finish?".

    ``{state, brief_id, session, task, runtime, events, [queue_position,
    ahead_running_s | started_ts, running_s], [outcome]}`` where ``state`` is
    queued|running|stalled|done|failed|cancelled|unknown, derived from which
    intake directory holds the brief. ``events`` is the tail of the brief's own
    slice of the operational feed; ``outcome`` appears once there is one.

    ``stalled`` is the one state NOT read off a directory: an unfinished brief
    (queued or running) in a session with NO live runtime. The directory answer
    there is honest and useless -- "queued" for 21 hours reads as "be patient"
    when the truth is "nobody will ever claim this". ``stalled_from`` keeps the
    directory answer, ``runtime.reason`` says why, and both a queued brief nobody
    will claim and a ``processing/`` orphan left by a crash surface the same way,
    because they are the same problem. Wanted the raw directory state? It is
    ``stalled_from``.

    ``runtime`` is that session's liveness (runtime_liveness) -- carried on EVERY
    reply, so a caller polling one brief can never again report progress about a
    session whose runtime is gone.

    ``wait_ms`` long-polls exactly like read_runtime_frame: block up to that long
    (capped board-side) for the state to CHANGE, then answer with the current
    state. Waiting out the cap is not an error -- the reply says ``running`` and
    the caller may wait again. A brief already in a terminal state never blocks,
    and neither does a STALLED one: waiting 30s for a dead runtime to claim
    something is the exact non-answer this face exists to stop giving.

    Live state, never sealed evidence: no chain verify, and it never raises.
    """
    session_dir = Path(session_dir)
    state, path = _brief_locate(session_dir, brief_id)
    runtime = runtime_liveness(session_dir)
    if wait_ms > 0 and state not in _BRIEF_TERMINAL and runtime["alive"]:
        deadline = time.monotonic() + min(int(wait_ms), _BRIEF_WAIT_CAP_MS) / 1000.0
        while time.monotonic() < deadline:
            time.sleep(_BRIEF_WAIT_TICK_S)
            now, now_path = _brief_locate(session_dir, brief_id)
            if now != state:
                state, path = now, now_path
                break
    events = read_runtime_events(session_dir)["events"]
    seg = _brief_events(events, brief_id)
    out = {"state": state, "brief_id": brief_id, "session": session_dir.name,
           "task": _brief_selector(path), "runtime": runtime,
           "events": seg[-_BRIEF_EVENTS:]}
    if state in ("queued", "running") and not runtime["alive"]:
        out["state"], out["stalled_from"] = "stalled", state
    if state == "queued":
        queue = _brief_queue(session_dir)
        if brief_id in queue:
            out["queue_position"] = queue.index(brief_id) + 1
        out["ahead_running_s"] = _ahead_running_s(session_dir, events)
    elif state == "running":
        started = next((e["ts"] for e in seg if e.get("kind") == "task_claimed"), None)
        if started is not None:
            out["started_ts"] = started
            out["running_s"] = round(max(time.time() - started, 0.0), 1)
    outcome = _brief_outcome(session_dir, brief_id, seg)
    if outcome is not None:
        out["outcome"] = outcome
    return out


def cancel_brief(session_dir: str | Path, brief_id: str) -> dict:
    """Ask the resident runtime to stop one brief -- ``{brief_id, session, state,
    requested}``, plus ``error`` when there is nothing to cancel.

    This writes ONE live-state marker (``<session>/cancel/<brief_id>``, the
    shared atomic drop) and NOTHING else. The runtime owns every mutation that
    follows: it checks the marker at the claim and at each NODE BOUNDARY, kills a
    campaign/rsi subprocess GROUP, files the brief into ``cancelled/`` and seals
    the ``runtime.task_cancelled`` row. One writer for the hash chain, and the
    board stays what it is -- a reader that drops a flag.

    A brief already ``done``/``failed``/``cancelled`` is refused with "already
    <state>; nothing to cancel" and no marker is written; an unknown id likewise.
    Cooperative by construction: cancelling never tears an episode mid-rollout,
    so a marker is a request that lands at the next boundary, not an instant kill.
    """
    session_dir = Path(session_dir)
    state, _ = _brief_locate(session_dir, brief_id)
    out = {"brief_id": brief_id, "session": session_dir.name, "state": state,
           "requested": False}
    if state == "unknown":
        return {**out, "error": "unknown brief"}
    if state in _BRIEF_TERMINAL:
        return {**out, "error": f"already {state}; nothing to cancel"}
    # brief_id is proven safe by the locate above (safe_child), so this join
    # cannot escape the session.
    cancel_dir = session_dir / "cancel"
    cancel_dir.mkdir(parents=True, exist_ok=True)
    drop(cancel_dir, brief_id, json.dumps({"ts": time.time()}))
    return {**out, "requested": True}


def _chain_rows(log_dir: Path) -> tuple[list[dict], int]:
    """Whole rows in seq order + a mid-write skip count -- same partial-tolerant
    line loop as _index_rows: the runtime appends rows.jsonl line by line, so a
    live poll can catch a half-written trailing (or non-row) line. Skip it (and
    count it) rather than crash; the next poll gets the whole row."""
    rows: list[dict] = []
    skipped = 0
    path = log_dir / "rows.jsonl"
    if not path.exists():
        return rows, skipped
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(row, dict) and "kind" in row and "data" in row:
            rows.append(row)
        else:
            skipped += 1
    return rows, skipped


def _session_rows(log_dir: Path) -> tuple[dict[str, list[dict]], int]:
    """Row ``data`` payloads grouped by kind, plus a skip count. Thin regroup
    over _chain_rows (which does the partial-tolerant read)."""
    rows, skipped = _chain_rows(log_dir)
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row["data"])
    return by_kind, skipped


def chain_rows(session_dir: str | Path) -> list[dict]:
    """Whole session chain rows (seq/kind/data/...) in order, mid-write tolerant.
    read_session groups payloads by kind and drops seq; this keeps rows whole so
    a reader can attribute one to a brief (used by mcp_server.run_task)."""
    return _chain_rows(Path(session_dir) / "session-log")[0]


def _session_mtime(log_dir: Path) -> float:
    """rows.jsonl mtime -- the LIVE signal; a new note appends to it."""
    try:
        return (log_dir / "rows.jsonl").stat().st_mtime
    except OSError:
        return 0.0


def read_session(session_dir: str | Path) -> dict:
    """One runtime session: note payloads grouped by kind, a mid-write skip
    count, and whether the hash chain still verifies.

    ``chain_ok`` reuses the writer's own SessionLog.load().verify() -- the one
    audited chain check -- rather than reimplementing the hash fold. A truncated
    trailing line (mid-write) makes load() choke, so it is caught and read as
    "not currently verifiable" (False); the skip count disambiguates that from a
    real tamper in the UI, and the next poll recovers.
    """
    session_dir = Path(session_dir)
    log_dir = session_dir / "session-log"
    by_kind, skipped = _session_rows(log_dir)
    try:
        chain_ok = SessionLog.load(log_dir).verify()
    except (OSError, json.JSONDecodeError):
        chain_ok = False
    return {"name": session_dir.name, "mtime": _session_mtime(log_dir),
            "chain_ok": chain_ok, "skipped": skipped,
            "kinds": {k: len(v) for k, v in by_kind.items()}, "rows": by_kind}


def runtime_python(session_dir: str | Path, pid) -> str | None:
    """The interpreter path iff ``pid`` is RIGHT NOW the harness_runtime serving
    ``session_dir``, else ``None``. The one liveness guard behind every face.

    A status file is a leftover, not a promise: one named pid 4086108 for three
    days after that process died, an agent read "runtime up", and an operator's
    brief rotted in a dead inbox. So the pid is checked against /proc, and
    checked STRUCTURALLY -- a substring scan for "harness_runtime.py" matches
    the very shell that is grepping for it, which is why the model-server face
    reads ``/proc/<pid>/exe`` instead:

    * ``argv[0]`` must be an existing ``python*`` file -- kills a recycled pid
      that landed on grep/ps/an editor. It is made absolute against THAT
      process's cwd but never ``resolve()``d: the venv's ``bin/python`` symlink
      is exactly what discriminates the robocasa runtime from the base one, and
      its target is identical for both.
    * some later arg must name ``harness_runtime.py`` -- kills an unrelated python.
    * some later arg must resolve to THIS session dir -- so a sibling runtime
      cannot answer for us, and ``session-robocasa`` never answers for
      ``session-robocasa-rsi`` (the prefix collision cockpit's find_runtime hit).

    Anything unreadable -- an exited pid, another user's process, a corrupt
    status file -- reads as ``None``: this layer never claims a liveness it
    cannot see.
    """
    try:
        pid = int(pid)
        argv = [a.decode(errors="replace")
                for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
        cwd = Path(f"/proc/{pid}/cwd").readlink()
    except (OSError, TypeError, ValueError):
        return None
    if not any(a.endswith("harness_runtime.py") for a in argv[1:]):
        return None
    target = Path(session_dir).resolve()
    # `--session-dir X` and `--session-dir=X` both put X in one arg once the
    # optional prefix is stripped; a relative one resolves against the RUNTIME's
    # cwd, never ours.
    if not any((cwd / a.removeprefix("--session-dir=")).resolve() == target
               for a in argv[1:]):
        return None
    exe = cwd / argv[0]              # an absolute argv[0] absorbs the cwd (pathlib)
    return str(exe) if exe.name.startswith("python") and exe.is_file() else None


def read_runtime_status(session_dir: str | Path) -> dict | None:
    """The resident runtime's LIVE status for one session
    (``<session>/runtime_status.json``: pid/render/mode/boot_ts/display),
    overwritten each boot. ``None`` when absent (a session that has not booted
    since the file existed) or mid-write -- a plain read, not a chain row: this
    is live operational state, not sealed evidence, so no chain verify.

    Two fields are DERIVED here, because the file on disk cannot know them and
    every reader that judged the pid for itself judged it wrong:

    * ``alive`` -- is that pid this session's harness_runtime right now
      (``runtime_python``)? A runtime that announced its own exit
      (``stopped_ts``) is dead too.
    * ``heartbeat_age_s`` -- seconds since the last stamp, or ``None`` on a file
      written before heartbeats existed. The second axis: ``alive`` says the
      process exists, this says how long since it last stamped -- together they
      separate "idle and listening" from "busy" from "wedged" from "dead".
    """
    path = Path(session_dir) / "runtime_status.json"
    try:
        status = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(status, dict):
        return None
    beat = status.get("heartbeat_ts")
    status["heartbeat_age_s"] = (round(time.time() - beat, 1)
                                 if isinstance(beat, (int, float)) else None)
    status["alive"] = (not status.get("stopped_ts")
                       and runtime_python(session_dir, status.get("pid")) is not None)
    return status


#: Long-poll bounds for read_runtime_frame: the blocking wait is capped (a
#: wedged caller frees itself inside the RPC timeout budget) and the file is
#: re-stat'ed on a short tick (the writer dumps ~every 30ms, so 10ms keeps the
#: added detection latency under half a frame).
_FRAME_WAIT_CAP_MS = 2000
_FRAME_WAIT_TICK_S = 0.01


def read_runtime_frame(session_dir: str | Path, after_ts: float = 0.0,
                       wait_ms: int = 0) -> dict:
    """The resident runtime's LIVE viewport frame for one session
    (``<session>/frame.jpg``, written by scripts/frame_dump.py: overwritten in
    place while a task runs). Same live-state family as runtime_status: a plain
    read, never a chain row, no verify.

    Returns ``{"jpeg_b64": ..., "ts": <mtime>, "age_s": <now - mtime>}`` -- the
    b64 is encoded HERE so every face ships the same dict and the TS side only
    decodes (zero business logic downstream). ``age_s`` lets a poller show a
    stale-frame placeholder without trusting its own clock against the file's.

    ``after_ts`` is the poller's cursor (the ``ts`` it last displayed): when the
    file has not changed since, the reply is the short
    ``{"unchanged": true, "ts": ..., "age_s": ...}`` -- no read, no b64. The
    cursor compares against the same round(mtime, 3) the full reply carries.

    ``wait_ms`` turns the read into a LONG POLL: instead of answering an
    unchanged (or absent) file immediately, re-stat it every
    ``_FRAME_WAIT_TICK_S`` until it changes or ``wait_ms`` (capped at
    ``_FRAME_WAIT_CAP_MS``) elapses, THEN answer as usual. The browser viewport
    re-issues the call the moment a reply lands, so its to-hand fps tracks the
    writer's dump rate with zero idle polling; ``wait_ms=0`` keeps the old
    immediate-answer behavior on every face.
    Absent or unreadable file (including mid-replace) -> ``{"error": "no frame"}``.
    """
    path = Path(session_dir) / "frame.jpg"
    deadline = (time.monotonic() + min(int(wait_ms), _FRAME_WAIT_CAP_MS) / 1000.0
                if wait_ms > 0 else None)
    while True:
        try:
            ts = round(path.stat().st_mtime, 3)
        except OSError:
            if deadline is not None and time.monotonic() < deadline:
                time.sleep(_FRAME_WAIT_TICK_S)
                continue
            return {"error": "no frame"}
        if after_ts and ts <= after_ts:
            if deadline is not None and time.monotonic() < deadline:
                time.sleep(_FRAME_WAIT_TICK_S)
                continue
            return {"unchanged": True, "ts": ts,
                    "age_s": round(max(time.time() - ts, 0.0), 3)}
        try:
            raw = path.read_bytes()
            # re-stat AFTER the read: an os.replace between stat and read would
            # pair the new bytes with the old ts and wedge the cursor one frame back.
            ts = round(path.stat().st_mtime, 3)
        except OSError:
            return {"error": "no frame"}
        return {"jpeg_b64": base64.b64encode(raw).decode("ascii"),
                "ts": ts, "age_s": round(max(time.time() - ts, 0.0), 3)}


def read_runtime_rollout(session_dir: str | Path) -> dict:
    """Latest task rollout MP4 for one session, as live downloadable state."""
    path = Path(session_dir) / "rollout.mp4"
    try:
        raw = path.read_bytes()
        stat = path.stat()
    except OSError:
        return {"error": "no rollout video"}
    return {"mp4_b64": base64.b64encode(raw).decode("ascii"),
            "ts": round(stat.st_mtime, 3), "size": stat.st_size}


def read_runtime_keyframes(session_dir: str | Path) -> dict:
    """The INDEX of one session's live keyframe stills
    (``<session>/keyframes/<seq:06d>-<kind>.jpg``, written by
    scripts/frame_dump's opstream listener and cleared by every boot).

    Returns ``{"frames": [{"seq", "kind", "ts"}, ...], "count": n}`` ordered by
    seq. Index ONLY -- no image bytes -- so a panel can poll it at event cadence
    for pennies and fetch a still on demand through read_runtime_keyframe. The
    seq is the runtime_events cursor, so a keyframe pins to the event a panel
    already draws.

    Same live-state family as read_runtime_frame: a plain directory read, never
    a chain row, no verify. An absent directory reads as an empty index --
    deleting the whole thing is legal and loses zero evidence.
    """
    frames: list[dict] = []
    try:
        entries = list((Path(session_dir) / "keyframes").iterdir())
    except OSError:
        return {"frames": [], "count": 0}
    for path in entries:
        seq, _, kind = path.stem.partition("-")
        if path.suffix != ".jpg" or not seq.isdigit() or not kind:
            continue  # a .tmp mid-publish, or anything else that wandered in
        try:
            ts = round(path.stat().st_mtime, 3)
        except OSError:
            continue
        frames.append({"seq": int(seq), "kind": kind, "ts": ts})
    frames.sort(key=lambda f: f["seq"])
    return {"frames": frames, "count": len(frames)}


def read_runtime_keyframe(session_dir: str | Path, seq: int) -> dict:
    """One keyframe still by its event seq: ``{"jpeg_b64", "seq", "kind"}``, or
    ``{"error": "no keyframe"}`` when that seq holds none (never captured, or
    the boot that captured it has been cleared -- the same non-event to a
    reader).

    The b64 is encoded HERE, like read_runtime_frame, so all three faces ship
    the identical dict and the TS side only decodes. ``kind`` comes off the
    filename, so the index and the image can never disagree.
    """
    try:
        hits = sorted((Path(session_dir) / "keyframes").glob(f"{int(seq):06d}-*.jpg"))
    except (OSError, ValueError):
        return {"error": "no keyframe"}
    for path in hits:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        return {"jpeg_b64": base64.b64encode(raw).decode("ascii"),
                "seq": int(seq), "kind": path.stem.partition("-")[2]}
    return {"error": "no keyframe"}


def read_runtime_events(session_dir: str | Path, after_seq: int = 0) -> dict:
    """The resident runtime's OPERATIONAL event feed for one session
    (``<session>/runtime_events.jsonl``, written by harness.opstream: truncated
    per boot, one JSON line per event) -- events with ``seq > after_seq`` plus
    ``last_seq``, the newest seq in the file. Live state, not sealed evidence:
    no chain verify, absent file reads as an empty feed.

    Cursor contract: a poller passes its last-seen seq and appends the returned
    events; ``last_seq < after_seq`` means the runtime re-booted (the file was
    truncated and seq restarted), so the poller resets its cursor to 0 and
    re-reads. Same partial-tolerant line loop as the other jsonl readers: a
    half-written trailing line is skipped and picked up whole on the next poll.
    """
    path = Path(session_dir) / "runtime_events.jsonl"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {"events": [], "last_seq": 0}
    events: list[dict] = []
    last_seq = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not (isinstance(row, dict) and isinstance(row.get("seq"), int)):
            continue
        last_seq = max(last_seq, row["seq"])
        if row["seq"] > after_seq:
            events.append(row)
    return {"events": events, "last_seq": last_seq}


def trajectories(session_dir: str | Path, *, role_of_seed=None) -> list[dict]:
    """Protocol-v0 trajectory samples: a PURE projection of one session's chain
    rows, one sample per plan/replan decision (``task.plan``), with that graph's
    ``task.verify`` results folded into ``o.verify`` and the episode's
    ``task.plan_complete`` stamping success/replans on every sample it closes.
    ``o.L`` = nodes done before the decision + nodes verified all-true under it.
    ``role_of_seed(seed) -> "dev" | "heldout"`` fills ``o.role`` when given;
    otherwise the role is derived from ``burned_blocks(session_dir.parent)``
    (heldout iff the seed sits in a burned heldout block), and ``o.role_source``
    says which (``caller`` / ``burned_blocks`` / ``no_store`` -> all dev)."""
    role_source = "caller"
    if role_of_seed is None:
        try:
            held = [(lo, hi) for lo, hi, role, _ in burned_blocks(Path(session_dir).parent)
                    if role == "heldout"]
            role_source = "burned_blocks"
        except ValueError:
            held, role_source = [], "no_store"
        role_of_seed = lambda s: "heldout" if any(lo <= s <= hi for lo, hi in held) else "dev"
    episode: list[Trajectory] = []
    out: list[Trajectory] = []
    for row in chain_rows(session_dir):
        kind, d = row["kind"], row["data"]
        if kind == "task.plan":
            seed = d.get("seed")
            x = {"mission": d.get("mission"), "sigma0": d.get("sigma0"),
                 "skills": list(d.get("skills") or []),
                 "visible": list(d.get("visible") or []),
                 "show_evidence": bool(d.get("show_evidence")),
                 "done": list(d.get("done") or []), "fault": d.get("fault")}
            y = {"graph": d.get("graph_id"), "rationale": d.get("rationale") or ""}
            o = {"legal": bool(d.get("legal")), "verify": {}, "L": len(x["done"]),
                 "success": None, "replans": None, "seed": seed, "block": d.get("block"),
                 "role": role_of_seed(seed) if seed is not None else None,
                 "role_source": role_source}
            episode.append(Trajectory(x, y, o))
        elif kind == "task.verify" and episode:
            o = episode[-1].o
            o["verify"][d["node"]] = d["results"]
            if all(v is True for v in d["results"].values()):
                o["L"] += 1
        elif kind == "task.plan_complete":
            for t in episode:
                t.o["success"] = bool(d.get("success"))
                t.o["replans"] = d.get("replans")
            out += episode
            episode = []
    return [{"id": t.id, **to_plain(t)} for t in out + episode]


def plan_index(session_dir: str | Path) -> list[dict]:
    """Per (task, graph_sha, embodiment, arm) plan evidence, a PURE projection of
    one session's chain: one episode per legal replan-0 ``task.plan`` row that
    carries ``graph_sha`` (older rows have none and are skipped), ``k`` from the
    closing ``task.plan_complete``, ``L`` = nodes verified all-true under that
    row. Rows sorted by key; each carries the graph (meta stripped) and the
    first row's facts/objects so a publisher can re-run Legal(G) on it."""
    keys, cur = {}, None
    for row in chain_rows(session_dir):
        kind, d = row["kind"], row["data"]
        if kind == "task.plan":
            cur = None
            if d.get("legal") and d.get("replan") == 0 and d.get("graph_sha"):
                key = (d.get("mission"), d["graph_sha"], d.get("embodiment"), d.get("arm"))
                cur = keys.setdefault(key, {
                    "task": key[0], "graph_sha": key[1], "embodiment": key[2], "arm": key[3],
                    "graph": {k: v for k, v in (d.get("graph") or {}).items()
                              if k not in ("planner", "rationale")},
                    "facts": list(d.get("facts") or []), "objects": list(d.get("objects") or []),
                    "seeds": [], "blocks": [], "seqs": [], "_L": [], "n": 0, "k": 0})
                cur["seeds"].append(d.get("seed"))
                if d.get("block") is not None and d["block"] not in cur["blocks"]:
                    cur["blocks"].append(d["block"])
                cur["seqs"].append(row.get("seq"))
                cur["_L"].append(0)
                cur["n"] += 1
        elif kind == "task.verify" and cur is not None:
            if all(v is True for v in d["results"].values()):
                cur["_L"][-1] += 1
        elif kind == "task.plan_complete" and cur is not None:
            cur["k"] += bool(d.get("success"))
            cur = None
    out = []
    for key in sorted(keys, key=lambda k: tuple(str(x) for x in k)):
        e = keys[key]
        L = e.pop("_L")
        e["L_mean"] = round(sum(L) / len(L), 4) if L else 0.0
        out.append(e)
    return out


def skill_evidence(session_dir: str | Path) -> list[dict]:
    """Per (skill, embodiment, executor) skill evidence {n, k}, a PURE projection
    of one session's ``task.verify`` seal rows: ``n`` counts verified nodes,
    ``k`` those whose bound predicates were all true. The node's skill comes from
    the enclosing ``task.plan`` graph, embodiment from that row, executor from
    the verify row's own ``executor`` key (``scripted`` on older rows that seal
    only ``driver``). Rows sorted by key; feeds bindings.<emb>.policies.<key>
    evidence (record.evidence[<emb>].by_executor) at publish time."""
    keys: dict[tuple, dict] = {}
    skill_of, emb = {}, None
    for row in chain_rows(session_dir):
        kind, d = row["kind"], row["data"]
        if kind == "task.plan":
            emb = d.get("embodiment")
            skill_of = {n.get("id"): n.get("skill") for n in (d.get("graph") or {}).get("nodes", [])}
        elif kind == "task.verify" and d.get("node") in skill_of:
            key = (skill_of[d["node"]], emb, d.get("executor") or "scripted")
            e = keys.setdefault(key, {"skill": key[0], "embodiment": key[1],
                                      "executor": key[2], "n": 0, "k": 0})
            e["n"] += 1
            e["k"] += all(v is True for v in d["results"].values())
    return [keys[k] for k in sorted(keys, key=lambda k: tuple(str(x) for x in k))]


def _record_rows(paths) -> dict[str, dict]:
    """name -> plain SkillRecordV0 dict for every readable record-shaped JSON
    file (has ``name`` and ``bindings``; capability/plan/recovery rows in the
    same skills root are skipped). Later paths overwrite earlier same-name rows."""
    out: dict[str, dict] = {}
    for path in paths:
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and isinstance(d.get("name"), str) and isinstance(d.get("bindings"), dict):
            out[d["name"]] = d
    return out


def skills(session_dir: str | Path) -> list[dict]:
    """Records overview, one row per skill sorted by name: ``{name, kind,
    description, bindings: {emb: [executor keys]}, evidence: {emb: {n, k,
    by_executor: {key: {n, k}}}}, limits, failure_modes, source}``.

    The library records (``skill-library/records``, what the runtime mounts)
    overlaid by the session's published copies (``<session>/skills/*.json``,
    the evolution-only write path: evidence.by_executor and tunables land
    there), so ``source`` says which one a row reflects. Executor keys follow
    protocol.executors_of: ``scripted`` always, plus the binding's ``policies``
    keys. A pure read; no session -> the library alone."""
    from harness.skill_library import ROOT as records_root  # loads the library once
    recs = _record_rows(sorted(records_root.glob("*.json")))
    session_root = Path(session_dir) / "skills"
    published = _record_rows(sorted(session_root.glob("*.json"))) if session_root.is_dir() else {}
    out = []
    for name in sorted(set(recs) | set(published)):
        d = published.get(name) or recs[name]
        bindings = {emb: sorted({"scripted", *((b or {}).get("policies") or {})})
                    for emb, b in sorted((d.get("bindings") or {}).items())}
        evidence = {}
        for emb, ev in sorted((d.get("evidence") or {}).items()):
            ev = ev or {}
            evidence[emb] = {"n": ev.get("n", 0), "k": ev.get("k", 0),
                             "by_executor": {key: {"n": r.get("n", 0), "k": r.get("k", 0)}
                                             for key, r in sorted((ev.get("by_executor") or {}).items())}}
        out.append({"name": name, "kind": d.get("kind", "segment"),
                    "description": d.get("description", ""),
                    "bindings": bindings, "evidence": evidence,
                    "limits": d.get("limits") or {},
                    "failure_modes": list(d.get("failure_modes") or []),
                    "source": "session" if name in published else "library"})
    return out


def split_trajectories(samples: list[dict]) -> dict[str, list[dict]]:
    """``{"dev": [...], "heldout": [...]}`` by each sample's ``o.role``."""
    return {r: [t for t in samples if t["o"]["role"] == r] for r in ("dev", "heldout")}


def export_trajectories(session_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Write ``<out_dir>/dev.jsonl`` and ``heldout.jsonl`` (one sample per line,
    ids stable) from ``trajectories(session_dir)``; returns the per-role counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for role, rows in split_trajectories(trajectories(session_dir)).items():
        (out_dir / f"{role}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in rows))
        counts[role] = len(rows)
    return counts


def cancelled_run(row: dict) -> bool:
    """True for a task result the operator actually STOPPED: the workload's
    node-boundary cancel folds in as a ``cancelled`` FAULT, so the run says so
    itself -- no second bookkeeping file, and an auditor reading the chain alone
    reaches the same conclusion this function does.

    Shared with harness_runtime, which asks it of the live ``workload.run``
    return (same ``faults`` shape) before filing a brief as cancelled. One
    predicate: if the runtime and this tally ever disagreed about what a
    cancellation is, the evidence would contradict the count."""
    return any(f.get("kind") == "cancelled" for f in row.get("faults") or [])


def session_progress(session_dir: str | Path) -> dict:
    """Mission-progress aggregate over one session's ``task.plan_complete`` rows.

    The mission→task→stage tree already lives in ``read_session``'s rows; this is
    the one place that FOLDS it into counts (task tallies, total replans/faults,
    stage pass-rate) so the cockpit's progress panel renders numbers it never
    computes -- the charter's hard rule that statistics live in board/, not the
    fork's TypeScript. Same mid-write-tolerant read as read_session (a live poll
    can land between appended rows); a partial trailing row is skipped, not fatal.

    ``stage_pass_rate`` is ``stages_passed / stages`` over every stage of every
    task node, or ``None`` when no stage has run yet (an empty tree is honest
    "no rate", never a fabricated 0/0). ``latest`` is the newest plan_complete
    row (append order) shaped for the pipeline view -- its goal, node tree, and
    run tallies -- or ``None`` before the first task seals, so the panel shows an
    idle state rather than inventing one. ``task_errors`` counts
    ``runtime.task_error`` rows (rejected briefs), a fault class distinct from a
    node failure inside a sealed plan.

    A run the OPERATOR stopped is tallied APART, in ``cancelled``, and kept out
    of tasks/succeeded/failed: a human pressing stop is not evidence about what
    the harness can do, and folding it into the failure tally is exactly how a
    later audit misreads an interruption as a capability gap. Its stages still
    count -- those rollouts really ran -- and ``latest`` still shows it, because
    the newest thing that happened is the newest thing that happened.
    """
    session_dir = Path(session_dir)
    by_kind, _skipped = _session_rows(session_dir / "session-log")
    sealed = by_kind.get("task.plan_complete", [])
    runs = [r for r in sealed if not cancelled_run(r)]
    succeeded = sum(1 for r in runs if r.get("success"))
    stages = stages_passed = 0
    for r in sealed:
        for node in (r.get("nodes") or {}).values():
            for stage in node.get("stages") or []:
                stages += 1
                if stage.get("success"):
                    stages_passed += 1
    latest = sealed[-1] if sealed else None
    return {
        "name": session_dir.name,
        "tasks": len(runs),
        "succeeded": succeeded,
        "failed": len(runs) - succeeded,
        "cancelled": len(sealed) - len(runs),
        "replans": sum(int(r.get("replans") or 0) for r in runs),
        "faults": sum(len(r.get("faults") or []) for r in runs),
        "task_errors": len(by_kind.get("runtime.task_error", [])),
        "stages": stages,
        "stages_passed": stages_passed,
        "stage_pass_rate": (stages_passed / stages) if stages else None,
        "latest": None if latest is None else {
            "goal": latest.get("goal"),
            "nodes": latest.get("nodes") or {},
            "success": latest.get("success"),
            "replans": latest.get("replans"),
            "actuations": latest.get("actuations"),
            "faults": latest.get("faults") or [],
        },
    }


def suite_result(session_dir: str | Path, sha: str | None = None) -> dict | None:
    """The sealed suite artifact of one session: ``<session>/suites/<sha>.json``,
    ``sha`` defaulting to the newest ``suite.sealed`` chain row's. None when the
    session sealed no suite (or the sha names no artifact / is not a hex digest)."""
    session_dir = Path(session_dir)
    if sha is None:
        sealed = _session_rows(session_dir / "session-log")[0].get("suite.sealed", [])
        if not sealed:
            return None
        sha = sealed[-1].get("sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        return None
    try:
        return json.loads((session_dir / "suites" / f"{sha}.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _campaign(session_dir: str | Path, task: str) -> dict | None:
    """``<session>/campaigns/evolve-<task>/campaign.json`` (scripts/evolve.py's
    atomic snapshot), or None when absent/unreadable. ``task`` rides the shared
    safe_child guard so a ``../`` task can never read outside the session."""
    path = safe_child(Path(session_dir) / "campaigns", f"evolve-{task}", Path.is_dir)
    if path is None:
        return None
    try:
        doc = json.loads((path / "campaign.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def rsi_run(session_dir: str | Path, task: str) -> dict | None:
    """One evolve campaign's state: the campaign.json fields (task, session,
    seeds, arm, best, cursor, status, rounds -- each round carrying ``per_seed``
    and ``needs``) plus ``latest`` (the newest round row, or None before the
    first lands), ``live`` (scripts/evolve.py's in-flight block: phase, round,
    seed/seed_index/seeds_total, node, nodes (the seed's node trail), seed_started_at,
    per_seed_partial, tried, message, messages (last 20), timings --
    live state, never sealed; null when the file predates it) and ``open_brief``
    (the inbox/processing evolve brief id for this task, so the page can stop it
    after a restart; null when none). None when no campaign exists."""
    doc = _campaign(session_dir, task)
    if doc is None:
        return None
    rounds = doc.get("rounds") or []
    return {**doc, "latest": rounds[-1] if rounds else None, "live": doc.get("live"),
            "open_brief": _open_brief(Path(session_dir), task)}


def _open_brief(session_dir: Path, task: str) -> str | None:
    """The brief id (intake filename) of the evolve brief still driving ``task``
    -- ``processing/`` first, then ``inbox/`` -- or None. Read from the intake
    dirs, not the per-boot feed, so a console restart can still cancel it."""
    for sub in ("processing", "inbox"):
        d = session_dir / sub
        for p in sorted(d.glob("*.json")) if d.is_dir() else ():
            try:
                b = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(b, dict) and b.get("kind") == "evolve" and b.get("task") == task:
                return p.name
    return None


def rsi_campaigns(session_dir: str | Path) -> list[dict]:
    """Every evolve campaign the session holds on disk (``campaigns/evolve-*/
    campaign.json``, so it survives a restart -- the per-boot feed does not):
    ``{task, status, cursor, rounds (count), best, seeds, arm, node_rate_best (the
    series' running-max node pass rate | null), updated
    (campaign.json mtime), live: {phase, message, nodes_done "k/n" | null} | null, open_brief}``,
    running first, then newest ``updated`` first."""
    session_dir = Path(session_dir)
    out = []
    for d in (session_dir / "campaigns").glob("evolve-*"):
        task = d.name[len("evolve-"):]
        doc = _campaign(session_dir, task)
        if doc is None:
            continue
        live = doc.get("live")
        out.append({
            "task": doc.get("task", task), "status": doc.get("status"),
            "cursor": doc.get("cursor"), "rounds": len(doc.get("rounds") or []),
            "best": doc.get("best"), "seeds": doc.get("seeds"), "arm": doc.get("arm"),
            "node_rate_best": (_series(doc) or [{}])[-1].get("node_rate", {}).get("best"),
            "updated": (d / "campaign.json").stat().st_mtime,
            "live": ({"phase": live.get("phase"), "message": live.get("message"),
                      "nodes_done": (f"{sum(n.get('ok') is True for n in live['nodes'])}/{len(live['nodes'])}"
                                     if live.get("nodes") else None)}
                     if isinstance(live, dict) else None),
            "open_brief": _open_brief(session_dir, task)})
    out.sort(key=lambda c: (c["status"] != "running", -c["updated"]))
    return out


def _rates(rows) -> tuple[float | None, dict]:
    """(mean node pass rate, {task: pass fraction}) over one per_seed list; a task
    passes for a seed when every node carrying that ``task`` is ok=true. (None, {})
    when no row carries nodes (rounds older than the trail)."""
    rows = [r for r in rows or () if r.get("nodes")]
    if not rows:
        return None, {}
    rate = sum(sum(n.get("ok") is True for n in r["nodes"]) / len(r["nodes"]) for r in rows) / len(rows)
    passes: dict[str, int] = {}
    for r in rows:
        per = {}
        for n in r["nodes"]:
            if n.get("task"):
                per[n["task"]] = per.get(n["task"], True) and n.get("ok") is True
        for t, ok in per.items():
            passes[t] = passes.get(t, 0) + ok
    return round(rate, 4), {t: round(k / len(rows), 4) for t, k in passes.items()}


def _series(doc: dict) -> list[dict]:
    out, best = [], None
    for r in doc.get("rounds") or []:
        nb, tb = _rates(r.get("per_seed"))
        na, ta = _rates(r.get("after_seeds"))
        cur = na if na is not None else nb
        best = cur if best is None or (cur is not None and cur > best) else best
        out.append({**{k: r.get(k) for k in ("round", "before", "after", "best", "per_seed", "needs")},
                    "node_rate": {"before": nb, "after": na, "best": best},
                    "by_task": {t: {"before": tb.get(t), "after": ta.get(t)} for t in sorted(set(tb) | set(ta))}})
    return out


def rsi_series(session_dir: str | Path, task: str) -> list[dict]:
    """Per-round {round, before, after, best, per_seed, needs, node_rate, by_task} of
    one evolve campaign, in order (the line-chart feed; ``per_seed`` = the kept suite's
    [{seed, success, first_death, failure_mode, nodes}], ``needs`` = what would unblock a
    round that tried nothing; ``node_rate`` = {before, after, best}: mean over seeds of
    ok-nodes/nodes (before from per_seed, after from after_seeds, best = running max of
    after-or-before); ``by_task`` = {task: {before, after}} pass fraction of each sub-task
    (every node of that task ok). Rounds without nodes read as null / {}). [] when no
    campaign exists."""
    return _series(_campaign(session_dir, task) or {})


def rsi_frames(session_dir: str | Path, task: str, round: int) -> list[str]:
    """The kept keyframe/video paths one evolve round recorded (``media`` of that
    round row, session-relative). [] when the campaign or round is absent."""
    doc = _campaign(session_dir, task) or {}
    for r in doc.get("rounds") or []:
        if r.get("round") == round:
            return [m for m in r.get("media") or [] if isinstance(m, str)]
    return []


# --- proposals inbox --------------------------------------------------------
#
# ``<session>/proposals/<id>.json``: what an outside proposer (the skill-author
# preset, an operator) asks the lightweight evolve loop to try next. One entry =
# ``{task, kind, payload, note}``; scripts/evolve.py consumes the oldest pending
# one for its task at the start of each round and stamps ``applied`` in place.

PROPOSAL_KINDS = ("tunables", "executor", "card")


def _proposal_error(doc) -> str | None:
    """The shape gate at the trust boundary (an LLM writes these): exactly
    ``{task:str, kind:PROPOSAL_KINDS, payload:dict, note?:str}``."""
    if not isinstance(doc, dict):
        return "proposal must be a JSON object"
    extra = set(doc) - {"task", "kind", "payload", "note"}
    if extra:
        return f"unknown proposal keys {sorted(extra)}"
    if not isinstance(doc.get("task"), str) or not doc["task"]:
        return "proposal needs task: str"
    if doc.get("kind") not in PROPOSAL_KINDS:
        return f"proposal kind must be one of {list(PROPOSAL_KINDS)}"
    if not isinstance(doc.get("payload"), dict):
        return "proposal needs payload: object"
    if not isinstance(doc.get("note", ""), str):
        return "proposal note must be a string"
    return None


def submit_proposal(session_dir: str | Path, raw: str) -> dict:
    """Atomically drop ``raw`` (a proposal JSON string) into ``<session>/proposals/``.
    The ONE write both faces share: ``{"submitted": <id>, "inbox": <dir>}`` or
    ``{"error": ...}`` when the shape gate rejects it. Ids sort in submission order."""
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"error": f"proposal is not JSON: {exc}"}
    err = _proposal_error(doc)
    if err:
        return {"error": err}
    inbox = Path(session_dir) / "proposals"
    inbox.mkdir(parents=True, exist_ok=True)
    pid = f"proposal-{time.time_ns():019d}-{uuid.uuid4().hex[:8]}"
    drop(inbox, f"{pid}.json", json.dumps(
        {"task": doc["task"], "kind": doc["kind"], "payload": doc["payload"],
         "note": doc.get("note", ""), "applied": None}, sort_keys=True))
    return {"submitted": pid, "inbox": str(inbox)}


def proposals(session_dir: str | Path) -> list[dict]:
    """Every proposal in the session's inbox, oldest first:
    ``{id, task, kind, payload, note, applied}`` -- ``applied`` is None while
    pending, else ``{round, ts}`` stamped by scripts/evolve.py. [] when none."""
    inbox = Path(session_dir) / "proposals"
    out = []
    for p in sorted(inbox.glob("proposal-*.json")) if inbox.is_dir() else ():
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(doc, dict):
            out.append({"id": p.stem, "task": doc.get("task"), "kind": doc.get("kind"),
                        "payload": doc.get("payload"), "note": doc.get("note", ""),
                        "applied": doc.get("applied")})
    return out


def discover_sessions(runs_dir: str | Path) -> list[dict]:
    """Every runtime session under runs_dir, newest first -- summary cards (no
    row payloads) for the sidebar. Sessions are tiny, so this reads each once.

    ``mode`` (execution/evolution, None before the first boot) rides along so a
    panel can offer only evolution sessions for an evolve brief; ``runtime_alive`` because this is the FIRST call anyone makes
    ("Unsure? Call sessions()"): a session with no live runtime is an inbox
    nothing will ever claim from, and finding that out only after the brief has
    sat there for a day is how the last three incidents went. Deliberately the
    bool and not the heartbeat age -- these rows are compared byte-for-byte
    across the three faces, and an age would jitter between two calls."""
    runs_dir = Path(runs_dir)
    out = []
    for p in sorted(runs_dir.iterdir()):
        if p.is_dir() and is_session(p):
            s = read_session(p)
            status = read_runtime_status(p)
            out.append({k: s[k] for k in ("name", "mtime", "chain_ok", "kinds", "skipped")}
                       | {"runtime_alive": bool(status and status["alive"]),
                          "mode": (status or {}).get("mode")})
    return sorted(out, key=lambda s: s["mtime"], reverse=True)


# --- host vitals ------------------------------------------------------------

#: nvidia-smi budget. A wedged driver (a hung GPU reset) must not stall the poll
#: behind it; the panel simply shows no GPU that tick and recovers on the next.
_NVSMI_TIMEOUT_S = 2.0


def _nvidia_smi(query: str) -> list[list[str]]:
    """One ``nvidia-smi --query-<...> --format=csv,noheader,nounits`` read, split
    into stripped cells. Any failure -- no driver, no binary, nonzero exit, a
    timeout -- reads as no rows, because a box without an NVIDIA GPU is a normal
    deployment, not an error."""
    try:
        proc = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=_NVSMI_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [[c.strip() for c in line.split(",")] for line in proc.stdout.splitlines() if line.strip()]


def _gpus() -> list[dict]:
    """Every visible NVIDIA GPU with its memory and its compute processes,
    biggest consumer first. Two nvidia-smi reads joined on the GPU uuid --
    ``--query-compute-apps`` has no index column, so the uuid is the only key
    that pairs a process with the card it sits on. Unparsable rows are dropped
    (a driver that answers ``[N/A]`` for a field is not an outage)."""
    gpus: dict[str, dict] = {}
    for row in _nvidia_smi("gpu=uuid,index,name,memory.used,memory.total"):
        if len(row) < 5:
            continue
        try:
            entry = {"index": int(row[1]), "name": row[2],
                     "used_mib": int(row[3]), "total_mib": int(row[4]), "procs": []}
        except ValueError:
            continue
        gpus[row[0]] = entry
    for row in _nvidia_smi("compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"):
        if len(row) < 4 or row[0] not in gpus:
            continue
        try:
            proc = {"pid": int(row[1]), "name": row[2], "used_mib": int(row[3])}
        except ValueError:
            continue
        gpus[row[0]]["procs"].append(proc)
    out = sorted(gpus.values(), key=lambda g: g["index"])
    for g in out:
        # The panel names the biggest consumer; the ranking is folded here so no
        # reader has to sort (statistics live in board/, never in the cockpit TS).
        g["procs"].sort(key=lambda p: p["used_mib"], reverse=True)
    return out


def _ram() -> dict:
    """Physical memory in GiB from /proc/meminfo: ``used`` is MemTotal minus
    MemAvailable (what a new allocation can actually get -- reclaimable cache is
    free, not used). A kernel without /proc reads as 0/0."""
    fields: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                fields[key] = int(rest.split()[0])  # kB
    except (OSError, IndexError, ValueError):
        return {"used_gb": 0.0, "total_gb": 0.0}
    total, avail = fields.get("MemTotal", 0), fields.get("MemAvailable", 0)
    return {"used_gb": round(max(total - avail, 0) / 1048576, 1),
            "total_gb": round(total / 1048576, 1)}


def host_vitals(path: str | Path = ".") -> dict:
    """The machine's LIVE resource headroom: every GPU's VRAM (with the compute
    processes holding it, biggest first), physical RAM, and the free space on
    the filesystem holding ``path`` -- ``{"gpu": [...], "ram": {...},
    "disk": {...}, "ts": <epoch>}``.

    Same live-state family as read_runtime_status: a plain sample of the box,
    never a chain row, no verify, and it NEVER raises. A host with no NVIDIA
    driver returns ``"gpu": []``; an unreadable /proc or filesystem returns
    zeros. The operator reads this to see a VRAM ceiling coming before it kills
    a resident runtime, so a failed probe must degrade to a quiet gap in the
    panel rather than take the whole poll down with it.

    ``path`` selects the filesystem to report (the board passes runs/, the tree
    a campaign actually fills); it is echoed back so the panel can say which
    mount the number describes.
    """
    try:
        st = os.statvfs(path)
        disk = {"path": str(path),
                "free_gb": round(st.f_bavail * st.f_frsize / 1073741824, 1),
                "total_gb": round(st.f_blocks * st.f_frsize / 1073741824, 1)}
    except OSError:
        disk = {"path": str(path), "free_gb": 0.0, "total_gb": 0.0}
    return {"gpu": _gpus(), "ram": _ram(), "disk": disk, "ts": time.time()}


# --- local model server -----------------------------------------------------

#: The ONE launcher this face may start, hardcoded. An ``action`` word is the
#: only thing a caller supplies -- never a path, never a command line. This face
#: is reachable from the operator's browser through the cockpit bridge, so a
#: caller-supplied script would be remote code execution on the harness box; it
#: is the same rule as a brief not naming its provider (CLAUDE.md).
_MODEL_SCRIPT = Path.home() / "models" / "launch_llamacpp.sh"
#: The identity a pid must prove before this face reports it as the server or
#: kills it: the launcher execs llama-server, so the running process IS the
#: server and its /proc/<pid>/exe is the unforgeable half of the check (argv
#: alone matches any shell that merely mentions the binary -- an editor writing
#: the launcher would have matched). The port pins WHICH llama-server it is.
_MODEL_BIN = "llama-server"
_MODEL_PORT = 30001
#: Health-probe budget. A server mid-load holds the socket without answering,
#: and that wait must not stall the operator's 5s poll behind it.
_MODEL_PROBE_TIMEOUT_S = 1.5
_MODEL_ACTIONS = ("status", "start", "stop")
#: The exact command an operator types when health says the model is STOPPED.
_MODEL_START_HINT = ("start it with `scripts/cockpit --with-model` (or set PH_WITH_MODEL=1 "
                     "in .env) or `python -m board.storecli model_server start`")


def _model_identity(pid: int) -> bool:
    """True iff ``pid`` is live AND is the whitelisted model server right now.

    The one guard behind both the scan and the kill: reading it again at kill
    time is what makes a recycled pid safe (the pidfile's number can be handed
    to an unrelated process between start and stop). Anything unreadable -- an
    exited pid, another user's process -- reads as "not the server", so this
    face never acts on something it cannot identify.
    """
    try:
        if os.path.basename(os.readlink(f"/proc/{pid}/exe")) != _MODEL_BIN:
            return False
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return f"--port {_MODEL_PORT}" in cmdline


def _find_model_server() -> int | None:
    """pid of the live model server, or None -- a /proc scan, never a pattern
    kill. Discovery is by identity, not by bookkeeping, so a server started
    outside the cockpit is reported the same as one this face spawned (the
    adopt half of adopt-or-spawn)."""
    try:
        pids = [int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()]
    except OSError:
        return None
    return next((pid for pid in sorted(pids) if _model_identity(pid)), None)


def _model_health() -> tuple[bool, str | None]:
    """``(healthy, model_id)`` from a GET on the server's ``/v1/models``.

    The load-vs-serving discriminator: the process is up for 1-2 minutes before
    it answers, so ``running and not healthy`` is exactly "still loading". Any
    failure -- refused, timed out, non-JSON, HTTP error -- is a healthy=False
    reading, never an exception.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{_MODEL_PORT}/v1/models",
                                    timeout=_MODEL_PROBE_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
    except Exception:      # URLError/HTTPError/timeout/JSON -- all read as "not serving yet"
        return False, None
    entries = data.get("data") or [] if isinstance(data, dict) else []
    first = entries[0] if entries else {}
    return True, (first.get("id") if isinstance(first, dict) else None)


def _model_vram(pid: int | None) -> int | None:
    """VRAM the server holds, from the same per-pid rows host_vitals already
    reports (one source for the number the operator compares against the meter
    right above this row)."""
    if pid is None:
        return None
    return next((p["used_mib"] for g in _gpus() for p in g["procs"] if p["pid"] == pid), None)


def _model_state() -> dict:
    """The status payload every model_server action answers with."""
    pid = _find_model_server()
    healthy, model = _model_health()
    return {"running": pid is not None, "pid": pid, "port": _MODEL_PORT,
            "healthy": healthy, "model": model, "vram_mib": _model_vram(pid)}


def model_server(action: str = "status", runs_dir: str | Path = ".") -> dict:
    """Start, stop, or read the local model server -- ``{"running", "pid",
    "port", "healthy", "model", "vram_mib"}``, plus an ``"error"`` key when an
    action could not be carried out.

    This switches the SERVICE PROCESS only. Which model a request goes to is the
    operator's route choice in the console's model picker; stopping the server
    hands its ~19 GB of VRAM back to the simulator, which is the whole reason
    the control exists.

    ``action`` is whitelisted to ``status``/``start``/``stop`` and the script it
    may launch is the module constant ``_MODEL_SCRIPT`` -- a caller supplies a
    word, never a path or a command line.

    - ``start`` adopts a server that is already up rather than spawning a second
      one, and spawns detached (``start_new_session``, i.e. setsid) so the
      server outlives whatever terminal or agent session started it.
    - ``stop`` sends SIGTERM to the ONE pid it can identify as this server
      (pidfile first, then the scan), re-checking ``/proc/<pid>/exe`` at kill
      time; a pattern kill is never used -- one has matched the killer's own
      shell in this repo's history.
    - ``status`` never mutates. ``running and not healthy`` means loading.

    Live state in the read_runtime_status family: no chain row, no verify, and
    it NEVER raises -- a failure is an ``"error"`` string beside a real status.
    """
    if action not in _MODEL_ACTIONS:
        return {**_model_state(), "error": f"unknown action: {action}"}
    pidfile = Path(runs_dir) / "model-server.pid"
    if action == "start":
        if _find_model_server() is None:
            try:
                if not _MODEL_SCRIPT.exists():
                    return {**_model_state(), "error": f"launcher not found: {_MODEL_SCRIPT}"}
                with open(Path(runs_dir) / "model-server.log", "ab") as log:
                    # The launcher execs the server, so this pid IS the server.
                    proc = subprocess.Popen([str(_MODEL_SCRIPT)], stdout=log, stderr=log,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
                pidfile.write_text(str(proc.pid))
                # ...but only once bash reaches the exec. Wait briefly for the
                # identity to become true rather than answering "not running"
                # about a process we just started (loading still takes minutes
                # after this -- that is the caller's healthy=False window).
                for _ in range(20):
                    if _model_identity(proc.pid):
                        break
                    time.sleep(0.05)
            except OSError as exc:
                return {**_model_state(), "error": f"spawn failed: {exc}"}
    elif action == "stop":
        try:
            recorded = int(pidfile.read_text().strip())
        except (OSError, ValueError):
            recorded = 0
        pid = recorded if _model_identity(recorded) else _find_model_server()
        if pid is None:
            return {**_model_state(), "error": "not running"}
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            return {**_model_state(), "error": f"kill {pid} failed: {exc}"}
        pidfile.unlink(missing_ok=True)
    return _model_state()


# --- pi0.5 policy server: the model_server pattern on :8000 ---------------------
#: openpi checkout that owns the interpreter and the fine-tuned checkpoints; the
#: serve script lives in THIS repo (scripts/serve_vla_openpi.py) but runs under
#: openpi's venv, exactly as its docstring shows. Same constant-not-argument rule
#: as _MODEL_SCRIPT: a caller picks a checkpoint dir, never an interpreter.
_OPENPI = Path.home() / "Desktop" / "Learning_based_model" / "openpi"
_POLICY_PYTHON = _OPENPI / ".venv" / "bin" / "python"
_POLICY_CHECKPOINT = _OPENPI / "checkpoints" / "pi05_robocasa_lora" / "gate2_bs8" / "199"
_POLICY_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "serve_vla_openpi.py"
_POLICY_CONFIG = "pi05_robocasa_lora"
_POLICY_PORT = 8000
_POLICY_START_HINT = ("start it with `scripts/cockpit --with-policy` (or set PH_WITH_POLICY=1 "
                      "in .env)")


def _policy_identity(pid: int) -> bool:
    """True iff ``pid`` is live and is serve_vla_openpi.py on _POLICY_PORT
    (cmdline only: the exe is a python, which proves nothing)."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return _POLICY_SCRIPT.name in cmdline and f"--port {_POLICY_PORT}" in cmdline


def _find_policy_server() -> int | None:
    try:
        pids = [int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()]
    except OSError:
        return None
    return next((pid for pid in sorted(pids) if _policy_identity(pid)), None)


def _policy_probe() -> tuple[bool, str | None]:
    """``(serving, checkpoint_sha)``: the port only listens once the weights are
    loaded, and the server's FIRST frame is its metadata, so one recv is the
    whole handshake. Any failure reads as not serving / unknown sha."""
    if not _serving(_POLICY_PORT):
        return False, None
    try:
        import msgpack
        import websockets.sync.client
        with websockets.sync.client.connect(f"ws://127.0.0.1:{_POLICY_PORT}", compression=None,
                                            max_size=None, open_timeout=_MODEL_PROBE_TIMEOUT_S) as ws:
            meta = msgpack.unpackb(ws.recv(timeout=_MODEL_PROBE_TIMEOUT_S))
        sha = meta.get("checkpoint_sha") if isinstance(meta, dict) else None
        return True, sha if isinstance(sha, str) else None
    except Exception:
        return True, None


def _policy_state() -> dict:
    pid = _find_policy_server()
    serving, sha = _policy_probe()
    return {"running": pid is not None, "pid": pid, "port": _POLICY_PORT,
            "serving": serving, "checkpoint_sha": sha}


def policy_server(action: str = "status", runs_dir: str | Path = ".",
                  checkpoint_dir: str | Path | None = None) -> dict:
    """Start, stop, or read the pi0.5 policy server -- ``{"running", "pid",
    "port", "serving", "checkpoint_sha"}`` plus ``"error"`` when an action
    could not be carried out. model_server's contract, one port over:
    adopt-or-spawn, setsid, pidfile under ``runs_dir``, exact-pid stop with an
    identity re-check, never raises. ``running and not serving`` is loading.
    ``checkpoint_dir`` defaults to ``$PH_POLICY_CHECKPOINT`` then the constant.
    """
    if action not in _MODEL_ACTIONS:
        return {**_policy_state(), "error": f"unknown action: {action}"}
    pidfile = Path(runs_dir) / "policy-server.pid"
    if action == "start":
        if _find_policy_server() is None:
            ckpt = Path(checkpoint_dir or os.environ.get("PH_POLICY_CHECKPOINT") or _POLICY_CHECKPOINT)
            if not _POLICY_PYTHON.exists():
                return {**_policy_state(), "error": f"interpreter not found: {_POLICY_PYTHON}"}
            if not ckpt.is_dir():
                return {**_policy_state(), "error": f"checkpoint not found: {ckpt}"}
            env = {**os.environ, "PYTHONPATH": str(_POLICY_SCRIPT.parent.parent),
                   "HF_LEROBOT_HOME": os.environ.get("HF_LEROBOT_HOME", str(Path.home() / "Desktop" / "datasets")),
                   "HF_HUB_OFFLINE": "1"}
            try:
                with open(Path(runs_dir) / "policy-server.log", "ab") as log:
                    proc = subprocess.Popen(
                        [str(_POLICY_PYTHON), str(_POLICY_SCRIPT), "--checkpoint-dir", str(ckpt),
                         "--config", _POLICY_CONFIG, "--port", str(_POLICY_PORT)],
                        cwd=str(_OPENPI), env=env, stdout=log, stderr=log,
                        stdin=subprocess.DEVNULL, start_new_session=True)
                pidfile.write_text(str(proc.pid))
            except OSError as exc:
                return {**_policy_state(), "error": f"spawn failed: {exc}"}
    elif action == "stop":
        try:
            recorded = int(pidfile.read_text().strip())
        except (OSError, ValueError):
            recorded = 0
        pid = recorded if _policy_identity(recorded) else _find_policy_server()
        if pid is None:
            return {**_policy_state(), "error": "not running"}
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            return {**_policy_state(), "error": f"kill {pid} failed: {exc}"}
        pidfile.unlink(missing_ok=True)
    return _policy_state()


# --- restart everything: the console cannot restart itself -------------------
_COCKPIT = Path(__file__).resolve().parent.parent / "scripts" / "cockpit"


def restart_services(runs_dir: str | Path = ".", build: bool = False) -> dict:
    """Kick ``scripts/cockpit --restart [--build]`` fully detached and return at
    once -> ``{"started", "pid", "log"}`` (+``"error"`` when the spawn failed).

    The caller is usually the console's MCP server, i.e. a process the restart
    is about to kill, so nothing here waits: setsid, stdio on
    ``runs/restart.log``, and the cockpit itself re-execs once more (see its
    --restart block). Progress is read back by ``health()["restart"]``.
    ``$PH_COCKPIT_BIN`` overrides the script (tests point it at a stub).
    """
    runs = Path(runs_dir)
    log = runs / "restart.log"
    argv = [os.environ.get("PH_COCKPIT_BIN") or str(_COCKPIT), "--restart"] + (["--build"] if build else [])
    try:
        runs.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as fh:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=fh, stderr=fh,
                                    start_new_session=True)
    except OSError as exc:
        return {"started": False, "pid": None, "log": str(log), "error": f"spawn failed: {exc}"}
    return {"started": True, "pid": proc.pid, "log": str(log)}


def _restart_state(runs_dir: Path) -> dict:
    """``{"state": idle|running|failed|done, "last": <last log line>}`` from
    runs/restart.log: the detached cockpit ends every restart with either
    ``restart done`` or an ``ERROR`` line, so the last line IS the state.
    ponytail: a restart that died mid-way (SIGKILL) reads running forever until
    the next restart truncates the log; add a pid check if that ever bites."""
    try:
        lines = (Path(runs_dir) / "restart.log").read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    last = next((ln for ln in reversed(lines) if ln.strip()), "")
    state = ("idle" if not last else "done" if "restart done" in last
             else "failed" if "ERROR" in last else "running")
    return {"state": state, "last": last}

# --- system health: the ONE first command --------------------------------------
#
# Three incidents, one shape: a piece of the pipeline was dead and every face
# still read normal, because no face was asked "is the SYSTEM up" -- only "what
# does this file say". This is that question, answered in one call over the
# pieces a brief actually travels through: a runtime per session (alive by /proc,
# not by its own leftover file), what is stacked up in each inbox, orphans left
# in processing/, the console the operator drives it from, and the model server.
#
# ``problems`` is the payload. An operator (or an agent) reads that list first
# and nothing else; ``ok`` is just ``not problems``. Every entry names the
# session and what to do about it, because a health line that says "degraded"
# with no subject is how the last three incidents stayed invisible.

#: The console's default port (cockpit's own default when .env sets none). The
#: cockpit face passes the port it actually resolved; only a bare call guesses.
_CONSOLE_PORT = 3080
#: Console probe budget: a TCP connect, no HTTP -- a listening socket is the
#: whole question and a wedged server must not stall the health read behind it.
_CONSOLE_PROBE_S = 1.0


def _serving(port: int) -> bool:
    """True iff something accepts a TCP connection on 127.0.0.1:port."""
    try:
        with socket.create_connection(("127.0.0.1", int(port)), _CONSOLE_PROBE_S):
            return True
    except OSError:
        return False


def _count(d: Path) -> int:
    try:
        return sum(1 for _ in d.glob("*.json"))
    except OSError:
        return 0


def health(runs_dir: str | Path = "runs", console_port: int = _CONSOLE_PORT) -> dict:
    """Is my system healthy RIGHT NOW -- one call, whole pipeline.

    ``{ok, problems, sessions, console, model, policy, restart, ts}``.

    * ``sessions`` -- every INTAKE session under runs/, i.e. every dir with an
      ``inbox/``. That is the exact set a brief can be dropped into, including a
      dir a submit created before any runtime booted; a campaign store also
      carries a session-log but has no inbox, so ``is_session`` would drag 20
      archived stores in here and bury the four that matter. Each is ``{name,
      mode, alive, pid, heartbeat_age_s, queued, processing, done, failed,
      reason}``.
    * ``console`` -- ``{port, serving}``: is the UI the operator actually uses up?
      The 2026-08-28 incident was live runtimes behind a dead console.
    * ``model`` -- ``_model_state()`` verbatim (running/healthy/vram). Flagged
      ONLY when ``PH_WITH_MODEL=1`` is in the environment (cockpit exports it
      from .env): stopping the model to hand ~19 GB back to the simulator is a
      normal operator move, not a fault -- unless the operator declared the VLM
      part of the stack, in which case a stopped server is the silent failure.
    * ``policy`` -- ``_policy_state()`` (pi0.5 on :8000: running/serving/
      checkpoint_sha); flagged only under ``PH_WITH_POLICY=1``, same rule.
    * ``restart`` -- ``{state, last}`` of the last ``cockpit --restart`` read
      from runs/restart.log (idle|running|failed|done), so the panel can show
      progress once the console is back.

    Each session row also carries ``state``: ``up`` (runtime alive), ``dormant``
    (dead runtime, nothing queued -- evidence left in place, never a problem),
    ``stalled`` (dead runtime WITH queued/claimed briefs) or ``wedged`` (alive
    but no beat while idle). The last two are always in ``problems``.

    ``problems`` (the thing to read) flags exactly what needs a hand NOW:

    * briefs waiting on a session with no live runtime -- ALL THREE incidents;
    * a runtime that is alive but has not beaten while IDLE (busy-vs-dead is only
      ambiguous while processing/ is non-empty, so an empty one turns a stale
      beat into a verdict: a wedged poll loop, never an operator's choice);
    * more than one brief in processing/ (a runtime claims serially, so the
      extras are crash orphans);
    * a console that is not serving.

    A stopped runtime with an EMPTY inbox is NOT a problem -- it is reported
    ``alive: false`` in ``sessions`` and left there. Retired sessions outnumber
    live ones on a real box, and a health face that is permanently red is one
    nobody reads, which is how the three incidents stayed invisible in the first
    place.

    Same live-state family as host_vitals: a plain sample of the box, never a
    chain row, no verify -- and it never raises, because the command an operator
    reaches for when things are broken must not be the next broken thing.
    """
    runs_dir = Path(runs_dir)
    problems: list[str] = []
    sessions = []
    try:
        entries = sorted(p for p in runs_dir.iterdir() if (p / "inbox").is_dir())
    except OSError:
        entries = []
    for p in entries:
        live = runtime_liveness(p)
        queued, processing = _count(p / "inbox"), _count(p / "processing")
        state = "up" if live["alive"] else "stalled" if queued + processing else "dormant"
        if state == "up" and processing == 0 and (live["heartbeat_age_s"] or 0) > _HEARTBEAT_STALE_S:
            state = "wedged"
        sessions.append({"name": p.name, "mode": live["mode"], "alive": live["alive"],
                         "pid": live["pid"], "heartbeat_age_s": live["heartbeat_age_s"],
                         "queued": queued, "processing": processing,
                         "done": _count(p / "done"), "failed": _count(p / "failed"),
                         "state": state, "reason": live["reason"]})
        if state == "stalled":
            problems.append(
                f"{p.name}: {queued} queued + {processing} claimed brief(s) and "
                f"NO RUNTIME -- {live['reason']}. Nothing will ever claim them; "
                "start it with scripts/cockpit")
        elif state == "wedged":
            problems.append(
                f"{p.name}: runtime pid {live['pid']} is alive but has not beaten "
                f"for {live['heartbeat_age_s']}s while IDLE (poll loop wedged); "
                f"{queued} brief(s) queued")
        elif processing > 1:
            problems.append(
                f"{p.name}: {processing} briefs in processing/ but a runtime claims "
                "one at a time -- the extras are crash orphans and re-queue on the "
                "next boot")
    serving = _serving(console_port)
    if not serving:
        problems.append(f"console: nothing serving on 127.0.0.1:{console_port} -- "
                        "the runtimes may be fine and the UI still gone "
                        "(2026-08-28); restart with scripts/cockpit")
    model = _model_state()
    if os.environ.get("PH_WITH_MODEL") == "1" and not model["running"]:
        problems.append(f"model: PH_WITH_MODEL=1 but nothing serving on "
                        f"127.0.0.1:{model['port']} (the VLM path cannot run) -- "
                        f"{_MODEL_START_HINT}")
    policy = _policy_state()
    if os.environ.get("PH_WITH_POLICY") == "1" and not policy["running"]:
        problems.append(f"policy: PH_WITH_POLICY=1 but nothing serving on "
                        f"127.0.0.1:{policy['port']} (the pi05 arm cannot run) -- "
                        f"{_POLICY_START_HINT}")
    return {"ok": not problems, "problems": problems, "sessions": sessions,
            "console": {"port": int(console_port), "serving": serving},
            "model": model, "policy": policy, "restart": _restart_state(runs_dir),
            "ts": time.time()}


# --- markdown feeds ---------------------------------------------------------

_RANGE = re.compile(r"(\d{4,5})-(\d{4,5})")
_STATE_ORDER = {"planned": 0, "reserved": 1, "burned": 2}


def parse_ledger(text: str) -> list[dict]:
    """Seed-block burn map from STATUS.md's ``区块预算`` section (board/report.py
    DISPLAY only; enforcement reads ``burned_blocks``).

    The ledger is prose, not a table, so this is a best-effort extraction: pull
    every ``NNNN-NNNN`` range from the budget section, classify each by the
    keyword in its sentence (已烧 -> burned, 留/需要时 -> reserved, else
    planned; 未烧 right after a range un-burns that one range), and keep the
    strongest state when a range recurs (a held-out block first planned then
    later burned reads as burned). A bullet wrapped
    onto continuation lines keeps its classification: lines without a keyword
    inherit the last keyword seen since the bullet started (a ``**`` line or a
    blank line resets it), and a line's own keyword takes precedence over the
    inherited one. The source line is carried for a tooltip so a fuzzy
    classification is always auditable.
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "区块预算" in ln), None)
    if start is None:
        return []
    seg: list[str] = []
    for ln in lines[start:]:
        if "frontier" in ln and seg:
            break
        seg.append(ln)
    found: dict[tuple[int, int], dict] = {}
    carry = ""  # state inherited by wrapped continuation lines of one **...** bullet
    for ln in seg:
        if not ln.strip() or ln.lstrip().startswith("**"):
            carry = ""
        burned = "已烧" in ln
        reserved = "留" in ln or "需要时" in ln
        state = "burned" if burned else "reserved" if reserved else carry or "planned"
        if burned or reserved:
            carry = state
        for m in _RANGE.finditer(ln):
            lo, hi = int(m.group(1)), int(m.group(2))
            if hi <= lo:
                continue
            # 未烧 right after a range explicitly un-burns it, even on a 已烧
            # line ("held-out 49350-49549 本相未烧" inside a burn bullet).
            r_state = "planned" if "未烧" in ln[m.end():m.end() + 12] else state
            key = (lo, hi)
            prev = found.get(key)
            if prev is None or _STATE_ORDER[r_state] > _STATE_ORDER[prev["state"]]:
                found[key] = {"lo": lo, "hi": hi, "state": r_state,
                              "line": re.sub(r"\*\*|`", "", ln).strip()}
    return sorted(found.values(), key=lambda r: r["lo"])


def _runs_of(seeds) -> list[tuple[int, int]]:
    """Seeds -> maximal consecutive inclusive ``[lo, hi]`` runs, sorted."""
    out: list[tuple[int, int]] = []
    for s in sorted({int(x) for x in seeds}):
        if out and s == out[-1][1] + 1:
            out[-1] = (out[-1][0], s)
        else:
            out.append((s, s))
    return out


def burned_blocks(runs_root: str | Path, status_md: str | Path | None = None,
                  ) -> list[tuple[int, int, str, str]]:
    """The seed ledger, DERIVED from evidence: every sealed preregistration under
    ``runs_root`` (any depth -- ``runs/<store>`` and ``runs/<session>/campaigns/
    <store>`` alike) burns its dev/selection seeds (role ``gate``) and held-out
    seeds (role ``heldout``) as inclusive ``(lo, hi, role, prereg_sha)`` runs.
    Calibration blocks never enter a prereg, so they never burn.

    Pre-store history (phase 1/2 blocks, held-out rescores) exists only in the
    operator's STATUS.md prose, so its burned rows are unioned in as role
    ``prose`` (sha ``""``): the guard is never weaker than the prose one was.
    STATUS.md is looked up at ``runs_root.parent / "STATUS.md"`` (the board's
    convention) unless ``status_md`` is given.

    Raises ``ValueError`` when no store exists at all: an absent ledger is not
    an empty one, and a caller that treated it as "nothing burned" could reuse
    every block ever gated.
    """
    runs_root = Path(runs_root)
    status_md = Path(status_md) if status_md else runs_root.parent / "STATUS.md"
    stores = sorted(p.parent for p in runs_root.rglob("index.jsonl"))
    if not stores:
        raise ValueError(f"no campaign store under {runs_root}: the burned set "
                         "cannot be derived, refusing to treat it as empty")
    out: list[tuple[int, int, str, str]] = []
    for store in stores:
        for row in _index_rows(store):
            if row["kind"] != "preregistration":
                continue
            try:
                payload = json.loads((store / "artifacts" / f"{row['sha']}.json").read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for key, role in (("dev", "gate"), ("selection", "gate"), ("heldout", "heldout")):
                for lo, hi in _runs_of(payload.get(key) or ()):
                    out.append((lo, hi, role, row["sha"]))
    if status_md.exists():
        out += [(r["lo"], r["hi"], "prose", "") for r in parse_ledger(status_md.read_text())
                if r.get("state") == "burned"]
    return sorted(set(out))


_ROUND = re.compile(r"^##\s*Round\s+(\d+)\s*-\s*([\d-]+)\s*-\s*(.*)$")


def parse_rounds(text: str) -> list[dict]:
    """progress.md ``## Round N - DATE - TITLE`` sections, latest first, each with
    its body text for a collapsible rounds feed."""
    lines = text.splitlines()
    heads = [(i, m) for i, ln in enumerate(lines) if (m := _ROUND.match(ln))]
    rounds = []
    for idx, (i, m) in enumerate(heads):
        end = heads[idx + 1][0] if idx + 1 < len(heads) else len(lines)
        body = "\n".join(lines[i + 1:end]).strip()
        rounds.append({"round": int(m.group(1)), "date": m.group(2),
                       "title": m.group(3).strip(), "body": body})
    rounds.sort(key=lambda r: r["round"], reverse=True)
    return rounds
