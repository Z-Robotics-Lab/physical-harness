#!/usr/bin/env python3
"""Read-only MCP server over the RSI board's parse layer (board/store.py).

The dsh cockpit (round-95 adoption) is an MCP client; this is the harness-side
MCP *server* it connects to for live reads. Every tool is a one-call passthrough
into board.store -- the SAME pure parse layer scripts/rsi_board.py serves over
HTTP -- so the two surfaces return byte-identical dicts and rsi_board can be
retired (rung 4) without losing a view.

The only writes are the brief lifecycle -- submit_brief/run_task drop a brief,
submit_skill_plan drops the task brief of a re-verified skill plan, cancel_brief
drops a stop marker -- and all land in LIVE intake dirs through the one shared
atomic drop. plan_skill_task (natural language -> validated skill chain) is a
read: it writes nothing and executes nothing. Nothing here writes the hash chain: runs/ is sealed
evidence and the resident runtime is its single writer.

Name-addressed reads (store/heldout/session) go through board.store.safe_child,
the one audited traversal guard, so a ``../`` name can never escape runs_dir.

    .venv/bin/python board/mcp_server.py --runs runs/    # stdio MCP server
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server import MCPServer

from board import cards as bc
from board import planning as bp
from board import store as bs
from board import vault as bv
from harness.manifest import discover

#: The default routing session -- the resident runtime cockpit always brings up.
#: submit_brief/run_task and the session-addressed reads fall back to it, so the
#: pre-routing single-runtime behavior is byte-identical when no session is named.
_DEFAULT_SESSION = "session-main"


class _Cfg:
    """Server config: the runs/ tree, the two markdown feeds, the default routing
    session, and its inbox that submit_brief drops into. Set once by configure()
    (main, or a test); the tools read it. Read defaults mirror rsi_board."""

    runs = Path("runs").resolve()
    status = runs.parent / "STATUS.md"
    progress = runs.parent / "progress.md"
    session = _DEFAULT_SESSION
    inbox = runs / _DEFAULT_SESSION / "inbox"
    #: Planner endpoint override for plan_skill_task (None = the planner card's
    #: own default, DeepSeek; tests point it at a fake server). Secrets never.
    planner_params = None


def configure(runs, status=None, progress=None, inbox=None,
              session=_DEFAULT_SESSION, planner_params=None) -> None:
    _Cfg.runs = Path(runs).resolve()
    _Cfg.status = Path(status).resolve() if status else _Cfg.runs.parent / "STATUS.md"
    _Cfg.progress = Path(progress).resolve() if progress else _Cfg.runs.parent / "progress.md"
    _Cfg.session = session
    _Cfg.inbox = Path(inbox).resolve() if inbox else _Cfg.runs / session / "inbox"
    _Cfg.planner_params = planner_params


def _route_inbox(session: str) -> Path | None:
    """The inbox a per-call ``session`` routes into, or ``None`` for an unknown
    one -- board.store.brief_inbox fed this server's configured defaults (the
    default session resolves to the configured inbox verbatim, no is_session
    gate, so a first submit can precede the runtime's first boot; any OTHER
    session is validated against runs/ through the shared guard)."""
    return bs.brief_inbox(_Cfg.runs, session, _Cfg.session, _Cfg.inbox)


def _session_dir(session: str) -> Path | None:
    """The session directory a per-call ``session`` addresses, or ``None``. The
    intake dirs are inbox's siblings (harness_runtime.boot), so the routed inbox
    IS the resolver -- one routing rule for the write faces and the lifecycle
    faces alike, including the configure(inbox=) override."""
    inbox = _route_inbox(session)
    return inbox.parent if inbox is not None else None


def _read(path: Path) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


mcp = MCPServer("physical-harness")


@mcp.tool()
def list_stores() -> list[dict]:
    """Every campaign store under runs/, newest first (summary cards)."""
    return bs.list_stores(_Cfg.runs)


@mcp.tool()
def store(name: str) -> dict:
    """Full structured view of one campaign store by name."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_store)
    return bs.store_detail(path) if path else {"error": "unknown store"}


@mcp.tool()
def heldout(name: str) -> dict:
    """Multi-block held-out comparison for a campaign (its block + rescores)."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_store)
    return bs.heldout_blocks(_Cfg.runs, name) if path else {"error": "unknown store"}


@mcp.tool()
def campaign_progress() -> list[dict]:
    """Every live campaign heartbeat under runs/ (runs/*/progress.json, written
    per finished episode by script-path batteries): done/total/label/rolling
    stats + a running flag. Live state, never sealed evidence."""
    return bs.campaign_progress(_Cfg.runs)


@mcp.tool()
def sessions() -> list[dict]:
    """Every runtime session under runs/, newest first (with chain badges).
    ``runtime_alive`` is false when NOTHING is serving that session's inbox --
    submitting there queues a brief no process will ever claim."""
    return bs.discover_sessions(_Cfg.runs)


@mcp.tool()
def session(name: str = _DEFAULT_SESSION) -> dict:
    """One runtime session: note payloads by kind plus chain_ok (hash-chain verify).
    ``name`` defaults to the resident session-main; pass another to read a second
    runtime's session (a ``../`` name is rejected by the shared guard)."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.read_session(path) if path else {"error": "unknown session"}


@mcp.tool()
def session_progress(name: str = _DEFAULT_SESSION) -> dict:
    """One session's mission-progress aggregate over its task.plan_complete rows
    (task tallies, total replans/faults, stage pass-rate, latest task tree).
    ``name`` defaults to session-main."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.session_progress(path) if path else {"error": "unknown session"}


@mcp.tool()
def suite_result(name: str = _DEFAULT_SESSION, sha: str | None = None) -> dict | None:
    """One session's sealed benchmark-suite artifact ({suite, arm, seeds,
    per_task:{n,k,L_mean,first_death}, prereg_sha, checkpoint_sha?}); ``sha``
    defaults to the newest ``suite.sealed`` row. null when none was sealed."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.suite_result(path, sha) if path else {"error": "unknown session"}


@mcp.tool()
def rsi_run(task: str, name: str = _DEFAULT_SESSION) -> dict | None:
    """One evolve campaign's state (campaigns/evolve-<task>/campaign.json: task,
    session, seeds, arm, rounds[] each with per_seed + needs, best, cursor,
    status) plus ``latest`` round.
    null when the session runs no campaign for that task."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.rsi_run(path, task) if path else {"error": "unknown session"}


@mcp.tool()
def rsi_series(task: str, name: str = _DEFAULT_SESSION) -> list[dict]:
    """Per-round {round, before, after, best, per_seed, needs} of one evolve
    campaign (the line-chart feed; per_seed = [{seed, success, first_death,
    failure_mode}], needs = what would unblock a round that tried nothing);
    [] when none exists."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.rsi_series(path, task) if path else {"error": "unknown session"}


@mcp.tool()
def rsi_frames(task: str, round: int, name: str = _DEFAULT_SESSION) -> list[str]:
    """Kept keyframe/video paths (session-relative) one evolve round recorded;
    [] when the campaign or round is absent. Paths only, never bytes."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.rsi_frames(path, task, round) if path else {"error": "unknown session"}


@mcp.tool()
def submit_proposal(proposal: dict, session: str = _DEFAULT_SESSION) -> dict:
    """Drop one proposal for the lightweight evolve loop into
    ``runs/<session>/proposals/``: ``{"task": <task>, "kind": "tunables"|"executor"|"card",
    "payload": {...}, "note": <why>}``. scripts/evolve.py consumes the oldest pending
    one for its task at the start of each round (sealed as ``rsi_proposal_applied``)
    and tries it INSTEAD of its built-in proposer; it still publishes only when the
    same-seed success count improves. Payloads: tunables ``{ref, path:[...], to,
    node?}``; executor ``{to, node?}``; card ``{path: plugins/candidates/<name>, to:
    <executor key>, ref: "module:attr", params?, node?}``. ``node`` defaults to the
    suite's commonest first-death node. Records are never written here."""
    path = bs.safe_child(_Cfg.runs, session, bs.is_session)
    return bs.submit_proposal(path, json.dumps(proposal)) if path else {"error": "unknown session"}


@mcp.tool()
def proposals(name: str = _DEFAULT_SESSION) -> list[dict]:
    """The session's proposal inbox, oldest first: ``{id, task, kind, payload,
    note, applied}`` (``applied`` = null while pending, else ``{round, ts}``)."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.proposals(path) if path else {"error": "unknown session"}


@mcp.tool()
def trajectories(name: str = _DEFAULT_SESSION) -> list[dict]:
    """Protocol-v0 trajectory samples projected from one session's chain: one
    per plan/replan decision (x: mission/sigma0/skills/done/fault, y: graph id +
    rationale, o: legal/verify/L/success/replans/seed). ``name`` defaults to
    session-main."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.trajectories(path) if path else {"error": "unknown session"}


@mcp.tool()
def plan_index(name: str = _DEFAULT_SESSION) -> list[dict]:
    """Per (task, graph_sha, embodiment, arm) plan evidence {n, k, L_mean, seeds,
    blocks, graph} projected from one session's chain -- the input to
    scripts/publish_plans.py. ``name`` defaults to session-main."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.plan_index(path) if path else {"error": "unknown session"}


@mcp.tool()
def skill_evidence(name: str = _DEFAULT_SESSION) -> list[dict]:
    """Per (skill, embodiment, executor) skill evidence {n, k} projected from one
    session's task.verify seal rows (executor defaults to scripted on rows that
    seal only driver). ``name`` defaults to session-main."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.skill_evidence(path) if path else {"error": "unknown session"}

@mcp.tool()
def skills(name: str = _DEFAULT_SESSION) -> list[dict]:
    """Records overview, one row per skill: name, kind, bindings (embodiment ->
    executor keys), evidence (embodiment -> {n, k, by_executor}), limits,
    failure_modes, and whether the row is the library record or the session's
    published copy. ``name`` defaults to session-main."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.skills(path) if path else {"error": "unknown session"}

@mcp.tool()
def trajectories_split(name: str = _DEFAULT_SESSION) -> dict:
    """``trajectories`` split ``{"dev": [...], "heldout": [...]}`` by ``o.role``
    (the seed's burned block role) -- the same rows ``storecli trajectories --out``
    writes as dev.jsonl / heldout.jsonl, without touching disk."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.split_trajectories(bs.trajectories(path)) if path else {"error": "unknown session"}


@mcp.tool()
def runtime_status(name: str = _DEFAULT_SESSION) -> dict | None:
    """One runtime session's LIVE status (pid/render/mode/boot_ts/display), or null
    when it has not booted since the file existed. Live state, not sealed evidence.
    ``alive`` is the verdict on that pid (it is checked against /proc -- a status
    file outlives its process, so the pid alone means nothing); ``heartbeat_age_s``
    is seconds since the runtime last stamped, so alive+small is idle-and-
    listening, alive+large is wedged. ``name`` defaults to session-main."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.read_runtime_status(path) if path else {"error": "unknown session"}


@mcp.tool()
def runtime_frame(name: str = _DEFAULT_SESSION, after_ts: float = 0.0,
                  wait_ms: int = 0) -> dict:
    """One runtime session's LIVE viewport frame (runs/<session>/frame.jpg,
    dumped offscreen by the frames overlay while a task runs): {jpeg_b64, ts,
    age_s}, or {"error": "no frame"} when none has been dumped. ``after_ts`` is
    the poller's cursor (the ts last displayed): an unchanged file returns the
    short {"unchanged": true, "ts", "age_s"} with no image bytes. ``wait_ms``
    long-polls: block up to that long (capped board-side) for the frame to
    change past the cursor before answering. Live state, never chain evidence.
    ``name`` defaults to session-main."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return (bs.read_runtime_frame(path, after_ts, wait_ms) if path
            else {"error": "unknown session"})


@mcp.tool()
def runtime_rollout(name: str = _DEFAULT_SESSION) -> dict:
    """Latest completed rollout MP4 for a session ({mp4_b64, ts, size}).
    It exists only when the runtime was booted with --frames and one task has
    finished. Live downloadable state, never chain evidence."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.read_runtime_rollout(path) if path else {"error": "unknown session"}


@mcp.tool()
def runtime_keyframes(name: str = _DEFAULT_SESSION) -> dict:
    """The INDEX of one session's live keyframe stills (runs/<session>/keyframes/,
    one JPEG pinned to an interesting runtime_events seq, cleared every boot):
    {frames: [{seq, kind, ts}], count}. Index only, no image bytes -- poll this,
    then fetch one still with runtime_keyframe. Live state, never chain evidence;
    an absent directory reads as an empty index. ``name`` defaults to session-main."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.read_runtime_keyframes(path) if path else {"error": "unknown session"}


@mcp.tool()
def runtime_keyframe(name: str = _DEFAULT_SESSION, seq: int = 0) -> dict:
    """One keyframe still by its runtime_events seq: {jpeg_b64, seq, kind}, or
    {"error": "no keyframe"} when that seq holds none. Live state, never chain
    evidence. ``name`` defaults to session-main."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.read_runtime_keyframe(path, seq) if path else {"error": "unknown session"}


@mcp.tool()
def runtime_events(name: str = _DEFAULT_SESSION, after_seq: int = 0) -> dict:
    """One runtime session's OPERATIONAL event feed (runtime_events.jsonl):
    events with seq > after_seq plus last_seq. last_seq < after_seq means the
    runtime re-booted (feed truncated); reset the cursor to 0 and re-read.
    Live progress (task_claimed/plan_built/node/stage/replan), never chain
    evidence."""
    path = bs.safe_child(_Cfg.runs, name, bs.is_session)
    return bs.read_runtime_events(path, after_seq) if path else {"error": "unknown session"}


@mcp.tool()
def health(console_port: int = 3080) -> dict:
    """**Ask this FIRST when anything looks wrong.** Is the whole pipeline up?

    ``{ok, problems, sessions, console, model, policy, restart, ts}`` in one call -- ``problems``
    is the list to read and everything else is the evidence behind it. It covers
    every piece a brief travels through: per session, whether a runtime is REALLY
    serving it (checked against /proc, not against its own leftover
    ``runtime_status.json``), that runtime's mode and heartbeat age, the inbox
    backlog, and crash orphans in processing/; then whether the console is
    serving and what the local model server is doing.

    Three incidents this answers and no other tool did: an RSI brief that sat 21h
    in a dead session's inbox; live runtimes behind a console that had been
    reaped; and a session whose runtime was long dead while its stale status file
    made every face report "runtime alive" and a brief held queue position 1
    forever. Same face the operator's ``scripts/cockpit --status`` prints.

    Live state, never sealed evidence, and it never raises."""
    return bs.health(_Cfg.runs, console_port)


@mcp.tool()
def host_vitals() -> dict:
    """The machine's LIVE resource headroom: {gpu: [{index, name, used_mib,
    total_mib, procs:[{pid, name, used_mib}]}], ram: {used_gb, total_gb},
    disk: {path, free_gb, total_gb}, ts}. The disk is the filesystem holding
    runs/. Live state, not sealed evidence, and it never raises: a host with no
    NVIDIA driver reports an empty gpu list."""
    return bs.host_vitals(_Cfg.runs)


@mcp.tool()
def model_server(action: str = "status") -> dict:
    """Start/stop/read the LOCAL model server (llama.cpp on 127.0.0.1:30001) ->
    {running, pid, port, healthy, model, vram_mib}, plus {error} when an action
    failed. action is one of status|start|stop; anything else is rejected, and
    the launcher script is a board constant -- no path or command may be passed.
    This switches the SERVICE PROCESS only, not which model a request routes to
    (that is the console's route picker). Stopping it hands ~19 GB of VRAM back
    to the simulator. Loading takes 1-2 minutes: running=true with healthy=false
    means loading. Live state, never sealed evidence, and it never raises."""
    return bs.model_server(action, _Cfg.runs)


@mcp.tool()
def policy_server(action: str = "status", checkpoint_dir: str = "") -> dict:
    """Start/stop/read the pi0.5 POLICY server (scripts/serve_vla_openpi.py on
    127.0.0.1:8000) -> {running, pid, port, serving, checkpoint_sha}, plus
    {error} when an action failed. action is status|start|stop; checkpoint_dir
    defaults to PH_POLICY_CHECKPOINT (the cockpit .env default). The operator
    starts it by hand -- never start it as a side effect. running=true with
    serving=false means the weights are still loading (minutes). Live state,
    never raises."""
    return bs.policy_server(action, _Cfg.runs, checkpoint_dir or None)


@mcp.tool()
def restart_services(build: bool = False) -> dict:
    """Restart EVERYTHING (runtimes + this console; the policy server only if it
    was serving) via a detached `scripts/cockpit --restart`; build=true runs
    `pnpm build` in ph-station first and aborts the restart if it fails.
    Returns {started, pid, log} immediately -- this very console dies moments
    later, so do not wait on it; read health()["restart"] once it is back
    (state idle|running|failed|done + last log line)."""
    return bs.restart_services(_Cfg.runs, build)


@mcp.tool()
def ledger() -> list:
    """Burned seed blocks, DERIVED from every sealed preregistration under runs/:
    ``[lo, hi, role, prereg_sha]`` rows, role in gate|heldout. Errors when no
    store exists at all (an absent ledger is not an empty one)."""
    return bs.burned_blocks(_Cfg.runs)


@mcp.tool()
def rounds() -> list[dict]:
    """progress.md round sections, latest first."""
    return bs.parse_rounds(_read(_Cfg.progress))


@mcp.tool()
def list_cards() -> list[dict]:
    """Every installed 机箱 card (plugins/*/manifest.toml), manifest read as data."""
    return bc.list_cards()


@mcp.tool()
def vault() -> dict:
    """The Skill Vault: the whole typed wiki graph (skill/package/capability nodes
    + the 9-relation edge vocabulary), a deterministic fold over sealed runs/ +
    manifests. Read it before planning: which tasks have a *promoted* skill."""
    return bv.build_graph(_Cfg.runs)


@mcp.tool()
def vault_node(id: str) -> dict:
    """One vault node as a wiki page: the node plus its ``out`` edges and
    ``backlinks`` (in-edges). Unknown id -> {"error": "unknown node"}."""
    return bv.node(bv.build_graph(_Cfg.runs), id)


@mcp.tool()
def vault_neighbors(id: str, relation: str | None = None) -> dict:
    """Adjacency (both directions) for one vault node, optionally one ``rel``."""
    return bv.neighbors(bv.build_graph(_Cfg.runs), id, relation)


# --- the session x task advisory (READ-ONLY; it never gates a submit) --------
#
# The mismatch it names: a robocasa mission dropped into session-main is ACCEPTED
# (the task string is in the manifest union, which is one table across every
# card), and refused seconds later inside a DIFFERENT process, in a log the
# operator is not reading. This puts that sentence in the answer they are already
# reading -- and nowhere else. It is not validation: submit_brief refuses nothing
# and names no provider by design (see its docstring), and moving the guard here
# would launder the runtime's authority into the MCP seam.
#
# SILENCE IS THE DEFAULT. Every unreadable input -- unknown task, a binding with
# no embodiment of its own, a chassis that will not fold, a dead runtime, an
# unprobeable interpreter -- yields NO warning key at all, because a wrong
# warning is exactly as bad as a wrong doc.

#: Interpreter probe budget. A wedged interpreter must not stall the submit
#: behind it; a timeout reads as unprobeable, i.e. as silence.
_PROBE_TIMEOUT_S = 10.0
#: Asked of the TARGET interpreter, never of ours: which of argv[1:] it cannot
#: import. find_spec is the machinery the mount itself uses, so an editable or
#: namespace install answers correctly where a dist-info glob would guess wrong.
_PROBE = ("import importlib.util as u, sys; "
          "print(' '.join(p for p in sys.argv[1:] if u.find_spec(p) is None))")


def _session_python(session_dir: Path) -> str | None:
    """The interpreter the session's LIVE runtime runs under, or ``None``.

    The venv is what decides whether a sim's packages import at all, and the two
    resident runtimes differ in exactly that -- which is the same question
    ``board.store.runtime_python`` answers for the liveness badge (argv[0], never
    ``/proc/<pid>/exe``: the venv's python symlink TARGET is identical for both).
    So this IS that call: one /proc guard, not two that can drift apart.
    """
    status = bs.read_runtime_status(session_dir)
    return bs.runtime_python(session_dir, status["pid"]) if status else None


def _compat_warning(brief: dict, session: str, session_dir: Path) -> str | None:
    """One sentence when this brief's task cannot mount in this session, else None.

    Two already-existing data sources, joined:

    * the manifest fold (``harness.manifest.discover``) -- ``task_bindings[task]``
      carries an ``env`` ref only when the mission rides a DIFFERENT simulator
      than the folded base (``harness_runtime._mount_plan`` overrides
      ``embodiment.env`` with it); the plugin dir that ref names is an embodiment
      card, and ``third_party`` is the top-level packages that card needs --
      recorded even for an ``enabled = false`` card, which every second-sim card is.
    * the target runtime's interpreter, asked whether it can import them.

    A binding with no ``env`` rides whatever base the session already mounted, so
    there is nothing to warn about and nothing is said.
    """
    if not isinstance(brief, dict) or not isinstance(brief.get("task"), str):
        return None
    try:
        registry = discover()
    except (OSError, ValueError):
        return None  # a chassis that will not fold has no advice to give
    env = (registry.task_bindings.get(brief["task"]) or {}).get("env")
    if not isinstance(env, str):
        return None
    card = env.split(":")[0].rsplit(".", 1)[-1]  # plugins.embodiment_robocasa:provider
    needs = registry.third_party.get(card)
    python = _session_python(session_dir) if needs else None
    if python is None:
        return None
    try:
        # cwd="/" so the sims/ shadow trap (CLAUDE.md) can never answer for it.
        probe = subprocess.run([python, "-c", _PROBE, *needs], cwd="/",
                               capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    missing = probe.stdout.split() if probe.returncode == 0 else []
    if not missing:
        return None
    return (f"task {brief['task']!r} binds {card}; {session}'s runtime runs "
            f"{python}, which cannot import {', '.join(missing)} -- it will "
            f"refuse this at mount. Submitted anyway; the runtime decides.")


@mcp.tool()
def submit_brief(brief: dict, session: str = _DEFAULT_SESSION) -> dict:
    """Drop a brief into a runtime session's inbox for it to claim.

    A brief is a selector+budgets. A task may also carry one bounded inert
    natural-language ``instruction``; providers/skills/oracles remain
    server-selected. Three kinds exist:

    * ``{"kind":"task","task":"stack","seed":90000}`` -- run one mission once.
    * ``{"kind":"campaign","campaign":"stack","dev":[[41000,41999]]}`` -- run a
      named hand-written campaign script (evolution-mode sessions only).
    * ``{"kind":"rsi","task":"kitchen_thaw"}`` -- the GENERIC self-improvement
      chain (evolution-mode sessions only). **The minimal form is the task name
      and nothing else.** The runtime allocates calibration/dev/held-out blocks
      off the live seed ledger, runs the ungoverned calibration, scores the
      go/no-go gate, picks the target node BY ATTRIBUTION, seals a prereg, runs
      the dev generations, and scores held-out once iff something promotes.
      Optional keys only OVERRIDE what it would otherwise decide from
      measurement: ``node`` (pin the target node; the override is recorded in the
      verdict), ``cal``/``dev``/``heldout`` (``[lo,hi]``, pin a block instead of
      allocating), ``workers``, ``floor``. A NO-GO is a normal outcome -- the
      chain stops at the gate, names the missing capability, and burns no dev
      seed. See docs/project-documentation.md §4.

    This tool does NO brief
    validation and names NO provider: it is a passthrough into
    board.store.submit_brief -- the ONE submit implementation the CLI face
    (``storecli submit_brief``) shares, doing only the shared atomic drop
    (brief_drop.drop -- temp write + os.replace) so the runtime never claims a
    half-written brief. The resident runtime re-validates ``_BRIEF_KEYS``
    server-side on claim and stays the SOLE authority, so an injected extra key
    rides through unchanged and hard-fails to failed/ there -- the MCP seam puts
    the LLM in front of the inbox but not in front of the guard.

    ``session`` only ROUTES (which runtime's inbox); it never touches the brief.
    It defaults to the resident session-main, so single-runtime behavior is
    unchanged. A non-default session is validated against runs/ (a real booted
    session, ``../`` rejected); an unknown one returns ``{"error": ...}``.

    A successful submit may carry one extra READ-ONLY key, ``warning``, when this
    session's runtime provably cannot mount this task's embodiment (see
    _compat_warning). It is advice, computed AFTER the drop, and it never changes
    whether the brief was delivered; when there is nothing certain to say the key
    is absent entirely.
    """
    res = bs.submit_brief(_Cfg.runs, json.dumps(brief), session,
                          _Cfg.session, _Cfg.inbox)
    if "submitted" in res:
        warning = _compat_warning(brief, session, Path(res["inbox"]).parent)
        if warning:
            res["warning"] = warning
    return res


@mcp.tool()
def brief_status(brief_id: str, session: str = _DEFAULT_SESSION,
                 wait_ms: int = 0) -> dict:
    """Where one brief is and what it did -- ONE call, no archaeology.

    Answers ``{state, brief_id, session, task, events, ...}`` where ``state`` is
    ``queued`` | ``running`` | ``done`` | ``failed`` | ``cancelled`` |
    ``unknown``, read off which of the runtime's intake directories holds the
    brief. ``events`` is the tail of THIS brief's slice of the operational feed
    (the same rows runtime_events serves, attributed by claim boundary), and
    ``outcome`` appears once there is one -- the sealed chain row when it names
    the brief (a rejected key, a cancel, a scheduled campaign/rsi), else the
    brief's own ``plan_complete`` event. A ``queued`` brief also carries
    ``queue_position`` (1 = next to be claimed) and ``ahead_running_s`` (how long
    the brief currently running has been running -- position 2 behind a chain
    three hours in is not position 2 behind one that just started); a ``running``
    one carries ``started_ts`` and ``running_s``.

    ``wait_ms`` long-polls: block up to that long (capped board-side) for the
    state to CHANGE, then answer with the current state. **Waiting out the cap is
    not an error** -- the reply just still says ``running``, and you may wait
    again. This is the tool to poll a long mission with; do NOT reconstruct its
    fate from runtime_events + session + session_progress.

    Live state, never sealed evidence: ``session()`` is the chain-verified read.
    """
    session_dir = _session_dir(session)
    return (bs.brief_status(session_dir, brief_id, wait_ms) if session_dir
            else {"error": f"unknown session {session!r}"})


@mcp.tool()
def cancel_brief(brief_id: str, session: str = _DEFAULT_SESSION) -> dict:
    """Stop one brief -- queued or already running -- and seal it as CANCELLED.

    Returns ``{brief_id, session, state, requested}``, plus ``error`` when there
    is nothing to cancel: a brief already ``done``/``failed``/``cancelled`` is
    refused ("already <state>; nothing to cancel") and NOTHING happens, and so is
    an unknown id.

    Cooperative, not a kill: this drops one marker and the resident runtime acts
    on it at a safe boundary -- a queued brief never starts; a running mission
    stops at its next NODE boundary (never mid-rollout, so a persistent episode
    is not torn) and seals an honest partial; a campaign/rsi subprocess is killed
    by process GROUP, so its worker pool leaves no orphans and its half-written
    store is marked incomplete. Expect ``state`` to still read ``running`` for a
    moment; poll brief_status to see it land in ``cancelled``. A cancel that
    arrives after the last node ran stops nothing, so that brief finishes
    ``done`` -- filing completed work as cancelled would be the same lie the
    other way round.

    A cancel is its OWN ending -- ``runtime.task_cancelled``, never
    ``runtime.task_error``, and excluded from session_progress's failure tally --
    because an operator stopping a run and a run failing must never be confusable
    when someone audits this later.
    """
    session_dir = _session_dir(session)
    return (bs.cancel_brief(session_dir, brief_id) if session_dir
            else {"error": f"unknown session {session!r}"})


@mcp.tool()
def run_task(task: str, seed: int, max_replans: int | None = None,
             max_actuations: int | None = None,
             session: str = _DEFAULT_SESSION) -> dict:
    """Submit a task brief and return its HANDLE immediately -- this does NOT wait.

    It used to block until the runtime finished, which made every long mission
    (a 31-node kitchen chain runs for many minutes) come back ``timeout`` while
    the runtime was in fact running it to a clean seal -- and the caller then
    spent a dozen tool calls reconstructing the truth. So: drop the brief through
    the SAME atomic path submit_brief uses (no second submit implementation) and
    hand back where it landed --
    ``{state, brief_id, session, task, queue_position, ahead_running_s, ...}``,
    a brief_status reply as of right now.

    **Then poll brief_status(brief_id, wait_ms=...) for the outcome.** It is the
    one call that answers "did it finish, and what happened"; cancel_brief stops
    it. Nothing here validates the brief or judges success -- the resident
    runtime stays the sole authority (an injected key hard-fails there and
    surfaces as ``state: failed`` with the sealed error).

    ``session`` routes to a second runtime (default session-main); an unknown one
    returns ``{"error": ...}`` before any submit. The submit's advisory
    ``warning`` (see submit_brief) rides onto the handle -- the same read-only
    key, carried so the reason a brief cannot mount survives the hop.
    """
    inbox = _route_inbox(session)
    if inbox is None:
        return {"error": f"unknown session {session!r}"}
    brief = {"kind": "task", "task": task, "seed": seed}
    if max_replans is not None:
        brief["max_replans"] = max_replans
    if max_actuations is not None:
        brief["max_actuations"] = max_actuations
    submitted = submit_brief(brief, session=session)
    out = bs.brief_status(inbox.parent, submitted["submitted"])
    return {**out, "warning": submitted["warning"]} if "warning" in submitted else out


# --- skill-graph planning (READ-ONLY plan, one explicit submit) ---------------


@mcp.tool()
def skill_library() -> dict:
    """Inspect the complete RoboCasa taxonomy and installed runtime skills.

    Returns the IS_A tree, HAS_STAGE annotations, DECOMPOSES_TO recipes,
    bounded dataset evidence, exact policy/driver binding state, and the runtime
    catalogue union. Read-only: no model call and no execution.
    """
    return bp.skill_library()


@mcp.tool()
def plan_skill_task(instruction: str, session: str = bp.DEFAULT_SESSION,
                    expand: bool = True, channel: str = "auto",
                    seed: int = bp.SCRATCH_SEED) -> dict:
    """Turn a natural-language task into a validated skill chain. PLANS ONLY --
    it executes nothing and writes nothing.

    Pipeline (board.planning -> plugins.task.skill_planning): retrieve the
    relevant subtree of the RoboCasa unified skill graph (IS_A taxonomy,
    progressive disclosure: a compact catalogue, never the whole graph) and the
    instruction-driven task bindings; route to ONE channel; ask the planner card
    (DeepSeek, strict JSON, one re-ask on bad JSON); gate the reply with the
    runtime's own ``validate_plan``; expand composites server-side by
    HAS_STAGE / DECOMPOSES_TO; check every leaf for a real policy/driver binding.

    Returns ``{status, goal, channel, selected_catalogue, composite_plan,
    expanded_plan, executable, missing_bindings, unbound_oracles, validation,
    graph_provenance, retrieval}``. ``status`` is ``executable`` (every leaf
    bound -- hand ``composite_plan`` to submit_skill_plan), ``planning_only``
    (the chain is symbolic; ``missing_bindings`` names each unbound leaf -- a
    RoboCasa annotation is NOT a controller), ``rejected`` (``validation.message``
    says why; an unparseable model reply after the one retry lands here too), or
    ``no_match`` (no installed vocabulary matches; the model was not called).

    ``channel`` pins a vocabulary (``robocasa_skill_graph`` or a task name such
    as ``pack_all_robocasa``) instead of routing by retrieval; ``expand=false``
    skips the leaf expansion. ``session`` only labels where a later submit would
    route. Live state, never sealed evidence."""
    return bp.plan_skill_task(instruction, session, expand=expand, channel=channel,
                              seed=seed, planner_params=_Cfg.planner_params)


@mcp.tool()
def submit_skill_plan(plan: dict, session: str = bp.DEFAULT_SESSION,
                      seed: int = bp.SCRATCH_SEED, max_replans: int | None = None,
                      max_actuations: int | None = None) -> dict:
    """Execute a plan_skill_task ``composite_plan`` record -- the ONE explicit
    submit; plan_skill_task itself never executes.

    The record is re-verified from scratch (channel must be an installed task
    binding, the plan must pass validate_plan against that task's current
    catalogue, every leaf must be bound); a planning_only or rejected record is
    refused with ``{"error", "status"}`` and nothing is written. An executable
    record becomes the ordinary task brief ``{"kind":"task","task",
    "instruction","seed"[,budgets]}`` dropped through the SAME atomic path
    submit_brief uses, into ``session``'s inbox, and the reply is that brief's
    brief_status handle (``state`` queued/running/stalled, ``brief_id``) --
    poll brief_status, stop with cancel_brief. The resident runtime re-plans
    and re-validates from the brief as the sole authority; the previewed chain
    is advisory and the runtime's sealed plan is the evidence."""
    return bp.submit_skill_plan(_Cfg.runs, plan, session, _Cfg.session, _Cfg.inbox,
                                seed=seed, max_replans=max_replans,
                                max_actuations=max_actuations)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--runs", type=Path, default=Path("runs"),
                        help="campaign runs directory (default: runs)")
    parser.add_argument("--status", type=Path, default=None,
                        help="STATUS.md (display-only prose; the ledger tool derives from runs/)")
    parser.add_argument("--progress", type=Path, default=None,
                        help="progress.md for the rounds feed (default: <runs>/../progress.md)")
    parser.add_argument("--session", default=_DEFAULT_SESSION,
                        help="default runtime session for submit_brief/run_task "
                             "when no per-call session is named (routes still "
                             "reach any other session by name)")
    args = parser.parse_args(argv)
    runs = args.runs.resolve()
    if not runs.is_dir():
        parser.error(f"runs directory not found: {runs}")
    configure(runs, args.status, args.progress, session=args.session)
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
