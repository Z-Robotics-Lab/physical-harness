#!/usr/bin/env python3
"""CLI face over the RSI board's parse layer (board/store.py).

The charter's "MCP 与 CLI 是同一函数的两个调用面": board/mcp_server.py serves the
LLM/chat over MCP; this serves the ph-station cockpit's read panels over stdout.
Both are one-call passthroughs into the SAME board.store functions, so the panel
renders the byte-identical dict the LLM gets -- no second statistics layer, no
reinterpretation. The fork host bridge (packages/host/dsh-ph-board) execFiles
this and JSON.parses stdout verbatim.

``health [PORT]`` is the FIRST call when something is wrong: one dict covering
every session's runtime liveness (asked of /proc, not of its own leftover status
file), inbox backlogs, crash orphans, the console, and the model server. It is
the same board.store.health the MCP face serves and ``scripts/cockpit --status``
prints, so the operator's terminal and the agent read the identical answer.

Three fns cover the brief LIFECYCLE, the same three board.store functions the
MCP face exposes: ``submit_brief`` drops one, ``brief_status <brief-id>
[--wait-ms N]`` says where it is and what it did (queued/running/done/failed/
cancelled, plus its own event tail and its outcome), and ``cancel_brief
<brief-id>`` stops it -- cooperatively, sealed as ``runtime.task_cancelled``,
never as an error. A long mission is polled with brief_status, never
reconstructed from raw files.

``plan_skill_task --instruction '<text>' [--channel X] [--session S]`` is the
natural-language planning READ (board/planning.py -> plugins.task.skill_planning):
skill-graph retrieval, DeepSeek strict JSON, the runtime's validate_plan, server-
side expansion, binding check -- ``status`` executable / planning_only / rejected
/ no_match, and it writes nothing. ``submit_skill_plan --plan '<record>'`` is
its ONE explicit execute: the record is re-verified and, only if executable,
becomes an ordinary task brief through the same shared drop below.

This face is read-only but for four write fns. ``model_server <action>`` is the
console's local-model switch (``status``/``start``/``stop``, default ``status``)
-- the launcher it may run is a constant in board.store, never an argument, and
the action word is whitelisted there. ``policy_server <action>`` is the same switch for
pi0.5 on :8000 and ``restart_services [build]`` kicks a detached cockpit --restart. ``cancel_brief`` drops one live marker and
nothing else (the runtime does every mutation that follows). The third is
``submit_brief`` (the cockpit's
submit button; ``--brief '<json>' --session <name>``), a passthrough into
board.store.submit_brief -- the SAME shared brief_drop atomic drop the MCP
face's submit_brief tool uses, zero validation (the resident runtime's
``_BRIEF_KEYS`` re-validation on claim is the SOLE authority). Briefs take
three kinds -- ``task``, ``campaign``, and ``rsi`` (the
generic self-improvement chain, minimal form ``{"kind":"rsi","task":"<task>"}``;
docs/project-documentation.md §4). An rsi run heartbeats ``runs/<store>/progress.json`` with a
``stage`` field (calibrate / gate / dev / done) that ``campaign_progress`` below
forwards verbatim, so the cockpit's 演进 panel shows where the chain is.

Name-addressed reads (store/heldout/session) go through board.store.safe_child,
the one audited traversal guard, so a ``../`` name can never escape runs_dir.

    python -m board.storecli list_stores --runs runs/     # -> JSON on stdout
    python -m board.storecli store stack-g1 --runs runs/   # name-addressed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from board import cards as bc
from board import planning as bp
from board import store as bs
from board import vault as bv


def _read(path: Path) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def dispatch(fn: str, name: str | None, runs: Path, status: Path, progress: Path,
             after: int = 0, relation: str | None = None, after_ts: float = 0.0,
             wait_ms: int = 0, brief: str | None = None,
             session: str | None = None, seq: int = 0, out: Path | None = None,
             sha: str | None = None, round: int = 0, checkpoint: str | None = None,
             instruction: str | None = None, channel: str = "auto",
             plan: str | None = None, seed: int = bp.SCRATCH_SEED,
             expand: bool = True, max_replans: int | None = None,
             max_actuations: int | None = None):
    """Return the same object the matching board/mcp_server.py tool returns.

    Raises KeyError for an unknown fn and ValueError for a rejected name, so
    main() can map both to an ``{"error": ...}`` line with a nonzero exit while
    every valid call is a bare board.store passthrough.
    """
    if fn == "submit_brief":
        # a write fn: raw passthrough into the shared atomic drop, zero
        # validation (see module docstring -- runtime is the sole authority).
        if brief is None:
            raise ValueError("submit_brief needs --brief")
        return bs.submit_brief(runs, brief, session or "session-main")
    if fn == "plan_skill_task":
        # READ: natural language -> validated skill chain; writes nothing. The
        # same board.planning function the MCP tool wraps (one implementation).
        if not instruction:
            raise ValueError("plan_skill_task needs --instruction")
        return bp.plan_skill_task(instruction, session or bp.DEFAULT_SESSION,
                                  expand=expand, channel=channel, seed=seed)
    if fn == "skill_library":
        return bp.skill_library()
    if fn == "submit_skill_plan":
        # the ONE explicit execute: re-verify the record, then the shared drop.
        if plan is None:
            raise ValueError("submit_skill_plan needs --plan '<composite_plan json>'")
        try:
            record = json.loads(plan)
        except ValueError as exc:
            raise ValueError(f"--plan is not JSON: {exc}") from exc
        return bp.submit_skill_plan(runs, record, session or bp.DEFAULT_SESSION,
                                    seed=seed, max_replans=max_replans,
                                    max_actuations=max_actuations)
    if fn in ("brief_status", "cancel_brief"):
        # The brief id rides the `name` slot (the model_server pattern); the
        # SESSION is the addressed thing, so it goes through the shared guard.
        path = bs.safe_child(runs, session or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        if not name:
            raise ValueError(f"{fn} needs a brief id as the name argument")
        return (bs.brief_status(path, name, wait_ms) if fn == "brief_status"
                else bs.cancel_brief(path, name))
    if fn in ("submit_proposal", "proposals"):
        # The session is the addressed thing (shared guard); submit_proposal's
        # raw JSON rides --brief into the ONE shared store write.
        path = bs.safe_child(runs, session or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        if fn == "proposals":
            return bs.proposals(path)
        if brief is None:
            raise ValueError("submit_proposal needs --brief (the proposal JSON)")
        return bs.submit_proposal(path, brief)
    if fn in ("rsi_run", "rsi_series", "rsi_frames"):
        # The TASK rides the `name` slot; the session is the addressed thing
        # (the brief_status pattern) and goes through the shared guard.
        path = bs.safe_child(runs, session or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        if not name:
            raise ValueError(f"{fn} needs a task as the name argument")
        if fn == "rsi_run":
            return bs.rsi_run(path, name)
        if fn == "rsi_series":
            return bs.rsi_series(path, name)
        return bs.rsi_frames(path, name, round)
    if fn == "list_stores":
        return bs.list_stores(runs)
    if fn == "cards":
        return bc.list_cards()
    if fn == "vault":
        return bv.build_graph(runs)
    if fn == "vault_node":
        return bv.node(bv.build_graph(runs), name or "")
    if fn == "vault_neighbors":
        return bv.neighbors(bv.build_graph(runs), name or "", relation)
    if fn == "campaign_progress":
        return bs.campaign_progress(runs)
    if fn == "sessions":
        return bs.discover_sessions(runs)
    if fn == "host_vitals":
        return bs.host_vitals(runs)
    if fn == "health":
        # The console PORT rides the `name` slot (the model_server pattern) --
        # scripts/cockpit --status passes the port it actually resolved from
        # .env, and a bare call falls back to the board's own default.
        return bs.health(runs, int(name) if name else bs._CONSOLE_PORT)
    if fn == "model_server":
        # The action rides the `name` slot; it is whitelisted board-side and
        # defaults to the read, so an omitted argument can never start or stop
        # anything. The launcher path is a board constant, never an argument.
        return bs.model_server(name or "status", runs)
    if fn == "policy_server":
        # Same slot rule as model_server; --checkpoint else PH_POLICY_CHECKPOINT.
        return bs.policy_server(name or "status", runs, checkpoint)
    if fn == "restart_services":
        # The name slot "build" opts into the ph-station rebuild first.
        return bs.restart_services(runs, build=(name == "build"))
    if fn == "ledger":
        return bs.burned_blocks(runs)
    if fn == "rounds":
        return bs.parse_rounds(_read(progress))
    if fn == "store":
        path = bs.safe_child(runs, name or "", bs.is_store)
        if path is None:
            raise ValueError("unknown store")
        return bs.store_detail(path)
    if fn == "heldout":
        path = bs.safe_child(runs, name or "", bs.is_store)
        if path is None:
            raise ValueError("unknown store")
        return bs.heldout_blocks(runs, name)
    # Session-addressed reads default to session-main when no name is given (the
    # resident runtime), so a caller can omit it; an explicit name still routes
    # to that session, and a ``../`` name is rejected by the shared guard.
    if fn == "session":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.read_session(path)
    if fn == "runtime_status":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.read_runtime_status(path)
    if fn == "runtime_frame":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.read_runtime_frame(path, after_ts, wait_ms)
    if fn == "runtime_rollout":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.read_runtime_rollout(path)
    if fn == "runtime_keyframes":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.read_runtime_keyframes(path)
    if fn == "runtime_keyframe":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.read_runtime_keyframe(path, seq)
    if fn == "runtime_events":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.read_runtime_events(path, after)
    if fn == "session_progress":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.session_progress(path)
    if fn == "suite_result":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.suite_result(path, sha)
    if fn == "trajectories":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.export_trajectories(path, out) if out else bs.trajectories(path)
    if fn == "plan_index":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.plan_index(path)
    if fn == "skill_evidence":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.skill_evidence(path)
    if fn == "skills":
        path = bs.safe_child(runs, name or "session-main", bs.is_session)
        if path is None:
            raise ValueError("unknown session")
        return bs.skills(path)
    raise KeyError(fn)


def serve(stdin, stdout, runs: Path, status: Path, progress: Path) -> int:
    """Resident line-JSON loop over the SAME dispatch (``storecli serve``).

    One request object per line ({"fn", "name"?, "after"?, "relation"?,
    "after_ts"?, "wait_ms"?}), one JSON reply line per request, strictly in
    order, flushed. The ph-station bridge keeps one of these alive for the
    取景窗 long poll: the ~60ms interpreter+import spawn cost was the measured
    browser fps ceiling, and this moves it off the per-frame path. Errors map
    to the same ``{"error": ...}`` dicts as one-shot mode and NEVER end the
    loop; EOF does.
    """
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            result = dispatch(req.get("fn", ""), req.get("name"), runs, status, progress,
                              int(req.get("after", 0)), req.get("relation"),
                              float(req.get("after_ts", 0.0)), int(req.get("wait_ms", 0)),
                              req.get("brief"), req.get("session"),
                              int(req.get("seq", 0)), sha=req.get("sha"),
                              round=int(req.get("round", 0)), checkpoint=req.get("checkpoint"),
                              instruction=req.get("instruction"),
                              channel=req.get("channel", "auto"), plan=req.get("plan"),
                              seed=int(req.get("seed", bp.SCRATCH_SEED)),
                              expand=bool(req.get("expand", True)),
                              max_replans=req.get("max_replans"),
                              max_actuations=req.get("max_actuations"))
        except KeyError:
            result = {"error": f"unknown fn: {req.get('fn', '')}"}
        except Exception as exc:  # bad JSON / rejected name / anything: reply, keep serving
            result = {"error": str(exc)}
        print(json.dumps(result), file=stdout, flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("fn", help="serve|health|skill_library|plan_skill_task|submit_skill_plan|submit_brief|brief_status|cancel_brief|submit_proposal|proposals|rsi_run|rsi_series|rsi_frames|list_stores|store|heldout|campaign_progress|sessions|session|session_progress|suite_result|trajectories|plan_index|skill_evidence|skills|runtime_status|runtime_frame|runtime_rollout|runtime_keyframes|runtime_keyframe|runtime_events|host_vitals|model_server|policy_server|restart_services|ledger|rounds|cards|vault|vault_node|vault_neighbors")
    parser.add_argument("name", nargs="?", default=None, help="store/session name, vault node id for vault_node/vault_neighbors, the brief id for brief_status/cancel_brief, the task for rsi_run/rsi_series/rsi_frames, the model_server/policy_server action (status|start|stop, default status), 'build' to make restart_services rebuild ph-station first, or the console port for health")
    parser.add_argument("--brief", default=None, help="submit_brief: the raw brief JSON string, dropped verbatim (zero validation; the runtime is the sole authority); submit_proposal: the raw proposal JSON {task, kind, payload, note}")
    parser.add_argument("--session", default=None, help="the runtime session addressed: whose inbox submit_brief routes into, and whose brief brief_status/cancel_brief names (default: session-main; plan_skill_task/submit_skill_plan default to session-robocasa)")
    parser.add_argument("--instruction", default=None, help="plan_skill_task: the natural-language task to plan (read-only; nothing is executed)")
    parser.add_argument("--channel", default="auto", help="plan_skill_task: pin the vocabulary (robocasa_skill_graph or a task name) instead of routing by retrieval")
    parser.add_argument("--no-expand", dest="expand", action="store_false", help="plan_skill_task: skip the server-side leaf expansion")
    parser.add_argument("--plan", default=None, help="submit_skill_plan: the composite_plan record JSON from plan_skill_task; re-verified from scratch, refused unless executable")
    parser.add_argument("--seed", type=int, default=bp.SCRATCH_SEED, help="plan_skill_task/submit_skill_plan: the task seed (default: the 424242 scratch seed, never burns the ledger)")
    parser.add_argument("--max-replans", type=int, default=None, help="submit_skill_plan: brief budget override")
    parser.add_argument("--max-actuations", type=int, default=None, help="submit_skill_plan: brief budget override")
    parser.add_argument("--relation", default=None, help="vault_neighbors: restrict adjacency to one rel")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="campaign runs directory (default: runs)")
    parser.add_argument("--status", type=Path, default=None, help="STATUS.md (display-only prose; the ledger fn derives from runs/)")
    parser.add_argument("--progress", type=Path, default=None, help="progress.md for the rounds feed (default: <runs>/../progress.md)")
    parser.add_argument("--after", type=int, default=0, help="runtime_events cursor: return only events with seq > AFTER")
    parser.add_argument("--after-ts", type=float, default=0.0, help="runtime_frame cursor: the ts last displayed; unchanged file -> short {unchanged} reply")
    parser.add_argument("--wait-ms", type=int, default=0, help="long poll: runtime_frame blocks up to WAIT_MS for the frame to change past --after-ts, brief_status for the brief's STATE to change; either way the answer is the current state, never a timeout error (capped board-side)")
    parser.add_argument("--seq", type=int, default=0, help="runtime_keyframe: the runtime_events seq whose pinned still to fetch")
    parser.add_argument("--sha", default=None, help="suite_result: the suite artifact sha to read (default: the session's newest suite.sealed row)")
    parser.add_argument("--round", type=int, default=0, help="rsi_frames: the evolve round whose kept media paths to list")
    parser.add_argument("--checkpoint", default=None, help="policy_server start: checkpoint dir (default: PH_POLICY_CHECKPOINT, then the board constant)")
    parser.add_argument("--out", type=Path, default=None, help="trajectories: write <OUT>/dev.jsonl and heldout.jsonl (split by burned block role) and print the counts")
    args = parser.parse_args(argv)
    runs = args.runs.resolve()
    status = args.status.resolve() if args.status else runs.parent / "STATUS.md"
    progress = args.progress.resolve() if args.progress else runs.parent / "progress.md"
    if args.fn == "serve":
        return serve(sys.stdin, sys.stdout, runs, status, progress)
    try:
        result = dispatch(args.fn, args.name, runs, status, progress, args.after, args.relation,
                          args.after_ts, args.wait_ms, args.brief, args.session, args.seq,
                          args.out, args.sha, args.round, args.checkpoint,
                          args.instruction, args.channel, args.plan, args.seed, args.expand,
                          args.max_replans, args.max_actuations)
    except KeyError:
        print(json.dumps({"error": f"unknown fn: {args.fn}"}))
        return 2
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        return 3
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
