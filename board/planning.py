"""The skill-graph planning faces: ``plan_skill_task`` and ``submit_skill_plan``.

Both call faces (board/mcp_server.py tools, ``storecli plan_skill_task`` /
``submit_skill_plan``) pass through these two functions, so the chat agent and
the ph-station panel get the one dict from the one code path -- the same
MCP-与-CLI-同一函数 discipline as the rest of the board.

``plan_skill_task`` is READ-ONLY: it writes nothing under runs/ (a preview is
not evidence). ``submit_skill_plan`` is the one write, and it is the SAME write
``submit_brief`` performs -- the shared atomic drop of a selector+budgets brief
into a runtime inbox -- gated by ``plugins.task.skill_planning.verify_plan_record``
re-validating the record from scratch: an unknown channel, a graph-only
(planning_only) record, a plan the validator refuses, or an unbound leaf is
refused HERE and nothing is dropped. The resident runtime then re-plans and
re-validates from the brief as the sole authority (its ``_BRIEF_KEYS`` gate is
untouched; the brief carries only ``kind/task/instruction/seed`` + budgets).

The pipeline module is imported lazily by name so importing ``board`` stays
plugin-free (``tests/test_cards.py``); resolving it is the first plugin code
either face runs, and only when a planning call is actually made.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from board import store as bs

#: The pipeline module, resolved on first use (plugins.task never imports board;
#: board imports plugins only by name, at call time).
PIPELINE = "plugins.task.skill_planning"
#: The kitchen runtime session: the graph vocabulary and every RoboCasa task.
DEFAULT_SESSION = "session-robocasa"
#: Scratch seed (never burns the ledger); the tool defaults mirror the pipeline.
SCRATCH_SEED = 424242

#: Operator override of the planner's model endpoint, read from the process
#: environment of the face (the console's storecli child, the MCP server):
#: PH_PLANNER_BASE_URL (an OpenAI-compatible /v1), PH_PLANNER_MODEL (optional,
#: else GET /models decides), PH_PLANNER_API_KEY_ENV (the env var NAMING the
#: key; default DEEPSEEK_API_KEY). The key itself is never read here.
ENV_BASE_URL = "PH_PLANNER_BASE_URL"
ENV_MODEL = "PH_PLANNER_MODEL"
ENV_KEY_ENV = "PH_PLANNER_API_KEY_ENV"


def _pipeline():
    return importlib.import_module(PIPELINE)


def planner_params_from_env(env: Mapping[str, str] | None = None) -> dict | None:
    """``endpoint_params`` for the planner when the operator points it at a
    different OpenAI-compatible server, else ``None`` (the card's own default:
    DeepSeek, key via DEEPSEEK_API_KEY / the console credential store)."""
    env = os.environ if env is None else env
    base = env.get(ENV_BASE_URL)
    if not base:
        return None
    ep: dict[str, str] = {"base_url": base,
                          "api_key_env": env.get(ENV_KEY_ENV) or "DEEPSEEK_API_KEY"}
    if env.get(ENV_MODEL):
        ep["model"] = env[ENV_MODEL]
    return {"endpoint_params": ep}


def plan_skill_task(instruction: str, session: str = DEFAULT_SESSION, *,
                    expand: bool = True, channel: str = "auto",
                    seed: int = SCRATCH_SEED,
                    planner_params: Mapping | None = None) -> dict:
    """Plan (never execute): retrieval -> compact catalogue -> DeepSeek strict
    JSON -> validate_plan -> server-side expansion -> binding check.

    Returns the pipeline's dict (``status`` in executable / planning_only /
    rejected / no_match), or ``{"error", "status": "rejected"}`` for a request
    refused before or around the model call (bad instruction, unknown channel,
    unreadable graph, unreachable endpoint). Never raises to the wire.
    """
    sp = _pipeline()
    params = planner_params if planner_params is not None else planner_params_from_env()
    try:
        return sp.plan_skill_task(instruction, session=session, expand=expand,
                                  channel=channel, seed=int(seed), planner_params=params)
    except (sp.PlanningError, sp.SkillGraphError) as exc:
        return {"error": str(exc), "status": sp.STATUS_REJECTED, "executable": False}
    except OSError as exc:
        # urllib's URLError/HTTPError are OSErrors: the model endpoint is down,
        # refused the key, or timed out. Honest failure, no invented plan.
        return {"error": f"planner endpoint unreachable: {exc}",
                "status": sp.STATUS_REJECTED, "executable": False}


def skill_library() -> dict:
    """Return the annotation taxonomy unioned with installed runtime skills."""
    sp = _pipeline()
    try:
        return sp.skill_library_snapshot()
    except (ValueError, OSError) as exc:
        return {"error": str(exc)}


def submit_skill_plan(runs_dir: str | Path, plan: Mapping, session: str = DEFAULT_SESSION,
                      default_session: str = "session-main",
                      default_inbox: str | Path | None = None, *,
                      seed: int = SCRATCH_SEED, max_replans: int | None = None,
                      max_actuations: int | None = None) -> dict:
    """Execute a VERIFIED plan record through the runtime: re-verify from
    scratch, then drop the resulting task brief with the shared atomic drop and
    answer with the brief's ``brief_status`` handle (poll it; cancel_brief stops
    it). A record that is not executable is refused with ``{"error", "status"}``
    and nothing is written."""
    sp = _pipeline()
    try:
        verdict = sp.verify_plan_record(plan, seed=int(seed), max_replans=max_replans,
                                        max_actuations=max_actuations)
    except sp.SkillGraphError as exc:
        return {"error": str(exc), "status": sp.STATUS_REJECTED, "submitted": False}
    if not verdict["ok"]:
        out = {"error": verdict["error"], "status": verdict["status"], "submitted": False}
        if "missing_bindings" in verdict:
            out["missing_bindings"] = verdict["missing_bindings"]
        return out
    res = bs.submit_brief(runs_dir, json.dumps(verdict["brief"]), session,
                          default_session, default_inbox)
    if "submitted" not in res:
        return {**res, "status": verdict["status"], "submitted": False}
    handle = bs.brief_status(Path(res["inbox"]).parent, res["submitted"])
    return {**handle, "submitted": True, "plan_id": verdict["plan_id"],
            "brief": verdict["brief"], "execution_note": verdict["execution_note"]}
