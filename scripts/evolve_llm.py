"""The LLM proposer of the lightweight evolve loop (scripts/evolve.py).

Each round the model reads a compact brief of the campaign -- ``rsi_projection``:
task, seeds, round history (tried + before->after + per-seed first death), this
round's per-seed node trails, the knobs of the first-death node's driver with their
current values and the card's failure-mode hints, the executors bound on that
skill with their evidence, the inbox proposals already consumed, ``needs`` and a
bounded excerpt of the dying node's log rows; each seed is its MILESTONE CHAIN with
its FIRST MISSING MILESTONE and the numeric divergence from the campaign's successful
reference there, the failing seeds grouped into ``clusters`` by (first missing
milestone, failure_mode) with the one the round targets named on ``target.cluster``,
the fixed causal ladder ``layers`` to diagnose top-down with, and the ``notebook``
of the last 10 rounds' notes, the gradient (``last_outcome`` = what the last round did to the
``score``, ``score_definition``, and the ``accepted_stack`` this round starts from) -- and answers ONE proposal in the
proposals-inbox shape (``PROPOSAL_SCHEMA``, with the ``layer`` it diagnosed at and an optional notebook
``notes``; a parameter-layer answer is refused while the target node's knobs are
``exhausted``, naming the higher layers): tunables / executor / card
(code-as-policy: the model writes a candidate card under ``plugins/candidates/<name>/``,
checked by scripts/plugin_doctor, dry-instantiated (``dry_run``) and preflighted on
ONE seed before it is mounted) / patch (exact-snippet ``edits`` -- ``{old, new}`` where
``old`` is copy-pasted out of ``functions`` (the stage classes' methods, verbatim, no line
numbers; ``module_sources`` is the same code numbered) and must
occur exactly once (exactly, or leniently: the same lines ignoring indentation) -- against ONE module of the first-death skill's card
(``first_death.modules``), applied by ``apply_edits`` (a unified ``diff`` through
``apply_diff`` is still accepted) to a COPY under the candidate dir, whose generated card (``PATCH_CARD``) instantiates the
patched stage class as an InprocExecutor through the same native-executor seam; the
installed card is never touched) / none. Two calls at most: call 1 = the compact brief
(``brief``, <= ``BRIEF_CHARS``) asking for a decision; a card / patch decision gets call 2,
the static code material (``MATERIAL_KEYS``: contract, reference card, driver source,
``module_sources``, ``functions``, primitives) inserted FIRST after the system message (prefix cache) and
the brief last -- when it answered without a payload, and equally when it wrote one BLIND
and was rejected (the live model invents a snippet on call 1: the material rides its repair). The answer is validated strictly (schema +
``scripts.evolve.from_proposal``); a rejected answer (bad JSON, bad payload, a (knob,
direction) or executor this campaign already tried, doctor
red, a contract miss at instantiation, an exception inside the executor on the
preflight seed) goes back to the model VERBATIM as a follow-up message, up to
``MAX_ATTEMPTS`` rejections per round (``attempts`` / ``calls`` in the audit file); after that the round is
an honest ``none`` with the last reason. An unreachable endpoint falls back to the
rules proposer with the reason on the round row. The brief carries what a coder
needs: the exact card template (concrete ref / skill / embodiment), the executor
contract, the reference card's full source (``REFERENCE_CARD``), the first-death
stage driver's source and the embodiment's primitives, bounded to ``PROMPT_CHARS``. The transport is the model_endpoint card (``plugins.model_endpoint``,
DeepSeek preset; ``PH_MODEL_ENDPOINT_FAKE`` routes to its fake for tests); the raw
answer is kept under ``campaigns/evolve-<task>/llm/round-<r>.json`` for audit, never
in the chain, and the api key never leaves the endpoint's Authorization header.
"""

from __future__ import annotations

import ast
import base64
import importlib
import inspect
import json
import math
import os
import re
import sys
import textwrap
import tomllib
import traceback
from collections import Counter
from pathlib import Path

from harness.config import sha_json
from harness.manifest import PLUGINS_ROOT, mount_params
from harness.protocol import content_id
from harness.registry import load_provider
from harness.skill_executor import SegmentExecutor, StepExecutor
from harness.skill_library import rearm, segment_specs
from scripts import plugin_doctor

ENDPOINT_REF = "plugins.model_endpoint:provider"
FAKE_REF = "plugins.model_endpoint:fake_provider"
#: Where a ``card`` answer lands (one dir per candidate); tests point it elsewhere.
CANDIDATES_ROOT = Path(os.environ.get("PH_CANDIDATES_ROOT") or PLUGINS_ROOT / "candidates")
#: The worked example every card answer is shown in full (the one real candidate).
REFERENCE_CARD = PLUGINS_ROOT / "candidates" / "grasp_geometric_robocasa"
MAX_LOG_LINES = 60
#: Prompt bound of call 2 (~48k DeepSeek tokens, measured ~2.5 chars/token on this JSON): a
#: round costs minutes of simulator, so the model gets the whole picture. Trimming order past
#: it, in this order: the log excerpt, then older rounds' per-seed detail, then the driver
#: source. The editable modules carry their own bound (``MODULE_CHARS``).
PROMPT_CHARS = 120_000
#: Bound of ``functions`` (the stage classes' methods, verbatim, for copy-paste).
FUNCTION_CHARS = 30_000
#: Bound of ``module_sources`` (the full text of every module a patch may edit): the stage's
#: own module is always whole, the others fall back to class/function extracts past it.
MODULE_CHARS = 60_000
#: Call-1 bound (the decision brief): the log excerpt goes first, then per_seed of
#: the 5 detailed rounds (older rounds are counts only, always).
BRIEF_CHARS = 12_000
#: Rejected answers per round (the decision call and the payload call do not count).
MAX_ATTEMPTS = 3
#: The static code material of call 2 (a card / patch decision without its payload).
MATERIAL_KEYS = ("card_template", "executor_contract", "reference_card", "scripted_driver_source",
                 "module_sources", "functions", "primitives", "obs_keys", "action_order")
KINDS = ("tunables", "executor", "card", "patch", "none")
#: Zetta's top-down causal ladder, highest layer first: the layer an answer claims to work
#: at. A higher layer that explains the failure forbids a lower-layer answer; the parameter
#: layer is refused outright once the target node's knobs are exhausted (``_exhausted``).
LAYERS = ("evaluation", "plan", "state", "recovery", "parameter")
LAYER_QUESTION = {
    "evaluation": "is the predicate / oracle right -- does 'success' mean what we think?",
    "plan": "is the node graph right -- a missing node (no nav before a place), a wrong order, "
            "a stage that cannot reach from where the previous one leaves the robot?",
    "state": "is the target / geometry right -- the point reached for, the dock, the frame?",
    "recovery": "is the repair right -- does the recovery primitive fire and undo the failure?",
    "parameter": "LAST RESORT: a knob of the dying driver.",
}
_NAME = re.compile(r"^[a-z][a-z0-9_]{2,40}$")

#: The exact reply shape (the proposals inbox shape + summary/rationale); ``payload``
#: is the FLAT object of ``PAYLOAD_BY_KIND[kind]``.
PROPOSAL_SCHEMA = {
    "decision": "tunables | executor | card | patch | none",
    "payload": "<the flat object described by payload_by_kind[decision], e.g. {\"to\": \"alt\"}; "
               "for card / patch you may omit it: the code material then comes in a second message>",
    "summary": "<1-3 sentences: what you saw this round, in Chinese>",
    "rationale": "<why this try -- the evidence chain: trail -> first missing milestone -> "
                 "divergence -> the code line>",
    "layer": "evaluation | plan | state | recovery | parameter: the HIGHEST layer of `layers` "
             "that explains this failure, chosen top-down",
    "notes": "<optional 1-3 sentences in Chinese for the lab notebook: what this round taught>",
}
PAYLOAD_BY_KIND = {
    "tunables": {"ref": "<tunables.ref>", "path": ["<tunables.path prefix>...", "<knob>"],
                 "to": "<number>", "node": "<optional node id>"},
    "executor": {"to": "<a key of executors>", "node": "<optional node id>"},
    "card": {"name": "<[a-z][a-z0-9_]{2,40}: the candidate dir name>",
             "files": {"manifest.toml": "<toml>", "__init__.py": "<python>"},
             "to": "<new executor key>", "ref": "<name>:provider (or <card_package>:provider)",
             "node": "<optional node id>"},
    "patch": {"name": "<optional [a-z][a-z0-9_]{2,40}: the candidate dir name, defaulted for you>",
              "module": "<one of first_death.modules>",
              "edits": [{"old": "<a snippet COPY-PASTED out of functions[<module>:<Class>.<method>] (the SAME "
                                "code with NO line-number prefix; module_sources is the same text "
                                "numbered). It must occur EXACTLY ONCE in that module>",
                         "new": "<what replaces it, same indentation>"}],
              "diff": "<optional alternative to edits: a unified diff, @@ hunks with exact context lines>",
              "to": "<optional new executor key, defaulted to `name`>", "node": "<optional node id>"},
    "none": {},
}

_RULES = """You are the proposer of a robot skill self-improvement loop. Each round the \
harness runs the task on fixed seeds, you read the round (per-seed node trails, where \
each seed died, its failure_mode, the log excerpt, what was tried before) and decide \
ONE change to try next; the harness re-runs the same seeds and keeps the change only if \
more seeds succeed. Reply with ONE JSON object and nothing else, exactly the shape of \
output_schema: exactly ONE of the payload shapes, matching decision. A card or patch \
decision may come without payload: you then get the code material (contract, reference \
card, the dying stage's driver source, the FULL numbered text of every editable module in \
module_sources, primitives) and write the full payload.
Diagnose TOP-DOWN before you choose. The layers, highest first: evaluation (is the \
predicate/oracle right?) -> plan (is the node graph/order right -- a missing node, a stage \
that cannot reach from where the previous one leaves the robot?) -> state (is the \
target/geometry right?) -> recovery (is the repair right?) -> parameter (a knob, LAST \
RESORT). Answer with `layer` = the HIGHEST layer that explains the evidence, and give the \
evidence chain in rationale (trail -> first missing milestone -> divergence -> the code). \
IF A HIGHER LAYER EXPLAINS THE FAILURE, NEVER PROPOSE A PARAMETER CHANGE: while the brief \
carries `exhausted` for the target node, a parameter-layer answer (a tunables decision \
included) is rejected. Each seed is its MILESTONE CHAIN (trail) with its first missing \
milestone and, when the campaign has a successful reference, the numeric divergence there; \
`clusters` groups the seeds by (first missing milestone, failure_mode) and the round targets \
`target.cluster`. `notebook` is what earlier rounds concluded -- build on it instead of \
re-deriving it, and add `notes` (1-3 Chinese sentences) with what THIS round taught.
Allowed answers:
- tunables: one knob of tunables.values (ref = tunables.ref, path = tunables.path + [knob]) \
to a new numeric value; do not repeat a (knob, direction) already in history.
- executor: switch the first-death node to another key of executors (never the current one).
- card: write a NEW code-as-policy executor for the first-death skill: a candidate card \
dir (files: manifest.toml + __init__.py, plain file names only): copy card_template \
EXACTLY (ref, skill, embodiment, transport are given -- use those strings), implement \
executor_contract, model the code on reference_card (a working card) and repair what \
scripted_driver_source does wrong for this failure_mode; reach the embodiment's helpers \
through primitives BY REF (importlib), never by import; read poses off env, not obs. \
It is doctor-checked, instantiated and run on one seed before the suite; any error \
comes back to you verbatim -- fix exactly that and answer again.
- patch: edit ONE module of first_death.modules (the scripted driver where the stage's \
constants and methods live). Answer patch with NO payload the first time: the module text \
comes back in the next message, and you cannot copy a snippet you have not been shown. \
`functions` is {"<module>:<Class>.<method>": its EXACT source} for every module you may \
edit -- the key's module is the one to send as `module` -- the SAME \
code with NO line-number prefix, so `old` is a COPY-PASTE out of it with zero transformation \
(if a class you want is not there, copy from module_sources and strip the prefix -- never \
invent a snippet); module_sources[module] is \
the whole module's text with every line prefixed "NNNN| " (the 1-based line number, for line \
numbers and context), and first_death.modules_full says which modules are \
whole and which are class/function extracts. Each edit is {old, new}: `old` is a snippet you \
COPY out of that material -- never retyped, never \
invented, never from a file you were not given -- and it must occur EXACTLY ONCE in the \
module (add surrounding lines to make it unique). The edits are applied to a COPY of the \
module and the patched stage class drives the first-death node; the installed card is \
untouched. An `old` found 0 or >1 times comes back to you with the count and the \
WHOLE enclosing function -- copy from that. (A retyped snippet whose only error is indentation \
still applies, but a copy-paste never needs that. A unified diff in "diff" instead of "edits" \
still works.)
- none: only when nothing is left to try -- the brief's untried lists what remains; while it \
is not empty, answer one of those instead (say why in rationale).
`score` is a tuple compared position by position (`score_definition`): whole-task successes \
first, then how FAR the seeds got (their milestones). Successes stay 0 for whole campaigns, so \
the ONLY gradient is the milestone position: PREFER A CHANGE THAT ADVANCES THE FURTHEST-REACHED \
MILESTONE, and NEVER trade a node that already passes for the target node -- a seed dying \
EARLIER than before is WORSE even when the success count is unchanged. `last_outcome` says what \
the last round did to that score. `accepted_stack` is what this campaign already accepted: those \
changes ARE this round's baseline (the run starts from them), so proposing one of them again is \
refused -- build the NEXT change on top.
summary: 1-3 sentences in Chinese on what this round shows. rationale: why this try.
Each seed's keyframes are the failure keyframes of its first-death node (first frame, \
stall / last-progress frame, last frame; 128px); when attached as images they are labelled \
"seed <n> keyframe <i>" in the same order.
`trial_evidence`（brief 的第一行）是你上一轮的改动在仿真里真实跑出来的结果——抛了什么异常、在哪一行，\
或者种子走到了第几步、距离动没动。先读它：是修自己写的代码，还是这条路本来就不通。
补丁自检清单（写 patch 前逐条对照，违反的当场驳回，不进仿真）：
1) 新用到的 self.<属性> 必须在这个类**已有的** __init__ / reset 里初始化——线上第 104、108 轮就是读了\
没人赋值的 self._last_d / self._replan。清单看 first_death.state_init；**如果你在 payload.node 里换了节点，\
就看 death_nodes 里那个节点那一行的 state_init / modules**，它们描述的是另一个类；
2) 只调用这个类自己或基类已经定义的方法，不要凭空 self.<method>()；
3) old != new：补丁必须真的改变行为，原样返回的编辑会被拒；
4) 不要重复被拒过的回答：最近 5 条**跨轮**记着，重复会被驳回并告诉你是哪一轮拒的；
5) 同一节点连续 3 轮死于运行时错误时 brief 会给 `repeat_failure`：改小——一个阶段、一个守卫、\
不引入新状态——或者换 death_nodes 里的另一个节点。"""


def _log_excerpt(seed: int, rows, dead: str | None, budget: int) -> list[str]:
    """The dying node's ``task.fault`` / ``task.verify`` rows of one seed, ``budget`` lines."""
    out = []
    for r in rows:
        if r["kind"] == "task.fault" or (r["kind"] == "task.verify" and r["data"].get("node") == dead):
            out.append(f"seed {seed} {r['kind']} "
                       + json.dumps(r["data"], sort_keys=True, default=str)[:400])
    return out[-budget:]


def first_missing(trail) -> str | None:
    """Zetta's FIRST MISSING MILESTONE: the first node of the seed's milestone chain (its
    node trail) the seed did not complete -- the earliest observable divergence a failure
    cluster is indexed by. None when every node passed."""
    return next((n.get("id") for n in trail or () if n.get("ok") is not True), None)


def _metrics(node: dict) -> dict:
    """The numeric state of one milestone: the node's steps and the scalars of its stall
    trace's last frame (d_eef_target, d_base_target, step) -- what a divergence compares."""
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    end = ((node.get("trace") or {}).get("end") or {}) if isinstance(node.get("trace"), dict) else {}
    return {k: v for k, v in ({"steps": node.get("steps")} | dict(end)).items() if num(v)}


def _divergence(node: dict, healthy) -> dict:
    """This seed's numeric divergence from the SUCCESSFUL REFERENCE at that milestone:
    ``{metric: {seed, reference, delta}}``. ``healthy`` is the campaign's reference index
    row for the milestone (``doc['reference'][milestone]``, written by the trail/reference
    side): ``{metric: number}``, or ``{metric: {"mean": number, ...}}`` for a distribution.
    Empty when there is no reference for it -- the divergence is simply not claimed."""
    out = {}
    for k, v in _metrics(node).items():
        ref = (healthy or {}).get(k)
        ref = ref.get("mean") if isinstance(ref, dict) else ref
        if isinstance(ref, (int, float)) and not isinstance(ref, bool):
            out[k] = {"seed": v, "reference": ref, "delta": round(v - ref, 4)}
    return out


def _seed_row(seed, s: dict, reference: dict) -> dict:
    """One seed for the brief: its MILESTONE CHAIN (the node trail, the dying node with its
    stall trace), the first missing milestone, and the divergence from the successful
    reference there when the campaign has one."""
    # the first-death row alone carries the causal evidence: the stall trace with its
    # per-step ``series``, where the target came from (``geometry``) and the segment that
    # parked the robot there (``upstream``). Dropping them leaves "the base never moved"
    # a fact with no author and no reach number to compare it against.
    # ``failure_mode`` rides only when the row HAS it: a node whose executor sealed none
    # leaves no key (scripts/evolve.py's trail assembly), and a null there reads as "no
    # stall" -- the same lie the trial line used to tell in every candidate round.
    trail = [{k: n.get(k) for k in ("id", "ok", "steps")}
             | ({"failure_mode": n["failure_mode"]} if "failure_mode" in n else {})
             | ({k: n[k] for k in ("trace", "geometry", "upstream") if n.get(k) is not None}
                if n.get("id") == s.get("first_death") else {})
             for n in s.get("trail") or []]
    ms = first_missing(trail) or s.get("first_death")
    row = {"seed": int(seed),
           **{k: s.get(k) for k in ("success", "first_death", "failure_mode", "fault", "keyframes")},
           "first_missing_milestone": ms, "trail": trail}
    if div := _divergence(next((n for n in trail if n.get("id") == ms), {}), (reference or {}).get(ms)):
        row["divergence"] = div
    return row


def clusters(rows: list) -> list[dict]:
    """The round's FAILURE CLUSTERS: the failing seeds grouped by (first missing milestone,
    failure_mode) -- earliest observable divergence -- biggest cluster first. The round
    targets the cluster of its target node (``target.cluster``)."""
    out: dict = {}
    for r in rows:
        if not r.get("success"):
            out.setdefault((r.get("first_missing_milestone"), r.get("failure_mode")), []).append(r["seed"])
    return sorted(({"milestone": k[0], "failure_mode": k[1], "seeds": v, "size": len(v)}
                   for k, v in out.items()), key=lambda c: (-c["size"], str(c["milestone"])))


def _notebook(rounds: list) -> list[dict]:
    """The LAB NOTEBOOK: the last 10 rounds' ``notes`` (what the round taught, the model's
    own words), so the campaign stops re-deriving the same conclusion every round. A note
    on its own is only a HYPOTHESIS -- 10 rounds in a row of the live campaign wrote almost
    the same sentence and not one observation -- so each row carries what was tried, what
    was measured (``_trial_line``) and how the round was judged: 假设 + 观测，成对。"""
    out = []
    for r in rounds:
        if not r.get("notes"):
            continue
        t = r.get("tried") or {}
        ev = r.get("trial_evidence")
        out.append({"round": r.get("round"), "notes": r["notes"],
                    "tried": f"{t.get('kind')} {t.get('node')}",
                    "measured": _trial_line(ev) if isinstance(ev, dict) else None,
                    "verdict": r.get("accepted_reason")})
    return out[-10:]


def _exhausted(proj: dict, untried: list, fd: dict | None = None) -> bool:
    """The brief's ``exhausted`` flag: every (knob, direction) of the target node is tried,
    so the PARAMETER layer is closed there and the answer must come from a higher one.
    ``fd`` is the node's driver row (default: the rotation head, what the brief is about)."""
    return bool(((fd if fd is not None else proj.get("first_death") or {}).get("tunables") or {}).get("values")) \
        and not any(str(u).startswith("tunables ") for u in untried)


def _stage_classes(ref: str, task: str | None) -> list[type]:
    """The scripted stage driver's classes for one sub-goal task (inside the driver's
    package, base first), off the driver module's ``_STAGES`` table; [] without one."""
    stages = getattr(importlib.import_module(ref.partition(":")[0]), "_STAGES", None)
    if not isinstance(stages, dict) or task not in stages:
        return []
    try:
        obj = stages[task][0]()
    except Exception:  # noqa: BLE001 -- a stage that needs the live env: no source, no crash
        return []
    top = type(obj).__module__.rsplit(".", 1)[0]
    return [c for c in reversed(type(obj).__mro__) if c is not object and c.__module__.startswith(top)]


def _stage_source(ref: str, task: str | None) -> str | None:
    """Those classes' source, each under a ``# module <name>`` line (a patch names one)."""
    cls = _stage_classes(ref, task)
    return "\n".join(f"# module {c.__module__}\n" + inspect.getsource(c) for c in cls) if cls else None


def _functions(cls: list[type], modules=(), budget: int = FUNCTION_CHARS) -> dict:
    """``{"<module>:<Class>.<method>": source}`` for the first-death stage classes AND every class / function
    of the modules a patch may edit -- the SAME text as in ``module_sources`` but with NO
    ``"NNNN| "`` prefix, so an edit's ``old`` is a copy-paste with zero transformation (4 of 6
    live rounds died retyping the snippet; a 7th invented one for a class that was not here --
    it wanted the drop stage while the target node was nav). Smallest module first: the
    mission's own modules fit before the big shared primitives one."""
    out: dict[str, str] = {}

    def add(qual: str, fn) -> None:
        nonlocal budget
        if qual in out or budget <= 0:
            return
        try:
            out[qual] = src = inspect.getsource(fn)
        except (OSError, TypeError):   # no source on disk
            return
        budget -= len(src)

    def scan(c: type) -> None:
        for name, fn in vars(c).items():
            fn = getattr(fn, "__func__", fn)
            if inspect.isfunction(fn):
                add(f"{c.__module__}:{c.__qualname__}.{name}", fn)

    mods = []
    for m in modules:
        try:
            mod = importlib.import_module(m)
            mods.append((len(inspect.getsource(mod)), mod))
        except Exception:   # noqa: BLE001 -- an unimportable module simply has no source here
            continue
    for c in cls:
        scan(c)
    for _, mod in sorted(mods, key=lambda kv: kv[0]):
        for o in vars(mod).values():
            if inspect.isclass(o) and o.__module__ == mod.__name__:
                scan(o)
            elif inspect.isfunction(o) and o.__module__ == mod.__name__:
                add(f"{mod.__name__}:{o.__qualname__}", o)
    return out


def _self_attrs(obj) -> set:
    """The ``self.<attr>`` names ASSIGNED in a source text / under an AST node."""
    if isinstance(obj, str):
        try:
            obj = ast.parse(textwrap.dedent(obj))
        except SyntaxError:
            return set()
    out = set()
    for n in ast.walk(obj):
        tgts = (n.targets if isinstance(n, ast.Assign) else
                [n.target] if isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.For)) else [])
        for t in tgts:
            for x in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == "self":
                    out.add(x.attr)
    return out


def _state_init(cls: list[type]) -> dict:
    """Where a patch MUST initialise the state it introduces: ``{"<module>:<Class>.<method>":
    [the self.<attr> it already sets]}`` for each stage class's constructor / reset methods.
    Live rounds 104 and 108 read ``self._last_d`` / ``self._replan`` that nobody ever
    assigned -- the patch applied, the doctor passed, and the trial raised AttributeError."""
    out = {}
    for c in cls:
        for name, fn in vars(c).items():
            fn = getattr(fn, "__func__", fn)
            if inspect.isfunction(fn) and (name == "__init__" or "reset" in name):
                try:
                    out[f"{c.__module__}:{c.__qualname__}.{name}"] = sorted(_self_attrs(inspect.getsource(fn)))
                except (OSError, TypeError):
                    continue
    return out


def _numbered(lines: list[str], a: int = 0, b: int | None = None) -> str:
    """Source lines as ``NNNN| <text>`` with 1-based numbers. The prefix is DISPLAY ONLY:
    ``apply_edits`` matches ``old`` against the raw source and strips nothing, so every
    message that says "copy `old` out of this text" has to say "without the prefix" too."""
    b = len(lines) if b is None else min(b, len(lines))
    return "\n".join(f"{i + 1:4d}| {lines[i]}" for i in range(max(0, a), b))


def _extract(src: str, budget: int) -> str:
    """A module past the budget: its top-level class / def blocks, whole ones only while
    ``budget`` lasts, numbered as in the file (so a snippet copied out of it is still
    exact) and every skipped stretch marked."""
    lines = src.split("\n")
    try:
        body = ast.parse(src).body
    except SyntaxError:
        return _numbered(lines)
    out, pos, left = [], 0, budget
    for n in body:
        if not isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        block = _numbered(lines, n.lineno - 1, n.end_lineno)
        if len(block) > left:
            continue
        left -= len(block)
        if n.lineno - 1 > pos:
            out.append(f"# ... lines {pos + 1}-{n.lineno - 1} omitted (extract)")
        out.append(block)
        pos = n.end_lineno
    if pos < len(lines):
        out.append(f"# ... lines {pos + 1}-{len(lines)} omitted (extract)")
    return "\n".join(out)


def _module_sources(modules, stage_module: str | None) -> tuple[dict, list[str]]:
    """``{module: "# file: <path>\\n<numbered source>"}`` for every module a patch may edit
    (``first_death.modules``) -- what an exact-snippet edit must be copied out of. The stage's
    own module is always whole; the others are whole while ``MODULE_CHARS`` lasts and
    class/function extracts after. Returns (sources, the modules given in full)."""
    out, full, left, repo = {}, [], MODULE_CHARS, PLUGINS_ROOT.parent
    for i, m in enumerate(sorted(modules, key=lambda m: m != stage_module)):
        try:
            f = Path(inspect.getsourcefile(importlib.import_module(m)))
            text = f.read_text()
        except Exception:  # noqa: BLE001 -- a module with no file on disk is simply not offered
            continue
        body = _numbered(text.split("\n"))
        if i and len(body) > left:
            body = _extract(text, max(left, 4_000))
        else:
            full.append(m)
        out[m] = f"# file: {f.relative_to(repo) if f.is_relative_to(repo) else f}\n{body}"
        left -= len(body)
    return out, full


def _primitives(ref: str) -> dict:
    """The embodiment package's ``drivers`` module (the helpers a card reaches by ref:
    constants + function signatures/docstrings) and its ``vla_io`` obs keys / action
    order, when the package has them."""
    pkg = ref.partition(":")[0].rpartition(".")[0]
    out = {}
    for name in ("drivers", "vla_io"):
        try:
            mod = importlib.import_module(f"{pkg}.{name}") if pkg else None
        except ImportError:
            mod = None
        if mod is None:
            continue
        if name == "drivers":
            out["primitives"] = {
                "ref": mod.__name__,
                "constants": {n: v for n, v in vars(mod).items() if n.isupper() and isinstance(v, (int, float))},
                "functions": {f"{n}{inspect.signature(f)}": inspect.getdoc(f) or ""
                              for n, f in vars(mod).items()
                              if inspect.isfunction(f) and f.__module__ == mod.__name__}}
        else:
            out["obs_keys"] = list(getattr(mod, "STATE_KEYS", ()))
            conv = getattr(mod, "lerobot_to_env", None)
            if conv is not None:
                out["action_order"] = inspect.getsource(conv)
    return out


def _contract(pkg: str, skill: str | None, emb: str | None) -> dict:
    """The exact card the harness mounts: template with the concrete strings, and the
    executor contract as the stage driver drives it (kitchen_driver / stage_extras)."""
    ref = f"{pkg}<name>:provider"
    return {
        "card_template": {
            "ref": ref, "to": "<new executor key, e.g. geometric2>",
            "manifest.toml": f'needs_sim = true\n[executors.<to>]\nskill = "{skill}"\n'
                             f'embodiment = "{emb}"\nref = "{ref}"\ntransport = "inproc"\n'
                             '[tunables]\n# your numeric knobs (evolve can perturb them later)\n',
            "__init__.py": "REF = \"" + ref + "\"\n"
                           "class Executor(harness.skill_executor.InprocExecutor): ...  # see executor_contract\n"
                           "class Policies:\n    def make_driver(self, spec): return Executor(...)\n"
                           "def provider(**params): return Policies()  # params: manifest [tunables] "
                           "(+ an evolve overlay) under params.get('tunables')\n"},
        "executor_contract": (
            "provider(**params) -> factory with make_driver(spec) -> a FRESH executor per segment "
            "(spec may be ignored; it is instantiated once with spec=None before the suite). The executor "
            "is a harness.skill_executor.StepExecutor (subclass InprocExecutor for the defaults): "
            "handshake() -> normalize_handshake('inproc', REF, {...meta}) (dict; transport must be 'inproc'); "
            "reset() (called at every segment entry); bind(env, target=None) -- the stage driver calls it "
            "right after enter_segment with the LIVE simulator env and the stage's target object name "
            "(or None): keep env, read poses off it through the primitives by ref "
            "(P = importlib.import_module(primitives.ref); P._eef(env), P._obj_pos(env, name), "
            "P._fixture(env, name), P._base_pose(env)); act(obs) -> numpy array of shape (ADIM,) = the raw "
            "env action (build it with P._arm_action(env, goal_world, grip) / P._base_action(...), "
            "which fill the slots correctly; action_order names every slot); done() -> bool read off env "
            "(the stage's own done() and step cap also end the segment); diagnostics() -> dict. "
            "Never import the embodiment package or another card. An exception or a wrong action shape "
            "inside act() fails the preflight seed and is sent back to you."),
    }


def _driver(before: dict, records: dict, emb: str, arm: str, binding: dict, node=None) -> dict:
    """The round's TARGET node (``node``, else the commonest first death) -> {node, skill,
    executor, rate, executors, tunables, embodiment, task}: what the rules proposer reads,
    projected for the model (same rearm / mount_params seams). ``embodiment`` is the key
    the skill's record binds under."""
    from scripts.evolve import _first_death   # noqa: PLC0415 -- evolve imports this module
    node = node or _first_death(before)
    if node is None:
        return {"node": None}
    runs = [s["nodes"][node] for s in before["seeds"].values() if node in s["nodes"]]
    skill, current = runs[0]["skill"], runs[0]["executor"]
    rec = records.get(skill)
    spec = segment_specs({skill: rec}, emb).get(skill) if rec else {}
    bound = sorted({"scripted", *((spec or {}).get("policies") or {})})
    ev = rec.evidence.get(emb) if rec else None
    ref = ((rearm(spec or {}, arm, current if current in bound else None).get("policy_provider"))
           or binding["policy"])
    params = mount_params(ref)
    tun = params.get("tunables") if isinstance(params.get("tunables"), dict) else params
    values = {k: v for k, v in tun.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    task = (spec or {}).get("task")
    cls = _stage_classes(ref, task)
    mods = sorted({c.__module__ for c in cls})
    return {"node": node, "skill": skill, "executor": current,
            # where a patch MUST initialise the state it introduces -- per NODE, not once for
            # the rotation head: the prompt's checklist points at it, and a round answering on
            # the other death node was reading a list that describes a DIFFERENT class
            "state_init": _state_init(cls),
            "embodiment": emb if rec is None or emb in rec.bindings else next(iter(rec.bindings), emb),
            "task": task,
            "modules": mods,
            # the same list, kept BEFORE a stuck round widens ``modules`` to the whole
            # pipeline: PATCH_CARD.make_stage() only swaps classes this stage's MRO
            # actually contains, so a patch to any other module is a no-op (write_patch)
            "stage_modules": mods,
            "rate": sum(r["success"] for r in runs) / len(runs),
            "executors": {k: dict((ev.by_executor if ev else {}).get(k) or {}) for k in bound},
            "tunables": {"ref": ref, "path": ["tunables"] if isinstance(params.get("tunables"), dict) else [],
                         "values": values,
                         "hints": {m: [k for k in ks if k in values]
                                   for m, ks in (params.get("tunable_hints") or {}).items()}}}


def rsi_projection(doc: dict, before: dict, records: dict, emb: str, arm: str, binding: dict,
                   log_excerpt: list[str]) -> dict:
    """The compact brief the model reads for the round after ``doc['cursor']``."""
    from scripts import evolve   # noqa: PLC0415 -- evolve imports this module
    rounds = doc.get("rounds") or []
    deaths = evolve.death_nodes(before, rounds)
    # One projection per death node, not only the rotation head: payload.node decides which
    # cluster the round is JUDGED on, which skill/task write_patch
    # stamps into the candidate's manifest + PATCH_CARD, which node apply() mounts the
    # executor under, and (below) which node's knobs / modules / state_init the round is
    # measured and refused against. Replayed over evolve-recycle_cans' 588 rounds: of the
    # 252 rounds the rotation put on nav-can1, 207 diagnosed drop-can1; 184 of the 363 patch
    # rounds (50.7%) edited code outside the judged node's stage -- the patch was mounted as
    # nav-can1's executor while the drop segment kept running the stock card (round 556's
    # 4243 after-row is verbatim the baseline). 0 of the 347 focused rounds scored above
    # baseline. _driver is pure reflection, no simulator: one call per death node.
    drivers = {d["node"]: _driver(before, records, emb, arm, binding, d["node"]) for d in deaths}
    fd = drivers.get(evolve._first_death(before, rounds)) or {"node": None}
    ref = fd.get("tunables", {}).get("ref") or binding["policy"]
    cls = _stage_classes(ref, fd.get("task"))
    rows = [_seed_row(seed, s, doc.get("reference") or {}) for seed, s in before["seeds"].items()]
    cl = clusters(rows)
    proj = {
        "task": doc["task"], "embodiment": emb, "seeds": doc["seeds"], "arm": arm,
        "round": int(doc.get("cursor") or 0) + 1,
        "applied": doc.get("applied"),
        "history": [{"round": r["round"], "proposer": r.get("proposer"),
                     "tried": {k: r["tried"][k] for k in ("kind", "node")}
                     | {"detail": {k: v for k, v in r["tried"]["detail"].items()
                                   if k in ("to", "from", "path", "ref", "module", "reason", "error",
                                            "hint", "patch_sha")}},
                     "before": r["before"], "after": r["after"], "published": r["published"],
                     "verdict": r.get("accepted_reason"),
                     # 模型自己那轮说了什么、那轮算什么结局：两个值早就在索引行上（index_row
                     # 保留 llm.summary 和 outcome），零新存储。573/588 轮 parent=0，
                     # 77 个 none 轮的理由是「repeated the same rejected answer」——
                     # 模型看不见自己写过什么，就只会把同一个想法重写一遍。
                     "summary": (r.get("llm") or {}).get("summary"), "outcome": r.get("outcome"),
                     **({"trial_evidence": r["trial_evidence"]} if r.get("trial_evidence") else {}),
                     "per_seed": [{k: s.get(k) for k in ("seed", "success", "first_death", "failure_mode")}
                                  for s in r.get("after_seeds") or r.get("per_seed") or []]}
                    for r in rounds],
        "this_round": {"count": before["count"], "seeds_total": len(before["seeds"]),
                       "per_seed": rows},
        # the diagnosis discipline: the ladder to answer from, the seeds' failure clusters
        # (first missing milestone x failure_mode) and what earlier rounds concluded
        "layers": LAYER_QUESTION, "clusters": cl, "notebook": _notebook(rounds),
        # the gradient: what the last round did to the score, and what is already accepted
        # scripts/evolve.py's structured row ({round, layer, kind, summary, before_score,
        # after_score, regressions}) plus this module's one-line reading of it
        "last_outcome": ({**doc["last_outcome"], "says": _last_outcome(rounds)}
                         if isinstance(doc.get("last_outcome"), dict) else _last_outcome(rounds)),
        "score_definition": doc.get("score_definition") or SCORE_DEF,
        "trial_evidence": _trial_evidence(doc, rounds),
        "accepted_stack": {"note": "已接受的改动（本轮从它们之上出发，它们就是新的 baseline；"
                                   "重复其中任何一条都会被驳回）。",
                           "changes": _accepted(doc, rounds)},
        "first_death": fd,
        # {node: the same shape as first_death} for every death node, so an answer that
        # names another node is materialised against THAT node (_try). Not in the brief:
        # the model already reads death_nodes + first_death, this is the lookup table.
        "drivers": drivers,
        "proposals_consumed": [{"round": r["round"], **r["proposal"]} for r in rounds if r.get("proposal")],
        "needs": rounds[-1].get("needs") if rounds else [],
        "log_excerpt": list(log_excerpt)[:MAX_LOG_LINES],
        **_contract(_card_package(CANDIDATES_ROOT), fd.get("skill"), fd.get("embodiment")),
        "reference_card": {f: (REFERENCE_CARD / f).read_text() for f in ("manifest.toml", "__init__.py")
                           if (REFERENCE_CARD / f).is_file()},
        "scripted_driver_source": _stage_source(ref, fd.get("task")),
        **_primitives(ref),
        "output_schema": PROPOSAL_SCHEMA, "payload_by_kind": PAYLOAD_BY_KIND,
        # every node the seeds die at, least-recently-targeted first: the round targets
        # the head of this list so no node (drop-can1 for 66 rounds) eats the campaign
        # ...each with what an answer ON that node needs: target.judged invites the switch,
        # and ``drivers`` is material, not brief -- without these the model reads the HEAD's
        # state_init / modules and writes a patch against another class (SelfCheckError
        # "reads self.<attr>", 4 live rounds) or against a module that stage never inherits.
        "death_nodes": [{k: d[k] for k in ("node", "seeds", "failure_mode", "rounds_targeted")}
                        | {k: v for k, v in (drivers.get(d["node"]) or {}).items()
                           if k in ("modules", "stage_modules", "state_init")}
                        for d in deaths],
        "target": {"node": fd.get("node"),
                   "cluster": next((c for c in cl if c["milestone"] == fd.get("node")), None),
                   "why": ("the least-recently-targeted of the seeds' first-death nodes"
                           if len(deaths) > 1 else "the only first-death node"),
                   # measured: with two death nodes the target alternates every round while the
                   # model kept patching the other one -- half of 300 rounds could not be accepted
                   # whatever they contained. The judgement follows payload.node, so say it.
                   "judged": ("本轮只在这个节点所属簇的种子上判定：改别的节点，这一轮无论多好都不会被"
                              f"接受。如果你要改的是 death_nodes 里的另一个节点，就在 payload.node 里"
                              f"写那个节点 id，判定会跟着走（当前默认 {fd.get('node')!r}）。")},
        "stuck_rounds": evolve.STUCK_ROUNDS,
    }
    if st := evolve.stuck_on(fd.get("node"), rounds):
        # stuck: name the whole stage pipeline as patchable and say the knobs are spent.
        # Widen EVERY death node's table: write_patch checks ``modules`` of payload.node, so
        # widening only the head's takes the promise back the moment the model switches node.
        for d in drivers.values():
            d["modules"] = sorted({*(d.get("modules") or []),
                                   *evolve.pipeline_modules((d.get("tunables") or {}).get("ref") or ref,
                                                            binding)})
        proj["stuck"] = {
            **st, "modules": fd["modules"],
            # ...minus the per-round bookkeeping ids ("executor patch_r412"): write_patch names
            # an unnamed patch after its round, so ~185 of the campaign's 205 rows were unique
            # to one round and could never be hit again -- noise, and it rode every prompt.
            "tried": sorted(f"tunables {k[1]} {_DIR[k[2]]}" if k[0] == "tunables" else f"executor {k[1]}"
                            for k in _tried_pairs(proj)
                            if not (k[0] == "executor" and str(k[1]).startswith("patch_r"))),
            "note": (f"{st['rounds']} rounds targeted {st['node']} with no improvement: parameter "
                     "tweaks on it are exhausted (stuck.tried lists them). Change CODE -- a patch "
                     "may name ANY module in stuck.modules now, not only the stage class's own -- "
                     "or target another node of death_nodes.")}
    # last, so a stuck round's widened fd["modules"] gets its source too: the real text of
    # every module a patch may edit, numbered, what an exact-snippet `old` is copied out of
    # ...for EVERY death node's modules, not the head's: an answer on the other node has to
    # copy its `old` out of a module the head may not even list. Costs nothing in the regime
    # that matters -- the two nodes of recycle_cans overlap in stage_extras (3 modules, not
    # 2x2) and every stuck round already widened this to the whole pipeline: the 50 unpruned
    # live audits all carried 4 modules / 70,892 chars, against 66,552 for this union and
    # 10,547 for a pre-stuck head alone. MODULE_CHARS still decides whole text vs extract.
    editable = sorted({m for d in drivers.values() for m in d.get("modules") or ()})
    proj["module_sources"], full = _module_sources(editable, cls[-1].__module__ if cls else None)
    for d in drivers.values():
        d["modules_full"] = [m for m in full if m in (d.get("modules") or ())]
    # the same code with no line numbers -- what `old` is copied from, for every editable module
    proj["functions"] = _functions(cls, editable)
    size = lambda: len(json.dumps(proj, sort_keys=True, default=str))
    if size() > PROMPT_CHARS:
        proj["log_excerpt"] = proj["log_excerpt"][-MAX_LOG_LINES // 4:]
    if size() > PROMPT_CHARS:
        for r in proj["history"][:-5]:
            r.pop("per_seed", None)
    if size() > PROMPT_CHARS and proj["scripted_driver_source"]:
        keep = max(2000, PROMPT_CHARS - size() + len(proj["scripted_driver_source"]))
        proj["scripted_driver_source"] = proj["scripted_driver_source"][:keep]
    return proj


def _fmt_score(v) -> str:
    return "(" + ", ".join(str(x) for x in v) + ")" if isinstance(v, (list, tuple)) else str(v)


def _score(r: dict) -> tuple:
    """A round's score before -> after: the campaign's ``score`` tuple when scripts/evolve.py
    wrote one, else the bare success counts (which are 0 -> 0 for whole campaigns)."""
    sc = r.get("score")
    if isinstance(sc, dict):
        return sc.get("before"), sc.get("after")
    return (r.get("before_score", r.get("score_before", r.get("before"))),
            r.get("after_score", r.get("score_after", r.get("after"))))


def _chain(row: dict) -> list:
    """One seed's node ids in plan order (the milestone chain), off whichever of ``nodes`` /
    ``trail`` the row carries."""
    ns = row.get("nodes") or row.get("trail") or []
    return [n.get("id") for n in ns if isinstance(n, dict)] if isinstance(ns, list) else list(ns)


def _moved(r: dict) -> list[str]:
    """Where each seed's death MOVED between this round's before-suite and its trial suite.
    Earlier in the chain = the trial lost ground = worse, however flat the success count is."""
    was = {s.get("seed"): s for s in r.get("per_seed") or []}
    out = []
    for row in r.get("after_seeds") or []:
        seed, b = row.get("seed"), (was.get(row.get("seed")) or {})
        a, b0 = row.get("first_death"), b.get("first_death")
        if a == b0:
            continue
        ch = _chain(row) or _chain(b)
        i, j = (ch.index(b0) if b0 in ch else -1), (ch.index(a) if a in ch else -1)
        way = "前移" if -1 < j < i else "后移" if j > i > -1 else "变为"
        out.append(f"种子 {seed} 的死亡点从 {b0} {way}到 {a} = "
                   + {"前移": "变差", "后移": "变好"}.get(way, "存疑"))
    return out


SCORE_DEF = ("score 是逐位比较的元组：第一位是整任务成功的种子数，其后是走到的里程碑数（越远越好）。"
             "整个战役成功数都是 0 时，唯一的梯度就是里程碑位——优先让最远到达的里程碑再往前一步；"
             "拿一个已经通过的节点去换目标节点（死亡点前移）算变差。"
             "修复节点（recover-<节点>）不计里程碑：它只因该节点失败才被插入，"
             "所以修好本体让 recover 消失是加分而不是退步，靠多插修复节点也换不到分。")


def _last_outcome(rounds: list) -> str | None:
    """The last round as ONE line of gradient: layer + kind, the score before -> after, and
    which way each seed's death moved. Round 86 wrote the right fix, scored 0 -> 0 and was
    filed "same" -- the movement is the signal the success count cannot carry."""
    if not rounds:
        return None
    r = rounds[-1]
    t = r.get("tried") or {}
    b, a = _score(r)
    moved = _moved(r)
    return (f"上一轮（第 {r.get('round')} 轮）：{(t.get('detail') or {}).get('layer') or r.get('layer') or '-'} "
            f"{t.get('kind')}，score {_fmt_score(b)} → {_fmt_score(a)}，"
            + ("；".join(moved) if moved else "各种子的死亡点不变 = 持平")
            + ("（已接受，是本轮的 baseline）" if r.get("accepted") or r.get("published")
               else "（未接受，本轮仍从已接受状态出发）")
            # 判定语和「持平」写在同一行：光说持平，模型看不出是分没动还是判定把它否了
            + (f"｜判定：{r['accepted_reason']}" if r.get("accepted_reason") else ""))


def _accepted(doc: dict, rounds: list) -> list[dict]:
    """The accepted stack: what this campaign has ALREADY accepted (campaign.json ``accepted``
    when scripts/evolve.py keeps one, else the published rounds). Each is the new baseline."""
    stack = doc.get("accepted_stack") or doc.get("accepted")
    if isinstance(stack, list):
        return stack
    keep = ("to", "from", "path", "ref", "module", "edits")
    return [{"round": r.get("round"), "kind": (r.get("tried") or {}).get("kind"),
             "node": (r.get("tried") or {}).get("node"),
             **{k: v for k, v in ((r.get("tried") or {}).get("detail") or {}).items() if k in keep}}
            for r in rounds if r.get("accepted") or r.get("published")]


def _trial_line(ev: dict) -> str:
    """``scripts.evolve.trial_evidence`` (the round row's, or ``last_outcome``'s flattened
    copy) as ONE Chinese line: the exception with its file:line, else per seed how far it got,
    how close it came and -- the point of the whole thing -- whether it diverged from the
    baseline at all. ``种子 4243 在 drop-can1 抛 AttributeError ...`` / ``种子 4243 在 nav-can1
    跑到第 41 步，d_eef 最小 0.57→0.55，底盘仍未移动，与基线逐步完全相同＝你的改动没有生效``.
    缺席也是一条读数：after 侧没有 trace 说「测不到」（不是「与基线不同」），节点根本没跑到
    说「没执行到」（不是「你的改动没有生效」），执行器没交回 failure_mode 说「测不到」
    （不是「卡顿没了」）。本战役 588 轮里 365 条 trial 记录：355 条被渲染成假的
    「第 N 步起与基线不同」（全是「第 1 步」）、6 条被渲染成「你的改动没有生效」；把这
    398 个有 trial_evidence 的轮次喂回本函数，带 failure_mode 键的那一版会在几乎每一条
    上打印「failure_mode reach_stall→无」——那个「无」是「有没有绑执行器」的函数。"""
    node, exc, out = ev.get("node") or "?", ev.get("exception"), []
    seeds = [r.get("seed") for r in ev.get("seeds") or ()]
    if isinstance(exc, dict) and exc.get("type"):
        where = f"（{Path(exc['file']).name}:{exc.get('line')}）" if exc.get("file") else ""
        return (f"种子 {seeds[0]} " if seeds else "") + \
            f"在 {node} 抛 {exc['type']}: {str(exc.get('message') or '')[:200]}{where} —— 这是你自己写的代码"
    for r in (ev.get("seeds") or ())[:4]:
        d = r.get("diff") if isinstance(r.get("diff"), dict) else r
        bits, num = [], lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        if d.get("steps_after") is None and d.get("ok_after") is not True:
            # 本战役 365 条 trial 记录里 6 条（131/232/292/357/371/389，全是 improved 轮）
            # 对一个更上游就死掉、根本没跑到的节点说「你的改动没有生效」——模型据此又改了
            # 一遍同一处代码。这里说实话。
            out.append(f"种子 {r.get('seed')} 上 {node} 这一轮根本没执行到"
                       "（更上游的节点先失败），这颗种子对你的改动没有任何读数")
            continue
        fb, fa = d.get("failure_mode_before"), d.get("failure_mode_after")
        if "failure_mode_after" not in d:
            # 键缺席 = 装上去的执行器根本没交回 failure_mode（D.merge_executor_diagnostics
            # 让沉默保持沉默）。旧代码把这种沉默渲染成「reach_stall→无」，在本战役
            # 每个候选轮都告诉模型卡顿治好了：被判定节点在 365 条候选 trial 记录里
            # 364 条读作 None（347 条跑满 cap），而脚本侧同样两个节点在 580 轮的基线行里
            # 照常报 stall —— 那个 None 是「有没有绑执行器」的函数，不是候选的功劳。
            if fb:
                bits.append(f"failure_mode 基线 {fb} → 本轮测不到（执行器没交回这个读数）")
        elif fb != fa:
            bits.append(f"failure_mode {fb or '无'}→{fa or '无'}")
        if d.get("steps_after") is not None:
            bits.append(f"跑到第 {d['steps_after']} 步"
                        + (f"（基线 {d['steps_before']} 步）" if d.get("steps_before") != d.get("steps_after") else ""))
        for k in ("d_eef", "d_base"):
            b, a = d.get(f"{k}_min_before"), d.get(f"{k}_min_after")
            # 缺席也要说出来：旧的 `if num(a)` 把已经测到的 before 值一起扔了，378 个有
            # trial_evidence 的轮次渲染出的行里一个距离数字都没有。
            if num(a):
                bits.append(f"{k} 最小 {b:.3f}→{a:.3f}" if num(b) else f"{k} 最小 {a:.3f}")
            elif num(b):
                bits.append(f"{k} 最小 基线 {b:.3f} → 本轮测不到")
        # 基线量到过、本轮量不到，才叫「测不到」；两边本来就没有 series（假舞台）不算丢读数
        had = num(d.get("d_eef_min_before")) or num(d.get("d_base_min_before"))
        if d.get("base_moved") is not None:
            bits.append("底盘移动了" if d["base_moved"] else "底盘仍未移动")
        elif had:
            bits.append("底盘是否移动测不到")
        if d.get("phase_changed"):
            bits.append("阶段序列变为 " + "→".join(str(x) for x in d["phase_changed"]))
        # 「没有生效」和「测不到」是两回事。旧代码把 355/365 条空 after series 说成
        # 「第 1 步起与基线不同」（_divergent 的空表分支），说的都是不存在的观测。
        # after 侧到底有没有 series，看它有没有留下任何读数（base_moved 非 None 就说明有）。
        measured = (num(d.get("d_eef_min_after")) or num(d.get("d_base_min_after"))
                    or d.get("base_moved") is not None or bool(d.get("phase_changed")))
        if had and not measured:
            bits.append("本轮该节点没留下逐步 trace，是测不到、不是没变化")
        elif d.get("first_divergent_step") is not None:
            bits.append(f"第 {d['first_divergent_step']} 步起与基线不同")
        elif d.get("steps_before") == d.get("steps_after"):
            bits.append("与基线逐步完全相同＝你的改动没有生效")
        if bits:
            out.append(f"种子 {r.get('seed')} 在 {node} " + "，".join(bits))
    return "；".join(out)


def _trial_evidence(doc: dict, rounds: list) -> str | None:
    """What the LAST round's change actually DID in the simulator, as the FIRST line of the
    brief -- until now the model saw only "score 0 → 0" and could not tell whether its own
    code ran, crashed or changed nothing. Read off ``scripts/evolve.py``'s measurement:
    ``last_outcome.trial_evidence`` (flattened) / the last round row's ``trial_evidence``."""
    lo = doc.get("last_outcome")
    ev = ((lo.get("trial_evidence") if isinstance(lo, dict) else None)
          or (rounds[-1].get("trial_evidence") if rounds else None) or doc.get("trial_evidence"))
    line = (_trial_line(ev) if isinstance(ev, dict) else
            "；".join(str(x) for x in ev) if isinstance(ev, (list, tuple)) else str(ev or "")).strip()
    return f"你上一轮的补丁跑了：{line[:900]}" if line else None


_RAISED = re.compile(r"Traceback|[A-Za-z]+Error|raised|抛")


def _runtime_streak(hist: list, node) -> int:
    """How many of the LAST rounds in a row targeted ``node`` and ended in a RUNTIME error of
    the model's own code (``trial_evidence`` / the try's reason). Three in a row means the
    model is rewriting the same broken idea: the brief asks for a smaller one."""
    n = 0
    for r in reversed(hist or ()):
        t = r.get("tried") or {}
        ev = r.get("trial_evidence")
        raised = (bool(ev.get("exception")) if isinstance(ev, dict) else bool(ev and _RAISED.search(str(ev)))) \
            or bool(_RAISED.search(json.dumps(t.get("detail"), default=str, ensure_ascii=False)))
        if t.get("node") != node or not raised:
            break
        n += 1
    return n


SMALLER = ("{node} 连续 {n} 轮死在运行时错误（你自己写的代码抛异常）。这一轮换个打法：做一个更小、"
           "自洽的改动——只改一个阶段、加一个守卫，不引入需要初始化的新状态——或者改去 death_nodes "
           "里的另一个节点。")


def _dock_bound(g: dict, up: dict) -> tuple[float, float, float] | None:
    """The standoff band that puts the point in reach, and the standoff the run is at now.

    The base parks on the ray dock->base (the loaded nav leg stops ``carry_stop`` from the dock
    it drove at), so a standoff ``s`` puts it at ``dock + s*u`` and the reach condition is
    ``|dock + s*u - point| <= reach_max`` -- one quadratic, whose solution is the CLOSED
    INTERVAL between BOTH roots: a standoff under the lower root has driven PAST the point and
    is out of reach on the other side. Returns ``(s_min, s_max, current)``, metres. On
    recycle_cans/4243 (dock 0.86,-1.815 from the carry row's trace_end; base 1.45,-1.728;
    point 0.488,-1.364) that band is [-0.739, 0.134] -- the lower root sits behind the dock,
    so only 0.134 binds -- against a standoff of 0.596. None when the row carries no dock, or
    when the ray never enters reach at all (discriminant < 0) -- then no standoff works and
    (a) is not a lever at all.
    """
    dock = ((up.get("trace_end") or {}).get("target") or ())[:2]
    base, point, reach = (g.get("base") or ())[:2], (g.get("point") or ())[:2], g.get("reach_max")
    if not all(len(v) == 2 for v in (dock, base, point)) \
            or not all(isinstance(v, (int, float)) for v in (*dock, *base, *point, reach)):
        return None
    ux, uy = base[0] - dock[0], base[1] - dock[1]
    cur = math.hypot(ux, uy)
    if cur < 1e-6:
        return None
    ux, uy = ux / cur, uy / cur
    wx, wy = point[0] - dock[0], point[1] - dock[1]
    b = ux * wx + uy * wy
    disc = b * b - (wx * wx + wy * wy - reach * reach)
    if disc < 0:
        return None
    r = math.sqrt(disc)
    return (round(b - r, 3), round(b + r, 3), round(cur, 3))


#: The knobs ``_reach_wall`` names as the only levers on a geometric gap. ``_try`` lets these
#: two -- and ONLY these two -- back through the ``seen`` gate while the wall stands: the
#: campaign spent carry_stop (r46 0.7, r52 0.5) and nudge_max (r45 0.2, r51 0.1) blind in its
#: first 55 rounds, before any geometry rode the brief, and the ``seen`` gate then refused
#: them for the 533 rounds that had the numbers. A knob tried without a target is not a
#: knob tried. Everything else on the node stays closed (``exhausted`` says so).
WALL_KNOBS = ("carry_stop", "nudge_max")

WALL_OPEN = (" 例外：reach_wall 点名的 carry_stop / nudge_max 仍然可以再答一次（按 reach_wall 里的目标"
             "数字答，不是再瞎试一个方向）——这一层的其它旋钮不行。")


def _reach_wall(proj: dict) -> str | None:
    """The target's drop/place point is farther from the base than the arm can reach.

    The stage commands the ARM only (``trace.series[].cmd.mode == "arm"`` on every row), so
    no edit inside it can close that gap, and clamping the point into reach moves it off the
    fixture the ``placed`` predicate scores -- both measured on recycle_cans/4243. Name the
    levers that CAN close it, or the model spends the campaign re-deriving the diagnosis.

    Read off the SEEDS' rows, not the rotation head's driver, so it shows on the rounds the
    rotation put on the other node too: 505 of the live campaign's 588 rounds, every one of
    them solving ``_dock_bound``. It is silent on rounds 1..83 only because the baseline trail
    carried no ``geometry`` before round 84. Note the reach: ``_seed_row`` keeps ``geometry``
    on the FIRST-DEATH row alone, so a wall on a later node is still invisible -- this fired
    every round because 4243's first death was drop-can1 throughout.

    The levers are the ones that exist AND that this module's own gates let through.
    "Insert a navigation node before this one" was in here and is not a lever: no answer kind
    edits the plan graph -- a patched planner copy is never imported, and all 18 planner-patch
    rounds ran with zero effect. Neither is patching the base driver's dead zone: ``drivers``
    is not in this node's ``stage_modules``, so ``write_patch`` refuses it on sight (it
    refused 171 live rounds that way). What is left is two TUNABLES -- the upstream standoff
    and the nudge bound -- each with the number it would have to reach and, honestly, whether
    it can get there. The model mentioned neither knob in any of those 505 rounds.
    """
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    for r in (proj.get("this_round") or {}).get("per_seed") or ():
        for n in r.get("trail") or ():
            g = n.get("geometry") or {}
            d, reach = g.get("d_base_point"), g.get("reach_max")
            if not num(d) or not num(reach) or d <= reach:
                continue
            gap = round(d - reach, 3)
            up = n.get("upstream") or {}
            drv = (proj.get("drivers") or {}).get(n["id"]) or proj.get("first_death") or {}
            tv = ((drv.get("tunables") or {}).get("values")) or {}
            cs, nm = tv.get("carry_stop"), tv.get("nudge_max")
            sm = drv.get("stage_modules") or ()
            bound = _dock_bound(g, up)
            d_up = (up.get("trace_end") or {}).get("d_base_target")   # where the CARRY leg parked
            near = "（它的 dock / carry_stop）。"
            if bound:
                lo, hi, cur = bound
                # the two are NOT the same quantity: carry_stop is the COMMAND to the loaded
                # leg, `cur` is where the base stands after the base_nudge recovery pulled it
                # in further. Live 4243: knob 0.65 -> leg parked 0.644 -> nudge -> 0.596, and
                # the bound is 0.134, so "set carry_stop to 0.134" is 0.054 short of the mark.
                near = ("沿 dock→base 这条射线解出来，底盘离 dock 的停靠距离得"
                        + (f"≤ {hi} m" if lo <= 0 else f"落在 {lo}–{hi} m 之间")
                        + f"，现在 {cur} m"
                        + (f"（carry 段停在 {d_up} m，base_nudge 又挪近了 {round(d_up - cur, 3)} m）"
                           if num(d_up) else "")
                        + "。注意 carry_stop 是发给 carry 段的命令值，不等于这个停靠距离："
                        + (f"现在 {cs} 兑现成 {d_up} m，要把停靠距离压到 {hi} m，得把它调到 "
                           f"{round(cs - (cur - hi), 3)} 上下"
                           if num(cs) and num(d_up) else f"要把停靠距离压到 {hi} m，得把它往下调 "
                           f"{round(cur - hi, 3)} m 左右")
                        + "。能不能兑现是另一回事：装载腿的已知天花板是「离 dock 0.6–0.8 m 就被台面/"
                          "家具边卡住」（drivers.NavigateDriver 的实测，0.50 已经量过太紧），所以它多半"
                          "停不到那么近。答一次，试验会把它实际停在哪量给你看。")
            nudge = (f"它的行程上限是 tunables nudge_max（现在 {nm} m）"
                     if num(nm) else "它的行程上限是 tunables nudge_max（0.15 m）")
            moved = (f"但它实测只把底盘挪了 {round(d_up - bound[2], 3)} m，"
                     if bound and num(d_up) else "但它实测挪不到那么多，")
            vcap = ("而那个死区在 drivers 里，不在这个节点的 stage_modules 里"
                    f"（{'、'.join(map(str, sm))}）——patch 它会被当场驳回，别往那儿写。"
                    if sm and not any(str(m).endswith(".drivers") for m in sm) else
                    "改它要动底盘驱动模块本身。")
            return (f"{n['id']}：目标点离底盘 {d} m，臂展只有 {reach} m，差 {gap} m。"
                    "这一段全程只发臂指令（trace.series 每一行 cmd.mode 都是 'arm'），所以在这一段内部"
                    "改任何参数或代码都不可能补上这个距离。把目标点夹进臂展内也不行：点会离开夹具，"
                    "placed 判据照样不过（已经实测过）。能补上这段距离的只有两个旋钮，都在 tunables 上："
                    f"(a) 让上游 {up.get('node') or '搬运'} 段的底盘停得更近——{near}"
                    "(b) 把已经在跑的 base_nudge 恢复节点的行程放大：planner 的 _RECOVERY_BY_MODE 把 "
                    "reach_stall 映射到 base_nudge，这个节点每轮都触发（trail 里的 recover-drop-can1），"
                    f"{nudge}，而要补的是 {gap} m —— 这个上限本身就不够，要走这条路得调到 ≥ {gap}。"
                    f"{moved}连现在的上限都没用满，所以先卡住它的不是这个上限，而是底盘驱动的摩擦死区"
                    f"（VCAP 实测：cmd 0.20 只走 0.20 mm/step，0.35 才 3.5）；{vcap}"
                    "这两个旋钮就算 campaign 已经试过，在这堵墙下也可以再答一次——但要按上面这两个数答。")
    return None


def brief(proj: dict) -> dict:
    """Call 1's compact brief: the projection minus ``MATERIAL_KEYS``, plus ``untried`` (what
    is left on the first-death node -- the only thing that makes a ``none`` acceptable), the
    last 5 rounds detailed and the older ones as counts, bounded to ``BRIEF_CHARS`` (log
    excerpt first, then the detailed rounds' per_seed)."""
    b = {k: v for k, v in proj.items() if k not in MATERIAL_KEYS and k != "drivers"}
    hist = b.get("history") or []
    old, b["history"] = hist[:-5], [dict(r) for r in hist[-5:]]
    if old:
        b["history_older"] = {"rounds": len(old), "published": sum(bool(r["published"]) for r in old),
                              "kinds": dict(Counter(r["tried"]["kind"] for r in old))}
    b["untried"] = _untried(proj, _tried_pairs(proj))
    fd = proj.get("first_death") or {}
    if (n := _runtime_streak(hist, fd.get("node"))) >= 3:
        b["repeat_failure"] = SMALLER.format(node=fd.get("node"), n=n)
    if wall := _reach_wall(proj):
        b["reach_wall"] = wall
    if _exhausted(proj, b["untried"]):
        b["exhausted"] = (f"tunables exhausted for {fd.get('node')}: every (knob, direction) is tried. "
                          f"The parameter layer is CLOSED here -- diagnose top-down and answer from "
                          f"{', '.join(LAYERS[:-1])}." + (WALL_OPEN if wall else ""))
    b["log_excerpt"] = list(b.get("log_excerpt") or [])
    size = lambda: len(json.dumps(b, sort_keys=True, default=str))
    while size() > BRIEF_CHARS and b["log_excerpt"]:
        del b["log_excerpt"][0]
    for r in b["history"]:
        if size() > BRIEF_CHARS:
            r.pop("per_seed", None)
    return b


def endpoint():
    """The model_endpoint card, mounted by ref the way planner_vlm does (its declared
    params; DeepSeek preset when the card names none); the fake when tests ask."""
    if os.environ.get("PH_MODEL_ENDPOINT_FAKE"):
        return load_provider(FAKE_REF, {})
    return load_provider(ENDPOINT_REF, mount_params(ENDPOINT_REF) or {"preset": "deepseek"})


def _parse(text: str) -> dict:
    try:
        ans = json.loads(text)
    except ValueError:
        start = text.find("{")
        if start < 0:
            raise ValueError("reply contains no JSON object")
        ans, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(ans, dict):
        raise ValueError("reply is not a JSON object")
    ans.setdefault("kind", ans.get("decision"))
    if ans["kind"] not in KINDS:
        raise ValueError(f"decision must be {'|'.join(KINDS)}, got {ans.get('kind')!r}")
    if not isinstance(ans.get("payload"), dict) and ans["kind"] not in ("none", "card", "patch"):
        raise ValueError("payload must be an object")
    if not isinstance(ans.get("summary"), str) or not ans["summary"].strip():
        # bookkeeping, never worth an attempt: evolve.py's last_outcome already falls back to
        # the round's own reason when it is empty. The live model was sent back for this one
        # missing line 22 times, in 11 rounds that burned 1,070,014 tokens between them and
        # 7 of which ended in none -- fill it in instead.
        ans["summary"] = next((t for k in ("rationale", "notes")
                               if (t := str(ans.get(k) or "").strip())), ans["kind"])[:300]
    ans["payload"] = dict(ans.get("payload") or {})
    if set(ans["payload"]) == {ans["kind"]} and isinstance(ans["payload"][ans["kind"]], dict):
        ans["payload"] = dict(ans["payload"][ans["kind"]])   # {"payload": {"executor": {...}}}: seen live
    if (lay := ans.get("layer")) is not None and lay not in LAYERS:
        raise ValueError(f"layer must be one of {'|'.join(LAYERS)} (Zetta's ladder, top-down), got {lay!r}")
    ans["rationale"] = str(ans.get("rationale") or "")
    return ans


def _card_package(root: Path) -> str:
    """Import name of ``<root>/<name>``: ``plugins.candidates.<name>`` for the repo root,
    else ``<name>`` with the root put on sys.path (a test's tmp root)."""
    repo = PLUGINS_ROOT.parent
    if root.resolve().is_relative_to(repo) and (root.resolve() != repo):
        return ".".join(root.resolve().relative_to(repo).parts) + "."
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return ""


def write_card(pay: dict, root: Path | None = None) -> str | None:
    """Materialise a ``card`` answer under ``root/<name>/`` and doctor it; fills
    ``pay['path']``. Returns the refusal reason (``doctor:<first finding>``) or None."""
    root = root or CANDIDATES_ROOT   # read at call time: a test points the root elsewhere
    name, files, ref = pay.get("name"), pay.get("files"), pay.get("ref")
    if not isinstance(name, str) or not _NAME.match(name):
        return f"doctor:card name {name!r} is not [a-z][a-z0-9_]{{2,40}}"
    if not isinstance(files, dict) or {"manifest.toml", "__init__.py"} - set(files):
        return "doctor:card files must include manifest.toml and __init__.py"
    if any(not isinstance(v, str) or Path(k).name != k or k in (".", "..") for k, v in files.items()):
        return "doctor:card files must be plain file names with string bodies"
    pkg = _card_package(root)
    mod = ref.partition(":")[0] if isinstance(ref, str) else ""
    if not mod or not (mod == pkg + name or mod.startswith(pkg + name + ".")):
        # one keystroke -- the model dropped the ``plugins.candidates.`` prefix -- ended
        # NINE consecutive rounds (17-25 of evolve-recycle_cans) in none, this refusal the
        # last word of every one of them; how many attempts each burned is gone with the
        # pruned audits. Give it the literal instead of the rule. It is not prefixed automatically: plugin_doctor reads
        # the ref out of the manifest the model itself wrote, so both copies must say it.
        return (f"doctor:ref {ref!r} must name a provider inside {pkg + name}: write it exactly as "
                f"{pkg + name}:provider, in `ref` AND in the ref line of the manifest.toml you send.")
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for k, v in files.items():
        (d / k).write_text(v)
    pay["path"] = str(d)
    return _doctor(d, ref, pay)


def _doctor(d: Path, ref: str, pay: dict) -> str | None:
    """A materialised candidate dir: plugin_doctor, then the dry instantiation."""
    importlib.invalidate_caches()   # the root went on sys.path before the dir existed
    try:
        rep = plugin_doctor.check(d)
    except Exception as exc:  # noqa: BLE001 -- a manifest that does not parse is a red
        return f"doctor:manifest {type(exc).__name__}: {exc}"
    red = next((r for r in rep.results if r.status == "FAIL"), None)
    if red is not None:
        return f"doctor:{red.name} {red.detail}"
    try:
        return dry_run(ref, dict(pay.get("params") or {}), pay.get("transport", "inproc"))
    except Exception as exc:  # noqa: BLE001 -- the ref the suite would mount must load
        return f"doctor:ref {ref!r} {type(exc).__name__}: {exc}"


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_diff(text: str, diff: str) -> str:
    """A unified diff applied in pure Python: each ``@@`` hunk's old side (context +
    removed lines, compared rstripped) must occur past the previous hunk -- the
    occurrence nearest the stated line wins, so a wrong line number is fine but a
    wrong context line is not; ``ValueError`` names the hunk and what was not found."""
    src, out, pos, n, i = text.split("\n"), [], 0, 0, 0
    lines = diff.splitlines()
    while i < len(lines):
        m = _HUNK.match(lines[i])
        i += 1
        if not m:
            continue   # ---/+++ headers, "diff --git", prose
        n, old, new = n + 1, [], []
        while i < len(lines) and not _HUNK.match(lines[i]) and not lines[i].startswith(("--- ", "+++ ", "diff ")):
            ln = lines[i]
            i += 1
            if ln.startswith("\\"):
                continue   # \ No newline at end of file
            if ln.startswith("-"):
                old.append(ln[1:])
            elif ln.startswith("+"):
                new.append(ln[1:])
            else:   # " ctx" -- or a blank whose leading space the model dropped
                old.append(ln[1:] if ln.startswith(" ") else ln)
                new.append(ln[1:] if ln.startswith(" ") else ln)
        if not old:
            raise ValueError(f"hunk {n} has no context or removed lines: nothing to anchor it on")
        want = [o.rstrip() for o in old]
        # ponytail: O(n*m) scan per hunk; the modules are <1k lines
        hits = [j for j in range(pos, len(src) - len(old) + 1)
                if [s.rstrip() for s in src[j:j + len(old)]] == want]
        if not hits:
            near = next((j for j in range(len(src)) for o in want if o.strip() and src[j].rstrip() == o), None)
            hint = ("" if near is None else "\nnearest matching line %d; the source there reads:\n%s"
                    % (near + 1, "\n".join(src[max(0, near - 2):near + 3])))
            raise ValueError(f"hunk {n} does not apply: these lines were not found (in order, past line {pos + 1}):\n"
                             + "\n".join(old) + hint)
        at = min(hits, key=lambda j: abs(j - (int(m.group(1)) - 1)))
        out += src[pos:at] + new
        pos = at + len(old)
    if n == 0:
        raise ValueError("no @@ hunk in the diff")
    return "\n".join(out + src[pos:])


def _enclosing(src: str, line: int) -> tuple[str, int, int] | None:
    """(what, first, last) of the innermost def / class containing the 1-based ``line``; the
    class when no function does; None when neither (or the module does not parse)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    best = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and n.lineno <= line <= (n.end_lineno or n.lineno) \
                and (best is None or n.lineno > best.lineno):
            best = n
    if best is None:
        return None
    kind = "class" if isinstance(best, ast.ClassDef) else "function"
    return f"{kind} {best.name}", best.lineno - 1, best.end_lineno


#: ``_near`` found not even the first line -- ``_elsewhere`` reads it as "not this module".
_NOWHERE = ("that snippet's first line occurs nowhere in the module -- copy `old` out of "
            "functions[\"<module>:<Class>.<method>\"] (its key names the module to patch); "
            "do not retype it.")


def _near(src: str, old: str, limit: int = 2, run: int = 1) -> str:
    """Where ``old`` comes CLOSEST to occurring: the first window of ``run`` consecutive
    non-blank lines of ``old`` that occurs in ``src``, shown as the WHOLE enclosing function
    (the class when no function encloses it, ±6 lines when neither), numbered, for up to
    ``limit`` such places -- a ±6-line window is not enough text to copy a snippet out of.
    ``run=1`` is a hint about the module the answer ALREADY named, so one line in common is
    fine; a claim about ANOTHER module needs more than one ``return None`` (see
    ``_elsewhere``). ``old`` shorter than ``run`` lines can never match: no hit, no claim."""
    lines, hits = src.split("\n"), []
    srp, want = [l.strip() for l in lines], [l.strip() for l in old.split("\n") if l.strip()]
    for k in range(len(want) - run + 1):
        hits = [i for i in range(len(srp) - run + 1) if srp[i:i + run] == want[k:k + run]][:limit]
        if hits:
            break
    if not hits:
        return _NOWHERE
    out = []
    for i in hits:
        span = _enclosing(src, i + 1)
        head = ("the module around line %d reads:" % (i + 1) if span is None else
                "line %d is in %s, which reads (copy `old` out of THIS text, WITHOUT the "
                "`NNNN| ` line-number prefixes):" % (i + 1, span[0]))
        a, b = (i - 6, i + 7) if span is None else (span[1], span[2])
        out.append(head + "\n" + _numbered(lines, a, b))
    return "\n\n".join(out)


def _lenient(src: list[str], old: str) -> tuple[int, int, int] | None:
    """The ONE window of ``src`` whose lines equal ``old``'s once leading indentation and
    trailing whitespace are ignored -- exactly what a RETYPED snippet gets wrong. Returns
    (start, length, indent delta) or None when it matches zero or several places."""
    want, srp = [l.strip() for l in old.split("\n")], [l.strip() for l in src]
    n = len(want)
    hits = [i for i in range(len(src) - n + 1) if srp[i:i + n] == want]
    if len(hits) != 1:
        return None
    j = next((k for k in range(n) if want[k]), 0)
    ind = lambda l: len(l) - len(l.lstrip())
    return hits[0], n, ind(src[hits[0] + j]) - ind(old.split("\n")[j])


def _shift(new: str, delta: int) -> str:
    """``new`` re-indented by ``delta`` columns, so a lenient match keeps the file's own
    indentation instead of the model's."""
    if not delta:
        return new
    return "\n".join(l if not l.strip() else " " * delta + l if delta > 0
                     else l[min(-delta, len(l) - len(l.lstrip())):] for l in new.split("\n"))


def apply_edits(text: str, edits: list, modes: list | None = None) -> str:
    """Exact-snippet edits: each ``{old, new}``'s ``old`` is replaced by ``new`` where it
    occurs EXACTLY ONCE -- ``"exact"`` (whitespace included) first, then ``"lenient"``: the
    same lines ignoring leading indentation and trailing whitespace, still exactly one hit,
    with ``new`` re-indented to the file. ``modes`` collects which mode matched per edit (the
    round detail records it). ``ValueError`` names the edit, the count and the full enclosing
    function."""
    for i, e in enumerate(edits, 1):
        old, new = (e or {}).get("old"), (e or {}).get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError(f"edit {i} must be {{old: <non-empty snippet>, new: <replacement>}}, got {e!r}")
        if old == new:
            raise ValueError(f"edit {i} changes nothing: old == new")
        n = text.count(old)
        span = None if n == 1 else _lenient(text.split("\n"), old)
        if n != 1 and span is None:
            raise ValueError(f"edit {i}: `old` occurs {n} times in the module, it must occur exactly once"
                             + (" (add the surrounding lines to make it unique)" if n > 1 else "")
                             + " -- not even ignoring indentation and trailing whitespace"
                             + f". You sent:\n{old}\n" + _near(text, old))
        if span is None:
            text = text.replace(old, new, 1)
        else:
            src = text.split("\n")
            text = "\n".join(src[:span[0]] + _shift(new, span[2]).split("\n") + src[span[0] + span[1]:])
        if modes is not None:
            modes.append("exact" if span is None else "lenient")
    return text


_IMPORT = re.compile(r"^from (plugins\.[\w.]+) import ([\w ,]+)$", re.M)


def _by_ref(text: str) -> str:
    """``from plugins.<pkg> import m as a`` -> ``a = importlib.import_module(...)``: the
    copy reaches the installed package BY REF like every card (tests/test_boundaries)."""
    def sub(m):
        return "\n".join("%s = importlib.import_module(%r)" % (n[-1], f"{m.group(1)}.{n[0]}")
                         for n in (x.strip().split(" as ") for x in m.group(2).split(",")))
    new = _IMPORT.sub(sub, text)
    if new == text:
        return text
    fut = "from __future__ import annotations"
    return new.replace(fut, fut + "\nimport importlib", 1) if fut in new else "import importlib\n" + new


#: The generated card of a patch answer: the installed stage table builds the stage
#: with the patched classes swapped in for the call (the copy's class for every class
#: the patched module defines, re-derived subclasses for the rest), and the executor
#: drives it through the native seam; the stage keeps done() / the cap on the driver.
PATCH_CARD = '''"""{name}: {module} patched by the evolve proposer (round {round}); executor {to} for {skill}."""
import importlib
import sys

from harness.skill_executor import InprocExecutor, normalize_handshake

REF = "{ref}"
INSTALLED = "{installed}"   # the card's _STAGES table -- read, never written
PATCHED = "{module}"        # the installed module whose copy ({base}.py) carries the diff
TASK = "{task}"


def _repoint(cls, mod):
    # ponytail: a REBUILT subclass keeps the old __class__ cell, so a zero-arg super() inside
    # it raises TypeError (stage_extras.NavToObjectDriver over a patched drivers.py); rebind
    # those cells with types.FunctionType when a patch needs a base module's subclasses.
    if cls is object:
        return cls
    if cls.__module__ == PATCHED:
        return getattr(mod, cls.__name__)
    bases = tuple(_repoint(b, mod) for b in cls.__bases__)
    if bases == cls.__bases__:
        return cls
    return type(cls.__name__, bases, {{k: v for k, v in vars(cls).items() if k not in ("__dict__", "__weakref__")}})


def make_stage():
    mod, orig = importlib.import_module(__name__ + ".{base}"), importlib.import_module(PATCHED)
    if hasattr(mod, "mount_tunables"):   # the copy sees the effective knobs of the installed module
        mod.mount_tunables(orig.tunables())
    factory = importlib.import_module(INSTALLED)._STAGES[TASK][0]
    swaps = {{}}
    for c in type(factory()).__mro__[:-1]:
        r = _repoint(c, mod)
        if r is not c:
            swaps[(sys.modules[c.__module__], c.__name__)] = r
    saved = {{k: getattr(*k) for k in swaps}}
    try:   # ponytail: a momentary process-wide swap for one factory call; single-threaded harness
        for (m, n), r in swaps.items():
            setattr(m, n, r)
        return factory()
    finally:
        for (m, n), v in saved.items():
            setattr(m, n, v)


class Executor(InprocExecutor):
    def __init__(self):
        self._env = self._stage = None

    def handshake(self):
        return normalize_handshake("inproc", REF, {{"patched": PATCHED}})

    def bind(self, env, target=None):
        self._env, self._stage = env, make_stage()

    def act(self, obs):
        return self._stage.act(self._env, obs)

    def done(self):
        return bool(self._stage.done(self._env)) or getattr(self._stage, "failure_mode", None) is not None

    def diagnostics(self):
        d = getattr(self._stage, "diagnostics", None)
        return dict(d(self._env)) if d is not None else {{}}


class Policies:
    def make_driver(self, spec):
        return Executor()


def provider(**params):
    return Policies()
'''


def _other_sources(module: str, fd: dict):
    """(name, source) of every editable module that is NOT the one the answer named."""
    for m in fd.get("modules") or ():
        if m == module:
            continue
        try:
            yield m, Path(inspect.getsourcefile(importlib.import_module(m))).read_text()
        except Exception:   # noqa: BLE001 -- a module with no source is simply not the answer
            continue


def _elsewhere(edits, module: str, fd: dict) -> str:
    """Where an unfound snippet actually lives. The model copies the right code out of
    ``functions`` and names the wrong module (measured: live round 102 sent
    ``PointPlaceDriver._act`` verbatim against recycle_driver, it lives in stage_extras).
    Exact-once first (zero false positives by construction); failing that, ``_near`` with
    ``run=3`` -- three CONSECUTIVE lines in common, and the wording drops to "MAY belong
    there". Replaying this function over the 136 "occurs nowhere" refusals of
    evolve-recycle_cans (112 rounds; their ``old`` survives verbatim in the audits' refusal
    text) against the four editable modules, ``module`` = recycle_driver as in 306 of the
    363 patch rounds: exact-once names a module for 46, the three-line fallback adds 1, and
    89 stay silent. The any-ONE-line rule this replaced added 9 instead -- 8 of them on one
    shared line (``return None``, ``else:``, ``self._ticks = 0``), each a confident wrong
    module, which costs two refusals: its own and the ``stage_modules`` gate behind it."""
    snippets = [o for e in (edits or ()) if isinstance(o := (e or {}).get("old"), str)]
    for old in snippets:
        for m, src in _other_sources(module, fd):
            if src.count(old) == 1:
                return f"\nThat snippet occurs EXACTLY ONCE in {m}: send `module`: {m!r} instead."
    for old in snippets:
        for m, src in _other_sources(module, fd):
            if (near := _near(src, old, run=3)) is not _NOWHERE:
                return (f"\nThree consecutive lines of that snippet occur in {m}, so `old` MAY "
                        f"belong there and not in {module}. If this is the code you meant, send "
                        f"`module`: {m!r} and copy `old` out of THIS text (without the `NNNN| ` "
                        f"line-number prefixes).\n{near}")
    return ""


def _patch_key(new: str) -> str:
    """Identity of a patched module for the ALREADY-RAN gate: the sha of its AST dump, so
    comments, blank lines and reflow do not mint a new one (unparsable -> the raw text).

    What it catches: the same code re-worded. What it does not: the same IDEA rewritten --
    a renamed variable, a constant folded differently, two statements swapped all change the
    dump. Replaying the 363 patch rounds of evolve-recycle_cans: the full-text sha this
    replaced saw 319 distinct modules and repeated only 44 (13 behind the stage_modules
    gate); the AST key sees 215 and repeats 148 (51 behind that gate). Catching "the model
    re-derived the same clamp in different words" is not this gate's job -- nothing
    mechanical does it, and 148 of 363 is what a mechanical key buys."""
    try:
        return content_id(ast.dump(ast.parse(new)))[:16]
    except SyntaxError:
        return content_id(new)[:16]


def write_patch(pay: dict, fd: dict, round_no: int = 0, root: Path | None = None) -> str | None:
    """Materialise a ``patch`` answer: the module copied under ``root/<name>/`` with the
    ``edits`` (or ``diff``) applied (imports of the installed package rewritten by ref), the card's
    ``[tunables]`` copied when the module reads its own manifest, a manifest binding
    ``<to>`` and the generated ``PATCH_CARD``; then the card checks (``_doctor``). Fills
    ``pay['path'] / ['ref']``. Returns the refusal (``patch:...`` / ``doctor:...``) or None."""
    root = root or CANDIDATES_ROOT
    name, module, to = pay.get("name"), pay.get("module"), pay.get("to")
    edits, diff = pay.get("edits"), pay.get("diff")
    if not isinstance(name, str) or not _NAME.match(name):
        name = pay["name"] = f"patch_r{round_no}"   # bookkeeping ids, never worth an attempt:
    if not isinstance(to, str) or not to:           # the live model burned 2 of its 3 on these
        to = pay["to"] = name
    if module not in (fd.get("modules") or []):
        return (f"patch:module must be one of node {fd.get('node')!r}'s modules {fd.get('modules')}, "
                f"got {module!r} (that list is death_nodes[{fd.get('node')!r}].modules -- "
                "first_death.modules is the DEFAULT node's, and payload.node moved this round off it)")
    # ...and inside that list, only a module this stage's MRO actually contains can change
    # anything: make_stage() swaps the classes whose ``__module__`` is the patched one, so a
    # patch to a module the stage never inherits from installs a card that runs the STOCK
    # code. Replayed over evolve-recycle_cans' 363 patch rounds: 171 (nav-can1 x
    # recycle_driver 153, x planner 11, drop-can1 x planner 7) were mechanically no-ops that
    # still passed the doctor and burned a whole suite (12,569 s of simulator, 48.8% of
    # the campaign's 25,777 s) -- a stuck round widens ``modules`` to the pipeline, and
    # the model spent the widening on the OTHER node's driver.
    if (sm := fd.get("stage_modules")) and module not in sm:
        return (f"patch:{module} cannot change node {fd.get('node')!r}: the card only swaps the classes "
                f"this stage inherits from, and those come from {', '.join(sm)}. A patch to {module} "
                f"would install a candidate that runs the stock code. Send `module`: one of {sm}, or "
                "put the node whose stage lives in that module in `payload.node`.")
    if edits is not None and not (isinstance(edits, list) and edits):
        return "patch:`edits` must be a non-empty list of {old, new} objects"
    if edits is None and not (isinstance(diff, str) and diff.strip()):
        return ("patch:payload needs `edits`: [{old, new}] -- each `old` copied verbatim out of "
                "module_sources[module] (a unified `diff` is still accepted instead)")
    mod = importlib.import_module(module)
    src = Path(inspect.getsourcefile(mod)).read_text()
    modes: list = []
    try:
        new = apply_edits(src, edits, modes) if edits is not None else apply_diff(src, diff)
    except ValueError as exc:
        return f"patch:{exc}{_elsewhere(edits, module, fd)}"
    if new == src:
        return "patch:the patch changes nothing"
    pkg, base = _card_package(root), module.rpartition(".")[2]
    ref = f"{pkg}{name}:provider"
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{base}.py").write_text(_by_ref(new))
    tun = ""
    if getattr(mod, "_MANIFEST", None) is not None:   # the copy reads [tunables] off ITS manifest
        tun = "\n[tunables]\n" + "".join(f"{k} = {v!r}\n" for k, v in tomllib.loads(
            Path(mod._MANIFEST).read_text()).get("tunables", {}).items() if isinstance(v, (int, float)))
    # A patched copy carries its ORIGIN card: the copy legitimately imports that
    # package (it IS a copy of it) and needs the same third_party, so the doctor's
    # boundary check must judge it against the origin, not as a standalone card.
    origin = module.split(".")[1] if module.startswith("plugins.") else ""
    om = Path("plugins") / origin / "manifest.toml"
    tp = list(tomllib.loads(om.read_text()).get("third_party", ())) if om.exists() else []
    # A copy may import whatever the ORIGINAL imported: fold the module's own
    # import roots in, so the doctor's boundary check judges the patch, not the
    # dependencies it inherited verbatim.
    tp += [r for r in sorted({m.split(".")[0] for _, m in _module_roots(new)}) if r not in tp]
    (d / "manifest.toml").write_text(
        f'needs_sim = true\npatched_from = "{origin}"\n'
        + (f"third_party = {list(tp)!r}\n" if tp else "")
        + f'[executors.{to}]\nskill = "{fd.get("skill")}"\nembodiment = "{fd.get("embodiment")}"\n'
        f'ref = "{ref}"\ntransport = "inproc"\n{tun}')
    (d / "__init__.py").write_text(PATCH_CARD.format(
        name=name, module=module, round=round_no, to=to, skill=fd.get("skill"), ref=ref,
        installed=str(fd.get("tunables", {}).get("ref", "")).partition(":")[0], base=base, task=fd.get("task")))
    pay["path"], pay["ref"], pay["match"] = str(d), ref, modes or ["diff"]
    pay["patch_sha"] = _patch_key(new)   # the resulting MODULE, keyed past comments/whitespace
    return _doctor(d, ref, pay)


def _module_roots(text: str):
    """(lineno, dotted module) for every import in a source text; unparsable -> none."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node.lineno, a.name
        elif isinstance(node, ast.ImportFrom) and not node.level:
            yield node.lineno, node.module or ""


def dry_run(ref: str, params: dict, transport: str = "inproc") -> str | None:
    """Mount the provider by ref, build one executor (``make_driver(None)``) and check
    the contract the stage driver relies on: a Step/SegmentExecutor whose ``handshake()``
    is a dict naming ``transport``. Returns the refusal (``doctor:...``) or None."""
    fac = load_provider(ref, params)
    try:
        ex = fac.make_driver(None)
    except Exception as exc:  # noqa: BLE001
        return f"doctor:{ref} make_driver(None) raised {type(exc).__name__}: {exc}"
    if not isinstance(ex, (StepExecutor, SegmentExecutor)):
        return (f"doctor:{ref} make_driver() returned {type(ex).__name__}, not a StepExecutor "
                "(handshake/reset/act/done/diagnostics) -- subclass harness.skill_executor.InprocExecutor")
    hs = ex.handshake()
    if not isinstance(hs, dict) or hs.get("transport") != transport:
        return f"doctor:{ref} handshake() must be normalize_handshake({transport!r}, REF, meta), got {hs!r}"
    return None


def _image_parts(proj: dict, session: Path | None) -> tuple[list[dict], list[dict]]:
    """The per-seed first-death keyframes as OpenAI content parts (a label text part
    then the image as a base64 data URL read from ``session/<path>``), and the same
    list for the audit with each image replaced by its path. ([], []) without a session."""
    parts, audit = [], []
    for s in proj.get("this_round", {}).get("per_seed") or []:
        for i, rel in enumerate(s.get("keyframes") or []):
            try:
                b = (session / rel).read_bytes() if session else None
            except OSError:
                b = None
            if b:
                label = {"type": "text", "text": f"seed {s['seed']} keyframe {i}: {rel}"}
                parts += [label, {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}]
                audit += [label, {"type": "image", "path": rel}]
    return parts, audit


_FRAME = re.compile(r'File "([^"]+)", line (\d+)')
#: scripts.evolve.self_check's finding: "<file>.py: <Class> reads self.<attr>, which ..."
_FINDING = re.compile(r"([\w.]+\.py): \w+ reads (self\.\w+)")


def _own_code(path, why: str, span: int = 8, most: int = 3) -> str:
    """The offending lines of the model's OWN candidate code, numbered: every traceback frame
    under ``path``, plus the lines a static self-check finding points at. Without it the model
    re-reads the INSTALLED source and rewrites its patch from scratch (measured: rounds 104
    and 108) instead of fixing the code it just wrote."""
    where: dict[Path, set] = {}
    for f, ln in _FRAME.findall(why or ""):
        where.setdefault(Path(f), set()).add(int(ln))
    for name, expr in _FINDING.findall(why or ""):
        q = Path(path or ".") / name
        if q.is_file():
            where.setdefault(q, set()).update(
                i for i, line in enumerate(q.read_text().split("\n"), 1) if expr in line)
    out = []
    for q, lines in sorted(where.items()):
        if not path or not str(q).startswith(str(path)) or not q.is_file():
            continue
        text = q.read_text().split("\n")
        out += [f"\n\n{q.name} -- YOUR OWN code, around line {ln} (fix THIS, not the installed source "
                f"you copied it from):\n" + _numbered(text, ln - 1 - span, ln + span) for ln in sorted(lines)]
    return "".join(out[:most])


def _prior_rejects(audit_dir: Path, keep: int = 5) -> dict:
    """The last ``keep`` rejected payload hashes of EARLIER rounds (read back off their audit
    files) -> why they were rejected, with the round. Live rounds 104 and 108 sent the same
    uninitialised-state patch: within a round it was already refused, across rounds it was not."""
    out = {}
    for f in sorted(audit_dir.glob("round-*.json"), key=lambda q: -int(re.sub(r"\D", "", q.name) or 0)):
        try:
            a = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for at in reversed(a.get("attempts") or []):
            if (dig := at.get("sha")) and dig not in out:
                out[dig] = f"第 {a.get('round')} 轮已经提过同一条回答并被拒：{str(at.get('reason') or '')[:300]}"
                if len(out) >= keep:
                    return out
    return out


REPAIR = ("Your proposal was rejected:\n{why}\n\nFix exactly that and output ONLY the corrected "
          "proposal JSON object (same output_schema, same decision unless the error says otherwise).")
#: A payload byte-identical to one already rejected this round: cheaper to say so than to
#: re-run the whole rejection, and the second one ends the round honestly.
REPEAT = ("你重复了一条已经被拒的回答（{why}）。必须换一个做法：改别的地方，或改用 {kinds}。\n"
          "只输出修改后的 proposal JSON 对象。")
REPEATED = "llm: repeated the same rejected answer"
ASK = ("You decided {kind}: {rationale}\n\nThe code material is above (materials). Output ONLY the full "
       "proposal JSON object now: decision {kind} with its complete payload (payload_by_kind.{kind}).")
_NEED = {"card": ("name", "files"), "patch": ("name", "module")}


def _needs_material(ans: dict) -> bool:
    """A card / patch decision whose payload is not there yet (call 2 supplies the material).
    A ``patch`` ALWAYS is, payload or not: call 1's brief carries not one line of source
    (``module_sources`` / ``functions`` are MATERIAL_KEYS), so any ``old`` written there is
    retyped from memory. Measured on evolve-recycle_cans: 108 of the 160 "old occurs 0
    times" refusals happened on attempt 0, and those 108 rounds cost 172,086 tokens each
    against 67,805 for a clean one -- 18.6M of the campaign's 59.4M, 48 ending in none.
    The call site's ``not step2`` guard still inserts the material at most once a round."""
    pay = ans["payload"]
    if ans["kind"] == "patch":
        return True
    return ans["kind"] in _NEED and not all(k in pay for k in _NEED[ans["kind"]])


_DIR = {True: "up", False: "down"}


def _tried_pairs(proj: dict, fd: dict | None = None) -> set:
    """What this campaign already tried on ``fd``'s node (default: the first-death node),
    from the round rows: ``("tunables", knob, went up?)`` (``detail.path/from/to`` on the same
    driver ref) and ``("executor", key, None)`` (an executor / card switch). The model may not
    repeat one. Node-scoped because ``_try`` answers on ``payload.node``, which is the OTHER
    death node half the rounds: the rejection used to list the head node's knobs under the
    other node's name."""
    fd = proj.get("first_death") or {} if fd is None else fd
    ref, node, out = (fd.get("tunables") or {}).get("ref"), fd.get("node"), set()
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    for r in proj.get("history") or ():
        t = r.get("tried") or {}
        d = t.get("detail") or {}
        if t.get("kind") == "tunables" and d.get("ref") == ref and d.get("path") \
                and num(d.get("from")) and num(d.get("to")):
            out.add(("tunables", d["path"][-1], d["to"] > d["from"]))
        elif t.get("kind") in ("executor", "card") and t.get("node") == node and d.get("to"):
            out.add(("executor", d["to"], None))
    return out


def _edit_key(module, edits) -> tuple:
    return (module, tuple(sorted((str((e or {}).get("old", "")).strip(),
                                  str((e or {}).get("new", "")).strip()) for e in edits)))


def _ran_patches(proj: dict) -> dict:
    """``patch_sha`` (``_patch_key``) -> the verdict of the round that ALREADY RAN that module.

    A patch that applies cleanly, runs, and simply fails to improve is in no reject list --
    nothing rejected it -- so the model, which never sees its own past code, re-derives it
    from the same evidence next round. Replayed over evolve-recycle_cans' 363 patch rounds:
    148 repeat a module already run (51 of them behind the ``stage_modules`` gate, which
    refuses 171 first); none of the 8 ``improved`` rounds is among them. The campaign's
    "same drop-point clamp for ~300 rounds" is real as an IDEA and not as code -- the wording
    drifts every round, and no sha reaches it."""
    out: dict = {}
    for r in proj.get("history") or ():
        sha = ((r.get("tried") or {}).get("detail") or {}).get("patch_sha")
        if sha:
            out.setdefault(str(sha), f"round {r.get('round')}: "
                           + str(r.get("verdict") or "it did not raise the score")[:200])
    return out


def _tried_values(proj: dict) -> set:
    """Every ``(knob, exact value)`` this campaign already SET, off the round rows.

    The wall's own knobs stay answerable while it stands -- a direction is not an answer
    there, the brief hands the model a NUMBER (carry_stop 0.188, nudge_max 0.364). But
    answering the SAME number again buys nothing and costs a whole suite (~44 s), and the
    wall stands in 505 of the campaign's 588 rounds: without this, one accepted-looking
    knob answer can be repeated verbatim until the operator stops it."""
    out = set()
    for r in proj.get("history") or ():
        t = r.get("tried") or {}
        d = t.get("detail") or {}
        if t.get("kind") == "tunables" and d.get("path") \
                and isinstance(d.get("to"), (int, float)) and not isinstance(d.get("to"), bool):
            out.add((str(d["path"][-1]), round(float(d["to"]), 6)))
    return out


def _accepted_repeat(proj: dict, pay: dict) -> str | None:
    """An accepted change IS this round's baseline (the run starts from it), so re-proposing
    it changes nothing. Refuses a patch whose (module, edits) is already in the stack."""
    if not isinstance(pay.get("edits"), list) or not pay["edits"]:
        return None
    key = _edit_key(pay.get("module"), pay["edits"])
    for c in (proj.get("accepted_stack") or {}).get("changes") or ():
        d = c.get("detail") if isinstance(c, dict) and isinstance(c.get("detail"), dict) else c
        if isinstance(d, dict) and isinstance(d.get("edits"), list) \
                and _edit_key(d.get("module"), d["edits"]) == key:
            return (f"this edit is ALREADY ACCEPTED (round {c.get('round')}): the accepted stack is "
                    "this round's baseline, the run already starts from it. Build the NEXT change "
                    "on top of it, or target another node of death_nodes.")
    return None


def _repeat_why(key: tuple, fd: dict, seen: set) -> str:
    """The rejection text for a (knob, direction) / executor already in ``seen``: what the
    campaign tried and what is still untried, so the answer has somewhere to go."""
    show = lambda k: f"{k[1]} {_DIR[k[2]]}" if k[0] == "tunables" else str(k[1])
    tun = fd.get("tunables") or {}
    if key[0] == "tunables":
        left = [f"{k} {d}" for k in sorted(tun.get("values") or {}) for d in ("down", "up")
                if ("tunables", k, d == "up") not in seen]
        hints = {m: ks for m, ks in (tun.get("hints") or {}).items() if ks}
    else:
        left = sorted(set(fd.get("executors") or {}) - {fd.get("executor")}
                      - {k[1] for k in seen if k[0] == "executor"})
        hints = {}
    return (f"{show(key)} was already tried in this campaign (tried: "
            f"{', '.join(sorted(show(k) for k in seen if k[0] == key[0]))}). Answer one still "
            f"untried instead: {', '.join(left) or 'nothing left -- decide none'}"
            + (f" (hinted for the failure mode: {json.dumps(hints, sort_keys=True)})" if hints else "") + ".")


def _untried(proj: dict, seen: set, fd: dict | None = None) -> list[str]:
    """What is still open on ``fd``'s node (default: the first-death node): every (knob,
    direction) not in ``seen``, every bound executor not tried, and a patch per module while
    no patch was tried. The model may answer ``none`` only when this is empty."""
    fd = proj.get("first_death") or {} if fd is None else fd
    tun = fd.get("tunables") or {}
    left = [f"tunables {k} {d}" for k in sorted(tun.get("values") or {}) for d in ("down", "up")
            if ("tunables", k, d == "up") not in seen]
    left += [f"executor {k}" for k in sorted(set(fd.get("executors") or {}) - {fd.get("executor")}
                                             - {k[1] for k in seen if k[0] == "executor"})]
    if not any(((r.get("tried") or {}).get("detail") or {}).get("module") for r in proj.get("history") or ()):
        left += [f"patch {m}" for m in fd.get("modules") or ()]
    return left


LOWER = ("the parameter layer is CLOSED on {node}: every (knob, direction) is already tried "
         "({tried}). Diagnose top-down and answer from a HIGHER layer -- {higher} -- "
         "{questions} Re-read the trail, the first missing milestone and its divergence, then "
         "change CODE (patch / card) or target another node of death_nodes.")


def _stamp(tried: dict, ans: dict, layer) -> dict:
    """The round's diagnosis on the try: the layer it claims (``detail.layer``, the round row
    and rsi_step carry it) and its notebook note (``detail.notes``)."""
    if layer:
        tried["detail"]["layer"] = layer
    if isinstance(ans.get("notes"), str) and ans["notes"].strip():
        tried["detail"]["notes"] = ans["notes"].strip()[:600]
    return tried


def _try(ans: dict, proj: dict, before: dict, round_no: int, preflight, seen: set | None = None,
         last: bool = False) -> dict:
    """One parsed answer -> this round's ``tried``; raises ValueError with the exact
    rejection text (validation / a (knob, direction) or executor ``seen`` already /
    doctor / dry instantiation / preflight seed / a ``none`` while ``_untried`` is not empty,
    unless ``last``: the final attempt takes the none honestly). ``seen`` is what THIS round
    already answered (it grows with every answer, so a round cannot repeat itself either);
    the campaign's own history is merged in per node, below."""
    from scripts.evolve import _none, from_proposal   # noqa: PLC0415 -- evolve imports this module
    seen = set() if seen is None else seen
    pay = ans["payload"]
    pay.setdefault("node", (proj.get("first_death") or {}).get("node"))   # the round's target; the model may override
    p = {"id": f"llm:round-{round_no}", "kind": ans["kind"], "payload": pay, "note": ans["rationale"]}
    # ...and once it overrides, EVERYTHING follows payload.node: the judgement already did
    # (evolve focuses the trial on that node's cluster), so the materialisation must too --
    # write_patch stamps fd's skill/task into the manifest and PATCH_CARD, and apply() mounts
    # the result under tried["node"]. Reading the rotation head here is what made 184 of the
    # 363 live patch rounds unexecutable: the drop-can1 patch was installed as nav-can1's
    # executor and the drop segment ran the stock card.
    if pay.get("node") and (proj.get("drivers") or {}) and pay["node"] not in proj["drivers"]:
        # A node with no projection this round has no modules, no stage MRO and no skill of
        # its own, so write_patch would stamp the ROTATION HEAD's identity into the manifest
        # and PATCH_CARD while apply() mounts the card under this node -- "edits A, judged B"
        # coming back through a second door, and target.judged now invites the model to write
        # this field every round. Refuse it by name instead.
        raise ValueError(f"payload.node must be one of death_nodes {sorted(proj['drivers'])}, "
                         f"got {pay['node']!r}. That node is not where any seed died this "
                         "round, so there is nothing to diagnose or patch on it.")
    fd = (proj.get("drivers") or {}).get(pay["node"]) or proj["first_death"]
    # ...including what the campaign already TRIED and what is still left: those are per node
    # too. Reading the head's here made the refusals lie -- "these are still untried on
    # drop-can1" listing nav-can1's knobs -- on the 207 of 252 nav-head rounds the model
    # diagnosed drop (measured on evolve-recycle_cans' 588 rounds).
    pairs = _tried_pairs(proj, fd) | seen
    left = _untried(proj, pairs, fd)
    # top-down: a tunables decision IS a parameter-layer answer, whatever it labels itself
    layer = ans.get("layer") or ("parameter" if ans["kind"] == "tunables" else None)
    # ...unless the round is standing at the reach wall: the only thing that closes a
    # 0.364 m geometric gap IS a parameter (the upstream standoff, nudge_max), and refusing
    # the layer while the brief's reach_wall points straight at it is one prompt telling the
    # model two opposite things. r55 onward the campaign never had a way through.
    if layer == "parameter" and _exhausted(proj, left, fd) and not _reach_wall(proj):
        raise ValueError(LOWER.format(
            node=fd.get("node"), higher=", ".join(LAYERS[:-1]),
            tried=", ".join(sorted(f"{k[1]} {_DIR[k[2]]}" for k in pairs if k[0] == "tunables")) or "-",
            questions=" ".join(f"{k}: {LAYER_QUESTION[k]}" for k in LAYERS[:-1])))
    if ans["kind"] == "none":
        if left and not last:   # none is for a round with nothing left
            raise ValueError("none is only allowed when nothing is left to try, and these are still "
                             f"untried on {fd.get('node')}: {', '.join(left)}. Answer one of them, or "
                             "say in rationale why each of them cannot help.")
        return _stamp(_none(f"llm: {ans['rationale'] or 'nothing to try'}", fd.get("node")), ans, layer)
    if ans["kind"] == "tunables":
        if pay.get("ref") != fd.get("tunables", {}).get("ref") or not isinstance(pay.get("to"), (int, float)) \
                or not (isinstance(pay.get("path"), list) and all(isinstance(x, str) for x in pay["path"])):
            raise ValueError(f"tunables payload must be {{ref: {fd.get('tunables', {}).get('ref')!r}, "
                             f"path: [str], to: number}}, got {pay}")
        cur = (fd.get("tunables", {}).get("values") or {}).get(pay["path"][-1] if pay["path"] else None)
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            key = ("tunables", pay["path"][-1], pay["to"] > cur)
            # the wall's OWN knobs stay answerable while it stands -- reopening the layer
            # (above) and then refusing the two knobs the brief points at is the same prompt
            # saying two opposite things, one gate later. ``seen`` (this round's own answers)
            # still refuses a repeat inside the round, and every other knob stays closed.
            val = (str(pay["path"][-1]), round(float(pay["to"]), 6))
            if val in _tried_values(proj):   # the exact number, whatever the direction says
                raise ValueError(
                    f"this campaign already set {val[0]} to {val[1]}; re-running the same "
                    "number cannot produce a different result. Answer a DIFFERENT number "
                    "(the brief's wall gives you the one the geometry asks for), or another "
                    "layer.")
            if key in seen or (key in pairs and not (key[1] in WALL_KNOBS and _reach_wall(proj))):
                raise ValueError(_repeat_why(key, fd, pairs))
            seen.add(key)
        tried = from_proposal(p, before)
    elif ans["kind"] == "executor":
        if pay.get("to") not in fd.get("executors", {}) or pay.get("to") == fd.get("executor"):
            raise ValueError(f"executor.to must be another key of {sorted(fd.get('executors', {}))}, got {pay.get('to')!r}")
        if ("executor", pay["to"], None) in pairs:
            raise ValueError(_repeat_why(("executor", pay["to"], None), fd, pairs))
        seen.add(("executor", pay["to"], None))
        tried = from_proposal(p, before)
    else:   # card / patch: materialised under the candidates root, then the same card path
        if why := (_accepted_repeat(proj, pay) if ans["kind"] == "patch" else None):
            raise ValueError(why)
        if why := (write_card(pay) if ans["kind"] == "card" else write_patch(pay, fd, round_no)):
            raise ValueError(why)
        if ans["kind"] == "patch" and (ran := _ran_patches(proj).get(str(pay.get("patch_sha")))):
            raise ValueError(
                f"patch:this exact patched module ALREADY RAN -- {ran}. Re-running it cannot "
                "produce a different result. Either change the SAME node a different way (a "
                "different phase, a different lever), or target another node of death_nodes.")
        tried = from_proposal({**p, "kind": "card", "payload": {k: pay[k] for k in ("path", "to", "ref", "params", "node") if k in pay}}, before)
        if ans["kind"] == "patch" and tried["kind"] == "card":
            # patch_sha too: the ALREADY-RAN gate reads it off the HISTORY rows, so a stamp
            # that stops here is a gate that never fires (it did not, for a whole campaign)
            tried["detail"].update(module=pay["module"],
                                   **{k: pay[k] for k in ("edits", "diff", "match", "patch_sha") if k in pay})
    if tried["kind"] == "none":
        raise ValueError(tried["detail"]["reason"])   # from_proposal's refusal: the answer was unusable
    if ans["kind"] in ("card", "patch") and preflight is not None:
        try:
            preflight(tried)
        except Exception as exc:  # noqa: BLE001 -- the executor's own failure, traceback and all
            # a SelfCheckError is a STATIC finding (scripts.evolve.self_check), not a crash:
            # it reads as itself, with the model's own offending lines under it
            why = (str(exc) if type(exc).__name__ == "SelfCheckError" else
                   f"preflight: the trial raised on seed {before and min(before['seeds'])}:\n"
                   + traceback.format_exc()[-3000:])
            if " reads self." in why:   # name the runtime error the static finding predicts
                why += (" -- reading it raises AttributeError the moment the trial runs (rounds 104 and "
                        "108 died exactly there); found statically, no simulator seed spent")
            raise ValueError(why + _own_code(pay.get("path"), why)) from exc
    return _stamp(tried, ans, layer)


def llm_propose(ep, proj: dict, before: dict, round_no: int, audit_dir: Path,
                max_tokens: int = 8192, session: Path | None = None, preflight=None) -> tuple[dict | None, dict]:
    """Up to ``MAX_ATTEMPTS`` model calls -> (tried | None, llm row). A rejected answer's
    exact error goes back as the next user message; after the last attempt ``tried`` is an
    honest none carrying that reason (``needs`` lists it). ``tried`` is None only when the
    endpoint itself failed (the row's ``reason`` says why; the caller falls back to the
    rules). An answer whose payload is byte-identical to one already rejected this round
    costs no attempt: it gets ``REPEAT`` back once, and a second identical answer ends the
    round with an honest none (``REPEATED``). ``max_tokens`` is the REPLY budget alone --
    4096 cut patch answers off mid-string (rounds 277/329/471 spent 3 x 4096 on nothing but
    "Unterminated string"); ``PROMPT_CHARS`` / ``BRIEF_CHARS`` bound the other direction,
    what we SEND, and do not move with it. ``preflight(tried)`` (a card's one-seed trial) may raise to reject. When the
    endpoint accepts images (``ep.images``) the seeds' failure keyframes ride along as
    image parts; the audit / prompt_sha keep their paths only, never the bytes."""
    from scripts.evolve import _none   # noqa: PLC0415 -- evolve imports this module
    b, materials = brief(proj), {k: proj[k] for k in MATERIAL_KEYS if k in proj}
    seen: set = set()   # this round's answers; _try merges the campaign's history per node
    # what the model's OWN last change did in the simulator comes FIRST, before the round input
    text = (((b["trial_evidence"] + "\n\n") if b.get("trial_evidence") else "")
            + "Round input:\n" + json.dumps(b, sort_keys=True, default=str)
            + "\n\nOutput ONLY the proposal JSON object now.")
    images, audit_images = _image_parts(proj, session) if getattr(ep, "images", False) else ([], [])
    messages = [{"role": "system", "content": _RULES},
                {"role": "user", "content": [{"type": "text", "text": text}, *images] if images else text}]
    audit_msgs = ([messages[0], {"role": "user", "content": [{"type": "text", "text": text}, *audit_images]}]
                  if images else list(messages))
    row = {"model": getattr(ep, "identity", repr(ep)), "prompt_sha": content_id(audit_msgs),
           "raw_sha": None, "summary": None, "rationale": None, "reason": None, "usage": None}
    audit = {"round": round_no, **row, "messages": audit_msgs, "brief": b, "materials": materials,
             "calls": 0, "raw": None, "attempts": [], "repeats": []}
    # the static code material (call 2): inserted after the system message -- once, whether
    # the model asked for it by answering without a payload or wrote one blind and was rejected
    mat = {"role": "user", "content": "Materials (static):\n" + json.dumps(materials, sort_keys=True, default=str)}
    tried, why, path, step2 = None, None, None, False
    # payload sha -> its rejection; a byte-identical answer is not a new attempt, and the
    # last 5 of EARLIER rounds count too (the same broken patch came back 4 rounds later)
    rejected = _prior_rejects(audit_dir)
    while len(audit["attempts"]) < MAX_ATTEMPTS:
        try:
            # ponytail: DeepSeek reasoning tokens count against max_tokens and left content
            # empty (3/3 production rounds); the proposer wants the answer, not the preamble
            raw = ep.chat(messages, temperature=0.0, max_tokens=max_tokens,
                          response_format={"type": "json_object"}, thinking={"type": "disabled"})
        except Exception as exc:  # noqa: BLE001 -- unreachable endpoint: rules take over
            row["reason"] = f"{type(exc).__name__}: {exc}"[:300]
            break
        audit["calls"] += 1
        usage = getattr(ep, "last_usage", None)
        if usage:
            row["usage"] = {k: (row["usage"] or {}).get(k, 0) + (usage.get(k) or 0) for k in ("prompt", "completion")}
        audit["raw"] = raw
        row["raw_sha"] = audit["raw_sha"] = sha_json(raw)
        ans, dig = None, None
        try:
            ans = _parse(raw)
            row["summary"], row["rationale"] = ans["summary"], ans["rationale"]
            if not step2 and _needs_material(ans):   # call 2: the static material FIRST (prefix
                step2 = True                           # cache), the brief last, then the ask
                ask = {"role": "user", "content": ASK.format(kind=ans["kind"], rationale=ans["rationale"][:1000])}
                for msgs in (messages, audit_msgs):
                    msgs.insert(1, mat)
                    msgs += [{"role": "assistant", "content": raw}, ask]
                continue
            # ``none`` carries no payload, so it can only look identical: its own ladder
            # (_untried, then the honest take on the last attempt) already bounds it.
            dig = sha_json([ans["kind"], ans["payload"]]) if ans["kind"] != "none" else None
            if dig in rejected:   # the same payload, byte for byte
                audit["repeats"].append({"raw": raw, "reason": rejected[dig]})
                if len(audit["repeats"]) > 1:   # twice: stop paying for it, end the round honestly
                    tried = _none(REPEATED, proj["first_death"].get("node"), needs=("proposal", REPEATED))
                    if path:
                        tried["detail"]["path"] = path
                    why = None
                    break
                nag = {"role": "user", "content": REPEAT.format(
                    why=rejected[dig].strip().splitlines()[0][:300],
                    kinds="/".join(k for k in KINDS if k != ans["kind"]))}
                messages += [{"role": "assistant", "content": raw}, nag]
                audit_msgs += [{"role": "assistant", "content": raw}, nag]
                continue
            tried = _try(ans, proj, before, round_no, preflight, seen,
                         last=len(audit["attempts"]) == MAX_ATTEMPTS - 1)
            why = None
            break
        except Exception as exc:  # noqa: BLE001 -- bad JSON / payload / doctor / preflight: the model repairs
            why = str(exc) if str(exc).startswith(("doctor:", "preflight:", "patch:")) else f"{type(exc).__name__}: {exc}"
            # 24 rounds died on a JSON error the model could not act on (rounds 277/329/471
            # each burned 3 x 4096 completion tokens): the answer ran out of room mid-string
            # and all it got back was "Unterminated string". Name the real cause first.
            # ``finish_reason == "length"`` is the endpoint SAYING it truncated; the token
            # count behind it is only an approximation (a reply that legitimately ends on
            # the last allowed token reads the same, and an endpoint that reports no usage
            # reads as "not truncated") -- kept for endpoints that send no finish_reason.
            if ans is None and (getattr(ep, "last_finish", None) == "length"
                                or ((usage or {}).get("completion") or 0) >= max_tokens):
                why = (f"你的回答在 max_tokens={max_tokens} 处被截断了，不是 JSON 写错了：把答案写"
                       "短一点（rationale 一两句就够，别把源码抄回来），再输出完整的 JSON。\n" + why)
            path = (ans or {}).get("payload", {}).get("path") or path   # the files stay for the operator
            if dig is not None:   # hashed BEFORE write_card/write_patch grew the payload
                rejected[dig] = why
            audit["attempts"].append({"raw": raw, "usage": usage, "reason": why, "sha": dig})
            repair = {"role": "user", "content": REPAIR.format(why=why[:4000])}
            if not step2 and (ans or {}).get("kind") in _NEED:
                step2 = True   # it wrote the payload blind (an invented diff/snippet): the
                for msgs in (messages, audit_msgs):   # real source it must copy from, now
                    msgs.insert(1, mat)
            messages += [{"role": "assistant", "content": raw}, repair]
            audit_msgs += [{"role": "assistant", "content": raw}, repair]
    if tried is None and why is not None:   # every attempt rejected: an honest none, not a rules try
        lines = why.strip().splitlines()   # a traceback: its first and last line say it
        tried = _none(f"llm: {len(audit['attempts'])} answers rejected; last: {why}"[:1500],
                      proj["first_death"].get("node"),
                      needs=("proposal", (lines[0] + " … " + lines[-1] if len(lines) > 1 else why)[:300]))
        if path:
            tried["detail"]["path"] = path
    audit.update(raw_sha=row["raw_sha"], summary=row["summary"], rationale=row["rationale"],
                 reason=row["reason"], usage=row["usage"], tried=tried)
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"round-{round_no}.json").write_text(json.dumps(audit, indent=1, sort_keys=True, default=str))
    return tried, row
