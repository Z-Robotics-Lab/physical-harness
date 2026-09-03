"""The LLM proposer of the lightweight evolve loop (scripts/evolve.py).

Each round the model reads a compact brief of the campaign -- ``rsi_projection``:
task, seeds, round history (tried + before->after + per-seed first death), this
round's per-seed node trails, the knobs of the first-death node's driver with their
current values and the card's failure-mode hints, the executors bound on that
skill with their evidence, the inbox proposals already consumed, ``needs`` and a
bounded excerpt of the dying node's log rows -- and answers ONE proposal in the
proposals-inbox shape (``PROPOSAL_SCHEMA``): tunables / executor / card
(code-as-policy: the model writes a candidate card under ``plugins/candidates/<name>/``,
checked by scripts/plugin_doctor, dry-instantiated (``dry_run``) and preflighted on
ONE seed before it is mounted) / none. The answer is validated strictly (schema +
``scripts.evolve.from_proposal``); a rejected answer (bad JSON, bad payload, doctor
red, a contract miss at instantiation, an exception inside the executor on the
preflight seed) goes back to the model VERBATIM as a follow-up message, up to
``MAX_ATTEMPTS`` per round (``attempts`` in the audit file); after that the round is
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

import base64
import importlib
import inspect
import json
import os
import re
import sys
import traceback
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
#: Prompt bound: ~12k DeepSeek tokens (measured ~2.5 chars/token on this JSON); over
#: it the log excerpt goes first, then older rounds' per-seed detail, then the driver source.
PROMPT_CHARS = 30_000
#: Answers per round: the first plus repairs of a rejected one.
MAX_ATTEMPTS = 3
_NAME = re.compile(r"^[a-z][a-z0-9_]{2,40}$")

#: The exact reply shape (the proposals inbox shape + summary/rationale); ``payload``
#: is the FLAT object of ``PAYLOAD_BY_KIND[kind]``.
PROPOSAL_SCHEMA = {
    "kind": "tunables | executor | card | none",
    "payload": "<the flat object described by payload_by_kind[kind], e.g. {\"to\": \"alt\"}>",
    "summary": "<1-3 sentences: what you saw this round, in Chinese>",
    "rationale": "<why this try>",
}
PAYLOAD_BY_KIND = {
    "tunables": {"ref": "<tunables.ref>", "path": ["<tunables.path prefix>...", "<knob>"],
                 "to": "<number>", "node": "<optional node id>"},
    "executor": {"to": "<a key of executors>", "node": "<optional node id>"},
    "card": {"name": "<[a-z][a-z0-9_]{2,40}: the candidate dir name>",
             "files": {"manifest.toml": "<toml>", "__init__.py": "<python>"},
             "to": "<new executor key>", "ref": "<name>:provider (or <card_package>:provider)",
             "node": "<optional node id>"},
    "none": {},
}

_RULES = """You are the proposer of a robot skill self-improvement loop. Each round the \
harness runs the task on fixed seeds, you read the round (per-seed node trails, where \
each seed died, its failure_mode, the log excerpt, what was tried before) and decide \
ONE change to try next; the harness re-runs the same seeds and keeps the change only if \
more seeds succeed. Reply with ONE JSON object and nothing else, exactly the shape of \
output_schema: exactly ONE of the payload shapes, matching kind.
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
- none: nothing worth trying (say why in rationale).
summary: 1-3 sentences in Chinese on what this round shows. rationale: why this try.
Each seed's keyframes are the failure keyframes of its first-death node (first frame, \
stall / last-progress frame, last frame; 128px); when attached as images they are labelled \
"seed <n> keyframe <i>" in the same order."""


def _log_excerpt(seed: int, rows, dead: str | None, budget: int) -> list[str]:
    """The dying node's ``task.fault`` / ``task.verify`` rows of one seed, ``budget`` lines."""
    out = []
    for r in rows:
        if r["kind"] == "task.fault" or (r["kind"] == "task.verify" and r["data"].get("node") == dead):
            out.append(f"seed {seed} {r['kind']} "
                       + json.dumps(r["data"], sort_keys=True, default=str)[:400])
    return out[-budget:]


def _stage_source(ref: str, task: str | None) -> str | None:
    """The scripted stage driver's source for one sub-goal task (its classes inside the
    embodiment package, base first), off the driver module's ``_STAGES`` table."""
    stages = getattr(importlib.import_module(ref.partition(":")[0]), "_STAGES", None)
    if not isinstance(stages, dict) or task not in stages:
        return None
    try:
        obj = stages[task][0]()
    except Exception:  # noqa: BLE001 -- a stage that needs the live env: no source, no crash
        return None
    top = type(obj).__module__.rsplit(".", 1)[0]
    return "\n".join(inspect.getsource(c) for c in reversed(type(obj).__mro__)
                     if c is not object and c.__module__.startswith(top))


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


def _driver(before: dict, records: dict, emb: str, arm: str, binding: dict) -> dict:
    """First-death node -> {node, skill, executor, rate, executors, tunables, embodiment,
    task}: what the rules proposer reads, projected for the model (same rearm /
    mount_params seams). ``embodiment`` is the key the skill's record binds under."""
    from scripts.evolve import _first_death   # noqa: PLC0415 -- evolve imports this module
    node = _first_death(before)
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
    return {"node": node, "skill": skill, "executor": current,
            "embodiment": emb if rec is None or emb in rec.bindings else next(iter(rec.bindings), emb),
            "task": (spec or {}).get("task"),
            "rate": sum(r["success"] for r in runs) / len(runs),
            "executors": {k: dict((ev.by_executor if ev else {}).get(k) or {}) for k in bound},
            "tunables": {"ref": ref, "path": ["tunables"] if isinstance(params.get("tunables"), dict) else [],
                         "values": values,
                         "hints": {m: [k for k in ks if k in values]
                                   for m, ks in (params.get("tunable_hints") or {}).items()}}}


def rsi_projection(doc: dict, before: dict, records: dict, emb: str, arm: str, binding: dict,
                   log_excerpt: list[str]) -> dict:
    """The compact brief the model reads for the round after ``doc['cursor']``."""
    rounds = doc.get("rounds") or []
    fd = _driver(before, records, emb, arm, binding)
    ref = fd.get("tunables", {}).get("ref") or binding["policy"]
    proj = {
        "task": doc["task"], "embodiment": emb, "seeds": doc["seeds"], "arm": arm,
        "round": int(doc.get("cursor") or 0) + 1,
        "applied": doc.get("applied"),
        "history": [{"round": r["round"], "proposer": r.get("proposer"),
                     "tried": {k: r["tried"][k] for k in ("kind", "node")}
                     | {"detail": {k: v for k, v in r["tried"]["detail"].items()
                                   if k in ("to", "from", "path", "ref", "reason", "error", "hint")}},
                     "before": r["before"], "after": r["after"], "published": r["published"],
                     "per_seed": [{k: s.get(k) for k in ("seed", "success", "first_death", "failure_mode")}
                                  for s in r.get("after_seeds") or r.get("per_seed") or []]}
                    for r in rounds],
        "this_round": {"count": before["count"], "seeds_total": len(before["seeds"]),
                       "per_seed": [{"seed": int(seed), **{k: s.get(k) for k in ("success", "first_death", "failure_mode", "fault", "keyframes")},
                                     "trail": [{k: n.get(k) for k in ("id", "ok", "steps", "failure_mode")}
                                               for n in s.get("trail") or []]}
                                    for seed, s in before["seeds"].items()]},
        "first_death": fd,
        "proposals_consumed": [{"round": r["round"], **r["proposal"]} for r in rounds if r.get("proposal")],
        "needs": rounds[-1].get("needs") if rounds else [],
        "log_excerpt": list(log_excerpt)[:MAX_LOG_LINES],
        **_contract(_card_package(CANDIDATES_ROOT), fd.get("skill"), fd.get("embodiment")),
        "reference_card": {f: (REFERENCE_CARD / f).read_text() for f in ("manifest.toml", "__init__.py")
                           if (REFERENCE_CARD / f).is_file()},
        "scripted_driver_source": _stage_source(ref, fd.get("task")),
        **_primitives(ref),
        "output_schema": PROPOSAL_SCHEMA, "payload_by_kind": PAYLOAD_BY_KIND,
    }
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
    if ans.get("kind") not in ("tunables", "executor", "card", "none"):
        raise ValueError(f"kind must be tunables|executor|card|none, got {ans.get('kind')!r}")
    if not isinstance(ans.get("payload"), dict) and ans["kind"] != "none":
        raise ValueError("payload must be an object")
    if not isinstance(ans.get("summary"), str) or not ans["summary"].strip():
        raise ValueError("summary must be a non-empty string")
    ans["payload"] = dict(ans.get("payload") or {})
    if set(ans["payload"]) == {ans["kind"]} and isinstance(ans["payload"][ans["kind"]], dict):
        ans["payload"] = dict(ans["payload"][ans["kind"]])   # {"payload": {"executor": {...}}}: seen live
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


def write_card(pay: dict, root: Path = CANDIDATES_ROOT) -> str | None:
    """Materialise a ``card`` answer under ``root/<name>/`` and doctor it; fills
    ``pay['path']``. Returns the refusal reason (``doctor:<first finding>``) or None."""
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
        return f"doctor:ref {ref!r} must name a provider inside {pkg + name}"
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for k, v in files.items():
        (d / k).write_text(v)
    pay["path"] = str(d)
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


REPAIR = ("Your proposal was rejected:\n{why}\n\nFix exactly that and output ONLY the corrected "
          "proposal JSON object (same output_schema, same kind unless the error says otherwise).")


def _try(ans: dict, proj: dict, before: dict, round_no: int, preflight) -> dict:
    """One parsed answer -> this round's ``tried``; raises ValueError with the exact
    rejection text (validation / doctor / dry instantiation / preflight seed)."""
    from scripts.evolve import _none, from_proposal   # noqa: PLC0415 -- evolve imports this module
    pay = ans["payload"]
    p = {"id": f"llm:round-{round_no}", "kind": ans["kind"], "payload": pay, "note": ans["rationale"]}
    fd = proj["first_death"]
    if ans["kind"] == "none":
        return _none(f"llm: {ans['rationale'] or 'nothing to try'}", fd.get("node"))
    if ans["kind"] == "tunables":
        if pay.get("ref") != fd.get("tunables", {}).get("ref") or not isinstance(pay.get("to"), (int, float)) \
                or not (isinstance(pay.get("path"), list) and all(isinstance(x, str) for x in pay["path"])):
            raise ValueError(f"tunables payload must be {{ref: {fd.get('tunables', {}).get('ref')!r}, "
                             f"path: [str], to: number}}, got {pay}")
        tried = from_proposal(p, before)
    elif ans["kind"] == "executor":
        if pay.get("to") not in fd.get("executors", {}) or pay.get("to") == fd.get("executor"):
            raise ValueError(f"executor.to must be another key of {sorted(fd.get('executors', {}))}, got {pay.get('to')!r}")
        tried = from_proposal(p, before)
    else:
        if why := write_card(pay):
            raise ValueError(why)
        tried = from_proposal({**p, "payload": {k: pay[k] for k in ("path", "to", "ref", "params", "node") if k in pay}}, before)
    if tried["kind"] == "none":
        raise ValueError(tried["detail"]["reason"])   # from_proposal's refusal: the answer was unusable
    if ans["kind"] == "card" and preflight is not None:
        try:
            preflight(tried)
        except Exception as exc:  # noqa: BLE001 -- the executor's own failure, traceback and all
            raise ValueError(f"preflight: the trial raised on seed {before and min(before['seeds'])}:\n"
                             + traceback.format_exc()[-3000:]) from exc
    return tried


def llm_propose(ep, proj: dict, before: dict, round_no: int, audit_dir: Path,
                max_tokens: int = 4096, session: Path | None = None, preflight=None) -> tuple[dict | None, dict]:
    """Up to ``MAX_ATTEMPTS`` model calls -> (tried | None, llm row). A rejected answer's
    exact error goes back as the next user message; after the last attempt ``tried`` is an
    honest none carrying that reason (``needs`` lists it). ``tried`` is None only when the
    endpoint itself failed (the row's ``reason`` says why; the caller falls back to the
    rules). ``preflight(tried)`` (a card's one-seed trial) may raise to reject. When the
    endpoint accepts images (``ep.images``) the seeds' failure keyframes ride along as
    image parts; the audit / prompt_sha keep their paths only, never the bytes."""
    from scripts.evolve import _none   # noqa: PLC0415 -- evolve imports this module
    text = ("Round input:\n" + json.dumps(proj, sort_keys=True, default=str)
            + "\n\nOutput ONLY the proposal JSON object now.")
    images, audit_images = _image_parts(proj, session) if getattr(ep, "images", False) else ([], [])
    messages = [{"role": "system", "content": _RULES},
                {"role": "user", "content": [{"type": "text", "text": text}, *images] if images else text}]
    audit_msgs = ([messages[0], {"role": "user", "content": [{"type": "text", "text": text}, *audit_images]}]
                  if images else list(messages))
    row = {"model": getattr(ep, "identity", repr(ep)), "prompt_sha": content_id(audit_msgs),
           "raw_sha": None, "summary": None, "rationale": None, "reason": None, "usage": None}
    audit = {"round": round_no, **row, "messages": audit_msgs, "raw": None, "attempts": []}
    tried, why, path = None, None, None
    for _ in range(MAX_ATTEMPTS):
        try:
            # ponytail: DeepSeek reasoning tokens count against max_tokens and left content
            # empty (3/3 production rounds); the proposer wants the answer, not the preamble
            raw = ep.chat(messages, temperature=0.0, max_tokens=max_tokens,
                          response_format={"type": "json_object"}, thinking={"type": "disabled"})
        except Exception as exc:  # noqa: BLE001 -- unreachable endpoint: rules take over
            row["reason"] = f"{type(exc).__name__}: {exc}"[:300]
            break
        usage = getattr(ep, "last_usage", None)
        if usage:
            row["usage"] = {k: (row["usage"] or {}).get(k, 0) + (usage.get(k) or 0) for k in ("prompt", "completion")}
        audit["raw"] = raw
        row["raw_sha"] = audit["raw_sha"] = sha_json(raw)
        ans = None
        try:
            ans = _parse(raw)
            row["summary"], row["rationale"] = ans["summary"], ans["rationale"]
            tried = _try(ans, proj, before, round_no, preflight)
            why = None
            break
        except Exception as exc:  # noqa: BLE001 -- bad JSON / payload / doctor / preflight: the model repairs
            why = str(exc) if str(exc).startswith(("doctor:", "preflight:")) else f"{type(exc).__name__}: {exc}"
            path = (ans or {}).get("payload", {}).get("path") or path   # the files stay for the operator
            audit["attempts"].append({"raw": raw, "usage": usage, "reason": why})
            repair = {"role": "user", "content": REPAIR.format(why=why[:4000])}
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
