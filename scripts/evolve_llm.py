"""Model-led program improvement through bounded inspection and measured trials.

Production rebuilds a compact request after each tool result; complete evidence
and model messages remain audit artifacts. Model claims never substitute for
measured outcomes.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import tomllib
from pathlib import Path

from harness.config import sha_json
from harness.manifest import PLUGINS_ROOT, mount_params
from harness.protocol import content_id
from harness.registry import load_provider
from harness.skill_executor import SegmentExecutor, StepExecutor
from harness.skill_library import rearm, segment_specs
from scripts import plugin_doctor
from scripts.evolve_evidence import (bound_class_index, compact_brief, compact_command, encoded, inspect_evidence,
                                     page as evidence_page, EvidenceWorkingSet, request_messages)

ENDPOINT_REF = "plugins.model_endpoint:provider"
FAKE_REF = "plugins.model_endpoint:fake_provider"
#: Where a ``card`` answer lands (one dir per candidate); tests point it elsewhere.
CANDIDATES_ROOT = Path(os.environ.get("PH_CANDIDATES_ROOT") or PLUGINS_ROOT / "candidates")
MAX_LOG_LINES = 60
KINDS = ("tunables", "executor", "card", "patch", "plan", "none")
# Optional diagnosis labels, without a prescribed search order.
LAYERS = ("evaluation", "plan", "state", "recovery", "parameter")
_NAME = re.compile(r"^[a-z][a-z0-9_]{2,40}$")

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
    # Preserve observations for every executed action: the model may diagnose an
    # upstream action even when that action's operational completion was true.
    trail = [{k: n[k] for k in ("id", "ok", "steps", "kind", "after", "task", "failure_mode",
                                 "trace", "trace_end", "geometry", "upstream") if k in n}
             for n in s.get("trail") or []]
    ms = first_missing(trail) or s.get("first_death")
    row = {"seed": int(seed),
           **{k: s.get(k) for k in ("success", "first_death", "failure_mode", "fault", "keyframes")},
           "first_missing_milestone": ms, "trail": trail}
    row.update({k: s[k] for k in ('evaluation', 'verification_observations', 'terminal_observation')
                if k in s})
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
            "provider(**params) returns a factory with make_driver(spec), producing a fresh "
            "harness.skill_executor.StepExecutor per segment. Subclass InprocExecutor for defaults. "
            "handshake() returns normalize_handshake('inproc', REF, metadata); reset() clears state; "
            "act(obs) returns the installed embodiment action layout shown in action_order or driver source; "
            "done() is an execution stop request, never a reward. diagnostics() returns observations. "
            "Use only helpers actually listed in primitives. If the installed driver calls bind(env, target=None), "
            "implement that hook using its documented arguments. No simulator-specific helper name is implied. "
            "Read installed source before writing code; reject unsupported action layouts. Candidate code "
            "cannot modify the evaluator, task predicate bindings or reference task goal. "
            "Measured trial errors are returned as evidence for the next model decision."),

    }


def _driver(before: dict, records: dict, emb: str, arm: str, binding: dict, node=None,
            applied: dict | None = None) -> dict:
    """Reflect the installed capability of one action that actually executed."""
    if node is None:
        return {"node": None}
    runs = [s["nodes"][node] for s in before["seeds"].values() if node in s.get("nodes", {})]
    if not runs:
        return {"node": node}
    skill, current = runs[0]["skill"], runs[0]["executor"]
    rec = records.get(skill)
    spec = segment_specs({skill: rec}, emb).get(skill) if rec else {}
    bound = {"scripted", *((spec or {}).get("policies") or {})}
    cards = (applied or {}).get("cards") or {}
    bound.update(k for k, c in cards.items() if c.get("skill") == skill)
    card = cards.get(current) or {}
    ev = rec.evidence.get(emb) if rec else None
    actual_ref = (runs[0].get("driver") or {}).get("ref") or card.get("ref")
    ref = actual_ref or (rearm(spec or {}, arm, current).get("policy_provider")
                         or binding.get("policy") if current in bound else None)
    # Unknown custom executors must not be projected as the stock implementation.
    if not ref or (not actual_ref and current not in bound):
        return {"node": node, "skill": skill, "executor": current,
                "unavailable": "executed provider binding was not recorded"}
    params = mount_params(ref)
    params.update(card.get("params") or {})
    for key, value in ((applied or {}).get("tunables") or {}).get(ref, {}).items():
        params[key] = ({**params[key], **value}
                       if isinstance(value, dict) and isinstance(params.get(key), dict) else value)
    tun = params.get("tunables") if isinstance(params.get("tunables"), dict) else params
    values = {k: v for k, v in tun.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    task = (spec or {}).get("task")
    cls = _stage_classes(ref, task)
    mods = sorted({c.__module__ for c in cls})
    return {"node": node, "skill": skill, "executor": current,
            "source_ref": ref, "state_init": _state_init(cls), "bound_classes": bound_class_index(cls),
            "embodiment": emb if rec is None or emb in rec.bindings else next(iter(rec.bindings), emb),
            "task": task, "modules": mods, "stage_modules": mods,
            "operational_completion_rate": sum(bool(r.get("success")) for r in runs) / len(runs),
            "executors": {k: dict((ev.by_executor if ev else {}).get(k) or {}) for k in sorted(bound)},
            "tunables": {"ref": ref, "path": ["tunables"] if isinstance(params.get("tunables"), dict) else [],
                         "values": values}}


def rsi_projection(doc: dict, before: dict, records: dict, emb: str, arm: str, binding: dict,
                   log_excerpt: list[str]) -> dict:
    """Structured evidence and capabilities for compact briefs and paged inspection."""
    from scripts import evolve  # noqa: PLC0415 -- evolve imports this module
    epoch_start = int(doc.get('epoch_start') or 0)
    rounds = [r for r in doc.get('rounds') or [] if r['round'] >= epoch_start]
    doc = {**doc, 'rounds': rounds}
    if (doc.get('last_outcome') or {}).get('round', epoch_start) < epoch_start:
        doc.pop('last_outcome', None)
    deaths = evolve.death_nodes(before, rounds)
    executed = set()
    for seed in before["seeds"].values():
        for n in seed.get("trail") or []:
            run = seed.get("nodes", {}).get(n.get("id"))
            record = records.get((run or {}).get("skill"))
            kind = n.get("kind") or getattr(record, "kind", None)
            if run and kind in ("segment", "manipulate"):
                executed.add(n["id"])
    drivers = {}
    for node in sorted(executed):
        driver = _driver(before, records, emb, arm, binding, node, doc.get("applied"))
        if driver.get("skill") in records and not driver.get("unavailable"):
            drivers[node] = driver
    observed = evolve._first_death(before)
    fd = dict(drivers.get(observed) or {"node": observed})
    rows = [_seed_row(seed, s, doc.get("reference") or {}) for seed, s in before["seeds"].items()]
    cl = clusters(rows)
    proj = {
        "task": doc["task"], "embodiment": emb, "seeds": doc["seeds"], "arm": arm,
        "round": int(doc.get("cursor") or 0) + 1,
        "epoch_start": epoch_start,
        "experiment_id": before.get("experiment_id"),
        "evaluation_contract": {k: v for k, v in (doc.get('evaluation_contract') or {}).items()
                                if k in ('sha', 'version', 'evidence_policy', 'task',
                                         'obligations', 'limitation')},
        "applied": doc.get("applied"),
        "history": [{"round": r["round"], "proposer": r.get("proposer"),
                     "tried": {k: r["tried"][k] for k in ("kind", "node")}
                     | {"detail": {k: v for k, v in r["tried"]["detail"].items()
                                   if k in ("to", "from", "path", "ref", "module", "reason", "error",
                                            "hint", "patch_sha")}},
                     "before": r["before"], "after": r["after"], "published": r["published"],
                     "verdict": r.get("accepted_reason"),
                     "summary": (r.get("llm") or {}).get("summary"), "outcome": r.get("outcome"),
                     "notes": (r.get("tried") or {}).get("detail", {}).get("notes"),
                     "llm": {k: v for k, v in (r.get("llm") or {}).items()
                             if k in ("status", "error", "reason")},
                     "experiments": r.get("experiments"),
                     **({"trial_evidence": r["trial_evidence"]} if r.get("trial_evidence") else {}),
                     "per_seed": [{k: s.get(k) for k in ("seed", "success", "first_death", "failure_mode")}
                                  for s in r.get("after_seeds") or r.get("per_seed") or []]}
                    for r in rounds],
        "this_round": {"count": before["count"], "seeds_total": len(before["seeds"]),
                       "per_seed": rows},
        # Failure observations and previous measured hypotheses, without assigning a target.
        "clusters": cl,
        "last_outcome": doc.get("last_outcome"),
        "score_definition": doc.get("score_definition") or SCORE_DEF,
        "trial_evidence": ((doc.get("last_outcome") or {}).get("trial_evidence")
                           or (rounds[-1].get("trial_evidence") if rounds else None)),
        "accepted_stack": doc.get("accepted_stack") or [],
        "first_death": fd,
        "drivers": drivers,
        "needs": rounds[-1].get("needs") if rounds else [],
        "log_excerpt": list(log_excerpt)[:MAX_LOG_LINES],
        "death_nodes": [{k: d[k] for k in ("node", "seeds", "failure_mode", "failed_observations")}
                        for d in deaths],
    }
    proj["diagnosis"] = doc.get("diagnosis")
    proj["experience"] = doc.get("experience")
    proj["plan_space"] = doc.get("plan_space")
    return proj


SCORE_DEF = ("分数来自冻结的独立验证条件。比较同一批种子的条件向量：至少一个条件改善，"
             "且任何已通过条件都不能回退。图节点数、控制器目标和自报诊断不产生奖励。"
             "整任务完成由独立终态 oracle 判定。开发接受不等于验证安装。")


def model_request_config(model=None, effort="off"):
    """Resolve operator choices against the installed endpoint's effort declaration."""
    from plugins.model_endpoint import reasoning_options
    if model is not None and (not isinstance(model, str) or not model.strip()
                              or len(model) > 200 or model != model.strip()
                              or any(ord(c) < 32 for c in model)):
        raise ValueError("llm_model must be a nonempty model ID of at most 200 characters")
    params = dict(mount_params(ENDPOINT_REF) or {"preset": "deepseek"})
    if model is not None:
        params["model"] = model
    return params, reasoning_options(effort, params.get("reasoning_efforts"))


def endpoint(model=None):
    """The model_endpoint card, mounted by ref the way planner_vlm does (its declared
    params; DeepSeek preset when the card names none); the fake when tests ask."""
    if os.environ.get("PH_MODEL_ENDPOINT_FAKE"):
        return load_provider(FAKE_REF, {})
    params, _ = model_request_config(model)
    return load_provider(ENDPOINT_REF, params)


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
        raise ValueError(f"layer must be one of {'|'.join(LAYERS)}, got {lay!r}")
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


def _write_files_once(path: Path, files: dict[str, str]) -> str | None:
    """An import name identifies one immutable candidate, including rejected candidates."""
    if path.is_symlink():
        return 'candidate path must not be a symlink'
    if path.exists():
        existing = {p.name: p.read_text() for p in path.iterdir()
                    if p.is_file() and p.suffix != '.pyc'}
        if existing != files:
            return 'candidate name is immutable; choose a new unused name for changed source'
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=path.parent, prefix='.candidate-'))
    try:
        for name, body in files.items():
            (temporary / name).write_text(body)
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return None


def candidate_digest(path: str | Path) -> str:
    """Content identity of all source/config files, excluding interpreter caches."""
    root = Path(path)
    return sha_json({str(p.relative_to(root)): p.read_bytes().hex() for p in root.rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'})


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
    if why := _write_files_once(d, files):
        return f'doctor:{why}'
    pay["path"] = str(d)
    pay['artifact_sha'] = candidate_digest(d)
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
            "inspect(view=source, module=..., symbol=...) for the chosen node; "
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
import types

from harness.skill_executor import InprocExecutor, normalize_handshake

REF = "{ref}"
INSTALLED = "{installed}"   # the card's _STAGES table -- read, never written
PATCHED = "{module}"        # the installed module whose copy ({base}.py) carries the diff
TASK = "{task}"


def _repoint(cls, mod, cache=None):
    # Rebuilt methods must close over the rebuilt class for zero-argument super().
    cache = {{}} if cache is None else cache
    if cls in cache:
        return cache[cls]
    if cls is object:
        return cls
    if cls.__module__ == PATCHED:
        return getattr(mod, cls.__name__)
    bases = tuple(_repoint(b, mod, cache) for b in cls.__bases__)
    if bases == cls.__bases__:
        return cls
    cell = (lambda value: lambda: value)(None).__closure__[0]

    def rebound(value):
        if isinstance(value, (staticmethod, classmethod)):
            return type(value)(rebound(value.__func__))
        if isinstance(value, property):
            return property(*(rebound(f) if f else None for f in
                              (value.fget, value.fset, value.fdel)), doc=value.__doc__)
        if not isinstance(value, types.FunctionType) or "__class__" not in value.__code__.co_freevars:
            return value
        closure = tuple(cell if name == "__class__" else old
                        for name, old in zip(value.__code__.co_freevars, value.__closure__))
        fn = types.FunctionType(value.__code__, value.__globals__, value.__name__,
                                value.__defaults__, closure)
        fn.__kwdefaults__, fn.__annotations__ = value.__kwdefaults__, dict(value.__annotations__)
        fn.__dict__.update(value.__dict__)
        fn.__qualname__, fn.__module__, fn.__doc__ = value.__qualname__, value.__module__, value.__doc__
        return fn

    namespace = {{k: rebound(v) for k, v in vars(cls).items()
                 if k not in ("__dict__", "__weakref__", "__classcell__")
                 and not isinstance(v, (types.MemberDescriptorType, types.GetSetDescriptorType))}}
    namespace["__classcell__"] = cell
    cache[cls] = result = type(cls)(cls.__name__, bases, namespace)
    return result


def make_stage():
    mod, orig = importlib.import_module(__name__ + ".{base}"), importlib.import_module(PATCHED)
    if hasattr(mod, "mount_tunables"):   # the copy sees the effective knobs of the installed module
        mod.mount_tunables(orig.tunables())
    factory = importlib.import_module(INSTALLED)._STAGES[TASK][0]
    swaps, cache = {{}}, {{}}
    for c in type(factory()).__mro__[:-1]:
        r = _repoint(c, mod, cache)
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
    generated_name = not isinstance(name, str) or not _NAME.match(name)
    generated_to = not isinstance(to, str) or not to
    if not isinstance(name, str) or not _NAME.match(name):
        name = pay["name"] = f"patch_r{round_no}"   # bookkeeping ids, never worth an attempt:
    if not isinstance(to, str) or not to:           # the live model burned 2 of its 3 on these
        to = pay["to"] = name
    if module not in (fd.get("modules") or []):
        return (f"patch:module must be one of node {fd.get('node')!r}'s modules {fd.get('modules')}, "
                f"got {module!r}; use drivers[payload.node].modules and that node's source material")
    # ...and inside that list, only a module this stage's MRO actually contains can change
    # anything: make_stage() swaps the classes whose ``__module__`` is the patched one, so a
    # patch to a module the stage never inherits from installs a card that runs the STOCK
    # code. Replayed over evolve-recycle_cans' 363 patch rounds: 171 (nav-can1 x
    # recycle_driver 153, x planner 11, drop-can1 x planner 7) were mechanically no-ops that
    # still passed the doctor and burned a whole suite (12,569 s of simulator, 48.8% of
    # the campaign's 25,777 s). The advertised modules must stay within this stage's
    # actual inheritance tree.
    if (sm := fd.get("stage_modules")) and module not in sm:
        return (f"patch:{module} cannot change node {fd.get('node')!r}: the card only swaps the classes "
                f"this stage inherits from, and those come from {', '.join(sm)}. A patch to {module} "
                f"would install a candidate that runs the stock code. Send `module`: one of {sm}, or "
                "put the node whose stage lives in that module in `payload.node`.")
    if edits is not None and not (isinstance(edits, list) and edits):
        return "patch:`edits` must be a non-empty list of {old, new} objects"
    if edits is None and not (isinstance(diff, str) and diff.strip()):
        return ("patch:payload needs `edits`: [{old, new}] -- each `old` copied verbatim out of "
                "the inspect source page (a unified `diff` is still accepted instead)")
    mod = importlib.import_module(module)
    src = Path(inspect.getsourcefile(mod)).read_text()
    modes: list = []
    try:
        new = apply_edits(src, edits, modes) if edits is not None else apply_diff(src, diff)
    except ValueError as exc:
        return f"patch:{exc}{_elsewhere(edits, module, fd)}"
    if new == src:
        return "patch:the patch changes nothing"
    if generated_name:
        name = pay['name'] = f'patch_r{round_no}_{_patch_key(new)[:10]}'
        if generated_to:
            to = pay['to'] = name
    pkg, base = _card_package(root), module.rpartition(".")[2]
    ref = f"{pkg}{name}:provider"
    d = root / name
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
    manifest = (
        f'needs_sim = true\npatched_from = "{origin}"\n'
        + (f"third_party = {list(tp)!r}\n" if tp else "")
        + f'[executors.{to}]\nskill = "{fd.get("skill")}"\nembodiment = "{fd.get("embodiment")}"\n'
        f'ref = "{ref}"\ntransport = "inproc"\n{tun}')
    entry = PATCH_CARD.format(
        name=name, module=module, round=round_no, to=to, skill=fd.get("skill"), ref=ref,
        installed=str(fd.get("tunables", {}).get("ref", "")).partition(":")[0], base=base, task=fd.get("task"))
    if why := _write_files_once(d, {f'{base}.py': _by_ref(new), 'manifest.toml': manifest, '__init__.py': entry}):
        return f'patch:{why}'
    pay["path"], pay["ref"], pay["match"] = str(d), ref, modes or ["diff"]
    pay["patch_sha"] = _patch_key(new)   # the resulting MODULE, keyed past comments/whitespace
    pay['artifact_sha'] = candidate_digest(d)
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


_FRAME = re.compile(r'File "([^"]+)", line (\d+)')
#: scripts.evolve.self_check's finding: "<file>.py: <Class> reads self.<attr>, which ..."
_FINDING = re.compile(r"([\w.]+\.py): \w+ reads (self\.\w+)")


def _edit_key(module, edits) -> tuple:
    return (module, tuple(sorted((str((e or {}).get("old", "")).strip(),
                                  str((e or {}).get("new", "")).strip()) for e in edits)))


def _same_context_history(proj: dict, node):
    eid = proj.get("experiment_id")
    return [r for r in proj.get("history") or [] if eid
            and (r.get("experiments") or {}).get("before") == eid
            and (r.get("tried") or {}).get("node") == node]


def _ran_patches(proj: dict, node=None) -> dict:
    """Identical code measured on the same node, baseline, seeds and evaluator."""
    out = {}
    for r in _same_context_history(proj, node):
        sha = ((r.get("tried") or {}).get("detail") or {}).get("patch_sha")
        if sha and r.get("after") is not None:
            out.setdefault(str(sha), f"round {r.get('round')}: {r.get('verdict') or r.get('outcome')}")
    return out


def _tried_values(proj: dict, fd: dict) -> set:
    out = set()
    for r in _same_context_history(proj, fd.get("node")):
        t = r.get("tried") or {}
        d = t.get("detail") or {}
        if t.get("kind") == "tunables" and d.get("ref") == (fd.get("tunables") or {}).get("ref") \
                and isinstance(d.get("to"), (int, float)) and not isinstance(d.get("to"), bool) \
                and r.get("after") is not None:
            out.add((tuple(d.get("path") or []), float(d["to"])))
    return out


def _accepted_repeat(proj: dict, pay: dict) -> str | None:
    """An identical patch already active on this exact node is a no-op."""
    if not isinstance(pay.get("edits"), list) or not pay["edits"]:
        return None
    fd = (proj.get("drivers") or {}).get(pay.get("node")) or {}
    for c in reversed(proj.get("accepted_stack") or []):
        d = c.get("detail") if isinstance(c.get("detail"), dict) else c
        if pay.get("node") and c.get("node") == pay.get("node") and d.get("to") == fd.get("executor") \
                and isinstance(d.get("edits"), list) \
                and _edit_key(d.get("module"), d["edits"]) == _edit_key(pay.get("module"), pay["edits"]):
            return f"this exact edit is already active on node {pay['node']} (round {c.get('round')})"
    return None


def _stamp(tried: dict, ans: dict, layer) -> dict:
    """The round's diagnosis on the try: the layer it claims (``detail.layer``, the round row
    and rsi_step carry it) and its notebook note (``detail.notes``)."""
    if layer:
        tried["detail"]["layer"] = layer
    if isinstance(ans.get("notes"), str) and ans["notes"].strip():
        tried["detail"]["notes"] = ans["notes"].strip()[:600]
    return tried


def _try(ans: dict, proj: dict, before: dict, round_no: int, seen: set | None = None) -> dict:
    """Validate a model-selected intervention against its actual installed capability.

    Exact duplicate checks require the same node and baseline experiment. The shared
    capability checks compile the candidate without running it. ProgramLearner owns
    measured execution and paired acceptance.
    """
    from scripts.evolve import _none, from_proposal  # noqa: PLC0415 -- evolve imports this module
    seen = set() if seen is None else seen
    pay = ans["payload"]
    kind, node = ans["kind"], pay.get("node")
    if kind not in ("plan", "none") and (not isinstance(node, str) or node not in (proj.get("drivers") or {})):
        raise ValueError(f"payload.node must explicitly name an executed driver in {sorted(proj.get('drivers') or {})}; got {node!r}")
    fd = (proj.get("drivers") or {}).get(node) or {}
    p = {"id": f"llm:round-{round_no}", "kind": kind, "payload": pay, "note": ans["rationale"]}
    layer = ans.get("layer")
    if kind == "none":
        return _stamp(_none(f"llm: {ans['rationale'] or 'nothing to try'}", node), ans, layer)
    if ans["kind"] == "tunables":
        if pay.get("ref") != fd.get("tunables", {}).get("ref") or not isinstance(pay.get("to"), (int, float)) \
                or not (isinstance(pay.get("path"), list) and all(isinstance(x, str) for x in pay["path"])):
            raise ValueError(f"tunables payload must be {{ref: {fd.get('tunables', {}).get('ref')!r}, "
                             f"path: [str], to: number}}, got {pay}")
        cur = (fd.get("tunables", {}).get("values") or {}).get(pay["path"][-1] if pay["path"] else None)
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            val = (tuple(pay["path"]), float(pay["to"]))
            if val in _tried_values(proj, fd):
                raise ValueError(f"this exact parameter value was measured on this node under the same baseline: {val}")
            exact = ("value", node, pay["ref"], *val)
            if exact in seen:
                raise ValueError("this exact parameter value was already proposed this round")
            seen.add(exact)
        tried = from_proposal(p, before)
    elif ans["kind"] == "plan":
        from plugins.rsi import interventions
        space = proj.get("plan_space")
        if not space:
            raise ValueError("this task exposes no validated plan intervention")
        interventions.validate_from_space(pay.get("graph"), space)
        tried = from_proposal(p, before)
    elif ans["kind"] == "executor":
        if pay.get("to") not in fd.get("executors", {}) or pay.get("to") == fd.get("executor"):
            raise ValueError(f"executor.to must be another key of {sorted(fd.get('executors', {}))}, got {pay.get('to')!r}")
        exact = ("executor", node, pay["to"])
        if exact in seen:
            raise ValueError("this exact executor switch was already proposed this round")
        if any((r.get("tried") or {}).get("detail", {}).get("to") == pay["to"]
               and (r.get("tried") or {}).get("kind") == "executor" and r.get("after") is not None
               for r in _same_context_history(proj, node)):
            raise ValueError("this executor switch was already measured on the same node and baseline")
        seen.add(exact)
        tried = from_proposal(p, before)
    else:   # card / patch: materialised under the candidates root, then the same card path
        if why := (_accepted_repeat(proj, pay) if ans["kind"] == "patch" else None):
            raise ValueError(why)
        if why := (write_card(pay) if ans["kind"] == "card" else write_patch(pay, fd, round_no)):
            raise ValueError(why)
        if ans["kind"] == "patch" and (ran := _ran_patches(proj, node).get(str(pay.get("patch_sha")))):
            raise ValueError(f"patch:this exact code was already measured on the same node and baseline: {ran}")
        tried = from_proposal({**p, "kind": "card", "payload": {k: pay[k] for k in ("path", "to", "ref", "params", "node", "artifact_sha") if k in pay}}, before)
        if ans["kind"] == "patch" and tried["kind"] == "card":
            # patch_sha too: the ALREADY-RAN gate reads it off the HISTORY rows, so a stamp
            # that stops here is a gate that never fires (it did not, for a whole campaign)
            tried["detail"].update(module=pay["module"],
                                   **{k: pay[k] for k in ("edits", "diff", "match", "patch_sha") if k in pay})
    if tried["kind"] == "none":
        raise ValueError(tried["detail"]["reason"])   # from_proposal's refusal: the answer was unusable
    return _stamp(tried, ans, layer)


AGENT_BUDGET = {"max_request_bytes": 24_000, "max_input_bytes": 96_000,
                "max_calls": 8, "max_tool_bytes": 8_000, "max_working_set_bytes": 8_000,
                "max_output_tokens": 4096, "max_read_calls": 2}
_AGENT_RULES = """Improve a robot program through measured experiments. Return JSON:
{"op":"inspect|trial|choose|stop","args":{...},"reason":"brief hypothesis/evidence","memo":"findings and next experiment, <=1000 characters"}.
Each request is self-contained. state.policy_id is observed; omitted trial parent
means incumbent_policy_id, NEVER the last viewed candidate. driver_index lists
all actions; capabilities[driver.capability] gives tunables, executor keys and
editable modules. inspect {view:catalog,node} gives that node's exact class/method
IDs; use a returned method ID as source.symbol. Never guess a provider's module
or class name. catalog_ref reads the full directory. Select the node explicitly.

trial args={kind,payload,parent_policy_id?,seed?}: one reset development episode.
Payloads: tunables={node,parameter,to:number} (declared only; legacy ref/path also
validated); executor={node,to:installed key}; neither needs a source read.
patch={node,module,edits:[{old:exact source,new:replacement}]}: read this parent's
code first; only its editable modules may change. card={node,name,files,to,ref}:
inspect node contract=true first. plan={graph}: inspect permitted actions; retain
existing calls/order/goals/verifier checkpoints, insert permitted actions only.
A neutral measured candidate can parent another edit, including in later cycles.
Working policies persist across cycles under the same incumbent and evaluator;
their measurements and exact edits are retained within state.workspace's limit.
A trial refreshes observed state, NOT the incumbent. Use feedback to revise
hypotheses and compose. Identical policy+seed measurements are cached, not new
evidence; change the experiment or use a different declared seed to learn more.
No gain from one edit does not disprove a conjunction of edits. Distinguish a
rejected hypothesis from an untested combination; the model selects both.

inspect args={view:catalog|node|parameter|source|trace|plan|history|evaluation,...}
or {requests:[{view,...},...]} for <=4 reads sharing one tool-byte allowance.
Node/trace/parameter require node. Parameter without name lists values; parameter:
name gives static uses, not write permission or runtime truth. Source node/module
gives a directory; symbol:"<class ID>.<method>" gives literal code (or module+symbol
/ absolute start,end). Continue symbols with cursor=next, NOT start/end. Each page
has sha/cursor/next; non-code partial pages are JSON fragments. Optional read args:
policy_id,seed,cursor,limit. node contract=true gives executor protocol/primitives.
view=evaluation resolves the frozen objectives/verifier sources.

read_calls_left bounds successful inspection batches between probes separately from probe_budget.
Only a NEW measured probe replenishes reads; zero reads still permits trials.
last_tool_result+retained_evidence is a bounded cache: don't reread resident pages.
Omitted detail keeps read references; errors identify commands to repair. Budgets
charge actual bytes/calls. Costs are not reward: use measured effects to decide
whether another probe or a full evaluation is worth its stated cost.

choose args={policy_id} runs the FULL paired development set for a measured working
candidate. Incumbent choice or stop args={reason} retains the incumbent. Probes
NEVER accept/install. In continuous mode stop/incumbent choice ends this cycle;
the controller continues. Only frozen independent verifier/terminal results count;
motion/distance/steps/graph size/authorable targets are not reward. Preserve
unknown/false/zero. working_policies keeps measured per-seed changes and costs;
historical replay/cycle memos are hypotheses, not executable parents; choose
parents only from working_policies. transfer_only means unknown or
different evaluator. In phase=selection only choose/stop: use an existing call,
never invent or auto-select a policy. Keep evidence and hypotheses distinct.
Cycle counters report repeated non-sampling/non-evaluation decisions. When they
grow, use retained experiments to revise the next hypothesis; repeating a stop
with the same rationale produces no evidence. End-cycle memo should state what
was measured, what remains unknown, and a concrete next experiment (parent,
intervention or evidence needed), not simply repeat the incumbent's score.
If a measured candidate warrants full evaluation, choose it; another probe is
not a prerequisite. A cycle ending does not discard its unaccepted candidates.
"""


def _agent_command(raw):
    try:
        value = json.loads(raw)
    except ValueError:
        start = raw.find("{")
        if start < 0:
            raise ValueError("reply contains no JSON object")
        value, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(value, dict):
        raise ValueError("reply must be an object")
    if "op" not in value:
        old = _parse(raw)
        return ({"op": "stop", "args": {"reason": old["rationale"]}, "reason": old["rationale"]}
                if old["kind"] == "none" else
                {"op": "trial", "args": {"kind": old["kind"], "payload": old["payload"]},
                 "reason": old["rationale"], "summary": old["summary"], "legacy_choose": True})
    if value["op"] not in ("inspect", "trial", "choose", "stop") or not isinstance(value.get("args"), dict):
        raise ValueError("op must be inspect/trial/choose/stop and args must be an object")
    return value


def llm_propose(ep=None, proj: dict | None = None, before: dict | None = None,
                round_no: int = 0, audit_dir: Path | None = None,
                max_tokens: int = 4096, session: Path | None = None,
                agent_tools: dict | None = None, budget: dict | None = None,
                evidence: EvidenceWorkingSet | None = None,
                model: str | None = None, effort: str = "off") -> tuple[dict, dict]:
    """One bounded evidence/probe loop; only the model chooses a final candidate."""
    if agent_tools is None:
        raise ValueError("agent_tools is required; use the measured program-policy loop")
    from scripts.evolve import _none  # noqa: PLC0415
    if proj is None or before is None or audit_dir is None:
        raise ValueError("proj, before and audit_dir are required")
    if any(not callable(agent_tools.get(name)) for name in ("trial", "choose", "projection")):
        raise ValueError("agent_tools requires trial, choose and projection callbacks")
    _, thinking_options = model_request_config(model, effort)
    overrides = (budget or {}).get("limits", budget or {})
    limits = {key: int(overrides.get(key, value)) for key, value in AGENT_BUDGET.items()}
    if any(value < 0 for value in limits.values()):
        raise ValueError("budget limits must be nonnegative")
    if limits["max_tool_bytes"] < 512:
        raise ValueError("max_tool_bytes must be at least 512")
    if limits["max_working_set_bytes"] < 640:
        raise ValueError("max_working_set_bytes must be at least 640")
    limits["max_output_tokens"] = min(limits["max_output_tokens"], int(max_tokens))
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    row = {"model": ENDPOINT_REF, "requested_model": model, "effort": effort,
           "method": "online_program_policy_v1", "prompt_sha": None,
           "raw_sha": None, "summary": None, "rationale": None, "reason": None, "memo": "", "usage": None,
           "status": None, "error": None, "calls": 0, "evidence_reads": 0, "evidence_refs": [], "trial_calls": 0,
           "decision_flow": {"requested": {}, "executed": {}, "errors": [], "other_errors": 0, "selection_calls": 0},
           "usage_complete": True,
           "stop_reason": None, "budget": {"limits": limits,
               "used": {"input_bytes": 0, "calls": 0, "tool_bytes": 0, "output_tokens": None},
               "usage_complete": True}}
    audit = {"round": round_no, "experiment_id": proj.get("experiment_id"), "messages": [],
             "brief": compact_brief(proj), "raw": None, "attempts": [], "requests": [],
             "events": [], "materials": {}, "materials_by_node": {}, "max_calls": limits["max_calls"]}
    tried, result, previous_command = None, None, None
    memo = str((proj.get("cycle_context") or {}).get("memo") or "")[-1000:]
    invalid_key, invalid_count = None, 0
    current = proj
    inspected, seen_by_policy = set(), {}
    usage_sum = {"prompt": 0, "completion": 0}
    used = row["budget"]["used"]
    working = evidence if evidence is not None else EvidenceWorkingSet(limits["max_working_set_bytes"])
    working.retain_sources(agent_tools['projection'])
    read_calls = 0

    def persist():
        row["memo"] = memo
        row["usage_complete"] = row["budget"]["usage_complete"]
        audit.update(row, tried=tried)
        target = audit_dir / f"round-{round_no}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=1, sort_keys=True, default=str))
        temporary.replace(target)

    def finish(reason, *, error=None):
        nonlocal tried
        row.update(reason=reason, stop_reason="model_error" if error else "budget_exhausted",
                   status="error" if error else "abstained", error=error)
        tried = _none(reason, needs=("model_endpoint",) if error else ("budget",))
        persist()
        return tried, row

    def feedback(value, view):
        nonlocal result
        # Every complete tool result is retained out of band; requests carry one page.
        audit["events"].append({"type": "tool", "view": view, "result": value, "sha": content_id(value)})
        result = evidence_page(value, view=view,
                               policy_id=value.get("policy_id") if isinstance(value, dict) else None,
                               max_bytes=working.page_limit(limits["max_tool_bytes"], trial=view == "trial"))
        working.add(result, trial=view == "trial")
        used["tool_bytes"] += len(encoded(result))

    persist()
    if not limits["max_calls"] or not limits["max_input_bytes"] or not limits["max_output_tokens"]:
        return finish("budget exhausted before the next model call")
    try:
        ep = ep if ep is not None else (endpoint(model) if model is not None else endpoint())
        row["model"] = getattr(ep, "identity", type(ep).__name__)
    except Exception as exc:  # noqa: BLE001 -- endpoint error is sealed, never substituted
        return finish(f"model endpoint failed: {exc}", error={"type": type(exc).__name__, "message": str(exc)[:2000], "stage": "endpoint"})

    while used["calls"] < limits["max_calls"]:
        # A failed trial may consume a probe without returning a new policy.
        # Refresh counters while retaining the currently observed policy.
        current = agent_tools["projection"](current.get("policy_id"))
        state = compact_brief(current)
        index_sha = content_id(state["source_index"])
        state["source_index_sha"] = index_sha
        body = {"state": state, **working.snapshot(),
                "read_calls_left": max(0, limits["max_read_calls"] - read_calls),
                "previous_command": previous_command, "memo": memo,
                "budget": row["budget"]}
        body, messages = request_messages(body, _AGENT_RULES,
            min(limits["max_request_bytes"], limits["max_input_bytes"] - used["input_bytes"]))
        input_bytes = len(encoded(messages))
        if ((state.get('probe_budget') or {}).get('used', 0) > 0
                and any(p.get("policy_id") != state.get("incumbent_policy_id")
                        for p in state.get("working_policies") or [])):
            selection = {"phase": "selection", "allowed_ops": ["choose", "stop"],
                         "state": {k: state[k] for k in ("task", "round", "policy_id", "incumbent_policy_id",
                                   "evaluation", "experiment_id", "development_seeds", "probe_budget",
                                   "working_policies", "workspace", "evaluation_budget", "development_cost", "cycle_context") if k in state},
                         "memo": memo, "budget": row["budget"],
                         "last_tool_result": (result if result and result.get("view") == "error" else None)}
            selection, selection_messages = request_messages(selection, _AGENT_RULES, limits["max_request_bytes"])
            selection_bytes = len(encoded(selection_messages))
            # Spend an existing call on the model's choice before exploration
            # consumes the remaining budget. Never auto-select a probe.
            if (used["calls"] + 1 == limits["max_calls"]
                    or input_bytes > limits["max_request_bytes"]
                    or used["input_bytes"] + input_bytes + selection_bytes > limits["max_input_bytes"]):
                body, messages, input_bytes = selection, selection_messages, selection_bytes
        if input_bytes > limits["max_request_bytes"]:
            return finish("budget exhausted: next request exceeds max_request_bytes")
        if used["input_bytes"] + input_bytes > limits["max_input_bytes"]:
            return finish("budget exhausted: next request exceeds cumulative max_input_bytes")
        used["input_bytes"] += input_bytes
        used["calls"] += 1
        flow = row['decision_flow']
        flow['selection_calls'] += int(body.get('phase') == 'selection')
        # Restored authority comes only from verified literal code visible in
        # this request, never a permission bit surviving after its page expired.
        for page in [body.get('last_tool_result'), *(body.get('retained_evidence') or [])]:
            if page and page.get('code_read') and isinstance(page.get('data'), dict) and page['data'].get('code'):
                inspected.update((page.get('policy_id'), node) for node in page['data'].get('owner_nodes', []))
        row["calls"] = used["calls"]
        if not audit["messages"]:
            audit["messages"] = list(messages)
            audit["brief"] = body["state"]
            row["prompt_sha"] = content_id(messages)
        options = {"max_tokens": limits["max_output_tokens"],
                   "response_format": {"type": "json_object"}, **thinking_options}
        request = {"call": row["calls"], "messages": messages, "prompt_sha": content_id(messages),
                   "input_bytes": input_bytes, "options": options}
        audit["requests"].append(request)
        persist()
        try:
            raw = ep.chat(messages, **options)
        except Exception as exc:  # noqa: BLE001
            row["budget"]["usage_complete"] = False
            used["output_tokens"], row["usage"] = None, None
            request["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            return finish(f"model endpoint chat failed: {exc}", error={**request["error"], "stage": "chat"})
        row["model"] = getattr(ep, "identity", row["model"])
        audit["raw"], row["raw_sha"] = raw, content_id(raw)
        usage = getattr(ep, "last_usage", None)
        valid_usage = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in usage_sum)
        if valid_usage:
            for key in usage_sum:
                usage_sum[key] += usage[key]
        else:
            row["budget"]["usage_complete"] = False
        row["usage"] = dict(usage_sum) if row["budget"]["usage_complete"] else None
        used["output_tokens"] = usage_sum["completion"] if row["budget"]["usage_complete"] else None
        request.update(raw=raw, raw_sha=row["raw_sha"], usage=usage,
                       finish_reason=getattr(ep, "last_finish", None))
        persist()
        command = None
        try:
            command = _agent_command(raw)
            previous_command = compact_command(command)
            op, args = command["op"], command["args"]
            flow['requested'][op] = flow['requested'].get(op, 0) + 1
            memo = str(command.get("memo") or memo)[-1000:]
            row["rationale"] = str(command.get("reason") or "")[:2000]
            row["summary"] = str(command.get("summary") or row["rationale"] or op)[:600]
            audit["events"].append({"type": "command", "command": command})
            if body.get("phase") == "selection" and op not in body["allowed_ops"]:
                raise ValueError("selection phase: choose a measured policy or stop; exploration budget is reserved for this decision")
            if op == "stop":
                flow['executed'][op] = flow['executed'].get(op, 0) + 1
                reason = str(args.get("reason") or row["rationale"] or "model stopped")
                tried = _none(reason, needs=("evidence",))
                row.update(status="abstained", reason=reason, stop_reason="model_stop")
                break
            if op == "inspect":
                if read_calls >= limits["max_read_calls"]:
                    raise ValueError("read budget exhausted for this sampling decision; trial, choose a measured policy, or stop")
                read_started = False
                reads = args.get("requests", [args])
                if (not isinstance(reads, list) or not 1 <= len(reads) <= 4
                        or any(not isinstance(read, dict) or "requests" in read for read in reads)):
                    raise ValueError("inspect requests must contain 1 to 4 flat read objects")
                remaining = working.page_limit(limits["max_tool_bytes"])
                if remaining < 512 * len(reads):
                    raise ValueError("batch exceeds tool-byte capacity; request fewer pages")
                for i, read in enumerate(reads):
                    policy = read.get("policy_id")
                    current = agent_tools["projection"](policy)
                    cap = remaining // (len(reads) - i)
                    if read.get("view") == "node" and read.get("contract"):
                        fd = (current.get("drivers") or {}).get(read.get("node"))
                        if not fd:
                            raise ValueError("inspect.args.node is required for node contract inspection; use an executed driver in driver_index")
                        material = {**_contract(_card_package(CANDIDATES_ROOT), fd.get("skill"), fd.get("embodiment")),
                                    **(_primitives(fd["source_ref"]) if fd.get("source_ref") else {})}
                        value = evidence_page(material, view="node", node=read.get("node"), policy_id=current.get("policy_id") or policy,
                                              cursor=read.get("cursor"), limit=read.get("limit"), max_bytes=cap)
                    else:
                        value = inspect_evidence(current, read, max_bytes=cap)
                    audit["events"].append({"type": "tool", "view": "inspect", "request": read,
                                             "result": value, "sha": content_id(value)})
                    result = value
                    working.add(value)
                    if not read_started:
                        read_calls += 1
                        read_started = True
                    size = len(encoded(value))
                    remaining -= size
                    used["tool_bytes"] += size
                    ref = {k: value[k] for k in ("view", "node", "policy_id", "sha", "cursor", "next")}
                    old_ref = next((r for r in row["evidence_refs"] if working.key(r) == working.key(ref)), None)
                    if old_ref is None:
                        row["evidence_refs"].append(ref)
                    else:
                        ref = old_ref
                    row["evidence_reads"] += 1
                    if read.get("view") == "node" and read.get("contract"):
                        inspected.add((current.get("policy_id") or policy, read.get("node")))
                    elif read.get("view") == "source" and value.get("code_read") is True:
                        owners = value["data"]["owner_nodes"]
                        ref["owner_nodes"] = owners
                        inspected.update((current.get("policy_id") or policy, owner) for owner in owners)
                invalid_key, invalid_count = None, 0
                flow['executed'][op] = flow['executed'].get(op, 0) + 1
                persist()
                continue
            if op == "choose":
                policy_id = args.get("policy_id")
                if not isinstance(policy_id, str) or not policy_id:
                    raise ValueError("choose requires policy_id from a measured trial")
                if policy_id == current.get("incumbent_policy_id"):
                    flow['executed'][op] = flow['executed'].get(op, 0) + 1
                    reason = "model retained the incumbent"
                    tried = _none(reason, needs=("evidence",))
                    row.update(status="abstained", reason=reason, stop_reason="model_stop")
                    break
                tried = agent_tools["choose"](policy_id)
                flow['executed'][op] = flow['executed'].get(op, 0) + 1
                row.update(status="proposed", stop_reason="chosen")
                break
            parent = args.get("parent_policy_id")
            current = agent_tools["projection"](parent)
            payload = args.get("payload")
            if args.get("kind") == "tunables" and isinstance(payload, dict) and "parameter" in payload:
                knobs = ((current.get("drivers") or {}).get(payload.get("node")) or {}).get("tunables") or {}
                parameter = payload["parameter"]
                if (not isinstance(parameter, str) or parameter not in (knobs.get("values") or {})
                        or set(payload) - {"node", "parameter", "to"}):
                    raise ValueError(f"tunables payload is {{node,parameter,to}}; declared parameters: {sorted(knobs.get('values') or {})}")
                payload = {"node": payload["node"], "ref": knobs.get("ref"),
                           "path": [*knobs.get("path", []), parameter], "to": payload.get("to")}
            ans = _parse(json.dumps({"kind": args.get("kind"), "payload": payload,
                                    "summary": row["summary"], "rationale": row["rationale"]}))
            node = ans["payload"].get("node")
            if ans["kind"] in ("patch", "card") and (current.get("policy_id") or parent, node) not in inspected:
                raise ValueError("requires_inspection: read source code or the executor contract for the chosen node under this parent_policy_id before patch/card trial")
            parent_before = agent_tools["baseline"](parent) if callable(agent_tools.get("baseline")) else before
            candidate = _try(ans, current, parent_before, round_no,
                             seen_by_policy.setdefault(current.get("policy_id") or parent, set()))
            if ans["kind"] == "tunables":
                candidate["detail"]["from"] = ((current.get("drivers") or {}).get(node, {})
                                                .get("tunables", {}).get("values", {}).get(ans["payload"]["path"][-1]))
            kwargs = {"parent_policy_id": parent}
            if args.get("seed") is not None:
                kwargs["seed"] = args["seed"]
            row["trial_calls"] += 1
            persist()
            receipt = agent_tools["trial"](candidate, **kwargs)
            flow['executed'][op] = flow['executed'].get(op, 0) + 1
            feedback(receipt, "trial")
            if not receipt.get("cached"):
                read_calls = 0
            # The next decision must see the sampled world's current evidence,
            # including motion and newly reached nodes, not the old parent's
            # observations. An omitted parent on a future trial still resolves
            # to the incumbent in the trial handler above.
            current = agent_tools["projection"](receipt["policy_id"])
            if command.get("legacy_choose"):
                tried = agent_tools["choose"](receipt["policy_id"])
                row.update(status="proposed", stop_reason="chosen")
                break
            invalid_key, invalid_count = None, 0
        except Exception as exc:  # noqa: BLE001 -- genuine tool/validation errors return to this model
            if command is None:
                previous_command = {"unparsed": True, "raw_sha": content_id(raw), "raw_preview": raw[:400]}
            error = {"type": type(exc).__name__, "message": str(exc)[:4000]}
            summary = {'op': command.get('op') if command else 'parse', 'type': type(exc).__name__,
                       'message': str(exc).split('\n')[0][:240]}
            previous = next((e for e in flow['errors'] if all(e[k] == v for k, v in summary.items())), None)
            if previous:
                previous['count'] += 1
            elif len(flow['errors']) < 8:
                flow['errors'].append({**summary, 'count': 1})
            else:
                flow['other_errors'] += 1
            audit["attempts"].append({"raw": raw, "reason": error["message"], "error": error})
            feedback({"error": error, "previous_command": previous_command,
                      "instruction": "Repair the previous command's arguments using this error, or trial/choose/stop within the remaining budgets."}, "error")
            key = content_id({"op": command.get("op"), "args": command.get("args")}) if command else None
            invalid_count = invalid_count + 1 if key is not None and key == invalid_key else 1
            invalid_key = key
            if key is not None and invalid_count >= 3:
                reason = "identical invalid tool arguments repeated three times without new evidence"
                tried = _none(reason, needs=("tool_protocol",))
                row.update(status="abstained", reason=reason, stop_reason="repeated_invalid_command")
                break
        persist()
    if tried is None:
        return finish("budget exhausted: maximum model calls reached without choose/stop")
    audit["messages"].append({"role": "assistant", "content": audit["raw"]})
    persist()
    return tried, row
