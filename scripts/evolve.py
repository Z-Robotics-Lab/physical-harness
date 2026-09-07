"""Evolve: an LLM agent repairs the task's driver card against measured failures.

One round = one agent session over a WORKING COPY of the card package that drives
the task (``plugins.embodiment_robocasa`` for the kitchen missions). The agent reads
the card's source, edits it, runs single development seeds in the simulator and
reads what happened (milestone trail, stall geometry, failure keyframes), then calls
``finish``. ``finish`` runs the full paired development suite against the incumbent;
the edit is accepted iff at least one frozen verify milestone (or task success) is
newly gained on some seed and none regresses on any seed. Accepted copies become
the next round's starting point. A markdown notebook of every round (hypothesis,
diff, measured outcome) is the memory the next round reads -- never raw traces.

The verify predicates and the mission planner are frozen and outside the copy;
``predicates.py`` inside the copy is refused. The harness never trains weights.

Process shape: the runtime spawns this script (``--mode evolution``); every suite
runs in a CHILD ``python scripts/evolve.py --suite <spec>`` whose import machinery
maps the card package onto the working copy (``PH_MODULE_OVERLAY``), so any edit in
the copy is exactly what runs -- no class-swap tricks, no stale modules, and a
crashing candidate cannot take the loop down.
"""

from __future__ import annotations

import argparse
import base64
import copy
import difflib
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

#: ``{"plugins.embodiment_robocasa": "/abs/workspace"}`` -- the child maps these
#: packages onto directories BEFORE anything imports them.
OVERLAY_ENV = "PH_MODULE_OVERLAY"


class _Overlay(importlib.abc.MetaPathFinder):
    def __init__(self, mapping: dict) -> None:
        self.mapping = mapping

    def find_spec(self, name, path=None, target=None):
        root = self.mapping.get(name)
        if root is None:
            return None
        return importlib.util.spec_from_file_location(
            name, Path(root) / "__init__.py", submodule_search_locations=[str(root)])


def install_overlay(mapping: dict) -> None:
    sys.meta_path.insert(0, _Overlay(dict(mapping)))


if os.environ.get(OVERLAY_ENV):   # the child, before the heavy imports below
    install_overlay(json.loads(os.environ[OVERLAY_ENV]))

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from board import store as bs
from harness import media
from harness.config import sha_json
from harness.definitions import CAPABILITIES
from harness.events import SessionLog
from harness.kernel import Kernel
from harness.manifest import discover, mount_params
from harness.registry import load_provider
from plugins.task import workload
from scripts import harness_runtime as hr
from scripts.brief_drop import drop
from scripts.rsi_campaign import _maybe_arm_frames

MODES = ("execution", "evolution")
#: JSON ``{provider ref: {param: value}}`` merged over a card's mount params by
#: ``harness.manifest.mount_params`` -- how a tunable reaches a driver.
OVERRIDE_ENV = "PH_MOUNT_PARAMS_OVERRIDE"
ENDPOINT_REF = "plugins.model_endpoint:provider"
FAKE_REF = "plugins.model_endpoint:fake_provider"
#: Full round rows kept in campaign.json; older rounds live in ``rounds/<n>.json``.
ROUNDS_KEPT = 20
#: Rejected working copies kept on disk beyond the incumbent lineage.
WORKSPACES_KEPT = 10
MIN_FREE_BYTES = 5 * 1024 ** 3
#: Files of the copy the agent may not touch: the reward, and the manifest whose
#: ``[tunables]`` the stock card (not the copy) is read from anyway.
PROTECTED = ("predicates.py", "manifest.toml")
MAX_LOG_LINES = 60
READ_MAX_LINES = 200
MAX_CONTEXT_CHARS = 120_000   # ~35k tokens per call; the round's conversation is resent on every call
MAX_IMAGES = 6


# ── the model endpoint ────────────────────────────────────────────────────────────

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
    if os.environ.get("PH_MODEL_ENDPOINT_FAKE"):
        return load_provider(FAKE_REF, {})
    params, _ = model_request_config(model)
    params.setdefault("timeout", 300.0)   # a thinking reply can take minutes; the card's default is 60 s
    return load_provider(ENDPOINT_REF, params)


# ── the seed suite (runs in the CHILD process) ────────────────────────────────────

def node_group(node: dict, graph: dict) -> str:
    """The sub-task label a plan node belongs to (``task`` on trail rows)."""
    return str(node.get("task") or node["id"]) if graph.get("tasks") else str(node["id"]).split("-", 1)[0]


class _Tap(SessionLog):
    """The per-seed ledger as a node trail: ``task.plan`` sets ``nodes`` (plan order,
    ``ok`` None), each ``task.verify`` fills that node's ``ok`` / ``steps`` /
    ``failure_mode``. ``on_change(nodes)`` fires at every change."""

    def __init__(self, on_change) -> None:
        super().__init__()
        self._on, self.nodes = on_change, []

    def append(self, kind: str, data: dict) -> int:
        seq = super().append(kind, data)
        if kind == "task.plan" and data.get("graph"):
            done = {n["id"]: n for n in self.nodes if n["ok"] is True}   # a replan keeps verified nodes
            self.nodes = [{**(done.get(n["id"]) or {"id": n["id"], "skill": n.get("skill"), "ok": None,
                                                    "steps": None, "failure_mode": None}),
                           "after": list(n.get("after") or []), "kind": n.get("kind", "manipulate"),
                           "task": node_group(n, data["graph"])}
                          for n in data["graph"].get("nodes") or []]
        elif kind == "task.verify" and (hit := [n for n in self.nodes if n["id"] == data.get("node")]):
            hit[0].update(ok=all((data.get("results") or {}).values()), steps=data.get("steps"),
                          failure_mode=(data.get("diagnostics") or {}).get("failure_mode"))
        else:
            return seq
        self._on(copy.deepcopy(self.nodes))
        return seq


def _log_excerpt(seed: int, rows, dead: str | None, budget: int) -> list[str]:
    """The dying node's ``task.fault`` / ``task.verify`` rows of one seed."""
    out = []
    for r in rows:
        if r["kind"] == "task.fault" or (r["kind"] == "task.verify" and r["data"].get("node") == dead):
            out.append(f"seed {seed} {r['kind']} " + json.dumps(r["data"], sort_keys=True, default=str)[:400])
    return out[-budget:]


def _link_upstream(trail: list, dead, skills: dict) -> None:
    """The first-death row's ``upstream``: the SEGMENT that ran just before it (which
    leg parked the base where the dying stage found it)."""
    i = next((k for k, n in enumerate(trail) if n["id"] == dead), None) if dead else None
    if i is None:
        return
    up = next((n for n in reversed(trail[:i]) if n.get("kind") == "segment" and n.get("steps")), None)
    if up is not None:
        trail[i]["upstream"] = {"node": up["id"], "skill": skills.get(up["id"]),
                                "steps": up["steps"], "trace_end": up.get("trace_end")}


def _get(budgets, binding: dict, key: str, default):
    v = (budgets or {}).get(key)
    return binding.get(key, default) if v is None else v


def run_suite(task: str, binding: dict, seeds: list[int], arm: str, skills_root: Path,
              tunables: dict, media_dir: Path | None = None, budgets: dict | None = None,
              progress=None, media_prefix: str = "media", cancelled=None, episode: bool = False) -> dict:
    """{count, seeds: {seed: {success, first_death, failure_mode, trail, ...}}, sha,
    elapsed_s, logs}. ``tunables`` = ``{provider ref: {param: value}}`` (OVERRIDE_ENV)."""
    os.environ[OVERRIDE_ENV] = json.dumps(tunables or {})
    per, logs = {}, []
    brief = {**hr.task_brief(task, binding), "arm": arm}
    if media_dir is not None:
        brief["media_dir"] = str(media_dir)
        brief["media_episode"] = bool(episode)   # whole-episode video (the retest: the round's rollout)
    tick = progress or (lambda **kw: None)
    t_suite = time.time()
    for i, seed in enumerate(seeds):
        t_seed = time.time()
        tick(seed_index=i, seed=seed, seeds_total=len(seeds), node=None, nodes=[], seed_started_at=t_seed)
        log = _Tap(lambda nodes: tick(nodes=nodes, node=next((n["id"] for n in nodes if n["ok"] is not True), None)))
        kernel = Kernel(CAPABILITIES, log=log)
        kernel.mount(hr._mount_plan(binding, skills_root, frames=_maybe_arm_frames()))
        out = workload.run(dict(brief), kernel, seed=seed,
                           max_replans=int(_get(budgets, binding, "max_replans", 3)),
                           max_actuations=int(_get(budgets, binding, "max_actuations", 3)),
                           segment_retries=int(binding.get("segment_retries", 0)), cancelled=cancelled)
        skills = {}
        for r in log.rows():
            if r["kind"] == "task.plan" and r["data"].get("graph"):
                skills.update({n["id"]: n["skill"] for n in r["data"]["graph"].get("nodes") or []})
        nodes, faults = out["nodes"], out.get("faults") or []
        dead = next((nid for nid, n in nodes.items() if not n["success"]), None)
        row = {"success": bool(out["success"]), "elapsed_s": round(time.time() - t_seed, 1),
               "trail": [{k: n[k] for k in ("id", "ok", "steps", "failure_mode", "after", "kind", "task")}
                         for n in log.nodes],
               "first_death": dead,
               "failure_mode": (nodes[dead].get("diagnostics") or {}).get("failure_mode") if dead else None,
               "fault": {k: faults[0].get(k) for k in ("kind", "node", "msg")} if faults else None,
               "keyframes": [f"{media_prefix}/{task}/{seed}/{f}" for f in
                             (media.dropped_of(media_dir, task, seed).get(dead) or {}).get("keyframes", [])]
               if media_dir is not None and dead else [],
               "nodes": {nid: {"skill": skills.get(nid), "success": bool(n["success"]),
                               "executor": n.get("executor") or "scripted",
                               "tunables_sha": (n.get("diagnostics") or {}).get("tunables_sha")}
                         for nid, n in nodes.items()}}
        for n in row["trail"]:   # final state from the result: a replan reset the live trail
            r = nodes.get(n["id"]) or {}
            diag = r.get("diagnostics") or {}
            if n["ok"] is None and "success" in r:
                n["ok"] = bool(r["success"])
            n["steps"] = n["steps"] if n["steps"] is not None else r.get("steps")
            if "failure_mode" in diag or n["failure_mode"] is not None:
                n["failure_mode"] = n["failure_mode"] or diag.get("failure_mode")
            else:
                del n["failure_mode"]   # never measured is not "no stall"
            if (diag.get("trace") or {}).get("end"):
                n["trace_end"] = diag["trace"]["end"]
            if diag.get("trace"):
                n["trace"], n["geometry"] = diag["trace"], diag.get("geometry")
        _link_upstream(row["trail"], dead, skills)
        per[str(seed)] = row
        logs += _log_excerpt(seed, log.rows(), dead, MAX_LOG_LINES // len(seeds))
        tick(per_seed_partial=per_seed({"seeds": per}))
    return {"count": sum(s["success"] for s in per.values()), "seeds": per, "sha": sha_json(per),
            "elapsed_s": round(time.time() - t_suite, 3), "logs": logs}


def per_seed(suite: dict) -> list[dict]:
    """The operator-facing per-seed summary sealed with every round: ``[{seed, success,
    first_death, failure_mode, elapsed_s, nodes: trail}]``."""
    return [{"seed": int(seed), **{k: s.get(k) for k in ("success", "first_death", "failure_mode", "elapsed_s")},
             "nodes": s.get("trail") or [],
             "tunables_sha": (s["nodes"].get(s["first_death"]) or {}).get("tunables_sha")
             if s.get("first_death") and s.get("nodes") else None}
            for seed, s in suite["seeds"].items()]


def _child_main(spec_path: Path) -> int:
    """``--suite <spec.json>``: run one suite, stream progress as ``@@{json}`` lines,
    write the result to ``spec.out``."""
    spec = json.loads(Path(spec_path).read_text())
    binding = discover().task_bindings[spec["task"]]
    marker = Path(spec["cancel_marker"]) if spec.get("cancel_marker") else None

    def progress(**kw):
        print("@@" + json.dumps(kw, default=str), flush=True)

    out = run_suite(spec["task"], binding, [int(s) for s in spec["seeds"]], spec["arm"],
                    Path(spec["skills_root"]), spec["tunables"],
                    media_dir=Path(spec["media_dir"]) if spec.get("media_dir") else None,
                    budgets=spec.get("budgets"), progress=progress, media_prefix=spec.get("media_prefix", "media"),
                    cancelled=(lambda: marker.exists()) if marker else None, episode=bool(spec.get("episode")))
    Path(spec["out"]).write_text(json.dumps(out, default=str))
    return 0


# ── score: frozen milestones, paired seeds ─────────────────────────────────────────

def milestones(row: dict) -> dict[str, bool]:
    """``{node: passed}`` over the seed's verify nodes (every node when the plan has no
    verify kind: those trails' ``ok`` IS the oracle's verify row) plus ``task``."""
    trail = row.get("trail") or []
    ver = [n for n in trail if n.get("kind") == "verify"] or trail
    m = {n["id"]: n.get("ok") is True for n in ver}
    m["task"] = bool(row.get("success"))
    return m


def compare(before: dict, after: dict) -> dict:
    """Paired, per seed: gains = milestones newly passed, regressions = milestones lost.
    Accepted iff any gain and no regression on ANY seed. Missing after-rows regress."""
    gains, losses = [], []
    for seed, b in before["seeds"].items():
        mb, ma = milestones(b), milestones((after.get("seeds") or {}).get(seed) or {})
        for k in sorted(set(mb) | set(ma)):
            if ma.get(k) and not mb.get(k):
                gains.append(f"{seed}:{k}")
            elif mb.get(k) and not ma.get(k):
                losses.append(f"{seed}:{k}")
    return {"gains": gains, "regressions": losses, "accepted": bool(gains) and not losses}


def score(suite: dict) -> list:
    """``[successes, mean milestone pass fraction]`` -- the chart's two numbers."""
    rows = list(suite["seeds"].values())
    frac = [sum(m.values()) / len(m) for m in map(milestones, rows) if m]
    return [sum(bool(r.get("success")) for r in rows), round(sum(frac) / len(frac), 4) if frac else 0.0]


# ── evidence as the agent reads it ─────────────────────────────────────────────────

def _f(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "?"


def _node_line(n: dict) -> str:
    mark = {True: "ok", False: "FAIL", None: "-"}[n.get("ok")]
    s = f"{n['id']} {mark}"
    if n.get("steps") is not None:
        s += f" {n['steps']}st"
    if n.get("failure_mode"):
        s += f" {n['failure_mode']}"
    end = n.get("trace_end") or {}
    if end and n.get("ok") is not True:
        s += (f" [eef→target {_f(end.get('d_eef_target'))} m, base→target {_f(end.get('d_base_target'))} m;"
              f" eef {end.get('eef')} target {end.get('target')} base {end.get('base')}]")
    elif end:
        s += f" [d_eef {_f(end.get('d_eef_target'))} d_base {_f(end.get('d_base_target'))}]"
    return s


def describe_seed(seed, s: dict, baseline_row: dict | None = None) -> str:
    trail = s.get("trail") or []
    ran = [n for n in trail if n.get("ok") is not None]
    left = len(trail) - len(ran)
    head = (f"seed {seed}: {'SUCCESS' if s.get('success') else 'FAIL'}"
            + (f", first death {s['first_death']}" if s.get("first_death") else "")
            + (f" ({s['failure_mode']})" if s.get("failure_mode") else "") + f", {s.get('elapsed_s')} s")
    lines = [head, "  " + " · ".join(_node_line(n) for n in ran) + (f" · ({left} nodes not reached)" if left else "")]
    dead = next((n for n in trail if n.get("id") == s.get("first_death")), None)
    if dead and dead.get("upstream"):
        u = dead["upstream"]
        lines.append(f"  upstream segment {u['node']} ended at base {(u.get('trace_end') or {}).get('base')} "
                     f"({u.get('steps')} steps)")
    if dead and dead.get("geometry"):
        lines.append(f"  target geometry: {json.dumps(dead['geometry'], default=str)[:400]}")
    if s.get("fault"):
        lines.append(f"  fault: {json.dumps(s['fault'], default=str)[:300]}")
    if baseline_row is not None:
        c = compare({"seeds": {"x": baseline_row}}, {"seeds": {"x": s}})
        lines.append("  vs incumbent on this seed: "
                     + (f"gained {[g.split(':', 1)[1] for g in c['gains']]} " if c["gains"] else "")
                     + (f"LOST {[g.split(':', 1)[1] for g in c['regressions']]}" if c["regressions"] else "")
                     + ("no milestone change" if not c["gains"] and not c["regressions"] else ""))
    return "\n".join(lines)


def describe_suite(suite: dict, baseline: dict | None = None, logs: bool = True) -> str:
    out = [describe_seed(seed, s, (baseline or {}).get("seeds", {}).get(seed) if baseline else None)
           for seed, s in suite["seeds"].items()]
    fails: dict = {}
    for seed, s in suite["seeds"].items():
        if not s.get("success"):
            fails.setdefault((s.get("first_death"), s.get("failure_mode")), []).append(seed)
    if len(fails) > 1 or any(len(v) > 1 for v in fails.values()):
        out.append("failure clusters (first death, mode → seeds): "
                   + "; ".join(f"{k[0]} {k[1] or ''} → {v}" for k, v in fails.items()))
    if logs and suite.get("logs"):
        out.append("log excerpt (fault / verify rows of the dying node):\n  " + "\n  ".join(suite["logs"][-8:]))
    return "\n".join(out)


def _image_part(path: Path) -> dict | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}}


def keyframe_parts(session: Path, suite: dict, limit: int = MAX_IMAGES) -> list[dict]:
    """Failure keyframes of each seed's first-death node, as image content parts."""
    parts = []
    for seed, s in suite["seeds"].items():
        for rel in s.get("keyframes") or []:
            if len(parts) >= limit:
                return parts
            if part := _image_part(session / rel):
                parts.append({"type": "text", "text": f"[seed {seed} {s.get('first_death')} keyframe {Path(rel).name}]"})
                parts.append(part)
    return parts


def _series(suite: dict, seed, node) -> list:
    row = next((n for n in (suite["seeds"].get(str(seed)) or {}).get("trail") or [] if n.get("id") == node), None)
    tr = (row or {}).get("trace")
    return (tr.get("series") or []) if isinstance(tr, dict) else []


# ── the working copy ───────────────────────────────────────────────────────────────

def card_package(binding: dict) -> tuple[str, Path]:
    """(import name, directory) of the package the task's policy provider lives in."""
    mod = binding["policy"].partition(":")[0]
    pkg = mod.rpartition(".")[0]
    if not pkg:
        raise ValueError(f"evolve needs the task's policy provider inside a package, got {binding['policy']!r}")
    m = importlib.import_module(pkg)
    if not getattr(m, "__file__", None):
        raise ValueError(f"{pkg} has no source directory to copy")
    return pkg, Path(m.__file__).resolve().parent


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Workspace:
    """A copy of the card package the agent edits. ``stock`` is the installed card
    (the protected files must stay byte-identical to it)."""

    def __init__(self, path: Path, stock: Path, parent: Path | None = None) -> None:
        # absolute throughout: the tools accept relative OR absolute paths inside the copy
        self.path, self.stock = Path(path).resolve(), Path(stock).resolve()
        self.parent = Path(parent).resolve() if parent else self.stock   # what ``diff``/``changed`` compare against

    @classmethod
    def create(cls, path: Path, source: Path, stock: Path) -> Workspace:
        if path.exists():
            shutil.rmtree(path)
        shutil.copytree(source, path, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        return cls(path, stock, source)

    def _resolve(self, rel: str) -> Path:
        p = (self.path / rel).resolve()
        if not p.is_relative_to(self.path) or p == self.path:
            raise ValueError(f"path must name a file inside the card copy (e.g. {self.path.name}/drivers.py or just "
                             f"drivers.py), got {rel!r}")
        return p

    def files(self) -> list[str]:
        out = []
        for p in sorted(self.path.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                rel = str(p.relative_to(self.path))
                out.append(f"{rel} ({p.stat().st_size} B{', protected' if p.name in PROTECTED else ''})")
        return out

    def read(self, rel: str, start: int = 1, end: int | None = None) -> str:
        lines = self._resolve(rel).read_text().split("\n")
        a = max(1, int(start or 1))
        b = min(len(lines), int(end) if end else len(lines), a + READ_MAX_LINES - 1)   # bounded: read again for more
        if a > len(lines):
            raise ValueError(f"{rel} has {len(lines)} lines")
        body = "\n".join(f"{i:4d}| {lines[i - 1]}" for i in range(a, b + 1))
        return body + ("" if b >= len(lines) else f"\n... ({len(lines) - b} more lines; read with start={b + 1})")

    def grep(self, pattern: str, path: str | None = None) -> str:
        rx = re.compile(pattern)
        hits = []
        files = [self._resolve(path)] if path else sorted(self.path.rglob("*.py"))
        for p in files:
            if "__pycache__" in p.parts or not p.is_file():
                continue
            for i, line in enumerate(p.read_text().split("\n"), 1):
                if rx.search(line):
                    hits.append(f"{p.relative_to(self.path)}:{i}: {line.strip()[:160]}")
        return "\n".join(hits[:80]) + (f"\n... {len(hits) - 80} more" if len(hits) > 80 else "") or "no match"

    def edit(self, rel: str, old: str, new: str) -> str:
        p = self._resolve(rel)
        if p.name in PROTECTED:
            raise ValueError(f"{rel} is frozen (the reward and the stock manifest); edit the drivers instead")
        if p.suffix != ".py":
            raise ValueError("only .py files take effect in the copy")
        text = p.read_text()
        if old == new:
            raise ValueError("old == new changes nothing")
        n = text.count(old) if old else 0
        if n == 1:
            return self._write(p, text.replace(old, new, 1), rel)
        # retyped snippet: the same lines ignoring indentation and trailing whitespace,
        # matching exactly once, with ``new`` re-indented to the file
        lines, want = text.split("\n"), [l.strip() for l in old.split("\n")]
        strip = [l.strip() for l in lines]
        hits = [i for i in range(len(lines) - len(want) + 1) if strip[i:i + len(want)] == want] if old.strip() else []
        if len(hits) == 1:
            i = hits[0]
            j = next((k for k in range(len(want)) if want[k]), 0)
            delta = (len(lines[i + j]) - len(lines[i + j].lstrip())) - (len(old.split("\n")[j]) - len(old.split("\n")[j].lstrip()))
            shifted = [(" " * delta + l if delta > 0 else l[min(-delta, len(l) - len(l.lstrip())):]) if l.strip() else l
                       for l in new.split("\n")]
            return self._write(p, "\n".join(lines[:i] + shifted + lines[i + len(want):]), rel)
        first = next((l for l in want if l), "")
        near = next((k for k, l in enumerate(strip) if l == first), None) if first else None
        hint = ("" if near is None else f" Its first line occurs at line {near + 1}; the file there reads:\n"
                + "\n".join(f"{k + 1:4d}| {lines[k]}" for k in range(max(0, near - 3), min(len(lines), near + len(want) + 3))))
        raise ValueError(f"`old` must occur exactly once in {rel}; it occurs {n} times (also ignoring indentation). "
                         f"Copy it verbatim out of a read result, without the `NNNN| ` prefixes.{hint}")

    def write(self, rel: str, content: str) -> str:
        p = self._resolve(rel)
        if p.name in PROTECTED or p.suffix != ".py":
            raise ValueError("only new/replaced .py files, never predicates.py or manifest.toml")
        p.parent.mkdir(parents=True, exist_ok=True)
        return self._write(p, content, rel)

    def _write(self, p: Path, text: str, rel: str) -> str:
        try:
            compile(text, rel, "exec")
        except SyntaxError as exc:
            raise ValueError(f"{rel} would not compile: {exc}") from None
        p.write_text(text)
        return f"{rel} written ({len(text.split(chr(10)))} lines)"

    def digest(self) -> str:
        """Content identity of the copy's .py files (what a run would execute)."""
        return sha_json({str(q.relative_to(self.path)): _sha(q) for q in sorted(self.path.rglob("*.py"))
                         if "__pycache__" not in q.parts})

    def protected_ok(self) -> str | None:
        for name in PROTECTED:
            a, b = self.path / name, self.stock / name
            if b.exists() and (not a.exists() or _sha(a) != _sha(b)):
                return f"{name} differs from the installed card; it is frozen"
        return None

    def diff(self, against: Path | None = None, limit: int = 160) -> str:
        """Unified diff of the copy against its parent (the incumbent's copy or the stock)."""
        against = self.parent if against is None else against
        out = []
        names = {p.relative_to(self.path) for p in self.path.rglob("*.py") if "__pycache__" not in p.parts}
        names |= {p.relative_to(against) for p in Path(against).rglob("*.py") if "__pycache__" not in p.parts}
        for rel in sorted(names):
            a, b = Path(against) / rel, self.path / rel
            ta = a.read_text().splitlines() if a.exists() else []
            tb = b.read_text().splitlines() if b.exists() else []
            if ta != tb:
                out += list(difflib.unified_diff(ta, tb, f"a/{rel}", f"b/{rel}", lineterm="", n=2))
        if len(out) > limit:
            out = out[:limit] + [f"... ({len(out) - limit} more diff lines)"]
        return "\n".join(out)

    def changed(self) -> bool:
        return bool(self.diff(limit=10 ** 9))


# ── the notebook: what crosses rounds ─────────────────────────────────────────────

class Notebook:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, text: str) -> None:
        with self.path.open("a") as f:
            f.write(text.rstrip("\n") + "\n\n")

    def text(self, limit: int = 12_000, diffs: int = 2) -> str:
        """The brief's view of the notebook: full entries for the last ``diffs`` rounds, older
        entries without their diff block (accepted diffs live on in the incumbent's code),
        the oldest collapsed to their header line once ``limit`` is reached. Resent on every
        model call, so it is the one piece of context worth keeping small."""
        if not self.path.exists():
            return "(no previous rounds)"
        entries = re.split(r"(?m)^(?=## Round )", self.path.read_text())
        entries = [re.sub(r"```diff\n.*?\n```\n?", "- diff: (see the incumbent's code)\n", e, flags=re.DOTALL)
                   if i < len(entries) - diffs else e for i, e in enumerate(entries)]
        body = "".join(entries)
        if len(body) <= limit:
            return body
        kept, size = [], 0
        for e in reversed(entries):
            if size + len(e) > limit:
                kept.append(e.split("\n", 1)[0] + "  (details elided)\n")
            else:
                kept.append(e)
                size += len(e)
        return "".join(reversed(kept))


# ── the agent: one round ────────────────────────────────────────────────────────────

SYSTEM = """You are a robotics engineer improving a scripted robot driver card in simulation.
You have a WORKING COPY of the card's Python package {pkg}; whatever you edit there is
exactly what the simulator runs when you call run/finish. Verification predicates
(predicates.py) and the mission planner are frozen: success is measured by them, never
by anything you write. Do not read simulator internals the installed drivers do not
already read.

Method (one hypothesis at a time):
1. Read the evidence: the milestone trail per seed, where the first death is, the stall
   geometry (eef/base vs target), the upstream segment that parked the base, keyframes.
2. Localise the cause at the highest layer that explains it (a recovery or approach
   decision before a numeric knob). Read the code that produced the observed numbers.
3. Make the smallest edit that tests the hypothesis; run the failing seed; read the
   result; iterate. Do not repeat an experiment the notebook already records.
4. When the failing seed gains its milestone without losing any, call finish. finish
   runs EVERY development seed paired against the incumbent; the edit is accepted only
   if some milestone or task success is newly gained and none regresses on any seed.

Reply with ONE JSON object per turn, no prose outside it. Inside JSON strings escape
double quotes (or quote code with single quotes); an unparsable reply wastes an action.
To save round trips send several actions at once as {{"actions": [{{...}}, {{...}}]}}: reads,
greps and edits run in order and return together; the batch stops at its first error or
at a run. Keep "thought" to two sentences; the notebook, not the chat, is your memory.
  {{"thought": "...", "action": "read", "path": "drivers.py", "start": 1, "end": 120}}
  {{"action": "grep", "pattern": "<regex over the copy's .py files>", "path": "<optional one file>"}}
  {{"action": "edit", "path": "<file>", "old": "<snippet occurring exactly once>", "new": "<replacement>"}}
  {{"action": "write", "path": "<new_or_whole_file>.py", "content": "..."}}
  {{"action": "tunable", "name": "<declared knob>", "value": <number>}}
  {{"action": "trace", "seed": <seed>, "node": "<node id>"}}   per-step motion series of that node in its last run
  {{"action": "run", "seed": <development seed>}}                one episode of the copy on that seed
  {{"action": "finish", "summary": "<what changed and why, <=600 chars>"}}
  {{"action": "give_up", "reason": "..."}}
Budget this round: {max_steps} actions, {max_probes} single-seed runs. finish is free.
"""


def _text(content) -> str:
    return content if isinstance(content, str) else " ".join(
        p.get("text", "[image]") for p in content if isinstance(p, dict) and p.get("type") != "image_url") or "[image]"


def _parse_actions(raw: str) -> list[dict]:
    """One action ``{"action": ...}`` or a batch ``{"actions": [...]}`` / ``[...]``."""
    try:
        v = json.loads(raw)
    except ValueError:
        i = min((k for k in (raw.find("{"), raw.find("[")) if k >= 0), default=-1)
        if i < 0:
            raise ValueError("reply contains no JSON object")
        v, _ = json.JSONDecoder().raw_decode(raw[i:])
    if isinstance(v, dict) and isinstance(v.get("actions"), list):
        v = v["actions"]
    batch = v if isinstance(v, list) else [v]
    if not batch or any(not isinstance(a, dict) or not isinstance(a.get("action"), str) for a in batch):
        raise TypeError('reply must be a JSON object with an "action" string, or {"actions": [...]} of them')
    return batch


class Agent:
    """One round's tool loop. ``run_seed(seeds, label) -> suite`` is the simulator."""

    def __init__(self, ep, ws: Workspace, *, pkg: str, tunables: dict, knobs: dict, run_seed,
                 dev_seeds: list[int], baseline: dict, session: Path, notebook: str, proposal: dict | None,
                 max_steps: int, max_probes: int, max_tokens: int, options: dict, cancelled, audit_path: Path,
                 tick, frontier: str | None = None) -> None:
        self.frontier = frontier
        # ``tunables`` = the OVERRIDE_ENV document, keyed by the card PACKAGE so it reaches
        # every provider the card hosts (harness.manifest.mount_params); ``knobs`` = the
        # effective numeric values the agent sees and may change.
        self.ep, self.ws, self.pkg, self.tunables, self.knobs = ep, ws, pkg, tunables, knobs
        self.knobs_from = dict(knobs)
        self.run_seed, self.dev_seeds, self.baseline, self.session = run_seed, dev_seeds, baseline, session
        self.max_steps, self.max_probes, self.max_tokens, self.options = max_steps, max_probes, max_tokens, options
        self.cancelled, self.audit_path, self.tick = cancelled, audit_path, tick
        self.images = bool(getattr(ep, "images", False))
        self.last_runs: dict[str, dict] = {}   # seed -> the last suite that ran it (baseline first)
        for seed in baseline["seeds"]:
            self.last_runs[seed] = baseline
        self.probes: list[dict] = []
        self.ran: dict[int, tuple] = {}   # seed -> (copy digest, knobs) of its last run
        self.usage = {"prompt": 0, "completion": 0, "cache_hit": 0}
        self.calls = 0
        self.actions: dict[str, int] = {}
        self.finishes: dict[str, int] = {}   # finish_reason counts ("length" = the answer was cut off)
        self.errors: list[str] = []
        self.messages = [{"role": "system", "content": SYSTEM.format(pkg=pkg, max_steps=max_steps, max_probes=max_probes)},
                         {"role": "user", "content": self._brief(notebook, proposal)}]
        self.raw: list[str] = []

    # -- prompt -------------------------------------------------------------------
    def _brief(self, notebook: str, proposal: dict | None):
        frontier = self.frontier or ""
        text = [f"# Task: {self.baseline['task']}  (development seeds {self.dev_seeds}; arm {self.baseline.get('arm')})",
                "Card copy files:\n  " + "\n  ".join(self.ws.files()),
                "Declared tunables (current effective values; change with the tunable action): "
                + json.dumps(self.knobs, sort_keys=True),
                "## Incumbent on the development seeds (what you must beat)\n" + describe_suite(self.baseline),
                *([frontier] if frontier else []),
                "## Notebook of previous rounds\n" + notebook]
        if proposal:
            text.append("## Operator proposal pending -- evaluate it first\n" + json.dumps(proposal, ensure_ascii=False)[:2000])
        text.append("Start by reading the code behind the first death, then state your hypothesis in `thought`.")
        parts = [{"type": "text", "text": "\n\n".join(text)}]
        if self.images:
            parts += keyframe_parts(self.session, self.baseline)
        return parts if len(parts) > 1 else parts[0]["text"]

    def _user(self, text: str, images: list | None = None) -> None:
        if images and self.images:
            # keyframes ride only the NEWEST message: every earlier image becomes a one-line
            # placeholder, or each call would resend every frame the round has ever shown
            for m in self.messages[1:]:
                if isinstance(m.get("content"), list):
                    m["content"] = [{"type": "text", "text": "[keyframe shown earlier]"} if p.get("type") == "image_url" else p
                                    for p in m["content"]]
            self.messages.append({"role": "user", "content": [{"type": "text", "text": text}, *images]})
        else:
            self.messages.append({"role": "user", "content": text})
        self._bound()

    def _bound(self) -> None:
        """Keep the conversation under MAX_CONTEXT_CHARS. Elision rewrites history, which
        invalidates the server's cached prefix for the next call, so when the cap is hit the
        oldest tool results are collapsed down to HALF the cap at once, not one at a time."""
        size = lambda: sum(len(json.dumps(m.get("content"), default=str)) for m in self.messages)
        if size() <= MAX_CONTEXT_CHARS:
            return
        i = 2
        while size() > MAX_CONTEXT_CHARS // 2 and i < len(self.messages) - 8:
            m = self.messages[i]
            if m["role"] == "user" and not (isinstance(m["content"], str) and m["content"].startswith("[elided")):
                m["content"] = f"[elided earlier result, {len(json.dumps(m['content'], default=str))} chars]"
            i += 1

    def _persist(self, status: str, extra: dict | None = None) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"status": status, "calls": self.calls, "usage": dict(self.usage), "actions": self.actions,
               "finish_reasons": self.finishes,
               "errors": self.errors[-20:], "probes": self.probes,
               "messages": [{"role": m["role"], "content": _text(m["content"])} for m in self.messages],
               **(extra or {})}
        tmp = self.audit_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str))
        os.replace(tmp, self.audit_path)

    # -- the loop ---------------------------------------------------------------------
    def loop(self) -> dict:
        """Returns ``{status: finished|gave_up|exhausted|error|cancelled, summary, reason, error}``."""
        steps, result, empties, plain_retry = 0, None, 0, False
        while result is None:
            if self.cancelled():
                result = {"status": "cancelled", "reason": "cancelled at an agent step"}
                break
            if steps >= self.max_steps:
                result = ({"status": "finished", "summary": "(action budget exhausted; evaluating the edits as they stand)",
                           "reason": "steps_exhausted"} if self.ws.changed() or self.knobs != self.knobs_from
                          else {"status": "exhausted", "reason": "action budget exhausted without an edit"})
                break
            self.tick(phase="propose", llm_calls=self.calls + 1)
            # after a reply the reasoning ate whole (finish_reason=length, empty content) the
            # retry runs WITHOUT thinking: the answer is one JSON object, reasoning is optional
            options = {"thinking": {"type": "disabled"}} if plain_retry else self.options
            raw, failure = None, None
            for attempt, pause in enumerate((0, 10, 30, 60)):   # transient network/API errors: retry, then seal
                if pause:
                    time.sleep(pause)
                try:
                    raw = self.ep.chat(self.messages, max_tokens=self.max_tokens,
                                       response_format={"type": "json_object"}, **options)
                    failure = None
                    break
                except Exception as exc:  # noqa: BLE001 -- the endpoint is infrastructure
                    failure = f"{type(exc).__name__}: {str(exc)[:600]}"
                    self.errors.append(f"model call failed (attempt {attempt + 1}): {failure}"[:300])
                    self._persist("running")
            if failure is not None:
                result = {"status": "error", "error": failure, "reason": "model_error"}
                break
            self.calls += 1
            u = getattr(self.ep, "last_usage", None) or {}
            for k in self.usage:
                self.usage[k] += int(u.get(k) or 0)   # cache_hit stays 0 on servers that report none
            finish = getattr(self.ep, "last_finish", None)
            self.finishes[str(finish)] = self.finishes.get(str(finish), 0) + 1
            if not (raw or "").strip():
                # thinking mode sometimes returns an empty content (or the reasoning ate
                # max_tokens): ask again without spending an action; the retry drops thinking.
                # Three in a row end the ROUND as unproductive, never the campaign.
                empties += 1
                plain_retry = True
                self.errors.append(f"empty reply (finish_reason={finish})")
                if empties >= 3:
                    result = {"status": "exhausted", "reason": f"3 consecutive empty replies (finish_reason={finish})"}
                    break
                self._user(f"error: your reply was empty (finish_reason={finish}). Answer with the JSON object only.")
                self._persist("running")
                continue
            empties, plain_retry = 0, False
            self.raw.append(raw)
            self.messages.append({"role": "assistant", "content": raw})
            steps += 1
            # a batch runs to its end, its first error, or its first run/finish/give_up; every
            # executed action counts against the budget, the results come back as ONE message
            outputs, images = [], []
            try:
                batch = _parse_actions(raw)
            except Exception as exc:  # noqa: BLE001 -- an unparsable reply is feedback, not a crash
                batch, outputs = [], [f"error: {exc}"]
                self.errors.append(outputs[0][:300])
            for k, act in enumerate(batch):
                name = act["action"]
                self.actions[name] = self.actions.get(name, 0) + 1
                steps += 1 if k else 0
                if name == "finish":
                    result = {"status": "finished", "summary": str(act.get("summary") or act.get("thought") or "")[:600],
                              "reason": "finish"}
                    break
                if name == "give_up":
                    result = {"status": "gave_up", "reason": str(act.get("reason") or act.get("thought") or "give_up")[:600]}
                    break
                head = f"[{k + 1}/{len(batch)} {name}] " if len(batch) > 1 else ""
                try:
                    text, images = self._tool(name, act)
                    outputs.append(head + text)
                except Exception as exc:  # noqa: BLE001 -- a wrong action is feedback, not a crash
                    msg = f"{head}error: {exc}"
                    self.errors.append(msg[:300])
                    outputs.append(msg + (" (rest of the batch skipped)" if k + 1 < len(batch) else ""))
                    break
                if name == "run" and k + 1 < len(batch):
                    outputs.append("(read this result before the rest of the batch; it was skipped)")
                    break
            if result is None:
                self._user("\n\n".join(outputs) + f"\n({max(0, self.max_steps - steps)} actions left)", images)
            self._persist("running")
        self._persist(result["status"], result)
        return result

    def _tool(self, name: str, a: dict) -> tuple[str, list]:
        if name == "read":
            return self.ws.read(str(a.get("path", "")), a.get("start", 1), a.get("end")), []
        if name == "grep":
            return self.ws.grep(str(a.get("pattern", "")), a.get("path") or None), []
        if name == "edit":
            return self.ws.edit(str(a.get("path", "")), str(a.get("old", "")), str(a.get("new", ""))), []
        if name == "write":
            return self.ws.write(str(a.get("path", "")), str(a.get("content", ""))), []
        if name == "tunable":
            k, v = a.get("name"), a.get("value")
            if k not in self.knobs or isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"tunable must name one of {sorted(self.knobs)} with a numeric value")
            self.knobs[k] = v
            self.tunables.setdefault(self.pkg, {}).setdefault("tunables", {})[k] = v
            return f"{k} = {v} for the next run (was {self.knobs_from.get(k)})", []
        if name == "trace":
            seed, node = str(a.get("seed")), a.get("node")
            suite = self.last_runs.get(seed)
            if suite is None:
                raise ValueError(f"seed must be one of {self.dev_seeds}")
            rows = _series(suite, seed, node)
            if not rows:
                ids = [n["id"] for n in (suite["seeds"][seed].get("trail") or []) if n.get("trace")]
                raise ValueError(f"no motion trace for {node!r} on seed {seed}; traced nodes: {ids}")
            step = max(1, len(rows) // 60)
            return (f"{node} on seed {seed}: {len(rows)} sampled steps (every {step}th shown)\n"
                    + "\n".join(json.dumps(r, default=str) for r in rows[::step])), []
        if name == "run":
            seed = a.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed not in self.dev_seeds:
                raise ValueError(f"run.seed must be one of the development seeds {self.dev_seeds}")
            if len(self.probes) >= self.max_probes:
                raise ValueError("single-seed run budget exhausted: finish or give_up")
            if why := self.ws.protected_ok():
                raise ValueError(why)
            # the simulator is deterministic per seed: the same code and knobs give the same
            # episode, so a repeat run buys nothing and is refused without spending a probe
            identity = (self.ws.digest(), json.dumps(self.tunables, sort_keys=True))
            if self.ran.get(seed) == identity:
                raise ValueError(f"seed {seed} already ran with exactly this code and these knobs (see its result above); "
                                 "edit something or run another seed")
            self.ran[seed] = identity
            label = f"probe-{len(self.probes)}"
            self.probes.append({"seed": seed, "label": label})
            self.tick(phase="probe", probe_index=len(self.probes) - 1)
            try:
                suite = self.run_seed([seed], label)
            except Exception as exc:  # noqa: BLE001 -- the candidate crashed the episode: evidence
                self.probes[-1]["error"] = str(exc)[:500]
                raise ValueError(f"the episode raised before finishing:\n{str(exc)[-2500:]}") from None
            finally:
                self.tick(phase="propose", llm_calls=self.calls)
            self.last_runs[str(seed)] = suite
            row = suite["seeds"][str(seed)]
            self.probes[-1].update(success=row["success"], first_death=row.get("first_death"),
                                   failure_mode=row.get("failure_mode"),
                                   compare=compare({"seeds": {str(seed): self.baseline["seeds"][str(seed)]}}, suite))
            text = describe_suite(suite, self.baseline)
            return text, keyframe_parts(self.session, suite, 3)
        raise ValueError(f"unknown action {name!r}; use read/grep/edit/write/tunable/trace/run/finish/give_up")


# ── campaign store ──────────────────────────────────────────────────────────────────

_INDEX_KEYS = ("round", "before", "after", "best", "parent", "outcome", "accepted", "accepted_reason",
               "published", "before_score", "after_score", "usage", "proposer", "needs", "confirm",
               "suite_sha", "proposal", "ts", "workspace", "llm", "evaluation")
_SEED_KEYS = ("seed", "success", "first_death", "failure_mode")


def index_row(r: dict) -> dict:
    """The compact stand-in for an archived round (chart numbers precomputed off the
    trails it drops, the same reading board.store._rates makes)."""
    nb, tb = bs._rates(r.get("per_seed"))
    na, ta = bs._rates(r.get("after_seeds"))
    t = r.get("tried") or {}
    return {**{k: r[k] for k in _INDEX_KEYS if k in r}, "sharded": True,
            "tried": {"kind": t.get("kind"), "node": t.get("node"),
                      "detail": {k: (v[:300] if isinstance(v, str) else v) for k, v in (t.get("detail") or {}).items()
                                 if k in ("summary", "reason", "error", "files", "tunables")}},
            "node_rate": {"before": nb, "after": na},
            "by_task": {k: {"before": tb.get(k), "after": ta.get(k)} for k in sorted({*tb, *ta})},
            "per_seed": [{k: s.get(k) for k in _SEED_KEYS} for s in r.get("per_seed") or ()],
            "after_seeds": [{k: s.get(k) for k in _SEED_KEYS} for s in r.get("after_seeds") or ()]}


class EvolveStore:
    """``campaigns/evolve-<task>/campaign.json``, written atomically and BOUNDED: the
    header plus the last ROUNDS_KEPT rounds in full; older rounds are written once to
    ``rounds/<n>.json`` and stand in the file as ``index_row``."""

    def __init__(self, session: Path, task: str) -> None:
        self.dir = session / "campaigns" / f"evolve-{task}"
        self.path = self.dir / "campaign.json"
        self.rounds_dir = self.dir / "rounds"

    def load(self) -> dict | None:
        return json.loads(self.path.read_text()) if self.path.exists() else None

    def round(self, n: int) -> dict | None:
        try:
            return json.loads((self.rounds_dir / f"{int(n)}.json").read_text())
        except (OSError, ValueError):
            return None

    def save(self, doc: dict) -> None:
        rows = doc.get("rounds") or []
        for i, r in enumerate(rows[:-ROUNDS_KEPT]):
            if not r.get("sharded"):
                rows[i] = self._archive(r)
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str))
        os.replace(tmp, self.path)

    def _archive(self, r: dict) -> dict:
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        p = self.rounds_dir / f"{int(r['round'])}.json"
        if not p.exists():
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(r, indent=1, sort_keys=True, default=str))
            os.replace(tmp, p)
        return index_row(r)


def take_proposal(session: Path, task: str, round_no: int) -> dict | None:
    """The oldest pending inbox proposal for ``task``, stamped ``applied`` in place."""
    for p in bs.proposals(session):
        if p["task"] == task and p["applied"] is None:
            path = session / "proposals" / f"{p['id']}.json"
            doc = json.loads(path.read_text())
            doc["applied"] = {"round": round_no, "ts": time.time()}
            drop(path.parent, path.name, json.dumps(doc, sort_keys=True))
            return {**p, "applied": doc["applied"]}
    return None


def _media(session: Path, task: str, seeds: list[int], prefix: str) -> list[str]:
    return [f"{prefix}/{task}/{seed}/{ent['file']}" for seed in seeds
            for ent in media.index_of(session / prefix, task, seed).values()]


def _dropped(session: Path, task: str, seeds: list[int], prefix: str) -> dict[str, dict]:
    return {f"{seed}/{node}": {"reason": d["reason"], "keyframes": [f"{prefix}/{task}/{seed}/{f}" for f in d["keyframes"]]}
            for seed in seeds for node, d in media.dropped_of(session / prefix, task, seed).items()}


def disk_guard(session: Path) -> str | None:
    free = shutil.disk_usage(session).free
    return f"free space {free / 1024 ** 3:.1f} GB < {MIN_FREE_BYTES / 1024 ** 3:.0f} GB" if free < MIN_FREE_BYTES else None


_ZH = {"idle": "等待", "baseline": "基线评测", "propose": "LLM 分析", "probe": "单种子试跑", "retest": "同种子复测",
       "confirm": "新种子确认", "done": "完成", "cancelled": "已取消", "failed": "失败", "paused_disk": "磁盘不足，已暂停"}


def _message(live: dict) -> str:
    head = f"第 {live['round']} 轮 {_ZH.get(live['phase'], live['phase'])}"
    if live["phase"] == "paused_disk":
        return live.get("disk") or head
    if live["phase"] == "failed":
        return f"第 {live['round']} 轮失败：{live.get('error') or '查看模型审计'}"
    if live["phase"] == "done":
        return f"已完成 {live['round']} 轮"
    if live["phase"] == "cancelled":
        return f"第 {live['round']} 轮边界取消"
    if live["phase"] == "propose" and live.get("llm_calls"):
        head += f"（第 {live['llm_calls']} 次模型调用）"
    if live["phase"] == "probe" and live.get("probe_index") is not None:
        head += f"（第 {live['probe_index'] + 1} 次）"
    if live.get("seed") is not None:
        done = sum(n["ok"] is True for n in live.get("nodes") or [])
        head += (f"：种子 {live['seed']} 运行中" + (f" ({live['node']})" if live.get("node") else "")
                 + (f" 节点 {done}/{len(live['nodes'])}" if live.get("nodes") else "")
                 + f"，{live['seed_index'] + 1}/{live['seeds_total']}")
    return head


# ── the round loop (parent) ─────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LLM agent repairs the task's driver card against measured failures")
    ap.add_argument("--mode", choices=MODES, default="execution")
    ap.add_argument("--suite", type=Path, help="(child) run one suite from this spec file")
    ap.add_argument("--task")
    ap.add_argument("--session", type=Path)
    ap.add_argument("--skills-root", type=Path)
    ap.add_argument("--seeds", type=int, nargs=2)
    ap.add_argument("--rounds", type=int, default=3, help="rounds this brief runs; 0 = until cancelled")
    ap.add_argument("--arm", default="auto")
    ap.add_argument("--cancel-marker", type=Path)
    ap.add_argument("--max-replans", type=int)
    ap.add_argument("--max-actuations", type=int)
    ap.add_argument("--confirm-seeds", type=int, default=2,
                    help="fresh development seeds a whole-task gain must also hold on (never installation evidence)")
    ap.add_argument("--llm-model")
    ap.add_argument("--llm-effort", default="off")
    ap.add_argument("--max-steps", type=int, default=40, help="agent actions per round")
    ap.add_argument("--max-probes", type=int, default=6, help="single-seed runs per round")
    ap.add_argument("--max-output-tokens", type=int, default=8192,
                    help="per reply; with a thinking effort the reasoning shares this budget")
    args = ap.parse_args(argv)
    if args.suite:
        return _child_main(args.suite)
    if args.rounds < 0 or args.max_steps < 1 or args.max_probes < 0:
        ap.error("rounds/probes must be nonnegative and steps positive")
    try:
        llm_params, thinking = model_request_config(args.llm_model, args.llm_effort)
    except ValueError as exc:
        ap.error(str(exc))
    if missing := [f"--{k}" for k in ("task", "session", "skills_root") if getattr(args, k) is None]:
        ap.error("required: " + ", ".join(missing))
    if args.mode != "evolution":
        print(json.dumps({"error": f"evolve refused in mode {args.mode!r}; assert --mode evolution"}))
        return 3
    binding = discover().task_bindings.get(args.task)
    if binding is None:
        raise SystemExit(f"no task binding for {args.task!r}")
    pkg, stock = card_package(binding)
    store = EvolveStore(args.session, args.task)
    doc = store.load() or {"task": args.task, "session": args.session.name, "seeds": list(args.seeds or [0, 1]),
                           "arm": args.arm, "rounds": [], "best": 0, "cursor": 0,
                           "incumbent": {"workspace": None, "tunables": {}, "round": 0}}
    for legacy in ("applied", "accepted_stack", "learning_replay", "cycle_budget", "run_budget", "cycle_context",
                   "transfer", "evaluation", "evaluation_contract", "evaluation_contracts", "reference", "reference_plan",
                   "plan_space", "diagnosis", "experience", "memory_prefix", "last_outcome", "prior_protocols",
                   "protocol_id", "working_candidates", "epoch_start", "confirm_base", "continuous"):
        doc.pop(legacy, None)   # the previous loop's state; the console would keep rendering it
    doc.update(status="running", stop_reason=None, card=pkg, llm_config={"model": llm_params.get("model"), "effort": args.llm_effort},
               score_definition="Frozen verify milestones per seed (plus task success), paired on the same development "
                                "seeds: accepted iff some milestone is newly gained and none regresses.")
    seeds, arm = doc["seeds"], doc["arm"]
    # a campaign written by the previous loop has no incumbent: start from the stock card
    incumbent = doc.setdefault("incumbent", {"workspace": None, "tunables": {}, "round": 0})
    if incumbent.get("workspace") and not (args.session / incumbent["workspace"]).is_dir():
        incumbent.update(workspace=None, note="incumbent copy missing on disk; restarted from the stock card")
    notebook = Notebook(store.dir / "notebook.md")
    budgets = {"max_replans": args.max_replans, "max_actuations": args.max_actuations}
    now = time.time()
    live = doc["live"] = {"phase": "idle", "round": doc["cursor"], "seeds_total": seeds[1] - seeds[0] + 1,
                          "seed_index": None, "seed": None, "node": None, "nodes": [], "messages": [], "message": "",
                          "started_at": now, "round_started_at": now, "phase_started_at": now, "seed_started_at": None,
                          "last_round_s": None, "per_seed_partial": [], "tried": None, "proposer": "llm",
                          "sim_s": sum((r.get("usage") or {}).get("sim_s", 0) for r in doc["rounds"])}

    def cancelled() -> bool:
        return args.cancel_marker is not None and args.cancel_marker.exists()

    def on_term(signum, frame):   # the runtime kills the process group on cancel: leave the truth behind
        doc.update(status="cancelled", stop_reason="cancelled")
        live.update(phase="cancelled", message=f"第 {live['round']} 轮被停止")
        store.save(doc)
        raise SystemExit(3)

    signal.signal(signal.SIGTERM, on_term)

    def tick(**kw) -> None:
        if kw.get("phase", live["phase"]) != live["phase"]:
            kw = {"phase_started_at": time.time(), "seed": None, "seed_index": None, "node": None, "nodes": [],
                  "seed_started_at": None, "per_seed_partial": [], **kw}
        live.update(kw)
        msg = _message(live)
        if msg != live["message"]:
            live["message"] = msg
            live["messages"] = (live["messages"] + [{"ts": time.time(), "text": msg}])[-20:]
        store.save(doc)

    round_cost = {"episode_attempts": 0, "sim_s": 0.0}

    def suite(seed_list: list[int], workspace: Path | None, tunables: dict, label: str, media_on: bool = True) -> dict:
        """One suite in a CHILD process mapped onto ``workspace`` (None = the stock card).
        The retest also records each seed's whole-episode video (the round's rollout)."""
        prefix = f"media/rsi/{args.task}/round-{live['round']}/{label}"
        work = store.dir / "work"
        work.mkdir(parents=True, exist_ok=True)
        spec_path, out_path = work / f"suite-{label}.json", work / f"suite-{label}.out.json"
        spec_path.write_text(json.dumps({
            "task": args.task, "seeds": seed_list, "arm": arm, "skills_root": str(args.skills_root),
            "tunables": tunables,
            "media_dir": str(args.session / prefix) if media_on else None, "media_prefix": prefix,
            "episode": label == "retest",
            "budgets": budgets, "out": str(out_path), "cancel_marker": str(args.cancel_marker) if args.cancel_marker else None}))
        env = {**os.environ}
        env.pop(OVERLAY_ENV, None)
        if workspace is not None:
            env[OVERLAY_ENV] = json.dumps({pkg: str(workspace)})
        cmd = [sys.executable, str(Path(__file__).resolve()), "--suite", str(spec_path)]
        t0 = time.monotonic()
        started = 0
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        err_buf: list[str] = []
        threading.Thread(target=lambda: err_buf.append(proc.stderr.read()), daemon=True).start()
        try:
            for line in proc.stdout:
                if line.startswith("@@"):
                    try:
                        kw = json.loads(line[2:])
                    except ValueError:
                        continue
                    started += int(kw.get("seed_started_at") is not None)
                    tick(**kw)
                if cancelled() and proc.poll() is None:
                    proc.terminate()
            proc.wait()
        finally:
            elapsed = time.monotonic() - t0
            round_cost["episode_attempts"] += started
            round_cost["sim_s"] += elapsed
            live["sim_s"] += elapsed
        if cancelled():
            raise RuntimeError("cancelled")
        err = (err_buf[0] if err_buf else "") or ""
        if proc.returncode != 0 or not out_path.exists():
            # the traceback, not the simulator's import warnings that pad stderr around it
            tb = err.rfind("Traceback (most recent call last)")
            raise RuntimeError(f"suite exited {proc.returncode}:\n{(err[tb:] if tb >= 0 else err).strip()[-3000:]}")
        out = json.loads(out_path.read_text())
        out.update(task=args.task, arm=arm, media=_media(args.session, args.task, seed_list, prefix) if media_on else [],
                   media_dropped=_dropped(args.session, args.task, seed_list, prefix) if media_on else {})
        return out

    tick()
    base = None
    completed = 0
    while args.rounds == 0 or completed < args.rounds:
        if cancelled():
            doc.update(status="cancelled", stop_reason="cancelled")
            tick(phase="cancelled")
            return 3
        if why := disk_guard(args.session):
            doc.update(status="paused_disk", stop_reason="paused_disk")
            tick(phase="paused_disk", disk=why)
            return 4
        rnd, started = int(doc["cursor"]) + 1, time.time()
        round_cost = {"episode_attempts": 0, "sim_s": 0.0}
        inc_dir = (args.session / incumbent["workspace"]) if incumbent.get("workspace") else None
        tick(phase="baseline" if base is None else "propose", round=rnd, round_started_at=started, tried=None)
        try:
            before = base or suite(list(range(seeds[0], seeds[1] + 1)), inc_dir, incumbent["tunables"], "baseline")
        except Exception as exc:
            if cancelled():
                doc.update(status="cancelled", stop_reason="cancelled")
                tick(phase="cancelled")
                return 3
            doc.update(status="failed", stop_reason="infrastructure_error")
            tick(phase="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        # the working copy: from the incumbent's copy, else the stock card
        ws_rel = f"campaigns/evolve-{args.task}/work/r{rnd}"
        ws = Workspace.create(args.session / ws_rel, inc_dir or stock, stock)
        tunables = copy.deepcopy(incumbent["tunables"])
        os.environ[OVERRIDE_ENV] = json.dumps(tunables)   # so mount_params reads the incumbent's knobs
        params = mount_params(binding["policy"])
        knobs_from = {k: v for k, v in (params.get("tunables") if isinstance(params.get("tunables"), dict) else params).items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool)}
        prop = take_proposal(args.session, args.task, rnd)
        tick(phase="propose", proposer="llm")
        try:
            ep = endpoint(args.llm_model)
        except Exception as exc:  # noqa: BLE001
            doc.update(status="failed", stop_reason="model_error")
            tick(phase="failed", error=f"model endpoint: {exc}")
            return 5
        agent = Agent(ep, ws, pkg=pkg, tunables=tunables, knobs=dict(knobs_from),
                      run_seed=lambda sl, label, ws=ws, tunables=tunables: suite(sl, ws.path, tunables, label),
                      dev_seeds=list(range(seeds[0], seeds[1] + 1)), baseline=before, session=args.session,
                      notebook=notebook.text(), proposal=prop, max_steps=args.max_steps, max_probes=args.max_probes,
                      max_tokens=args.max_output_tokens, options=thinking, cancelled=cancelled,
                      audit_path=store.dir / "llm" / f"round-{rnd}.json", tick=tick,
                      frontier=failure_frontier(doc["rounds"], before, list(range(seeds[0], seeds[1] + 1))))
        outcome = agent.loop()
        if outcome["status"] == "cancelled" or cancelled():
            doc.update(status="cancelled", stop_reason="cancelled")
            tick(phase="cancelled")
            return 3
        after, compared, accepted, why, confirm, exc_text = None, None, False, outcome.get("reason") or "", None, None
        diff = ws.diff()
        tun_changes = {k: v for k, v in agent.knobs.items() if knobs_from.get(k) != v}
        parent_round = int(incumbent.get("round") or 0)
        tried_kind = "none"
        # the model finished, OR it became unusable (error / silence) with edits pending: the
        # simulator does not need the model, so pending edits are still measured, never lost
        if outcome["status"] == "finished" or (outcome["status"] in ("error", "exhausted") and (diff or tun_changes)):
            if not diff and not tun_changes:
                why = "finish without any edit: nothing to evaluate"
            elif ws.protected_ok():
                why = ws.protected_ok()
            else:
                tried_kind = "edit"
                tick(phase="retest", tried={"kind": "edit", "node": before["seeds"][str(seeds[0])].get("first_death")})
                try:
                    after = suite(list(range(seeds[0], seeds[1] + 1)), ws.path, tunables, "retest")
                    compared = compare(before, after)
                    accepted, why = compared["accepted"], (
                        f"gained {compared['gains']}" if compared["accepted"] else
                        f"regressed {compared['regressions']}" if compared["regressions"] else "no milestone gained")
                except Exception as exc:  # noqa: BLE001 -- the candidate's own failure is the round's verdict
                    if cancelled():
                        doc.update(status="cancelled", stop_reason="cancelled")
                        tick(phase="cancelled")
                        return 3
                    exc_text = str(exc)[-2000:]
                    why = f"candidate suite failed: {exc_text[:300]}"
        if accepted and after["count"] > before["count"] and args.confirm_seeds > 0:
            cs = list(range(seeds[1] + 1, seeds[1] + 1 + args.confirm_seeds))
            tick(phase="confirm")
            try:
                cb = suite(cs, inc_dir, incumbent["tunables"], "confirm-before", media_on=False)
                ca = suite(cs, ws.path, tunables, "confirm-after", media_on=False)
                check = compare(cb, ca)
                confirm = {"seeds": cs, "before": cb["count"], "after": ca["count"], "regressions": check["regressions"]}
                if check["regressions"]:
                    accepted, why = False, f"fresh seeds regressed {check['regressions']}"
            except Exception as exc:  # noqa: BLE001
                if cancelled():
                    doc.update(status="cancelled", stop_reason="cancelled")
                    tick(phase="cancelled")
                    return 3
                accepted, why = False, f"confirmation suite failed: {str(exc)[-300:]}"
                confirm = {"seeds": cs, "before": None, "after": None, "error": str(exc)[-500:]}
            seeds[1] = cs[-1]
            tick(seeds_total=seeds[1] - seeds[0] + 1)
        if accepted:
            incumbent = doc["incumbent"] = {"workspace": ws_rel, "round": rnd, "tunables": tunables}
        kept = after if accepted else before
        outcome_word = ("error" if outcome["status"] == "error" else "none" if tried_kind == "none" else
                        "improved" if accepted else "worse" if compared and compared["regressions"] else
                        "same" if compared else "error")
        files = sorted({l[6:] for l in diff.split("\n") if l.startswith("+++ b/")})
        # the console's learning chart reads evaluation.{before,after}.{progress,successes,episodes}
        # and segments epochs by (protocol_id, objective_id)
        sample = lambda suite: {"successes": score(suite)[0], "episodes": len(suite["seeds"]), "progress": score(suite)[1]}
        eval_row = {"protocol_id": "milestones-v1",
                    "objective_id": sha_json({"task": args.task, "card": pkg, "milestones": "verify nodes + terminal success"}),
                    "before": sample(before), "after": sample(after) if after else None,
                    "acceptance": {"accepted": accepted, "reason": why}}
        row = {"round": rnd, "ts": time.time(), "proposer": "llm",
               "tried": {"kind": tried_kind, "node": before["seeds"][str(seeds[0])].get("first_death"),
                         "detail": {"summary": outcome.get("summary") or outcome.get("reason"), "files": files,
                                    "tunables": tun_changes, "diff": diff[:20_000], "reason": why, "error": exc_text}},
               "before": before["count"], "after": after["count"] if after else None,
               "best": max(int(doc["best"]), kept["count"]), "suite_sha": after["sha"] if after else None,
               "before_score": score(before), "after_score": score(after) if after else None,
               "outcome": outcome_word, "accepted": accepted, "accepted_reason": why, "published": False,
               "evaluation": eval_row,
               "parent": parent_round,
               "regression": {"lost": compared["regressions"] if compared else []}, "confirm": confirm,
               "workspace": ws_rel, "needs": [] if tried_kind != "none" else ["edit"],
               "per_seed": per_seed(before), "after_seeds": per_seed(after) if after else [],
               "media": list(dict.fromkeys([*before.get("media", []), *(after.get("media", []) if after else [])])),
               "media_dropped": {**{f"before/{k}": v for k, v in before.get("media_dropped", {}).items()},
                                 **({f"after/{k}": v for k, v in after.get("media_dropped", {}).items()} if after else {})},
               "probes": agent.probes, "proposal": {k: prop[k] for k in ("id", "kind", "note")} if prop else None,
               "usage": {"llm_tokens": dict(agent.usage), "model_calls": agent.calls,
                         "episode_attempts": round_cost["episode_attempts"], "sim_s": round(round_cost["sim_s"], 3),
                         "wall_s": round(time.time() - started, 3)},
               "llm": {"model": getattr(ep, "identity", type(ep).__name__), "requested_model": args.llm_model,
                       "effort": args.llm_effort, "status": outcome["status"], "calls": agent.calls,
                       "actions": agent.actions, "finish_reasons": agent.finishes, "errors": agent.errors[-8:], "summary": outcome.get("summary"),
                       "reason": outcome.get("reason"), "error": outcome.get("error"), "stop_reason": outcome.get("reason")}}
        doc["rounds"].append(row)
        doc.update(cursor=rnd, best=row["best"])
        verdict = ("ACCEPTED" if accepted else "REJECTED" if tried_kind == "edit" else
                   "NO EDIT" if outcome["status"] in ("gave_up", "exhausted") else outcome["status"].upper())
        entry = [(f"## Round {rnd} — {verdict}  (successes {before['count']} → {after['count'] if after else '-'} "
                  f"of {len(before['seeds'])}; {why})"),
                 f"- hypothesis / summary: {outcome.get('summary') or outcome.get('reason') or '-'}"]
        if tun_changes:
            entry.append(f"- tunables: {json.dumps(tun_changes)}")
        for p in agent.probes:
            c = p.get("compare") or {}
            entry.append(f"- probe seed {p['seed']}: " + (f"error {p['error'][:200]}" if p.get("error") else
                         f"{'success' if p.get('success') else 'fail at ' + str(p.get('first_death'))} "
                         f"{p.get('failure_mode') or ''} gains {c.get('gains')} lost {c.get('regressions')}"))
        if diff:
            entry.append("```diff\n" + diff[:6000] + "\n```")
        if after:
            entry.append("- result per seed:\n  " + "\n  ".join(
                describe_seed(s, after["seeds"][s], before["seeds"][s]).split("\n")[0] for s in after["seeds"]))
        notebook.append("\n".join(entry))
        _trim_workspaces(store.dir / "work", keep={Path(incumbent["workspace"]).name} if incumbent.get("workspace") else set(),
                         last=rnd)
        _trim_episodes(args.session, args.task, int(incumbent.get("round") or 0), rnd)
        completed += 1
        if outcome["status"] == "error":
            doc.update(status="failed", stop_reason="model_error")
            tick(phase="failed", error=outcome.get("error"), last_round_s=round(time.time() - started, 1))
            print(json.dumps({"task": args.task, "cursor": rnd, "status": "failed", "error": outcome.get("error")}),
                  file=sys.stderr)
            return 5
        tick(phase="idle", last_round_s=round(time.time() - started, 1))
        base = None if confirm else kept
    doc.update(status="done", stop_reason="round_limit")
    tick(phase="done")
    print(json.dumps({"task": args.task, "cursor": doc["cursor"], "best": doc["best"], "status": "done"}))
    return 0


def failure_frontier(rounds: list[dict], baseline: dict, seeds: list[int], window: int = 5) -> str:
    """Per development seed: where the incumbent dies, how many single-seed runs the last
    ``window`` rounds spent on it, and the last round that gained a milestone there. A
    fact table (Zetta's failure clusters), not a target list."""
    recent = [r for r in rounds if r.get("round")][-window:]
    if not recent:
        return ""
    lines = [f"## Failure frontier (last {len(recent)} rounds: {recent[0]['round']}–{recent[-1]['round']})",
             "seed | incumbent dies at | runs spent on it | last round with a gain on it"]
    for seed in seeds:
        row = baseline["seeds"].get(str(seed)) or {}
        death = "SUCCESS" if row.get("success") else f"{row.get('first_death')} {row.get('failure_mode') or ''}".strip()
        probes = sum(1 for r in recent for p in r.get("probes") or [] if p.get("seed") == seed)
        gained = [r["round"] for r in rounds if any(g.startswith(f"{seed}:") for p in r.get("probes") or []
                                                    for g in (p.get("compare") or {}).get("gains") or [])]
        lines.append(f"{seed} | {death} | {probes} | {gained[-1] if gained else 'never'}")
    return "\n".join(lines)


def _trim_episodes(session: Path, task: str, keep_round: int, last: int) -> None:
    """Whole-episode videos are the largest thing a round leaves: keep the incumbent's
    and the last WORKSPACES_KEPT rounds', drop the rest (index entries included)."""
    for d in (session / "media" / "rsi" / task).glob("round-*"):
        m = re.fullmatch(r"round-(\d+)", d.name)
        if not m or int(m.group(1)) == keep_round or int(m.group(1)) > last - WORKSPACES_KEPT:
            continue
        for idx in d.glob(f"retest/{task}/*/index.json"):
            try:
                data = json.loads(idx.read_text())
                if data.get("files", {}).pop("episode", None) is not None:
                    (idx.parent / "episode.mp4").unlink(missing_ok=True)
                    idx.write_text(json.dumps(data, sort_keys=True, indent=1))
            except (OSError, ValueError):
                continue


def _trim_workspaces(work: Path, keep: set[str], last: int) -> None:
    """Rejected copies older than WORKSPACES_KEPT rounds go; the incumbent's stays."""
    for d in work.glob("r*"):
        m = re.fullmatch(r"r(\d+)", d.name)
        if d.is_dir() and m and d.name not in keep and int(m.group(1)) <= last - WORKSPACES_KEPT:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
