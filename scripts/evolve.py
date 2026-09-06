"""Online program-policy RSI: bounded exploration and frozen paired evaluation.

The runtime starts this process for an evolution-mode brief. The original task
binding supplies the evaluator; candidate controllers and inserted actions cannot
edit its objectives. Accepted development overlays stay inside their campaign.
Installation requires the independent verification battery in plugins/rsi.

Round artifacts retain logs, before/after trajectories, media paths and model
feedback. Runtime seals completed rounds; cancellation is a separate outcome.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from board import store as bs
from harness import media
from harness.config import Mount, Patch, Profile, resolve_plan, sha_json
from harness.definitions import CAPABILITIES
from harness.events import SessionLog
from harness.kernel import Kernel
from harness.manifest import discover, mount_params
from harness.protocol import SkillRecordV0, to_plain
from harness.registry import load_provider
from plugins.rsi import diagnosis, evaluation, experience, interventions
from plugins.rsi.learner import ProgramLearner, intervention_summary
from plugins.task import workload
from scripts import evolve_llm
from scripts import harness_runtime as hr
from scripts.brief_drop import drop
from scripts.rsi_campaign import _maybe_arm_frames

MODES = ("execution", "evolution")
#: JSON ``{provider ref: {param: value}}`` merged over a card's mount params by
#: ``harness.manifest.mount_params`` -- how a tunables trial reaches a driver.
OVERRIDE_ENV = "PH_MOUNT_PARAMS_OVERRIDE"
#: Extra card roots (harness.manifest.discover); a ``card`` proposal appends its
#: candidate dir for the round's suite, on top of whatever the process was given.
EXTRA_ENV = "PH_PLUGINS_EXTRA"
_BASE_EXTRA = os.environ.get(EXTRA_ENV, "")
PLANNER_REF = "scripts.evolve:planner_provider"
#: Full round rows kept in campaign.json; older rounds live in ``rounds/<round>.json``
#: and stand in the file as an index row. The live recycle_cans campaign reached 42 MB
#: over 490 rounds -- 19 MB of it per-seed node trails nothing reads twice.
ROUNDS_KEPT = 20
#: ``llm/round-*.json`` kept whole; older audits are rewritten to their summary (never
#: deleted). 482 files x ~285 KB = 145 MB on the live box.
AUDITS_KEPT = 50
#: Model-written candidate cards kept beyond the referenced ones (``gc_candidates``).
CANDIDATES_KEPT = 20
#: The loud stop (``disk_guard``): free space on the filesystem holding ``runs/``, and
#: the size of one campaign directory. Better a named pause than a full disk mid-write.
MIN_FREE_BYTES = 5 * 1024 ** 3
MAX_CAMPAIGN_BYTES = 2 * 1024 ** 3

#: What an INDEX row keeps verbatim off a round: everything the round loop, the proposer
#: and the RSI page read. Dropped: ``per_seed``/``after_seeds`` node trails (the bulk),
#: ``media``/``media_dropped`` (rsi_frames reads the shard), ``trial_evidence`` (only the
#: LAST round's is ever read), the card source in ``tried.detail.edits`` and the model's
#: raw ``rationale`` / prompt shas -- all still in the shard, and in the llm audit.
_INDEX_KEYS = ("round", "before", "after", "best", "parent", "layer", "notes", "outcome",
               "accepted", "accepted_reason", "published", "before_score", "after_score",
               "usage", "proposer", "needs", "confirm", "trial", "stuck", "regression",
               "burned", "suite_sha", "proposal", "ts", "evaluation", "transfer", "experiments",
               "policy", "run_budget", "cycle_budget", "cycle_outcome", "stop_reason", "memo")
_TRIED_KEYS = ("skill", "ref", "path", "from", "to", "module", "executor", "reason",
               "hint", "error", "layer", "needs", "match", "name", "patch_sha", "artifact_sha")
_SEED_KEYS = ("seed", "success", "first_death", "failure_mode")


#: Free text an index row keeps only a head of (the whole thing stays in the shard and,
#: for the model's own words, in the llm audit): 500 rounds x a 1500-char refusal reason
#: is a third of the file on its own.
_CLIP = 300


def _clip(d: dict, *keys) -> dict:
    return {k: (v[:_CLIP] if k in keys and isinstance(v, str) else v) for k, v in d.items()}


def _seeds_index(rows) -> list[dict]:
    """One round's per-seed summary WITHOUT the node trail: what the proposer's history
    and the failure clusters read (the trail itself is 19 MB of the live campaign)."""
    return [{k: s.get(k) for k in _SEED_KEYS} for s in rows or ()]


def index_row(r: dict, baseline: bool = False) -> dict:
    """The compact stand-in for an archived round. Carries the chart's numbers already
    computed off the trails it drops (``node_rate`` / ``by_task``: board.store._rates,
    the same reading the RSI page's line and heat strip make), so the page can show 500
    rounds without ever opening a shard. ``baseline`` keeps round 1's trails: EVERY round
    re-reads them (``cluster_seeds`` scores the origin cluster off ``rounds[0]``, and falls
    back to the round's own, weaker cluster when they carry no trail), and a shard read that
    silently failed would read as "no cluster" -- 21 KB against that."""
    t = r.get("tried") or {}
    out = _clip({k: r[k] for k in _INDEX_KEYS if k in r}, "notes", "accepted_reason")
    nb, tb = bs._rates(r.get("per_seed"))
    na, ta = bs._rates(r.get("after_seeds"))
    return {**out, "sharded": True, "tried_kind": t.get("kind"), "node": t.get("node"),
            "tried": {"kind": t.get("kind"), "node": t.get("node"),
                      "detail": _clip({k: v for k, v in (t.get("detail") or {}).items()
                                       if k in _TRIED_KEYS}, "reason", "error")},
            "llm": _clip({k: v for k, v in (r.get("llm") or {}).items()
                          if k in ("model", "requested_model", "effort", "summary", "reason", "status", "error", "method", "calls",
                                   "evidence_reads", "trial_calls", "budget", "usage_complete", "stop_reason", "decision_flow")},
                         "summary", "reason") or None,
            "node_rate": {"before": nb, "after": na},
            "by_task": {k: {"before": tb.get(k), "after": ta.get(k)} for k in sorted({*tb, *ta})},
            "per_seed": [{**s, **({"nodes": full.get("nodes") or []} if baseline else {})}
                         for s, full in zip(_seeds_index(r.get("per_seed")), r.get("per_seed") or ())],
            "after_seeds": _seeds_index(r.get("after_seeds"))}


class EvolveStore:
    """``campaigns/evolve-<task>/campaign.json``, written atomically (tmp+rename) and
    BOUNDED: the header plus the last ``ROUNDS_KEPT`` rounds in full. Every older round
    is written ONCE to ``rounds/<round>.json`` (never rewritten) and stands in
    campaign.json as an ``index_row`` -- so the file is O(1) in rounds instead of the
    42 MB / 186 MB the 490-round live campaign reached, and the board's faces fit through
    the 1 MB pipe the station bridge gives them.

    A legacy (unsharded) file is migrated on the first ``load``: the original is copied
    to ``campaign.json.bak`` beside it and the sharded shape written atomically over it.
    The loader reads both shapes and the migration is idempotent (an index row is
    recognised by ``sharded``)."""

    def __init__(self, session: Path, task: str) -> None:
        self.dir = session / "campaigns" / f"evolve-{task}"
        self.path = self.dir / "campaign.json"
        self.rounds_dir = self.dir / "rounds"

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        doc = json.loads(self.path.read_text())
        if any(not r.get("sharded") for r in (doc.get("rounds") or [])[:-ROUNDS_KEPT]):
            bak = self.path.with_suffix(".json.bak")
            if not bak.exists():
                bak.write_bytes(self.path.read_bytes())
            self.save(doc)   # shards in place, atomically
        return doc

    def round(self, n: int) -> dict | None:
        """The archived FULL row of round ``n`` (media list, node trails, card source),
        or None when it is still inside the window / predates sharding."""
        try:
            return json.loads((self.rounds_dir / f"{int(n)}.json").read_text())
        except (OSError, ValueError):
            return None

    def save(self, doc: dict) -> None:
        rows = doc.get("rounds") or []
        for i, r in enumerate(rows[:-ROUNDS_KEPT]):
            if not r.get("sharded"):
                rows[i] = self._archive(r, baseline=i == 0)
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        # compact: the file is an index the board reads, not one an operator diffs -- and
        # indent=1 is a 1.65x tax on every one of the hundreds of writes a round makes.
        tmp.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
        os.replace(tmp, self.path)

    def _archive(self, r: dict, baseline: bool = False) -> dict:
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        p = self.rounds_dir / f"{int(r['round'])}.json"
        if not p.exists():   # written once, never rewritten
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(r, indent=1, sort_keys=True, default=str))
            os.replace(tmp, p)
        return index_row(r, baseline)


# ── maintenance: the three things that grow without a bound ──────────────────────

_AUDIT_KEEP = ("round", "experiment_id", "model", "requested_model", "effort", "prompt_sha", "raw_sha", "summary", "rationale",
               "reason", "usage", "calls", "status", "error", "requests", "events", "method",
               "evidence_reads", "evidence_refs", "trial_calls", "budget", "usage_complete", "stop_reason", "decision_flow")


def _round_no(p: Path) -> int:
    try:
        return int(re.sub(r"\D", "", p.stem) or 0)
    except ValueError:
        return 0


def prune_audits(llm_dir: Path, keep: int = AUDITS_KEPT, dry_run: bool = False) -> list[str]:
    """``campaigns/<c>/llm/round-<r>.json`` older than the last ``keep``: REWRITTEN to
    their summary -- round, decision, layer, summary, rationale, reason, usage, the
    attempt count and every sha (prompt, raw, and each rejected attempt's payload sha with
    its reason, which is what ``evolve_llm._prior_rejects`` reads back) -- with the raw
    model text, the whole prompt and the materials dropped. Chosen over gzip because the
    only thing anything reads out of an old audit IS that summary (~1 KB against ~285 KB
    whole, ~40 KB gzipped). An audit is NEVER deleted: the row survives, only its bulk
    goes. Idempotent (``pruned``). Returns one line per file rewritten."""
    out = []
    for f in sorted(llm_dir.glob("round-*.json"), key=_round_no)[:-keep or None]:
        try:
            a = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(a, dict) or a.get("pruned"):
            continue
        t = a.get("tried") or {}
        small = {k: a[k] for k in _AUDIT_KEEP if k in a}
        # Keep the request/result identities after archival, not a second copy
        # of all source, traces and model messages forever.
        if 'requests' in small:
            small['requests'] = [{k: v for k, v in request.items() if k not in ('messages', 'raw')}
                                 for request in small['requests']]
        if 'events' in small:
            small['events'] = [{k: v for k, v in event.items() if k != 'result'}
                              for event in small['events']]
        small |= {"pruned": True, "decision": t.get("kind"), "node": t.get("node"),
                  "layer": (t.get("detail") or {}).get("layer"),
                  "tried": {"kind": t.get("kind"), "node": t.get("node"),
                            "detail": {k: v for k, v in (t.get("detail") or {}).items()
                                       if k in _TRIED_KEYS}},
                  "attempt_count": len(a.get("attempts") or []),
                  "attempts": [{k: at.get(k) for k in ("sha", "reason")}
                               for at in a.get("attempts") or ()]}
        text = json.dumps(small, indent=1, sort_keys=True, default=str)
        out.append(f"prune audit {f.name}: {f.stat().st_size} -> {len(text)} bytes")
        if not dry_run:
            f.write_text(text)
    return out


def _tracked(root: Path) -> set[str] | None:
    """The HAND-WRITTEN candidate cards = the ones git tracks (plugins/candidates is
    git-ignored, so everything else in it is a run artefact). None when git cannot answer:
    the GC then deletes nothing, because it cannot tell the two apart."""
    try:
        p = subprocess.run(["git", "ls-files", "-z", "--", "."], cwd=root, check=False,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return {q.split("/")[0] for q in p.stdout.split("\0") if q}


def _referenced(runs: Path) -> set[str]:
    """Every candidate card name any campaign under ``runs`` still stands on: the names
    mentioned by its ``applied`` overlay or its ``accepted_stack`` (a published round is
    accepted by definition, so an installed skill's binding is never cut). Read as text
    off both refs (``plugins.candidates.<name>:provider``) and paths
    (``.../candidates/<name>/...``) -- over-keeping here costs a directory, under-keeping
    costs a skill record."""
    names: set[str] = set()
    for c in sorted(runs.glob("*/campaigns/evolve-*/campaign.json")):
        try:
            doc = json.loads(c.read_text())
        except (OSError, ValueError):
            continue
        # accepted_stack is the authority; the accepted/published ROWS are the same
        # answer for a campaign written before the stack existed (evolve_llm._accepted).
        # ponytail: campaigns are the index -- a card published by a campaign whose
        # campaign.json has been deleted is not seen. Scan runs/*/skills if that happens.
        blob = json.dumps([doc.get("applied"), doc.get("accepted_stack"), doc.get("working_candidates"),
                           [r for r in doc.get("rounds") or ()
                            if r.get("accepted") or r.get("published")]], default=str)
        names |= set(re.findall(r"candidates[./\\]([A-Za-z_]\w*)", blob))
    return names


def gc_candidates(root: Path | None = None, runs: Path | None = None,
                  keep: int = CANDIDATES_KEPT, dry_run: bool = False) -> list[str]:
    """Delete model-written candidate cards, OLDEST FIRST, keeping: every git-tracked
    (hand-written) card, every card a campaign under ``runs`` still references
    (``_referenced``), and the newest ``keep`` of the rest by mtime. 318 dirs / 17 MB on
    the live box. Returns one line per deletion (the whole log in ``dry_run``)."""
    root = Path(root or evolve_llm.CANDIDATES_ROOT)
    runs = Path(runs if runs is not None else REPO_ROOT / "runs")
    if not root.is_dir():
        return []
    tracked = _tracked(root)
    if tracked is None:
        return [f"gc: {root} is not a readable git checkout -- nothing deleted"]
    pinned = tracked | _referenced(runs)
    cards = sorted((d for d in root.iterdir()
                    if d.is_dir() and d.name not in pinned and not d.name.startswith(("_", "."))),
                   key=lambda d: d.stat().st_mtime)
    out = []
    for d in cards[:-keep or None]:
        out.append(f"gc candidate {d.name} (mtime {int(d.stat().st_mtime)}, "
                   f"{sum(f.stat().st_size for f in d.rglob('*') if f.is_file())} bytes)")
        if not dry_run:
            shutil.rmtree(d, ignore_errors=True)
    return out


def _bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.is_dir() else 0


def disk_guard(session: Path, campaign_dir: Path) -> str | None:
    """The operator message when the next round must NOT start -- the filesystem holding
    ``runs/`` is under ``MIN_FREE_BYTES``, or this campaign's directory is over
    ``MAX_CAMPAIGN_BYTES`` -- else None. Both numbers are named in the message: 26 GB of
    593 GB left with a 186 MB campaign dir on 2026-09-04 is how close this got."""
    free = shutil.disk_usage(session).free
    if free < MIN_FREE_BYTES:
        return (f"磁盘只剩 {free / 1e9:.1f} GB（下限 {MIN_FREE_BYTES / 1e9:.0f} GB），"
                f"本轮不开始：status=paused_disk")
    if (used := _bytes(campaign_dir)) > MAX_CAMPAIGN_BYTES:
        return (f"战役目录 {campaign_dir.name} 已 {used / 1e9:.1f} GB"
                f"（上限 {MAX_CAMPAIGN_BYTES / 1e9:.0f} GB），本轮不开始：status=paused_disk")
    return None


def maintain(store: EvolveStore, session: Path, dry_run: bool = False) -> list[str]:
    """One round's housekeeping, run BEFORE the round: prune this campaign's audits and
    GC the candidate cards. Cheap -- a glob and a stat walk plus the campaign headers.

    The GC runs ONLY when this loop's session sits in the repo's own ``runs/`` and the
    cards sit in the repo's own ``plugins/candidates``: the card store is repo-global
    while the reference set is read out of campaigns, so a loop pointed at a scratch
    ``runs/`` (every e2e test) would judge the repo's cards against campaigns it cannot
    see. Anything but that exact pairing and it deletes nothing."""
    out = prune_audits(store.dir / "llm", dry_run=dry_run)
    runs, root = REPO_ROOT / "runs", Path(evolve_llm.CANDIDATES_ROOT)
    if session.parent.resolve() == runs.resolve() and root.resolve() == (REPO_ROOT / "plugins" / "candidates").resolve():
        out += gc_candidates(root, runs, dry_run=dry_run)
    return out


# ── the executor-switch seam: a planner wrapper that stamps node.executor ────────

class _Forced:
    def __init__(self, inner, executors: dict, graph=None) -> None:
        self._inner, self._executors, self._graph = inner, dict(executors), graph

    def plan(self, brief):
        plan = copy.deepcopy(self._graph) if self._graph else dict(self._inner.plan(brief))
        plan["nodes"] = [{**n, "executor": self._executors[n["id"]]}
                         if n.get("id") in self._executors else n
                         for n in plan.get("nodes") or ()]
        return plan

    def __getattr__(self, name):
        return getattr(self._inner, name)


def planner_provider(inner: str, inner_params=None, executors=None, graph=None) -> _Forced:
    return _Forced(load_provider(inner, dict(inner_params or {})), executors or {}, graph)


def node_group(node: dict, graph: dict) -> str:
    """The sub-task label a plan node belongs to (``task`` on trail rows, the key
    rsi_series groups by): the node's ``task`` when the graph carries mission
    ``tasks``, else its stage word before the first ``-`` (nav-can1 -> nav)."""
    return str(node.get("task") or node["id"]) if graph.get("tasks") else str(node["id"]).split("-", 1)[0]


class _Tap(SessionLog):
    """The per-seed ledger as a node trail: ``task.plan`` sets ``nodes`` (plan order,
    ``ok`` None, ``after``/``kind`` = the graph's edges and node kind, ``task`` = its
    sub-task label (``node_group``), so the trail draws as
    a graph; a replan resets all but the verified-ok nodes and rewrites every edge), each ``task.verify`` fills that node's ``ok`` (+
    ``steps`` / ``failure_mode`` when the row carries them). The node in flight is
    the first not yet verified ok -- an inference (no node-start row exists), not
    a reading. ``on_change(nodes)`` fires at every change."""

    def __init__(self, on_change) -> None:
        super().__init__()
        self._on, self.nodes = on_change, []

    def append(self, kind: str, data: dict) -> int:
        seq = super().append(kind, data)
        if kind == "task.plan" and data.get("graph"):
            done = {n["id"]: n for n in self.nodes if n["ok"] is True}   # a replan keeps verified nodes
            self.nodes = [{**(done.get(n["id"]) or {"id": n["id"], "skill": n.get("skill"), "ok": None,
                                                    "steps": None, "failure_mode": None}),
                           "after": list(n.get("after") or []), "kind": n.get("kind", "manipulate"),   # validator's default
                           "task": node_group(n, data["graph"])}
                          for n in data["graph"].get("nodes") or []]   # edges always from the latest graph
        elif kind == "task.verify" and (hit := [n for n in self.nodes if n["id"] == data.get("node")]):
            hit[0].update(ok=all((data.get("results") or {}).values()), steps=data.get("steps"),
                          failure_mode=(data.get("diagnostics") or {}).get("failure_mode"))
        else:
            return seq
        self._on(copy.deepcopy(self.nodes))
        return seq


def _mount(binding: dict, skills_root: Path, executors: dict, graph=None):
    plan = hr._mount_plan(binding, skills_root, frames=_maybe_arm_frames())
    if not executors and not graph:
        return plan
    m = next(m for m in plan.mounts if m.capability == "task.planner")
    forced = Mount("task.planner", PLANNER_REF,
                   {"inner": m.provider, "inner_params": dict(m.params),
                    "executors": dict(executors), "graph": graph})
    return resolve_plan(Profile("evolve", plan.mounts),
                        patches=(Patch("evolve", override=(forced,)),))


# ── look: the seed suite, in-process ──────────────────────────────────────────────

def _get(budgets, binding: dict, key: str, default):
    """Budget precedence: the brief's value, else the task binding's, else the default."""
    v = (budgets or {}).get(key)
    return binding.get(key, default) if v is None else v


def run_suite(task: str, binding: dict, seeds: list | None, arm: str, skills_root: Path,
              applied: dict, media_dir: Path | None = None, budgets: dict | None = None,
              progress=None, seed_list: list | None = None, media_prefix: str = "media") -> dict:
    """{count, seeds: {seed: {success, first_death, fault, nodes}}, sha}. ``media_dir``
    (<session>/media) turns on the workload's segment recorder: kept-on-success clips.
    ``progress(**live)`` is called at every seed boundary and node change.
    ``seeds`` is the inclusive range [lo, hi]; ``seed_list`` names the seeds explicitly
    instead. Development acceptance always compares the complete paired suite."""
    os.environ[OVERRIDE_ENV] = json.dumps(applied["tunables"])
    cards = applied.get("cards") or {}
    for card in cards.values():
        if card.get('artifact_sha') != evolve_llm.candidate_digest(card['path']):
            raise ValueError('candidate source changed after validation')
    os.environ[EXTRA_ENV] = ":".join(r for r in (_BASE_EXTRA, *(c["path"] for c in cards.values())) if r)
    per, logs, graphs = {}, [], []
    brief = {**hr.task_brief(task, binding), "arm": arm}
    if cards:   # a candidate card's executor: bind it into this suite's records (the plan
        # validator's view) and segment specs (rearm's route) -- in memory, never on disk
        specs = brief["segment_specs"] = copy.deepcopy(brief.get("segment_specs") or {})
        recs = brief["records"] = {k: to_plain(v) if isinstance(v, SkillRecordV0) else copy.deepcopy(v)
                                   for k, v in (brief.get("records") or {}).items()}
        for key, c in cards.items():
            specs.setdefault(c["skill"], {}).setdefault("policies", {})[key] = _binding(c)
            emb = brief["embodiment"]
            rec = recs.setdefault(c["skill"], {"id": c["skill"], "name": c["skill"]})
            rec.setdefault("bindings", {}).setdefault(emb, {}).setdefault("policies", {})[key] = _binding(c)
    if media_dir is not None:
        brief["media_dir"] = str(media_dir)
    tick = progress or (lambda **kw: None)
    t_suite = time.time()
    seq = [int(x) for x in seed_list] if seed_list is not None else list(
        range(int(seeds[0]), int(seeds[1]) + 1))
    for i, seed in enumerate(seq):
        t_seed = time.time()
        tick(seed_index=i, seed=seed, seeds_total=len(seq), node=None, nodes=[], seed_started_at=t_seed)
        log = _Tap(lambda nodes: tick(nodes=nodes, node=next(
            (n["id"] for n in nodes if n["ok"] is not True), None)))
        kernel = Kernel(CAPABILITIES, log=log)
        kernel.mount(_mount(binding, skills_root, applied["executors"], applied.get("graph")))
        out = workload.run(dict(brief), kernel, seed=seed,
                           max_replans=int(_get(budgets, binding, "max_replans", 3)),
                           max_actuations=int(_get(budgets, binding, "max_actuations", 3)),
                           segment_retries=int(binding.get("segment_retries", 0)))
        skills = {}
        for r in log.rows():
            if r["kind"] == "task.plan" and r["data"].get("graph"):
                if not graphs:
                    graphs.append(copy.deepcopy(r["data"]["graph"]))
                skills.update({n["id"]: n["skill"] for n in r["data"]["graph"].get("nodes") or []})
        nodes, faults = out["nodes"], out.get("faults") or []
        dead = next((nid for nid, n in nodes.items() if not n["success"]), None)
        per[str(seed)] = {
            "success": bool(out["success"]),
            "verification_observations": out.get("verification_observations", []),
            "terminal_observation": out.get("terminal_observation"),
            "elapsed_s": round(time.time() - t_seed, 1),
            "trail": [{k: n[k] for k in ("id", "ok", "steps", "failure_mode", "after", "kind", "task")}
                      for n in log.nodes],
            "first_death": dead,
            "failure_mode": (nodes[dead].get("diagnostics") or {}).get("failure_mode") if dead else None,
            "fault": {k: faults[0].get(k) for k in ("kind", "node", "msg")} if faults else None,
            # the first-death node's failure keyframes (session-relative paths; the LLM brief's images)
            "keyframes": [f"{media_prefix}/{task}/{seed}/{f}" for f in (media.dropped_of(media_dir, task, seed)
                                                                .get(dead) or {}).get("keyframes", [])]
            if media_dir is not None and dead else [],
            "nodes": {nid: {"skill": skills.get(nid), "success": bool(n["success"]),
                            "executor": n.get("executor") or "scripted",
                            **({'driver': n['driver']} if n.get('driver') else {}),
                            "tunables_sha": (n.get("diagnostics") or {}).get("tunables_sha")}
                      for nid, n in nodes.items()}}
        for n in per[str(seed)]["trail"]:   # final state from the result: a replan reset the live
            r = nodes.get(n["id"]) or {}      # trail, and the verify row carries no steps/diagnostics
            diag = r.get("diagnostics") or {}
            if n["ok"] is None and "success" in r:
                n["ok"] = bool(r["success"])
            n["steps"] = n["steps"] if n["steps"] is not None else r.get("steps")
            # Missing executor observations stay absent through projection and inspection;
            # absence must never become a measured "no stall" claim.
            if "failure_mode" in diag or n["failure_mode"] is not None:
                n["failure_mode"] = n["failure_mode"] or diag.get("failure_mode")
            else:
                del n["failure_mode"]
            if (diag.get("trace") or {}).get("end"):   # where this segment ENDED, every node:
                n["trace_end"] = diag["trace"]["end"]  # the reference index + the upstream row
            if diag.get("trace"):
                # Passed upstream actions can cause downstream failure. Keep their
                # observed response available to the model instead of preselecting a cause.
                n["trace"] = diag["trace"]
                n["geometry"] = diag.get("geometry")
        _link_upstream(per[str(seed)]["trail"], dead, skills)
        logs += evolve_llm._log_excerpt(seed, log.rows(), dead,
                                        evolve_llm.MAX_LOG_LINES // len(seq))
        tick(per_seed_partial=per_seed({"seeds": per}))
    return {"count": sum(s["success"] for s in per.values()), "seeds": per, "sha": sha_json(per),
            "elapsed_s": round(time.time() - t_suite, 3),
            "logs": logs, "plan": graphs[0] if graphs else None}


def _merge(a: dict, b: dict) -> dict:
    """Two ``run_suite`` results over disjoint seeds as one (the preflight seed + the rest)."""
    per = {**a["seeds"], **b["seeds"]}
    return {"count": sum(s["success"] for s in per.values()), "seeds": per, "sha": sha_json(per),
            "elapsed_s": round(a["elapsed_s"] + b["elapsed_s"], 3), "logs": a["logs"] + b["logs"]}


def _link_upstream(trail: list, dead, skills: dict) -> None:
    """Give the first-death row an ``upstream``: ``{node, skill, steps, trace_end}`` of
    the SEGMENT that ran just before it (verify/decide nodes move nothing; a recovery
    node between them is skipped -- the question is which leg parked the base where the
    dying stage found it). Without it "the base never moved" is a fact with no author."""
    i = next((k for k, n in enumerate(trail) if n["id"] == dead), None) if dead else None
    if i is None:
        return
    up = next((n for n in reversed(trail[:i]) if n.get("kind") == "segment" and n.get("steps")), None)
    if up is not None:
        trail[i]["upstream"] = {"node": up["id"], "skill": skills.get(up["id"]),
                                "steps": up["steps"], "trace_end": up.get("trace_end")}


def update_reference(doc: dict, kept: dict, rnd: int) -> dict:
    """The campaign's SUCCESSFUL REFERENCE INDEX (Zetta, cheap version): per plan
    segment, the LAST round in which it passed -- ``reference: {node: {node, seed,
    steps, d_eef, d_base, round} | null}``, null = the node has run and never passed.
    The healthy baseline a death is measured against ("nav-can1 passed on 4243 with
    d_base 0.13; here it is 1.03"). Kept across rounds, never reset."""
    ref = doc.setdefault("reference", {})
    for seed, s in kept["seeds"].items():
        for n in s.get("trail") or []:
            if n.get("kind") != "segment":
                continue
            ref.setdefault(n["id"], None)
            if n.get("ok") and n.get("steps") is not None:
                end = n.get("trace_end") or {}
                ref[n["id"]] = {"node": n["id"], "seed": int(seed), "steps": n["steps"],
                                "d_eef": end.get("d_eef_target"),
                                "d_base": end.get("d_base_target"), "round": rnd}
    return ref


def per_seed(suite: dict) -> list[dict]:
    """The operator-facing per-seed summary sealed with every round (rsi_step /
    campaign.json): ``[{seed, success, first_death, failure_mode, tunables_sha, elapsed_s,
    nodes: [{id, ok, steps, failure_mode, after, kind, task, trace_end?, trace?, geometry?,
    upstream?}]}]`` (the knobs the dying node ran under; the node trail's final state;
    ``trace_end`` = every traced segment's last {eef, target, base, d_eef_target,
    d_base_target, step}; and on the FIRST-DEATH row alone ``trace`` = its stall geometry
    {start, stall, end} + the downsampled per-step ``series``, ``geometry`` = where its
    target came from (fixture bbox / dock, the knobs used, the arm's reach_max), and
    ``upstream`` = the segment that ran before it) -- the seed detail that otherwise
    lives only in this process."""
    return [{"seed": int(seed), **{k: s.get(k) for k in ("success", "first_death", "failure_mode")},
             "elapsed_s": s.get("elapsed_s"), "nodes": s.get("trail") or [],
             "evaluation": s.get("evaluation"),
             "verification_observations": s.get('verification_observations') or [],
             "terminal_observation": s.get('terminal_observation'),
             "tunables_sha": (s["nodes"].get(s["first_death"]) or {}).get("tunables_sha")
             if s.get("first_death") else None}
            for seed, s in suite["seeds"].items()]


# ── the trial's own evidence: what the MODEL'S code did in the simulator ──────────

def _exception(exc: BaseException) -> dict:
    """A raise inside the trial, kept whole enough to fix: ``{type, message, file, line,
    traceback}`` (the tail, <= 15 lines). Today a raise leaves only ``repr(exc)``."""
    fr = (traceback.extract_tb(exc.__traceback__) or [None])[-1]
    return {"type": type(exc).__name__, "message": str(exc)[:500],
            "file": fr.filename if fr else None, "line": fr.lineno if fr else None,
            "traceback": "".join(traceback.format_exception(
                type(exc), exc, exc.__traceback__)).rstrip().splitlines()[-15:]}


def _row(suite: dict, seed, node) -> dict:
    """One node's trail row on one seed of a suite result ({} when it never ran)."""
    return next((n for n in ((suite.get("seeds") or {}).get(str(seed)) or {}).get("trail") or []
                 if n.get("id") == node), {})


def _series(node: dict) -> list:
    tr = node.get("trace")
    return (tr.get("series") or []) if isinstance(tr, dict) else []


def _phases(series: list) -> list[str]:
    """The phase sequence of a series, consecutive repeats collapsed."""
    out: list[str] = []
    for r in series:
        p = r.get("phase")
        if p is not None and (not out or out[-1] != str(p)):
            out.append(str(p))
    return out


def _minimum(series: list, key: str, node: dict, end_key: str):
    """The closest the segment ever got; without a series, its last frame's distance."""
    vals = [r[key] for r in series if isinstance(r.get(key), (int, float))]
    if vals:
        return min(vals)
    end = ((node.get("trace") or {}).get("end") if isinstance(node.get("trace"), dict) else None) \
        or node.get("trace_end") or {}
    return end.get(end_key)


def _divergent(base: list, trial: list):
    """The first step at which the trial's series stops matching the baseline's (phase,
    d_eef, d_base); the first extra step when one simply runs longer; None when equal.
    A MISSING series is not a divergence: with an empty after-side the length branch below
    returned the baseline's first step, so 355 of the recycle_cans campaign's 365 trial
    rows told the model "第 1 步起与基线不同" (all of them "step 1") when the truth was
    that nothing was measured."""
    if not base or not trial:
        return None
    key = lambda r: (r.get("phase"), r.get("d_eef"), r.get("d_base"))
    for b, a in zip(base, trial):
        if key(b) != key(a):
            return a.get("step")
    longer = trial if len(trial) > len(base) else base
    return longer[min(len(base), len(trial))].get("step") if len(base) != len(trial) else None


def trial_evidence(before: dict, after: dict, node, seeds: list, exc: dict | None = None) -> dict:
    """What the TRIAL'S OWN CODE did, per trial seed: the target node's per-step ``trace``
    and ``geometry`` under the trial (the evidence the baseline already carries) plus the
    ``diff`` against that seed's BASELINE row -- ``{phase_changed (the trial's phase
    sequence when it differs), first_divergent_step, base_moved, d_eef_min_before/after,
    d_base_min_before/after, steps_before/after, ok_after, failure_mode_before/after
    (each present only when THAT side actually reported one)}``
    -- and the ``exception`` the executor
    raised (preflight or suite), whole. Without it a round only ever said "0 -> 0"."""
    ev = {"node": node, "exception": exc, "seeds": []}
    for s in seeds:
        b, a = _row(before, s, node), _row(after, s, node)
        sb, sa = _series(b), _series(a)
        pb, pa = _phases(sb), _phases(sa)
        xy = lambda r: ((r.get("base") or [0.0, 0.0]) + [0.0, 0.0])[:2]
        ev["seeds"].append({
            "seed": int(s),
            **{k: a[k] for k in ("trace", "geometry") if a.get(k) is not None},
            "diff": {"phase_changed": pa if pa != pb else [],
                     "first_divergent_step": _divergent(sb, sa),
                     "base_moved": (max(math.hypot(xy(r)[0] - xy(sa[0])[0],
                                                   xy(r)[1] - xy(sa[0])[1]) for r in sa) > 0.01)
                                   if sa else None,
                     "d_eef_min_before": _minimum(sb, "d_eef", b, "d_eef_target"),
                     "d_eef_min_after": _minimum(sa, "d_eef", a, "d_eef_target"),
                     "d_base_min_before": _minimum(sb, "d_base", b, "d_base_target"),
                     "d_base_min_after": _minimum(sa, "d_base", a, "d_base_target"),
                     "steps_before": b.get("steps"), "steps_after": a.get("steps"),
                     "ok_after": a.get("ok"),
                     # failure_mode rides ONLY when the row carries the key. An executor
                     # that reports no failure_mode leaves none (D.merge_executor_
                     # diagnostics), and "the candidate never answered" must not render as
                     # "reach_stall→无": over the campaign's 588 rounds the judged node
                     # read None on 364 of the 365 candidate trial rows, 347 of them at the
                     # segment cap, while the scripted baseline rows carried the stall in
                     # 580 of those rounds -- i.e. the old unconditional key said "cured"
                     # in every candidate round.
                     **{f"failure_mode_{w}": r["failure_mode"]
                        for w, r in (("before", b), ("after", a)) if "failure_mode" in r}}})
    return ev


def _evidence_summary(ev: dict | None) -> dict | None:
    """``last_outcome``'s copy of it: the numbers and the exception, no per-step series."""
    return ev and {"node": ev["node"], "exception": ev["exception"],
                   "seeds": [{"seed": r["seed"], **r["diff"]} for r in ev["seeds"]]}


# ── self-check: the candidate's own code, read statically before the simulator ────

class SelfCheckError(Exception):
    """A static finding about the candidate's code, raised where the doctor's findings
    are raised (inside the preflight) so it rides the model's repair loop."""


def _defs(node: ast.ClassDef) -> tuple[set, set]:
    """(what the class READS on self, what it ASSIGNS): literal ``self.x`` by context,
    plus its methods, its class attributes and ``setattr(self, "x", ...)``."""
    reads, writes = set(), set()
    for st in node.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            writes.add(st.name)
        elif isinstance(st, ast.Assign):
            writes |= {t.id for t in st.targets if isinstance(t, ast.Name)}
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            writes.add(st.target.id)
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
            (reads if isinstance(n.ctx, ast.Load) else writes).add(n.attr)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr" \
                and len(n.args) > 1 and isinstance(n.args[0], ast.Name) and n.args[0].id == "self" \
                and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
            writes.add(n.args[1].value)
    return reads, writes


def _imports(tree: ast.Module) -> dict:
    """``name -> object`` for the file's imports; whatever will not import is left out
    (a base we cannot resolve makes its class unjudgeable, never a finding)."""
    ns = {}
    for n in ast.walk(tree):
        try:
            if isinstance(n, ast.Import):
                for a in n.names:
                    ns[a.asname or a.name.split(".")[0]] = importlib.import_module(
                        a.name if a.asname else a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
                mod = importlib.import_module(n.module)
                for a in n.names:
                    if hasattr(mod, a.name):
                        ns[a.asname or a.name] = getattr(mod, a.name)
        except Exception:  # noqa: BLE001 -- unimportable: the name is simply unknown
            continue
    return ns


def _base_names(obj: type) -> set:
    """What an imported base supplies: its ``dir()`` plus every ``self.x`` its own
    source assigns (a base's ``__init__`` attributes are invisible to ``dir()``)."""
    names = set(dir(obj))
    for c in getattr(obj, "__mro__", ()):
        try:
            node = ast.parse(textwrap.dedent(inspect.getsource(c))).body[0]
        except (OSError, TypeError, SyntaxError, IndentationError):
            continue   # no source (C, exec'd): its dir() is all we know
        if isinstance(node, ast.ClassDef):
            names |= _defs(node)[1]
    return names


def _known(cls: ast.ClassDef, classes: dict, ns: dict, seen: tuple = ()) -> set | None:
    """Every attribute name the class can legitimately read on self -- its own
    assignments plus each base's -- or None when a base cannot be resolved."""
    out = _defs(cls)[1]
    for b in cls.bases:
        if isinstance(b, ast.Name) and b.id == "object":
            continue
        if isinstance(b, ast.Name) and b.id in classes and b.id not in seen:
            more = _known(classes[b.id], classes, ns, seen + (cls.name,))
        else:
            obj = ns.get(b.id) if isinstance(b, ast.Name) else (
                getattr(ns.get(b.value.id), b.attr, None)
                if isinstance(b, ast.Attribute) and isinstance(b.value, ast.Name) else None)
            more = _base_names(obj) if isinstance(obj, type) else None
        if more is None:
            return None
        out |= more
    return out


def _check_source(src: str, where: str) -> list[str]:
    tree = ast.parse(src)
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    ns, out = _imports(tree), []
    for cls in classes.values():
        known = _known(cls, classes, ns)
        if known is None:
            continue   # a base we cannot import: nothing is claimed about this class
        for a in sorted(_defs(cls)[0] - known):
            out.append(f"{where}: {cls.name} reads self.{a}, which {cls.name} never assigns and no "
                       f"base of it has -- initialise it (e.g. in __init__) or read what exists")
    return out


def self_check(tried: dict) -> str | None:
    """The candidate's OWN code, read statically before the simulator runs it: every
    attribute the new code reads on ``self`` must be assigned somewhere in its class (or
    come from a resolvable base -- the measured failure is ``self._last_d`` used and never
    initialised), and no edit may be a no-op. Returns the doctor-shaped refusal text (the
    repair loop's material) or None.

    Limits -- a cheap AST read, not a type checker: only literal ``self.x`` and
    ``setattr(self, "x", ...)`` count as assignments (a base that injects attributes
    dynamically is invisible); a class whose base does not import is skipped rather than
    guessed at; nothing about types, arguments or control flow is checked."""
    d = tried.get("detail") or {}
    for i, e in enumerate(d.get("edits") or (), 1):   # normally caught by apply_edits first
        if isinstance(e, dict) and e.get("old") == e.get("new"):
            return f"doctor:self-check: edit {i} changes nothing: old == new"
    findings: list[str] = []
    for py in sorted(Path(d["path"]).glob("*.py")) if d.get("path") else ():
        try:
            findings += _check_source(py.read_text(), py.name)
        except SyntaxError as exc:
            return f"doctor:self-check: {py.name} does not parse: {exc}"
    return ("doctor:self-check (static, no sim): " + "; ".join(findings[:5])) if findings else None


# ── try: the proposals inbox first, then the built-in proposer ────────────────────

def _none(reason: str, node=None, needs=("proposal",)) -> dict:
    """``needs`` = what WOULD give the proposer something to try next round."""
    return {"kind": "none", "node": node, "detail": {"reason": reason, "needs": list(needs)}}


def _binding(c: dict) -> dict:
    return {"ref": c["ref"], "params": dict(c.get("params") or {}),
            "transport": c.get("transport", "inproc")}


def death_nodes(before: dict, history: list | None = None) -> list[dict]:
    """Observed failure locations and their history, without selecting an intervention.

    An upstream actuator linked to a failed verification is an attribution hypothesis.
    The model can select another evidenced action from its installed capability catalog.
    """
    rows: dict[str, dict] = {}
    for seed, s in sorted(before["seeds"].items(), key=lambda kv: int(kv[0])):
        node = s.get("first_death")
        observed_node = node
        by_id = {n['id']: n for n in s.get('trail') or []}
        if node in by_id and by_id[node].get('kind') in ('verify', 'decide', 'perceive'):
            pending, seen = list(by_id[node].get('after') or []), set()
            while pending:
                parent = pending.pop()
                if parent in seen or parent not in by_id:
                    continue
                seen.add(parent)
                if by_id[parent].get('kind', 'manipulate') in ('segment', 'manipulate'):
                    node = parent
                    break
                pending.extend(by_id[parent].get('after') or [])
        if node is None and s.get('terminal_mismatch'):
            node = next((n['id'] for n in reversed(s.get('trail') or [])
                         if n.get('kind', 'manipulate') in ('segment', 'manipulate')), None)
        if not node:
            continue
        d = rows.setdefault(node, {"node": node, "seeds": [],
                                               "failure_mode": None, "rounds_targeted": 0,
                                               "last_round": 0, "failed_observations": []})
        d["seeds"].append(int(seed))
        if observed_node and observed_node not in d['failed_observations']:
            d['failed_observations'].append(observed_node)
        d["failure_mode"] = d["failure_mode"] or s.get("failure_mode")
    for r in history or ():
        d = rows.get((r.get("tried") or {}).get("node"))
        if d is not None:
            d["rounds_targeted"] += 1
            d["last_round"] = max(d["last_round"], int(r.get("round") or 0))
    return sorted(rows.values(), key=lambda d: d["node"])


def _first_death(before: dict, history: list | None = None):
    """A stable first failure observation for legacy readers; never a proposal default."""
    nodes = death_nodes(before, history)
    return nodes[0]["node"] if nodes else None


def _ok_map(suite: dict) -> dict[str, dict[str, bool]]:
    """``{seed: {node: passed?}}`` from each seed's node trail."""
    return {str(seed): {n["id"]: n.get("ok") is True for n in (s.get("trail") or [])}
            for seed, s in suite["seeds"].items()}


def take_proposal(session: Path, task: str, round_no: int) -> dict | None:
    """The oldest pending inbox proposal for ``task`` (board.store.proposals), stamped
    ``applied={round, ts}`` in place (atomic rewrite) so it is consumed exactly once."""
    for p in bs.proposals(session):
        if p["task"] == task and p["applied"] is None:
            path = session / "proposals" / f"{p['id']}.json"
            doc = json.loads(path.read_text())
            doc["applied"] = {"round": round_no, "ts": time.time()}
            drop(path.parent, path.name, json.dumps(doc, sort_keys=True))
            return {**p, "applied": doc["applied"]}
    return None


def from_proposal(p: dict, before: dict) -> dict:
    """Normalize an explicit model/operator candidate without choosing its target."""
    pay = dict(p["payload"])
    node = pay.pop("node", None)
    runs = [s["nodes"][node] for s in before["seeds"].values()
            if isinstance(node, str) and node in s["nodes"]]
    tag = {"proposal": p["id"], "note": p["note"]}
    if p['kind'] == 'plan':
        if not isinstance(pay.get('graph'), dict):
            return _none('plan proposal requires a graph object', node)
        return {'kind': 'plan', 'node': node, 'detail': {**tag, **pay}}
    if not isinstance(node, str) or not runs:
        return {"kind": "none", "node": node,
                "detail": {**tag, "reason": f"proposal requires an explicit node the suite ran ({node!r})"}}
    need = {"tunables": ("ref", "path", "to"), "executor": ("to",),
            "card": ("path", "to", "ref"), "plan": ("graph",)}[p["kind"]]
    if missing := [k for k in need if k not in pay]:
        return {"kind": "none", "node": node,
                "detail": {**tag, "reason": f"{p['kind']} proposal lacks {missing}"}}
    frm = runs[0]["executor"]
    if p["kind"] == "tunables":   # the knob's current value (None when the ref/path is unknown)
        frm = mount_params(pay["ref"])
        for k in pay["path"]:
            frm = frm.get(k) if isinstance(frm, dict) else None
        frm = frm if isinstance(frm, (int, float)) and not isinstance(frm, bool) else None
    return {"kind": p["kind"], "node": node,
            "detail": {"skill": runs[0]["skill"], "executor": runs[0]["executor"],
                       "from": frm, **tag, **pay}}


def apply(tried: dict, applied: dict) -> dict:
    out = {"executors": dict(applied["executors"]),
           "tunables": json.loads(json.dumps(applied["tunables"])),
           "cards": dict(applied.get("cards") or {})}
    if applied.get("graph"):
        out["graph"] = copy.deepcopy(applied["graph"])
    d = tried["detail"]
    if tried["kind"] == "executor":
        out["executors"][tried["node"]] = d["to"]
    elif tried["kind"] == "card":
        out["executors"][tried["node"]] = d["to"]
        out["cards"][d["to"]] = {"skill": d["skill"], "path": d["path"], "ref": d["ref"],
                                 "params": dict(d.get("params") or {}),
                                 "transport": d.get("transport", "inproc"),
                                 "artifact_sha": d.get('artifact_sha') or evolve_llm.candidate_digest(d['path'])}
    elif tried["kind"] == "tunables":
        cur = out["tunables"].setdefault(d["ref"], {})
        for p in d["path"][:-1]:
            cur = cur.setdefault(p, {})
        cur[d["path"][-1]] = d["to"]
    elif tried["kind"] == "plan":
        out["graph"] = copy.deepcopy(d["graph"])
    return out


def _media(session: Path, task: str, seeds: list, prefix: str = 'media') -> list[str]:
    """Session-relative paths of the clips kept so far (harness.media index), the
    list the board's rsi_frames face returns verbatim."""
    return [f"{prefix}/{task}/{seed}/{ent['file']}"
            for seed in range(int(seeds[0]), int(seeds[1]) + 1)
            for ent in media.index_of(session / prefix, task, seed).values()]


def _dropped(session: Path, task: str, seeds: list, prefix: str = 'media') -> dict[str, dict]:
    """``{"<seed>/<node>": {reason, keyframes: [session-relative paths]}}`` of the
    segments that left no clip -- the honest side of ``media`` (rsi_frames' ``dropped``)."""
    return {f"{seed}/{node}": {"reason": d["reason"],
                               "keyframes": [f"{prefix}/{task}/{seed}/{f}" for f in d["keyframes"]]}
            for seed in range(int(seeds[0]), int(seeds[1]) + 1)
            for node, d in media.dropped_of(session / prefix, task, seed).items()}


# ── the round loop ────────────────────────────────────────────────────────────────

_ZH = {"idle": "等待", "baseline": "基线评测", "propose": "选试验", "retest": "同种子复测",
       "confirm": "新种子确认", "publish": "发布", "done": "完成", "cancelled": "已取消",
       "failed": "失败", "paused_disk": "磁盘不足，已暂停"}


def _message(live: dict) -> str:
    """One short operator sentence for the live block (the page shows it verbatim)."""
    head = f"第 {live['round']} 轮 {_ZH.get(live['phase'], live['phase'])}"
    if live["phase"] == "paused_disk":
        return live.get("disk") or head
    if live['phase'] == 'failed':
        return f"LLM 提案失败，第 {live['round']} 轮已停止：{live.get('error') or '查看模型审计'}"
    if live["phase"] == "propose" and live.get("proposer") == "llm":
        head = f"LLM 分析第 {live['round']} 轮…"
    if live["phase"] == "done":
        return f"已完成 {live['round']} 轮"
    if live["phase"] == "cancelled":
        return f"第 {live['round']} 轮边界取消"
    t = live.get("tried")
    if t and live["phase"] in ("retest", "confirm", "publish"):
        head += f"（{t['kind']} @ {t['node']}）"
    if live["seed"] is not None:
        done = sum(n["ok"] is True for n in live.get("nodes") or [])
        head += (f"：种子 {live['seed']} 运行中" + (f" ({live['node']})" if live["node"] else "")
                 + (f" 节点 {done}/{len(live['nodes'])}" if live.get("nodes") else "")
                 + f"，{live['seed_index'] + 1}/{live['seeds_total']}")
    return head

def _evaluate_suite(out: dict, contract: dict) -> dict:
    """Attach only execution-owned verification observations to a fixed ruler."""
    out = copy.deepcopy(out)
    for row in out['seeds'].values():
        row['execution_success'] = row['success']
        row['evaluation'] = evaluation.evaluate(
            contract, row.get('verification_observations'), row.get('terminal_observation'))
        row['success'] = row['evaluation']['complete']
        if row['execution_success'] and not row['success']:
            row['terminal_mismatch'] = True
    out['count'] = sum(row['success'] for row in out['seeds'].values())
    out['sha'] = sha_json(out['seeds'])
    return out


def _diagnose_suite(out: dict) -> dict:
    findings, fingerprint, traces = [], set(), []
    for seed, row in out['seeds'].items():
        for observation in row.get('verification_observations') or []:
            if observation.get('blocked_reads'):
                findings.append({'kind': 'untrusted_verifier_dependency', 'channel': 'evaluation',
                                 'seed': seed, 'node': observation['node']['id'],
                                 'evidence': {'source': observation['source'],
                                              'blocked_reads': observation['blocked_reads']}})
                fingerprint.add('evaluation:untrusted_verifier_dependency')
        if row.get('terminal_mismatch'):
            findings.append({'kind': 'terminal_mismatch', 'channel': 'evaluation', 'seed': seed,
                             'evidence': {'execution_success': True, 'terminal': row['evaluation']['terminal']}})
            fingerprint.add('evaluation:terminal_mismatch')
        for node in row.get('trail') or []:
            if not node.get('trace'):
                continue
            diag = diagnosis.analyze_trace(node['trace'])
            traces.append({'seed': seed, 'node': node['id'], **diag})
            fingerprint.update(diag.get('fingerprint') or [])
            findings.extend({**f, 'seed': seed, 'node': node['id']} for f in diag['findings'])
    return {'status': 'observed' if findings else 'unknown', 'findings': findings,
            'fingerprint': sorted(fingerprint), 'traces': traces,
            'limits': 'Sampled controller observations support hypotheses, never reward or proof of physical impossibility.'}


def _validate_try(tried: dict, projection: dict, brief_data: dict, reference: dict) -> None:
    """Validate an explicit candidate against installed capabilities and the frozen task."""
    kind, detail = tried['kind'], tried['detail']
    if kind == 'none':
        return
    if kind == 'plan':
        interventions.validate_candidate(detail['graph'], reference, brief_data)
        return
    driver = (projection.get('drivers') or {}).get(tried['node'])
    if driver is None:
        raise ValueError('candidate must explicitly select an observed action from intervention_space')
    if kind == 'tunables':
        knobs = driver.get('tunables') or {}
        path = detail.get('path') or []
        values = knobs.get('values') or {}
        if (detail.get('ref') != knobs.get('ref') or not path
                or path[:-1] != knobs.get('path', []) or path[-1] not in values
                or type(detail.get('to')) not in (int, float) or not math.isfinite(detail['to'])):
            raise ValueError('tunables must name a finite installed driver parameter from intervention_space')
    elif kind == 'executor':
        if detail.get('to') not in driver.get('executors', {}):
            raise ValueError('executor must be bound to the target skill')
    elif kind == 'card':
        path = Path(detail['path'])
        if not path.is_dir():
            raise ValueError('candidate card directory is missing')
        if why := evolve_llm._doctor(path, detail['ref'], detail):
            raise ValueError(why)
        if why := self_check(tried):
            raise SelfCheckError(why)
        digest = evolve_llm.candidate_digest(path)
        if detail.get('artifact_sha') and detail['artifact_sha'] != digest:
            raise ValueError('candidate content differs from its proposal identity')
        detail['artifact_sha'] = digest
    else:
        raise ValueError(f'unsupported intervention {kind!r}')


def evaluator_identity(binding: dict, records: dict) -> dict:
    """Bind experiments to installed source and configuration, excluding generated candidates."""
    roots = {REPO_ROOT / 'harness', REPO_ROOT / 'plugins' / 'task', REPO_ROOT / 'plugins' / 'rsi'}
    for ref in binding.values():
        if isinstance(ref, str) and ':' in ref:
            spec = importlib.util.find_spec(ref.partition(':')[0])
            if spec and spec.origin and Path(spec.origin).is_file():
                roots.add(Path(spec.origin).parent)
    files = {str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p): sha_json(p.read_text())
             for root in roots for p in root.iterdir() if p.is_file() and p.suffix in ('.py', '.toml')}
    for p in (REPO_ROOT / 'scripts' / name for name in ('evolve.py', 'evolve_llm.py', 'evolve_evidence.py')):
        files[str(p.relative_to(REPO_ROOT))] = sha_json(p.read_text())
    return {'binding': binding, 'records': {k: to_plain(v) for k, v in records.items()},
            'sources': files}


def _policy_ancestor(learner: ProgramLearner, policy_id: str) -> bool:
    current = learner.selected_id
    while current is not None:
        if current == policy_id:
            return True
        current = learner.policies[current]['parent_id']
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Bounded online program-policy learning with fixed verification objectives')
    ap.add_argument('--mode', choices=MODES, default='execution')
    ap.add_argument('--task')
    ap.add_argument('--session', type=Path)
    ap.add_argument('--skills-root', type=Path)
    ap.add_argument('--seeds', type=int, nargs=2)
    ap.add_argument('--rounds', type=int, default=None,
                    help='Completed learning cycles in this brief; 0 means no cycle-count limit')
    ap.add_argument('--continuous', action='store_true',
                    help='Renew bounded cycle budgets until cancellation, error, or the round limit')
    ap.add_argument('--arm', default='auto')
    ap.add_argument('--cancel-marker', type=Path)
    ap.add_argument('--max-replans', type=int)
    ap.add_argument('--max-actuations', type=int)
    ap.add_argument('--confirm-seeds', type=int, default=2,
                    help='Additional development seeds for a task-success gain; never held-out installation evidence')
    ap.add_argument('--proposer', choices=('llm',), default='llm')
    ap.add_argument('--llm-model', help='Model ID served by the installed model endpoint')
    ap.add_argument('--llm-effort', default='off', help='Effort declared by the installed model endpoint')
    ap.add_argument('--max-model-calls', type=int, default=8)
    ap.add_argument('--max-input-bytes', type=int, default=96000)
    ap.add_argument('--max-output-tokens', type=int, default=4096)
    ap.add_argument('--max-probe-episodes', type=int, default=3)
    ap.add_argument('--gc', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    if args.rounds is None:
        args.rounds = 0 if args.continuous else 3
    if args.rounds < 0:
        ap.error('rounds must be nonnegative')
    if min(args.max_model_calls, args.max_input_bytes, args.max_probe_episodes) < 0 or args.max_output_tokens < 1:
        ap.error('model/input/probe budgets must be nonnegative; output tokens must be positive')
    try:
        llm_params, _ = evolve_llm.model_request_config(args.llm_model, args.llm_effort)
    except ValueError as exc:
        ap.error(str(exc))
    if args.gc:
        messages = gc_candidates(runs=args.session.parent if args.session else REPO_ROOT / 'runs',
                                 dry_run=args.dry_run)
        if args.session and args.task:
            messages += prune_audits(EvolveStore(args.session, args.task).dir / 'llm', dry_run=args.dry_run)
        print('\n'.join(messages) or 'gc: nothing to do')
        return 0
    if missing := [f'--{k}' for k in ('task', 'session', 'skills_root') if getattr(args, k) is None]:
        ap.error('required: ' + ', '.join(missing))
    if args.mode != 'evolution':
        print(json.dumps({'error': f'evolve refused in mode {args.mode!r}; assert --mode evolution'}))
        return 3
    binding = discover().task_bindings.get(args.task)
    if binding is None:
        raise SystemExit(f'no task binding for {args.task!r}')
    brief_data = hr.task_brief(args.task, binding)
    records, emb = hr._binding_records(binding), brief_data['embodiment']
    identity = evaluator_identity(binding, records)
    if keys := Counter(k for record in records.values() for k in record.bindings if emb not in record.bindings):
        emb = keys.most_common(1)[0][0]
    store = EvolveStore(args.session, args.task)
    doc = store.load() or {'task': args.task, 'session': args.session.name,
                           'seeds': list(args.seeds or [0, 1]), 'arm': args.arm,
                           'rounds': [], 'best': 0, 'cursor': 0,
                           'applied': {'executors': {}, 'tunables': {}}}
    if (doc.get('protocol_id') != evaluation.VERSION
            or (doc.get('evaluation_contract') or {}).get('context') != identity):
        # Old observations remain hypotheses across a source/evaluator change.
        # They never populate this decision's executable working policies or
        # comparisons. Tag legacy replay before clearing its original contract.
        for observation in doc.get('learning_replay', []):
            observation.setdefault('contract_sha', (doc.get('evaluation_contract') or {}).get('sha'))
            observation.setdefault('task', args.task)
        # The former ruler and its accepted state cannot silently become this experiment.
        if doc['rounds']:
            doc.setdefault('prior_protocols', []).append({
                'through_round': doc['cursor'], 'applied': doc['applied'],
                'accepted_stack': doc.get('accepted_stack', []), 'best': doc['best'],
                'reason': 'New frozen verifier protocol; historical rows retain their original meaning.'})
        doc.update(protocol_id=evaluation.VERSION, applied={'executors': {}, 'tunables': {}},
                   accepted_stack=[], best=0, epoch_start=int(doc['cursor']) + 1,
                   evaluation_contract=None, reference_plan=None, reference={}, confirm_base=None)
        doc.pop('memory_prefix', None)
        doc.pop('last_outcome', None)
        doc.pop('cycle_context', None)
    seeds, arm, applied = doc['seeds'], doc['arm'], doc['applied']
    epoch_start = doc['epoch_start']
    epoch_rows = lambda: [row for row in doc['rounds'] if row['round'] >= epoch_start]
    cycles = 0
    target = float('inf') if args.rounds == 0 else args.rounds
    doc.update(status='running', continuous=args.continuous, stop_reason=None)
    doc['llm_config'] = {'model': llm_params.get('model'), 'effort': args.llm_effort}
    doc['working_candidates'] = []  # Process-local measurements never become restored executable authority.
    doc['score_definition'] = ('Frozen independent verification vector; accept a strict component gain on paired seeds '
                               'only when no true component regresses. Node count and controller targets carry no reward.')
    contract = doc.get('evaluation_contract')
    reference_plan = doc.get('reference_plan')
    memory_path = args.session.parent / 'rsi-experience.json'
    memory_rows = experience.read_experiences(memory_path)
    memory_prefix = doc.setdefault('memory_prefix', max((r['sequence'] for r in memory_rows), default=0) + 1)
    prior_tasks = len({r['task'] for r in memory_rows if r['sequence'] < memory_prefix
                       and r['task'] != args.task
                       and r.get('evidence', {}).get('protocol_id') == evaluation.VERSION
                       and r.get('evidence', {}).get('evidence_policy') == evaluation.EVIDENCE_POLICY})
    now = time.time()
    live = doc['live'] = {'phase': 'idle', 'round': doc['cursor'], 'seeds_total': seeds[1] - seeds[0] + 1,
                          'seed_index': None, 'seed': None, 'node': None, 'nodes': [], 'messages': [],
                          'message': '', 'started_at': now, 'round_started_at': now, 'phase_started_at': now,
                          'seed_started_at': None, 'last_round_s': None, 'per_seed_partial': [],
                          'tried': None, 'proposer': args.proposer,
                          'sim_s': sum((row.get('usage') or {}).get('sim_s', 0) for row in doc['rounds'])}
    budgets = {'max_replans': args.max_replans, 'max_actuations': args.max_actuations}
    run_budget = {'scope': 'submitted_brief', 'limits': {
        'model_calls': None if args.continuous else args.max_model_calls,
        'input_bytes': None if args.continuous else args.max_input_bytes,
        'output_tokens_per_call': args.max_output_tokens,
        'probe_episodes': None if args.continuous else args.max_probe_episodes},
        'used': {'model_calls': 0, 'input_bytes': 0, 'probe_episodes': 0, 'full_evaluations': 0}}
    doc['run_budget'] = run_budget

    def cancelled():
        return args.cancel_marker is not None and args.cancel_marker.exists()

    def tick(**kw):
        if kw.get('phase', live['phase']) != live['phase']:
            kw = {'phase_started_at': time.time(), 'seed': None, 'seed_index': None,
                  'node': None, 'nodes': [], 'seed_started_at': None, 'per_seed_partial': [], **kw}
        live.update(kw)
        msg = _message(live)
        if msg != live['message']:
            live['message'] = msg
            live['messages'] = (live['messages'] + [{'ts': time.time(), 'text': msg}])[-20:]
        store.save(doc)

    def suite(seed_range, overlay, *, media_enabled=True, label=None):
        prefix = f"media/rsi/{args.task}/epoch-{epoch_start}/round-{live['round']}/{label or live['phase']}"
        out, started_episodes = None, 0
        started_sampling = time.monotonic()

        def sampling_tick(**kw):
            nonlocal started_episodes
            if kw.get('seed_started_at') is not None:
                started_episodes += 1
            tick(**kw)

        try:
            out = run_suite(args.task, binding, seed_range, arm, args.skills_root, overlay,
                            media_dir=args.session / prefix if media_enabled else None,
                            budgets=budgets, progress=sampling_tick, media_prefix=prefix)
        finally:
            # Failed sampling still costs time and started episodes. Reused
            # baselines never enter this wrapper and therefore cost nothing twice.
            elapsed = out['elapsed_s'] if out is not None else time.monotonic() - started_sampling
            attempts = max(started_episodes, len(out.get('seeds', {})) if out is not None else 0)
            round_sampling['episode_attempts'] += attempts
            round_sampling['sim_s'] += elapsed
            live['sim_s'] += elapsed
        out['experiment_id'] = sha_json({'binding': binding, 'arm': arm, 'budgets': budgets,
                                         'overlay': overlay, 'seeds': seed_range,
                                         'evaluator': contract['sha'] if contract else None})
        out['media'] = _media(args.session, args.task, seed_range, prefix) if media_enabled else []
        out['media_dropped'] = _dropped(args.session, args.task, seed_range, prefix) if media_enabled else {}
        return _evaluate_suite(out, contract) if contract else out

    tick()
    base, learner = None, None
    while cycles < target:
        if cancelled():
            doc.update(status='cancelled', stop_reason='cancelled')
            tick(phase='cancelled')
            return 3
        cycle_limits = {key: maximum if args.continuous else max(0, maximum - run_budget['used'][key])
                        for key, maximum in (('model_calls', args.max_model_calls),
                                             ('input_bytes', args.max_input_bytes),
                                             ('probe_episodes', args.max_probe_episodes))}
        cycle_budget = doc['cycle_budget'] = {
            'scope': 'learning_cycle', 'cycle': cycles + 1,
            'limits': {**cycle_limits, 'output_tokens_per_call': args.max_output_tokens},
            'used': {'model_calls': 0, 'input_bytes': 0, 'probe_episodes': 0, 'full_evaluations': 0}}
        if not all(cycle_limits.values()):
            doc['stop_reason'] = 'budget_exhausted'
            break
        if why := disk_guard(args.session, store.dir):
            doc.update(status='paused_disk', stop_reason='paused_disk')
            doc['cursor'] = int(doc['cursor']) + 1
            doc['rounds'].append({'round': doc['cursor'], 'tried': _none(why, needs=('disk',)),
                                  'before': doc['best'], 'after': doc['best'], 'best': doc['best'],
                                  'outcome': 'none', 'accepted': False, 'published': False,
                                  'accepted_reason': 'paused_disk', 'paused_disk': why,
                                  'needs': ['disk'], 'per_seed': [], 'ts': time.time()})
            tick(phase='paused_disk', round=doc['cursor'], disk=why)
            return 4
        for line in maintain(store, args.session):
            print(line, file=sys.stderr)
        rnd, started = int(doc['cursor']) + 1, time.time()
        round_started = time.monotonic()
        round_sampling = {'episode_attempts': 0, 'sim_s': 0.0}
        previous_efficiency = experience.development_report(epoch_rows(), epoch_start=epoch_start)
        tick(phase='baseline' if base is None else 'propose', round=rnd, round_started_at=started,
             cycle=cycles + 1, tried=None)
        try:
            before = base or suite(seeds, applied)
        except Exception as exc:
            doc.update(status='failed', stop_reason='infrastructure_error')
            tick(phase='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        if contract is None:
            reference_plan = before.get('plan')
            if not reference_plan:
                raise ValueError('baseline produced no server plan from which to freeze evaluation')
            contract = evaluation.compile_contract(reference_plan, records, task=args.task,
                                                   predicates=brief_data.get('predicates'),
                                                   terminal_ref=brief_data['embodiment'], identity=identity)
            doc['evaluation_contract'], doc['reference_plan'] = contract, reference_plan
            doc.setdefault('evaluation_contracts', {})[contract['sha']] = contract
            before = _evaluate_suite(before, contract)
            before['experiment_id'] = sha_json({'binding': binding, 'arm': arm, 'budgets': budgets,
                                                 'overlay': applied, 'seeds': seeds,
                                                 'evaluator': contract['sha']})
        tick(phase='propose')
        os.environ[OVERRIDE_ENV] = json.dumps(applied['tunables'])
        diag = _diagnose_suite(before)
        retrieved = experience.retrieve_experiences(memory_path, task=args.task, diagnosis=diag,
                                                    before_sequence=memory_prefix,
                                                    protocol_id=evaluation.VERSION,
                                                    evidence_policy=evaluation.EVIDENCE_POLICY)
        doc['diagnosis'], doc['experience'] = diag, {'retrieved': retrieved}
        doc['plan_space'] = interventions.plan_space(reference_plan, brief_data)

        def project_policy(overlay, observed, *, previous_efficiency=previous_efficiency,
                           round_sampling=round_sampling):
            # Resolve source/parameters against the actual working parent. Source
            # evidence for the incumbent cannot authorize an unrelated branch.
            os.environ[OVERRIDE_ENV] = json.dumps(overlay.get('tunables', {}))
            try:
                view = evolve_llm.rsi_projection({**doc, 'applied': overlay}, observed, records, emb,
                                                 arm, binding, observed.get('logs') or [])
            finally:
                os.environ[OVERRIDE_ENV] = json.dumps(applied.get('tunables', {}))
            view.update(plan_space=doc['plan_space'], diagnosis=_diagnose_suite(observed),
                        experience={'retrieved': retrieved}, learning_replay=doc.get('learning_replay', []),
                        cycle_context={**doc.get('cycle_context', {}), 'continuous': args.continuous,
                                       'cycle': cycles + 1},
                        declared_development_seeds=list(range(seeds[0], seeds[1] + 1)),
                        development_cost={'accepted_updates': previous_efficiency['accepted_updates'],
                                          'previous_rounds': previous_efficiency['cost'],
                                          'current_round': copy.deepcopy(round_sampling)})
            return view

        probe_media, probe_dropped = [], {}

        def run_policy(seed_range, overlay, scope):
            if cancelled():
                raise RuntimeError('experiment cancelled')
            index = len(learner.probes)
            label = f'probe-{index}' if scope == 'probe' else 'retest'
            tick(phase='retest', experiment_scope=scope, probe_index=index,
                 **({'tried': learner.probes[-1]['tried']} if scope == 'probe' else {}))
            try:
                out = suite(seed_range, overlay, label=label)
                if scope == 'probe':
                    probe_media.extend(out.get('media', []))
                    probe_dropped.update({f'{label}/{k}': v for k, v in out.get('media_dropped', {}).items()})
                return out
            finally:
                os.environ[OVERRIDE_ENV] = json.dumps(applied['tunables'])
                tick(phase='propose', experiment_scope=None)

        if (learner is not None and learner.initial_id == learner._identity(applied)
                and learner.seeds == seeds and learner.contract['sha'] == contract['sha']
                and learner.baseline.get('experiment_id') == before.get('experiment_id')):
            learner.begin_cycle(max_probes=cycle_limits['probe_episodes'], project=project_policy, run=run_policy)
        else:
            learner = ProgramLearner(applied=applied, baseline=before, contract=contract, seeds=seeds,
                project=project_policy, validate=lambda t, p: _validate_try(t, p, brief_data, reference_plan),
                apply=apply, run=run_policy, observations=per_seed,
                max_probes=cycle_limits['probe_episodes'])
            evidence_working_set = evolve_llm.EvidenceWorkingSet(evolve_llm.AGENT_BUDGET['max_working_set_bytes'])
        projection = learner.projection()

        prop, llm, proposer = take_proposal(args.session, args.task, rnd), None, 'inbox'
        if prop:
            tick(proposer='inbox')
            tried = from_proposal(prop, before)
        else:
            proposer = 'llm'
            tick(proposer='llm')
            tried, llm = evolve_llm.llm_propose(None, projection, before, rnd,
                store.dir / 'llm', session=args.session, max_tokens=args.max_output_tokens,
                agent_tools={'trial': learner.trial, 'choose': learner.choose,
                             'projection': learner.projection, 'baseline': learner.observations},
                evidence=evidence_working_set,
                model=args.llm_model, effort=args.llm_effort,
                budget={'max_calls': cycle_limits['model_calls'],
                        'max_input_bytes': cycle_limits['input_bytes'],
                        'max_output_tokens': args.max_output_tokens})
            used = (llm.get('budget') or {}).get('used') or {}
            cycle_budget['used']['model_calls'] = used.get('calls', 0)
            cycle_budget['used']['input_bytes'] = used.get('input_bytes', 0)
            run_budget['used']['model_calls'] += used.get('calls', 0)
            run_budget['used']['input_bytes'] += used.get('input_bytes', 0)
        cycle_budget['used']['probe_episodes'] = len(learner.probes)
        cycle_budget['used']['full_evaluations'] = learner.full_calls
        run_budget['used']['probe_episodes'] += len(learner.probes)
        run_budget['used']['full_evaluations'] += learner.full_calls
        model_error = (llm or {}).get('status') == 'error'
        after, trial, trial_row, exc_info, confirm = before, applied, None, None, None
        try:
            if tried['kind'] != 'none':
                if proposer == 'llm':
                    if learner.selected_suite is None:
                        raise ValueError('model candidate lacks a selected full paired evaluation')
                    trial, after = learner.selected_overlay, learner.selected_suite
                else:
                    _validate_try(tried, projection, brief_data, reference_plan)
                    trial = apply(tried, applied)
                    tick(phase='retest', tried=tried)
                    after = suite(seeds, trial)
                    run_budget['used']['full_evaluations'] += 1
                    cycle_budget['used']['full_evaluations'] += 1
                trial_row = {'scope': 'full', 'seeds': list(range(seeds[0], seeds[1] + 1)),
                             'target_pass': sum(m.get(tried['node'], False) for m in _ok_map(after).values())}
        except Exception as exc:  # noqa: BLE001 -- candidate failures are experimental outcomes
            exc_info = _exception(exc)
            tried['detail']['error'] = str(exc)
        compared = evaluation.compare(before, after, contract)
        accepted = tried['kind'] != 'none' and exc_info is None and compared['accepted']
        why = compared['reason'] if exc_info is None else f"candidate rejected: {exc_info['message']}"
        if tried['kind'] == 'none':
            why = tried['detail'].get('reason') or 'no candidate proposed'
        if accepted and after['count'] > before['count'] and args.confirm_seeds > 0:
            # These are additional DEVELOPMENT seeds. Installation remains a separate battery.
            cs = [seeds[1] + 1, seeds[1] + args.confirm_seeds]
            tick(phase='confirm')
            try:
                cb, ca = suite(cs, applied, media_enabled=False), suite(cs, trial, media_enabled=False)
                check = evaluation.compare(cb, ca, contract)
                confirm = {'seeds': cs, 'before': cb['count'], 'after': ca['count']}
                confirm['evaluation'] = {'objective_id': contract['sha'],
                                          'before': per_seed(cb), 'after': per_seed(ca)}
                confirm['experiments'] = {'before': cb['experiment_id'], 'after': ca['experiment_id']}
                missing_terminal = any(s['evaluation']['terminal'] is None
                                       for su in (cb, ca) for s in su['seeds'].values())
                if check['regressions'] or check['before'] is None or missing_terminal:
                    accepted, why = False, 'additional development seeds regressed'
                    if missing_terminal:
                        why = 'additional development task-terminal evidence unavailable'
            except Exception as exc:  # noqa: BLE001 -- failed measurement must reject and retain evidence
                exc_info = _exception(exc)
                accepted, why = False, 'additional development evaluation failed'
                confirm = {'seeds': cs, 'before': None, 'after': None, 'error': exc_info['message']}
            seeds[1] = cs[1]
            tick(seeds_total=seeds[1] - seeds[0] + 1)
            base = None
        recorded = None
        if trial_row and exc_info is None:
            ancestry = [state['tried'] for key, state in learner.policies.items()
                        if state['tried'] is not None and _policy_ancestor(learner, key)] or [tried]
            strategies = [experience.intervention_strategy(
                {**atom, 'detail': {**atom['detail'], 'reference_graph': reference_plan}},
                summary=((llm or {}).get('summary') or atom['detail'].get('note', '')) if len(ancestry) == 1 else '',
                reference=f'{args.session.name}/{args.task}#round-{rnd}') for atom in ancestry]
            strategy = strategies[0] if len(strategies) == 1 else {
                'kind': 'composition', 'scope': 'program', 'reference': strategies[0]['reference'],
                'summary': 'Jointly evaluated program edits; individual causal effects are not established. '
                           + ' '.join(s['summary'] for s in strategies)}
            recorded = experience.record_experience(memory_path, task=args.task, diagnosis=diag,
                intervention=strategy,
                accepted=accepted, evidence={'before_sha': before['sha'], 'after_sha': after['sha'],
                    'round': rnd, 'session': args.session.name, 'suite_scope': 'full',
                    'protocol_id': evaluation.VERSION, 'evidence_policy': evaluation.EVIDENCE_POLICY})
        if accepted:
            applied = trial
            doc['accepted_stack'].append({'round': rnd, 'kind': tried['kind'],
                'detail': {'node': tried['node'], **tried['detail']},
                'policy_id': learner.selected_id,
                'ancestry': [{**{k: state[k] for k in ('parent_id', 'tried')}, 'policy_id': key}
                             for key, state in learner.policies.items() if state['tried'] is not None
                             and _policy_ancestor(learner, key)]})
        kept = after if accepted else before
        measured_after = trial_row is not None
        before_sum = evaluation.summary(before, contract)
        after_sum = evaluation.summary(after, contract) if measured_after else None
        before_score = [before_sum['successes'], before_sum['progress']]
        after_score = [after_sum['successes'], after_sum['progress']] if after_sum else None
        outcome = 'error' if model_error else 'none' if tried['kind'] == 'none' else ('improved' if compared['accepted'] else
                  'worse' if compared['regressions'] else 'same')
        eval_row = {'protocol_id': evaluation.VERSION, 'objective_id': contract['sha'],
                    'before': before_sum, 'after': after_sum,
                    'acceptance': {'accepted': accepted, 'reason': why},
                    'installation': {'status': 'not_evaluated',
                        'reason': 'Paired development evidence only; blind twin, held-out and sensing degradation battery required.'}}
        prior = epoch_rows()
        evidence = trial_evidence(before, after, tried['node'], (trial_row or {}).get('seeds', []), exc_info)
        candidate_id = learner.selected_id or (learner._identity(trial) if measured_after else None)
        cycle_reason = (llm or {}).get('stop_reason')
        cycle_outcome = ('error' if model_error else 'updated' if accepted else
                         'budget_exhausted' if cycle_reason == 'budget_exhausted' else
                         'no_update' if tried['kind'] != 'none' or learner.probes
                         or (llm or {}).get('status') == 'rejected' else 'abstained')
        row = {'round': rnd, 'tried': tried, 'before': before['count'],
               'after': after['count'] if measured_after else None,
               'best': max(doc['best'], kept['count']), 'suite_sha': after['sha'] if measured_after else None,
               'experiments': {'before': before.get('experiment_id'),
                               'after': after.get('experiment_id') if measured_after else None},
               'published': False, 'accepted': accepted, 'accepted_reason': why,
               'before_score': before_score, 'after_score': after_score, 'outcome': outcome,
               'parent': max((r['round'] for r in prior if r.get('accepted')), default=0),
               'layer': tried['detail'].get('layer'), 'notes': tried['detail'].get('notes'),
               'trial': trial_row, 'trial_evidence': evidence, 'confirm': confirm,
               'regression': {'lost': compared['regressions']}, 'burned': [],
               'evaluation': eval_row, 'diagnosis': diag,
               'experience': {'retrieved': retrieved, 'recorded': recorded},
               'learning': learner.report(), 'run_budget': copy.deepcopy(run_budget),
               'cycle_budget': copy.deepcopy(cycle_budget), 'cycle_outcome': cycle_outcome,
               'stop_reason': cycle_reason, 'memo': str((llm or {}).get('memo') or '')[-1000:],
               'policy': {'before_id': learner.initial_id, 'candidate_id': candidate_id,
                          'active_id': candidate_id if accepted else learner.initial_id,
                          'parent_id': learner.policies[learner.selected_id]['parent_id'] if learner.selected_id
                                       else learner.initial_id if candidate_id else None,
                          'updated': accepted, 'representation': 'program_overlay'},
               'usage': {'llm_tokens': (llm or {}).pop('usage', None),
                         'model_calls': llm.get('calls') if llm is not None else 0,
                         'input_bytes': ((llm.get('budget') or {}).get('used', {}).get('input_bytes')
                                         if llm is not None else 0),
                         'episode_attempts': round_sampling['episode_attempts'],
                         'sim_s': round(round_sampling['sim_s'], 3), 'sim_s_saved': 0.0,
                         'wall_s': round(time.monotonic() - round_started, 3)},
               'per_seed': per_seed(before), 'after_seeds': per_seed(after) if measured_after else [],
               'needs': tried['detail'].get('needs', []) if tried['kind'] == 'none' else [],
               'media': list(dict.fromkeys([*before.get('media', []), *probe_media, *after.get('media', [])])),
               'media_dropped': {**probe_dropped, **{f'before/{k}': v for k, v in before.get('media_dropped', {}).items()},
                                 **({f'after/{k}': v for k, v in after.get('media_dropped', {}).items()}
                                    if measured_after else {})},
               'proposal': {k: prop[k] for k in ('id', 'kind', 'note')} if prop else None,
               'proposer': proposer, 'llm': llm, 'ts': time.time()}
        if row['usage']['model_calls'] == 0:
            row['usage']['llm_tokens'] = {'prompt': 0, 'completion': 0}
        transfer = row['transfer'] = {**experience.development_report([*prior, row], epoch_start=epoch_start),
                    'prior_tasks': prior_tasks, 'memory_prefix': memory_prefix,
                    'condition': 'warm' if prior_tasks else 'cold',
                    'claim': 'One development task; no paired transfer or scale claim.'}
        doc['rounds'].append(row)
        cycles += 1
        previous_context = doc.get('cycle_context') or {}
        doc['cycle_context'] = {'previous_round': rnd, 'cycle_outcome': cycle_outcome,
                                'stop_reason': cycle_reason, 'reason': why[:1000], 'memo': row['memo'],
            'cycles_without_update': 0 if accepted else previous_context.get('cycles_without_update', 0) + 1,
            'cycles_without_sample': 0 if learner.probes else previous_context.get('cycles_without_sample', 0) + 1,
            'cycles_without_full_evaluation': 0 if learner.full_calls else previous_context.get('cycles_without_full_evaluation', 0) + 1}
        doc.update(cursor=rnd, applied=applied, best=row['best'], evaluation=eval_row, transfer=transfer,
                   last_outcome={k: row[k] for k in ('round', 'layer', 'outcome', 'accepted', 'accepted_reason',
                                                   'before_score', 'after_score', 'trial_evidence')})
        doc['last_outcome']['regressions'] = compared['regressions']
        doc['last_outcome']['trial_evidence'] = _evidence_summary(evidence)
        replay = doc.setdefault('learning_replay', [])
        for p in learner.probes:
            previous = next((r for r in replay if r.get('contract_sha') == contract['sha']
                             and r.get('policy_id') == p['policy_id'] and r.get('seeds') == p['seeds']), None)
            if previous:
                replay.remove(previous)
            replay.append({'round': rnd, 'task': args.task, 'contract_sha': contract['sha'], 'seeds': p['seeds'],
                'tried': intervention_summary(p['tried']), 'parent_id': p['parent_id'],
                'comparison': p.get('comparison'), 'error': p.get('error'),
                'measurement_sha': p.get('measurement_sha'), 'policy_id': p['policy_id'],
                'sample_count': (previous or {}).get('sample_count', 0) + 1})
        doc['learning_replay'] = replay[-24:]
        doc['working_candidates'] = [intervention_summary(s['tried']) for s in learner.policies.values()
                                     if s['tried'] is not None]
        update_reference(doc, kept, rnd)
        if model_error:
            doc.update(status='failed', stop_reason='model_error')
            tick(phase='failed', error=llm.get('reason'), last_round_s=round(time.time() - started, 1))
            print(json.dumps({'task': args.task, 'cursor': rnd, 'status': 'failed',
                              'error': llm.get('error'), 'reason': llm.get('reason')}), file=sys.stderr)
            return 5
        tick(phase='idle', last_round_s=round(time.time() - started, 1))
        base = None if confirm else kept
        if cycles >= target:
            doc['stop_reason'] = 'round_limit'
            break
        if (not args.continuous and any(run_budget['used'][key] >= run_budget['limits'][key]
                                      for key in ('model_calls', 'input_bytes', 'probe_episodes'))):
            doc['stop_reason'] = 'budget_exhausted'
            break
        if proposer == 'llm' and not any(cycle_budget['used'].values()):
            # A request that cannot fit even once will not become executable by
            # resetting an identical allowance. Never spin through empty cycles.
            doc['stop_reason'] = 'budget_exhausted'
            break
    doc['status'] = 'done'
    tick(phase='done')
    print(json.dumps({'task': args.task, 'cursor': doc['cursor'], 'best': doc['best'],
                      'protocol_id': evaluation.VERSION, 'status': doc['status']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
