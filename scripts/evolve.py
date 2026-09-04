"""The lightweight RSI loop: look -> try -> re-run -> publish, one round at a time.

Spawned by ``scripts/harness_runtime._run_evolve`` (an ``evolve`` brief, evolution
mode only) as its own process group. One round: run the task's seed suite in-process
(the SAME ``_mount_plan``/``task_brief``/``workload.run`` a task brief uses), read
each seed's first-death node + fault signature + per-node executor, let the LLM
proposer (scripts/evolve_llm: the model reads the round's trails + log excerpt and
answers a tunables / executor / code-as-policy card / driver-patch proposal, repairing a rejected one
up to 3 times from the exact error -- a card is preflighted on the first seed alone;
``--proposer rules`` or an unreachable endpoint falls back to) the built-in rules proposer pick ONE change -- the first-death node's executor (a bound policy whose
``evidence.by_executor`` beats the measured rate) else a one-dimensional +/-20%
tunables perturbation of that node's driver (its card's mount params, applied via
``PH_MOUNT_PARAMS_OVERRIDE``) else nothing, with the honest reason -- re-run the
same seeds (the TARGET NODE'S CLUSTER first: a trial that does not make that node
pass stops there and never spends the rest of the suite), and score the result as
``(successes, milestones, target_pass)`` (``score``). Two levels follow from it: the
round is ACCEPTED -- its change joins ``accepted_stack``/``applied`` and becomes the
next round's baseline -- when the score rose and no node that passed stopped passing
(``regressions``); it is PUBLISHED, whole-task evidence into the skill record, only
when the success count itself improves: the skill record with
the measured ``by_executor`` row folded in goes through the evolution-only skills
root door (``InMemorySkillGraph.publish``, the same one scripts/publish_plans.py
uses). Every round lands atomically in ``campaigns/evolve-<task>/campaign.json``
(rounds[], best, cursor, status, ``reference`` = the last successful pass of every
plan segment) -- BOUNDED: the header plus the last ``ROUNDS_KEPT`` rounds in full,
every older round written once to ``rounds/<round>.json`` and left in the file as a
compact ``index_row`` (see ``EvolveStore``). The same bound is kept on the two other
things that grew without one -- the llm audits (``prune_audits``) and the candidate
cards (``gc_candidates``, also ``--gc``) -- and a round does not start at all when
``disk_guard`` says the disk or the campaign dir is over the line (``paused_disk``).
The round row carries the kept suite's per-seed summary
(``per_seed``) and, when nothing was tried, ``needs`` -- what would unblock the
proposer, plus ``stuck`` when the round's node has taken ``STUCK_ROUNDS`` rounds
without improving (the brief then widens what may be patched there). A trial that
improves is re-scored over its ORIGIN FAILURE CLUSTER (``regression``, the baseline
seeds that shared its first missing milestone) before the fresh-seed confirm and is
not published when it lost one of them; a confirm seed that blocks a publish is burned
into the dev seeds (``burned``) and fresh confirm seeds are drawn above them. The
model's causal ``layer`` and its notebook ``notes`` ride the round row too; the runtime seals the ``rsi_step`` rows off it. The same file carries a
``live`` block (phase / seed / node / partial per-seed, rewritten at every phase and
seed boundary): live state the board's rsi_run shows, never sealed.
With ``PH_RSI_FRAMES`` set (the runtime passes its frame.jpg when --frames is on)
the suite's episodes are mirrored to that file, same one-writer lock as an rsi chain.
The ``proposals/`` inbox (board.store.submit_proposal) comes first: a pending entry
for this task is consumed at the start of the round (``rsi_proposal_applied``) and
tried instead of the built-in proposer -- a ``card`` proposal mounts its candidate
dir through ``PH_PLUGINS_EXTRA`` for that round's suite and, if it wins, its
binding is published into the record.
Cancel is checked at the round boundary (``--cancel-marker``); a resubmitted task
resumes from ``cursor``. Media never enters this file's outputs beyond paths read
from ``media/<task>/<seed>/index.json``.

    scripts/evolve.py --mode evolution --task kitchen_thaw --session runs/session-x \\
        --skills-root runs/session-x/skills --seeds 1 2 --rounds 3 --arm auto
    scripts/evolve.py --gc [--dry-run]        # the maintenance pass alone, for the operator
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

from harness.config import Mount, Patch, Profile, resolve_plan, sha_json
from harness.definitions import CAPABILITIES
from harness.events import SessionLog
from harness import media
from harness.kernel import Kernel
from harness.manifest import discover, mount_params
from harness.protocol import SkillRecordV0, to_plain
from harness.registry import load_provider
from harness.skill_library import rearm, segment_specs
from plugins.graphs import InMemorySkillGraph
from plugins.task import workload
from scripts import harness_runtime as hr
from scripts.brief_drop import drop
from scripts import evolve_llm
from scripts.rsi_campaign import _maybe_arm_frames
from board import store as bs

MODES = ("execution", "evolution")
#: JSON ``{provider ref: {param: value}}`` merged over a card's mount params by
#: ``harness.manifest.mount_params`` -- how a tunables trial reaches a driver.
OVERRIDE_ENV = "PH_MOUNT_PARAMS_OVERRIDE"
#: Extra card roots (harness.manifest.discover); a ``card`` proposal appends its
#: candidate dir for the round's suite, on top of whatever the process was given.
EXTRA_ENV = "PH_PLUGINS_EXTRA"
_BASE_EXTRA = os.environ.get(EXTRA_ENV, "")
PLANNER_REF = "scripts.evolve:planner_provider"
#: Consecutive rounds with nothing tried that end a BOUNDED loop (--rounds > 0: status
#: done, needs on the row). An unbounded loop (--rounds 0, the console's 开始/继续)
#: never stops on its own -- the operator's 停止 is the only end -- so a model that has
#: nothing to try is throttled instead: wait NONE_BACKOFF_S[0] after the first empty
#: round, doubling up to NONE_BACKOFF_S[1], reset by the next real try.
MAX_NONE = 2
NONE_BACKOFF_S = (float(os.environ.get("PH_NONE_BACKOFF_S", "60")), 600.0)
#: Rounds that targeted ONE node without improving before the round is called stuck
#: (brief keys ``stuck_rounds`` / ``stuck``): the brief then says parameter tweaks on
#: that node are exhausted and widens the patchable modules to the card's whole stage
#: pipeline (``pipeline_modules``), so the model changes code or targets another node.
STUCK_ROUNDS = 6


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
               "burned", "suite_sha", "proposal", "ts")
_TRIED_KEYS = ("skill", "ref", "path", "from", "to", "module", "executor", "reason",
               "hint", "error", "layer", "needs", "match", "name", "patch_sha")
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
                          if k in ("model", "summary", "reason")}, "summary", "reason") or None,
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

_AUDIT_KEEP = ("round", "model", "prompt_sha", "raw_sha", "summary", "rationale",
               "reason", "usage", "calls")


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
        blob = json.dumps([doc.get("applied"), doc.get("accepted_stack"),
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
    def __init__(self, inner, executors: dict) -> None:
        self._inner, self._executors = inner, dict(executors)

    def plan(self, brief):
        plan = dict(self._inner.plan(brief))
        plan["nodes"] = [{**n, "executor": self._executors[n["id"]]}
                         if n.get("id") in self._executors else n
                         for n in plan.get("nodes") or ()]
        return plan

    def __getattr__(self, name):
        return getattr(self._inner, name)


def planner_provider(inner: str, inner_params=None, executors=None) -> _Forced:
    return _Forced(load_provider(inner, dict(inner_params or {})), executors or {})


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


def _mount(binding: dict, skills_root: Path, executors: dict):
    plan = hr._mount_plan(binding, skills_root, frames=_maybe_arm_frames())
    if not executors:
        return plan
    m = next(m for m in plan.mounts if m.capability == "task.planner")
    forced = Mount("task.planner", PLANNER_REF,
                   {"inner": m.provider, "inner_params": dict(m.params),
                    "executors": dict(executors)})
    return resolve_plan(Profile("evolve", plan.mounts),
                        patches=(Patch("evolve", override=(forced,)),))


# ── look: the seed suite, in-process ──────────────────────────────────────────────

def _get(budgets, binding: dict, key: str, default):
    """Budget precedence: the brief's value, else the task binding's, else the default."""
    v = (budgets or {}).get(key)
    return binding.get(key, default) if v is None else v


def run_suite(task: str, binding: dict, seeds: list | None, arm: str, skills_root: Path,
              applied: dict, media_dir: Path | None = None, budgets: dict | None = None,
              progress=None, seed_list: list | None = None) -> dict:
    """{count, seeds: {seed: {success, first_death, fault, nodes}}, sha}. ``media_dir``
    (<session>/media) turns on the workload's segment recorder: kept-on-success clips.
    ``progress(**live)`` is called at every seed boundary and node change.
    ``seeds`` is the inclusive range [lo, hi]; ``seed_list`` names the seeds explicitly
    instead (a focused trial's cluster is not contiguous)."""
    os.environ[OVERRIDE_ENV] = json.dumps(applied["tunables"])
    cards = applied.get("cards") or {}
    os.environ[EXTRA_ENV] = ":".join(r for r in (_BASE_EXTRA, *(c["path"] for c in cards.values())) if r)
    per, logs = {}, []
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
        tick(seed_index=i, seed=seed, node=None, nodes=[], seed_started_at=t_seed)
        log = _Tap(lambda nodes: tick(nodes=nodes, node=next(
            (n["id"] for n in nodes if n["ok"] is not True), None)))
        kernel = Kernel(CAPABILITIES, log=log)
        kernel.mount(_mount(binding, skills_root, applied["executors"]))
        out = workload.run(dict(brief), kernel, seed=seed,
                           max_replans=int(_get(budgets, binding, "max_replans", 3)),
                           max_actuations=int(_get(budgets, binding, "max_actuations", 3)),
                           segment_retries=int(binding.get("segment_retries", 0)))
        skills = {}
        for r in log.rows():
            if r["kind"] == "task.plan" and r["data"].get("graph"):
                skills.update({n["id"]: n["skill"] for n in r["data"]["graph"].get("nodes") or []})
        nodes, faults = out["nodes"], out.get("faults") or []
        dead = next((nid for nid, n in nodes.items() if not n["success"]), None)
        per[str(seed)] = {
            "success": bool(out["success"]),
            "elapsed_s": round(time.time() - t_seed, 1),
            "trail": [{k: n[k] for k in ("id", "ok", "steps", "failure_mode", "after", "kind", "task")}
                      for n in log.nodes],
            "first_death": dead,
            "failure_mode": (nodes[dead].get("diagnostics") or {}).get("failure_mode") if dead else None,
            "fault": {k: faults[0].get(k) for k in ("kind", "node", "msg")} if faults else None,
            # the first-death node's failure keyframes (session-relative paths; the LLM brief's images)
            "keyframes": [f"media/{task}/{seed}/{f}" for f in (media.dropped_of(media_dir, task, seed)
                                                                .get(dead) or {}).get("keyframes", [])]
            if media_dir is not None and dead else [],
            "nodes": {nid: {"skill": skills.get(nid), "success": bool(n["success"]),
                            "executor": n.get("executor") or "scripted",
                            "tunables_sha": (n.get("diagnostics") or {}).get("tunables_sha")}
                      for nid, n in nodes.items()}}
        for n in per[str(seed)]["trail"]:   # final state from the result: a replan reset the live
            r = nodes.get(n["id"]) or {}      # trail, and the verify row carries no steps/diagnostics
            diag = r.get("diagnostics") or {}
            if n["ok"] is None and "success" in r:
                n["ok"] = bool(r["success"])
            n["steps"] = n["steps"] if n["steps"] is not None else r.get("steps")
            # ``failure_mode`` only when somebody measured it: an executor that seals none
            # leaves NO key (D.merge_executor_diagnostics), and that absence has to survive
            # all the way to _trial_line, which renders it "测不到". A None here reads as
            # "no stall" -- the fake cure every candidate round used to be told about.
            if "failure_mode" in diag or n["failure_mode"] is not None:
                n["failure_mode"] = n["failure_mode"] or diag.get("failure_mode")
            else:
                del n["failure_mode"]
            if (diag.get("trace") or {}).get("end"):   # where this segment ENDED, every node:
                n["trace_end"] = diag["trace"]["end"]  # the reference index + the upstream row
            if diag.get("trace") and n["id"] == dead:
                n["trace"] = diag["trace"]   # the stall geometry (with the per-step ``series``)
                n["geometry"] = diag.get("geometry")   # and the target's provenance.
        _link_upstream(per[str(seed)]["trail"], dead, skills)
        logs += evolve_llm._log_excerpt(seed, log.rows(), dead,
                                        evolve_llm.MAX_LOG_LINES // len(seq))
        tick(per_seed_partial=per_seed({"seeds": per}))
    return {"count": sum(s["success"] for s in per.values()), "seeds": per, "sha": sha_json(per),
            "elapsed_s": round(time.time() - t_suite, 3),
            "logs": logs}   # the dying nodes' fault/verify rows: the LLM proposer's log excerpt


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
    """The DISTINCT first-death nodes of this round's seeds, LEAST-RECENTLY-TARGETED
    first: ``[{node, seeds, failure_mode, rounds_targeted, last_round}]``. Rotating over
    this list is what keeps one node (the commonest death) from eating the campaign --
    with 4243 dying at drop-can1 and 4244 at nav-can1, nav-can1 never got a round.
    Ties (nothing targeted yet) go to the node the most seeds die at, then by name."""
    rows: dict[str, dict] = {}
    for seed, s in sorted(before["seeds"].items(), key=lambda kv: int(kv[0])):
        if not s.get("first_death"):
            continue
        d = rows.setdefault(s["first_death"], {"node": s["first_death"], "seeds": [],
                                               "failure_mode": None, "rounds_targeted": 0,
                                               "last_round": 0})
        d["seeds"].append(int(seed))
        d["failure_mode"] = d["failure_mode"] or s.get("failure_mode")
    for r in history or ():
        d = rows.get((r.get("tried") or {}).get("node"))
        if d is not None:
            d["rounds_targeted"] += 1
            d["last_round"] = max(d["last_round"], int(r.get("round") or 0))
    return sorted(rows.values(), key=lambda d: (d["last_round"], -len(d["seeds"]), d["node"]))


def _first_death(before: dict, history: list | None = None):
    """The node this round targets: the least-recently-targeted first-death node
    (without ``history``, the commonest -- the fallback for a call that has none)."""
    nodes = death_nodes(before, history)
    return nodes[0]["node"] if nodes else None


def stuck_on(node, history: list | None, rounds: int | None = None) -> dict | None:
    """``{node, rounds}`` when the last ``rounds`` (``STUCK_ROUNDS``) rounds that targeted
    ``node`` all failed to improve -- rounds on other nodes do not break the streak, or
    rotation would hide it. None while the streak is shorter."""
    streak = 0
    for r in reversed(history or ()):
        if (r.get("tried") or {}).get("node") != node:
            continue
        if r.get("published") or r.get("outcome") == "improved":
            break
        streak += 1
    return {"node": node, "rounds": streak} if node and streak >= (rounds or STUCK_ROUNDS) else None


def cluster_seeds(rounds: list, before: dict, milestone) -> list[int]:
    """The ORIGIN CLUSTER of a milestone: the seeds that shared it as their first missing
    milestone in the campaign's BASELINE round (round 1's per_seed) -- the cohort a fix has
    to keep, seeds it has already won included.

    The fallback below CHANGES that meaning, and only where the origin cluster cannot be
    read at all: with no trail in round 1 there is no baseline cohort, so this scores the
    cluster of THIS round instead -- a seed already repaired has left the cluster and stops
    being guarded against a later round losing it again. Weaker than the origin cluster,
    stronger than the empty list a baseline-only reading returns."""
    if not milestone:   # nothing died: there is no cluster to regress against
        return []
    # round 1's rows can EXIST and carry no trail (the live campaign's baseline row is
    # [{seed: 4243, nodes: []}, {seed: 4244, nodes: []}]), and an empty trail has no first
    # missing milestone -- so this returned [] for all 588 rounds of evolve-recycle_cans.
    # That is NOT why the historical regression never ran there: regression() is only
    # reached under ``if published``, and that campaign published 0 rounds (0 accepted,
    # best 0). An empty cluster silently skipping the check is a second way to lose it.
    r0 = (rounds[0].get("per_seed") if rounds else None) or []
    rows = r0 if any(r.get("nodes") for r in r0) else per_seed(before)
    return sorted(int(r["seed"]) for r in rows
                  if evolve_llm.first_missing(r.get("nodes")) == milestone)


def regression(rounds: list, before: dict, after: dict, milestone) -> dict | None:
    """HISTORICAL REGRESSION over the origin cluster (Zetta's acceptance step before the
    fresh-seed confirm): ``{seeds, before, after, lost}`` = how many of that cluster's seeds
    succeed under the accepted state vs under the trial, and WHICH ones the trial lost. The
    retest re-runs EVERY dev seed and the dev range only ever grows (a burned confirm seed
    joins it), so the cluster is always inside both suites and this reads their results --
    no seed is re-run twice. A non-empty ``lost`` blocks the publish (a net win that swaps
    one cluster seed for two is still a regression). None when the cluster is empty."""
    cs = cluster_seeds(rounds, before, milestone)
    ok = lambda suite, s: bool((suite["seeds"].get(str(s)) or {}).get("success"))
    hit = lambda suite: sum(ok(suite, s) for s in cs)
    return {"seeds": cs, "before": hit(before), "after": hit(after),
            "lost": [s for s in cs if ok(before, s) and not ok(after, s)]} if cs else None


def verdict(tried: dict, trial_row: dict | None, regs: list, confirm: dict | None,
            published: bool, bs_score: tuple, as_score: tuple) -> tuple[bool, str]:
    """ACCEPT, level one: ``(accepted, why)`` for a round that ran -- the score rose and
    nothing that passed stopped passing. (Level two, the publish, is decided upstream.)

    A FOCUSED trial is judged by the SAME score as a full one. Refusing it outright --
    "focused trial: <node> passed on no seed of its cluster", without ever reading the
    score -- ended 347 of the recycle_cans campaign's 588 rounds (nav-can1 196,
    drop-can1 151); nothing written in those rounds could have been accepted.
    ``regressions`` already skips the seeds a focused trial never ran (they carry no
    evidence either way) and the publish still demands scope == "full". What a focused
    accept CANNOT be is the next round's baseline: its suite is ``_merge(before, done)``,
    so every seed the trial never ran keeps a row measured under the previous state, and
    downstream nothing tells those rows from fresh ones (they ride under the round's own
    ``suite_sha``). ``next_baseline`` drops it instead -- one extra suite for the round
    after a focused accept, against a stale row that would otherwise propagate forever."""
    if tried["kind"] == "none" or trial_row is None:
        return False, "nothing tried"
    if regs:
        return False, "regressed: " + ", ".join(f"{g['seed']}/{g['node']}" for g in regs[:4])
    if confirm and not published:   # the fresh seeds refused it: not a baseline either
        return False, f"confirm {confirm['before']} -> {confirm['after']}"
    if as_score > bs_score:
        return True, f"score {list(bs_score)} -> {list(as_score)}, no node regressed"
    return False, f"score {list(bs_score)} -> {list(as_score)}"


def next_baseline(accepted: bool, trial_row: dict | None, kept: dict) -> dict | None:
    """The suite the next round starts from -- None means "re-run it". Only a FOCUSED
    accept returns None: its suite carries baseline rows for every seed the trial never
    ran (see ``verdict``), so the round after it pays one full retest under the accepted
    state rather than scoring against rows the accepted change was never run on."""
    return None if accepted and (trial_row or {}).get("scope") == "focused" else kept


# ── score: the gradient. Whole-task success alone is flat (0/2 for 90 rounds) ─────

def _ok_map(suite: dict) -> dict[str, dict[str, bool]]:
    """``{seed: {node: passed?}}`` from each seed's node trail."""
    return {str(seed): {n["id"]: n.get("ok") is True for n in (s.get("trail") or [])}
            for seed, s in suite["seeds"].items()}


def _recoveries(*suites: dict) -> set[str]:
    """The node ids that are REPAIRS: the planner inserts a ``recover-<node>`` only after
    ``<node>`` failed. Read from EVERY suite given, so a kind known on one side only (the
    trial replanned, the baseline did not) is still known."""
    return {n["id"] for su in suites for s in (su.get("seeds") or {}).values()
            for n in (s.get("trail") or []) if n.get("kind") == "recovery"}


def _repair(nid: str, rec: set) -> bool:
    """Is this node a repair -- ``recover-<node>`` by the planner's naming, or a trail
    row the task validator kinded ``recovery`` (``_recoveries``)?"""
    return nid.startswith("recover-") or nid in rec


def score(suite: dict, node=None) -> tuple[int, int, int]:
    """The suite's LEXICOGRAPHIC score ``(successes, milestones, target_pass)``:
    whole-task wins first, then the furthest-progress signal (nodes passed, summed over
    seeds), then how many seeds pass the round's TARGET node. Success alone is flat --
    0/2 on every candidate for 90 rounds -- so a change that moves a death EARLIER
    scores strictly lower and a change that moves it later scores strictly higher.

    REPAIR nodes do not count as milestones: a recovery exists only because something
    failed, so counting it scored a run that NEEDED a recovery above the same run that
    no longer needs one -- and paid a patch for inserting repairs."""
    oks, rec = _ok_map(suite), _recoveries(suite)
    return (sum(bool(s.get("success")) for s in suite["seeds"].values()),
            sum(sum(ok for nid, ok in m.items() if not _repair(nid, rec)) for m in oks.values()),
            sum(bool(m.get(node)) for m in oks.values()) if node else 0)


def regressions(before: dict, after: dict) -> list[dict]:
    """``[{seed, node, was_ok_now_not}]`` -- every node that PASSED under the accepted
    state and no longer passes under the trial. A non-empty list refuses acceptance,
    whatever the score did.

    Two things are NOT regressions. A seed the trial never ran (a focused scope keeps
    the baseline's rows for the rest) carries no evidence either way. And a REPAIR that
    is gone from the trial's plan BECAUSE THE NODE IT REPAIRS NOW PASSES: the planner
    inserts ``recover-<node>`` only after ``<node>`` failed, so the fix working makes the
    repair vanish -- reading that as a loss refused eight measurably improving rounds of
    the recycle_cans campaign (131..389, milestones 9 -> 13..18, target_pass 0 -> 1).

    Everything else still counts: a repair that RAN AGAIN and failed, a repair whose
    target still fails, and any ordinary node the trial's plan dropped -- a plan that
    silently drops work is not a win."""
    a, rec, out = _ok_map(after), _recoveries(before, after), []
    for seed, m in sorted(_ok_map(before).items(), key=lambda kv: int(kv[0])):
        now = a.get(seed)
        if now is None:
            continue        # the trial never ran this seed: it says nothing about its nodes
        for nid, ok in m.items():
            if not ok or now.get(nid):
                continue
            if nid not in now and _repair(nid, rec) \
                    and now.get(nid.removeprefix("recover-"), True):
                continue    # the repair is gone because its target passes -- that is the win
            out.append({"seed": int(seed), "node": nid, "was_ok_now_not": True})
    return out


def focus_seeds(rounds: list, before: dict, node, seeds: list) -> list[int]:
    """The FOCUSED trial's scope: the target node's failure cluster (``cluster_seeds``,
    else the seeds dying there this round), inside the dev range. Empty when it is the
    whole range -- there is nothing to save by narrowing."""
    full = list(range(int(seeds[0]), int(seeds[1]) + 1))
    cs = [s for s in cluster_seeds(rounds, before, node) if s in full] or \
         sorted(int(s) for s, v in before["seeds"].items()
                if v.get("first_death") == node and int(s) in full)
    return cs if 0 < len(cs) < len(full) else []


def pipeline_modules(ref: str, binding: dict) -> list[str]:
    """The other modules of the card's stage pipeline, patchable once a node is stuck:
    the stage-table module (``ref``), its package's shared ``drivers``, and the mission
    planner. Only the importable ones (``write_patch`` imports what it is given)."""
    out = set()
    mod = ref.partition(":")[0]
    for m in {mod, f"{mod.rpartition('.')[0]}.drivers", (binding.get("planner") or "").partition(":")[0]}:
        try:
            if m and importlib.util.find_spec(m):
                out.add(m)
        except (ImportError, ValueError):
            pass
    return sorted(out)


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


def from_proposal(p: dict, before: dict, node_default: str | None = None) -> dict:
    """A proposal as this round's ``tried`` -- the same {kind, node, detail} shape the
    built-in proposer emits (so apply/publish need no second path), plus
    ``detail.proposal`` (id) and ``detail.note``. ``payload.node`` else ``node_default``
    (the round's rotated target, handed in so the caller and this do not each compute a
    first-death node of their own) else the commonest first-death node; a node the suite
    never ran is an honest ``none``."""
    pay = dict(p["payload"])
    node = pay.pop("node", None) or node_default or _first_death(before)
    runs = [s["nodes"][node] for s in before["seeds"].values() if node in s["nodes"]]
    tag = {"proposal": p["id"], "note": p["note"]}
    if not runs:
        return {"kind": "none", "node": node,
                "detail": {**tag, "reason": f"proposal names no node the suite ran ({node!r})"}}
    need = {"tunables": ("ref", "path", "to"), "executor": ("to",), "card": ("path", "to", "ref")}[p["kind"]]
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


def _tunables(params: dict) -> tuple[dict, list]:
    """Numeric knobs + the key path they live under (``[tunables]`` table or top-level)."""
    t = params.get("tunables")
    nested = isinstance(t, dict)
    src = t if nested else params
    num = {k: v for k, v in src.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return num, (["tunables"] if nested else [])


def propose(before: dict, records: dict, emb: str, arm: str, binding: dict,
            round_no: int, applied: dict, history: list | None = None) -> dict:
    """``history`` = earlier round rows: an executor switch already tried on the
    first-death node (won or lost) is not proposed again, nor is a (knob,
    direction) tunables step. Knobs the card's ``[tunable_hints]`` ties to the
    node's failure_mode go first, each in both directions (-30% then +30%)."""
    node = _first_death(before, history)   # rotates over the distinct first-death nodes
    if node is None:
        return _none("no first death: every seed succeeded", needs=())
    modes = Counter(s.get("failure_mode") for s in before["seeds"].values()
                    if s["first_death"] == node and s.get("failure_mode"))
    mode = modes.most_common(1)[0][0] if modes else None
    runs = [s["nodes"][node] for s in before["seeds"].values() if node in s["nodes"]]
    skill, current = runs[0]["skill"], runs[0]["executor"]
    rate = sum(r["success"] for r in runs) / len(runs)
    rec = records.get(skill)
    if rec is None:
        return _none(f"no skill record for {skill!r}", node)
    spec = segment_specs({skill: rec}, emb).get(skill) or {}
    bound = {"scripted", *(spec.get("policies") or {})}
    ev = rec.evidence.get(emb)
    cands = {k: v for k, v in (ev.by_executor if ev else {}).items()
             if k in bound and k != current and v.get("n")}
    if cands:
        best = max(sorted(cands), key=lambda k: cands[k]["k"] / cands[k]["n"])
        if cands[best]["k"] / cands[best]["n"] > rate:
            return {"kind": "executor", "node": node,
                    "detail": {"skill": skill, "from": current, "to": best,
                               "evidence": dict(cands[best]), "measured": rate}}
    # no evidence says another bound executor is better: one honest attempt at any
    # not yet tried on this node beats none
    tried = {r["tried"]["detail"].get("to") for r in history or ()
             if r["tried"]["kind"] in ("executor", "card") and r["tried"]["node"] == node}
    if untried := sorted(bound - {current} - tried):
        return {"kind": "executor", "node": node,
                "detail": {"skill": skill, "from": current, "to": untried[0],
                           "evidence": dict(cands.get(untried[0]) or {}), "measured": rate}}
    ref = (rearm(spec, arm, current if current in bound else None).get("policy_provider")
           or binding["policy"])
    params = mount_params(ref)
    tun, path = _tunables(params)
    hinted = [k for k in (params.get("tunable_hints") or {}).get(mode) or () if k in tun]
    done = set()   # (knob, went up?) steps already tried on this node (a proposal's row has no numeric from)
    for r in history or ():
        d = r["tried"]["detail"]
        if r["tried"]["kind"] == "tunables" and r["tried"]["node"] == node \
                and isinstance(d.get("from"), (int, float)) and isinstance(d.get("to"), (int, float)):
            done.add((d["path"][-1], d["to"] > d["from"]))
    for key in [*hinted, *sorted(set(tun) - set(hinted))]:
        for f in (0.7, 1.3):
            # the card re-types the overlay (int stays int): a knob the step leaves
            # where it is (0, a small int) is no trial -- skip it rather than burn a suite
            to = type(tun[key])(tun[key] * f)
            if to == tun[key] or (key, to > tun[key]) in done:
                continue
            return {"kind": "tunables", "node": node,
                    "detail": {"skill": skill, "executor": current, "ref": ref, "path": [*path, key],
                               "from": tun[key], "to": to, "hint": mode if key in hinted else None}}
    return _none(f"no untried executor for {skill!r} and no untried tunables step on {ref!r}",
                 node, needs=(f"tunables on {ref}", "evidence for another executor", "proposal"))


def apply(tried: dict, applied: dict) -> dict:
    out = {"executors": dict(applied["executors"]),
           "tunables": json.loads(json.dumps(applied["tunables"])),
           "cards": dict(applied.get("cards") or {})}
    d = tried["detail"]
    if tried["kind"] == "executor":
        out["executors"][tried["node"]] = d["to"]
    elif tried["kind"] == "card":
        out["executors"][tried["node"]] = d["to"]
        out["cards"][d["to"]] = {"skill": d["skill"], "path": d["path"], "ref": d["ref"],
                                 "params": dict(d.get("params") or {}),
                                 "transport": d.get("transport", "inproc")}
    elif tried["kind"] == "tunables":
        cur = out["tunables"].setdefault(d["ref"], {})
        for p in d["path"][:-1]:
            cur = cur.setdefault(p, {})
        cur[d["path"][-1]] = d["to"]
    return out


# ── publish: evidence write-back through the evolution-only skills-root door ───────

def publish(skills_root: Path, rec, emb: str, tried: dict, after: dict) -> tuple[str, dict]:
    d = to_plain(rec)
    node, det = tried["node"], tried["detail"]
    key = det["to"] if tried["kind"] in ("executor", "card") else det["executor"]
    runs = [s["nodes"][node] for s in after["seeds"].values() if node in s["nodes"]]
    ev = d.setdefault("evidence", {}).setdefault(emb, {"n": 0, "k": 0})
    row = ev.setdefault("by_executor", {}).setdefault(key, {"n": 0, "k": 0})
    row["n"] += len(runs)
    row["k"] += sum(r["success"] for r in runs)
    if tried["kind"] == "card":   # the candidate's binding earns its place in the record
        b = d.setdefault("bindings", {}).setdefault(emb, {})
        b.setdefault("policies", {})[key] = _binding(det)
    if tried["kind"] == "tunables":
        b = d.setdefault("bindings", {}).setdefault(emb, {})
        slot = b.get("policies", {}).get(key, b)   # the policy entry; scripted rides the binding
        cur = slot.setdefault("params", {})
        for p in det["path"][:-1]:
            cur = cur.setdefault(p, {})
        cur[det["path"][-1]] = det["to"]
    return InMemorySkillGraph(root=str(skills_root)).publish(d), d


def _media(session: Path, task: str, seeds: list) -> list[str]:
    """Session-relative paths of the clips kept so far (harness.media index), the
    list the board's rsi_frames face returns verbatim."""
    return [f"media/{task}/{seed}/{ent['file']}"
            for seed in range(int(seeds[0]), int(seeds[1]) + 1)
            for ent in media.index_of(session / "media", task, seed).values()]


def _dropped(session: Path, task: str, seeds: list) -> dict[str, dict]:
    """``{"<seed>/<node>": {reason, keyframes: [session-relative paths]}}`` of the
    segments that left no clip -- the honest side of ``media`` (rsi_frames' ``dropped``)."""
    return {f"{seed}/{node}": {"reason": d["reason"],
                               "keyframes": [f"media/{task}/{seed}/{f}" for f in d["keyframes"]]}
            for seed in range(int(seeds[0]), int(seeds[1]) + 1)
            for node, d in media.dropped_of(session / "media", task, seed).items()}


# ── the round loop ────────────────────────────────────────────────────────────────

_ZH = {"idle": "等待", "baseline": "基线评测", "propose": "选试验", "retest": "同种子复测",
       "confirm": "新种子确认", "publish": "发布", "done": "完成", "cancelled": "已取消",
       "paused_disk": "磁盘不足，已暂停"}


def _message(live: dict) -> str:
    """One short operator sentence for the live block (the page shows it verbatim)."""
    head = f"第 {live['round']} 轮 {_ZH.get(live['phase'], live['phase'])}"
    if live["phase"] == "paused_disk":
        return live.get("disk") or head
    if live["phase"] == "propose" and live.get("proposer") == "llm":
        head = f"LLM 分析第 {live['round']} 轮…"
    if live["phase"] == "done":
        return f"已完成 {live['round']} 轮"
    if live["phase"] == "waiting":
        return (f"第 {live['round']} 轮没有可试方案，{live.get('wait_s', 0)} 秒后继续"
                f"（连续第 {live.get('nones', 1)} 次，按停止结束）")
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

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=MODES, default="execution")
    ap.add_argument("--task")
    ap.add_argument("--gc", action="store_true",
                    help="run the maintenance pass alone (prune audits + GC candidate cards) "
                         "and exit; --dry-run prints what it would do and changes nothing")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--session", type=Path)
    ap.add_argument("--skills-root", type=Path)
    ap.add_argument("--seeds", type=int, nargs=2, default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arm", default="auto")
    ap.add_argument("--cancel-marker", type=Path, default=None)
    ap.add_argument("--max-replans", type=int, default=None)
    ap.add_argument("--max-actuations", type=int, default=None)
    ap.add_argument("--confirm-seeds", type=int, default=2,
                    help="fresh scratch seeds (above the block) a debug-seed win must hold on "
                         "before publish; 0 disables")
    ap.add_argument("--proposer", choices=("llm", "rules"), default="llm",
                    help="llm: the model_endpoint card reads the round and answers the try "
                         "(rules fallback when unreachable/invalid); rules: the built-in proposer only")
    args = ap.parse_args(argv)
    if args.gc:   # the operator's door: no task, no session, no round
        log = gc_candidates(runs=(args.session.parent if args.session else REPO_ROOT / "runs"),
                            dry_run=args.dry_run)
        if args.session and args.task:
            log = prune_audits(EvolveStore(args.session, args.task).dir / "llm",
                               dry_run=args.dry_run) + log
        print("\n".join(log) or "gc: nothing to do")
        return 0
    if missing := [f"--{k}" for k in ("task", "session", "skills_root")
                   if getattr(args, k) is None]:
        ap.error("the following arguments are required: " + ", ".join(missing))
    budgets = {"max_replans": args.max_replans, "max_actuations": args.max_actuations}
    if args.mode != "evolution":
        print(json.dumps({"error": f"evolve writes a skills root: refused in mode "
                                   f"{args.mode!r}; assert --mode evolution"}))
        return 3
    binding = discover().task_bindings.get(args.task)
    if binding is None:
        raise SystemExit(f"no task binding for {args.task!r}")
    records = hr._binding_records(binding)
    emb = hr.task_brief(args.task, binding)["embodiment"]
    if keys := Counter(k for r in records.values() for k in r.bindings if emb not in r.bindings):
        emb = keys.most_common(1)[0][0]   # robocasa: the records bind under the card's short name, not the env ref

    store = EvolveStore(args.session, args.task)
    doc = store.load() or {"task": args.task, "session": args.session.name,
                           "seeds": list(args.seeds or [0, 1]), "arm": args.arm,
                           "rounds": [], "best": 0, "cursor": 0, "status": "running",
                           "applied": {"executors": {}, "tunables": {}}}
    seeds, arm, applied = doc["seeds"], doc["arm"], doc["applied"]
    # ``rounds`` counts rounds with a REAL try (a rejected / empty round is not an idea
    # tried); a resume whose --rounds does not reach past the tries so far means "N more"
    # (the console's 开始/继续 sends a task-only brief, so rounds is the default):
    # otherwise the round loop is empty and the brief "finishes" in a blink.
    tries = sum(r["tried"]["kind"] != "none" for r in doc["rounds"])
    unbounded = args.rounds <= 0
    target = float("inf") if unbounded else (args.rounds if args.rounds > tries else tries + args.rounds)
    doc["status"] = "running"
    # live = where the loop is RIGHT NOW (rsi_run's ``live``): rewritten with the
    # doc at every phase/seed/node boundary. One writer, tmp+rename -> no race.
    now = time.time()
    live = doc["live"] = {"phase": "idle", "round": doc["cursor"], "seeds_total": int(seeds[1]) - int(seeds[0]) + 1,
                          "seed_index": None, "seed": None, "node": None, "started_at": now,
                          "round_started_at": None, "phase_started_at": now, "last_round_s": None,
                          "per_seed_partial": [], "tried": None, "message": "", "messages": [],
                          "nodes": [], "seed_started_at": None, "proposer": args.proposer,
                          "sim_s": round(sum((r.get("usage") or {}).get("sim_s") or 0 for r in doc["rounds"]), 3)}

    def suite(seed_range, overlay, media=True, seed_list=None):
        out = run_suite(args.task, binding, seed_range, arm, args.skills_root, overlay,
                        media_dir=args.session / "media" if media else None, budgets=budgets,
                        progress=tick, seed_list=seed_list)
        live["sim_s"] = round(live["sim_s"] + out["elapsed_s"], 3)
        return out

    def tick(**kw) -> None:
        if kw.get("phase", live["phase"]) != live["phase"]:
            kw = {"phase_started_at": time.time(), "seed": None, "seed_index": None, "node": None,
                  "nodes": [], "seed_started_at": None, "per_seed_partial": [], **kw}
        live.update(kw)
        msg = _message(live)
        if msg != live["message"]:   # rolling operator log: the last 20 distinct messages
            live["message"] = msg
            live["messages"] = (live["messages"] + [{"ts": time.time(), "text": msg}])[-20:]
        store.save(doc)

    def preflight(tried: dict) -> None:
        """A card's trial on the FIRST seed alone, before the suite: an exception inside
        the executor comes back to the model as a repair, not a burned suite. The seed's
        result is kept and merged into the retest (never run twice). The static
        ``self_check`` runs first -- an attribute the class never assigns costs no sim at
        all -- and either way the raise is kept whole in ``pre["exc"]`` for the round's
        ``trial_evidence``, not only in the repair text."""
        pre.pop("exc", None)   # a later attempt's outcome, not the last one's
        if why := self_check(tried):
            try:
                raise SelfCheckError(why)
            except SelfCheckError as exc:
                pre["exc"] = _exception(exc)
                raise
        try:
            pre["out"] = suite([int(seeds[0]), int(seeds[0])], apply(tried, applied))
        except Exception as exc:  # noqa: BLE001 -- the executor's own failure IS the finding
            pre["exc"] = _exception(exc)
            raise
        finally:   # mount_params reads the accepted overlay again, not the trial's
            os.environ[OVERRIDE_ENV] = json.dumps(applied["tunables"])

    tick()
    base, pre, r, nones = None, {}, doc["cursor"], 0
    # stop: the target reached, or MAX_NONE rounds in a row with nothing tried (a stuck
    # model cannot loop forever); a none whose ``needs`` is empty (nothing could unblock
    # it: every seed succeeded) ends the loop at once.
    def cancelled() -> bool:
        return args.cancel_marker is not None and args.cancel_marker.exists()

    while tries < target and (unbounded or nones < MAX_NONE):
        if cancelled():
            doc["status"] = "cancelled"
            tick(phase="cancelled")
            return 3
        # bounded growth, checked before anything is spent: the loud stop first, then the
        # cheap housekeeping (this campaign's old audits, the candidate cards nothing needs)
        if msg := disk_guard(args.session, store.dir):
            doc["status"] = "paused_disk"
            doc["rounds"].append(
                {"round": r + 1, "tried": _none(msg, None, needs=("disk",)), "before": doc["best"],
                 "after": doc["best"], "best": doc["best"], "outcome": "none", "accepted": False,
                 "published": False, "accepted_reason": "paused_disk", "paused_disk": msg,
                 "needs": ["disk"], "per_seed": [], "ts": time.time()})
            tick(phase="paused_disk", round=r + 1, disk=msg)
            print(msg, file=sys.stderr)
            return 4
        for line in maintain(store, args.session):
            print(line, file=sys.stderr)
        if unbounded and nones:
            # Throttle, never stop: an empty round costs one model call and no sim.
            wait = min(NONE_BACKOFF_S[0] * 2 ** (nones - 1), NONE_BACKOFF_S[1])
            tick(phase="waiting", wait_s=int(wait), nones=nones)
            t_wait = time.time()
            while time.time() - t_wait < wait:
                if cancelled():
                    doc["status"] = "cancelled"
                    tick(phase="cancelled")
                    return 3
                time.sleep(min(2.0, wait))
        r += 1
        t_round = time.time()
        tick(phase="baseline" if base is None else "propose", round=r, round_started_at=t_round, tried=None)
        sim_s0 = live["sim_s"]
        before = base or suite(seeds, applied)
        tick(phase="propose")
        os.environ[OVERRIDE_ENV] = json.dumps(applied["tunables"])  # mount_params: the accepted overlay, not the last trial's
        prop = take_proposal(args.session, args.task, r)
        proposer, llm, tried = "inbox", None, None
        pre.clear()
        if prop:
            tried = from_proposal(prop, before, _first_death(before, doc["rounds"]))
        elif args.proposer == "llm":
            proposer = "llm"
            try:
                ep = evolve_llm.endpoint()
                tried, llm = evolve_llm.llm_propose(
                    ep, evolve_llm.rsi_projection(doc, before, records, emb, arm, binding, before.get("logs") or []),
                    before, r, store.path.parent / "llm", session=args.session, preflight=preflight)
            except Exception as exc:  # noqa: BLE001 -- no endpoint card / mount error: the rules take over
                llm = {"model": None, "prompt_sha": None, "raw_sha": None, "summary": None,
                       "rationale": None, "reason": f"{type(exc).__name__}: {exc}"[:300]}
        if tried is None:
            proposer = "rules"
            tried = propose(before, records, emb, arm, binding, r, applied, doc["rounds"])
        tick(tried=tried)
        after, published, confirm, regr, burned = before, False, None, None, []
        trial_row, saved_s, trial_exc = None, 0.0, pre.get("exc")
        if tried["kind"] != "none":
            trial = apply(tried, applied)
            tick(phase="retest")
            full = list(range(int(seeds[0]), int(seeds[1]) + 1))
            focus = focus_seeds(doc["rounds"], before, tried["node"], seeds)
            done = pre.get("out")   # the preflight seed already ran under this trial
            ran = {int(x) for x in (done or {"seeds": {}})["seeds"]}

            def run_seeds(want, trial=trial, ran=ran):   # the seeds of ``want`` not already run
                nonlocal done
                todo = [x for x in want if x not in ran]
                if todo:
                    out = suite(None, trial, seed_list=todo)
                    done = _merge(done, out) if done else out
                    ran.update(todo)
                return done

            scope, target_pass = "full", 0
            try:
                if focus:   # NODE-FOCUSED TRIAL: the cluster of the target node first
                    scope, foc = "focused", run_seeds(focus)
                    target_pass = sum(bool((_ok_map(foc).get(str(x)) or {}).get(tried["node"]))
                                      for x in focus)
                # ``set(full) <= ran`` : the preflight seed plus the focus already covered
                # the whole dev range, so run_seeds(full) below is a no-op and the round is
                # a full retest under any name -- round 556 ran 4243 in preflight and 4244
                # in the focus (usage.sim_s 59.8, two whole seeds) and was still filed
                # "focused". It was not robbed of an accept -- its score stood still
                # ([0,9,1] -> [0,9,1]) and it is refused on the score either way; what the
                # old filing cost was the TRUTH of the refusal, which said the node passed
                # on no seed of its cluster instead of saying the score did not move.
                if scope == "full" or target_pass or set(full) <= ran:
                    after = run_seeds(full)
                    scope, target_pass = "full", score(after, tried["node"])[2]
                else:   # the target still does not pass: the rest of the suite is not spent.
                    # Seeds outside the focus keep their baseline result, so the score compares.
                    after = _merge(before, done)
                    saved_s = round((before.get("elapsed_s") or 0.0)
                                    * (len(full) - len(focus)) / len(full), 3)
            except Exception as exc:  # noqa: BLE001 -- the trial's failure is the round's finding
                tried["detail"]["error"] = repr(exc)
                trial_exc = _exception(exc)   # type/message/where/traceback tail, not a repr
                after, scope, target_pass = before, "full", 0
            trial_row = {"scope": scope, "seeds": focus if scope == "focused" else full,
                         "target_pass": target_pass}
            published = scope == "full" and after["count"] > before["count"]
            if published:   # the whole origin cluster first, then the fresh seeds
                regr = regression(doc["rounds"], before, after, tried["node"])
                if regr and regr["lost"]:
                    published = False   # it fixed this round's seed and broke one the cluster had
            if published and args.confirm_seeds > 0:
                # ASPIRE's debug-vs-eval split, light: the win must hold on fresh scratch seeds
                # right above the block (never the ledger), SAME overlay, against the accepted
                # state's count on those seeds (measured once per accepted state)
                cs = [int(seeds[1]) + 1, int(seeds[1]) + args.confirm_seeds]
                tick(phase="confirm")
                cb = doc.get("confirm_base")
                if not cb or cb["seeds"] != cs:
                    cb = doc["confirm_base"] = {"seeds": cs, "count": suite(cs, applied, media=False)["count"]}
                try:
                    ca = suite(cs, trial, media=False)["count"]
                except Exception as exc:  # noqa: BLE001
                    tried["detail"]["error"] = repr(exc)
                    trial_exc = trial_exc or _exception(exc)
                    ca = -1
                confirm = {"seeds": cs, "before": cb["count"], "after": ca}
                published = ca >= cb["count"]
                if published:
                    cb["count"] = ca   # the trial is the next accepted state on these seeds too
                elif ca >= 0:
                    # the held-out seed forced another edit: it is burned as held-out. It joins
                    # the dev list (the range grows over it) and the next round draws fresh
                    # confirm seeds above -- confirm_base re-measures itself on the new pair.
                    burned = list(range(cs[0], cs[1] + 1))
                    seeds[1] = cs[1]   # ``seeds`` IS doc["seeds"]
                    tick(seeds_total=int(seeds[1]) - int(seeds[0]) + 1)
            if published:
                tick(phase="publish")
                skill = tried["detail"]["skill"]
                tried["detail"]["digest"], d = publish(
                    args.skills_root, records[skill], emb, tried, after)
                records[skill] = SkillRecordV0.from_dict(d)   # later rounds build on what was published
        # what the model's own code DID in the simulator: the target node's per-step
        # evidence under the trial, its diff against the baseline seed, and any raise
        evidence = trial_evidence(before, after, tried["node"],
                                  (trial_row or {}).get("seeds") or [], trial_exc) \
            if (trial_row or trial_exc) else None
        # ── ACCEPT: the two levels. A partial win (the score rose and nothing that passed
        # stopped passing) joins the campaign's accepted state and becomes the next round's
        # baseline; only a whole-task win also publishes evidence into the skill record.
        bs_score, as_score = score(before, tried["node"]), score(after, tried["node"])
        regs = regressions(before, after)
        accepted, why = verdict(tried, trial_row, regs, confirm, published, bs_score, as_score)
        if published and not accepted:   # a whole-task win is the accepted state by definition
            accepted, why = True, f"published: {before['count']} -> {after['count']} ({why})"
        if accepted:
            applied = trial
            doc.setdefault("accepted_stack", []).append(
                {"round": r, "kind": tried["kind"],
                 "detail": {"node": tried["node"],
                            **{k: tried["detail"][k] for k in
                               ("skill", "ref", "path", "from", "to", "module", "edits")
                               if k in tried["detail"]}},
                 "score": list(as_score)})
        kept = after if accepted else before
        doc["best"] = max(int(doc["best"] or 0), kept["count"])
        update_reference(doc, kept, r)   # the successful-reference index, campaign-wide
        doc["rounds"].append({
            "round": r, "tried": tried, "before": before["count"], "after": after["count"],
            # the streak the brief widened on (None while the node is not stuck)
            "stuck": stuck_on(tried["node"], doc["rounds"]),
            # the diagnosis: which causal layer the try claims, what the round taught, the
            # origin cluster re-scored under it, and the confirm seeds it burned into dev
            "layer": tried["detail"].get("layer"), "notes": tried["detail"].get("notes"),
            "regression": regr, "burned": burned,
            "best": doc["best"], "suite_sha": after["sha"], "published": published,
            # the hypothesis tree: which accepted state this try grew from, and how it went
            "parent": max((x["round"] for x in doc["rounds"]
                           if x.get("accepted") or x.get("published")), default=0),
            # outcome on the SCORE TUPLE, not the success count: a death that moves earlier
            # is ``worse``, never ``same`` -- that is the gradient
            "outcome": ("none" if tried["kind"] == "none" else "improved" if as_score > bs_score
                        else "worse" if as_score < bs_score else "same"),
            "before_score": list(bs_score), "after_score": list(as_score),
            "accepted": accepted, "accepted_reason": why, "trial": trial_row,
            "trial_evidence": evidence, "confirm": confirm,
            "usage": {"llm_tokens": llm.pop("usage", None) if llm else None,
                      "sim_s": round(live["sim_s"] - sim_s0, 3), "sim_s_saved": saved_s},
            "per_seed": per_seed(kept), "after_seeds": per_seed(after),
            "needs": tried["detail"].get("needs", []) if tried["kind"] == "none" else [],
            "media": _media(args.session, args.task, seeds),
            "media_dropped": _dropped(args.session, args.task, seeds), "ts": time.time(),
            "proposal": {k: prop[k] for k in ("id", "kind", "note")} if prop else None,
            "proposer": proposer, "llm": llm})
        # what the model is told plainly next round: what this try did to the score, and
        # which seed/node it pushed backwards
        doc["last_outcome"] = {
            "round": r, "layer": tried["detail"].get("layer"), "kind": tried["kind"],
            "summary": (llm or {}).get("summary") or tried["detail"].get("reason")
                       or f"{tried['kind']} @ {tried['node']}",
            "before_score": list(bs_score), "after_score": list(as_score),
            "outcome": doc["rounds"][-1]["outcome"], "accepted": accepted,
            "accepted_reason": why, "regressions": regs,
            "trial_evidence": _evidence_summary(evidence)}
        doc["cursor"], doc["applied"] = r, applied
        tick(phase="idle", last_round_s=round(time.time() - t_round, 1))
        base = next_baseline(accepted, trial_row, kept)
        if tried["kind"] != "none":
            tries, nones = tries + 1, 0
        elif not tried["detail"].get("needs"):
            break   # nothing could unblock it: every seed succeeded -- genuinely done
        else:
            nones += 1
    doc["status"] = "done"
    tick(phase="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
