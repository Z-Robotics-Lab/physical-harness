"""The lightweight RSI loop: look -> try -> re-run -> publish, one round at a time.

Spawned by ``scripts/harness_runtime._run_evolve`` (an ``evolve`` brief, evolution
mode only) as its own process group. One round: run the task's seed suite in-process
(the SAME ``_mount_plan``/``task_brief``/``workload.run`` a task brief uses), read
each seed's first-death node + fault signature + per-node executor, let the built-in
proposer pick ONE change -- the first-death node's executor (a bound policy whose
``evidence.by_executor`` beats the measured rate) else a one-dimensional +/-20%
tunables perturbation of that node's driver (its card's mount params, applied via
``PH_MOUNT_PARAMS_OVERRIDE``) else nothing, with the honest reason -- re-run the
same seeds, and publish when the success count improves: the skill record with
the measured ``by_executor`` row folded in goes through the evolution-only skills
root door (``InMemorySkillGraph.publish``, the same one scripts/publish_plans.py
uses). Every round lands atomically in ``campaigns/evolve-<task>/campaign.json``
(rounds[], best, cursor, status) with the kept suite's per-seed summary
(``per_seed``) and, when nothing was tried, ``needs`` -- what would unblock the
proposer; the runtime seals the ``rsi_step`` rows off it. The same file carries a
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
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


class EvolveStore:
    """``campaigns/evolve-<task>/campaign.json``, written atomically (tmp+rename)."""

    def __init__(self, session: Path, task: str) -> None:
        self.path = session / "campaigns" / f"evolve-{task}" / "campaign.json"

    def load(self) -> dict | None:
        return json.loads(self.path.read_text()) if self.path.exists() else None

    def save(self, doc: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
        os.replace(tmp, self.path)


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


class _Tap(SessionLog):
    """The per-seed ledger as a node trail: ``task.plan`` sets ``nodes`` (plan order,
    ``ok`` None; a replan resets all but the verified-ok nodes), each ``task.verify`` fills that node's ``ok`` (+
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
            self.nodes = [done.get(n["id"]) or {"id": n["id"], "skill": n.get("skill"), "ok": None,
                                                "steps": None, "failure_mode": None}
                          for n in data["graph"].get("nodes") or []]
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


def run_suite(task: str, binding: dict, seeds: list, arm: str, skills_root: Path,
              applied: dict, media_dir: Path | None = None, budgets: dict | None = None,
              progress=None) -> dict:
    """{count, seeds: {seed: {success, first_death, fault, nodes}}, sha}. ``media_dir``
    (<session>/media) turns on the workload's segment recorder: kept-on-success clips.
    ``progress(**live)`` is called at every seed boundary and node change."""
    os.environ[OVERRIDE_ENV] = json.dumps(applied["tunables"])
    cards = applied.get("cards") or {}
    os.environ[EXTRA_ENV] = ":".join(r for r in (_BASE_EXTRA, *(c["path"] for c in cards.values())) if r)
    per = {}
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
    for i, seed in enumerate(range(int(seeds[0]), int(seeds[1]) + 1)):
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
            "trail": [{k: n[k] for k in ("id", "ok", "steps", "failure_mode")} for n in log.nodes],
            "first_death": dead,
            "failure_mode": (nodes[dead].get("diagnostics") or {}).get("failure_mode") if dead else None,
            "fault": {k: faults[0].get(k) for k in ("kind", "node", "msg")} if faults else None,
            "nodes": {nid: {"skill": skills.get(nid), "success": bool(n["success"]),
                            "executor": n.get("executor") or "scripted",
                            "tunables_sha": (n.get("diagnostics") or {}).get("tunables_sha")}
                      for nid, n in nodes.items()}}
        for n in per[str(seed)]["trail"]:   # final state from the result: a replan reset the live
            r = nodes.get(n["id"]) or {}      # trail, and the verify row carries no steps/diagnostics
            if n["ok"] is None and "success" in r:
                n["ok"] = bool(r["success"])
            n["steps"] = n["steps"] if n["steps"] is not None else r.get("steps")
            n["failure_mode"] = n["failure_mode"] or (r.get("diagnostics") or {}).get("failure_mode")
        tick(per_seed_partial=per_seed({"seeds": per}))
    return {"count": sum(s["success"] for s in per.values()), "seeds": per, "sha": sha_json(per)}


def per_seed(suite: dict) -> list[dict]:
    """The operator-facing per-seed summary sealed with every round (rsi_step /
    campaign.json): ``[{seed, success, first_death, failure_mode, tunables_sha, elapsed_s,
    nodes: [{id, ok, steps, failure_mode}]}]`` (the knobs the dying node ran under; the
    node trail's final state) -- the seed detail that otherwise lives only in this process."""
    return [{"seed": int(seed), **{k: s.get(k) for k in ("success", "first_death", "failure_mode")},
             "elapsed_s": s.get("elapsed_s"), "nodes": s.get("trail") or [],
             "tunables_sha": (s["nodes"].get(s["first_death"]) or {}).get("tunables_sha")
             if s.get("first_death") else None}
            for seed, s in suite["seeds"].items()]


# ── try: the proposals inbox first, then the built-in proposer ────────────────────

def _none(reason: str, node=None, needs=("proposal",)) -> dict:
    """``needs`` = what WOULD give the proposer something to try next round."""
    return {"kind": "none", "node": node, "detail": {"reason": reason, "needs": list(needs)}}


def _binding(c: dict) -> dict:
    return {"ref": c["ref"], "params": dict(c.get("params") or {}),
            "transport": c.get("transport", "inproc")}


def _first_death(before: dict):
    deaths = Counter(s["first_death"] for s in before["seeds"].values() if s["first_death"])
    return deaths.most_common(1)[0][0] if deaths else None


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
    """A proposal as this round's ``tried`` -- the same {kind, node, detail} shape the
    built-in proposer emits (so apply/publish need no second path), plus
    ``detail.proposal`` (id) and ``detail.note``. ``payload.node`` else the commonest
    first-death node; a node the suite never ran is an honest ``none``."""
    pay = dict(p["payload"])
    node = pay.pop("node", None) or _first_death(before)
    runs = [s["nodes"][node] for s in before["seeds"].values() if node in s["nodes"]]
    tag = {"proposal": p["id"], "note": p["note"]}
    if not runs:
        return {"kind": "none", "node": node,
                "detail": {**tag, "reason": f"proposal names no node the suite ran ({node!r})"}}
    need = {"tunables": ("ref", "path", "to"), "executor": ("to",), "card": ("path", "to", "ref")}[p["kind"]]
    if missing := [k for k in need if k not in pay]:
        return {"kind": "none", "node": node,
                "detail": {**tag, "reason": f"{p['kind']} proposal lacks {missing}"}}
    return {"kind": p["kind"], "node": node,
            "detail": {"skill": runs[0]["skill"], "executor": runs[0]["executor"],
                       "from": runs[0]["executor"], **tag, **pay}}


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
    deaths = Counter(s["first_death"] for s in before["seeds"].values() if s["first_death"])
    if not deaths:
        return _none("no first death: every seed succeeded", needs=())
    node = deaths.most_common(1)[0][0]
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


def _dropped(session: Path, task: str, seeds: list) -> dict[str, str]:
    """``{"<seed>/<node>": reason}`` of the segments that left no clip -- the
    honest side of ``media`` (read via rsi_run's latest round)."""
    return {f"{seed}/{node}": why
            for seed in range(int(seeds[0]), int(seeds[1]) + 1)
            for node, why in media.dropped_of(session / "media", task, seed).items()}


# ── the round loop ────────────────────────────────────────────────────────────────

_ZH = {"idle": "等待", "baseline": "基线评测", "propose": "选试验", "retest": "同种子复测",
       "publish": "发布", "done": "完成", "cancelled": "已取消"}


def _message(live: dict) -> str:
    """One short operator sentence for the live block (the page shows it verbatim)."""
    head = f"第 {live['round']} 轮 {_ZH.get(live['phase'], live['phase'])}"
    if live["phase"] == "done":
        return f"已完成 {live['round']} 轮"
    if live["phase"] == "cancelled":
        return f"第 {live['round']} 轮边界取消"
    t = live.get("tried")
    if t and live["phase"] in ("retest", "publish"):
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
    ap.add_argument("--task", required=True)
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--skills-root", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arm", default="auto")
    ap.add_argument("--cancel-marker", type=Path, default=None)
    ap.add_argument("--max-replans", type=int, default=None)
    ap.add_argument("--max-actuations", type=int, default=None)
    args = ap.parse_args(argv)
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

    store = EvolveStore(args.session, args.task)
    doc = store.load() or {"task": args.task, "session": args.session.name,
                           "seeds": list(args.seeds or [0, 1]), "arm": args.arm,
                           "rounds": [], "best": 0, "cursor": 0, "status": "running",
                           "applied": {"executors": {}, "tunables": {}}}
    seeds, arm, applied = doc["seeds"], doc["arm"], doc["applied"]
    doc["status"] = "running" if doc["cursor"] < args.rounds else "done"
    # live = where the loop is RIGHT NOW (rsi_run's ``live``): rewritten with the
    # doc at every phase/seed/node boundary. One writer, tmp+rename -> no race.
    now = time.time()
    live = doc["live"] = {"phase": "idle", "round": doc["cursor"], "seeds_total": int(seeds[1]) - int(seeds[0]) + 1,
                          "seed_index": None, "seed": None, "node": None, "started_at": now,
                          "round_started_at": None, "phase_started_at": now, "last_round_s": None,
                          "per_seed_partial": [], "tried": None, "message": "", "messages": [],
                          "nodes": [], "seed_started_at": None}

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

    tick()
    base = None
    for r in range(doc["cursor"] + 1, args.rounds + 1):
        if args.cancel_marker is not None and args.cancel_marker.exists():
            doc["status"] = "cancelled"
            tick(phase="cancelled")
            return 3
        t_round = time.time()
        tick(phase="baseline" if base is None else "propose", round=r, round_started_at=t_round, tried=None)
        before = base or run_suite(args.task, binding, seeds, arm, args.skills_root, applied,
                                     media_dir=args.session / "media", budgets=budgets, progress=tick)
        tick(phase="propose")
        os.environ[OVERRIDE_ENV] = json.dumps(applied["tunables"])  # mount_params: the accepted overlay, not the last trial's
        prop = take_proposal(args.session, args.task, r)
        tried = (from_proposal(prop, before) if prop
                 else propose(before, records, emb, arm, binding, r, applied, doc["rounds"]))
        tick(tried=tried)
        after, published = before, False
        if tried["kind"] != "none":
            trial = apply(tried, applied)
            tick(phase="retest")
            try:
                after = run_suite(args.task, binding, seeds, arm, args.skills_root, trial,
                                  media_dir=args.session / "media", budgets=budgets, progress=tick)
            except Exception as exc:  # noqa: BLE001 -- the trial's failure is the round's finding
                tried["detail"]["error"] = repr(exc)
                after = before
            published = after["count"] > before["count"]
            if published:
                tick(phase="publish")
                applied = trial
                skill = tried["detail"]["skill"]
                tried["detail"]["digest"], d = publish(
                    args.skills_root, records[skill], emb, tried, after)
                records[skill] = SkillRecordV0.from_dict(d)   # later rounds build on what was published
        kept = after if published else before
        doc["best"] = max(int(doc["best"] or 0), kept["count"])
        doc["rounds"].append({
            "round": r, "tried": tried, "before": before["count"], "after": after["count"],
            "best": doc["best"], "suite_sha": after["sha"], "published": published,
            "per_seed": per_seed(kept), "after_seeds": per_seed(after),
            "needs": tried["detail"].get("needs", []) if tried["kind"] == "none" else [],
            "media": _media(args.session, args.task, seeds),
            "media_dropped": _dropped(args.session, args.task, seeds), "ts": time.time(),
            "proposal": {k: prop[k] for k in ("id", "kind", "note")} if prop else None})
        doc["cursor"], doc["applied"] = r, applied
        tick(phase="idle", last_round_s=round(time.time() - t_round, 1))
        base = kept
    doc["status"] = "done"
    tick(phase="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
