#!/usr/bin/env python3
"""The GENERIC RSI chain: one task name in, the whole discipline out.

    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/rsi_campaign.py \
        --task inventory_build --out runs/rsi-inventory --stop-after gate

Five hand-written campaign scripts (stack/place/grasp_cube/clear_build) plus a
probe script per task all encode the SAME chain with a different task welded in.
This module is that chain with the task as an ARGUMENT. There is no ``if task ==``
branch anywhere below, and adding a task is still installing a plugin dir.

The chain, in order (docs/project-documentation.md §4):

a. **allocate** -- one contiguous 650-seed frontier past every burned block
   (board.store.burned_blocks: DERIVED from sealed preregistrations under runs/),
   split cal 150 / dev 300 / held-out 200. The caller may pin any of the three.
b. **calibrate** -- N baseline episodes through the GENERIC task path
   (``plugins.task.workload.run`` under an EMPTY skills root, so the arm is
   ungoverned by construction). Yields chain base rate, per-node x per-mechanism
   first-death, and wall clock. Calibration NEVER gates and is always re-runnable.
c. **gate** -- M7 §3 / M6 §4 as five scored criteria. Not proceeding is a RESULT,
   not a failure: the verdict names which capability is missing and burns no dev
   seed.
d. **prereg** -- content-hashed, sealed BEFORE any dev seed runs.
e. **dev campaign** -- ``plugins.rsi.workload.run`` FROM-SCRATCH on the node the
   ATTRIBUTION chose (never the caller). Threshold comes from the search
   (plugins/rsi/stats/search.py), recovery shape from the repertoire, both
   selected by measurement.
f. **held-out** -- run_campaign scores it once, and only on a promotion.
g. **fold** -- the caller (harness_runtime._run_rsi) copies published records into
   the session skills root; the 两态铁律 audit is unchanged.
h. **ledger** -- a STATUS.md-shaped entry is PRINTED for the operator. Never
   appended: the ledger stays human-authored.

Honest boundaries are first-class, not error paths -- see ``recovery_support``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from board.store import burned_blocks
from harness.spec_tabletop import STACK_PHASE_HEIGHT
from scripts.alloc_seeds import next_block

#: Block sizes, the shape every evidence plan has used since M5.
CAL_N, DEV_N, HELDOUT_N = 150, 300, 200

#: ``plugins/policies/drivers.py`` seeds numpy with ``spec.seed * 7919 + 11``,
#: which overflows int32 above this. Allocating past it would hand back a block
#: that crashes on its first episode, so refuse at allocation time.
SEED_CEILING = 542479

#: The two methods ``plugins/rsi/governed.py`` calls on a driver when a rule
#: fires. A driver without them cannot be governed at all -- no recovery
#: primitive is invented for it (see ``recovery_support``).
RECOVERY_PROTOCOL = ("retarget", "on_handback")

#: kind -> the death mechanism a first-death at that kind MEANS. Derived from the
#: node kind, not from a per-task table: an actuating node dies by burning its
#: budget without reaching its own done; its verify dies when the node CLAIMED
#: done and the live world disagreed (the drop-shaped death a recovery targets).
MECHANISM = {
    "manipulate": "actuation_stall", "segment": "actuation_stall",
    "verify": "verify_drop", "perceive": "perceive_read", "decide": "decide_gate",
}

#: The node kinds that dispatch a governed rollout, i.e. the ones a Bundle can
#: mount on at all (``plugins/task/workload.py`` _manipulate / _segment).
ACTUATING = ("manipulate", "segment")


# ── a. allocate ──────────────────────────────────────────────────────────────

def allocate(burned: list[tuple], *, floor: int = 0, cal=None, dev=None,
             heldout=None) -> dict:
    """Claim cal/dev/held-out blocks past the burned set as inclusive [lo,hi].

    ``burned`` is ``board.store.burned_blocks`` output: ``(lo, hi, role, sha)``
    rows derived from every sealed preregistration. A pinned dev/held-out block
    is re-checked against it here, so the CLI path is guarded like the runtime.

    ONE contiguous ``CAL_N+DEV_N+HELDOUT_N`` block is taken and split, so the
    three are disjoint by construction rather than by three separate scans that
    could interleave. Any of the three may be pinned by the caller -- a pinned
    calibration block is the normal case, since calibration never gates and a
    measured block stays re-measurable forever.
    """
    taken = [(int(lo), int(hi)) for lo, hi, *_ in burned]
    lo, hi = next_block(CAL_N + DEV_N + HELDOUT_N, taken, floor=floor)
    if hi > SEED_CEILING:
        raise ValueError(
            f"allocator reached seed {hi} but the driver's RandomState seeding "
            f"({{seed}}*7919+11) overflows above {SEED_CEILING}; the ledger frontier "
            "has outrun the seeding scheme -- fix the seeding before allocating here")
    auto = {"cal": (lo, lo + CAL_N - 1),
            "dev": (lo + CAL_N, lo + CAL_N + DEV_N - 1),
            "heldout": (lo + CAL_N + DEV_N, hi - 1)}
    pinned = {"cal": cal, "dev": dev, "heldout": heldout}
    blocks = {k: (tuple(v) if v is not None else auto[k]) for k, v in pinned.items()}
    # Pinned blocks bypass the split above, so re-assert disjointness HERE.
    # ``Preregistration`` catches a dev/held-out overlap too, but only at step d
    # -- after the calibration set has already been paid for -- and it never sees
    # the calibration block at all.
    for a, b in (("cal", "dev"), ("cal", "heldout"), ("dev", "heldout")):
        (alo, ahi), (blo, bhi) = blocks[a], blocks[b]
        if alo <= bhi and blo <= ahi:
            raise ValueError(
                f"{a} block [{alo},{ahi}] overlaps {b} block [{blo},{bhi}]; "
                "the three roles must be disjoint (a block that both calibrates "
                "and gates is not evidence)")
    for k in ("dev", "heldout"):
        alo, ahi = blocks[k]
        for blo, bhi in taken:
            if alo <= bhi and blo <= ahi:
                raise ValueError(f"{k} block [{alo},{ahi}] hits burned [{blo},{bhi}]")
    return blocks


def seeds(block: tuple[int, int]) -> list[int]:
    """Inclusive [lo,hi] -> the seed list."""
    return list(range(int(block[0]), int(block[1]) + 1))


# ── b. calibrate (the GENERIC probe: no per-task probe script) ───────────────

def _binding(task: str) -> dict:
    from harness.manifest import discover

    binding = discover().task_bindings.get(task)
    if binding is None:
        raise SystemExit(
            f"no task binding for {task!r}; install a plugin dir that declares it")
    return binding


#: SOFT per-episode wall cap, enforced INSIDE the worker by SIGALRM. A healthy
#: kitchen_thaw episode is ~60-110s. SIGALRM still earns its keep against a
#: Python-level spin -- but only that: a Python signal handler runs between
#: bytecodes, so a worker parked inside MuJoCo's C code never reaches it. That
#: is exactly what happened on 2026-08-28 (8 workers at 100% CPU for 3h, alarm
#: never fired, 140 finished episodes lost), which is why the cap below exists.
EPISODE_WALL_S = 600

#: HARD per-episode cap, enforced by the PARENT (harness.executor watched path,
#: SIGKILL). 2x the soft cap: a worker past twice its own alarm has proved the
#: alarm cannot reach it. Either cap yields the same honest row --
#: first_death="wall_timeout", which attribute() counts as ungoverned (charged
#: to nobody, never a target).
EPISODE_HARD_WALL_S = 2 * EPISODE_WALL_S


class _EpisodeWallTimeout(Exception):
    pass


def _lost_row(job: tuple, reason: str, seconds: float = EPISODE_HARD_WALL_S) -> dict:
    """The row an episode nobody could finish returns. ``reason`` is its
    ``first_death``: ``wall_timeout`` (capped) or ``worker_died`` (the child
    vanished without a result -- a segfault or the OOM killer). Neither is a
    node id, so ``attribute()`` charges it to nobody and can never target it.

    Doubles as the executor's ``on_lost(item, reason)`` -- the default
    ``seconds`` is the hard cap the parent enforces."""
    return {
        "seed": job[1], "success": False, "first_death": reason,
        "graph": [], "node_ok": {}, "node_stages": {},
        "replans": 0, "actuations": 0, "budget_exhaust": False,
        "seconds": float(seconds), "wall_timeout": reason == "wall_timeout",
    }


def _probe_one(job: tuple[str, int, int, int]) -> dict:
    """ONE ungoverned episode of ``task`` at ``seed``, plus its node graph.

    A clean pool task: fresh kernel, fresh mounts, the same ``_mount_plan`` the
    resident runtime builds for a ``{"kind":"task"}`` brief -- pointed at an EMPTY
    skills root, which is what makes the arm baseline (``assemble_bundle`` matches
    nothing, so every node runs ``governed_rollout(spec, None)``).

    The plan is asked for BEFORE the run so each node's declared ``kind`` and
    ``after`` edges travel with the row; that is the whole per-task attribution
    table the hand-written probes used to hard-code.
    """
    task, seed, max_replans, max_actuations = job
    import signal

    def _alarm(signum, frame):  # noqa: ARG001 -- signal handler shape
        raise _EpisodeWallTimeout

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(EPISODE_WALL_S)
    try:
        return _probe_one_uncapped(task, seed, max_replans, max_actuations)
    except _EpisodeWallTimeout:
        return _lost_row(job, "wall_timeout", EPISODE_WALL_S)
    finally:
        signal.alarm(0)


#: True once THIS pool worker won the frame.lock and armed the overlay.
_FRAMES = False


def _maybe_arm_frames() -> bool:
    """Arm the 取景窗 overlay in exactly ONE pool worker, if the spawning
    runtime asked for it (``PH_RSI_FRAMES`` = the session's frame.jpg path).

    One writer, chosen by an O_EXCL lockfile keyed by pid: ten workers
    overwriting one frame.jpg would flicker between ten kitchens. A stale lock
    (dead pid, e.g. a previous chain's worker) is stolen. Live state, not
    evidence -- the overlay delegates verbatim and a lost frame never moves an
    episode (scripts/frame_dump.py's own contract).
    """
    global _FRAMES
    if _FRAMES:
        return True
    dest = os.environ.get("PH_RSI_FRAMES")
    if not dest:
        return False
    lock = Path(dest).with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text().strip() or "0"), 0)
            return False          # a live worker already streams
        except (ValueError, ProcessLookupError):
            try:
                lock.write_text(str(os.getpid()))   # steal the stale lock
            except OSError:
                return False
    from scripts import frame_dump
    frame_dump.arm(dest)
    _FRAMES = True
    return True


def _probe_one_uncapped(task: str, seed: int, max_replans: int,
                        max_actuations: int) -> dict:
    from harness.definitions import CAPABILITIES
    from harness.events import SessionLog
    from harness.kernel import Kernel
    from plugins.task import workload

    from scripts import harness_runtime as hr

    binding = _binding(task)
    with tempfile.TemporaryDirectory() as empty_skills:
        kernel = Kernel(CAPABILITIES, log=SessionLog())
        kernel.mount(hr._mount_plan(binding, Path(empty_skills),
                                    frames=_maybe_arm_frames()))
        brief = hr.task_brief(task, binding)
        # The planner call the loop's first attempt makes, verbatim -- so the
        # graph recorded here is the graph that ran.
        scene = kernel.resolve("graph.scene", consumer="rsi")
        plan = kernel.resolve("task.planner", consumer="rsi").plan(
            {**brief, "scene": scene.snapshot({}), "budget": max_actuations})
        graph = [{"id": n["id"], "skill": n["skill"],
                  "kind": n.get("kind", "manipulate"),
                  "after": list(n.get("after") or []),
                  "args": dict(n.get("args") or {})}
                 for n in (plan.get("nodes") or [])]
        t0 = time.perf_counter()
        out = workload.run(brief, kernel, seed=seed, max_replans=max_replans,
                           max_actuations=max_actuations)
        dt = time.perf_counter() - t0

    nodes = out["nodes"]
    # First death = the first node the loop RAN that did not end true. nodes_out
    # is insertion-ordered, and the loop breaks on the first fault, so insertion
    # order IS execution order.
    first_death = next((nid for nid, n in nodes.items() if not n["success"]), "none")
    faults = out.get("faults") or []
    return {
        "seed": seed, "success": bool(out["success"]),
        "first_death": first_death,
        "graph": graph,
        "node_ok": {nid: bool(n["success"]) for nid, n in nodes.items()},
        # the target node's own stage ledger, for the repair-shape read below
        "node_stages": {nid: n.get("stages") or [] for nid, n in nodes.items()},
        "replans": out["replans"], "actuations": out["actuations"],
        "budget_exhaust": bool(faults and faults[-1].get("kind") == "budget"),
        "seconds": round(dt, 3),
    }


#: Chain steps after calibration that the progress bar still has to cross
#: (gate, dev, held-out). The bar's denominator is ``len(cal) + this`` so ONE
#: heartbeat file tracks the whole chain: calibration dominates the wall clock,
#: and the card stays live (``done < total``) through the tail stages instead of
#: retiring the moment the last calibration episode lands.
POST_CAL_STEPS = 3


def calibrate(task: str, block, *, workers: int = 10, out_dir: Path | None = None,
              max_replans: int = 3, max_actuations: int = 40,
              extra: dict | None = None) -> dict:
    """Run the ungoverned calibration set and fold the numbers the gate reads."""
    from harness.executor import LocalPoolExecutor

    ss = seeds(block)
    tick = None
    if out_dir is not None:
        from scripts.campaign_progress import tracker

        tick = tracker(out_dir, len(ss) + POST_CAL_STEPS,
                       label=f"rsi {task} · calibrate", extra=extra)
    jobs = [(task, s, max_replans, max_actuations) for s in ss]
    if workers <= 1 or len(ss) == 1:
        rows = []
        for j in jobs:
            rows.append(_probe_one(j))
            if tick is not None:
                tick(rows[-1])
    else:
        # timeout= arms the parent-side watchdog: the serial branch above has no
        # parent to watch it and keeps the SIGALRM soft cap alone.
        rows = LocalPoolExecutor().map(_probe_one, jobs, workers=workers,
                                       on_result=tick,
                                       timeout=EPISODE_HARD_WALL_S,
                                       on_lost=_lost_row)
    rows.sort(key=lambda r: r["seed"])
    return summarize(task, rows, block)


def summarize(task: str, rows: list[dict], block) -> dict:
    """Chain base rate + node x mechanism first-death + wall clock. Never gates."""
    n = len(rows)
    # First row WITH a graph, not row 0: a written-off episode carries an empty
    # graph, and the lowest seed dying that way would blank the kind table and
    # silently make every death ungoverned.
    graph = next((r["graph"] for r in rows if r["graph"]), [])
    kinds = {g["id"]: g["kind"] for g in graph}
    by_node = Counter(r["first_death"] for r in rows)
    by_mech = Counter(MECHANISM.get(kinds.get(r["first_death"], ""), "none")
                      for r in rows if r["first_death"] != "none")
    secs = [r["seconds"] for r in rows]
    successes = sum(r["success"] for r in rows)
    return {
        "task": task, "arm": "baseline", "n": n, "block": list(block),
        "successes": successes,
        "base_rate": round(successes / n, 4) if n else None,
        "graph": graph,
        "first_death_by_node": dict(by_node),
        "first_death_by_mechanism": dict(by_mech),
        "first_death_by_node_x_mechanism": {
            nid: {"kind": kinds.get(nid, "?"),
                  "mechanism": MECHANISM.get(kinds.get(nid, ""), "none"),
                  "deaths": c}
            for nid, c in sorted(by_node.items(), key=lambda kv: -kv[1])
            if nid != "none"},
        "budget_exhaust": sum(r["budget_exhaust"] for r in rows),
        "replans_total": sum(r["replans"] for r in rows),
        "seconds_total": round(sum(secs), 3),
        "seconds_per_episode": round(sum(secs) / n, 3) if n else None,
        "episodes": rows,
    }


# ── attribution: WHICH node, decided by the data ─────────────────────────────

def attribute(cal: dict) -> dict:
    """Fold first-deaths onto the ACTUATING node each one is attributable to.

    A verify node has nothing to govern of its own -- it is the oracle that
    caught the preceding sub-goal dropping what it claimed to hold -- so its
    deaths are charged BACKWARDS along ``after`` to the nearest actuating
    ancestor, which is the node a recovery would fire inside. Deaths at a
    perceive/decide node are charged to nobody: they are ungoverned by
    construction, and a majority there is the M6 c3 attribution pivot.

    The target is the argmax over governable charges. The caller never picks it;
    that is the point (a caller left to choose would choose a flattering node).
    """
    graph = {g["id"]: g for g in cal.get("graph") or []}

    def owner(nid: str) -> str | None:
        seen = set()
        while nid in graph and nid not in seen:
            seen.add(nid)
            g = graph[nid]
            if g["kind"] in ACTUATING:
                return nid
            after = [a for a in g["after"] if a in graph]
            if not after:
                return None
            nid = after[-1]
        return None

    governable: Counter = Counter()
    ungoverned: Counter = Counter()
    for nid, count in (cal.get("first_death_by_node") or {}).items():
        if nid == "none":
            continue
        target = owner(nid)
        (governable if target else ungoverned)[target or nid] += count
    ranked = sorted(governable.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"governable": dict(governable), "ungoverned": dict(ungoverned),
            "governable_deaths": sum(governable.values()),
            "ungoverned_deaths": sum(ungoverned.values()),
            "target": ranked[0][0] if ranked else None,
            "target_deaths": ranked[0][1] if ranked else 0,
            "ranked": ranked}


# ── the honest boundary: does this embodiment HAVE recovery primitives? ──────

def node_spec_kwargs(task: str, node: dict) -> dict:
    """The EpisodeSpec kwargs the node runs under -- the SAME lookup the loop
    does (``SKILL_SPECS`` for manipulate, the card's ``segment_specs`` for
    segment), so the campaign governs the operating point the chain actually ran."""
    from harness.registry import load_provider
    from plugins.task.workload import SKILL_SPECS

    from scripts.harness_runtime import _load_attr

    binding = _binding(task)
    if node["kind"] == "segment":
        table = _load_attr(binding["segment_specs"]) if "segment_specs" in binding else {}
        base = dict(_load_attr(binding["episode"])) if "episode" in binding else {}
        kwargs = {**base, **dict(table.get(node["skill"]) or {})}
    else:
        kwargs = dict(SKILL_SPECS.get(node["skill"]) or {})
    if not kwargs:
        raise ValueError(
            f"node {node['id']!r} (skill {node['skill']!r}, kind {node['kind']!r}) has "
            "no execution binding; nothing to build an EpisodeSpec from")
    by_object = kwargs.pop("task_by_object", None)
    if by_object is not None:
        obj = node["args"].get("object")
        if obj not in by_object:
            raise ValueError(f"node {node['id']!r}: object {obj!r} has no task binding")
        kwargs["task"] = by_object[obj]
    if isinstance(kwargs.get("stages"), str):
        kwargs["stages"] = load_provider(kwargs["stages"])
    kwargs.pop("horizon", None)
    kwargs.pop("percept_provider", None)
    return kwargs


def embodiment_card(binding: dict) -> str:
    """The plugin dir name of the embodiment card this task's episodes run on.

    A binding that names its own ``env`` rides a second simulator; one that omits
    it rides the base fold's default. Either way the card name is the middle
    component of the provider ref -- the same plugin dir name whose manifest
    declares ``[recoveries.*]`` repair shapes.
    """
    from plugins.rsi.governed import DEFAULT_ENV_REF

    return (binding.get("env") or DEFAULT_ENV_REF).split(":")[0].split(".")[1]


def recovery_support(task: str, node: dict) -> dict:
    """Can a recovery primitive fire inside this node AT ALL? Answered, not assumed.

    Three independent facts, each reported by name rather than collapsed into a
    bare False, because "no primitive exists" and "the primitive exists but this
    node is not reachable by the campaign path" are different honest answers and
    the operator needs to know which one they got:

    * the embodiment card must register a recovery ``Strategy`` (repertoire),
      AND -- because a documented no-op ``retarget``/``on_handback`` would pass a
      bare ``hasattr`` probe while doing nothing -- its driver must be able to
      EXECUTE one: an episodic-segment driver needs ``enter_segment`` (the
      isolated-node campaign path) and ``make_recovery`` (build the actor for its
      own 12-dim action space). The kitchen driver now has both, so RoboCasa's
      grasp node is governable; a card with neither is reported as such and
      nothing is invented to fill the hole.
    * an un-importable driver is reported as un-importable, never silently as
      "unsupported" -- usually it means the wrong venv for this embodiment.
    """
    from harness.registry import load_provider

    binding = _binding(task)
    out = {"node": node["id"], "kind": node["kind"], "skill": node["skill"],
           "driver_ref": binding.get("policy"), "supported": False}
    if node["kind"] not in ACTUATING:
        out["reason"] = (f"节点 kind={node['kind']} 不驱动 governed_rollout，"
                         "没有可挂治理的面")
        return out
    try:
        kwargs = node_spec_kwargs(task, node)
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        out["reason"] = f"无法为该节点构造 EpisodeSpec: {exc}"
        return out
    try:
        from harness.spec import EpisodeSpec

        driver = load_provider(binding["policy"]).make_driver(
            EpisodeSpec(seed=0, **kwargs))
    except ImportError as exc:
        out["reason"] = (f"驱动 {binding.get('policy')!r} 不可导入 ({exc}) —— "
                         "无法确认该本体的恢复原语（多半是 venv 不对）")
        return out
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"驱动 {binding.get('policy')!r} 无法构造: {exc}"
        return out
    from plugins.rsi import repertoire

    out["driver"] = type(driver).__name__
    out["card"] = card = embodiment_card(binding)
    out["repertoire"] = strategies = repertoire.strategies_for(card)
    # Every blocker, not just the first: "this embodiment has no primitives" and
    # "this node is not reachable by the campaign path" are independent facts and
    # an operator told only one of them would fix the wrong thing.
    blockers = []
    if not strategies:
        blockers.append(
            f"该本体（卡 {card}）无注册恢复原语，RSI 无从下手: "
            "该卡的 manifest.toml 没有声明任何 [recoveries.*]。"
            f"robosuite 侧的 servo_descend/servo_probe 是模板（{repertoire.strategies_for('embodiment_robosuite')}），"
            "但它们说的是 tabletop 的 above/descend/close/lift 词汇，换本体即无意义。"
            "为新本体注册恢复原语是先决条件，不是本次可以现编的东西。")
    missing = [m for m in RECOVERY_PROTOCOL if not hasattr(driver, m)]
    if missing:
        blockers.append(
            f"驱动 {type(driver).__name__} 缺 {missing}"
            "（plugins/rsi/governed.py 在规则触发时调用它们）。")
    if node["kind"] == "segment":
        # The isolated-segment campaign path (governed.isolated_segment_rollout)
        # drives the target sub-goal in a fresh world per seed, so a persistent
        # mission IS governable now -- but only through a driver that can enter a
        # sub-goal (enter_segment) and build a recovery actor for its own action
        # space (make_recovery). A no-op retarget/on_handback is not enough.
        seg_missing = [m for m in ("enter_segment", "make_recovery")
                       if not hasattr(driver, m)]
        if seg_missing:
            blockers.append(
                f"segment 驱动 {type(driver).__name__} 缺 {seg_missing}: 隔离节点 "
                "campaign 路径需要 enter_segment（驱动子目标）+ make_recovery（在该本体 "
                "12 维动作空间里执行恢复）—— 有文档的 no-op 不算原语在场。")
    out["blockers"] = blockers
    out["supported"] = not blockers
    out["reason"] = (" ".join(blockers) if blockers
                     else f"驱动 {type(driver).__name__} 实现 {list(RECOVERY_PROTOCOL)}")
    return out


def _segment_isolation(task: str, node: dict, cal: dict) -> tuple[str, tuple[str, ...], int]:
    """The isolated-node rollout plumbing for an episodic ``segment`` target.

    Returns ``(mission_task, sub_goal_tasks, horizon)``: the mission env task the
    ONE world opens under, the ordered sub-goal tasks to drive (the target's
    ``segment`` ancestors up to and including it, mapped through the card's
    SEGMENT_SPECS -- the prefix re-establishes the target's precondition, the
    round-108 "drive from reset to the node's precondition" probe), and the
    episode horizon (above the summed caps so the target never truncates)."""
    from scripts.harness_runtime import _load_attr

    binding = _binding(task)
    graph = {g["id"]: g for g in cal.get("graph") or []}
    seg_specs = _load_attr(binding["segment_specs"])
    episode = dict(_load_attr(binding["episode"])) if "episode" in binding else {}
    chain: list[dict] = []
    nid: str | None = node["id"]
    seen: set[str] = set()
    while nid in graph and nid not in seen:
        seen.add(nid)
        g = graph[nid]
        if g["kind"] == "segment":
            chain.append(g)
        after = [a for a in g["after"] if a in graph]
        nid = after[-1] if after else None
    chain.reverse()  # oldest ancestor first, the target last
    subgoals = tuple(dict(seg_specs[g["skill"]])["task"] for g in chain)
    return episode.get("task", task), subgoals, int(episode.get("horizon", 4000))


def _plan_graph(task: str, max_actuations: int) -> list[dict]:
    """The task's node graph WITHOUT running an episode: the planner is a pure
    function, so a node-level run can read kinds/after edges up front (to decide
    isolation and the sub-goal chain) before paying for any calibration."""
    from harness.definitions import CAPABILITIES
    from harness.events import SessionLog
    from harness.kernel import Kernel

    from scripts import harness_runtime as hr

    binding = _binding(task)
    with tempfile.TemporaryDirectory() as empty_skills:
        kernel = Kernel(CAPABILITIES, log=SessionLog())
        kernel.mount(hr._mount_plan(binding, Path(empty_skills)))
        brief = hr.task_brief(task, binding)
        scene = kernel.resolve("graph.scene", consumer="rsi")
        plan = kernel.resolve("task.planner", consumer="rsi").plan(
            {**brief, "scene": scene.snapshot({}), "budget": max_actuations})
    return [{"id": n["id"], "skill": n["skill"], "kind": n.get("kind", "manipulate"),
             "after": list(n.get("after") or []), "args": dict(n.get("args") or {})}
            for n in (plan.get("nodes") or [])]


def _isolated_probe_one(job: tuple) -> dict:
    """ONE ungoverned isolated-node episode: drive the prefix sub-goals + the
    target on a fresh world, score the TARGET segment boolean. summarize-shaped,
    with the graph reduced to the single target node so ``attribute`` charges the
    residual to it (owner() sees kind=segment == ACTUATING and stops there)."""
    task, node_id, node_skill, mission_task, subgoals, horizon, seed = job
    from harness.spec import EpisodeSpec
    from plugins.rsi.governed import isolated_segment_rollout

    binding = _binding(task)
    from scripts.harness_runtime import _load_attr

    episode = dict(_load_attr(binding["episode"])) if "episode" in binding else {}
    spec = EpisodeSpec(
        seed=seed, task=mission_task,
        percept_noise=float(episode.get("percept_noise", 0.012)), horizon=horizon,
        env_provider=binding.get("env"), policy_provider=binding["policy"],
        percept_provider=binding.get("percept"), segment_isolate=subgoals)
    t0 = time.perf_counter()
    out = isolated_segment_rollout(spec, None)
    dt = time.perf_counter() - t0
    node = {"id": node_id, "skill": node_skill, "kind": "segment", "after": [], "args": {}}
    return {
        "seed": seed, "success": bool(out["success"]),
        "first_death": "none" if out["success"] else node_id,
        "graph": [node],
        "node_ok": {node_id: bool(out["success"])},
        "node_stages": {node_id: out.get("stages") or []},
        "replans": 0, "actuations": len(subgoals),
        "budget_exhaust": bool(out.get("steps") and out["steps"] >= horizon),
        "seconds": round(dt, 3),
    }


def calibrate_isolated(task: str, node: dict, mission_task: str, subgoals: tuple,
                       horizon: int, block, *, workers: int = 10,
                       out_dir: Path | None = None, extra: dict | None = None) -> dict:
    """Node-isolated calibration: the generic probe (``calibrate``) restricted to
    the ONE target segment, base rate = the target sub-goal boolean. Same
    summarize/attribute machinery, never gates, always re-runnable."""
    from harness.executor import LocalPoolExecutor

    ss = seeds(block)
    tick = None
    if out_dir is not None:
        from scripts.campaign_progress import tracker

        tick = tracker(out_dir, len(ss) + POST_CAL_STEPS,
                       label=f"rsi {task} · calibrate node {node['id']}", extra=extra)
    jobs = [(task, node["id"], node["skill"], mission_task, subgoals, horizon, s)
            for s in ss]
    if workers <= 1 or len(ss) == 1:
        rows = []
        for j in jobs:
            rows.append(_isolated_probe_one(j))
            if tick is not None:
                tick(rows[-1])
    else:
        rows = LocalPoolExecutor().map(_isolated_probe_one, jobs, workers=workers,
                                       on_result=tick)
    rows.sort(key=lambda r: r["seed"])
    return summarize(task, rows, block)


def repair_shape(cal: dict, node_id: str, allowed: list[str]) -> str:
    """Pick the repertoire strategy by the node's MEASURED dominant failing stage.

    ``plugins/rsi/governed.py:_is_place_recovery`` says a repair is place-shaped
    iff a phase names the place vocabulary; the same vocabulary decides here, so
    a node whose deaths are seated-placement deaths gets the place-shaped repair
    and a node whose deaths are grasp deaths gets the grasp-shaped one. Measured,
    not chosen.

    ``allowed`` is the embodiment card's REGISTERED repertoire, so this can only
    ever return a repair that exists for the robot in question.
    """
    failed: Counter = Counter()
    for row in cal.get("episodes") or []:
        for stage in row["node_stages"].get(node_id) or []:
            if not stage["success"]:
                failed[stage["name"]] += 1
                break
    place_shaped = bool(failed) and failed.most_common(1)[0][0] in STACK_PHASE_HEIGHT
    want = "replace" if place_shaped else "regrasp"
    return want if want in allowed else allowed[0]


# ── c. the mechanical gate (M7 §3, M6 §4) ───────────────────────────────────

#: Ceiling above which there is no residual left to learn from (M6 §4 c2).
BASE_CEILING = 0.90
#: A calibration + one dev generation longer than this defers the campaign
#: (M7 §3.5). Two hours, the number the design wrote down.
BUDGET_SECONDS = 2 * 3600


def gate(cal: dict, attribution: dict, support: dict | None, *,
         workers: int = 10, dev_n: int = DEV_N) -> dict:
    """Score the five go/no-go criteria and return proceed + the evidence.

    Every criterion is a scored FACT with its own numbers attached. A NO-GO is a
    finished result: it names the missing capability and burns nothing.
    """
    n, base = cal["n"], cal["base_rate"] or 0.0
    per_ep = cal["seconds_per_episode"] or 0.0
    # cal + one dev generation (both arms of a paired gate over the dev prefix)
    est = cal["seconds_total"] + (2 * dev_n * per_ep / max(workers, 1))
    criteria = [
        {"id": "c1_base_degenerate", "fail": base in (0.0, 1.0),
         "detail": f"base_rate={base:.4g} ({cal['successes']}/{n})",
         "verdict": "基率 0% 或 100% 则无残余可学，任何门都学不到东西"},
        {"id": "c2_base_ceiling", "fail": base >= BASE_CEILING,
         "detail": f"base_rate={base:.4g} vs ceiling {BASE_CEILING}",
         "verdict": "基率已在天花板上 → 诚实 null，不烧 dev/held-out"},
        {"id": "c3_budget_exhaust_dominant",
         "fail": cal["budget_exhaust"] > (n - cal["successes"]) - cal["budget_exhaust"]
                 and cal["budget_exhaust"] > 0,
         "detail": f"budget_exhaust={cal['budget_exhaust']} of {n - cal['successes']} failures",
         "verdict": "多数死于预算耗尽 → 调 max_actuations/horizon（配置），不是 RSI"},
        {"id": "c4_attribution",
         "fail": attribution["ungoverned_deaths"] >= attribution["governable_deaths"]
                 or attribution["target"] is None,
         "detail": (f"governable={attribution['governable_deaths']} "
                    f"ungoverned={attribution['ungoverned_deaths']} "
                    f"target={attribution['target']}"),
         "verdict": "链主要死在未治理节点 → 归因 pivot，先要那个节点的能力，不进化"},
        {"id": "c5_recovery_primitive",
         "fail": support is None or not support["supported"],
         "detail": (support or {}).get("reason", "无目标节点，未做恢复原语检查"),
         "verdict": "目标节点没有可用的恢复原语 → RSI 无从下手"},
        {"id": "c6_wall_clock", "fail": est > BUDGET_SECONDS,
         "detail": f"est cal+1gen={est/3600:.2f}h at {workers} workers "
                   f"({per_ep:.3g}s/episode)",
         "verdict": "标定 + 一代 dev 超 2h → 今夜只标定，不抢跑长集"},
    ]
    failed = [c["id"] for c in criteria if c["fail"]]
    return {"proceed": not failed, "failed": failed, "criteria": criteria,
            "target_node": attribution["target"] if not failed else None,
            # verdict AND the measurement that tripped it: a NO-GO an operator
            # cannot act on is barely better than no NO-GO at all.
            "missing_capability": [f"{c['verdict']}（{c['detail']}）"
                                   for c in criteria if c["fail"]],
            "estimated_seconds_cal_plus_one_generation": round(est, 1)}


# ── d/e/f. prereg + dev campaign + held-out ─────────────────────────────────

def build_prereg(task: str, node: dict, cal: dict, dev_block, heldout_block,
                 allowed: list[str]):
    """The preregistration for the chosen node, FROM-SCRATCH.

    ``critic_budget=0`` is what makes the trigger search structurally unable to
    reach a privileged feature -- the "prefer non-privileged" rule expressed as a
    budget rather than a preference. A privileged rule can only enter by raising
    the budget, and ``run_campaign`` runs the transfer ablation at every promotion
    regardless, so a privileged gain always arrives with its collapse curve.
    """
    from plugins.rsi.campaign import Preregistration

    kwargs = node_spec_kwargs(task, node)
    prereg_task = kwargs["task"]
    seg_isolate: tuple[str, ...] | None = None
    horizon = 900
    binding = _binding(task)
    if node["kind"] == "segment" and binding.get("episodic"):
        # Node-level RSI on a persistent mission: the campaign opens the MISSION
        # env and scores the isolated TARGET sub-goal (governed.isolated_segment_
        # rollout). ``task`` becomes the mission task so make_env resolves; the
        # sub-goal chain + horizon ride the prereg.
        prereg_task, seg_isolate, horizon = _segment_isolation(task, node, cal)
    return Preregistration(
        dev=tuple(seeds(dev_block)), heldout=tuple(seeds(heldout_block)),
        percept_noise=float(kwargs.get("percept_noise", 0.012)),
        critic_budget=0, action_budget=0, recovery_sensor_sd=0.020,
        max_generations=2, task=prereg_task, policy="scripted",
        stages=kwargs.get("stages"),
        terminal_label=bool(kwargs.get("terminal_label", False)),
        scale_dev_by_power=True, require_judgement=True,
        recovery_name=repair_shape(cal, node["id"], allowed),
        parent_store=None, segment_isolate=seg_isolate, horizon=horizon,
    )


def _campaign_kernel(task: str, out: Path):
    """base_profile + the task binding's own policy/env/percept + an on-disk
    skill graph. The same mount ``harness_runtime._mount_plan`` gives the task,
    minus the planner (a campaign drives one node, not the graph)."""
    from harness.config import Mount, Patch, resolve_plan
    from harness.definitions import CAPABILITIES
    from harness.events import SessionLog
    from harness.kernel import Kernel
    from profiles import base_profile

    binding = _binding(task)
    override = [Mount("policy.driver", binding["policy"]),
                Mount("graph.skill", "plugins.graphs:skill_graph_provider",
                      {"root": str(out / "skills")})]
    for cap, key in (("embodiment.env", "env"), ("percept.model", "percept")):
        if key in binding:
            override.append(Mount(cap, binding[key]))
    kernel = Kernel(CAPABILITIES, log=SessionLog(out / "session-log"))
    kernel.mount(resolve_plan(base_profile(),
                              patches=(Patch("rsi_campaign", override=tuple(override)),)))
    return kernel


# ── h. the ledger entry (printed, never appended) ────────────────────────────

def ledger_entry(task: str, blocks: dict, cal: dict, verdict: dict,
                 result: dict | None) -> str:
    """A STATUS.md 区块预算-shaped paragraph for the operator to paste (or not).

    Printed rather than appended on purpose: STATUS.md is human-authored, and a
    second writer racing the operator's own edit is exactly the corruption the
    one-prose-ledger rule exists to prevent.
    """
    head = (f"**标定 {blocks['cal'][0]}-{blocks['cal'][1]}({cal['n']} 席, baseline 臂, "
            f"通用 RSI 路径 scripts/rsi_campaign.py --task {task}):** 链基率 "
            f"{(cal['base_rate'] or 0):.2%}({cal['successes']}/{cal['n']}), "
            f"首死 {cal['first_death_by_node']}, 按机制 {cal['first_death_by_mechanism']}, "
            f"~{cal['seconds_per_episode']}s/集; 标定块永不再当门禁/held-out(可复测)。")
    unburned = (f" dev {blocks['dev'][0]}-{blocks['dev'][1]} / held-out "
                f"{blocks['heldout'][0]}-{blocks['heldout'][1]} **未烧**。")
    if not verdict["proceed"]:
        return (head + f" **门禁 NO-GO**({', '.join(verdict['failed'])}): "
                + " ".join(verdict["missing_capability"]) + unburned)
    if result is None:
        return (head + f" **门禁 GO**, 目标节点 {verdict['target_node']}; "
                "本次只跑到门禁(--stop-after gate)," + unburned)
    r = result.get("result", {})
    held = r.get("heldout") or {}
    tail = (f" 门禁 GO, 目标节点 {verdict['target_node']}。**dev 蓄水池 "
            f"{blocks['dev'][0]}-{blocks['dev'][1]}(功效缩放前缀):** "
            f"{r.get('generations')} 代 {r.get('promoted')} 晋级, "
            f"final_sha {str(r.get('final_sha'))[:12]}, rules={r.get('rules')}。")
    if held:
        tail += (f"**held-out {blocks['heldout'][0]}-{blocks['heldout'][1]}"
                 f"({held.get('n')} 席, 评一次):** {held.get('base_rate', 0):.1%} → "
                 f"{held.get('governed_rate', 0):.1%}, fixed {held.get('fixed')} / "
                 f"broken {held.get('broken')}, p={held.get('p_value')}。")
    else:
        tail += f"零晋级 = 诚实 null, held-out {blocks['heldout'][0]}-{blocks['heldout'][1]} **未烧**。"
    return head + tail


# ── the chain ────────────────────────────────────────────────────────────────

def run_chain(task: str, out: Path, *, workers: int = 10, stop_after: str = "heldout",
              cal=None, dev=None, heldout=None, node: str | None = None,
              floor: int = 0, max_replans: int = 3, max_actuations: int = 40,
              runs_root: Path | None = None) -> dict:
    """a -> h for one task. Returns the whole chain's report; writes it to
    ``<out>/rsi_report.json`` and heartbeats ``<out>/progress.json`` per stage."""
    from scripts.campaign_progress import write_progress

    out.mkdir(parents=True, exist_ok=True)
    # derived from sealed preregistrations: no store at all refuses, never "nothing burned"
    blocks = allocate(burned_blocks(runs_root or REPO / "runs"),
                      floor=floor, cal=cal, dev=dev, heldout=heldout)
    report: dict = {"task": task, "blocks": {k: list(v) for k, v in blocks.items()},
                    "stage": "calibrate"}

    # ONE heartbeat denominator for the whole chain (see POST_CAL_STEPS): the
    # calibration episodes plus gate/dev/held-out, so the 演进 panel's card stays
    # live through the tail stages and its `stage` field names where the chain is.
    cal_n = len(seeds(blocks["cal"]))
    total = cal_n + POST_CAL_STEPS

    def beat(stage: str, done: int, **extra) -> None:
        report["stage"] = stage
        write_progress(out, done, total, label=f"rsi {task} · {stage}",
                       extra={"stage": stage, "task": task,
                              "blocks": {k: list(v) for k, v in blocks.items()},
                              **extra})

    # Node-level RSI on an episodic ``segment`` target calibrates the TARGET node
    # ALONE (round-108 template): the whole-chain rate on a persistent mission is
    # 0% by construction (thawed needs every sub-goal) and would trip c1, so an
    # explicit segment ``node`` override isolates it -- drive the prefix sub-goals
    # to the node's precondition, score the node's own boolean.
    binding = _binding(task)
    graph_nodes = _plan_graph(task, max_actuations)
    isolate = (node is not None
               and {g["id"]: g for g in graph_nodes}.get(node, {}).get("kind") == "segment"
               and bool(binding.get("episodic")))
    report["isolated_node"] = node if isolate else None

    beat("calibrate", 0)
    cal_extra = {"stage": "calibrate", "task": task,
                 "blocks": {k: list(v) for k, v in blocks.items()}}
    if isolate:
        target_node = {g["id"]: g for g in graph_nodes}[node]
        mission_task, subgoals, horizon = _segment_isolation(
            task, target_node, {"graph": graph_nodes})
        report["segment_isolate"] = {"mission_task": mission_task,
                                     "sub_goals": list(subgoals), "horizon": horizon}
        cal = calibrate_isolated(task, target_node, mission_task, subgoals, horizon,
                                 blocks["cal"], workers=workers, out_dir=out,
                                 extra={**cal_extra, "node": node})
    else:
        cal = calibrate(task, blocks["cal"], workers=workers, out_dir=out,
                        max_replans=max_replans, max_actuations=max_actuations,
                        extra=cal_extra)
    report["calibration"] = {k: v for k, v in cal.items() if k != "episodes"}
    (out / "calibration.json").write_text(json.dumps(cal, indent=1, sort_keys=True,
                                                     default=str))

    beat("gate", cal_n)
    attribution = attribute(cal)
    report["attribution"] = attribution
    target_id = node or attribution["target"]
    graph = {g["id"]: g for g in cal["graph"]}
    target = graph.get(target_id) if target_id else None
    support = recovery_support(task, target) if target else None
    report["recovery_support"] = support
    verdict = gate(cal, attribution, support, workers=workers)
    if node is not None:
        verdict["node_override"] = node
    report["gate"] = verdict
    beat("gate", cal_n + 1, verdict="GO" if verdict["proceed"] else "NO-GO",
         target_node=verdict["target_node"], failed=verdict["failed"])

    if not verdict["proceed"] or stop_after == "gate":
        report["stage"] = "stopped"
        report["ledger_entry"] = ledger_entry(task, blocks, cal, verdict, None)
        (out / "rsi_report.json").write_text(json.dumps(report, indent=1,
                                                        sort_keys=True, default=str))
        beat("stopped", total, verdict="GO" if verdict["proceed"] else "NO-GO",
             target_node=verdict["target_node"], failed=verdict["failed"])
        return report

    # d. seal the prereg BEFORE a single dev seed runs.
    from plugins.rsi.workload import run as rsi_run

    prereg = build_prereg(task, target, cal, blocks["dev"], blocks["heldout"],
                          support["repertoire"])
    kernel = _campaign_kernel(task, out)
    stamped = dataclasses.replace(
        prereg,
        env_provider=kernel.provider_ref("embodiment.env"),
        policy_provider=kernel.provider_ref("policy.driver"),
        percept_provider=kernel.provider_ref("percept.model"),
        reasoner=kernel.resolve("reasoner.proposer", consumer="rsi").identity).sha()
    report["preregistration_sha"] = stamped
    report["recovery_name"] = prereg.recovery_name
    beat("dev", cal_n + 1, prereg_sha=stamped[:12],
         target_node=verdict["target_node"], recovery=prereg.recovery_name)

    # e/f. dev generations, then held-out ONCE and only on a promotion.
    outcome = rsi_run(prereg, out, kernel, workers=workers)
    report["result"] = outcome["result"]
    report["skills"] = outcome["skills"]
    report["stage"] = "done"
    report["ledger_entry"] = ledger_entry(task, blocks, cal, verdict, outcome)
    (out / "rsi_report.json").write_text(json.dumps(report, indent=1, sort_keys=True,
                                                    default=str))
    beat("done", total, promoted=outcome["result"].get("promoted"),
         target_node=verdict["target_node"])
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--stop-after", choices=("gate", "heldout"), default="heldout")
    ap.add_argument("--node", default=None,
                    help="override the attribution's target node (audited, not free: "
                         "the override is recorded in the gate verdict)")
    ap.add_argument("--floor", type=int, default=0)
    ap.add_argument("--max-replans", type=int, default=3)
    ap.add_argument("--max-actuations", type=int, default=40)
    for name in ("cal", "dev", "heldout"):
        ap.add_argument(f"--{name}", default=None, metavar="LO:HI",
                        help=f"pin the {name} block (inclusive)")
    args = ap.parse_args(argv)

    def block(spec):
        if spec is None:
            return None
        lo, hi = (int(x) for x in spec.split(":", 1))
        return (lo, hi)

    report = run_chain(
        args.task, args.out, workers=args.workers, stop_after=args.stop_after,
        cal=block(args.cal), dev=block(args.dev), heldout=block(args.heldout),
        node=args.node, floor=args.floor, max_replans=args.max_replans,
        max_actuations=args.max_actuations)
    print(json.dumps({k: v for k, v in report.items() if k != "calibration"},
                     indent=1, sort_keys=True, default=str))
    print("\n=== STATUS.md 账本条目（给操作员，未自动写入）===")
    print(report["ledger_entry"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
