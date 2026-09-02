"""``harness.contracts.TaskPlanner`` + PREDICATES for ``recycle_cans`` (M7 on
RoboCasa, RecycleSodaCans): four soda cans scattered across the kitchen, each
fetched and clustered on the counter beside the stove in ONE persistent episode.

Sibling of ``plugins/mission_kitchen_thaw/planner.py`` -- same two halves, but
the chain is a PER-CAN loop instead of one linear appliance path:

    survey -> plan-order -> [ nav -> at -> grasp -> grasped -> carry ->
    drop -> placed ] x4 cans -> sweep -> report            (32 nodes)

* survey (perceive) reads every can's LIVE pose + the base pose + the
  ``get_ep_meta`` scene fingerprint. plan-order (decide) ranks the cans by
  distance from the surveyed base pose -- the recorded fetch-order decision;
  the GRAPH order stays the planner's fixed can1..can4 (a deterministic planner
  is a pure fn of the task; the base loop skips finished nodes, so the ranking
  is the decide node's reading, not a graph mutation -- same stance as
  kitchen_thaw's plan node).
* every grasped verify is SECURE_DZ-shaped: robocasa's ``check_obj_grasped`` is
  a false-positive latch (carry-probe), so the verify ALSO requires the can's
  live z to have risen SECURE_DZ above its surveyed entry z -- the relational
  proof the can actually moved with the hand.
* placed verifies read RecycleSodaCans's own success primitives per can (within
  0.25 m of the stove bbox + counter contact + gripper released); sweep
  (perceive) re-reads all four cans for leftovers; report (decide) cross-checks
  the sealed segments against the live oracle, headline ``recycled`` = the
  env's own ``_check_success``.

Predicates reach the robocasa card's parametric primitives purely by ref via
``load_provider`` (never a sibling import; tests/test_boundaries.py).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from harness import opstream, protocol
from harness.registry import load_provider
from harness.skill_library import RECORDS, catalogue_of, segment_specs, select

# ── card constants ────────────────────────────────────────────────────────────
#: RecycleSodaCans's own object names (its _check_success reads them).
CANS: tuple[str, ...] = ("can1", "can2", "can3", "can4")

#: The relational secure-grasp margin (mirrors drivers.GraspDriver.SECURE_DZ).
SECURE_DZ = 0.08

#: The robocasa card's parametric predicate module (reached by ref only).
_P = "plugins.embodiment_robocasa.predicates"

#: Per-can chain: (segment id, segment skill, verify id or None, verify skill).
_CHAIN: tuple[tuple[str, str, str | None, str | None], ...] = tuple(
    step
    for c in CANS
    for step in (
        (f"nav-{c}", f"nav_{c}", f"at-{c}", f"v_at_{c}"),
        (f"grasp-{c}", f"grasp_{c}", f"grasped-{c}", f"v_grasped_{c}"),
        (f"carry-{c}", f"carry_{c}", None, None),
        (f"drop-{c}", f"drop_{c}", f"placed-{c}", f"v_placed_{c}"),
    )
)
_SEG_IDS: tuple[str, ...] = tuple(seg for seg, *_ in _CHAIN)


# ── symbolic layer ────────────────────────────────────────────────────────────

#: The card's slice of the static skill library (skill-library/records): the
#: symbolic contracts Supported/Covered judge; CATALOGUE is its typed view.
SKILL_RECORDS = select(RECORDS, "robocasa", (
    "survey", "plan_order", "sweep", "report",
    *(skill for _, skill, _, _ in _CHAIN),
    *(vskill for _, _, _, vskill in _CHAIN if vskill)))
CATALOGUE: dict[str, dict[str, type]] = catalogue_of(SKILL_RECORDS)
#: sigma0 facts the card declares true at reset (no live predicate binding
#: exists for the library vocabulary yet), the base of every Supported chain.
INITIAL_FACTS: tuple[str, ...] = tuple([*(f"present({c})" for c in CANS), "gripper_free()"])

ORACLES: tuple[str, ...] = ("staged", "reported")

PREDICATES: dict[str, str] = {
    "survey": "plugins.mission_recycle_cans.planner:survey",
    "plan_order": "plugins.mission_recycle_cans.planner:plan_order",
    "sweep": "plugins.mission_recycle_cans.planner:sweep",
    "report": "plugins.mission_recycle_cans.planner:report",
    **{f"v_at_{c}": f"plugins.mission_recycle_cans.planner:v_at_{c}" for c in CANS},
    **{f"v_grasped_{c}": f"plugins.mission_recycle_cans.planner:v_grasped_{c}"
       for c in CANS},
    **{f"v_placed_{c}": f"plugins.mission_recycle_cans.planner:v_placed_{c}"
       for c in CANS},
}

#: 16 driven sub-goals x up to 1600 capped steps each, plus in-episode retries.
EPISODE: dict[str, Any] = {
    "task": "recycle_cans",
    "percept_noise": 0.012,
    "percept_provider": "plugins.embodiment_robocasa.percept:provider",
    "horizon": 8000,
}

SEGMENT_SPECS: dict[str, dict[str, Any]] = segment_specs(SKILL_RECORDS, "robocasa")


def _emit_plan() -> Mapping:
    nodes: list[dict] = [
        {"id": "survey", "skill": "survey", "kind": "perceive", "args": {},
         "after": []},
        {"id": "plan-order", "skill": "plan_order", "kind": "decide", "args": {},
         "after": ["survey"]},
    ]
    verify_list: list[dict] = []
    prev = "plan-order"
    for seg_id, seg_skill, ver_id, ver_skill in _CHAIN:
        nodes.append({"id": seg_id, "skill": seg_skill, "kind": "segment",
                      "args": {}, "after": [prev]})
        verify_list.append({"after": seg_id, "predicate": "staged"})
        if ver_id is not None:
            nodes.append({"id": ver_id, "skill": ver_skill, "kind": "verify",
                          "args": {}, "after": [seg_id]})
            prev = ver_id
        else:
            prev = seg_id
    nodes.append({"id": "sweep", "skill": "sweep", "kind": "perceive",
                  "args": {}, "after": [prev]})
    nodes.append({"id": "report", "skill": "report", "kind": "decide",
                  "args": {}, "after": ["sweep"]})
    verify_list.append({"after": "report", "predicate": "reported"})
    return json.loads(json.dumps({
        "goal": "in ONE persistent kitchen episode, gather all four scattered "
                "soda cans and cluster them on the counter beside the stove -- "
                "survey, rank the cans by distance, then per can drive "
                "nav/grasp/carry/drop segments with live-state verifies, "
                "retrying a failed sub-goal in the SAME world, sweeping for "
                "strays and closing with a machine report",
        "nodes": nodes,
        "verify": verify_list,
    }, sort_keys=True))


#: Stage word of a node id -> the embodiment_robocasa recovery a no_progress
#: replan inserts before it (a done recover-<id> re-inserts byte-identically from
#: the fault's recoveries_done -- protocol.recover_plan).
_RECOVERY: dict[str, str] = {"nav": "redock_retry", "carry": "redock_retry",
                            "grasp": "regrasp_kitchen", "drop": "reapproach"}
#: failure_mode -> stage word -> the repair that overrides _RECOVERY: an arm that
#: stalled reaching a drop point is at its envelope edge, so the base moves
#: (base_nudge) instead of re-approaching from where it stood.
_RECOVERY_BY_MODE: dict[str, dict[str, str]] = {"reach_stall": {"drop": "base_nudge"}}


class RecycleCansPlanner:
    """Deterministic emitter of the fixed 32-node per-can mission graph; the
    in-episode retry is the base loop re-running a failed node (max_replans)."""

    def plan(self, brief: Mapping) -> Mapping:
        task = brief.get("task")
        if task != "recycle_cans":
            raise ValueError(
                f"RecycleCansPlanner only plans 'recycle_cans', got {task!r}")
        # Stateless emitter: done recover-<id> nodes re-insert byte-identically;
        # a no_progress refusal (the graph was tried twice at that node) gets the
        # card's repair for that stage before it instead of a third copy.
        return protocol.recover_plan(_emit_plan(), brief.get("fault"), _RECOVERY, _RECOVERY_BY_MODE)

    @property
    def identity(self) -> str:
        return "recycle_cans_planner@v3"  # v3: per-stage recovery table


def provider(**params: Any) -> RecycleCansPlanner:
    return RecycleCansPlanner(**params)


# ── predicate layer ───────────────────────────────────────────────────────────

def _episode(ctx):
    ep = getattr(ctx, "episode", None)
    if ep is None:
        raise ValueError(
            "recycle_cans predicate reached with no persistent episode; the "
            "binding must declare episodic=true (workload threads ctx.episode)")
    return ep


def _obj_z(env, name: str) -> float:
    return float(np.asarray(env.sim.data.body_xpos[env.obj_body_id[name]])[2])


def _plain(v):
    # numpy scalars -> plain python so the fingerprint survives json.dumps
    # (opstream.emit silently drops a non-serialisable event).
    if v is None or isinstance(v, (bool, str)):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


def survey():
    """PERCEIVE: every can's LIVE pose + the base pose + the scene fingerprint,
    sealed as facts (the SECURE_DZ verifies and plan-order read them back)."""
    def pred(node: Mapping, ctx) -> dict:
        ep = _episode(ctx)
        env, obs = ep.env, ep.obs
        cans = {c: [float(v) for v in np.asarray(obs[f"{c}_pos"])[:3]]
                for c in CANS}
        bid = env.sim.model.body_name2id("mobilebase0_base")
        base_xy = [float(v) for v in np.asarray(env.sim.data.body_xpos[bid])[:2]]
        try:
            meta = env.get_ep_meta()
        except Exception:  # noqa: BLE001 -- a scene with no meta still surveys
            meta = {}
        scene = {"lang": (str(meta["lang"]) if meta.get("lang") is not None else None),
                 "layout_id": _plain(getattr(env, "layout_id", meta.get("layout_id"))),
                 "style_id": _plain(getattr(env, "style_id", meta.get("style_id")))}
        opstream.emit("scene_meta", **{k: v for k, v in scene.items() if v is not None})
        ok = all(np.isfinite(v) and abs(v) < 100
                 for pos in cans.values() for v in pos)
        return {"success": bool(ok),
                "facts": {"scene": scene, "cans": cans, "base_xy": base_xy},
                "privilege": ["privileged.object_z"]}
    return pred


def plan_order():
    """DECIDE: rank the four cans by distance from the surveyed base pose --
    the deterministic fetch-order reading. Informational for execution (the
    graph order is fixed by the planner; the loop skips finished nodes), sealed
    for the report to cross-check."""
    def pred(node: Mapping, ctx) -> dict:
        sv = ctx.nodes_out.get("survey") or {}
        facts = sv.get("facts") or {}
        cans, base = facts.get("cans") or {}, facts.get("base_xy")
        ready = bool(sv.get("success") and base and set(CANS) <= set(cans))
        order = (sorted(CANS, key=lambda c: float(
            np.hypot(cans[c][0] - base[0], cans[c][1] - base[1])))
            if ready else [])
        return {"success": ready,
                "decision": {"order": order, "scene": facts.get("scene")}}
    return pred


def _mk_at(name: str):
    def factory():
        prim = load_provider(f"{_P}:base_near_obj", {"name": name, "th": 1.5})

        def pred(node: Mapping, ctx) -> dict:
            return {"success": bool(prim(_episode(ctx).env))}
        return pred
    return factory


def _mk_grasped(name: str):
    """SECURE_DZ-shaped grasp verify: never the bare ``check_obj_grasped`` --
    that latch passes with the gripper merely touching the can (carry-probe's
    false-positive finding). The z-rise evidence is EITHER the can's live z
    above its surveyed entry z, OR the grasp segment's own sealed success --
    GraspDriver.done is itself SECURE_DZ-gated from segment-entry z, which stays
    truthful when a failed first attempt knocked the can somewhere lower and the
    in-episode retry grasped it from there (measured: seed 4243 can2, the
    survey-z-only verify falsely failed a real regrasp). Both branches still
    require the latch to hold NOW."""
    def factory():
        grasped = load_provider(f"{_P}:obj_grasped_any", {"name": name})

        def pred(node: Mapping, ctx) -> dict:
            ep = _episode(ctx)
            facts = (ctx.nodes_out.get("survey") or {}).get("facts") or {}
            pos0 = (facts.get("cans") or {}).get(name)
            if not pos0:
                return {"success": False}
            risen = _obj_z(ep.env, name) > float(pos0[2]) + SECURE_DZ
            seg_secure = bool((ctx.nodes_out.get(f"grasp-{name}") or {})
                              .get("success"))
            return {"success": bool(grasped(ep.env) and (risen or seg_secure))}
        return pred
    return factory


def _mk_placed(name: str):
    def factory():
        near = load_provider(f"{_P}:obj_near_fixture",
                             {"name": name, "fixture": "stove", "th": 0.25})
        on_counter = load_provider(f"{_P}:obj_on_counter", {"name": name})
        released = load_provider(f"{_P}:gripper_far", {"name": name})

        def pred(node: Mapping, ctx) -> dict:
            env = _episode(ctx).env
            return {"success": bool(near(env) and on_counter(env)
                                    and released(env))}
        return pred
    return factory


for _c in CANS:
    globals()[f"v_at_{_c}"] = _mk_at(_c)
    globals()[f"v_grasped_{_c}"] = _mk_grasped(_c)
    globals()[f"v_placed_{_c}"] = _mk_placed(_c)


def sweep():
    """PERCEIVE: re-read every can live after the four chains -- which are
    delivered (near stove + on a counter) and which are strays. Success = the
    positions are readable; a stray is a sealed fact, not a failure (report and
    the operator read it)."""
    def factory_preds():
        return {c: (load_provider(f"{_P}:obj_near_fixture",
                                  {"name": c, "fixture": "stove", "th": 0.25}),
                    load_provider(f"{_P}:obj_on_counter", {"name": c}))
                for c in CANS}
    preds = factory_preds()

    def pred(node: Mapping, ctx) -> dict:
        ep = _episode(ctx)
        env = ep.env
        pos = {c: [float(v) for v in
                   np.asarray(env.sim.data.body_xpos[env.obj_body_id[c]])[:3]]
               for c in CANS}
        delivered = [c for c in CANS if preds[c][0](env) and preds[c][1](env)]
        strays = [c for c in CANS if c not in delivered]
        ok = all(np.isfinite(v) for p in pos.values() for v in p)
        return {"success": bool(ok),
                "facts": {"delivered": delivered, "strays": strays, "cans": pos},
                "privilege": ["privileged.object_z"]}
    return pred


def report():
    """DECIDE: cross-check the sealed segment outcomes against the live oracle;
    headline ``recycled`` is RecycleSodaCans's own _check_success (clustered +
    near stove + on counter + released, all four)."""
    def factory_preds():
        return {c: load_provider(f"{_P}:obj_near_fixture",
                                 {"name": c, "fixture": "stove", "th": 0.25})
                for c in CANS}
    near = factory_preds()

    def pred(node: Mapping, ctx) -> dict:
        ep = _episode(ctx)
        out = ctx.nodes_out
        live: dict[str, Any] = {}
        ok = True
        for c in CANS:
            try:
                live[f"{c}_near_stove"] = bool(near[c](ep.env))
            except Exception:  # noqa: BLE001 -- an unreadable check breaks the report
                live[f"{c}_near_stove"] = None
                ok = False
        try:
            recycled = bool(ep.env._check_success())
        except Exception:  # noqa: BLE001
            recycled, ok = None, False
        segments = {sid: bool((out.get(sid) or {}).get("success"))
                    for sid in _SEG_IDS}
        sweep_facts = (out.get("sweep") or {}).get("facts") or {}
        return {"success": bool(ok),
                "decision": {"live": live, "segments": segments,
                             "strays": sweep_facts.get("strays"),
                             "recycled": recycled}}
    return pred


if __name__ == "__main__":
    plan = RecycleCansPlanner().plan({"task": "recycle_cans"})
    assert set(plan) == {"goal", "nodes", "verify"} and plan["goal"]
    ids = [n["id"] for n in plan["nodes"]]
    assert len(ids) == 32 and len(set(ids)) == 32, len(ids)
    kinds = [n.get("kind", "manipulate") for n in plan["nodes"]]
    assert kinds.count("segment") == 16 and kinds.count("verify") == 12
    assert kinds.count("perceive") == 2 and kinds.count("decide") == 2
    assert "manipulate" not in kinds
    seen: list[str] = []
    for n in plan["nodes"]:
        assert n["skill"] in CATALOGUE, n
        assert all(a in seen for a in n["after"]), n["id"]
        seen.append(n["id"])
        if n.get("kind") in ("perceive", "decide", "verify"):
            assert n["skill"] in PREDICATES, n
    for _, seg_skill, _, _ in _CHAIN:
        assert seg_skill in SEGMENT_SPECS, seg_skill
    assert json.dumps(plan, sort_keys=True) == json.dumps(
        RecycleCansPlanner().plan({"task": "recycle_cans"}), sort_keys=True)
    for ref in PREDICATES.values():
        assert callable(load_provider(ref))

    # decide ranks by surveyed distance (pure-fn check on fakes)
    from dataclasses import dataclass

    @dataclass
    class _Ctx:
        episode: Any
        nodes_out: dict
    facts = {"cans": {"can1": [3.0, 0.0, 1.0], "can2": [1.0, 0.0, 1.0],
                      "can3": [2.0, 0.0, 1.0], "can4": [0.5, 0.0, 1.0]},
             "base_xy": [0.0, 0.0]}
    ctx = _Ctx(None, {"survey": {"success": True, "facts": facts}})
    dec = plan_order()({}, ctx)
    assert dec["success"] and dec["decision"]["order"] == \
        ["can4", "can2", "can3", "can1"]
    # missing survey -> honest failure
    assert plan_order()({}, _Ctx(None, {}))["success"] is False
    try:
        RecycleCansPlanner().plan({"task": "stack"})
    except ValueError:
        pass
    else:
        raise AssertionError("wrong task must fail loudly")
    print("plugins/mission_recycle_cans/planner.py self-check OK")
