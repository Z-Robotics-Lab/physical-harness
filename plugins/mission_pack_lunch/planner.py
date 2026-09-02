"""``harness.contracts.TaskPlanner`` + PREDICATES for ``pack_lunch`` (M7 on
RoboCasa, PackFoodByTemp): two hot items in the stove area + two cold items in
the (already-open) fridge, packed by temperature into two tupperwares on the
dining counter, ONE persistent episode.

Same two halves as the recycle_cans sibling; the shape is a per-item loop with
one extra DETERMINISTIC decision:

    survey -> sort-temp -> [ nav -> at -> grasp -> grasped -> carry ->
    pack -> packed ] x4 items -> report                    (31 nodes)

* sort-temp (decide) assigns each item its target container BY TEMPERATURE
  ATTRIBUTE -- hot -> tupperware0, cold -> tupperware1 -- a pure function of
  the item names (PackFoodByTemp's own _check_success accepts either consistent
  assignment; this is the fixed one the driver's stage table is authored from,
  so the decision and the actuation can never disagree).
* grasped verifies are SECURE_DZ-shaped (grasped AND live z risen above the
  surveyed entry z) -- never the bare check_obj_grasped latch.
* packed verifies read PackFoodByTemp's own primitive per item
  (check_obj_in_receptacle against the ASSIGNED tupperware + gripper released);
  report cross-checks live, headline ``packed`` = the env's own _check_success.

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
#: PackFoodByTemp's own object names; temperature is the name's own attribute.
#: Chain order is shallow-reach-first: the open-counter item (hot1, on a plate
#: beside the stove), then the stove-pan item (hot0), then the fridge-rack items
#: (cold0/cold1) -- appliance-interior grasps are the measured hard tail
#: (calibration-r1's fridge enclosure finding), so the graph banks the
#: reachable work before attacking them.
ITEMS: tuple[str, ...] = ("hot1", "hot0", "cold0", "cold1")

#: The deterministic temperature -> container assignment (the decide node's
#: function AND the driver stage table's authoring source).
TARGET: dict[str, str] = {"hot0": "tupperware0", "hot1": "tupperware0",
                          "cold0": "tupperware1", "cold1": "tupperware1"}

SECURE_DZ = 0.08
_P = "plugins.embodiment_robocasa.predicates"

_CHAIN: tuple[tuple[str, str, str | None, str | None], ...] = tuple(
    step
    for it in ITEMS
    for step in (
        (f"nav-{it}", f"nav_{it}", f"at-{it}", f"v_at_{it}"),
        (f"grasp-{it}", f"grasp_{it}", f"grasped-{it}", f"v_grasped_{it}"),
        (f"carry-{it}", f"carry_{it}", None, None),
        (f"pack-{it}", f"pack_{it}", f"packed-{it}", f"v_packed_{it}"),
    )
)
_SEG_IDS: tuple[str, ...] = tuple(seg for seg, *_ in _CHAIN)


# ── symbolic layer ────────────────────────────────────────────────────────────

#: The card's slice of the static skill library (skill-library/records): the
#: symbolic contracts Supported/Covered judge; CATALOGUE is its typed view.
SKILL_RECORDS = select(RECORDS, "robocasa", (
    "survey", "sort_temp", "report",
    *(skill for _, skill, _, _ in _CHAIN),
    *(vskill for _, _, _, vskill in _CHAIN if vskill)))
CATALOGUE: dict[str, dict[str, type]] = catalogue_of(SKILL_RECORDS)
#: sigma0 facts the card declares true at reset (no live predicate binding
#: exists for the library vocabulary yet), the base of every Supported chain.
INITIAL_FACTS: tuple[str, ...] = tuple([*(f"present({it})" for it in ITEMS), "gripper_free()"])

ORACLES: tuple[str, ...] = ("staged", "reported")

PREDICATES: dict[str, str] = {
    "survey": "plugins.mission_pack_lunch.planner:survey",
    "sort_temp": "plugins.mission_pack_lunch.planner:sort_temp",
    "report": "plugins.mission_pack_lunch.planner:report",
    **{f"v_at_{it}": f"plugins.mission_pack_lunch.planner:v_at_{it}"
       for it in ITEMS},
    **{f"v_grasped_{it}": f"plugins.mission_pack_lunch.planner:v_grasped_{it}"
       for it in ITEMS},
    **{f"v_packed_{it}": f"plugins.mission_pack_lunch.planner:v_packed_{it}"
       for it in ITEMS},
}

EPISODE: dict[str, Any] = {
    "task": "pack_lunch",
    "percept_noise": 0.012,
    "percept_provider": "plugins.embodiment_robocasa.percept:provider",
    "horizon": 8000,
}

SEGMENT_SPECS: dict[str, dict[str, Any]] = segment_specs(SKILL_RECORDS, "robocasa")


def _emit_plan() -> Mapping:
    nodes: list[dict] = [
        {"id": "survey", "skill": "survey", "kind": "perceive", "args": {},
         "after": []},
        {"id": "sort-temp", "skill": "sort_temp", "kind": "decide", "args": {},
         "after": ["survey"]},
    ]
    verify_list: list[dict] = []
    prev = "sort-temp"
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
    nodes.append({"id": "report", "skill": "report", "kind": "decide",
                  "args": {}, "after": [prev]})
    verify_list.append({"after": "report", "predicate": "reported"})
    return json.loads(json.dumps({
        "goal": "in ONE persistent kitchen episode, pack the two hot stove-area "
                "items into one tupperware and the two cold fridge items into "
                "the other on the dining counter -- survey, decide the "
                "temperature->container assignment, then per item drive "
                "nav/grasp/carry/pack segments with live-state verifies, "
                "retrying a failed sub-goal in the SAME world, closing with a "
                "machine report",
        "nodes": nodes,
        "verify": verify_list,
    }, sort_keys=True))


#: Failed segment's stage word (its node id up to the first "-") -> the recovery
#: primitive this card's embodiment declares ([recoveries.*], embodiment_robocasa)
#: that a no_progress replan inserts before it: a stalled base leg redocks, a
#: grasp re-seats, a placement re-approaches. Keyed by node id (not the fault's
#: failure_mode -- a done recover-<id> re-inserts byte-identically from the
#: fault's recoveries_done, protocol.recover_plan); a stage with no entry is
#: left as-is (nothing to work with).
_RECOVERY: dict[str, str] = {"nav": "redock_retry", "carry": "redock_retry",
                            "grasp": "regrasp_kitchen", "pack": "reapproach"}
#: failure_mode -> stage word -> the repair that overrides _RECOVERY: an arm that
#: stalled reaching a pack point is at its envelope edge, so the base moves
#: (base_nudge) instead of re-approaching from where it stood.
_RECOVERY_BY_MODE: dict[str, dict[str, str]] = {"reach_stall": {"pack": "base_nudge"}}


class PackLunchPlanner:
    """Deterministic emitter of the fixed 31-node per-item mission graph."""

    def plan(self, brief: Mapping) -> Mapping:
        task = brief.get("task")
        if task != "pack_lunch":
            raise ValueError(
                f"PackLunchPlanner only plans 'pack_lunch', got {task!r}")
        # Stateless emitter: done recover-<id> nodes re-insert byte-identically;
        # a no_progress refusal gets the card's repair for that stage before it.
        return protocol.recover_plan(_emit_plan(), brief.get("fault"), _RECOVERY, _RECOVERY_BY_MODE)

    @property
    def identity(self) -> str:
        return "pack_lunch_planner@v2"  # v2: answers no_progress with a recovery node


def provider(**params: Any) -> PackLunchPlanner:
    return PackLunchPlanner(**params)


# ── predicate layer ───────────────────────────────────────────────────────────

def _episode(ctx):
    ep = getattr(ctx, "episode", None)
    if ep is None:
        raise ValueError(
            "pack_lunch predicate reached with no persistent episode; the "
            "binding must declare episodic=true (workload threads ctx.episode)")
    return ep


def _obj_z(env, name: str) -> float:
    return float(np.asarray(env.sim.data.body_xpos[env.obj_body_id[name]])[2])


def _plain(v):
    if v is None or isinstance(v, (bool, str)):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


def survey():
    """PERCEIVE: every item's + tupperware's LIVE pose and the scene
    fingerprint, sealed as facts (SECURE_DZ verifies and sort-temp read back)."""
    def pred(node: Mapping, ctx) -> dict:
        ep = _episode(ctx)
        env, obs = ep.env, ep.obs
        objects = {o: [float(v) for v in np.asarray(obs[f"{o}_pos"])[:3]]
                   for o in (*ITEMS, "tupperware0", "tupperware1")}
        try:
            meta = env.get_ep_meta()
        except Exception:  # noqa: BLE001 -- a scene with no meta still surveys
            meta = {}
        scene = {"lang": (str(meta["lang"]) if meta.get("lang") is not None else None),
                 "layout_id": _plain(getattr(env, "layout_id", meta.get("layout_id"))),
                 "style_id": _plain(getattr(env, "style_id", meta.get("style_id")))}
        opstream.emit("scene_meta", **{k: v for k, v in scene.items() if v is not None})
        ok = all(np.isfinite(v) and abs(v) < 100
                 for pos in objects.values() for v in pos)
        return {"success": bool(ok),
                "facts": {"scene": scene, "objects": objects},
                "privilege": ["privileged.object_z"]}
    return pred


def sort_temp():
    """DECIDE: the temperature -> container assignment, a pure function of the
    item names' own temperature attribute (hot* -> tupperware0, cold* ->
    tupperware1). The driver's stage table is authored from the SAME function,
    so decision and actuation cannot disagree; sealed for the report."""
    def pred(node: Mapping, ctx) -> dict:
        sv = ctx.nodes_out.get("survey") or {}
        facts = sv.get("facts") or {}
        ready = bool(sv.get("success")
                     and set(ITEMS) <= set(facts.get("objects") or {}))
        assignment = {it: ("tupperware0" if it.startswith("hot")
                           else "tupperware1") for it in ITEMS}
        assert assignment == TARGET  # the one deterministic function, two views
        return {"success": ready,
                "decision": {"assignment": assignment,
                             "scene": facts.get("scene")}}
    return pred


def _mk_at(name: str):
    def factory():
        prim = load_provider(f"{_P}:base_near_obj", {"name": name, "th": 1.5})

        def pred(node: Mapping, ctx) -> dict:
            return {"success": bool(prim(_episode(ctx).env))}
        return pred
    return factory


def _mk_grasped(name: str):
    """SECURE_DZ-shaped grasp verify (see mission_recycle_cans: never the bare
    check_obj_grasped latch; the segment's own sealed SECURE_DZ success is the
    alternative z-evidence after a disturbed retry)."""
    def factory():
        grasped = load_provider(f"{_P}:obj_grasped_any", {"name": name})

        def pred(node: Mapping, ctx) -> dict:
            ep = _episode(ctx)
            facts = (ctx.nodes_out.get("survey") or {}).get("facts") or {}
            pos0 = (facts.get("objects") or {}).get(name)
            if not pos0:
                return {"success": False}
            risen = _obj_z(ep.env, name) > float(pos0[2]) + SECURE_DZ
            seg_secure = bool((ctx.nodes_out.get(f"grasp-{name}") or {})
                              .get("success"))
            return {"success": bool(grasped(ep.env) and (risen or seg_secure))}
        return pred
    return factory


def _mk_packed(name: str):
    def factory():
        inside = load_provider(f"{_P}:obj_in_receptacle",
                               {"name": name, "receptacle": TARGET[name]})
        released = load_provider(f"{_P}:gripper_far", {"name": name})

        def pred(node: Mapping, ctx) -> dict:
            env = _episode(ctx).env
            return {"success": bool(inside(env) and released(env))}
        return pred
    return factory


for _it in ITEMS:
    globals()[f"v_at_{_it}"] = _mk_at(_it)
    globals()[f"v_grasped_{_it}"] = _mk_grasped(_it)
    globals()[f"v_packed_{_it}"] = _mk_packed(_it)


def report():
    """DECIDE: cross-check sealed segments against the live oracle; headline
    ``packed`` is PackFoodByTemp's own _check_success (either consistent
    assignment, all four packed, gripper released)."""
    def factory_preds():
        return {it: load_provider(f"{_P}:obj_in_receptacle",
                                  {"name": it, "receptacle": TARGET[it]})
                for it in ITEMS}
    inside = factory_preds()

    def pred(node: Mapping, ctx) -> dict:
        ep = _episode(ctx)
        out = ctx.nodes_out
        live: dict[str, Any] = {}
        ok = True
        for it in ITEMS:
            try:
                live[f"{it}_in_{TARGET[it]}"] = bool(inside[it](ep.env))
            except Exception:  # noqa: BLE001 -- an unreadable check breaks the report
                live[f"{it}_in_{TARGET[it]}"] = None
                ok = False
        try:
            packed = bool(ep.env._check_success())
        except Exception:  # noqa: BLE001
            packed, ok = None, False
        segments = {sid: bool((out.get(sid) or {}).get("success"))
                    for sid in _SEG_IDS}
        return {"success": bool(ok),
                "decision": {"live": live, "segments": segments,
                             "packed": packed}}
    return pred


if __name__ == "__main__":
    plan = PackLunchPlanner().plan({"task": "pack_lunch"})
    assert set(plan) == {"goal", "nodes", "verify"} and plan["goal"]
    ids = [n["id"] for n in plan["nodes"]]
    assert len(ids) == 31 and len(set(ids)) == 31, len(ids)
    kinds = [n.get("kind", "manipulate") for n in plan["nodes"]]
    assert kinds.count("segment") == 16 and kinds.count("verify") == 12
    assert kinds.count("perceive") == 1 and kinds.count("decide") == 2
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
        PackLunchPlanner().plan({"task": "pack_lunch"}), sort_keys=True)
    for ref in PREDICATES.values():
        assert callable(load_provider(ref))

    # the decide is the same pure function TARGET was authored from
    from dataclasses import dataclass

    @dataclass
    class _Ctx:
        episode: Any
        nodes_out: dict
    facts = {"objects": {o: [0.0, 0.0, 1.0]
                         for o in (*ITEMS, "tupperware0", "tupperware1")}}
    dec = sort_temp()({}, _Ctx(None, {"survey": {"success": True,
                                                 "facts": facts}}))
    assert dec["success"] and dec["decision"]["assignment"] == TARGET
    assert sort_temp()({}, _Ctx(None, {}))["success"] is False
    try:
        PackLunchPlanner().plan({"task": "stack"})
    except ValueError:
        pass
    else:
        raise AssertionError("wrong task must fail loudly")
    print("plugins/mission_pack_lunch/planner.py self-check OK")
