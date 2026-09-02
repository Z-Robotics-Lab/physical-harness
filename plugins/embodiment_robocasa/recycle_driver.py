"""Composite ``policy.driver`` for the recycle_cans persistent mission
(RecycleSodaCans): four scattered soda cans, each fetched by the same
nav -> grasp -> carry -> drop stage quartet and clustered on the counter beside
the stove. The stage tables ride ``drivers.py`` / ``stage_extras.py`` primitives
untouched; only the cluster-drop targeting is new here.

Can homes are the task's OWN object placements (RecycleSodaCans._get_obj_cfgs):
can1 on the microwave counter, can2 on the cabinet counter, can3/can4 on the
dining counter by the stool -- static facts of the env class, so the nav table
is authored, not sensed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plugins.embodiment_robocasa import drivers as D
from plugins.embodiment_robocasa import stage_extras as X

#: can -> the registered fixture it spawns on (RecycleSodaCans._get_obj_cfgs).
CAN_HOME: dict[str, str] = {
    "can1": "counter_microwave",
    "can2": "counter_cab",
    "can3": "dining_counter",
    "can4": "dining_counter",
}
CANS: tuple[str, ...] = tuple(CAN_HOME)


class ClusterDropDriver(X.PointPlaceDriver):
    """Drop the held can on the counter BESIDE the stove, clustered: the drop
    point sits just past the stove's bbox toward the stove-side counter
    (RecycleSodaCans wants every can within 0.25 of the stove AND on a counter),
    with a small per-can lateral spread so four cans neither stack nor scatter
    (neighbour gap ~0.07 m << the 0.25 cluster threshold). The three geometry
    knobs are the card's [tunables] drop_edge_margin / drop_spread / drop_dz."""

    def __init__(self, obj_name: str, slot: int):
        super().__init__(obj_name)
        t = D.tunables()
        self.slot = slot           # 0..3, deterministic per can
        self._point = None
        self._edge_margin = t["drop_edge_margin"]   # past the stove bbox edge, toward the counter
        self._spread = t["drop_spread"]             # per-can lateral pitch along the stove edge
        self._drop_dz = t["drop_dz"]                # release height above the stove-top plane

    def _drop_point(self, env) -> np.ndarray:
        if self._point is None:
            stove = D._fixture(env, "stove")
            counter = D._fixture(env, "counter_stove")
            sites = np.vstack(stove.get_ext_sites(relative=False))
            center = sites.mean(0)
            top_z = float(sites[:, 2].max())
            # toward the stove-side counter, horizontal, unit
            u = np.asarray(counter.pos, float)[:2] - center[:2]
            n = float(np.linalg.norm(u))
            u = u / n if n > 1e-6 else np.array([1.0, 0.0])
            # stove half-extent along u (bbox corners projected on u)
            half = float(np.max(np.abs((sites[:, :2] - center[:2]) @ u)))
            v = np.array([-u[1], u[0]])          # along the stove edge
            xy = (center[:2] + u * (half + self._edge_margin)
                  + v * self._spread * (self.slot - 1.5))
            self._point = np.array([xy[0], xy[1], top_z + self._drop_dz])
        return self._point

    def done(self, env) -> bool:
        import robocasa.utils.object_utils as OU

        return bool(
            OU.obj_fixture_bbox_min_dist(env, self.obj_name, D._fixture(env, "stove"))
            <= 0.25
            and OU.check_obj_any_counter_contact(env, self.obj_name)
            and OU.gripper_obj_far(env, obj_name=self.obj_name))


def _stages() -> dict[str, tuple[Any, int]]:
    """spec.task -> (stage factory, step cap). Caps mirror kitchen_driver's
    per-stage smoke budgets (nav 250 / grasp 600 / loaded carry 450 / place 300)."""
    table: dict[str, tuple[Any, int]] = {}
    for can in CANS:
        home = CAN_HOME[can]
        table[f"nav_{can}"] = (
            lambda h=home, c=can: X.NavToObjectDriver(h, c), 250)
        table[f"grasp_{can}"] = (lambda c=can: D.GraspDriver(c), 600)
        # 700: a cross-kitchen loaded leg measured 553 capped steps (stow 80 +
        # VCAP arc over 3.5 m, seed 4243) -- 450 starved it just short.
        table[f"carry_{can}"] = (
            lambda: D.NavigateDriver("stove", carry=True), 700)
        table[f"drop_{can}"] = (
            lambda c=can, s=CANS.index(can): ClusterDropDriver(c, s), 300)
    return table


_STAGES = _stages()


def provider(**params: Any) -> X.CompositePolicies:
    return X.CompositePolicies(_STAGES, "robocasa_recycle_cans@v1", **params)


if __name__ == "__main__":
    # Base-importable self-check: the table covers every stage the mission's
    # SEGMENT_SPECS can re-task to, and dispatch arms the right stage class.
    assert set(_STAGES) == {f"{p}_{c}" for c in CANS
                            for p in ("nav", "grasp", "carry", "drop")}
    drv = provider().make_driver(object())

    class _S:
        task = "nav_can3"
    drv.enter_segment(object(), _S())
    assert isinstance(drv._stage, X.NavToObjectDriver)
    assert drv._stage.fixture_name == "dining_counter" and drv._stage.obj_name == "can3"

    class _S2:
        task = "drop_can2"
    drv.enter_segment(object(), _S2())
    assert isinstance(drv._stage, ClusterDropDriver) and drv._stage.slot == 1
    print("plugins/embodiment_robocasa/recycle_driver.py self-check OK")
