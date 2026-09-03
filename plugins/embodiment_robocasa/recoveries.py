"""This card's recovery repair shapes, declared in its manifest's ``[recoveries.*]``.

The kitchen vocabulary (unclench/raise/reseat/clench, backout/redock) is
DISJOINT from the tabletop card's above/descend/close/lift, which is what keeps
``plugins/rsi/governed.py``'s ``_is_place_recovery`` False here. These phases are
executed by ``RobocasaRecoveryActor`` (plugins/embodiment_robocasa/recovery.py)
against PandaOmron's 12-dim action space, reusing the verified base_mode/arm_mode
primitives in this card's ``drivers.py`` -- never the tabletop RecoveryActor.

The first batch answers the two MEASURED kitchen deaths (calibration r3): a false
grasp enclosure and a loaded-nav stall. A card declaring no ``[recoveries.*]`` has
no recovery primitives at all, and ``plugins/rsi/repertoire.py`` says so verbatim
rather than substituting another card's.
"""

from __future__ import annotations

from dataclasses import dataclass

#: (phase, duration, dx, dy) -- the planar offsets are unused by the kitchen
#: actor (it re-reads a live pose) and stay 0.0 to keep one Step shape repo-wide.
Step = tuple[str, int, float, float]


@dataclass(frozen=True, slots=True)
class Strategy:
    """One named repair shape (satisfies ``harness.contracts.RecoveryStrategy``)."""

    name: str
    steps: tuple[Step, ...]
    rationale: str

    @property
    def length(self) -> int:
        """Upper bound: servo segments may finish early."""
        return sum(d for _n, d, _x, _y in self.steps)

    @property
    def uses_feedback(self) -> bool:
        return any(n.startswith("servo_") for n, _d, _x, _y in self.steps)


REGRASP_KITCHEN = Strategy(
    "regrasp_kitchen",
    (("unclench", 6, 0.0, 0.0), ("raise", 22, 0.0, 0.0),
     ("reseat", 30, 0.0, 0.0), ("clench", 20, 0.0, 0.0)),
    "Kitchen grasp recovery: release, lift clear, re-descend onto a fresh live "
    "meat pose, close in place. GraspDriver.done's SECURE_DZ is the real-"
    "enclosure judge -- a fired trigger (observable.finger_gap closed-on-nothing) "
    "means the grip latched on air, and this re-seats it. Arm mode throughout.",
)

REDOCK_RETRY = Strategy(
    "redock_retry",
    (("backout", 20, 0.0, 0.0), ("redock", 120, 0.0, 0.0)),
    "Kitchen nav/at recovery: back the base straight out, re-drive the fixture "
    "dock (fresh NavigateDriver), then hand back so the stalled segment retries "
    "from a clean approach. For a loaded-transport stall (failure_mode "
    "\"nav_stall\"); a loaded leg keeps its grip. Base mode throughout.",
)


# Reach repairs (overnight-goal item 2): what a planner inserts as a recovery node
# after a segment failed with failure_mode "reach_stall", instead of retrying the
# same graph verbatim. Phases rehover/redescend/nudge are executed by
# RobocasaRecoveryActor against the ACTIVE stage's live target (object pose, or a
# place stage's drop point).

REAPPROACH = Strategy(
    "reapproach",
    (("rehover", 25, 0.0, 0.0), ("redescend", 30, 0.0, 0.0)),
    "Reach recovery: lift to the hover height above the live target, then "
    "descend onto its CURRENT pose -- a target that moved (plowed slab, nudged "
    "can) is re-acquired instead of chased from a stalled pose. Arm mode.",
)

BASE_NUDGE = Strategy(
    "base_nudge",
    (("nudge", 20, 0.0, 0.0), ("rehover", 25, 0.0, 0.0)),
    "Reach recovery: drive the base at most nudge_max (tunable, 0.15 m) toward the target xy (the "
    "arm stalled at the edge of its envelope), then re-hover. Base mode then arm.",
)

RELEASE_RESET = Strategy(
    "release_reset",
    (("unclench", 6, 0.0, 0.0), ("raise", 20, 0.0, 0.0), ("rehover", 25, 0.0, 0.0)),
    "Reach recovery: open the fingers, lift clear, return to hover over the live "
    "target -- clears a wedged/snagged approach before the segment retries. Arm mode.",
)
