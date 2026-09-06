"""The observation feature contract.

This is Governor's central mechanism and the place it diverges from both source
projects. Zetta's critics read simulator-internal quantities such as
``privileged.dishwasher.rack.residual_to_success``; its README forbids letting
privileged state "become hidden task control" but ships no mechanism enforcing
it. Here the separation is structural: every feature declares its class, a
critic declares the features it reads, and the gate can require a privilege
budget of zero.

Classes
-------
OBSERVABLE  A real robot could measure this from proprioception, joint sensors,
            or gripper state. Safe for a skill intended to transfer.
PRIVILEGED  Only a simulator knows it (ground-truth object pose, success
            residuals, contact ground truth). Usable for diagnosis, never
            silently usable for control.
DERIVED     A pure function of other declared features; inherits the highest
            privilege class of its inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Privilege(Enum):
    """How obtainable a feature is outside a simulator."""

    OBSERVABLE = "observable"
    PRIVILEGED = "privileged"

    @property
    def cost(self) -> int:
        """Budget units consumed by reading one feature of this class."""
        return 0 if self is Privilege.OBSERVABLE else 1


@dataclass(frozen=True, slots=True)
class Feature:
    """One named scalar a critic may read at control frequency."""

    name: str
    privilege: Privilege
    extract: Callable[[Mapping[str, np.ndarray]], float]
    doc: str

    def __post_init__(self) -> None:
        if not self.name or " " in self.name:
            raise ValueError(f"feature name must be a non-empty bare token, got {self.name!r}")
        expected = "observable." if self.privilege is Privilege.OBSERVABLE else "privileged."
        if not self.name.startswith(expected):
            raise ValueError(
                f"feature {self.name!r} is declared {self.privilege.value} "
                f"but its name does not start with {expected!r}; the namespace IS the declaration"
            )


def _f(x) -> float:
    return float(np.asarray(x).reshape(-1)[0]) if np.asarray(x).size == 1 else float(np.asarray(x))


REGISTRY: dict[str, Feature] = {}


def register(feature: Feature) -> Feature:
    """Add one feature to the process-wide catalog published to proposers."""
    if feature.name in REGISTRY:
        raise ValueError(f"duplicate feature {feature.name!r}")
    REGISTRY[feature.name] = feature
    return feature


def privilege_cost(names) -> int:
    """Budget consumed by a critic reading `names`.

    Raises for an unknown feature: a proposer may not invent features, which is
    how a candidate is prevented from silently reaching into raw observations.
    """
    total = 0
    for n in names:
        if n not in REGISTRY:
            raise KeyError(f"unknown feature {n!r}; declared catalog is {sorted(REGISTRY)}")
        total += REGISTRY[n].privilege.cost
    return total


def extract(obs: Mapping[str, np.ndarray], names) -> dict[str, float]:
    """Project a raw environment observation onto the declared feature vector."""
    return {n: REGISTRY[n].extract(obs) for n in names}


# Populate REGISTRY with the base catalog the moment the machinery is imported.
# The import sits at the BOTTOM on purpose: feature_catalog imports register/
# Feature/Privilege back from this module, which are all defined above, so the
# cycle resolves cleanly. This is what makes the round-69 empty-registry state
# unreachable -- you cannot import the feature machinery without the catalog.
import harness.feature_catalog  # noqa: F401  base catalog population (see module docstring)
