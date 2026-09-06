"""Intervention permissions derived from the installed task and its execution graph.

The task's goal, predicate calls and existing skill arguments are immutable.
An offline candidate may insert installed skills between the original calls; the ordinary
task validator still checks typing, dependency order and verification coverage.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping


def plan_space(plan: dict, brief: dict) -> dict:
    """Return the actual graph and installed vocabulary available to a proposer."""
    return {
        "graph": copy.deepcopy(plan),
        "skills": {name: {key: typ.__name__ for key, typ in args.items()}
                   for name, args in brief["catalogue"].items()},
        "oracles": list(brief["oracles"]),
        "rules": [
            "Keep goal, tasks, existing skill calls and verification entries unchanged.",
            "Keep existing calls in their original order; insert installed segment skills.",
            "Every inserted action requires an existing oracle and legal dependencies.",
            "Additional nodes earn no reward; only the frozen evaluation objectives count.",
            "The proposed graph is fixed for each episode; in-episode retries remain budgeted.",
        ],
    }


def validate_candidate(plan: dict, reference: dict, brief: dict) -> None:
    """Reject changes to evaluation authority before normal workload validation.

    The validator is resolved by reference to preserve the card import boundary.
    Runtime validation repeats the protocol checks against live world grounding.
    """
    from harness.registry import load_attr

    if not isinstance(plan, Mapping):
        raise ValueError("plan.graph must be an object")
    for key in ("goal", "tasks"):
        if plan.get(key) != reference.get(key):
            raise ValueError(f"plan may not change the server-owned {key}")
    original = {n["id"]: n for n in reference["nodes"]}
    proposed = {n.get("id"): n for n in plan.get("nodes", []) if isinstance(n, Mapping)}
    retained_order = [n.get("id") for n in plan.get("nodes", []) if n.get("id") in original]
    if retained_order != list(original):
        raise ValueError("plan must retain the original checkpoint order")
    for nid, node in original.items():
        other = proposed.get(nid)
        if other is None:
            raise ValueError(f"plan must retain original call {nid!r}")
        for key in ("skill", "args", "kind", "task"):
            if other.get(key) != node.get(key):
                raise ValueError(f"plan may not change {nid!r}.{key}")
    for node in plan.get("nodes", []):
        if node.get("id") not in original and node.get("kind", "manipulate") not in ("segment", "manipulate"):
            raise ValueError("new nodes must be installed action skills, not new evaluation predicates")
    for entry in reference["verify"]:
        if entry not in plan.get("verify", []):
            raise ValueError("plan must retain every original verification entry")
    validate = load_attr("plugins.task.validate:validate_plan")
    ok, why = validate(plan, brief["catalogue"], brief["oracles"],
                       requirements=brief.get("planning_context"))
    if not ok:
        raise ValueError(why)
    if plan == reference:
        raise ValueError("plan does not change the execution graph")


def validate_from_space(plan: dict, space: dict) -> None:
    """Validate against the same JSON vocabulary that the proposer was shown."""
    types = {t.__name__: t for t in (str, int, float, bool, list, dict)}
    brief = {"catalogue": {name: {k: types[v] for k, v in args.items()}
                            for name, args in space["skills"].items()},
             "oracles": space["oracles"]}
    validate_candidate(plan, space["graph"], brief)
