"""A fixed ruler for development trials, independent of the candidate graph.

Only card-bound verification predicates and the embodiment's task terminal
oracle supply labels. Controller ``done()``, diagnostics, graph size and
declared ``ensures`` do not. The latter describe an obligation; they are not
observations that it held. Boolean checkpoints deliberately remain boolean:
the existing predicate API does not expose trustworthy continuous residuals.

The caller compiles once from the server's original plan and bindings, before
mounting a candidate, and persists the resulting contract with the campaign.
This module consumes execution evidence; execution never calls RSI back.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

from harness.config import sha_json
from harness.protocol import parse_pred_ref, pred_ref_str

VERSION = "fixed-verification-v2"
EVIDENCE_POLICY = 'world-dependencies-v1'


def _semantic(node: Mapping) -> dict:
    return {"kind": node.get("kind", "manipulate"), "skill": node["skill"],
            "args": copy.deepcopy(dict(node.get("args") or {}))}


def _ensures(record, args: Mapping) -> list[str]:
    refs = record.get("ensures", ()) if isinstance(record, Mapping) else getattr(record, "ensures", ())
    out = []
    for ref in refs:
        name, values = parse_pred_ref(ref)
        out.append(pred_ref_str((name, *(args.get(v, v) for v in values))))
    return sorted(set(out))


def compile_contract(plan: Mapping, records: Mapping | None = None, *, task: str = "",
                     predicates: Mapping | None = None, terminal_ref: str | None = None,
                     identity: Mapping | None = None) -> dict:
    """Freeze unique, grounded verification obligations from a server-owned plan.

    Renaming, repeating or inserting candidate nodes cannot change this ruler.
    Actuating nodes are excluded: even an installed controller's stopping
    threshold may be a tunable. Perception/decision success means readable or
    well-formed in existing cards, not that the physical task succeeded.
    ``terminal_ref`` names the server-selected embodiment provider; its
    ``terminal_success`` is measured independently before the world closes.
    """
    predicates, records = predicates or {}, records or {}
    obligations: dict[str, dict] = {}
    for node in plan.get("nodes") or ():
        if node.get("kind") != "verify":
            continue
        source = predicates.get(node["skill"])
        if not isinstance(source, str) or not source:
            continue
        semantic = _semantic(node)
        key = sha_json({"authority": "predicate", "source": source, **semantic})
        obligations.setdefault(key, {"id": key, "authority": "predicate", "source": source,
                                    "semantic": semantic,
                                    "ensures": _ensures(records.get(node["skill"]), semantic["args"]),
                                    "nodes": []})["nodes"].append(str(node["id"]))
    if terminal_ref:
        semantic = {"task": task}
        key = sha_json({"authority": "embodiment.terminal_success", "source": terminal_ref,
                       **semantic})
        obligations[key] = {"id": key, "authority": "embodiment.terminal_success",
                            "source": terminal_ref, "semantic": semantic, "ensures": [], "nodes": []}
    rows = sorted(obligations.values(), key=lambda row: row["id"])
    # Node ids are display aliases only. They neither identify the ruler nor
    # give duplicate occurrences additional weight.
    identity = {"version": VERSION, 'evidence_policy': EVIDENCE_POLICY,
                "task": task, "context": copy.deepcopy(dict(identity or {})),
                "obligations": [{k: v for k, v in row.items() if k != "nodes"} for row in rows]}
    return {**identity, "obligations": rows, "sha": sha_json(identity),
            "available": bool(rows),
            "limitation": "Boolean verifier checkpoints; no continuous residual or causal diagnosis is inferred."}


def _check_contract(contract: Mapping) -> None:
    identity = {"version": contract.get("version"), 'evidence_policy': contract.get('evidence_policy'),
                "task": contract.get("task"),
                "context": contract.get("context", {}),
                "obligations": [{k: v for k, v in row.items() if k != "nodes"}
                                for row in contract.get("obligations", ())]}
    if (contract.get("version") != VERSION or contract.get('evidence_policy') != EVIDENCE_POLICY
            or contract.get("sha") != sha_json(identity)):
        raise ValueError("evaluation contract identity does not match its contents")
    ids = [row["id"] for row in contract["obligations"]]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation contract contains duplicate obligations")


class EvaluationMonitor:
    """Accumulate only observations matching the frozen authority and semantics.

    A repeated check replaces its prior reading; it never contributes another
    reward. An unobserved condition is None, not success. This is checkpoint
    evidence, not a claim that a transient grasp still holds at task end.
    """

    def __init__(self, contract: Mapping):
        _check_contract(contract)
        self.contract = copy.deepcopy(dict(contract))
        self.vector = {row["id"]: None for row in contract["obligations"]}

    def observe(self, node: Mapping, result: Mapping) -> None:
        semantic = _semantic(node)
        for row in self.contract["obligations"]:
            if row["authority"] != "predicate" or row["semantic"] != semantic:
                continue
            if result.get("authority") != row["authority"] or result.get("source") != row["source"]:
                continue
            clean = (result.get('evidence_policy') == self.contract['evidence_policy']
                     and result.get('blocked_reads') == [])
            value = result.get("success") if clean else None
            self.vector[row["id"]] = value if type(value) is bool else None

    def observe_terminal(self, result: Mapping | None) -> None:
        result = result or {}
        for row in self.contract["obligations"]:
            if row["authority"] == "embodiment.terminal_success" \
                    and result.get("authority") == row["authority"] \
                    and result.get("source") == row["source"]:
                value = result.get("success")
                self.vector[row["id"]] = value if type(value) is bool else None

    def snapshot(self) -> dict:
        terminal = [self.vector[row["id"]] for row in self.contract["obligations"]
                    if row["authority"] == "embodiment.terminal_success"]
        return {"contract_sha": self.contract["sha"], "vector": dict(self.vector),
                "complete": bool(terminal) and all(value is True for value in terminal),
                "terminal": terminal[0] if len(terminal) == 1 else None,
                "observed": sum(value is not None for value in self.vector.values()),
                "passed": sum(value is True for value in self.vector.values()),
                "total": len(self.vector),
                "available": any(value is not None for value in self.vector.values())}


def evaluate(contract: Mapping, observations, terminal_observation: Mapping | None = None) -> dict:
    monitor = EvaluationMonitor(contract)
    for row in observations or ():
        monitor.observe(row["node"], row)
    monitor.observe_terminal(terminal_observation)
    return monitor.snapshot()


def _reading(seed: Mapping, contract: Mapping) -> dict:
    reading = seed.get("evaluation") or {}
    ids = {row["id"] for row in contract["obligations"]}
    if reading.get("contract_sha") != contract["sha"] or set(reading.get("vector") or {}) != ids:
        raise ValueError("trial lacks observations under this evaluation contract")
    if any(value is not None and type(value) is not bool for value in reading["vector"].values()):
        raise ValueError("evaluation vector contains a non-boolean observation")
    return reading["vector"]


def summary(suite: Mapping, contract: Mapping) -> dict:
    _check_contract(contract)
    vectors = [_reading(seed, contract) for seed in (suite.get("seeds") or {}).values()]
    terminal = [row["id"] for row in contract["obligations"]
                if row["authority"] == "embodiment.terminal_success"]
    passed = sum(x is True for v in vectors for x in v.values())
    denominator = len(vectors) * len(contract["obligations"])
    return {"successes": sum(bool(terminal) and all(v[k] is True for k in terminal) for v in vectors),
            "episodes": len(vectors), "passed": passed,
            "progress": passed / denominator if denominator else 0.0,
            "obligations": len(contract["obligations"]),
            "observed": sum(x is not None for v in vectors for x in v.values()),
            "contract_sha": contract["sha"]}


def compare(before: Mapping, after: Mapping, contract: Mapping) -> dict:
    """Paired development acceptance: any strict gain and no lost true condition.

    This is not a statistical installation gate. Missing seeds or mismatched
    rulers refuse comparison; caller-selected subsets must subset both sides
    explicitly and cannot be presented as whole-suite evidence.
    """
    _check_contract(contract)
    result = {"accepted": False, "comparable": False, "reason": "", "before": None, "after": None,
              "gains": [], "regressions": [], "contract_sha": contract["sha"]}
    if not contract["obligations"]:
        return {**result, "reason": "evaluation unavailable: no independent verifier"}
    a, b = before.get("seeds") or {}, after.get("seeds") or {}
    if not a or set(a) != set(b):
        return {**result, "reason": "paired evaluation requires the same nonempty seed set"}
    try:
        result.update(before=summary(before, contract), after=summary(after, contract))
        if not result["before"]["observed"] and not result["after"]["observed"]:
            return {**result, "reason": "evaluation unavailable: no independent observation"}
        for seed in sorted(a, key=str):
            av, bv = _reading(a[seed], contract), _reading(b[seed], contract)
            for key in av:
                change = {"seed": seed, "obligation": key, "before": av[key], "after": bv[key]}
                if av[key] is True and bv[key] is not True:
                    result["regressions"].append(change)
                elif av[key] is not True and bv[key] is True:
                    result["gains"].append(change)
    except ValueError as exc:
        return {**result, "reason": str(exc)}
    accepted = bool(result["gains"]) and not result["regressions"]
    return {**result, "accepted": accepted, "comparable": True,
            "reason": ("fixed verification vector improved without regressions" if accepted else
                       "fixed verification condition regressed" if result["regressions"] else
                       "fixed verification vector did not improve")}
