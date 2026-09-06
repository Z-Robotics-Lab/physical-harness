"""Bounded cross-task diagnosis/intervention memory and censored transfer reports.

Records carry evaluator evidence references, not executable patches. Retrieval is
a hypothesis for a fresh paired experiment, never installation authority.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path


def intervention_strategy(tried: dict, *, summary: str = "", reference: str = "") -> dict:
    """Describe a measured intervention without copying its task's absolute settings.

    This is a description, not an acceptance or recording gate. For a plan, the
    caller may supply ``detail.reference_graph`` from the server's previous plan;
    only inserted call types/counts survive, never graph arguments or source.
    """
    if not isinstance(reference, str):
        raise TypeError("intervention reference must be a string")
    if not isinstance(summary, str):
        raise TypeError("intervention summary must be a string")
    rationale = summary.strip()
    kind = tried.get("kind", "none")
    detail = tried.get("detail") or {}
    layer = detail.get("layer")
    scope = layer if isinstance(layer, str) and layer else {
        "tunables": "parameter", "plan": "plan",
    }.get(kind, "controller")
    if kind == "tunables":
        path = detail.get("path")
        path = ".".join(path) if isinstance(path, list) and all(
            isinstance(part, str) for part in path) else "<undeclared path>"
        before, after = detail.get("from"), detail.get("to")
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      and math.isfinite(v) for v in (before, after))
        if numeric and rationale:
            # Keep the proposed mechanism, without exporting this task's settings.
            rationale = re.sub(
                r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])",
                lambda match: "<task-specific value>" if any(
                    math.isclose(float(match[0]), v, rel_tol=1e-9, abs_tol=1e-12)
                    for v in (before, after)) else match[0], rationale)
        change = "relative change unavailable"
        if numeric:
            if before == after:
                change = "unchanged"
            elif before == 0:
                change = "increased from zero" if after > 0 else "decreased from zero"
            else:
                relative = (after - before) / abs(before) * 100
                if math.isfinite(relative):
                    change = f"{relative:+.6g}% relative to its previous magnitude"
        summary = (f"Adjust declared parameter {path}: {change}. "
                   "Resolve the current task's declaration and scale before a fresh paired trial.")
    elif kind == "executor":
        summary = ("Switch the same action to another installed executor. "
                   "Resolve an available executor in the current task and compare paired outcomes.")
    elif kind == "plan":
        graph, previous = detail.get("graph"), detail.get("reference_graph")
        if isinstance(graph, dict) and isinstance(previous, dict):
            old = {node.get("id") for node in previous.get("nodes", [])}
            inserted = [node for node in graph.get("nodes", []) if node.get("id") not in old]
            types = Counter((node.get("kind", "manipulate"), node.get("skill", "unspecified"))
                            for node in inserted)
            labels = ", ".join(f"{kind}/{skill} x{count}"
                               for (kind, skill), count in sorted(types.items())) or "none"
            summary = (f"Insert {len(inserted)} installed action calls ({labels}). "
                       "Map these call types to the current task; retain its original objectives.")
        else:
            summary = ("Change the plan with installed action insertions; inserted types/counts "
                       "are unknown because the previous graph was not supplied.")
    else:
        summary = rationale or (
            f"Change the {scope} implementation through a candidate executor; "
            "derive a task-specific implementation and compare paired outcomes."
            if kind in ("card", "patch") else "No executable intervention was specified.")
    if kind in ("tunables", "executor", "plan") and rationale:
        summary += f" Reported rationale: {rationale}"
    return {"kind": kind, "scope": scope, "summary": summary, "reference": reference}


def read_experiences(path: str | Path) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text())
        return data.get("records", []) if isinstance(data, dict) else []
    except (OSError, ValueError):
        return []


def record_experience(path: str | Path, *, task: str, diagnosis: dict,
                      intervention: dict, accepted: bool, evidence: dict,
                      max_records: int = 512) -> dict | None:
    """Append one trusted evaluator result; return None for uninformative evidence.

    ``evidence`` requires distinct ``before_sha``/``after_sha`` and a ``round``.
    Caller supplies measured paired outcomes; model notes alone cannot enter.
    ``intervention`` retains only kind/scope/summary/reference, never source code.
    Optional protocol/policy identities are preserved for compatible retrieval.
    Returned ``sequence`` lets a transfer evaluation freeze its training prefix.
    """
    fingerprint = sorted(set(diagnosis.get("fingerprint") or []))
    if not fingerprint or not task or not isinstance(accepted, bool):
        return None
    if not evidence.get("before_sha") or not evidence.get("after_sha") \
            or evidence["before_sha"] == evidence["after_sha"] or "round" not in evidence:
        return None
    strategy = {k: intervention[k][:500] for k in ("kind", "scope", "summary", "reference")
                if isinstance(intervention.get(k), str)}
    if not strategy.get("kind") or strategy["kind"] == "none" or max_records < 1:
        return None
    record = {"task": task, "fingerprint": fingerprint, "intervention": strategy,
              "accepted": accepted, "evidence": {k: evidence[k] for k in (
                  "before_sha", "after_sha", "round", "session", "suite_scope",
                  "protocol_id", "evidence_policy") if k in evidence}}
    record["id"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = read_experiences(path)
        if old := next((r for r in rows if r.get("id") == record["id"]), None):
            return old
        record["sequence"] = max((r.get("sequence", 0) for r in rows), default=0) + 1
        rows.append(record)
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".experience-")
        try:
            with os.fdopen(fd, "w") as out:
                json.dump({"schema": "rsi-experience-v1", "records": rows[-max_records:]}, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    return record


def retrieve_experiences(path: str | Path, *, task: str, diagnosis: dict,
                         limit: int = 5, before_sequence: int | None = None,
                         protocol_id: str | None = None,
                         evidence_policy: str | None = None) -> list[dict]:
    """Retrieve matching structural diagnoses from OTHER tasks in a frozen prefix.

    Both positive and negative measured outcomes are shown. No task-name or node
    match contributes to the score; no source code is copied into a new task.
    Supplied protocol/policy filters require exact recorded identities; legacy
    rows without them remain readable but cannot satisfy those filters.
    """
    query = set(diagnosis.get("fingerprint") or [])
    if not query or limit <= 0:
        return []
    scored = []
    for row in read_experiences(path):
        if row.get("task") == task or (before_sequence is not None
                                      and row.get("sequence", 0) >= before_sequence):
            continue
        evidence = row.get("evidence") or {}
        if any(expected is not None and evidence.get(key) != expected
               for key, expected in (("protocol_id", protocol_id),
                                     ("evidence_policy", evidence_policy))):
            continue
        other = set(row.get("fingerprint") or [])
        overlap = query & other
        if overlap:
            scored.append({**row, "similarity": len(overlap) / len(query | other)})
    return sorted(scored, key=lambda r: (-r["similarity"], -int(r["accepted"]),
                                        -r["sequence"]))[:limit]


def development_report(rows: list[dict], *, epoch_start: int) -> dict:
    """Summarize current-epoch improvements and additive, per-round costs.

    Missing costs remain unknown independently for each field. Historical run
    budgets are cumulative counters and are deliberately not read here.
    """
    current = sorted((row for row in rows if isinstance(row, dict)
                      and type(row.get("round")) is int and row["round"] >= epoch_start),
                     key=lambda row: row["round"])

    def accepted(row):
        evaluation = row.get("evaluation")
        acceptance = evaluation.get("acceptance") if isinstance(evaluation, dict) else None
        if isinstance(acceptance, dict) and "accepted" in acceptance:
            return acceptance["accepted"] is True
        return row.get("accepted") is True

    def valid(value):
        return type(value) in (int, float) and value >= 0 and (type(value) is int or math.isfinite(value))

    def total(values):
        if not all(valid(value) for value in values):
            return None
        try:
            result = sum(values)
        except OverflowError:
            return None
        return result if valid(result) else None

    def costs(selected):
        fields = {key: [] for key in ("episode_attempts", "model_calls", "input_bytes", "sim_s", "wall_s", "llm_tokens")}
        for row in selected:
            usage = row.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            for key, observed in fields.items():
                value = usage.get(key)
                if key == "llm_tokens":
                    llm = row.get("llm")
                    incomplete = isinstance(llm, dict) and llm.get("usage_complete") is False
                    value = total([value.get("prompt"), value.get("completion")]) \
                        if isinstance(value, dict) and not incomplete else None
                observed.append(value)
        return {key: total(values) for key, values in fields.items()}

    improvements = [index for index, row in enumerate(current) if accepted(row)]
    first = improvements[0] if improvements else None
    previous = next((index for index in reversed(improvements) if index < len(current) - 1), None)
    return {"first_accepted_round": current[first]["round"] - epoch_start + 1 if first is not None else None,
            "total_trials": len(current), "censored": first is None, "accepted_updates": len(improvements),
            "cost": {"total": costs(current),
                     "first_accepted": costs(current[:first + 1]) if first is not None else None,
                     "since_previous_acceptance": costs(current[previous + 1:] if previous is not None else current)}}


def transfer_report(trials: list[dict]) -> dict:
    """Paired cold/warm first-improvement counts, retaining right-censored tasks.

    Each measured row supplies ``task, condition ('cold'|'warm'), trial (1-based),
    accepted``; optional ``training_tasks`` identifies the frozen memory size.
    The caller must run the same task/seeds/budget for each pair. This function
    reports counts only and does not claim the supplied experiments prove scale.
    """
    grouped = {}
    for row in trials:
        condition = row.get("condition")
        if condition not in ("cold", "warm") or not isinstance(row.get("accepted"), bool):
            raise ValueError("transfer rows require cold/warm and a measured accepted boolean")
        index = row.get("trial")
        if not isinstance(index, int) or isinstance(index, bool) or index < 1:
            raise ValueError("trial must be a positive integer")
        key = (row["task"], row.get("training_tasks", 0), condition)
        cohort = grouped.setdefault(key, {})
        if index in cohort:
            raise ValueError("duplicate trial in transfer cohort")
        cohort[index] = row["accepted"]
    pairs = {}
    for (task, training, condition), cohort in sorted(grouped.items()):
        if sorted(cohort) != list(range(1, max(cohort) + 1)):
            raise ValueError("transfer trial history must be contiguous")
        first = min((i for i, accepted in cohort.items() if accepted), default=None)
        pairs.setdefault((task, training), {"task": task, "training_tasks": training})[condition] = {
            "first_accepted_trial": first, "trials_observed": len(cohort), "censored": first is None}
    rows = list(pairs.values())
    for row in rows:
        cold, warm = row.get("cold", {}), row.get("warm", {})
        a, b = cold.get("first_accepted_trial"), warm.get("first_accepted_trial")
        row["paired"] = bool(cold and warm)
        row["trials_saved"] = a - b if a is not None and b is not None else None
    return {"schema": "rsi-transfer-v1", "pairs": rows,
            "paired_tasks": sum(r["paired"] for r in rows),
            "censored_runs": sum(v["censored"] for r in rows for c in ("cold", "warm")
                                 if (v := r.get(c))),
            "claim": "Descriptive paired counts; scale requires held-out tasks and fixed budgets."}
