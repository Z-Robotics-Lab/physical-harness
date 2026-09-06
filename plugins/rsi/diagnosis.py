"""Action/response observations from embodiment-declared trace coordinates.

This is diagnostic evidence, never a reward or a proof of reachability. A response
span describes only the motions actually observed; missing excitation does not
establish what would have happened under another controller.
"""

from __future__ import annotations

import math

import numpy as np


def _vector(row: dict, field: str, axes: list[int]):
    try:
        value = np.asarray([row[field][i] for i in axes], dtype=float)
        return value if np.isfinite(value).all() else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _window(series: list, spec: dict) -> list:
    """Last contiguous phase with a fixed diagnostic target and finite state."""
    out = []
    for row in reversed(series):
        state = _vector(row, spec["state"], spec["axes"])
        target = _vector(row, spec.get("target", "target"), spec["axes"])
        if state is None or target is None:
            break
        if out and (row.get("phase") != out[0][0].get("phase")
                    or np.linalg.norm(target - out[0][2]) > spec["noise"]):
            break
        out.append((row, state, target))
    return list(reversed(out))


def _group(series: list, spec: dict) -> dict:
    win = _window(series, spec)
    unknown = {"kind": "unknown", "evidence": {"samples": len(win)}}
    if len(win) < 3:
        return unknown
    rows, states, targets = zip(*win)
    commands = []
    for row in rows[:-1]:
        cmd = row.get("cmd")
        if not isinstance(cmd, dict) or not isinstance(cmd.get("nonzero"), list):
            return unknown
        if "mode" in spec and "mode" not in cmd:
            return unknown
        active_mode = "mode" not in spec or cmd.get("mode") == spec["mode"]
        commands.append(active_mode and bool(set(spec["commands"]) & set(cmd["nonzero"])))
    states, targets = np.asarray(states), np.asarray(targets)
    residual = targets[-1] - states[-1]
    noise = float(spec["noise"])
    distances = np.linalg.norm(targets - states, axis=1)
    movement = float(np.max(np.linalg.norm(states - states[0], axis=1)))
    progress = float(distances[0] - distances[-1])
    delta = np.diff(states, axis=0)
    observed = delta[np.linalg.norm(delta, axis=1) > noise]
    rank, outside = 0, float(np.linalg.norm(residual))
    if len(observed):
        _, singular, vectors = np.linalg.svd(observed, full_matrices=False)
        rank = int(np.sum(singular > noise * math.sqrt(len(observed))))
        basis = vectors[:rank]
        outside = float(np.linalg.norm(residual - (residual @ basis.T) @ basis))
    evidence = {"samples": len(win), "first_step": rows[0].get("step"),
                "last_step": rows[-1].get("step"), "phase": rows[-1].get("phase"),
                "commanded_intervals": sum(commands), "observed_intervals": len(commands),
                "residual": [round(float(x), 6) for x in residual],
                "residual_norm": round(float(distances[-1]), 6),
                "movement": round(movement, 6), "progress": round(progress, 6),
                "response_rank": rank, "residual_outside_response_span": round(outside, 6),
                "noise": noise}
    if distances[-1] <= noise:
        kind = "residual_small"
    elif not any(commands):
        kind = "unexcited_in_samples"
    elif movement <= noise:
        kind = "commanded_without_observed_motion"
    elif progress < -noise:
        kind = "moved_away"
    elif outside > noise:
        kind = "residual_outside_observed_span"
    elif progress > noise:
        kind = "progressing"
    else:
        kind = "no_net_progress"
    return {"kind": kind, "evidence": evidence}


def analyze_trace(trace: dict | None, groups: dict | None = None) -> dict:
    """Return bounded findings using a trace's declared groups (or explicit adapter).

    Group schema: ``{state, axes, target?, commands, mode?, noise}``. Coordinates
    and noise are supplied by the embodiment, never inferred from task names.
    Legacy traces without this schema remain unknown. Signed native commands may
    be retained in ``cmd.values``; their coordinate frame is not guessed here.
    """
    trace = trace if isinstance(trace, dict) else {}
    series = trace.get("series") or []
    groups = groups if groups is not None else trace.get("groups", {})
    findings = []
    for channel, spec in sorted(groups.items()):
        try:
            if not spec["axes"] or not spec["commands"] or float(spec["noise"]) <= 0:
                raise ValueError("invalid coordinates")
            finding = _group(series, spec)
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            finding = {"kind": "unknown", "evidence": {"reason": "invalid group declaration"}}
        findings.append({"channel": channel, **finding})
    fingerprint = sorted(f"{f['channel']}:{f['kind']}" for f in findings
                         if f["kind"] not in ("unknown", "residual_small", "progressing"))
    return {"schema": "action-response-v1", "status": "observed" if any(
                f["kind"] != "unknown" for f in findings) else "unknown",
            "findings": findings, "fingerprint": fingerprint,
            "coverage": trace.get("sampling", {"kind": "unspecified"}),
            "limits": "Sampled commands and observed motion are diagnostic only; "
                      "they do not establish causality, controllability, or task success."}
