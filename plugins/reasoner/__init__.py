"""LLM recovery proposals through the registered model-endpoint transport.

The campaign owns experiments and statistical gates. This adapter owns the
model request, strict proposal parsing, and its auditable model_proposal record.
It never substitutes a searched candidate for a missing or invalid answer.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from governor.proposer import ProposerError, build_brief, parse_proposal
from harness.manifest import mount_params
from harness.registry import load_provider

_ENDPOINT = "plugins.model_endpoint:provider"
_FAKE = "plugins.model_endpoint:fake_provider"
_SYSTEM = """Propose one robot recovery rule from the evidence and response_schema.
Use only catalog features and offered recovery_strategies. Return one JSON object,
without prose or code fences. If evidence does not justify a candidate, return
{"kind":"none","reason":"your evidence-based reason"}. Never invent evidence."""


class LlmReasoner:
    """A model proposal is untrusted until the shared governor parser accepts it."""

    def __init__(self, *, endpoint: str = _ENDPOINT,
                 endpoint_params: Mapping[str, Any] | None = None,
                 attempts: int = 2, max_tokens: int = 2048,
                 temperature: float = 0.0, seed: int = 0) -> None:
        if attempts < 1 or max_tokens < 1:
            raise ValueError("attempts and max_tokens must be positive")
        self._ref = _FAKE if os.environ.get("PH_MODEL_ENDPOINT_FAKE") else endpoint
        self._params = dict(mount_params(endpoint) if endpoint_params is None else endpoint_params)
        self._ep = load_provider(self._ref, self._params)
        self._attempts = attempts
        self._decode = {"max_tokens": max_tokens, "temperature": temperature, "seed": seed}
        self.last_audit: dict = {}

    @property
    def identity(self) -> str:
        return "llm_reasoner:" + json.dumps({"ref": self._ref, "params": self._params,
            "endpoint": self._ep.identity, "attempts": self._attempts,
            "decode": self._decode}, sort_keys=True, separators=(",", ":"))

    def propose(self, brief: Mapping) -> Mapping:
        # Doctor checks the protocol shape with no campaign traces; it must not
        # spend a model request or claim a model abstention from that empty input.
        if not brief.get("traces"):
            return {"rule": None, "status": "not_evaluated", "reason": "no traces"}
        prereg, generation = brief["prereg"], brief["generation"]
        parent = brief.get("parent")
        rule_index = len(parent.rules) + 1 if parent is not None else generation
        traces, labels = brief["traces"], brief["labels"]
        strategies = tuple(brief.get("strategies", ()))
        evidence = build_brief(traces, labels, generation=generation,
                              privilege_budget=prereg.critic_budget, strategies=strategies)
        # Thresholds need measured values, not just a separation score. These
        # summaries expose the observed scale without selecting a threshold.
        evidence["observed_values"] = {}
        for feature in (item["name"] for item in evidence["catalog"]):
            groups = {}
            for success in (True, False):
                arrays = [np.asarray(t[feature], dtype=float).ravel()
                          for t, label in zip(traces, labels) if bool(label) == success]
                values = np.concatenate(arrays) if arrays else np.array([])
                finite = values[np.isfinite(values)]
                groups["success" if success else "failure"] = {
                    "samples": int(finite.size), "quantiles_0_25_50_75_100":
                    np.quantile(finite, [0, .25, .5, .75, 1]).tolist() if finite.size else None}
            evidence["observed_values"][feature] = groups
        n_steps = max(len(next(iter(t.values()))) for t in traces)
        audit = {"generation": generation, "identity": self.identity,
                 "status": "pending", "attempts": [], "rejections": []}
        self.last_audit = audit

        def finish(status, reason, rule=None):
            audit.update(status=status, reason=reason)
            if brief.get("store") is not None:
                brief["store"].put("model_proposal", audit)
            return {"rule": rule.canonical() if rule is not None else None,
                    "status": status, "reason": reason}

        for attempt in range(self._attempts):
            messages = [{"role": "system", "content": _SYSTEM}, {"role": "user",
                "content": json.dumps({**evidence, "attempt": attempt,
                    "previous_rejections": audit["rejections"]}, sort_keys=True)}]
            row = {"messages": messages, "raw": None, "usage": None}
            audit["attempts"].append(row)
            try:
                raw = self._ep.chat(messages, **self._decode)
            except Exception as exc:
                # Preserve the failed request before propagating the endpoint
                # failure. Nothing in this path generates an alternate proposal.
                audit["error"] = {"type": type(exc).__name__, "message": str(exc), "stage": "request"}
                finish("error", str(exc))
                raise
            row.update(raw=raw, usage=getattr(self._ep, "last_usage", None),
                       finish_reason=getattr(self._ep, "last_finish", None))
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict) and payload.get("kind") == "none":
                    reason = payload.get("reason")
                    if not isinstance(reason, str) or not reason.strip():
                        raise ProposerError("an abstention needs a non-empty reason")
                    return finish("abstained", reason)
                if isinstance(payload, dict) and payload.get("feature") not in evidence["observed_values"]:
                    raise ProposerError("feature is not in the observed catalog")
                rule = parse_proposal(payload, generation=rule_index,
                    privilege_budget=prereg.critic_budget, strategies=strategies,
                    recovery_sensor_sd=prereg.recovery_sensor_sd, n_steps=n_steps)
            except (ValueError, TypeError, OverflowError) as exc:
                audit["rejections"].append(str(exc))
                continue
            return finish("proposed", "validated model candidate", rule)
        return finish("rejected", "candidate validation budget exhausted")


def provider(**params: Any) -> LlmReasoner:
    return LlmReasoner(**params)
