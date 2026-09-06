"""RSI campaign workload: run a campaign through the kernel, publish its skills.

This is the strict form of the OS main loop's Verify -> Adapt edge
(ARCHITECTURE.md): it consumes ``embodiment.env``, ``policy.driver`` and
``graph.skill`` through the kernel rather than importing providers directly,
so swapping any of them is a mount-plan edit, not a code edit here.

Thin on purpose. All of the campaign logic -- generations, paired gating,
held-out scoring, the transfer-ablation curve -- stays exactly where it was in
``plugins.rsi.campaign.run_campaign``; this module's only job is to (1) stamp the
kernel-resolved provider refs onto the preregistration before the campaign
runs, so the mount identity that produced this run's episodes is itself
preregistered and enters the content hash, and (2) turn every rule the
campaign promoted into a ``SkillRecord`` published to the resolved skill
graph. Every field on that record is a readout of the campaign's own paired
experiments -- preconditions are the trigger that armed the rule, effects and
judgement are held-out paired-gate numbers, capability_boundary is the
declared privilege plus the transfer-ablation curve -- never hand annotation
(ARCHITECTURE.md: "capability boundaries...不再是标注, 是配对实验的输出").
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from harness.kernel import Kernel
from plugins.rsi import campaign
from plugins.rsi.campaign import CampaignStore, Preregistration

#: The one SkillRecord "kind" this workload publishes. A fixed vocabulary word
#: keeps the graph queryable across campaigns instead of free text per run.
SKILL_KIND = "grasp_recovery"

#: The kernel consumer name every capability this workload resolves is
#: accounted under -- see harness/kernel.py's per-resolution audit trail.
CONSUMER = "rsi"


def _promoted_generation_records(store: CampaignStore, *, min_seq: int = 0) -> list[dict]:
    """Every `generation` artifact the campaign promoted, oldest generation first.

    `run_campaign`'s return value is a campaign SUMMARY (counts, held-out
    numbers, rule_ids); the trigger and recovery that earned each promotion
    live only in the per-generation artifacts it wrote to `store` along the
    way. Reading them back through the same content-addressed store -- rather
    than threading a second copy of the same data out of `run_campaign` --
    keeps this a read of measured campaign output, not a bookkeeping path that
    could drift from it.

    `min_seq` scopes the read to index rows written at or after that sequence
    number. `CampaignStore` is append-only, so a `store_root` pointed at a
    directory that already holds a prior campaign would otherwise resurface
    that campaign's promotions too; `run` passes the store's size from just
    before this call, so only artifacts THIS call produced are read back.
    """
    if not store.index_path.exists():
        return []
    records: list[dict] = []
    with store.index_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["seq"] < min_seq or row.get("kind") != "generation":
                continue
            payload = store.read(row["sha"])
            if payload.get("promoted"):
                records.append(payload)
    records.sort(key=lambda rec: rec["generation"])
    return records


def _mount_plan_sha(kernel: Kernel) -> str | None:
    """The most recent `kernel.mount` event's plan sha, if this kernel logs one.

    `Kernel` has no public accessor for its session log -- harness/ is owned
    by the orchestrator and not ours to extend -- so this is best-effort: a
    kernel built with no log, or whose providers were mounted one at a time
    through `Kernel.provide` rather than `Kernel.mount(plan)`, simply has no
    discoverable mount_plan_sha, and the field is omitted from published
    records rather than published as a guess.
    """
    log = getattr(kernel, "_log", None)
    if log is None:
        return None
    plan_sha: str | None = None
    for row in log.rows():
        if row.get("kind") == "kernel.mount":
            plan_sha = row["data"].get("plan_sha")
    return plan_sha


def run(
    prereg: Preregistration,
    store_root: str | Path,
    kernel: Kernel,
    *,
    workers: int = 10,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one RSI campaign through the kernel and publish its promoted skills.

    Resolves every capability this workload consumes -- `embodiment.env`,
    `policy.driver`, `percept.model`, `exec.rollouts`, `graph.skill` and
    `reasoner.proposer` -- as consumer "rsi" so the kernel's audit trail records
    the dependency, then stamps the preregistration with the resolved env/policy
    provider refs before running the campaign. Those refs travel on the spec as
    strings, never as the provider objects themselves, because a live hook does
    not survive multiprocessing spawn while a "module:factory" string pickles,
    content-hashes and audits cleanly (harness/registry.py; ARCHITECTURE.md's
    "L0 迁移方式").

    The mounted `reasoner.proposer` always drives candidate generation. Its
    declared identity is sealed by `run_campaign`; absence of an identity never
    switches execution to a different proposer.

    After the campaign returns, every rule it promoted is published as a
    `SkillRecord` to the resolved `graph.skill` provider. Returns
    ``{"result": <run_campaign's summary dict>, "skills": [<published digests>]}``.
    """
    kernel.resolve("embodiment.env", consumer=CONSUMER)
    kernel.resolve("policy.driver", consumer=CONSUMER)
    kernel.resolve("percept.model", consumer=CONSUMER)
    executor = kernel.resolve("exec.rollouts", consumer=CONSUMER)
    skill_graph = kernel.resolve("graph.skill", consumer=CONSUMER)
    reasoner = kernel.resolve("reasoner.proposer", consumer=CONSUMER)

    # Workers rebuild providers from the bare ref string, which cannot carry
    # Mount params. Refusing here beats silently running a different provider
    # configuration in every worker than the one the kernel constructed.
    for cap in ("embodiment.env", "policy.driver", "percept.model"):
        params = kernel.provider_params(cap)
        if params:
            raise ValueError(
                f"{cap} was mounted with params {params}, which cannot travel "
                "to rollout workers at L0; use a parameter-free provider ref")

    prereg2 = dataclasses.replace(
        prereg,
        env_provider=kernel.provider_ref("embodiment.env"),
        policy_provider=kernel.provider_ref("policy.driver"),
        percept_provider=kernel.provider_ref("percept.model"),
    )
    store = CampaignStore(Path(store_root))
    start_seq = store.size()
    result = campaign.run_campaign(prereg2, store, workers=workers, verbose=verbose,
                                   executor=executor, reasoner=reasoner)

    mount_plan_sha = _mount_plan_sha(kernel)
    heldout = result.get("heldout") or {}
    promoted = _promoted_generation_records(store, min_seq=start_seq)
    final_bundle_sha = promoted[-1].get("child_sha") if promoted else None
    # Held-out judgement verdict, promoted to a first-class field so consumers
    # (e.g. a skill tree) can filter on it without reaching into bundle_evidence.
    # Bundle-level, like the evidence it summarises: one held-out-vs-blind gate
    # scored over the whole chain, shared by every rule in it.
    #   None  -> require_judgement off (no heldout_vs_blind in result): not tested
    #   True  -> judgement ran and cleared p<alpha & fixed>broken on THIS block
    #   False -> judgement ran but shrank to non-significant
    # Single held-out block only; a settled verdict needs >=3 blocks (round 52),
    # so True means "this run established it", not "definitively established".
    judgement = result.get("heldout_vs_blind")
    heldout_judgement_established = (
        None if judgement is None
        else judgement["p_value"] < prereg.alpha
        and judgement["fixed"] > judgement["broken"]
    )
    digests: list[str] = []
    for rec in promoted:
        rule = rec["rule"]
        # Evidence attribution matters: the held-out numbers, the blind-twin
        # judgement and the ablation curve are properties of the FINAL BUNDLE
        # (the whole promoted chain scored once), not of any single rule in it.
        # A rule's own effect is its dev gate against its parent -- that is the
        # paired experiment that promoted it. Publishing bundle-level numbers
        # onto each rule would let a reader multiply one chain's +delta by the
        # number of rules in it.
        skill_record: dict[str, Any] = {
            "kind": SKILL_KIND,
            "policy": prereg.policy,
            "task": prereg.task,
            "generation": rec.get("generation"),
            "preconditions": rule["trigger"],
            "recovery": rule["recovery"],
            "effects": {"dev_gate_vs_parent": rec.get("dev_gate")},
            "judgement_dev": rec.get("blind_gate"),
            "heldout_judgement_established": heldout_judgement_established,
            "bundle_sha": rec.get("child_sha"),
            "bundle_evidence": {
                "note": ("bundle-level: shared by every rule in the promoted "
                         "chain, not attributable to this rule alone"),
                "final_bundle_sha": final_bundle_sha,
                "heldout": result.get("heldout"),
                "judgement": result.get("heldout_vs_blind"),
                "ablation": result.get("ablation"),
                "declared_privilege": heldout.get("declared_privilege"),
            },
            "prereg_sha": result.get("preregistration_sha"),
        }
        if mount_plan_sha is not None:
            skill_record["mount_plan_sha"] = mount_plan_sha
        digests.append(skill_graph.publish(skill_record))

    kernel.note("rsi.campaign_complete", {
        "prereg_sha": result.get("preregistration_sha"),
        "rules": result.get("rules"),
        "skills": digests,
        "heldout": {k: heldout[k] for k in ("n", "base_rate", "governed_rate",
                                            "fixed", "broken", "p_value")
                    if k in heldout},
    })
    return {"result": result, "skills": digests}
