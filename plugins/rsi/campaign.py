"""Campaign lifecycle: preregistration, atomic generations, content-hashed artifacts.

Adopted from Zetta (Zetta-Embodiment/zetta/evolution/): a preregistered seed
partition frozen before any episode runs, exactly one critic-recovery pair
appended per generation with the parent frozen, and every artifact written under
its content hash so a campaign cannot be edited in place.

Deliberate divergences
----------------------
* **Gain is measured against the PARENT, not the ungoverned policy.** Zetta's
  same-seed gate compares a candidate against its parent; reporting each
  generation against the raw baseline would let one good rule keep collecting
  credit for later generations that added nothing.
* **The privilege ablation runs at every promotion, not once at the end.**
  local-archive/docs/retired-from-public/headline-finding.md is the reason: a gain measured under a privileged
  percept is not the same quantity as a transferable gain, and finding that out
  only at the end of a campaign means every intermediate decision was made on
  the wrong number.
* **The held-out block is a test, not a promotion gate, by default.** Gating
  promotion on held-out every generation consumes it as a training signal.
  Promotion is decided on dev; held-out is scored once per campaign.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from harness.spec import EpisodeSpec, StageSpec
from plugins.rsi.gate import PairedResult, ablation_curve, paired_gate
from plugins.rsi.governed import DEFAULT_ENV_REF, DEFAULT_PERCEPT_REF, Bundle, RecoverySpec, Rule
from plugins.rsi.stats.search import DEFAULT_EARLINESS, DEFAULT_FP_PENALTY, Trigger, search_triggers

#: Threshold a `gt` trigger can never fail, so the twin fires the moment it arms.
_ALWAYS = 1e12


def blind_twin(rule: Rule) -> Rule:
    """The rule with its judgement removed: same recovery, same arm step, a
    trigger no state can fail. One constructor shared by the campaign and beam
    judgement gates, so both paths compare against the field-for-field
    identical twin (round 45's control, round 77 extends it to beam)."""
    return Rule(f"{rule.rule_id}-blind",
                Trigger(rule.trigger.feature, "gt", -_ALWAYS, 1,
                        rule.trigger.arm_after, "value"),
                rule.recovery)


def sha_json(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str).encode()
    ).hexdigest()


#: Round 90 fields that fold out of the content hash at their default value so a
#: sealed prereg (which predates them) rebuilds to its archived sha; see
#: Preregistration._hash_payload.
_HASH_FOLD_DEFAULTS = (("recovery_name", "regrasp"),
                       ("parent_store", None),
                       ("parent_final_sha", None),
                       ("reasoner", None),
                       ("segment_isolate", None),
                       ("horizon", 900))


@dataclass(frozen=True, slots=True)
class Preregistration:
    """The seed partition, frozen before any episode runs.

    Written first and hashed; every later artifact records this hash, so a
    campaign that quietly re-partitioned its seeds is detectable after the fact.
    """

    dev: tuple[int, ...]
    heldout: tuple[int, ...]
    percept_noise: float
    critic_budget: int
    action_budget: int
    recovery_sensor_sd: float
    max_generations: int
    #: Which world and which frozen policy this campaign governs. Part of the
    #: preregistration because a promoted skill is only meaningful against the
    #: policy and task it was earned on; changing either silently would make the
    #: artifact record a claim it never tested.
    task: str = "lift"
    policy: str = "scripted"
    #: Scale each generation's dev sample to the effect it can still see. With a
    #: constant size the gate is progressively underpowered as residual effects
    #: shrink; round 17 measured a candidate worth +6.5pp held-out rejected at
    #: p=0.065 on 120 seeds. `dev` becomes an ORDERED RESERVOIR and each
    #: generation takes a prefix sized from the PREVIOUS generation's residual
    #: rate -- decided before the candidate is judged, so it is not optional
    #: stopping.
    scale_dev_by_power: bool = False
    power_fix_share: float = 0.80
    power_target: float = 0.80
    min_fixed: int = 3
    alpha: float = 0.05
    #: Search the recovery program per generation, not just the trigger. The
    #: searched program is adopted only if it clears the same paired gate the
    #: trigger must clear, against the hand-written program as baseline.
    search_recovery: bool = False
    #: Seeds used for the recovery coordinate descent; a subset of `dev`.
    recovery_search_n: int = 60
    #: Rank trigger candidates by an out-of-sample shadow score instead of the
    #: in-sample search score. Costs no rollouts and demotes candidates that only
    #: fit the half they were searched on (round 7 measured a shrinkage of +0.45
    #: on one such candidate).
    screen_triggers: bool = False
    #: The trigger objective's hand-tuned weights. These live HERE, not as module
    #: globals, because round 26 measured that one of them was worth 5.0pp of
    #: held-out success: a constant that can move the headline is part of what
    #: was preregistered, and belongs in the content hash with everything else.
    #: Keeping them as defaults bound at import time also made an A/B silently
    #: run the same arm twice (round 29) -- rebinding the module attribute after
    #: import cannot reach a default already bound to the function.
    #: Generation 1 has no measured yield to carry forward. Later generations
    #: never use this; see `plan_generation`.
    prior_discordance_yield: float = 0.7
    #: Floor on the carried-forward estimate, so one unlucky generation cannot
    #: demand an unbounded reservoir.
    min_discordance_yield: float = 0.05
    #: `screen()` splits each generation in half, so a screened generation must
    #: be sized larger to leave the search the same evidence.
    screen_search_fraction: float = 0.5
    #: A candidate must out-net a blind twin of itself -- same recovery, same
    #: arm step, fired without looking at state. Off only to reproduce the
    #: pre-round-45 promotion rule.
    require_judgement: bool = True
    #: Promotion now depends on TWO tests, and round 47 found only the first was
    #: powered: at 60 seeds the judgement test could see the established rules
    #: (which beat their twins 86-99% of discordant pairs) but not a modest one
    #: at 70%, which needs 49 pairs rather than 20. Planning for the weaker
    #: effect is the point -- a rule that judges moderately still judges.
    judgement_fix_share: float = 0.70
    #: Discordant pairs per seed in the judgement comparison. Measured from the
    #: previous generation; this is the generation-1 prior (observed 0.35-0.54
    #: across four campaigns, so the low end).
    prior_judgement_yield: float = 0.35
    earliness: float = DEFAULT_EARLINESS
    fp_penalty: float = DEFAULT_FP_PENALTY
    #: Round 90: the recovery strategy every proposed rule wires (a name in
    #: plugins.rsi.repertoire). Default "regrasp" folds OUT of the content hash
    #: (`_hash_payload`), so every prereg sealed before round 90 rebuilds to its
    #: exact archived sha; place-g1 sets "replace", the place-shaped repair, which
    #: enters the hash like any other non-default field.
    recovery_name: str = "regrasp"
    #: Round 90 seeding: a sealed campaign store whose FINAL PROMOTED bundle this
    #: campaign starts from -- generations append onto it, gain is measured against
    #: it by the existing parent/child machinery -- instead of the empty bundle.
    #: None (folded out of the hash) is a from-scratch campaign; place-g1 seeds
    #: from stack-g1.
    parent_store: str | None = None
    #: The final promoted child_sha the parent store MUST rebuild to. run_campaign
    #: asserts it before seeding and fails loud on mismatch, so a parent that
    #: drifted since sealing cannot silently reroot this campaign. None (folded
    #: out) when parent_store is None.
    parent_final_sha: str | None = None
    #: Model identity is required for new runs and sealed before experiments.
    #: None folds out only for rebuilding historical preregistrations.
    reasoner: str | None = None
    #: R2 stage chain the governed rollout scores (harness/stages.py). In the
    #: preregistration because "what stage chain this bundle was scored
    #: against" is a conclusion-moving fact: asdict recursion carries it into
    #: sha(). None == no stage overlay, byte-identical legacy path. Before the
    #: provider triple, which stays the literal tail the seam guard pins.
    stages: tuple[StageSpec, ...] | None = None
    #: R2 round 79: score episodes on the embodiment's full-task terminal boolean
    #: instead of the shared sub-goal. Changing the success criterion is a new
    #: claim about what this campaign measured, so it enters the content hash.
    #: Before the provider triple, which stays the literal tail the seam guard pins.
    terminal_label: bool = False
    #: M7 node-level RSI: the ordered persistent-episode SUB-GOAL tasks each episode
    #: drives (``task`` names the mission env), the LAST being the target node scored.
    #: None (folded out of the hash) is the one-shot rollout every robosuite campaign
    #: uses; a robocasa segment campaign sets it so ``governed_rollout`` routes to the
    #: isolated-segment path. Conclusion-moving (which node was scored), so it enters
    #: the content hash when non-default -- byte-identical rebuild for every prior
    #: seal. Task-shape config, so it sits before the provider tail (seam guard).
    segment_isolate: tuple[str, ...] | None = None
    #: Episode horizon (env steps) each rollout runs under. Default 900 == the
    #: EpisodeSpec default the robosuite campaigns use, folded out of the hash so
    #: every prior seal rebuilds byte-identical; a persistent-episode segment
    #: campaign sets it above the summed sub-goal caps so the target segment never
    #: truncates (kitchen nav+grasp needs >1150). Task-shape, before the tail.
    horizon: int = 900
    #: L0 capability-seam refs ("module:factory" strings, harness/registry.py):
    #: which embodiment.env / policy.driver provider built this run's episodes.
    #: None keeps every existing archived campaign's replay path byte-identical
    #: (see governor.env.make_env / plugins.policies.drivers.make_driver's dispatch --
    #: no ref falls back to the original code exactly). Threaded into every
    #: EpisodeSpec by `_specs` below. Preregistered rather than passed
    #: out-of-band because ARCHITECTURE.md's rule is that anything able to move
    #: a conclusion enters the content hash, and which provider built an
    #: episode qualifies even when that provider is a byte-identical adapter.
    #: percept alone defaults to the EFFECTIVE constant rather than None: unlike
    #: env/policy (whose None falls back to fixed legacy code, kept for
    #: byte-identical replay), a None percept resolves to a swappable constant
    #: that would select behaviour without entering the hash -- the L1 rung 1
    #: caveat. Defaulting to the constant puts "which percept provider built
    #: this episode" in the content hash even on the default path (round 29's
    #: precedent: constants that can move a conclusion enter the hash).
    env_provider: str | None = None
    policy_provider: str | None = None
    percept_provider: str | None = DEFAULT_PERCEPT_REF

    def __post_init__(self) -> None:
        overlap = set(self.dev) & set(self.heldout)
        if overlap:
            raise ValueError(f"dev and held-out seeds overlap: {sorted(overlap)[:5]}")
        # An explicit None would hash as None while behaving as the constant --
        # the exact desync the default above closes. Normalise it away.
        if self.percept_provider is None:
            object.__setattr__(self, "percept_provider", DEFAULT_PERCEPT_REF)

    def _hash_payload(self) -> dict:
        """asdict for the content hash, with the round-90 fields folded out at
        their defaults. Appending a plain field would move every predating
        archive's sha (asdict adds a key); folding the default keeps each sealed
        prereg byte-identical while a non-default value -- place-g1 sets all
        three -- still enters the hash for free through asdict."""
        d = asdict(self)
        for name, default in _HASH_FOLD_DEFAULTS:
            if d[name] == default:
                del d[name]
        return d

    def sha(self) -> str:
        return sha_json(self._hash_payload())


@dataclass
class CampaignStore:
    """Append-only, content-addressed artifact directory."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        (self.root / "artifacts").mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def put(self, kind: str, payload: dict) -> str:
        """Write one artifact under its content hash and index it. Never overwrites."""
        digest = sha_json(payload)
        path = self.root / "artifacts" / f"{digest}.json"
        if not path.exists():
            path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
        with self.index_path.open("a") as fh:
            fh.write(json.dumps({"seq": self.size(), "kind": kind, "sha": digest,
                                 "time": time.time()}, default=str) + "\n")
        return digest

    def size(self) -> int:
        if not self.index_path.exists():
            return 0
        return sum(1 for _ in self.index_path.open())

    def read(self, digest: str) -> dict:
        return json.loads((self.root / "artifacts" / f"{digest}.json").read_text())


def stage_attribution(results: Sequence[dict]) -> dict | None:
    """Per-stage failure attribution over governed rollout results.

    Round 78's ruling made ``stages is not None`` the only opt-in, so the
    presence of the ``stages`` key on a rollout result IS the switch here:
    stageless results return None and nothing is written (runs/demo and the
    lift anchors stay byte-identical). Episodes are bucketed by their FURTHEST
    reached stage, and each bucket counts the episode's TERMINAL label -- the
    same boolean the gate consumed -- so the buckets' terminal_success total
    equals the gate's label count by construction: a descriptive table that
    could silently diverge from the gated boolean would be worse than none.
    ``first_failure`` histograms the residual failures by the first stage
    whose predicate did not hold; ``(terminal)`` catches episodes whose every
    stage held but whose terminal label still failed, ``(none)`` catches
    episodes that reached no stage at all.
    """
    if not results or "stages" not in results[0]:
        return None
    names = [s["name"] for s in results[0]["stages"]]
    per_stage = {n: {"reached": 0, "success": 0} for n in names}
    furthest = {n: {"episodes": 0, "terminal_success": 0} for n in (*names, "(none)")}
    first_failure = dict.fromkeys((*names, "(terminal)"), 0)
    for r in results:
        deepest = "(none)"
        for s in r["stages"]:
            if s["reached"]:
                per_stage[s["name"]]["reached"] += 1
                deepest = s["name"]
            if s["success"]:
                per_stage[s["name"]]["success"] += 1
        furthest[deepest]["episodes"] += 1
        furthest[deepest]["terminal_success"] += int(bool(r["success"]))
        if not r["success"]:
            failed = next((s["name"] for s in r["stages"] if not s["success"]), "(terminal)")
            first_failure[failed] += 1
    return {"n": len(results),
            "successes": int(sum(bool(r["success"]) for r in results)),
            "stage_order": names, "per_stage": per_stage,
            "furthest": furthest, "first_failure": first_failure}


def _specs(seeds: Sequence[int], prereg: Preregistration) -> list[EpisodeSpec]:
    return [EpisodeSpec(seed=s, task=prereg.task, policy=prereg.policy,
                        percept_noise=prereg.percept_noise,
                        stages=prereg.stages,
                        terminal_label=prereg.terminal_label,
                        env_provider=prereg.env_provider,
                        policy_provider=prereg.policy_provider,
                        percept_provider=prereg.percept_provider,
                        segment_isolate=prereg.segment_isolate,
                        horizon=prereg.horizon) for s in seeds]


def propose_rule(
    traces, labels, *, generation: int, prereg: Preregistration,
    dev_specs=None, executor=None, workers: int = 10,
    parent: Bundle | None = None, store: CampaignStore | None = None,
) -> Rule | None:
    """Deterministic proposer: the best admissible trigger over residual failures.

    Historical research helper with zero external API calls. It is never the
    default proposer of run_campaign or the mounted reasoner card.

    With `dev_specs` and an `executor`, the round-88 repair tie-break applies to
    the in-sample ranked path: the objective is blind to fire time, so many arm
    variants tie at the float-exact top score and the family dedup keeps
    whichever enumerated first -- the divergence onset, not the arm that repairs
    most. The tied candidates are replayed on the dev block and the max-fixed
    one kept; a dev replay for SELECTION is licensed use of a dev block. The
    audit, including what the cap dropped, is sealed in `store` when given.
    """
    if not any(labels) or all(labels):
        return None
    # Mint the id from the rule's 1-indexed CHAIN POSITION, not `generation`, so a
    # rule appended onto a seeded parent (place-g1 seeds stack-g1's g1) does not
    # reuse the parent's id and collide (round 92). Byte-identical for from-scratch
    # campaigns: every prior generation promoted (run_campaign breaks on rejection),
    # so len(parent.rules) == generation - 1 there; beam mints its own id downstream.
    rule_id = f"g{(len(parent.rules) if parent is not None else 0) + 1}"
    if prereg.screen_triggers:
        from plugins.rsi.stats.screen import screen

        screened = screen(traces, labels, privilege_budget=prereg.critic_budget, pool=8,
                          earliness=prereg.earliness, fp_penalty=prereg.fp_penalty)
        if screened:
            # No repair tie-break on this path: screening already re-ranked the
            # family-deduped pool by OUT-of-sample shadow score, and replaying
            # the in-sample top-score ties would override that ordering with
            # the very evidence screening exists to discount. The tie-break is
            # an in-sample-ranking repair only; the fall-through below is that
            # ranking, so it gets the tie-break.
            return Rule(rule_id=rule_id, trigger=screened[0].trigger,
                        recovery=RecoverySpec(name=prereg.recovery_name,
                                              sensor_sd=prereg.recovery_sensor_sd))
        # too few episodes to split; fall through to the in-sample ranking
    ranked = search_triggers(traces, labels, privilege_budget=prereg.critic_budget, top_k=3,
                             earliness=prereg.earliness, fp_penalty=prereg.fp_penalty)
    if not ranked:
        return None
    trigger = ranked[0].trigger
    if dev_specs is not None and executor is not None:
        from governor.proposer import break_tie_by_repair

        # recovery_name threads the campaign's ACTUAL repair (place-g1's `replace`)
        # into the replay, so `fixed` measures the repair the campaign will run.
        trigger, selection = break_tie_by_repair(
            traces, labels, privilege_budget=prereg.critic_budget,
            recovery_sensor_sd=prereg.recovery_sensor_sd, dev_specs=dev_specs,
            executor=executor, workers=workers, default=trigger, parent=parent,
            recovery_name=prereg.recovery_name,
            earliness=prereg.earliness, fp_penalty=prereg.fp_penalty)
        if store is not None:
            store.put("tie_break", {"preregistration_sha": prereg.sha(),
                                    "generation": generation, **selection})
    return Rule(
        rule_id=rule_id,
        trigger=trigger,
        recovery=RecoverySpec(name=prereg.recovery_name, sensor_sd=prereg.recovery_sensor_sd),
    )


@dataclass
class GenerationRecord:
    """One generation's decision and the evidence behind it."""

    generation: int
    rule_id: str
    trigger: str
    parent_sha: str
    child_sha: str
    dev: PairedResult
    promoted: bool
    reason: str


def _seed_from_parent(prereg: Preregistration, *, verbose: bool = True) -> Bundle:
    """Rebuild the parent store's final promoted bundle, assert it matches the
    preregistered parent_final_sha, and return it rebudgeted to THIS campaign's
    budgets.

    The rebuild reuses ``plugins.rsi.rebuild.rebuild_final_bundle`` -- the same
    child_sha-asserted reconstruction the rescore and probe scripts use (they
    re-export it) -- so there is one reconstruction, not a second that could
    drift. The parent's own budgets only had to reproduce the sealed sha (the
    assertion pins that); the returned bundle carries prereg's budgets because the
    new generations search under those. The parent's promoted rules are
    observable, so their behaviour is budget-invariant -- rebudgeting only widens
    the view the NEW rules may read, which is the point of the higher critic_budget.
    """
    from harness.registry import load_provider
    from plugins.rsi.rebuild import (
        read_store_artifacts,
        rebuild_final_bundle,
        rebuild_preregistration,
    )

    archived = read_store_artifacts(prereg.parent_store)
    prereg_payloads = archived.get("preregistration", [])
    if not prereg_payloads:
        raise ValueError(f"parent store {prereg.parent_store} has no preregistration artifact")
    parent_prereg = rebuild_preregistration(prereg_payloads[0])
    # Registry order (round 69/85): declared_privilege reads the feature catalog,
    # populated when the embodiment plugin is imported. Load the parent's own
    # provider ref first, exactly as rescore_heldout.run_rescore does, so a fresh
    # process (a direct run_campaign, not via workload.run) is safe too.
    load_provider(parent_prereg.env_provider or "plugins.embodiment_robosuite:provider", {})
    parent = rebuild_final_bundle(parent_prereg, archived.get("generation", []))
    if parent.sha() != prereg.parent_final_sha:
        raise AssertionError(
            f"parent store {prereg.parent_store} rebuilds to bundle {parent.sha()[:12]}, "
            f"but the prereg pinned parent_final_sha {str(prereg.parent_final_sha)[:12]}: "
            "refusing to seed from a parent that does not match its preregistration")
    if verbose:
        print(f"seeded from {prereg.parent_store} final bundle {parent.sha()[:12]} "
              f"({len(parent.rules)} rule(s)), rebudgeted to critic={prereg.critic_budget} "
              f"action={prereg.action_budget}")
    return Bundle(rules=parent.rules, critic_budget=prereg.critic_budget,
                  action_budget=prereg.action_budget)


def run_campaign(
    prereg: Preregistration, store: CampaignStore, *, workers: int = 10, verbose: bool = True,
    executor=None, reasoner=None,
) -> dict:
    """Drive generations until nothing further clears the dev gate.

    ``reasoner`` must be supplied explicitly by the workload's mounted seam.
    Model proposals still pass the existing development, blind-twin, held-out,
    and sensor-degradation measurements. No search replaces a model candidate.
    """
    if reasoner is None:
        raise ValueError("run_campaign requires an explicit LLM reasoner")
    if prereg.search_recovery:
        raise ValueError("search_recovery is unavailable in LLM-only campaigns")
    identity = getattr(reasoner, "identity", None)
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("run_campaign requires a non-empty reasoner.identity")
    prereg = replace(prereg, reasoner=identity)
    prereg_sha = store.put("preregistration", prereg._hash_payload())
    if verbose:
        print(f"preregistration {prereg_sha[:12]}  dev={len(prereg.dev)} heldout={len(prereg.heldout)}")

    reservoir = _specs(prereg.dev, prereg)
    dev_specs = reservoir
    if prereg.parent_store is not None:
        bundle = _seed_from_parent(prereg, verbose=verbose)
    else:
        bundle = Bundle(rules=(), critic_budget=prereg.critic_budget,
                        action_budget=prereg.action_budget)
    history: list[GenerationRecord] = []
    plans: list[dict] = []

    prev_residual_rate: float | None = None
    #: Discordant pairs per residual failure, carried forward from the last
    #: completed generation. Generation 1 has nothing to measure and uses the
    #: prior; every later generation is sized on this campaign's own evidence.
    prev_yield: float | None = None
    prev_j_yield: float | None = None
    for gen in range(1, prereg.max_generations + 1):
        # Size this generation BEFORE anything about its candidate is known, from
        # the previous generation's residual rate. Measurement, search and gate
        # then all run on the SAME slice: sizing the gate differently from the
        # search would judge a candidate on a population it was not fitted to.
        if prereg.scale_dev_by_power:
            from plugins.rsi.stats.power import plan_generation

            rate = prev_residual_rate if prev_residual_rate is not None else 0.5
            plan = plan_generation(gen, round(rate * len(reservoir)), len(reservoir),
                                   len(reservoir), fix_share=prereg.power_fix_share,
                                   alpha=prereg.alpha, power=prereg.power_target,
                                   discordance_yield=(prev_yield if prev_yield is not None
                                                      else prereg.prior_discordance_yield),
                                   search_fraction=(prereg.screen_search_fraction
                                                    if prereg.screen_triggers else 1.0))
            # The generation must satisfy BOTH tests it will be judged by.
            jplan = plan_generation(gen, round(rate * len(reservoir)), len(reservoir),
                                    len(reservoir), fix_share=prereg.judgement_fix_share,
                                    alpha=prereg.alpha, power=prereg.power_target,
                                    discordance_yield=1.0)
            j_seeds = int(round(jplan.discordant_needed /  # noqa: RUF046  parity-pinned numeric path
                                max(prev_j_yield if prev_j_yield is not None
                                    else prereg.prior_judgement_yield, 1e-6)))
            if prereg.require_judgement and j_seeds > plan.seeds_used:
                plan = replace(plan, seeds_needed=max(plan.seeds_needed, j_seeds),
                               seeds_used=min(max(plan.seeds_used, j_seeds), len(reservoir)),
                               capped=j_seeds > len(reservoir))
                if verbose:
                    print(f"  judgement test needs {jplan.discordant_needed} pairs "
                          f"-> {j_seeds} seeds; generation sized to {plan.seeds_used}")
            dev_specs = reservoir[: plan.seeds_used]
            plans.append(asdict(plan))
            if verbose:
                print(f"  {plan.line()}")

        # Residual failures under the CURRENT bundle are the target population.
        from plugins.rsi.gate import _run
        from plugins.rsi.parallel import default_executor
        ex = executor or default_executor()
        cur = ex.map(
            _run, [(s, bundle if bundle.rules else None) for s in dev_specs],
            workers=workers)
        labels = [r["success"] for r in cur]
        rate = float(np.mean(labels))
        if verbose:
            print(f"\ngen {gen}: current dev rate {rate:.1%} ({sum(labels)}/{len(labels)})")
        # "Where do the residual failures land" as a sealed artifact, not a
        # transient dict. stages=None writes nothing (round 78's opt-in).
        attribution = stage_attribution(cur)
        if attribution is not None:
            store.put("stage_attribution", {"preregistration_sha": prereg_sha,
                                            "generation": gen, "table": attribution})
        if all(labels):
            if verbose:
                print("  no residual failures on dev; campaign converged")
            break

        # Resolve the candidate through the mounted reasoner seam (dead until R7:
        # the loop hard-called propose_rule and never consulted the mount). The
        # brief carries live objects -- this is an in-process seam, not a
        # serialized one -- so a reasoner may return either a Rule or its
        # canonical dict; both are accepted.
        from plugins.rsi.repertoire import strategies_for
        provider_parts = (prereg.env_provider or DEFAULT_ENV_REF).partition(":")[0].split(".")
        card = provider_parts[1] if len(provider_parts) > 1 and provider_parts[0] == "plugins" else ""
        brief = {"traces": [r["trace"] for r in cur], "labels": labels,
                 "generation": gen, "prereg": prereg, "dev_specs": dev_specs,
                 "executor": ex, "workers": workers, "parent": bundle, "store": store,
                 # The installed recovery vocabulary the model may name.
                 "strategies": tuple(strategies_for(card))}
        proposed = reasoner.propose(brief).get("rule")
        if isinstance(proposed, Mapping):
            from plugins.rsi.rebuild import rule_from_canonical
            rule = rule_from_canonical(proposed)
        else:
            rule = proposed
        if rule is None:
            if verbose:
                print("  proposer produced no admissible candidate; stopping")
            break

        child = bundle.append(rule)
        child.assert_atomic_child_of(bundle)     # one rule appended, parent frozen
        dev_result = paired_gate(dev_specs, child, baseline=bundle if bundle.rules else None,
                                 workers=workers, executor=executor)

        # The blind control was a per-report check by hand until round 45, when a
        # third policy grew a rule that fired on 200/200 held-out episodes and
        # tied its own blind arm. Beating the ungoverned baseline is not enough:
        # a recovery that runs unconditionally can do that on a weak policy
        # without the critic judging anything. The win has to come from choosing
        # WHEN, so every candidate now runs against a blind twin of itself.
        blind_child = bundle.append(blind_twin(rule))
        # Head to head on the SAME seeds, not two separate comparisons against the
        # parent: netting 35 against a blind twin's 34 is one episode of noise, and
        # that is exactly what slipped through when this gate first shipped.
        blind_result = paired_gate(dev_specs, child, baseline=blind_child, workers=workers,
                                  executor=executor)
        judged = (blind_result.p_value < prereg.alpha
                  and blind_result.fixed > blind_result.broken)

        promoted = (dev_result.fixed >= prereg.min_fixed
                    and dev_result.p_value < prereg.alpha
                    and dev_result.fixed > dev_result.broken
                    and (judged or not prereg.require_judgement))
        reason = ("promoted" if promoted else
                  "rejected (no judgement vs its blind twin: "
                  f"delta={blind_result.delta:+.1%} p={blind_result.p_value:.4f})"
                  if not judged and prereg.require_judgement else
                  f"rejected (fixed={dev_result.fixed} broken={dev_result.broken} p={dev_result.p_value:.4f})")
        if verbose:
            print(f"  vs its blind twin (same recovery, unconditional at t={rule.trigger.arm_after}): "
                  f"{blind_result.line()}")
        rec = GenerationRecord(gen, rule.rule_id, rule.trigger.describe(), bundle.sha(),
                               child.sha(), dev_result, promoted, reason)
        # Sized from a SEALED generation, never from the candidate being planned.
        residual_on_slice = max(len(dev_specs) - sum(labels), 1)
        prev_yield = max((dev_result.fixed + dev_result.broken) / residual_on_slice,
                         prereg.min_discordance_yield)
        prev_j_yield = max((blind_result.fixed + blind_result.broken) / max(len(dev_specs), 1),
                           prereg.min_discordance_yield)
        history.append(rec)
        store.put("generation", {
            "blind_gate": asdict(blind_result),
            "preregistration_sha": prereg_sha, "generation": gen,
            "rule": rule.canonical(), "parent_sha": bundle.sha(), "child_sha": child.sha(),
            "dev_gate": asdict(dev_result), "promoted": promoted, "reason": reason,
        })
        if verbose:
            print(f"  candidate: {rule.trigger.describe()}")
            print(f"  dev gate vs parent: {dev_result.line()}")
            print(f"  -> {reason}")
        if not promoted:
            break
        bundle = child
        prev_residual_rate = 1.0 - dev_result.governed_rate

    # --- held-out is scored ONCE, after the campaign, as a test --------------
    result: dict = {"preregistration_sha": prereg_sha, "power_plans": plans,
                    "generations": len(history),
                    "promoted": sum(1 for h in history if h.promoted),
                    "final_sha": bundle.sha(), "rules": [r.rule_id for r in bundle.rules]}
    if bundle.rules:
        held = _specs(prereg.heldout, prereg)
        final = paired_gate(held, bundle, baseline=None, workers=workers, executor=executor)
        result["heldout"] = asdict(final)
        if verbose:
            print(f"\nheld-out (scored once, n={len(held)}): {final.line()}")
        # The judgement test decides promotion on the SAME slice the trigger was
        # searched on, so its estimate is inflated: round 50 measured a shrinkage
        # of +5.0pp for one policy and +15.7pp for another. Re-run it on held-out
        # seeds so "the win is judgement, not extra control steps" is a held-out
        # claim rather than an in-sample one.
        if bundle.rules and prereg.require_judgement:
            blind = Bundle(rules=tuple(blind_twin(r) for r in bundle.rules),
                           critic_budget=bundle.critic_budget,
                           action_budget=bundle.action_budget)
            held_blind = paired_gate(held, bundle, baseline=blind, workers=workers,
                                     executor=executor)
            result["heldout_vs_blind"] = asdict(held_blind)
            if verbose:
                established = (held_blind.p_value < prereg.alpha
                               and held_blind.fixed > held_blind.broken)
                print(f"held-out vs blind twin: {held_blind.line()}  "
                      f"-> judgement {'established' if established else 'NOT established'}")
        curve = ablation_curve(held, bundle, workers=workers, executor=executor)
        result["ablation"] = [(sd, asdict(r)) for sd, r in curve]
        if verbose:
            print("held-out transfer ablation:")
            for sd, r in curve:
                tag = "GROUND TRUTH (privileged)" if sd == 0.0 else "onboard sensor"
                print(f"  sensor_sd={sd:.3f}  {r.line()}  [{tag}]")
        store.put("campaign_result", result)
    return result
