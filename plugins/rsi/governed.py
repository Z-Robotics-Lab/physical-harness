"""Governed rollout: the frozen policy under a critic-recovery bundle.

Ownership
---------
This module holds the raw observation and is the only place that does. The
critic and the recovery each receive a :class:`FeatureView` built under their
own budget, and both invariants are asserted before every dispatch. That is the
structural answer to the leak recorded in local-archive/docs/retired-from-public/headline-finding.md, where a
hand-written recovery read ``obs["cube_pos"]`` directly and inflated a +13.3%
(n.s.) result into a reported +50%.

The percept model IS the ablation ladder
----------------------------------------
A recovery that re-approaches must know where the object is. Modelling that as
``estimate.cube_pos = true + N(0, sensor_sd)`` puts the sim-to-real question
inside the harness instead of beside it: ``sensor_sd=0`` is ground truth and is
declared PRIVILEGED, while a positive ``sensor_sd`` is what an onboard sensor
would actually deliver. Sweeping it produces the transfer curve directly, and no
separate "ablation mode" can drift out of sync with the thing being measured.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

# Transitional laundering point, same class as the proposer and the motor
# vocabulary: driver construction is the policy.driver capability, reached
# through the governor shell until the shared vocabulary refactor lands.
from governor.policy import make_driver
from harness import opstream
from harness.episode_log import chain_start, chain_step
from harness.invariant import (
    assert_privilege_budget,
    assert_view_reconstructable,
    record_view,
)
from harness.percept import PRIVILEGED_SENSOR_SD, PrivilegePolicy, project
from harness.spec import STACK_PHASE_HEIGHT, EpisodeSpec
from plugins.rsi.recovery import RecoveryActor
from plugins.rsi.stats.search import Trigger

#: A perfect percept is ground truth, so it costs privilege; a noisy one does not.


#: The embodiment a spec resolves to when it names no ``env_provider``: the
#: robosuite card, by ref string. A bare EpisodeSpec (tests, demos, the
#: governor.env legacy entry) resolves to exactly the environment/success/
#: object_key it did through the old ``_LegacyEmbodiment -> governor.env`` path,
#: which was itself a facade over ``plugins.embodiment_robosuite.env`` -- so the
#: None-provider path stays byte-identical (parity). Mirrors DEFAULT_PERCEPT_REF
#: below: governed.py (RSI) names the card by ref, never by import, and no longer
#: reaches back through the governor.env shim. R5 folds this into the manifest so
#: the ref stops being hard-coded here.
DEFAULT_ENV_REF = "plugins.embodiment_robosuite:provider"


def _embodiment(spec: EpisodeSpec):
    """Resolve the embodiment ONCE per rollout, by ref: the spec's, or the default."""
    from harness.registry import load_provider

    return load_provider(spec.env_provider or DEFAULT_ENV_REF)



@dataclass(frozen=True, slots=True)
class RecoverySpec:
    """A bounded, scripted repair. Phases and durations only -- no free code."""

    #: Name of a strategy in plugins.rsi.repertoire. Different SHAPES of repair, which
    #: is the axis that matters -- round 6 searched durations and the gain did not
    #: survive a clean gate.
    name: str = "regrasp"
    #: Explicit program overriding the named strategy; None uses the repertoire.
    program: tuple[tuple[str, int], ...] | None = None
    #: Std-dev of the percept the recovery re-reads. 0.0 == ground truth == privileged.
    sensor_sd: float = 0.020
    max_invocations: int = 1

    @property
    def percept_privilege(self) -> int:
        return 1 if self.sensor_sd <= PRIVILEGED_SENSOR_SD else 0

    def steps(self):
        """Resolve to (phase, duration, dx, dy) steps."""
        from plugins.rsi.repertoire import strategy

        if self.program is not None:
            return tuple((n, d, 0.0, 0.0) for n, d in self.program)
        return strategy(self.name).steps


@dataclass(frozen=True, slots=True)
class Rule:
    """One critic-recovery pair. The unit a generation appends."""

    rule_id: str
    trigger: Trigger
    recovery: RecoverySpec = field(default_factory=RecoverySpec)

    def canonical(self) -> dict:
        """Byte-stable form used for parent-freeze verification and hashing."""
        return {
            "rule_id": self.rule_id,
            "trigger": {
                "feature": self.trigger.feature, "op": self.trigger.op,
                "threshold": round(float(self.trigger.threshold), 9),
                "dwell": int(self.trigger.dwell), "arm_after": int(self.trigger.arm_after),
                # The running reducer arrived in round 14 and this form was
                # written before it. Omitting it made two rules that behave
                # differently hash identically, which blinded BOTH freeze
                # checks at once: `parent_sha` is computed from this same form,
                # so it could not act as a backstop.
                "reducer": self.trigger.reducer,
            },
            "recovery": {
                "name": self.recovery.name,
                "strategy": self.recovery.name,
                "program": [list(p) for p in self.recovery.steps()],
                "sensor_sd": round(float(self.recovery.sensor_sd), 9),
                "max_invocations": int(self.recovery.max_invocations),
            },
        }

    def declared_privilege(self) -> int:
        return self.trigger.privilege + self.recovery.percept_privilege


@dataclass(frozen=True, slots=True)
class Bundle:
    """An ordered chain of rules, grown one rule per generation.

    Zetta constrains each generation to ``append_exactly_one_critic_recovery_pair``
    with ``preserve_parent_rules_byte_for_byte`` (Zetta-Embodiment/zetta/evolution/
    stages.py:1858). The point is attribution: if a generation may rewrite its
    parent, a measured gain cannot be assigned to the one change under test.
    :meth:`assert_atomic_child_of` enforces it here rather than documenting it.
    """

    rules: tuple[Rule, ...] = ()
    critic_budget: int = 0
    action_budget: int = 0
    parent_sha: str | None = None

    def canonical(self) -> dict:
        return {
            "rules": [r.canonical() for r in self.rules],
            "critic_budget": self.critic_budget,
            "action_budget": self.action_budget,
        }

    def sha(self) -> str:
        payload = json.dumps(self.canonical(), separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def declared_privilege(self) -> int:
        """Worst-case privilege across the chain; a bundle is as privileged as its worst rule."""
        return max((r.declared_privilege() for r in self.rules), default=0)

    def append(self, rule: Rule) -> Bundle:
        """The only sanctioned growth operation."""
        return Bundle(rules=(*self.rules, rule), critic_budget=self.critic_budget,
                      action_budget=self.action_budget, parent_sha=self.sha())

    def assert_atomic_child_of(self, parent: Bundle) -> None:
        """Reject any generation that did more than append exactly one rule."""
        if len(self.rules) != len(parent.rules) + 1:
            raise ValueError(
                f"generation must append exactly one rule: parent has {len(parent.rules)}, "
                f"child has {len(self.rules)}"
            )
        for i, (a, b) in enumerate(zip(parent.rules, self.rules)):
            if a.canonical() != b.canonical():
                raise ValueError(f"parent rule {i} ({a.rule_id}) was modified; parents are frozen")
        if self.parent_sha != parent.sha():
            raise ValueError("child does not record its parent's hash")


#: Where the onboard percept implementation LIVES now (L1 rung 1): the
#: embodiment plugin. governor keeps only the seam and this named default so
#: direct EpisodeSpec users (tests, demos) keep working unchanged. The rung 1
#: caveat (constant selects behaviour without entering any hash) is closed at
#: L1 rung 3: Preregistration.percept_provider defaults to THIS constant, so
#: the effective ref enters the content hash even on the default path.
DEFAULT_PERCEPT_REF = "plugins.embodiment_robosuite.percept:provider"


def _percept_object(obs, spec: EpisodeSpec, sensor_sd: float, draw: int) -> np.ndarray:
    """Dispatch to the mounted percept provider (spec ref, or the default)."""
    from harness.registry import load_provider

    ref = spec.percept_provider or DEFAULT_PERCEPT_REF
    return load_provider(ref).object_estimate(obs, spec, sensor_sd, draw)


def _place_object(obs, spec: EpisodeSpec, sensor_sd: float, draw: int) -> np.ndarray:
    """The PLACE goal for a place-shaped recovery: the provider's independent
    cubeB estimate, drawn from a distinct stream so it does not repeat the noise
    the policy already acted on. Only place-shaped strategies reach this."""
    from harness.registry import load_provider

    ref = spec.percept_provider or DEFAULT_PERCEPT_REF
    return load_provider(ref).place_estimate(obs, spec, sensor_sd, draw)


def _is_place_recovery(steps) -> bool:
    """A recovery is place-shaped if any phase names the place vocabulary
    (over_b/place/release/retreat). Grasp/servo strategies use none of these, so
    this is False for every sealed strategy -- the grasp path stays byte-identical."""
    return any(name in STACK_PHASE_HEIGHT for name, *_ in steps)


def score_terminal(embodiment, obs, spec: EpisodeSpec, start_z: float, env) -> bool:
    """The episode/segment terminal boolean, scored on the LIVE obs + env handle
    (env still alive). Extracted from ``governed_rollout`` so the persistent
    segment path (plugins/task/workload.py) scores its sub-goal terminal through
    the SAME predicate a one-shot rollout does -- no second success dialect.

    ``spec.terminal_label`` False is the byte-identical legacy sub-goal path
    (embodiment.success); True swaps in the embodiment's full-task terminal, which
    some predicates (Stack's _check_success) can only read off the live handle.
    An embodiment asked for a terminal label it cannot produce fails loud (round 79).
    """
    if spec.terminal_label:
        terminal = getattr(embodiment, "terminal_success", None)
        if terminal is None:
            raise ValueError(
                f"spec.terminal_label is set but embodiment {type(embodiment).__name__} "
                "has no terminal_success; refusing to fall back to the sub-goal (round 79)")
        return bool(terminal(obs, spec, start_z, env=env))
    return embodiment.success(obs, spec, start_z)


def open_episode(spec: EpisodeSpec):
    """Open a PERSISTENT episode: make env, reset, build + observe the driver --
    the make+reset+observe prefix ``governed_rollout`` runs before it drives,
    factored so a persistent EpisodeContext (plugins/task/workload.py, M7) opens
    the world the SAME way a one-shot rollout does. After a successful return,
    the caller owns ``env.close()`` and fires it once at mission end. Initialization
    failures close the acquired environment here before propagating the error.
    Returns ``(embodiment, env, obs, driver)``, the live handles a segment reuses.
    """
    embodiment = _embodiment(spec)
    env = embodiment.make_env(spec)
    try:
        obs = env.reset()
        driver = make_driver(spec)
        driver.observe_once(obs)
    except BaseException as error:
        try:
            env.close()
        except BaseException as cleanup_error:  # noqa: BLE001 -- preserve the original initialization failure
            error.add_note(f"Episode initialization cleanup also failed: {cleanup_error!r}")
        raise
    return embodiment, env, obs, driver


def governed_segment(env, obs, driver, spec: EpisodeSpec, bundle: Bundle | None,
                     *, step_budget: int) -> dict:
    """Drive an ALREADY-OPEN env for a bounded step span, scoring stages and
    firing critic-recovery, then hand the env back ALIVE -- no make, no reset, no
    close. This is the bracketed middle of ``governed_rollout``, extracted so both
    it and M7's persistent ``_segment`` handler run the ONE critic-firing +
    stage-scoring loop (never a second copy). ``governed_rollout`` calls it with
    ``step_budget=spec.horizon`` and a fresh env, byte-identical to before the
    extraction; the persistent path calls it on a shared env for the remaining
    horizon so consequences carry forward across sub-goals.

    Returns the drive readouts (final ``obs``, ``steps``, ``policy_steps``,
    ``fires``, ``chain``, ``critic_privilege_used``, ``trace``, and ``stages`` iff
    ``spec.stages`` is set). The TERMINAL boolean is NOT scored here -- the caller
    runs :func:`score_terminal` on the returned obs while the env is still alive,
    because a persistent segment's terminal (a sub-goal) differs from a rollout's.
    """
    critic_names = PrivilegePolicy(critic_budget=bundle.critic_budget if bundle else 0).critic_names()
    action_names = PrivilegePolicy(critic_budget=bundle.action_budget if bundle else 0).critic_names()
    # --- R2 stage overlay: a SCORER over the policy's schedule, never a
    # controller. Everything is gated behind `stages is not None` so the
    # stageless path stays byte-identical (runs/demo, runs/demo-r1 anchors).
    # The scorer view may read privileged features -- a scorer is not a critic,
    # embodiment.success already reads ground truth -- so its reads are
    # accounted PER STAGE and never merged into `history`, the chain, or
    # assert_privilege_budget: those feed the zero-privilege search.
    stages = spec.stages
    if stages is not None:
        scorer_names = PrivilegePolicy(critic_budget=1).critic_names()
        bounds: list[int] = []
        for s in stages:
            bounds.append(s.budget + (bounds[-1] if bounds else 0))
        stage_results: list[dict] = []
        stage_idx = 0
        stage_entered = 0

        def _score_stage(exited_step: int | None) -> None:
            """Score the current stage on the CURRENT obs and advance."""
            nonlocal stage_idx, stage_entered
            sv = project(obs, scorer_names, step=k, episode=f"s{spec.seed}")
            stage_results.append({
                "name": stages[stage_idx].name,
                "entered_step": stage_entered, "exited_step": exited_step,
                "success": stages[stage_idx].holds(sv), "reached": True,
                "privilege_used": sv.privilege_used(),
            })
            # Operational feed only (no-op outside the resident runtime;
            # campaigns run in an unarmed subprocess). Fires at stage
            # cadence, never per step, so the hot loop stays cheap.
            opstream.emit("stage_transition", stage=stages[stage_idx].name,
                          success=stage_results[-1]["success"],
                          entered_step=stage_entered, exited_step=exited_step)
            stage_entered = k
            stage_idx += 1

    rules = list(bundle.rules) if bundle else []
    # Keyed by CHAIN POSITION, not rule_id: nothing forbids two rules sharing an
    # id before sealing, and keying by id silently merges their invocation budget
    # and dwell accumulator -- round 92's appended place rule (id "g1", colliding
    # with the parent grasp "g1") could never fire because the grasp rule reset
    # the shared dwell every step and ate the shared invocation. Index-keying is
    # behaviorally identical for every unique-id bundle (all sealed ones are).
    consec = [0] * len(rules)
    used = [0] * len(rules)
    fires: list[dict] = []
    history: dict[str, list[float]] = {}
    chain = chain_start()
    privilege_used = 0
    recovery: RecoveryActor | None = None
    t = 0            # env steps consumed in THIS segment, against step_budget
    k = 0            # POLICY-owned steps: the clock a trigger's arm_after counts in

    while t < step_budget:
        if recovery is not None and recovery.done:
            driver.on_handback()
            recovery = None
        if recovery is not None:
            action = recovery.act(obs)
        elif driver.exhausted:
            break
        else:
            action = driver.act(obs)

        policy_owned = recovery is None
        obs, _r, done, _info = env.step(action)
        t += 1
        if done:
            break

        # The critic governs the POLICY, so it neither observes nor fires while a
        # scripted recovery owns control, and those steps stay out of the trace.
        # Without this the search reads a mixture: round 16's second generation
        # proposed a trigger armed at t=129, inside a recovery program, on a
        # policy whose own clock is 100 steps long. Trigger `arm_after` therefore
        # counts policy-owned steps, which is also the only clock that means the
        # same thing across episodes of different length.
        if not policy_owned:
            continue

        view = project(obs, critic_names, step=k, episode=f"s{spec.seed}")
        logged = record_view(view)
        for name, value in view.snapshot().items():
            history.setdefault(name, []).append(value)
        chain = chain_step(chain, logged.digest)
        k += 1
        # Stage boundaries live on the DRIVER's schedule clock: on_handback
        # supersedes the interrupted phase, so that clock can jump several
        # cumulative budgets in one hop -- hence a while, scoring every crossed
        # stage on the current (post-recovery) obs, not only the first.
        if stages is not None:
            pk = getattr(driver, "k", k)
            while stage_idx < len(stages) and pk >= bounds[stage_idx]:
                _score_stage(exited_step=k)
        if not rules:
            continue

        assert_view_reconstructable(view, logged)
        triggered: Rule | None = None
        triggered_idx = -1
        for i, rule in enumerate(rules):
            if used[i] >= rule.recovery.max_invocations:
                consec[i] = 0
                continue
            trig = rule.trigger
            if k - 1 < trig.arm_after:
                consec[i] = 0
                continue
            # Instantaneous reading, value-only: the reducer is NOT applied here
            # (search enumerates only what this loop honors, search.RUNTIME_REDUCERS).
            # `crosses` is the one shared op/threshold predicate -- the same test
            # Trigger.fire_step steps offline, so search scores what runtime fires.
            hit = trig.crosses(view[trig.feature])
            consec[i] = consec[i] + 1 if hit else 0
            if triggered is None and consec[i] >= trig.dwell:
                triggered = rule
                triggered_idx = i
        assert_privilege_budget(view, bundle.critic_budget, role="critic")
        privilege_used = max(privilege_used, view.privilege_used())
        if triggered is None:
            continue

        used[triggered_idx] += 1
        fires.append({"rule_id": triggered.rule_id, "step": k - 1, "env_step": t - 1})
        act_view = project(obs, action_names, step=k - 1, episode=f"s{spec.seed}")
        assert_view_reconstructable(act_view, record_view(act_view))
        assert_privilege_budget(act_view, bundle.action_budget, role="recovery")
        steps = triggered.recovery.steps()
        draw = used[triggered_idx]
        if hasattr(driver, "make_recovery"):
            # Embodiment-owned recovery (M7 robocasa): the driver builds the actor
            # for its OWN action space (PandaOmron's 12-dim, base_mode discipline)
            # and reads the target off the LIVE world it already holds -- the
            # kitchen stage drivers are privileged scripted oracles, so their
            # recovery reads env truth the same way, never the tabletop percept
            # seam (whose object_key does not resolve a sub-goal task). Robosuite
            # drivers have no such method, so the two branches below are untouched.
            recovery = driver.make_recovery(triggered.recovery, obs, draw, spec)
        elif _is_place_recovery(steps):
            # Place-shaped repair: aim the actor at a fresh cubeB estimate (the
            # place goal) and point the driver's target_b at the same estimate,
            # so the phases it resumes after handback stay consistent with what
            # the recovery placed against. The seat height is already carried by
            # STACK_PHASE_HEIGHT['place'], so the estimate needs no z offset.
            goal = _place_object(obs, spec, triggered.recovery.sensor_sd, draw)
            driver.retarget_place(goal)
            recovery = RecoveryActor(steps, goal, height_offset=spec.grasp_height_offset)
        else:
            goal = _percept_object(obs, spec, triggered.recovery.sensor_sd, draw)
            driver.retarget(goal)
            recovery = RecoveryActor(steps, goal, height_offset=spec.grasp_height_offset)
        consec = [0] * len(consec)

    if stages is not None:
        # The loop can exit with boundaries still uncrossed on the ledger: an
        # exhaust break lands before the in-loop check, and a hand-back jump
        # can even exhaust the driver outright. Settle crossed stages first,
        # then score the in-progress stage on the final obs, then record the
        # stages the policy never reached.
        pk = getattr(driver, "k", k)
        while stage_idx < len(stages) and pk >= bounds[stage_idx]:
            _score_stage(exited_step=k)
        if stage_idx < len(stages) and pk > (bounds[stage_idx - 1] if stage_idx else 0):
            _score_stage(exited_step=None)
        for s in stages[stage_idx:]:
            stage_results.append({"name": s.name, "entered_step": None, "exited_step": None,
                                  "success": False, "reached": False, "privilege_used": 0})

    seg = {
        "obs": obs,
        "steps": t,
        "policy_steps": k,
        "fires": fires,
        "chain": chain,
        "critic_privilege_used": privilege_used,
        "trace": {name: np.asarray(v) for name, v in history.items()},
    }
    if stages is not None:
        # The stages key only on the opt-in path: stages=None emits nothing new.
        # Per-stage results are measurement output like trace, not hashed identity.
        seg["stages"] = stage_results
    return seg


def governed_rollout(spec: EpisodeSpec, bundle: Bundle | None) -> dict:
    """Run one episode, optionally under a critic-recovery bundle.

    Control ownership, not schedule splicing
    ----------------------------------------
    The frozen policy owns every step until a critic fires; then a scripted
    :class:`RecoveryActor` owns the next ``len(program)`` steps and hands control
    back. Recovery steps do not advance the policy's own clock, so the policy
    resumes where it was interrupted -- which is what the earlier
    splice-into-the-phase-queue implementation did, expressed in a way that also
    works for a policy with no phases at all.

    ``bundle=None`` runs the policy alone; both arms of a paired gate go through
    this identical code path.

    make + reset + observe (the prefix) -> :func:`governed_segment` (the drive,
    bounded to the full horizon) -> :func:`score_terminal` (env still alive) ->
    close. The drive is factored so M7's persistent segment path reuses this ONE
    loop on an already-open env; passing ``step_budget=spec.horizon`` on a fresh
    env is byte-identical to the pre-extraction single loop.

    A ``spec.segment_isolate`` spec routes to :func:`isolated_segment_rollout`
    instead: node-level RSI on a persistent-episode mission scores the TARGET
    sub-goal in a fresh world per seed, which every downstream gate (paired_gate,
    ablation) reaches through this one function unchanged.
    """
    if spec.segment_isolate:
        return isolated_segment_rollout(spec, bundle)
    embodiment = _embodiment(spec)
    env = embodiment.make_env(spec)
    try:
        obs = env.reset()
        driver = make_driver(spec)
        driver.observe_once(obs)
        start_z = float(np.asarray(obs[embodiment.object_key(spec)])[2])
        seg = governed_segment(env, obs, driver, spec, bundle, step_budget=spec.horizon)
        obs = seg["obs"]
        success = score_terminal(embodiment, obs, spec, start_z, env)
    finally:
        env.close()
    result = {
        "seed": spec.seed,
        "task": spec.task,
        "policy": driver.identity,
        "success": success,
        "steps": seg["steps"],
        "policy_steps": seg["policy_steps"],
        "fires": seg["fires"],
        "fired_at": seg["fires"][0]["step"] if seg["fires"] else None,
        "fired_rules": sorted({f["rule_id"] for f in seg["fires"]}),
        "chain": seg["chain"],
        "critic_privilege_used": seg["critic_privilege_used"],
        "declared_privilege": bundle.declared_privilege() if bundle else 0,
        "trace": seg["trace"],
    }
    if "stages" in seg:
        result["stages"] = seg["stages"]
    return result


def isolated_segment_rollout(spec: EpisodeSpec, bundle: Bundle | None) -> dict:
    """One node-isolated rollout of a persistent-episode mission (M7, node-level RSI).

    ``spec.segment_isolate`` is the ordered sub-goal tasks to drive on the ONE
    world ``spec`` opens (``spec.task`` is the MISSION env). The prefix sub-goals
    run UNGOVERNED to re-establish the target's precondition (the round-108
    "drive from reset to the target node's precondition" probe), then the LAST
    one -- the target -- runs under ``bundle`` and is the segment scored. Each
    seed is an independent fresh episode, so the dev/held-out blocks stay the
    clean same-seed paired population McNemar assumes -- the reason this
    per-seed-fresh path was chosen over reusing one persistent world (whose
    carried-over state would correlate the pairs).

    Returns the ``governed_rollout`` shape (the target segment's trace + fires,
    the segment boolean as ``success``), so paired_gate/ablation/run_campaign
    consume it with no branch of their own.
    """
    subgoals = spec.segment_isolate
    embodiment, env, obs, driver = open_episode(spec)
    if not hasattr(driver, "enter_segment"):
        env.close()
        raise ValueError(
            f"segment_isolate needs an episodic-segment driver (enter_segment); "
            f"{type(driver).__name__} is obs-only retargetable, no isolated path")
    seg: dict | None = None
    success = False
    try:
        cursor = 0
        for i, subtask in enumerate(subgoals):
            is_target = i == len(subgoals) - 1
            seg_spec = spec.child(task=subtask)
            driver.enter_segment(env, seg_spec)
            seg = governed_segment(env, obs, driver, seg_spec,
                                   bundle if is_target else None,
                                   step_budget=max(spec.horizon - cursor, 1))
            obs = seg["obs"]
            cursor += seg["steps"]
            success = bool(driver.segment_success(env))
            # A prefix that failed never established the target's precondition;
            # the target did not run, so this seed is an honest upstream failure
            # (rare -- e.g. kitchen nav is 150/150 -- but scored, not dropped).
            if not success and not is_target:
                break
    finally:
        env.close()
    if seg is None:
        seg = {"steps": 0, "policy_steps": 0, "fires": [], "chain": chain_start(),
               "critic_privilege_used": 0, "trace": {}}
    result = {
        "seed": spec.seed, "task": spec.task, "policy": driver.identity,
        "success": success, "steps": seg["steps"], "policy_steps": seg["policy_steps"],
        "fires": seg["fires"],
        "fired_at": seg["fires"][0]["step"] if seg["fires"] else None,
        "fired_rules": sorted({f["rule_id"] for f in seg["fires"]}),
        "chain": seg["chain"], "critic_privilege_used": seg["critic_privilege_used"],
        "declared_privilege": bundle.declared_privilege() if bundle else 0,
        "trace": seg["trace"],
    }
    if "stages" in seg:
        result["stages"] = seg["stages"]
    return result
