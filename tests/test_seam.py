"""Regression for the L0 capability seam: provider identity travels with specs.

`EpisodeSpec` carries `env_provider` / `policy_provider` as "module:factory"
strings, not module-global hooks, because a hook does not survive
multiprocessing spawn while a string pickles, content-hashes, and audits
cleanly (see governor/env.py, ARCHITECTURE.md's "L0 迁移方式", and
local-archive/docs/retired-from-public/observability-design.md's spawn findings). This file covers:

(a) the new fields default to the pre-seam shape, and a spec with refs set
    still pickles;
(b) each new plugin adapter satisfies its harness.contracts Protocol;
(c) make_env dispatch equivalence, with one short real env build/step/close;
(d) make_driver dispatch equivalence.
"""

from __future__ import annotations

import dataclasses as dc
import json
import pickle

import numpy as np
import pytest

from governor.env import NOMINAL_SCHEDULE, EpisodeSpec, make_env
from harness.contracts import EnvProvider, PolicyFactory, Reasoner
from harness.registry import load_provider
from plugins.policies.drivers import ScriptedDriver, make_driver

ENV_REF = "plugins.embodiment_robosuite:provider"
POLICY_REF = "plugins.policies:provider"
REASONER_REF = "plugins.reasoner:provider"


# --- (a) EpisodeSpec: unchanged shape when the new fields are omitted -------

def test_episode_spec_omits_new_fields_unchanged():
    """A spec built with no provider refs is exactly what it was pre-seam."""
    s = EpisodeSpec(seed=7)
    assert dc.asdict(s) == {
        "seed": 7, "task": "lift", "robot": "Panda", "horizon": 900,
        "percept_noise": 0.020, "arm_noise": 0.02, "kp": 8.0, "policy": "scripted",
        "schedule": NOMINAL_SCHEDULE, "grasp_height_offset": 0.0, "stages": None,
        "terminal_label": False, "segment_isolate": None,
        "env_provider": None, "policy_provider": None, "percept_provider": None,
    }


def test_new_fields_are_appended_at_the_end():
    """This codebase's defaulted-field-ordering rule, checked structurally.

    R2 ruling: `stages` is task-shape config, not a provider ref, so it sits
    BEFORE the provider block -- the triple stays the literal tail this pins.
    Round 79 slots `terminal_label` in the same task-shape band, still before
    the provider tail."""
    names = [f.name for f in dc.fields(EpisodeSpec)]
    assert names[-3:] == ["env_provider", "policy_provider", "percept_provider"]
    # M7 node-level RSI's segment_isolate joins the task-shape band ahead of the
    # provider tail (like stages/terminal_label); the triple keeps the literal tail.
    assert names[-4] == "segment_isolate"
    assert names[-5] == "terminal_label"
    assert names[-6] == "stages"
    assert all(f.default is None for f in dc.fields(EpisodeSpec)
              if f.name in ("env_provider", "policy_provider", "stages", "segment_isolate"))


def test_episode_spec_with_provider_refs_pickles():
    """Strings survive multiprocessing spawn; that is the whole point."""
    s = EpisodeSpec(seed=0, env_provider=ENV_REF, policy_provider=POLICY_REF)
    blob = pickle.dumps(s)
    assert pickle.loads(blob) == s


def test_episode_spec_with_stages_pickles():
    """A stage chain must survive multiprocessing spawn like every other field."""
    from harness.spec import Clause, StageSpec

    chain = (StageSpec("grasp", (Clause("observable.finger_gap", "gt", 0.01),), 62),)
    s = EpisodeSpec(seed=0, stages=chain)
    assert pickle.loads(pickle.dumps(s)) == s


# --- (b) each provider satisfies its harness contract -----------------------
# Loaded the same way the kernel would mount them: through the registry ref
# string, not a direct import -- exercising the actual "module:factory" path.

def test_robosuite_embodiment_satisfies_env_provider():
    p = load_provider(ENV_REF)
    assert isinstance(p, EnvProvider)
    # sorted TASKS: the four staged tasks + the M7 clear_workspace mode-0 family
    # (clearall builds the persistent env; clear{milk,bread,cereal,can} name each
    # sub-goal's object_key -- plugins/embodiment_robosuite/env.py TASKS).
    assert p.tasks() == ("clearall", "clearbread", "clearcan", "clearcereal",
                         "clearmilk", "lift", "pickcan", "pickmilk", "stack")


def test_governor_policies_satisfies_policy_factory():
    assert isinstance(load_provider(POLICY_REF), PolicyFactory)


def test_model_reasoner_satisfies_reasoner(tmp_path, monkeypatch):
    reply = tmp_path / "model.json"
    reply.write_text(json.dumps({"kind": "none", "reason": "No measured intervention yet."}))
    monkeypatch.setenv("PH_MODEL_ENDPOINT_FAKE", str(reply))
    assert isinstance(load_provider(REASONER_REF), Reasoner)


def test_model_reasoner_propose_thin_adapter_round_trips(tmp_path, monkeypatch):
    """Not required by the contract shape check above, but this is new code:
    a brief in, a plain-Mapping proposal out, matching governor.proposer's
    own return shape (a Rule or None) collapsed through Rule.canonical().
    """
    from plugins.rsi.campaign import Preregistration

    prereg = Preregistration(dev=tuple(range(20)), heldout=tuple(range(20, 30)),
                             percept_noise=0.02, critic_budget=0, action_budget=0,
                             recovery_sensor_sd=0.02, max_generations=1)
    traces, labels = [], []
    for i in range(20):
        failing = i % 2 == 0
        traces.append({
            "observable.finger_gap": np.full(40, 0.001 if failing else 0.04),
            "observable.eef_z": np.linspace(1.0, 0.9, 40),
            "observable.gripper_effort": np.zeros(40),
            "observable.joint_speed": np.zeros(40),
        })
        labels.append(not failing)

    reply = tmp_path / "model.json"
    reply.write_text(json.dumps({"feature": "observable.finger_gap", "op": "lt", "threshold": 0.02,
                                 "dwell": 1, "arm_after": 10, "reducer": "value", "recovery": "regrasp"}))
    monkeypatch.setenv("PH_MODEL_ENDPOINT_FAKE", str(reply))
    reasoner = load_provider(REASONER_REF)
    out = reasoner.propose({"traces": traces, "labels": labels, "generation": 1,
                            "prereg": prereg, "strategies": ("regrasp",)})
    assert isinstance(out, dict) and "rule" in out
    assert out["status"] == "proposed" and out["rule"]["rule_id"] == "g1"
    assert out["rule"]["trigger"]["threshold"] == 0.02 and reasoner.identity.startswith("llm_reasoner:")


# --- (c) make_env dispatch equivalence, with ONE short real env build -------

@pytest.mark.robosuite
def test_make_env_dispatch_equivalence_and_smoke():
    """A ref'd spec builds through the plugin and behaves like a real env; a
    ref-less spec still resolves to the identical class (dispatch equivalence).
    """
    ref_spec = EpisodeSpec(seed=0, task="lift", env_provider=ENV_REF)
    plain_spec = EpisodeSpec(seed=0, task="lift")

    env = make_env(ref_spec)
    try:
        obs = env.reset()
        assert "robot0_eef_pos" in obs
        action = np.zeros(env.action_spec[0].shape, dtype=np.float32)
        env.step(action)
    finally:
        env.close()

    baseline = make_env(plain_spec)
    try:
        assert type(baseline) is type(env)
    finally:
        baseline.close()


# --- (d) make_driver dispatch equivalence ------------------------------------

def test_make_driver_dispatch_equivalence():
    ref_spec = EpisodeSpec(seed=0, policy="scripted", policy_provider=POLICY_REF)
    plain_spec = EpisodeSpec(seed=0, policy="scripted")

    ref_driver = make_driver(ref_spec)
    plain_driver = make_driver(plain_spec)
    assert type(ref_driver) is type(plain_driver) is ScriptedDriver


def test_preregistration_provider_fields_are_appended_at_the_end():
    """Same guard EpisodeSpec has: the refs must stay last, defaulted.

    The defaults are deliberately asymmetric: env/policy None falls back to
    fixed legacy code (byte-identical replay), but percept's None would resolve
    to a swappable constant outside the hash, so its default IS the constant --
    the effective ref enters the content hash even on the default path.
    """
    import dataclasses as dc

    from plugins.rsi.campaign import Preregistration
    from plugins.rsi.governed import DEFAULT_PERCEPT_REF

    fields = {f.name: f for f in dc.fields(Preregistration)}
    names = list(fields)
    assert names[-3:] == ["env_provider", "policy_provider", "percept_provider"]
    # R2 ruling: `stages` sits directly before the provider block, like on
    # EpisodeSpec -- the triple keeps the literal tail. Round 79's
    # terminal_label and M7's segment_isolate/horizon join the same task-shape band.
    assert names[-4] == "horizon"
    assert names[-5] == "segment_isolate"
    assert names[-6] == "terminal_label"
    assert names[-7] == "stages"
    assert fields["stages"].default is None
    assert fields["env_provider"].default is None
    assert fields["policy_provider"].default is None
    assert fields["percept_provider"].default == DEFAULT_PERCEPT_REF


def _prereg(**kw):
    from plugins.rsi.campaign import Preregistration

    return Preregistration(dev=(0, 1), heldout=(2, 3), percept_noise=0.02,
                           critic_budget=0, action_budget=0,
                           recovery_sensor_sd=0.02, max_generations=1, **kw)


def test_preregistration_percept_provider_default_is_the_constant():
    from plugins.rsi.governed import DEFAULT_PERCEPT_REF

    assert _prereg().percept_provider == DEFAULT_PERCEPT_REF


def test_percept_ref_moves_the_prereg_hash():
    """The core invariant of L1 rung 3: the effective percept ref is content."""
    assert _prereg().sha() != _prereg(percept_provider="other.mod:provider").sha()


def test_explicit_none_percept_is_coerced_into_the_hash():
    """An explicit None must not reopen the hash=None/behaviour=constant split."""
    from plugins.rsi.governed import DEFAULT_PERCEPT_REF

    p = _prereg(percept_provider=None)
    assert p.percept_provider == DEFAULT_PERCEPT_REF
    assert p.sha() == _prereg().sha()


def test_specs_threads_the_percept_ref():
    """The default path delivers the real ref to every EpisodeSpec, so
    _percept_object never takes its None fallback for campaign episodes."""
    from plugins.rsi.campaign import _specs
    from plugins.rsi.governed import DEFAULT_PERCEPT_REF

    assert _specs([0], _prereg())[0].percept_provider == DEFAULT_PERCEPT_REF


def test_grasp_height_offset_moves_only_descend_and_close():
    """Round 60: the Sawyer correction must not touch above/lift, and 0.0 must
    reproduce the Panda action bit for bit."""
    import numpy as np

    from harness.spec import EpisodeSpec
    from plugins.policies.drivers import FrozenPolicy

    obs = {"robot0_eef_pos": np.array([0.0, 0.0, 0.9])}
    plain = FrozenPolicy(EpisodeSpec(seed=1))
    shifted = FrozenPolicy(EpisodeSpec(seed=1, grasp_height_offset=-0.01))
    plain.target = np.array([0.0, 0.0, 0.82])
    shifted.target = np.array([0.0, 0.0, 0.82])
    for phase in ("above", "lift"):
        assert np.array_equal(plain.act(obs, phase), shifted.act(obs, phase)), phase
    for phase in ("descend", "close"):
        assert not np.array_equal(plain.act(obs, phase), shifted.act(obs, phase)), phase
    default = FrozenPolicy(EpisodeSpec(seed=1, grasp_height_offset=0.0))
    default.target = plain.target
    for phase in ("above", "descend", "close", "lift"):
        assert np.array_equal(plain.act(obs, phase), default.act(obs, phase))


def test_sawyer_providers_satisfy_their_contracts():
    from harness.contracts import EnvProvider, PolicyFactory
    from harness.registry import load_provider

    assert isinstance(load_provider("plugins.embodiment_robosuite:sawyer_provider"), EnvProvider)
    assert isinstance(load_provider("plugins.policies:sawyer_scripted_provider"), PolicyFactory)


def test_recovery_actor_honours_the_grasp_height_offset():
    """Round 61: repair must be embodiment-corrected the same way the policy is.

    offset=0.0 reproduces the Panda behaviour bit for bit; a Sawyer offset
    moves descend/close goals and leaves above/lift alone.
    """
    import numpy as np

    from plugins.rsi.recovery import RecoveryActor

    target = np.array([0.0, 0.0, 0.82])
    obs = {"robot0_eef_pos": np.array([0.0, 0.0, 0.9])}
    prog = (("above", 1, 0.0, 0.0), ("descend", 1, 0.0, 0.0),
            ("close", 1, 0.0, 0.0), ("lift", 1, 0.0, 0.0))
    plain = RecoveryActor(prog, target)
    zero = RecoveryActor(prog, target, height_offset=0.0)
    shifted = RecoveryActor(prog, target, height_offset=-0.010)
    acts = {name: (plain.act(obs), zero.act(obs), shifted.act(obs))
            for name in ("above", "descend", "close", "lift")}
    for name, (a, b, c) in acts.items():
        assert np.array_equal(a, b), f"offset=0 changed {name}"
        if name in ("descend", "close"):
            assert not np.array_equal(a, c), f"offset ignored in {name}"
        else:
            assert np.array_equal(a, c), f"offset leaked into {name}"


def test_spec_reexport_is_object_identical():
    """Round 77: the vocabulary split re-exports, never copies. A copy would
    be value-equal yet silently break the hidden contract that EpisodeSpec's
    schedule default and every import path resolve to one object."""
    import harness.spec
    import harness.spec_tabletop

    assert harness.spec.PHASE_HEIGHT is harness.spec_tabletop.PHASE_HEIGHT
    assert harness.spec.NOMINAL_SCHEDULE is harness.spec_tabletop.NOMINAL_SCHEDULE


def test_phase_height_and_schedule_values_are_frozen():
    """test_reducers checks only key membership, so a value typo during the
    round 77 code-motion would pass the whole suite and surface only as a
    demo parity break. Pin the literals through the re-export surface."""
    from harness.spec import NOMINAL_SCHEDULE, PHASE_HEIGHT

    assert PHASE_HEIGHT == {"above": 0.10, "descend": 0.005,
                            "close": 0.005, "lift": 0.25}
    assert NOMINAL_SCHEDULE == (("above", 25), ("descend", 25),
                                ("close", 12), ("lift", 38))


# --- R2 rung B: the stack baseline policy ------------------------------------

STACK_POLICY_REF = "plugins.policies:stack_scripted_provider"


def test_stack_phase_height_and_schedule_values_are_frozen():
    """Same trap as round 77: a value typo here would pass every other test
    and surface only as a rollout parity break. Pin the literals."""
    from harness.spec import STACK_PHASE_HEIGHT, STACK_SCHEDULE

    assert STACK_PHASE_HEIGHT == {"over_b": 0.12, "place": 0.05,
                                  "release": 0.05, "retreat": 0.22}
    assert STACK_SCHEDULE == (("above", 25), ("descend", 25), ("close", 12),
                              ("lift", 30), ("over_b", 25), ("place", 25),
                              ("release", 10), ("retreat", 25))


def test_stack_scripted_driver_smoke():
    """Two t=0 targets (xy noised independently, z exact), 8-phase progression,
    grip closed exactly on {close, lift, over_b, place}, exhaustion after the
    schedule's total duration -- and all six PolicyDriver members present."""
    from harness.spec import STACK_SCHEDULE
    from plugins.policies.drivers import StackScriptedDriver, phase_at

    d = StackScriptedDriver(EpisodeSpec(seed=3, task="stack"))
    for member in ("observe_once", "act", "retarget", "on_handback",
                   "exhausted", "identity"):
        assert hasattr(d, member)
    assert d.identity == "stack_scripted@v1"

    obs = {"cubeA_pos": np.array([0.0, 0.0, 0.82]),
           "cubeB_pos": np.array([0.1, 0.1, 0.83]),
           "robot0_eef_pos": np.array([0.0, 0.0, 1.0])}
    d.observe_once(obs)
    assert d.target_a[2] == 0.82 and d.target_b[2] == 0.83     # z exact
    noise_a, noise_b = d.target_a[:2] - [0.0, 0.0], d.target_b[:2] - [0.1, 0.1]
    assert np.any(noise_a != 0.0) and np.any(noise_b != 0.0)   # xy noised
    assert not np.allclose(noise_a, noise_b)                   # independent draws

    seen, grips = [], {}
    total = sum(dur for _, dur in STACK_SCHEDULE)
    for _ in range(total):
        phase = phase_at(STACK_SCHEDULE, d.k)
        grips[phase] = d.act(obs)[-1]
        if phase not in seen:
            seen.append(phase)
    assert seen == [name for name, _ in STACK_SCHEDULE]
    assert {p for p, g in grips.items() if g == 1.0} == {"close", "lift",
                                                         "over_b", "place"}
    assert d.exhausted


def test_percept_rng_accepts_large_seeds():
    """RandomState seeds must be < 2**32, so the derived percept seed wraps.

    Regression for M6 round 107: seed=990000 (inventory_smoke's own documented
    default) made seed*7919+11 overflow and raise at the first observe_once.
    Every driver's t=0 percept routes through the one _percept_rng helper, so
    FrozenPolicy alone covers them all.
    """
    from plugins.policies.drivers import FrozenPolicy

    obs = {"cube_pos": np.array([0.0, 0.0, 0.82])}
    target = FrozenPolicy(EpisodeSpec(seed=990000)).observe_once(obs)
    assert target[2] == 0.82


def test_stack_provider_satisfies_policy_factory():
    assert isinstance(load_provider(STACK_POLICY_REF), PolicyFactory)


def test_stack_dispatch_leaves_the_lift_path_untouched():
    """The ref resolves the new driver; a plain spec still resolves the exact
    legacy class -- _default_make_driver has zero stack knowledge."""
    from plugins.policies.drivers import StackScriptedDriver

    ref_spec = EpisodeSpec(seed=0, task="stack", policy_provider=STACK_POLICY_REF)
    assert type(make_driver(ref_spec)) is StackScriptedDriver
    assert type(make_driver(EpisodeSpec(seed=0))) is ScriptedDriver


@pytest.mark.robosuite
def test_stack_env_smoke():
    """De-risks the obs['cubeB_pos'] direct-read assumption at CI time: build
    Stack once, reset, read both cubes, step one StackScriptedDriver action."""
    spec = EpisodeSpec(seed=0, task="stack", env_provider=ENV_REF,
                       policy_provider=STACK_POLICY_REF)
    env = make_env(spec)
    try:
        obs = env.reset()
        assert "cubeA_pos" in obs and "cubeB_pos" in obs
        driver = make_driver(spec)
        driver.observe_once(obs)
        env.step(driver.act(obs))
    finally:
        env.close()
