"""Round 90 rung 3: the place campaign -- prereg fields, parent seeding, and the
repair-aware tie-break brought into the campaign proposer.

Everything here is offline: the tie-break and seeding tests monkeypatch
gate._run with a seed-deterministic fake (the test_stage_attribution trick) and
use a serial executor, so the real propose_rule / run_campaign / rebuild paths
run over synthetic rollouts against real content-addressed stores. The
stack-g1 backward-compat proof skips when the sealed store is not in the
checkout (the worktree), which is fine -- the self-contained fold test pins the
same invariant without it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from plugins.rsi import gate
from plugins.rsi.campaign import (
    CampaignStore,
    Preregistration,
    _seed_from_parent,
    propose_rule,
    run_campaign,
    sha_json,
)
from plugins.rsi.governed import Bundle, RecoverySpec, Rule
from plugins.rsi.stats.search import Trigger

REPO_ROOT = Path(__file__).resolve().parent.parent
_NEW_FIELDS = ("recovery_name", "parent_store", "parent_final_sha", "reasoner",
               "segment_isolate", "horizon")


def _prereg(**kw) -> Preregistration:
    return Preregistration(dev=(0, 1), heldout=(2, 3), percept_noise=0.02,
                           critic_budget=0, action_budget=0,
                           recovery_sensor_sd=0.02, max_generations=1, **kw)


# --- the three new fields enter the hash, defaults fold out ------------------

def test_each_new_field_moves_the_prereg_hash():
    base = _prereg().sha()
    assert _prereg(recovery_name="replace").sha() != base
    assert _prereg(parent_store="/runs/stack-g1").sha() != base
    assert _prereg(parent_final_sha="deadbeef").sha() != base
    # and the defaults reproduce the default hash (folded out, not merely equal).
    assert _prereg(recovery_name="regrasp").sha() == base
    assert _prereg(parent_store=None, parent_final_sha=None).sha() == base


def test_defaults_fold_out_like_a_predating_archive():
    """The definitive backward-compat mechanism without the sealed store: a
    prereg with every folded field at default (the round-90 trio plus R7's
    reasoner) hashes to EXACTLY what an archive sealed before them (no such keys)
    would hash to."""
    p = _prereg()
    payload = dataclasses.asdict(p)
    predating = {k: v for k, v in payload.items() if k not in _NEW_FIELDS}
    assert sha_json(predating) == p.sha()


def test_stack_g1_prereg_rebuilds_to_its_sealed_sha():
    """The definitive proof against the real seal: rebuild stack-g1's archived
    preregistration and assert its sha is unchanged by the three new fields."""
    root = REPO_ROOT / "runs" / "stack-g1"
    if not root.exists():
        pytest.skip("runs/stack-g1 not present in this checkout")
    from scripts.parity_check import read_store_artifacts, rebuild_preregistration

    archived = read_store_artifacts(root)
    payload = archived["preregistration"][0]
    prereg = rebuild_preregistration(payload)
    assert prereg.sha() == sha_json(payload)  # archived index sha == rebuilt sha


# --- propose_rule: recovery_name threading + repair-aware tie-break ----------

def _peaked(n=40, N=80, span=0.035):
    """Failing (even) episodes peel finger_gap from 0.04 at t=10, widest at t=50:
    diverges at onset AND peaks late, so many arm variants tie at the top score.
    (Same fixture shape as tests/test_proposer.py's tie-break test.)"""
    from harness.spec import EpisodeSpec

    traces, labels, specs = [], [], []
    for s in range(n):
        failing = s % 2 == 0
        fg = np.full(N, 0.04)
        if failing:
            for t in range(10, N):
                diff = span * (t - 9) / 41 if t <= 50 else span * (N - t) / 30
                fg[t] = 0.04 - diff
        traces.append({
            "observable.finger_gap": fg,
            "observable.eef_z": np.linspace(1.0, 0.9, N),
            "observable.gripper_effort": np.zeros(N),
            "observable.joint_speed": np.zeros(N),
        })
        labels.append(not failing)
        specs.append(EpisodeSpec(seed=s))
    return traces, labels, specs


class _SerialExecutor:
    def __init__(self):
        self.map_calls = 0

    def map(self, fn, items, *, workers):
        self.map_calls += 1
        return [fn(item) for item in items]


def test_tiebreak_picks_max_fixed_and_threads_recovery_name(monkeypatch):
    traces, labels, dev_specs = _peaked()

    def _fake_run(job):
        # A failing dev seed is repaired ONLY by a late arm (>= 40); the tie-break
        # must reject the early-arm default the dedup would keep.
        spec, bundle = job
        failing = spec.seed % 2 == 0
        fixed = failing and bundle.rules[-1].trigger.arm_after >= 40
        return {"success": (not failing) or fixed, "fired_at": 0, "trace": {}}
    monkeypatch.setattr(gate, "_run", _fake_run)

    ex = _SerialExecutor()
    rule = propose_rule(traces, labels, generation=1,
                        prereg=_prereg(recovery_name="replace"),
                        dev_specs=dev_specs, executor=ex, workers=1)
    assert rule.trigger.arm_after >= 40, "tie-break must arm at the peak, not the onset"
    assert rule.recovery.name == "replace", "the proposed recovery must be prereg.recovery_name"
    assert ex.map_calls >= 2, "a genuine tie must have been replayed"


def test_no_dev_block_keeps_the_ranked_head_and_recovery_name(monkeypatch):
    """No dev_specs -> no replay (byte-for-byte the old ranked[0] pick), but the
    recovery name is still threaded from the prereg."""
    from plugins.rsi.stats.search import search_triggers

    traces, labels, _ = _peaked()
    monkeypatch.setattr(gate, "_run", lambda job: (_ for _ in ()).throw(
        AssertionError("no replay must run without a dev block")))
    default = search_triggers(traces, labels, privilege_budget=0, top_k=3)[0].trigger
    rule = propose_rule(traces, labels, generation=1, prereg=_prereg(recovery_name="replace"))
    assert rule.trigger == default
    assert rule.recovery.name == "replace"


def test_seeded_rule_id_does_not_collide_with_parent():
    """Round 92: the minted id is the rule's 1-indexed chain position, so a rule
    appended onto a one-rule parent (g1) is g2, and a from-scratch proposal is
    still g1 (byte-identical minting for the unseeded path)."""
    traces, labels, _ = _peaked()
    parent = Bundle(rules=(_rule("g1", 41),), critic_budget=0, action_budget=0)
    assert propose_rule(traces, labels, generation=1, prereg=_prereg(),
                        parent=parent).rule_id == "g2"
    assert propose_rule(traces, labels, generation=1, prereg=_prereg()).rule_id == "g1"


# --- run_campaign seeding from a sealed parent store ------------------------

def _rule(rule_id: str, arm_after: int) -> Rule:
    return Rule(rule_id, Trigger("observable.finger_gap", "lt", 0.0018, 1, arm_after, "value"),
                RecoverySpec(program=(("descend", 10), ("lift", 40)), sensor_sd=0.02))


def _sealed_parent(root: Path) -> tuple[Preregistration, Bundle]:
    """A run_campaign-shaped store with one promoted generation: real prereg,
    real rule canonical, real child_sha chain (critic_budget=0, like stack-g1)."""
    prereg = Preregistration(
        dev=tuple(range(41000, 41020)), heldout=tuple(range(42000, 42010)),
        percept_noise=0.02, critic_budget=0, action_budget=0,
        recovery_sensor_sd=0.02, max_generations=1, task="stack", policy="scripted")
    parent = Bundle(rules=(), critic_budget=0, action_budget=0)
    child = parent.append(_rule("g1", 41))
    store = CampaignStore(root)
    prereg_sha = store.put("preregistration", prereg._hash_payload())
    store.put("generation", {
        "preregistration_sha": prereg_sha, "generation": 1, "rule": _rule("g1", 41).canonical(),
        "parent_sha": parent.sha(), "child_sha": child.sha(),
        "dev_gate": {}, "blind_gate": {}, "promoted": True, "reason": "promoted"})
    store.put("campaign_result", {
        "preregistration_sha": prereg_sha, "generations": 1, "promoted": 1,
        "final_sha": child.sha(), "rules": ["g1"]})
    return prereg, child


def _place_prereg(store: Path, parent_final_sha: str, *, critic_budget: int = 1) -> Preregistration:
    return Preregistration(
        dev=tuple(range(41100, 41120)), heldout=tuple(range(41200, 41210)),
        percept_noise=0.02, critic_budget=critic_budget, action_budget=0, recovery_sensor_sd=0.02,
        max_generations=1, task="stack", policy="scripted", recovery_name="replace",
        parent_store=str(store), parent_final_sha=parent_final_sha)


def test_seed_from_parent_rebuilds_asserts_and_rebudgets(tmp_path):
    _prereg_parent, child = _sealed_parent(tmp_path / "stack")
    seeded = _seed_from_parent(_place_prereg(tmp_path / "stack", child.sha()), verbose=False)
    # the parent's promoted rule is carried, rebudgeted to THIS campaign's budgets.
    assert [r.canonical() for r in seeded.rules] == [r.canonical() for r in child.rules]
    assert seeded.critic_budget == 1 and seeded.action_budget == 0
    # the assertion pinned the parent at ITS budget (0); the rebudgeted bundle
    # is a fresh root, so its sha differs from the sealed parent sha.
    assert seeded.sha() != child.sha()


def test_seed_from_parent_rejects_a_sha_mismatch(tmp_path):
    _prereg_parent, _child = _sealed_parent(tmp_path / "stack")
    with pytest.raises(AssertionError, match="does not match its preregistration"):
        _seed_from_parent(_place_prereg(tmp_path / "stack", "0" * 64), verbose=False)


def test_run_campaign_grows_a_generation_onto_the_seeded_parent(tmp_path, monkeypatch):
    """End to end: the campaign seeds from the parent and its gen-1 record chains
    off the rebudgeted parent bundle, not off an empty one."""
    _prereg_parent, child = _sealed_parent(tmp_path / "stack")

    def _fake_run(job):
        # Even seeds fail ungoverned. The SEEDED parent (1 non-blind rule) leaves
        # them failing -- residual place failures -- and only the child (2 non-blind
        # rules: parent + the new place rule) fixes them, so a generation is
        # proposed, promotes, and its parent_sha is the seeded bundle.
        spec, bundle = job
        failing = spec.seed % 2 == 0
        real = 0 if bundle is None else sum(
            1 for r in bundle.rules if not r.rule_id.endswith("-blind"))
        return {"success": (not failing) or real >= 2,
                "fired_at": 0 if real and failing else None,
                "trace": {"observable.finger_gap": np.full(60, 0.001 if failing else 0.04),
                          "observable.eef_z": np.linspace(1.0, 0.9, 60),
                          "observable.gripper_effort": np.zeros(60),
                          "observable.joint_speed": np.zeros(60)}}
    monkeypatch.setattr(gate, "_run", _fake_run)

    # critic_budget=0 here: this test proves the parent_sha CHAIN (budget-
    # independent) over an observable-only fake trace; the budget-1 rebudget is
    # proven by test_seed_from_parent_rebuilds and privileged search by the smoke.
    place = _place_prereg(tmp_path / "stack", child.sha(), critic_budget=0)
    store = CampaignStore(tmp_path / "place")
    seeded = _seed_from_parent(place, verbose=False)  # what run_campaign starts from
    from plugins.reasoner import provider
    response = tmp_path / "model-reply.json"
    response.write_text(json.dumps({"feature": "observable.finger_gap", "op": "lt", "threshold": 0.02,
                                    "dwell": 1, "arm_after": 10, "reducer": "value", "recovery": "replace"}))
    monkeypatch.setenv("PH_MODEL_ENDPOINT_FAKE", str(response))
    run_campaign(place, store, workers=1, verbose=False, executor=_SerialExecutor(), reasoner=provider())

    index = [json.loads(line) for line in store.index_path.open()]
    gen = next(store.read(r["sha"]) for r in index if r["kind"] == "generation")
    # gen1's parent is the SEEDED (rebudgeted) bundle, carrying the parent's rule.
    assert gen["parent_sha"] == seeded.sha()
    assert gen["rule"]["recovery"]["name"] == "replace"
    # and its minted id does not collide with the parent's g1 (round 92).
    assert gen["rule"]["rule_id"] == "g2"

    # Reproduction round-trip (place-g2's blocker): rebuild_final_bundle must SEED
    # from the parent, else the first child_sha assertion fails on a seeded store.
    from plugins.rsi.rebuild import read_store_artifacts, rebuild_final_bundle
    archived = read_store_artifacts(store.root)
    rebuilt = rebuild_final_bundle(place, archived["generation"])
    result = next(store.read(r["sha"]) for r in index if r["kind"] == "campaign_result")
    assert rebuilt.sha() == result["final_sha"], "seeded store must rebuild to its sealed final"


# --- the script's prereg + smoke CLI ----------------------------------------

def test_place_prereg_pins_the_parent_and_recovery():
    from scripts.place_campaign import PARENT_FINAL_SHA, PARENT_STORE, place_prereg

    p = place_prereg()
    assert p.recovery_name == "replace"
    assert p.critic_budget == 1 and p.action_budget == 0
    assert p.parent_store == PARENT_STORE
    assert p.parent_final_sha == PARENT_FINAL_SHA
    # round-92 rematch (place-g2): fresh blocks, v1's 46000-46266 / 47000-47199 burned.
    assert p.dev == tuple(range(46267, 47000))
    assert p.heldout == tuple(range(47200, 47400))
    assert p.sha()  # constructs and hashes cleanly


def test_place_campaign_help_smoke():
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "scripts/place_campaign.py", "--help"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0
    assert "--smoke" in proc.stdout and "--out" in proc.stdout
