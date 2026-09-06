"""Unit tests for the RSI workload plugin and scripts/parity_check.py's pure parts.

No simulation anywhere in this file. `plugins.rsi.campaign.run_campaign` is
monkeypatched throughout the workload section below, so no episode, env, or
policy driver is ever built. The parity-check section only exercises
`read_store_artifacts` / `rebuild_preregistration` / `compare_generations` /
`compare_heldout` -- plain file I/O and dict comparison against small,
hand-built store fixtures -- never `run_parity_check` itself, which drives a
real campaign and is explicitly the orchestrator's job to run.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from harness.config import sha_json
from harness.definitions import CAPABILITIES
from harness.events import SessionLog
from harness.fakes import FakeReasoner
from harness.kernel import Kernel
from plugins.graphs import InMemorySkillGraph
from plugins.rsi import campaign, workload
from plugins.rsi.campaign import CampaignStore, Preregistration
from scripts import parity_check

REPO_ROOT = Path(__file__).resolve().parent.parent


def _prereg(**overrides) -> Preregistration:
    kwargs = {"dev": tuple(range(20)), "heldout": tuple(range(20, 30)),
             "percept_noise": 0.02, "critic_budget": 0, "action_budget": 0,
             "recovery_sensor_sd": 0.02, "max_generations": 2,
             "task": "lift", "policy": "scripted"}
    kwargs.update(overrides)
    return Preregistration(**kwargs)


# =====================================================================
# plugins.rsi.workload.run
# =====================================================================

class _FakeEnvProvider:
    """Satisfies harness.contracts.EnvProvider; must never actually be called."""

    def make_env(self, spec):
        raise AssertionError("make_env must not run: run_campaign is monkeypatched")

    def tasks(self):
        return ("lift",)

    def object_key(self, spec):
        return "cube_pos"

    def success(self, obs, spec, start_z):
        return False

class _FakePolicyFactory:
    """Satisfies harness.contracts.PolicyFactory; must never actually be called."""

    def make_driver(self, spec):
        raise AssertionError("make_driver must not run: run_campaign is monkeypatched")


class _FakePercept:
    def object_estimate(self, obs, spec, sensor_sd, draw):
        return None


class _FakeExecutor:
    def map(self, fn, items, *, workers):
        return [fn(item) for item in items]


def _kernel_with_fakes() -> Kernel:
    k = Kernel(CAPABILITIES)
    k.provide("embodiment.env", _FakeEnvProvider(), ref="tests.fakes:env")
    k.provide("policy.driver", _FakePolicyFactory(), ref="tests.fakes:policy")
    k.provide("percept.model", _FakePercept(), ref="tests.fakes:percept")
    k.provide("exec.rollouts", _FakeExecutor(), ref="tests.fakes:executor")
    k.provide("graph.skill", InMemorySkillGraph(), ref="plugins.graphs:skill_graph_provider")
    # Even a test reasoner without identity must be forwarded explicitly.
    k.provide("reasoner.proposer", FakeReasoner(), ref="harness.fakes:reasoner_provider")
    return k


#: A canonical rule (Rule.canonical()'s shape: rule_id/trigger/recovery) the
#: fakes below write into fake generation artifacts, mirroring exactly what
#: governor/campaign.py's own run_campaign stores (verified against a real
#: archived generation artifact in runs/campaign-pj-scripted).
_TRIGGER = {"feature": "observable.finger_gap", "op": "lt", "threshold": 0.0018,
           "dwell": 1, "arm_after": 41, "reducer": "value"}
_RECOVERY = {"name": "regrasp", "strategy": "regrasp",
            "program": [["descend", 10, 0.0, 0.0]], "sensor_sd": 0.02,
            "max_invocations": 1}
_DEV_GATE = {"n": 10, "base_rate": 0.5, "governed_rate": 0.9, "fixed": 4,
            "broken": 0, "fires": 4, "p_value": 0.01, "declared_privilege": 0}
_HELDOUT = {"n": 20, "base_rate": 0.4, "governed_rate": 0.8, "fixed": 8,
           "broken": 0, "fires": 8, "p_value": 0.001, "declared_privilege": 0}
_ABLATION = [[0.0, _HELDOUT], [0.02, _HELDOUT]]

#: Sentinel distinguishing "omit heldout_vs_blind entirely" (require_judgement
#: off: campaign.py never writes the key) from an explicit value including None.
_OMIT = object()


def _fake_run_campaign_factory(captured: dict, *, second_generation_promoted: bool | None,
                               heldout_vs_blind=_HELDOUT):
    """Build a `run_campaign` fake that writes campaign-store-shaped artifacts.

    Generation 1 is always promoted. `second_generation_promoted=None` omits a
    second generation entirely; `False` writes a REJECTED second generation
    (proving a rejected generation contributes no skill); `True` writes a
    second PROMOTED generation with a distinct trigger (proving two promotions
    produce two skills). `heldout_vs_blind=_OMIT` drops the key from the
    result, mirroring a require_judgement=False campaign.
    """
    def fake_run_campaign(prereg, store, *, workers=10, verbose=True, executor=None,
                          reasoner=None):
        captured["prereg"] = prereg
        captured["store"] = store
        captured["reasoner"] = reasoner
        # Mirror the real run_campaign: the sealed prereg payload is _hash_payload
        # (round-90 fields fold out at their defaults), so prereg_sha == prereg.sha().
        prereg_sha = store.put("preregistration", prereg._hash_payload())

        rule_g1 = {"rule_id": "g1", "trigger": _TRIGGER, "recovery": _RECOVERY}
        store.put("generation", {
            "preregistration_sha": prereg_sha, "generation": 1, "rule": rule_g1,
            "parent_sha": "parent0", "child_sha": "child1", "dev_gate": _DEV_GATE,
            "blind_gate": _DEV_GATE, "promoted": True, "reason": "promoted",
        })
        rules = ["g1"]
        n_generations = 1

        if second_generation_promoted is not None:
            n_generations = 2
            rule_g2 = {"rule_id": "g2", "trigger": {**_TRIGGER, "arm_after": 58},
                      "recovery": _RECOVERY}
            store.put("generation", {
                "preregistration_sha": prereg_sha, "generation": 2, "rule": rule_g2,
                "parent_sha": "child1", "child_sha": "child2", "dev_gate": _DEV_GATE,
                "blind_gate": _DEV_GATE, "promoted": bool(second_generation_promoted),
                "reason": "promoted" if second_generation_promoted else "rejected (fixed=0 broken=1)",
            })
            if second_generation_promoted:
                rules.append("g2")

        result = {
            "preregistration_sha": prereg_sha, "power_plans": [], "generations": n_generations,
            "promoted": len(rules), "final_sha": rules[-1], "rules": rules,
            "heldout": _HELDOUT, "ablation": _ABLATION,
        }
        if heldout_vs_blind is not _OMIT:
            result["heldout_vs_blind"] = heldout_vs_blind
        store.put("campaign_result", result)
        return result

    return fake_run_campaign


def test_provider_refs_are_threaded_onto_the_preregistration(tmp_path, monkeypatch):
    kernel = _kernel_with_fakes()
    captured: dict = {}
    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory(captured, second_generation_promoted=None))

    workload.run(_prereg(), tmp_path / "store", kernel, workers=2, verbose=False)

    assert captured["prereg"].env_provider == "tests.fakes:env"
    assert captured["prereg"].policy_provider == "tests.fakes:policy"
    assert captured["prereg"].env_provider == kernel.provider_ref("embodiment.env")
    assert captured["prereg"].policy_provider == kernel.provider_ref("policy.driver")
    # everything else about the preregistration is untouched by the threading
    assert captured["prereg"].task == "lift" and captured["prereg"].policy == "scripted"


def test_kernel_resolutions_are_accounted_under_consumer_rsi(tmp_path, monkeypatch):
    kernel = _kernel_with_fakes()
    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory({}, second_generation_promoted=None))

    workload.run(_prereg(), tmp_path / "store", kernel, workers=2, verbose=False)

    resolved = {r.capability: r.consumer for r in kernel.resolutions()}
    assert resolved == {"embodiment.env": "rsi", "policy.driver": "rsi",
                        "percept.model": "rsi", "exec.rollouts": "rsi",
                        "graph.skill": "rsi", "reasoner.proposer": "rsi"}
    assert not any(r.privileged for r in kernel.resolutions())


def test_promoted_rule_publishes_exactly_one_skill_with_matching_preconditions(tmp_path, monkeypatch):
    """One promoted generation, one rejected generation -> exactly one skill."""
    kernel = _kernel_with_fakes()
    captured: dict = {}
    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory(captured, second_generation_promoted=False))

    prereg = _prereg()
    out = workload.run(prereg, tmp_path / "store", kernel, workers=2, verbose=False)

    assert len(out["skills"]) == 1
    skills = kernel.resolve("graph.skill", consumer="test").skills()
    assert len(skills) == 1
    skill = skills[0]
    assert skill["kind"] == "grasp_recovery"
    assert skill["policy"] == prereg.policy
    assert skill["task"] == prereg.task
    assert skill["preconditions"] == _TRIGGER
    assert skill["recovery"] == _RECOVERY
    # Round 54 review fix: a rule's own effect is its dev gate vs parent; the
    # held-out numbers, judgement and ablation are bundle-level evidence shared
    # by the whole chain, and are labelled as such.
    assert skill["effects"] == {"dev_gate_vs_parent": skill["effects"]["dev_gate_vs_parent"]}
    assert skill["bundle_evidence"]["heldout"] == _HELDOUT
    assert skill["bundle_evidence"]["judgement"] == _HELDOUT
    assert skill["bundle_evidence"]["ablation"] == _ABLATION
    assert skill["bundle_evidence"]["declared_privilege"] == 0
    assert "not attributable to this rule alone" in skill["bundle_evidence"]["note"]
    assert skill["prereg_sha"] == captured["prereg"].sha()
    assert "mount_plan_sha" not in skill, "no session log on this kernel -- must be omitted"


def test_skill_established_true(tmp_path, monkeypatch):
    """heldout_vs_blind clears p<alpha & fixed>broken -> verdict True."""
    kernel = _kernel_with_fakes()
    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory({}, second_generation_promoted=None))

    workload.run(_prereg(), tmp_path / "store", kernel, workers=2, verbose=False)

    skill = kernel.resolve("graph.skill", consumer="test").skills()[0]
    assert skill["heldout_judgement_established"] is True


def test_skill_established_false(tmp_path, monkeypatch):
    """Judgement ran but shrank to non-significant -> verdict False, not None."""
    kernel = _kernel_with_fakes()
    shrunk = dict(_HELDOUT, p_value=0.2, fixed=3, broken=3)
    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory({}, second_generation_promoted=None,
                                                   heldout_vs_blind=shrunk))

    workload.run(_prereg(), tmp_path / "store", kernel, workers=2, verbose=False)

    skill = kernel.resolve("graph.skill", consumer="test").skills()[0]
    assert skill["heldout_judgement_established"] is False


def test_skill_judgement_none_when_not_run(tmp_path, monkeypatch):
    """require_judgement off (no heldout_vs_blind key): the field is still
    PRESENT, with an explicit None -- untested is not the same as failed, and
    the always-present contract is what makes it first-class (contrast
    mount_plan_sha, which is omitted when unknown)."""
    kernel = _kernel_with_fakes()
    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory({}, second_generation_promoted=None,
                                                   heldout_vs_blind=_OMIT))

    workload.run(_prereg(), tmp_path / "store", kernel, workers=2, verbose=False)

    skill = kernel.resolve("graph.skill", consumer="test").skills()[0]
    assert "heldout_judgement_established" in skill
    assert skill["heldout_judgement_established"] is None


def test_returned_digests_match_graph_skills(tmp_path, monkeypatch):
    kernel = _kernel_with_fakes()
    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory({}, second_generation_promoted=True))

    out = workload.run(_prereg(), tmp_path / "store", kernel, workers=2, verbose=False)

    skills = kernel.resolve("graph.skill", consumer="test").skills()
    assert len(out["skills"]) == 2
    assert len(skills) == 2
    assert set(out["skills"]) == {sha_json(s) for s in skills}
    assert len(set(out["skills"])) == 2, "two distinct rules must not collide to one digest"


def test_no_promoted_generations_publishes_nothing(tmp_path, monkeypatch):
    kernel = _kernel_with_fakes()

    def fake_run_campaign(prereg, store, *, workers=10, verbose=True, executor=None,
                          reasoner=None):
        prereg_sha = store.put("preregistration", dataclasses.asdict(prereg))
        store.put("generation", {
            "preregistration_sha": prereg_sha, "generation": 1,
            "rule": {"rule_id": "g1", "trigger": _TRIGGER, "recovery": _RECOVERY},
            "parent_sha": "parent0", "child_sha": "child1", "dev_gate": _DEV_GATE,
            "blind_gate": _DEV_GATE, "promoted": False, "reason": "rejected (fixed=0 broken=1)",
        })
        # bundle.rules stays empty: run_campaign omits "heldout"/"ablation" entirely.
        result = {"preregistration_sha": prereg_sha, "power_plans": [], "generations": 1,
                  "promoted": 0, "final_sha": "parent0", "rules": []}
        store.put("campaign_result", result)
        return result

    monkeypatch.setattr(campaign, "run_campaign", fake_run_campaign)

    out = workload.run(_prereg(), tmp_path / "store", kernel, workers=2, verbose=False)

    assert out["skills"] == []
    assert kernel.resolve("graph.skill", consumer="test").skills() == ()


def test_mount_plan_sha_omitted_with_no_kernel_log(tmp_path, monkeypatch):
    kernel = _kernel_with_fakes()
    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory({}, second_generation_promoted=None))

    workload.run(_prereg(), tmp_path / "store", kernel, workers=2, verbose=False)

    skill = kernel.resolve("graph.skill", consumer="test").skills()[0]
    assert "mount_plan_sha" not in skill


def test_mount_plan_sha_present_when_kernel_mounted_a_plan(tmp_path, monkeypatch):
    """Real plugin adapters, mounted through Kernel.mount -- still no simulation:
    run_campaign is monkeypatched below, so nothing ever calls make_env/make_driver.
    """
    from harness.config import Mount, Profile, resolve_plan

    log = SessionLog()
    kernel = Kernel(CAPABILITIES, log=log)
    plan = resolve_plan(Profile("base", (
        Mount("embodiment.env", "plugins.embodiment_robosuite:provider"),
        Mount("policy.driver", "plugins.policies:provider"),
        Mount("percept.model", "plugins.embodiment_robosuite.percept:provider"),
        Mount("exec.rollouts", "harness.executor:provider"),
        Mount("graph.skill", "plugins.graphs:skill_graph_provider"),
        Mount("reasoner.proposer", "harness.fakes:reasoner_provider"),
    )))
    kernel.mount(plan)

    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory({}, second_generation_promoted=None))
    workload.run(_prereg(), tmp_path / "store", kernel, workers=2, verbose=False)

    skill = kernel.resolve("graph.skill", consumer="test").skills()[0]
    assert skill["mount_plan_sha"] == plan.sha()


def test_mount_plan_sha_is_path_portable(tmp_path, monkeypatch):
    """The frontier item: a skill record's mount_plan_sha must not carry the
    --out path. Two runs identical except for where skills persist publish
    byte-identical mount_plan_sha, so the record travels between archives."""
    from harness.config import Mount, Profile, resolve_plan

    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory({}, second_generation_promoted=None))

    def sha_for_out(out: Path) -> str:
        kernel = Kernel(CAPABILITIES, log=SessionLog())
        kernel.mount(resolve_plan(Profile("base", (
            Mount("embodiment.env", "plugins.embodiment_robosuite:provider"),
            Mount("policy.driver", "plugins.policies:provider"),
            Mount("percept.model", "plugins.embodiment_robosuite.percept:provider"),
            Mount("exec.rollouts", "harness.executor:provider"),
            Mount("graph.skill", "plugins.graphs:skill_graph_provider",
                  {"root": str(out / "skills")}),
            Mount("reasoner.proposer", "harness.fakes:reasoner_provider"),
        ))))
        workload.run(_prereg(), out / "store", kernel, workers=2, verbose=False)
        return kernel.resolve("graph.skill", consumer="test").skills()[0]["mount_plan_sha"]

    assert sha_for_out(tmp_path / "run-a") == sha_for_out(tmp_path / "run-b")


def test_reused_store_root_does_not_resurface_a_prior_runs_promotions(tmp_path, monkeypatch):
    """CampaignStore is append-only; a second run into the same store_root must
    publish only ITS OWN promotions, not the first run's too."""
    kernel = _kernel_with_fakes()
    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory({}, second_generation_promoted=None))
    store_root = tmp_path / "shared-store"

    first = workload.run(_prereg(), store_root, kernel, workers=2, verbose=False)
    second = workload.run(_prereg(), store_root, kernel, workers=2, verbose=False)

    assert len(first["skills"]) == 1
    assert len(second["skills"]) == 1, "the second call must not also re-publish the first run's rule"


# --- white-box coverage of the two module-private helpers ------------------

def test_promoted_generation_records_filters_and_orders(tmp_path):
    store = CampaignStore(tmp_path / "store")
    store.put("generation", {"generation": 2, "promoted": True, "rule": "r2"})
    store.put("generation", {"generation": 1, "promoted": True, "rule": "r1"})
    store.put("generation", {"generation": 3, "promoted": False, "rule": "r3"})
    store.put("preregistration", {"anything": True})  # non-generation kind, ignored

    records = workload._promoted_generation_records(store)
    assert [r["rule"] for r in records] == ["r1", "r2"]


def test_promoted_generation_records_respects_min_seq(tmp_path):
    store = CampaignStore(tmp_path / "store")
    store.put("generation", {"generation": 1, "promoted": True, "rule": "old"})
    cutoff = store.size()
    store.put("generation", {"generation": 1, "promoted": True, "rule": "new"})

    assert [r["rule"] for r in workload._promoted_generation_records(store)] == ["old", "new"]
    assert [r["rule"] for r in workload._promoted_generation_records(store, min_seq=cutoff)] == ["new"]


def test_promoted_generation_records_empty_store(tmp_path):
    store = CampaignStore(tmp_path / "empty-store")
    assert workload._promoted_generation_records(store) == []


def test_mount_plan_sha_none_when_kernel_has_no_log():
    kernel = Kernel(CAPABILITIES)
    assert workload._mount_plan_sha(kernel) is None


def test_mount_plan_sha_reads_the_most_recent_kernel_mount_event():
    log = SessionLog()
    kernel = Kernel(CAPABILITIES, log=log)
    log.append("capability.provide", {"capability": "graph.skill"})  # unrelated row
    log.append("kernel.mount", {"plan_sha": "sha-one", "capabilities": []})
    log.append("kernel.mount", {"plan_sha": "sha-two", "capabilities": []})
    assert workload._mount_plan_sha(kernel) == "sha-two"


# =====================================================================
# scripts/parity_check.py -- pure parts only (no simulation, no kernel)
# =====================================================================

def _write_store(root: Path, prereg_payload: dict, generations: list[dict],
                 campaign_result: dict | None) -> CampaignStore:
    store = CampaignStore(root)
    store.put("preregistration", prereg_payload)
    for gen in generations:
        store.put("generation", gen)
    if campaign_result is not None:
        store.put("campaign_result", campaign_result)
    return store


def test_parity_check_module_imports_cleanly():
    """Importing the module (done at collection time via the top-level import
    above) must not touch robosuite, a kernel, or a campaign -- confirmed here
    by checking its public surface exists rather than re-importing."""
    assert callable(parity_check.main)
    assert callable(parity_check.run_parity_check)


def test_read_store_artifacts_groups_by_kind_in_index_order(tmp_path):
    store = _write_store(
        tmp_path / "store", {"task": "lift"},
        [{"generation": 1, "rule": "r1"}, {"generation": 2, "rule": "r2"}],
        {"rules": ["g1"]},
    )
    by_kind = parity_check.read_store_artifacts(store.root)
    assert [g["generation"] for g in by_kind["generation"]] == [1, 2]
    assert by_kind["preregistration"] == [{"task": "lift"}]
    assert by_kind["campaign_result"] == [{"rules": ["g1"]}]


def test_read_store_artifacts_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parity_check.read_store_artifacts(tmp_path / "does-not-exist")


def test_rebuild_preregistration_round_trips_and_restores_tuples():
    original = _prereg(dev=tuple(range(5)), heldout=tuple(range(5, 8)))
    payload = json.loads(json.dumps(dataclasses.asdict(original), default=str))
    rebuilt = parity_check.rebuild_preregistration(payload)
    assert rebuilt == original
    assert isinstance(rebuilt.dev, tuple) and isinstance(rebuilt.heldout, tuple)


def test_rebuild_preregistration_drops_fields_the_dataclass_no_longer_has():
    original = _prereg()
    payload = dataclasses.asdict(original)
    payload["a_field_this_dataclass_no_longer_has"] = "legacy"
    assert parity_check.rebuild_preregistration(payload) == original


def test_rebuild_preregistration_restores_stage_specs():
    """A staged archive (runs/stack-g1) stores the chain as dicts; the rollout
    consumes StageSpec attributes, so the rebuild must restore the dataclasses."""
    from harness.stages import Clause, StageSpec

    original = _prereg(stages=(
        StageSpec("grasp", (Clause("observable.finger_gap", "gt", 0.01),), 62),))
    payload = json.loads(json.dumps(dataclasses.asdict(original), default=str))
    rebuilt = parity_check.rebuild_preregistration(payload)
    assert rebuilt == original
    assert isinstance(rebuilt.stages[0], StageSpec)
    assert isinstance(rebuilt.stages[0].success[0], Clause)


def test_rebuild_preregistration_defaults_provider_refs_for_pre_seam_archives():
    """runs/campaign-pj-scripted's real preregistration artifact predates
    env_provider/policy_provider and has no such keys at all."""
    payload = dataclasses.asdict(_prereg())
    del payload["env_provider"]
    del payload["policy_provider"]
    rebuilt = parity_check.rebuild_preregistration(payload)
    assert rebuilt.env_provider is None
    assert rebuilt.policy_provider is None


def test_compare_generations_all_match():
    gens = [{"generation": 1, "rule": {"rule_id": "g1"}, "dev_gate": _DEV_GATE, "promoted": True}]
    rule_r, gate_r, promo_r = parity_check.compare_generations(gens, gens)
    assert rule_r.ok and gate_r.ok and promo_r.ok


def test_compare_generations_detects_a_rule_mismatch():
    old = [{"generation": 1, "rule": {"rule_id": "g1"}, "dev_gate": _DEV_GATE, "promoted": True}]
    new = [{"generation": 1, "rule": {"rule_id": "g1-different"}, "dev_gate": _DEV_GATE,
           "promoted": True}]
    rule_r, gate_r, promo_r = parity_check.compare_generations(old, new)
    assert not rule_r.ok and rule_r.detail == "generations [1] differ"
    assert gate_r.ok and promo_r.ok


def test_compare_generations_detects_a_dev_gate_mismatch():
    old = [{"generation": 1, "rule": {"rule_id": "g1"}, "dev_gate": _DEV_GATE, "promoted": True}]
    new_gate = dict(_DEV_GATE, fixed=_DEV_GATE["fixed"] + 1)
    new = [{"generation": 1, "rule": {"rule_id": "g1"}, "dev_gate": new_gate, "promoted": True}]
    rule_r, gate_r, promo_r = parity_check.compare_generations(old, new)
    assert rule_r.ok and promo_r.ok
    assert not gate_r.ok and gate_r.detail == "generations [1] differ"


def test_compare_generations_detects_a_promotion_mismatch():
    old = [{"generation": 1, "rule": {"rule_id": "g1"}, "dev_gate": _DEV_GATE, "promoted": True}]
    new = [{"generation": 1, "rule": {"rule_id": "g1"}, "dev_gate": _DEV_GATE, "promoted": False}]
    rule_r, gate_r, promo_r = parity_check.compare_generations(old, new)
    assert rule_r.ok and gate_r.ok
    assert not promo_r.ok and promo_r.detail == "generations [1] differ"


def test_compare_generations_detects_a_generation_count_mismatch():
    old = [{"generation": 1, "rule": {}, "dev_gate": _DEV_GATE, "promoted": True}]
    rule_r, gate_r, promo_r = parity_check.compare_generations(old, [])
    assert not rule_r.ok and not gate_r.ok and not promo_r.ok
    assert "count differs" in rule_r.detail


def test_compare_heldout_matches():
    assert parity_check.compare_heldout(_HELDOUT, dict(_HELDOUT)).ok


def test_compare_heldout_detects_a_fixed_mismatch():
    other = dict(_HELDOUT, fixed=_HELDOUT["fixed"] + 1)
    result = parity_check.compare_heldout(_HELDOUT, other)
    assert not result.ok and "fixed" in result.detail


def test_compare_heldout_detects_a_rate_drift():
    """Rates are compared directly now, not only through the derived delta."""
    other = dict(_HELDOUT, governed_rate=_HELDOUT["governed_rate"] + 0.05)
    result = parity_check.compare_heldout(_HELDOUT, other)
    assert not result.ok and "governed_rate" in result.detail


def test_compare_heldout_detects_symmetric_drift():
    """Both rates shifting together is the realistic signature of an env or
    RNG parity break: fixed/broken and delta can all agree while the block
    baseline itself moved. The direct rate comparison catches it."""
    other = dict(_HELDOUT,
                 base_rate=_HELDOUT["base_rate"] + 0.05,
                 governed_rate=_HELDOUT["governed_rate"] + 0.05)
    result = parity_check.compare_heldout(_HELDOUT, other)
    assert not result.ok
    assert "base_rate" in result.detail and "governed_rate" in result.detail


def test_compare_heldout_both_none_passes():
    assert parity_check.compare_heldout(None, None).ok


def test_compare_heldout_one_none_fails():
    assert not parity_check.compare_heldout(_HELDOUT, None).ok
    assert not parity_check.compare_heldout(None, _HELDOUT).ok


# --- sanity checks against the real archived campaign, if present ----------

def test_read_store_artifacts_reads_the_real_archived_campaign():
    root = REPO_ROOT / "runs" / "campaign-pj-scripted"
    if not root.exists():
        pytest.skip("runs/campaign-pj-scripted not present in this checkout")
    by_kind = parity_check.read_store_artifacts(root)
    assert len(by_kind["preregistration"]) == 1
    assert len(by_kind["generation"]) == 2
    assert len(by_kind["campaign_result"]) == 1
    assert [g["generation"] for g in by_kind["generation"]] == [1, 2]

    prereg = parity_check.rebuild_preregistration(by_kind["preregistration"][0])
    assert prereg.task == "lift"
    assert prereg.policy == "scripted"
    assert prereg.env_provider is None
    assert prereg.policy_provider is None


def test_the_real_archive_compared_against_itself_is_all_pass():
    root = REPO_ROOT / "runs" / "campaign-pj-scripted"
    if not root.exists():
        pytest.skip("runs/campaign-pj-scripted not present in this checkout")
    by_kind = parity_check.read_store_artifacts(root)
    gen_results = parity_check.compare_generations(by_kind["generation"], by_kind["generation"])
    heldout = by_kind["campaign_result"][0]["heldout"]
    heldout_result = parity_check.compare_heldout(heldout, heldout)
    assert all(r.ok for r in (*gen_results, heldout_result))


def test_campaign_completion_enters_the_session_chain(tmp_path, monkeypatch):
    """L1 rung 3: the campaign's outcome and the mounts that produced it sit in
    ONE verifiable ledger."""
    log = SessionLog()
    kernel = Kernel(CAPABILITIES, log=log)
    kernel.provide("embodiment.env", _FakeEnvProvider(), ref="tests.fakes:env")
    kernel.provide("policy.driver", _FakePolicyFactory(), ref="tests.fakes:policy")
    kernel.provide("percept.model", _FakePercept(), ref="tests.fakes:percept")
    kernel.provide("exec.rollouts", _FakeExecutor(), ref="tests.fakes:executor")
    kernel.provide("graph.skill", InMemorySkillGraph(), ref="plugins.graphs:skill_graph_provider")
    kernel.provide("reasoner.proposer", FakeReasoner(), ref="harness.fakes:reasoner_provider")
    captured: dict = {}
    monkeypatch.setattr(campaign, "run_campaign",
                        _fake_run_campaign_factory(captured, second_generation_promoted=False))

    out = workload.run(_prereg(), tmp_path / "store", kernel, workers=2, verbose=False)

    rows = [r for r in log.rows() if r["kind"] == "rsi.campaign_complete"]
    assert len(rows) == 1
    assert rows[0]["data"]["skills"] == out["skills"]
    assert rows[0]["data"]["prereg_sha"] == out["result"]["preregistration_sha"]
    resolve_kinds = [r["kind"] for r in log.rows()]
    assert resolve_kinds.index("capability.resolve") < resolve_kinds.index("rsi.campaign_complete"), \
        "the completion must sit downstream of the resolutions in the same chain"
    assert log.verify(), "the combined ledger no longer verifies"


def test_workload_forwards_identityless_reasoner_without_selecting_an_alternative(tmp_path, monkeypatch):
    kernel = _kernel_with_fakes()
    expected = kernel.resolve("reasoner.proposer", consumer="test")
    captured = {}
    monkeypatch.setattr(campaign, "run_campaign",
        _fake_run_campaign_factory(captured, second_generation_promoted=None))
    workload.run(_prereg(), tmp_path / "store", kernel, workers=1, verbose=False)
    assert captured["reasoner"] is expected
