"""The mounted LLM reasoner owns proposals; no implicit search path exists.

Tests use the same registered fake endpoint as model-backed runtime tests and
keep the real parser, campaign gates, artifact store, and workload wiring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import json
import pytest

from harness.manifest import discover
from plugins.rsi import gate
from plugins.rsi.campaign import (
    CampaignStore,
    Preregistration,
    run_campaign,
)
from scripts.plugin_doctor import check

_REPO = Path(__file__).resolve().parent.parent
_PROPOSAL = {"feature": "observable.finger_gap", "op": "lt", "threshold": 0.02,
             "dwell": 1, "arm_after": 10, "reducer": "value", "recovery": "regrasp"}


@pytest.fixture(autouse=True)
def model_reply(tmp_path, monkeypatch):
    path = tmp_path / "model-reply.json"
    path.write_text(json.dumps(_PROPOSAL))
    monkeypatch.setenv("PH_MODEL_ENDPOINT_FAKE", str(path))
    return path


class _SerialExecutor:
    def map(self, fn, items, *, workers):
        return [fn(item) for item in items]


def _fake_run(job):
    """Even seeds fail ungoverned and are repaired by any armed rule; peaked
    finger_gap traces give the search a candidate. No simulator, no episodes."""
    spec, bundle = job
    failing = spec.seed % 2 == 0
    real = 0 if bundle is None else sum(
        1 for r in bundle.rules if not r.rule_id.endswith("-blind"))
    fg = np.full(60, 0.04)
    if failing:
        for t in range(10, 60):
            fg[t] = 0.04 - 0.035 * min(t - 9, 40) / 40
    return {"success": (not failing) or real >= 1,
            "fired_at": 0 if real and failing else None,
            "trace": {"observable.finger_gap": fg,
                      "observable.eef_z": np.linspace(1.0, 0.9, 60),
                      "observable.gripper_effort": np.zeros(60),
                      "observable.joint_speed": np.zeros(60)}}


def _prereg(**kw):
    base = {"dev": tuple(range(90000, 90040)), "heldout": tuple(range(90100, 90140)),
            "percept_noise": 0.02, "critic_budget": 0, "action_budget": 0,
            "recovery_sensor_sd": 0.02, "max_generations": 1,
            "task": "lift", "policy": "scripted"}
    base.update(kw)
    return Preregistration(**base)


class _RecordingReasoner:
    """Record the live seam brief and delegate to the actual model adapter."""

    def __init__(self, identity=None):
        from plugins.reasoner import provider
        self.inner = provider()
        self.briefs = []
        self.identity = identity or self.inner.identity

    def propose(self, brief):
        self.briefs.append(brief)
        return self.inner.propose(brief)


# --- (1) the seam is live: the mounted reasoner is invoked with the brief ----

def test_run_campaign_invokes_the_mounted_reasoner_with_the_brief(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_run", _fake_run)
    reasoner = _RecordingReasoner()
    store = CampaignStore(tmp_path / "c")
    run_campaign(_prereg(), store, workers=1, verbose=False,
                 executor=_SerialExecutor(), reasoner=reasoner)
    assert reasoner.briefs, "the mounted reasoner was never called -- the seam is dead"
    brief = reasoner.briefs[0]
    assert brief["generation"] == 1
    assert set(brief) >= {"traces", "labels", "generation", "prereg",
                          "dev_specs", "executor", "workers", "parent", "store"}
    assert len(brief["traces"]) == len(brief["labels"]) == 40


def test_campaign_requires_an_explicit_reasoner_before_any_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_run", lambda _: pytest.fail("experiment must not run"))
    with pytest.raises(ValueError, match="explicit LLM reasoner"):
        run_campaign(_prereg(), CampaignStore(tmp_path / "a"), verbose=False)


def test_campaign_refuses_search_recovery_before_model_or_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_run", lambda _: pytest.fail("experiment must not run"))
    reasoner = _RecordingReasoner()
    with pytest.raises(ValueError, match="search_recovery.*LLM-only"):
        run_campaign(_prereg(search_recovery=True), CampaignStore(tmp_path / "a"),
                     verbose=False, reasoner=reasoner)
    assert reasoner.briefs == []


@pytest.mark.parametrize("identity", [None, "", "   "])
def test_campaign_refuses_unattributed_reasoners_before_experiments(identity, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_run", lambda _: pytest.fail("experiment must not run"))
    reasoner = _RecordingReasoner()
    reasoner.identity = identity
    with pytest.raises(ValueError, match="non-empty reasoner.identity"):
        run_campaign(_prereg(), CampaignStore(tmp_path / "a"), verbose=False, reasoner=reasoner)


class _Env:
    def make_env(self, spec):
        raise AssertionError("make_env must not run: gate._run is faked")

    def tasks(self):
        return ("lift",)

    def object_key(self, spec):
        return "cube_pos"

    def success(self, obs, spec, start_z):
        return False


class _Policy:
    def make_driver(self, spec):
        raise AssertionError("make_driver must not run: gate._run is faked")


class _Percept:
    def object_estimate(self, obs, spec, sensor_sd, draw):
        return None


def _workload_kernel(reasoner_obj, reasoner_ref):
    from harness.definitions import CAPABILITIES
    from harness.kernel import Kernel
    from plugins.graphs import InMemorySkillGraph

    k = Kernel(CAPABILITIES)
    k.provide("embodiment.env", _Env(), ref="tests.fakes:env")
    k.provide("policy.driver", _Policy(), ref="tests.fakes:policy")
    k.provide("percept.model", _Percept(), ref="tests.fakes:percept")
    k.provide("exec.rollouts", _SerialExecutor(), ref="tests.fakes:executor")
    k.provide("graph.skill", InMemorySkillGraph(), ref="plugins.graphs:skill_graph_provider")
    k.provide("reasoner.proposer", reasoner_obj, ref=reasoner_ref)
    return k


def _store_artifacts(root):
    import json
    rows = [json.loads(line) for line in (Path(root) / "index.jsonl").open()]
    return [(r["kind"], r["sha"]) for r in rows]


def test_workload_and_direct_campaign_use_the_same_explicit_model(tmp_path, monkeypatch):
    import dataclasses
    from plugins.reasoner import provider as reasoner_provider
    from plugins.rsi import workload

    monkeypatch.setattr(gate, "_run", _fake_run)
    kernel = _workload_kernel(reasoner_provider(), "plugins.reasoner:provider")
    workload.run(_prereg(), tmp_path / "wired", kernel, workers=1, verbose=False)
    stamped = dataclasses.replace(
        _prereg(), env_provider=kernel.provider_ref("embodiment.env"),
        policy_provider=kernel.provider_ref("policy.driver"),
        percept_provider=kernel.provider_ref("percept.model"))
    run_campaign(stamped, CampaignStore(tmp_path / "base"), workers=1,
                 verbose=False, executor=_SerialExecutor(), reasoner=reasoner_provider())
    assert _store_artifacts(tmp_path / "wired") == _store_artifacts(tmp_path / "base")


def test_rsi_run_drives_a_mounted_reasoner_that_declares_an_identity(tmp_path, monkeypatch):
    """The finding the verifier caught: no production path resolved
    reasoner.proposer, so an LLM card (plugins.model_qwen) could never be
    consulted. rsi_run now drives it -- and its identity lands in the seal."""
    monkeypatch.setattr(gate, "_run", _fake_run)
    from plugins.rsi import workload

    reasoner = _RecordingReasoner("qwen38(model=x)")
    kernel = _workload_kernel(reasoner, "plugins.model_qwen:provider")
    out = workload.run(_prereg(), tmp_path / "c", kernel, workers=1, verbose=False)

    assert reasoner.briefs, "the mounted reasoner was never consulted through rsi_run"
    prereg_art = CampaignStore(tmp_path / "c").read(out["result"]["preregistration_sha"])
    assert prereg_art["reasoner"] == "qwen38(model=x)"


# --- (3) model identity is content: it moves the prereg sha; None folds out ---

def test_reasoner_identity_moves_the_prereg_hash():
    """The core of closing the env-var smuggling, same shape as the percept-ref
    guard: which model proposed the rules is content, the default carries none."""
    assert _prereg().sha() == _prereg(reasoner=None).sha()          # default folds
    assert _prereg().sha() != _prereg(reasoner="qwen38(model=x)").sha()


def test_run_campaign_stamps_the_reasoner_identity_into_the_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_run", _fake_run)
    ident = "qwen38(model=qwen3-8b,base=http://h:30000/v1,temp=0.0,seed=0,attempts=2)"
    store = CampaignStore(tmp_path / "c")
    res = run_campaign(_prereg(), store, workers=1, verbose=False,
                       executor=_SerialExecutor(), reasoner=_RecordingReasoner(ident))
    prereg_art = store.read(res["preregistration_sha"])
    assert prereg_art["reasoner"] == ident
    # Different explicit model identities define different preregistrations.
    default = run_campaign(_prereg(), CampaignStore(tmp_path / "d"), workers=1,
                           verbose=False, executor=_SerialExecutor(),
                           reasoner=_RecordingReasoner("other-model"))
    assert res["preregistration_sha"] != default["preregistration_sha"]


# --- (4) a prereg predating the field rebuilds byte-identical (hash fold) -----

def test_prereg_predating_the_reasoner_field_rebuilds_byte_identical():
    from plugins.rsi.rebuild import rebuild_preregistration

    sealed = _prereg()
    payload = sealed._hash_payload()             # a sealed archive: no reasoner key
    assert "reasoner" not in payload
    rebuilt = rebuild_preregistration(payload)
    assert rebuilt.reasoner is None
    assert rebuilt.sha() == sealed.sha()


# --- (5) the qwen card: doctorable, degrades, unfolded, sha-neutral ----------

def test_doctor_greens_the_qwen_card_committed():
    rep = check(_REPO / "plugins" / "model_qwen")
    assert rep.green, [(r.tier, r.name, r.detail) for r in rep.results if r.status == "FAIL"]
    a = [r for r in rep.results if r.tier == "A" and r.name == "reasoner.proposer"]
    assert a and a[0].status == "PASS"           # Tier A shape always runs
    b = [r for r in rep.results if r.tier == "B" and r.name == "reasoner.proposer"]
    assert b and b[0].status in ("SKIP", "PASS")  # SKIP when the endpoint is down


def test_doctor_skips_the_qwen_reasoner_when_the_endpoint_is_down(tmp_path):
    # point it at a dead port so available() is deterministically False here.
    body = ('needs_sim = false\n[mounts."reasoner.proposer"]\n'
            'ref = "plugins.model_qwen:provider"\n'
            'params = { base_url = "http://127.0.0.1:9/v1", attempts = 1 }\n')
    d = tmp_path / "qwen_dead"
    d.mkdir()
    (d / "manifest.toml").write_text(body)
    rep = check(d)
    assert rep.green, [(r.tier, r.name, r.detail) for r in rep.results if r.status == "FAIL"]
    b = [r for r in rep.results if r.tier == "B" and r.name == "reasoner.proposer"]
    assert b and b[0].status == "SKIP" and "unreachable" in b[0].detail


def test_enabled_false_qwen_card_leaves_the_llm_endpoint_adapter_mounted():
    reg = discover()                              # would raise on a folded duplicate
    reasoner_mounts = [m for m in reg.mounts if m.capability == "reasoner.proposer"]
    assert len(reasoner_mounts) == 1
    assert reasoner_mounts[0].provider == "plugins.reasoner:provider"
    assert reasoner_mounts[0].params["endpoint"] == "plugins.model_endpoint:provider"
    assert "top_k" not in reasoner_mounts[0].params


def test_qwen_provider_reports_a_model_identity():
    from harness.contracts import Reasoner
    from harness.registry import load_provider

    p = load_provider("plugins.model_qwen:provider",
                      {"model": "qwen3-8b", "base_url": "http://h:30000/v1"})
    assert isinstance(p, Reasoner)
    assert "qwen3-8b" in p.identity and "h:30000" in p.identity


def _model_artifact(store):
    rows = [json.loads(line) for line in (store.root / "index.jsonl").read_text().splitlines()]
    return store.read(next(r["sha"] for r in rows if r["kind"] == "model_proposal"))


def test_endpoint_adapter_seals_raw_request_usage_and_validated_candidate(tmp_path, monkeypatch):
    from plugins.reasoner import provider
    monkeypatch.setattr(gate, "_run", _fake_run)
    store = CampaignStore(tmp_path / "c")
    reasoner = provider()
    reasoner._ep.last_usage = {"prompt": 17, "completion": 9}
    run_campaign(_prereg(), store, workers=1, verbose=False,
                 executor=_SerialExecutor(), reasoner=reasoner)
    audit = _model_artifact(store)
    assert audit["status"] == "proposed"
    assert audit["attempts"][0]["usage"] == {"prompt": 17, "completion": 9}
    assert json.loads(audit["attempts"][0]["raw"]) == _PROPOSAL
    prompt = json.loads(audit["attempts"][0]["messages"][1]["content"])
    assert "regrasp" in prompt["recovery_strategies"]
    values = prompt["observed_values"]["observable.finger_gap"]
    assert values["success"]["quantiles_0_25_50_75_100"] == [0.04] * 5
    assert values["failure"]["quantiles_0_25_50_75_100"][0] < 0.006
    assert values["failure"]["samples"] == 1200
    assert audit["identity"] == reasoner.identity


@pytest.mark.parametrize("response,status", [
    ({"kind": "none", "reason": "no supported change"}, "abstained"),
    ({"feature": "invented"}, "rejected"),
])
def test_none_and_validation_exhaustion_leave_audited_nulls(response, status, model_reply, tmp_path, monkeypatch):
    from plugins.reasoner import provider
    model_reply.write_text(json.dumps(response))
    monkeypatch.setattr(gate, "_run", _fake_run)
    store = CampaignStore(tmp_path / "c")
    result = run_campaign(_prereg(), store, workers=1, verbose=False,
                          executor=_SerialExecutor(), reasoner=provider())
    audit = _model_artifact(store)
    assert audit["status"] == status
    assert result["rules"] == [] and result["promoted"] == 0
    assert len(audit["attempts"]) == (1 if status == "abstained" else 2)


def test_model_error_is_sealed_then_raised_without_substitute(tmp_path, monkeypatch):
    from plugins.reasoner import provider
    monkeypatch.setattr(gate, "_run", _fake_run)
    reasoner = provider()
    def fail(*args, **kwargs):
        raise OSError("endpoint unavailable")
    monkeypatch.setattr(reasoner._ep, "chat", fail)
    store = CampaignStore(tmp_path / "c")
    with pytest.raises(OSError, match="endpoint unavailable"):
        run_campaign(_prereg(), store, workers=1, verbose=False,
                     executor=_SerialExecutor(), reasoner=reasoner)
    audit = _model_artifact(store)
    assert audit["status"] == "error"
    assert audit["error"] == {"type": "OSError", "message": "endpoint unavailable", "stage": "request"}
    assert len(audit["attempts"]) == 1 and audit["attempts"][0]["raw"] is None


def test_registered_but_unobserved_feature_is_not_an_admissible_proposal(model_reply, tmp_path, monkeypatch):
    from plugins.reasoner import provider
    model_reply.write_text(json.dumps({**_PROPOSAL, "feature": "privileged.object_z"}))
    monkeypatch.setattr(gate, "_run", _fake_run)
    store = CampaignStore(tmp_path / "c")
    result = run_campaign(_prereg(critic_budget=1), store, workers=1, verbose=False,
                          executor=_SerialExecutor(), reasoner=provider())
    audit = _model_artifact(store)
    assert audit["status"] == "rejected" and result["rules"] == []
    assert audit["rejections"] == ["feature is not in the observed catalog"] * 2


def test_recovery_catalog_is_scoped_to_the_executing_embodiment(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_run", _fake_run)
    reasoner = _RecordingReasoner()
    store = CampaignStore(tmp_path / "c")
    result = run_campaign(_prereg(env_provider="plugins.embodiment_robocasa:provider"),
        store, workers=1, verbose=False, executor=_SerialExecutor(), reasoner=reasoner)
    assert "regrasp_kitchen" in reasoner.briefs[0]["strategies"]
    assert "regrasp" not in reasoner.briefs[0]["strategies"]
    assert result["rules"] == []
    assert _model_artifact(store)["status"] == "rejected"


def test_default_card_doctor_never_calls_a_model_without_traces(monkeypatch):
    from plugins.reasoner import provider
    reasoner = provider()
    monkeypatch.setattr(reasoner._ep, "chat", lambda *_args, **_kwargs: pytest.fail("no network for shape"))
    assert reasoner.propose({"task": "shape-only"})["status"] == "not_evaluated"
    assert check(_REPO / "plugins" / "reasoner").green


def test_endpoint_params_follow_the_installed_manifest(monkeypatch):
    import plugins.reasoner as module
    seen = []
    class Endpoint:
        identity = "test-model"
    monkeypatch.delenv("PH_MODEL_ENDPOINT_FAKE")
    monkeypatch.setattr(module, "mount_params", lambda ref: {"model": "installed-model"})
    monkeypatch.setattr(module, "load_provider", lambda ref, params: seen.append((ref, params)) or Endpoint())
    reasoner = module.provider()
    assert seen == [("plugins.model_endpoint:provider", {"model": "installed-model"})]
    assert "installed-model" in reasoner.identity
