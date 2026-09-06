"""Round 85 rung 2: per-stage failure attribution as a sealed store artifact.

gate._run is monkeypatched with a seed-deterministic fake that carries the
stage overlay (the test_round25_rerun trick), so run_campaign's aggregation
runs over synthetic rollouts against a real content-addressed store. The
opt-in stays round 78's ruling: the "stages" key's existence on a rollout
result is the ONLY switch, and stages=None writes nothing and changes
nothing. The load-bearing invariant is that the descriptive table cannot
silently diverge from the gated boolean -- the furthest-stage buckets count
the TERMINAL label, so their total equals the label count the gate consumed.
"""

from __future__ import annotations

import json

import numpy as np

from harness.spec import Clause, StageSpec
from plugins.rsi import gate
from plugins.rsi.campaign import CampaignStore, Preregistration, run_campaign, stage_attribution

ALWAYS = (Clause("observable.finger_gap", "gt", -1.0),)
CHAIN = (StageSpec("grasp", ALWAYS, 62), StageSpec("place", ALWAYS, 38))


class _FakeExecutor:
    def map(self, fn, items, *, workers):
        return [fn(item) for item in items]


def _stage(name: str, reached: bool, success: bool) -> dict:
    return {"name": name, "entered_step": 0 if reached else None, "exited_step": None,
            "success": success, "reached": reached, "privilege_used": 1 if reached else 0}


def _staged(success: bool) -> list[dict]:
    """Successes clear both stages; failures die in grasp and never reach place."""
    if success:
        return [_stage("grasp", True, True), _stage("place", True, True)]
    return [_stage("grasp", True, False), _stage("place", False, False)]


def _fake_run(job):
    """Even seeds fail ungoverned; the real bundle fixes them; the blind twin
    fixes nothing, so the judgement head-to-head separates cleanly. Stage data
    rides along ONLY when the spec carries a chain, exactly like governed_rollout."""
    spec, bundle = job
    failing = spec.seed % 2 == 0
    if bundle is None or not bundle.rules or all(r.rule_id.endswith("-blind") for r in bundle.rules):
        success = not failing
    else:
        success = True
    out = {
        "success": success,
        "fired_at": 0 if bundle is not None and bundle.rules and failing else None,
        "trace": {
            "observable.finger_gap": np.full(80, 0.001 if failing else 0.04),
            "observable.eef_z": np.linspace(1.0, 0.9, 80),
            "observable.gripper_effort": np.zeros(80),
            "observable.joint_speed": np.zeros(80),
        },
    }
    if spec.stages is not None:
        out["stages"] = _staged(success)
    return out


def _prereg(stages: tuple[StageSpec, ...] | None) -> Preregistration:
    return Preregistration(
        dev=tuple(range(41000, 41020)), heldout=tuple(range(42000, 42010)),
        percept_noise=0.02, critic_budget=0, action_budget=0,
        recovery_sensor_sd=0.02, max_generations=1, stages=stages)


def _campaign(tmp_path, monkeypatch, stages):
    monkeypatch.setattr(gate, "_run", _fake_run)
    from plugins.reasoner import provider
    response = tmp_path / "model-reply.json"
    response.write_text(json.dumps({"feature": "observable.finger_gap", "op": "lt", "threshold": 0.02,
                                    "dwell": 1, "arm_after": 10, "reducer": "value", "recovery": "regrasp"}))
    monkeypatch.setenv("PH_MODEL_ENDPOINT_FAKE", str(response))
    store = CampaignStore(tmp_path / "store")
    result = run_campaign(_prereg(stages), store, workers=1, verbose=False,
                          executor=_FakeExecutor(), reasoner=provider())
    index = [json.loads(line) for line in store.index_path.open()]
    return store, result, index


def test_campaign_seals_the_stage_attribution_artifact(tmp_path, monkeypatch):
    store, _result, index = _campaign(tmp_path, monkeypatch, CHAIN)

    rows = [r for r in index if r["kind"] == "stage_attribution"]
    assert len(rows) == 1  # one generation ran, one measurement attributed
    art = store.read(rows[0]["sha"])
    gen_art = store.read(next(r["sha"] for r in index if r["kind"] == "generation"))
    assert art["generation"] == 1
    assert art["preregistration_sha"] == gen_art["preregistration_sha"]

    t = art["table"]
    assert t["n"] == 20 and t["successes"] == 10
    assert t["stage_order"] == ["grasp", "place"]
    assert t["per_stage"] == {"grasp": {"reached": 20, "success": 10},
                              "place": {"reached": 10, "success": 10}}
    assert t["furthest"] == {"grasp": {"episodes": 10, "terminal_success": 0},
                             "place": {"episodes": 10, "terminal_success": 10},
                             "(none)": {"episodes": 0, "terminal_success": 0}}
    assert t["first_failure"] == {"grasp": 10, "place": 0, "(terminal)": 0}


def test_attribution_successes_equal_the_gated_label_count(tmp_path, monkeypatch):
    """The table describes the SAME measurement the dev gate's baseline arm
    scored, so its bucket totals must reproduce the gated numbers exactly."""
    store, _result, index = _campaign(tmp_path, monkeypatch, CHAIN)

    t = store.read(next(r["sha"] for r in index if r["kind"] == "stage_attribution"))["table"]
    dev_gate = store.read(next(r["sha"] for r in index if r["kind"] == "generation"))["dev_gate"]
    assert sum(b["terminal_success"] for b in t["furthest"].values()) == t["successes"]
    assert t["successes"] == round(dev_gate["base_rate"] * dev_gate["n"])
    assert sum(t["first_failure"].values()) == t["n"] - t["successes"]


def test_stages_none_writes_nothing_and_changes_nothing(tmp_path, monkeypatch):
    """Round 78's byte-identity discipline at the artifact layer: a stages=None
    campaign's index carries exactly the expected kinds and the result dict
    exactly the legacy keys -- no stage artifact, no new key, no new config.
    The model proposal audit is sealed independently of the stage overlay."""
    _store, result, index = _campaign(tmp_path, monkeypatch, None)

    assert [r["kind"] for r in index] == ["preregistration", "model_proposal", "generation",
                                          "campaign_result"]
    assert set(result) == {"preregistration_sha", "power_plans", "generations", "promoted",
                           "final_sha", "rules", "heldout", "heldout_vs_blind", "ablation"}


# --- the aggregation itself, on hand-built results ---------------------------

def test_stage_attribution_is_none_without_the_stages_key():
    assert stage_attribution([]) is None
    assert stage_attribution([{"success": True, "trace": {}}]) is None


def test_attribution_edge_buckets_keep_the_invariant():
    """(terminal) catches every-stage-held-but-terminal-failed; (none) catches
    zero-reach episodes; the terminal_success total still equals the labels."""
    results = [
        {"success": True, "stages": _staged(True)},
        {"success": False, "stages": _staged(False)},
        {"success": False, "stages": _staged(True)},   # sub-goals held, task failed
        {"success": False, "stages": [_stage("grasp", False, False),
                                      _stage("place", False, False)]},  # reached nothing
    ]
    t = stage_attribution(results)
    assert t["n"] == 4 and t["successes"] == 1
    assert t["furthest"]["(none)"] == {"episodes": 1, "terminal_success": 0}
    assert t["furthest"]["place"] == {"episodes": 2, "terminal_success": 1}
    assert t["first_failure"] == {"grasp": 2, "place": 0, "(terminal)": 1}
    assert sum(b["terminal_success"] for b in t["furthest"].values()) == t["successes"]
    assert sum(t["first_failure"].values()) == t["n"] - t["successes"]
