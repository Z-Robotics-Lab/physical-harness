"""Recovery seals complete RSI evidence under the round's original evaluator."""

from __future__ import annotations

import copy
from types import SimpleNamespace

from harness.events import SessionLog
from plugins.rsi import evaluation
from scripts.evolve import EvolveStore, ROUNDS_KEPT
from scripts.harness_runtime import _seal_rounds


TASK = "sealing_fixture"
CHECK = {"id": "check-object", "kind": "verify", "skill": "placed", "args": {"object": "cup"}}


def _contract(source_version):
    return evaluation.compile_contract(
        {"nodes": [CHECK]}, task=TASK, predicates={"placed": "fixture:placed"},
        terminal_ref="fixture:world", identity={"source_digest": source_version})


def _seed(contract, checkpoint):
    return {"seed": 17, "success": False, "first_death": "finish", "failure_mode": "unfinished",
            "nodes": [{"id": "move", "kind": "segment", "ok": checkpoint, "steps": 4,
                       "trace_end": [{"step": 4, "position": [0.1, 0.2, 0.3]}]}],
            "evaluation": evaluation.evaluate(
                contract, [{"node": CHECK, "authority": "predicate", "source": "fixture:placed",
                            "evidence_policy": "world-dependencies-v1", "blocked_reads": [],
                            "success": checkpoint}],
                {"authority": "embodiment.terminal_success", "source": "fixture:world", "success": False})}


def _suite(seeds):
    return {"seeds": {str(seed["seed"]): seed for seed in seeds}}


def _round(number, contract):
    before, after = [_seed(contract, False)], [_seed(contract, True)]
    checked = evaluation.compare(_suite(before), _suite(after), contract)
    return {"round": number, "tried": {"kind": "patch", "node": "move",
                                       "detail": {"module": "fixture.driver",
                                                  "edits": [{"old": "action = 0", "new": "action = 1"}]}},
            "before": 0, "after": 0, "best": 0, "accepted": True, "published": False,
            "accepted_reason": checked["reason"], "outcome": "improved", "parent": number - 1,
            "before_score": [0, 0.0], "after_score": [0, 0.5],
            "per_seed": before, "after_seeds": after, "suite_sha": f"suite-{number}",
            "experiments": {"before": f"experiment-{number}-before", "after": f"experiment-{number}-after"},
            "evaluation": {"protocol_id": evaluation.VERSION, "objective_id": contract["sha"],
                           "before": checked["before"], "after": checked["after"],
                           "acceptance": {"accepted": True, "reason": checked["reason"]},
                           "installation": {"status": "not_evaluated"}},
            "trial_evidence": {"node": "move", "exception": None,
                               "seeds": [{"seed": 17, "diff": {"first_divergent_step": 2},
                                          "trace": {"series": [{"step": 2, "action": [1.0]}]}}]},
            "diagnosis": {"fingerprint": ["control:inactive", "response:stalled"]},
            "experience": {"retrieved": [{"id": "prior-case"}], "recorded": {"id": f"case-{number}"}},
            "transfer": {"prior_tasks": 1, "first_accepted_round": 1, "total_trials": number},
            "proposal": {"id": f"proposal-{number}", "kind": "patch", "note": "Measured intervention"},
            "llm": {"model": "fixture", "summary": "Restore a missing action",
                    "rationale": "Full rationale must survive index compaction."}}


def _store(session, rounds, contracts, current):
    store = EvolveStore(session, TASK)
    store.save({"task": TASK, "rounds": copy.deepcopy(rounds), "cursor": len(rounds),
                "evaluation_contract": current,
                "evaluation_contracts": {contract["sha"]: contract for contract in contracts}})
    return store


def test_recovery_seals_full_archived_evidence_once_and_preserves_the_hash_chain(tmp_path):
    contract = _contract("original-source")
    rounds = [_round(number, contract) for number in range(1, ROUNDS_KEPT + 3)]
    store = _store(tmp_path, rounds, [contract], contract)
    archived = store.load()["rounds"][1]
    assert archived["sharded"] and "nodes" not in archived["per_seed"][0]
    assert "evaluation" not in archived["after_seeds"][0]
    assert "trial_evidence" not in archived and "rationale" not in archived["llm"]

    log_path = tmp_path / "session-log"
    runtime = SimpleNamespace(log=SessionLog(log_path))
    _seal_rounds(runtime, "brief-1", TASK, store.path)
    events = runtime.log.rows()
    steps = [event for event in events if event["kind"] == "rsi_step"]
    assert len(steps) == len(rounds)
    sealed = next(event["data"] for event in steps if event["data"]["round"] == 2)
    full = rounds[1]
    for key in ("tried", "per_seed", "after_seeds", "llm", "trial_evidence", "evaluation",
                "experiments", "diagnosis", "experience", "transfer"):
        assert sealed[key] == full[key], key
    assert sealed["evaluation_contract"] == contract
    assert evaluation.compare(_suite(sealed["per_seed"]), _suite(sealed["after_seeds"]),
                              sealed["evaluation_contract"])["accepted"]
    assert [(event["kind"], event["data"]["round"]) for event in events[:4]] == [
        ("rsi_proposal_applied", 1), ("rsi_step", 1), ("rsi_proposal_applied", 2), ("rsi_step", 2)]

    # Restart the actual ledger writer, then repeat the recovery scan.
    reopened = SimpleNamespace(log=SessionLog.load(log_path))
    original_chain = reopened.log.rows()[-1]["chain"]
    _seal_rounds(reopened, "brief-resumed", TASK, store.path)
    assert len(reopened.log.rows()) == len(events)
    assert reopened.log.rows()[-1]["chain"] == original_chain
    assert SessionLog.load(log_path).verify()


def test_historical_rounds_use_their_own_contract_after_a_source_epoch_changes(tmp_path):
    original, current = _contract("source-v1"), _contract("source-v2")
    assert original["sha"] != current["sha"]
    rounds = [_round(1, original), _round(2, current)]
    store = _store(tmp_path, rounds, [original, current], current)
    runtime = SimpleNamespace(log=SessionLog(tmp_path / "session-log"))

    _seal_rounds(runtime, "brief-after-upgrade", TASK, store.path)
    steps = [event["data"] for event in SessionLog.load(tmp_path / "session-log").rows()
             if event["kind"] == "rsi_step"]
    assert [step["evaluation_contract"] for step in steps] == [original, current]
    for step in steps:
        contract = step["evaluation_contract"]
        assert contract["sha"] == step["evaluation"]["objective_id"]
        assert evaluation.compare(_suite(step["per_seed"]), _suite(step["after_seeds"]), contract)["accepted"]
    wrong_ruler = evaluation.compare(_suite(steps[0]["per_seed"]), _suite(steps[0]["after_seeds"]), current)
    assert not wrong_ruler["accepted"] and "lacks observations" in wrong_ruler["reason"]
    assert SessionLog.load(tmp_path / "session-log").verify()
