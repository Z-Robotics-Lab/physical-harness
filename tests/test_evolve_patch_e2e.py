"""The ``patch`` answer kind + the two-step prompt, end to end through the REAL runtime +
scripts/evolve.py on a fake stage-driver card (tests/fakes/patch_stage.py), the
model_endpoint FAKE answering a fixed sequence: round 1 a tunables decision with its
payload is ONE call; round 2 a patch decision without payload gets call 2 (the code
material inserted first, the brief last), its diff does not apply (wrong context) -> the
exact rejection goes back -> the repaired diff lands on a COPY of the module under the
candidates root (the installed file's hash is unchanged), the generated card passes the
doctor + dry run + the one-seed preflight, drives the trial suite and wins (0/2 -> 2/2).
Plus the pure-Python diff applier's edge cases and the recycle_cans call-1 brief bound."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fakes.patch_stage import EMB
from test_evolve_e2e import _CARD as _E2E_CARD
from test_mission_e2e import _Runtime

from scripts import evolve_llm

MODULE = "fakes.patch_stage"
INSTALLED = Path(__file__).parent / "fakes" / "patch_stage.py"
_CARD = _E2E_CARD.replace("test_evolve_e2e:env_provider", EMB).replace("test_evolve_e2e:", "fakes.patch_stage:") \
    .replace("[task_bindings.e2e_evolve]", "[task_bindings.e2e_patch]")
TASK = "e2e_patch"

BAD = "--- a/patch_stage.py\n+++ b/patch_stage.py\n@@ -30,2 +30,2 @@\n class GrabStage:\n-    STOP = 0.99\n+    STOP = 0.4\n"
GOOD = "--- a/patch_stage.py\n+++ b/patch_stage.py\n@@ -1,3 +1,3 @@\n     # the loaded standoff: the scripted value never closes the grab\n-    STOP = 0.65\n+    STOP = 0.4\n"


def _patch(diff):
    return {"kind": "patch", "payload": {"name": "grab_stop", "module": MODULE, "to": "patched", "diff": diff},
            "summary": "把 STOP 调小。", "rationale": "the standoff never closes"}


CANNED = [
    {"decision": "tunables", "payload": {"ref": "fakes.patch_stage:policy_provider", "path": ["stall_k"], "to": 28},
     "summary": "两颗种子都死在 grab-0。", "rationale": "先试 knob"},          # round 1: one call
    {"decision": "patch", "summary": "STOP 太大。", "rationale": "grab never closes"},   # round 2 call 1: no payload
    _patch(BAD),                                                                       # call 2: does not apply
    _patch(GOOD),                                                                      # repair: applies, wins
]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    sha0 = hashlib.sha256(INSTALLED.read_bytes()).hexdigest()
    rt = _Runtime(runs, card=_CARD, canned=CANNED, mode="evolution",
                  env={"PH_CANDIDATES_ROOT": str(runs / "candidates")})
    rt.campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    try:
        rt.run({"kind": "evolve", "task": TASK, "seeds": [1, 2], "rounds": 4, "arm": "auto"})
        rt.sha = (sha0, hashlib.sha256(INSTALLED.read_bytes()).hexdigest())
        yield rt
    finally:
        rt.stop()


def test_patch_lands_on_a_copy_after_two_steps_and_a_repair(runtime):
    doc = json.loads(runtime.campaign.read_text())
    r1, r2 = doc["rounds"][:2]
    audits = {p.name: json.loads(p.read_text()) for p in (runtime.campaign.parent / "llm").glob("round-*.json")}
    a1, a2 = audits["round-1.json"], audits["round-2.json"]
    # round 1: a decision with its payload is one call; the brief carries no code material
    assert (r1["proposer"], r1["tried"]["kind"], r1["published"]) == ("llm", "tunables", False)
    assert a1["calls"] == 1 and a1["attempts"] == [] and [m["role"] for m in a1["messages"]] == ["system", "user"]
    assert "Materials" not in a1["messages"][1]["content"] and "reference_card" not in a1["brief"]
    assert a1["brief"]["first_death"]["modules"] == [MODULE] and "scripted_driver_source" in a1["materials"]
    # round 2: call 2 = [system, materials, brief, assistant, ask]; the bad hunk's exact text went back
    assert a2["calls"] == 3 and len(a2["attempts"]) == 1
    roles = [m["role"] for m in a2["messages"]]
    assert roles == ["system", "user", "user", "assistant", "user", "assistant", "user"]
    assert a2["messages"][1]["content"].startswith("Materials (static):") and "executor_contract" in a2["messages"][1]["content"]
    assert a2["messages"][2]["content"].startswith("Round input:") and a2["messages"][4]["content"].startswith("You decided patch")
    assert a2["attempts"][0]["reason"].startswith("patch:hunk 1 does not apply") and "STOP = 0.99" in a2["attempts"][0]["reason"]
    assert "nearest matching line" in a2["attempts"][0]["reason"] and "class GrabStage" in a2["attempts"][0]["reason"]
    assert a2["attempts"][0]["reason"] in a2["messages"][6]["content"]
    # the repaired diff: applied on a copy, the installed file untouched, the card bound and published
    assert (r2["proposer"], r2["tried"]["kind"], r2["tried"]["detail"]["to"]) == ("llm", "card", "patched")
    assert r2["tried"]["detail"]["module"] == MODULE and r2["tried"]["detail"]["diff"] == GOOD
    assert (r2["before"], r2["after"], r2["published"]) == (0, 2, True)
    assert runtime.sha[0] == runtime.sha[1] and "STOP = 0.65" in INSTALLED.read_text()
    cand = runtime.runs / "candidates" / "grab_stop"
    assert r2["tried"]["detail"]["path"] == str(cand) and "STOP = 0.4" in (cand / "patch_stage.py").read_text()
    assert 'PATCHED = "fakes.patch_stage"' in (cand / "__init__.py").read_text()
    assert '[executors.patched]\nskill = "grab"' in (cand / "manifest.toml").read_text()
    assert doc["applied"]["cards"]["patched"]["ref"] == "grab_stop:provider"
    rec = json.loads((runtime.session / "skills" / f"{r2['tried']['detail']['digest']}.json").read_text())
    assert rec["bindings"][EMB]["policies"]["patched"]["ref"] == "grab_stop:provider"
    assert doc["status"] == "done" and doc["best"] == 2


def test_apply_diff_anchors_on_context_and_rejects_what_it_cannot_find():
    src = "a\nb\n\nc = 1\nd\n"
    assert evolve_llm.apply_diff(src, "@@ -99,2 +99,2 @@\n b\n\n-c = 1\n+c = 2\n") == "a\nb\n\nc = 2\nd\n"
    assert evolve_llm.apply_diff(src, "@@ -1 +1 @@\n-a\n+a0\n+a1\n@@ -5 +6 @@\n-d\n+e\n") == "a0\na1\nb\n\nc = 1\ne\n"
    with pytest.raises(ValueError, match="hunk 1 does not apply"):
        evolve_llm.apply_diff(src, "@@ -1 +1 @@\n-zz\n+y\n")
    with pytest.raises(ValueError, match="hunk 2 does not apply"):   # hunks apply in order, never backwards
        evolve_llm.apply_diff(src, "@@ -5 +5 @@\n-d\n+e\n@@ -1 +1 @@\n-a\n+z\n")
    with pytest.raises(ValueError, match="no @@ hunk"):
        evolve_llm.apply_diff(src, "just prose")
    with pytest.raises(ValueError, match="nothing to anchor"):
        evolve_llm.apply_diff(src, "@@ -1,0 +1 @@\n+x\n")


def test_recycle_cans_call_1_brief_stays_under_12k_chars():
    from test_evolve_llm_e2e import recycle_cans_projection
    proj, _ = recycle_cans_projection()
    proj["log_excerpt"] = [f"seed 4243 task.fault {json.dumps({'k': 'x' * 380})}"] * evolve_llm.MAX_LOG_LINES
    proj["history"] = [{"round": r, "proposer": "llm", "tried": {"kind": "tunables", "node": "drop-can1", "detail": {}},
                        "before": 0, "after": 0, "published": False,
                        "per_seed": [{"seed": s, "success": False, "first_death": "drop-can1", "failure_mode": "reach_stall"}
                                     for s in (4243, 4244)]} for r in range(1, 9)]
    b = evolve_llm.brief(proj)
    assert len(json.dumps(b, sort_keys=True)) <= 12_000
    assert [r["round"] for r in b["history"]] == [4, 5, 6, 7, 8] and b["history_older"] == {"rounds": 3, "published": 0, "kinds": {"tunables": 3}}
    assert b["first_death"]["modules"] == ["plugins.embodiment_robocasa.recycle_driver", "plugins.embodiment_robocasa.stage_extras"]
    assert not any(k in b for k in evolve_llm.MATERIAL_KEYS) and b["payload_by_kind"]["patch"]["module"]
    assert proj["scripted_driver_source"].startswith("# module plugins.embodiment_robocasa.stage_extras\n")
