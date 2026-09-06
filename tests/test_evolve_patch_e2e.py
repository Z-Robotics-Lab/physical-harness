"""Real runtime patch trials with bounded inspection and executable repair.

The endpoint is an explicit fixture. Older complete proposal syntax exercises
the compatibility mapping to trial/choose after an actual source read; dedicated
agent runtime tests separately cover explicit dynamic policy-id selection.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
from fakes.patch_stage import EMB
from test_evolve_e2e import _CARD as _E2E_CARD
from test_mission_e2e import _Runtime

from scripts import evolve_llm
from scripts.evolve_evidence import compact_brief, inspect_evidence

MODULE = "fakes.patch_stage"
INSTALLED = Path(__file__).parent / "fakes" / "patch_stage.py"
_CARD = _E2E_CARD.replace("test_evolve_e2e:env_provider", EMB).replace("test_evolve_e2e:", "fakes.patch_stage:") \
    .replace("[task_bindings.e2e_evolve]", "[task_bindings.e2e_patch]")
TASK = "e2e_patch"

TWICE = [{"old": "    def done(self, env):", "new": "    def completed(self, env):"}]        # 2 places
ABSENT = [{"old": "class GrabStage:\n    STOP = 0.99", "new": "class GrabStage:\n    STOP = 0.4"}]
# RETYPED out of `functions` at the wrong indentation (what the live model does): the lenient
# match still lands it, and `new` is re-indented to the file's own columns.
GOOD = [{"old": "# the loaded standoff: the scripted value never closes the grab\nSTOP = 0.65",
         "new": "# patched by the proposer\nSTOP = 0.4"}]
DIFF = "--- a/patch_stage.py\n+++ b/patch_stage.py\n@@ -1,3 +1,3 @@\n class GrabStage:\n-    STOP = 0.99\n+    STOP = 0.4\n"


def _patch(edits=None, diff=None):
    pay = {"node": "grab-0", "name": "grab_stop", "module": MODULE, "to": "patched"}
    pay["edits" if edits is not None else "diff"] = edits if edits is not None else diff
    return {"kind": "patch", "payload": pay, "summary": "把 STOP 调小。", "rationale": "the standoff never closes"}


CANNED = [
    {"op": "inspect", "args": {"view": "parameter", "node": "grab-0", "parameter": "stall_k"}},
    {"decision": "tunables", "payload": {"node": "grab-0", "ref": "fakes.patch_stage:policy_provider", "path": ["stall_k"], "to": 28},
     "summary": "两颗种子都死在 grab-0。", "rationale": "先试 knob"},
    {"op": "inspect", "args": {"view": "source", "node": "grab-0", "module": MODULE,
                               "start": 1, "end": len(INSTALLED.read_text().splitlines())}},
    _patch(TWICE),                                                                     # call 2: `old` twice
    _patch(ABSENT),                                                                    # repair 1: `old` absent
    _patch(GOOD),                                                                      # repair 2: applies, wins
]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    from test_evolve_llm_e2e import _params_card

    runs = tmp_path_factory.mktemp("runs")
    _params_card(runs, "fakes.patch_stage:policy_provider", nested=False)
    sha0 = hashlib.sha256(INSTALLED.read_bytes()).hexdigest()
    rt = _Runtime(runs, card=_CARD, canned=CANNED, mode="evolution",
                  env={"PH_CANDIDATES_ROOT": str(runs / "candidates")})
    rt.campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    try:
        rt.run({"kind": "evolve", "task": TASK, "seeds": [1, 2], "rounds": 2, "arm": "auto", "confirm_seeds": 0})
        rt.sha = (sha0, hashlib.sha256(INSTALLED.read_bytes()).hexdigest())
        yield rt
    finally:
        rt.stop()


def test_patch_edits_land_on_a_copy_after_inspection_and_two_repairs(runtime):
    doc = json.loads(runtime.campaign.read_text())
    r1, r2 = doc["rounds"][:2]
    audits = {p.name: json.loads(p.read_text()) for p in (runtime.campaign.parent / "llm").glob("round-*.json")}
    a1, a2 = audits["round-1.json"], audits["round-2.json"]
    # The compact initial request contains no automatic source or trajectory expansion.
    assert (r1["proposer"], r1["tried"]["kind"], r1["published"]) == ("llm", "tunables", False)
    assert a1["calls"] == 2 and a1["attempts"] == []
    assert [m["role"] for m in a1["messages"]] == ["system", "user", "assistant"]
    assert "reference_card" not in a1["brief"]
    driver = a1["brief"]["driver_index"]["grab-0"]
    assert driver["capability"] in a1["brief"]["capabilities"]
    assert a1["brief"]["catalog_ref"]["view"] == "catalog"
    assert "modules" not in driver
    assert a1["materials"] == {} and a1["evidence_reads"] == 1
    inspected = next(e["result"] for e in a2["events"] if e["type"] == "tool" and e["view"] == "inspect")
    assert inspected["view"] == "source" and inspected["complete"]
    assert inspected["data"]["code"] == INSTALLED.read_text()
    assert "    STOP = 0.65" in inspected["data"]["code"]
    assert a2["calls"] == 4 and len(a2["attempts"]) == 2
    assert a2["messages"][-1]["role"] == "assistant"
    assert all([m["role"] for m in request["messages"]] == ["system", "user"] for request in a2["requests"])
    assert a2["evidence_reads"] == 1 and a2["materials_by_node"] == {}
    first_policy = r1["learning"]["probes"][0]["policy_id"]
    retained = next(p for p in a2["brief"]["working_policies"] if p["policy_id"] == first_policy)
    assert retained["tried"]["kind"] == "tunables" and retained["measurements"]
    assert all(p["policy_id"] != first_policy for p in a2["brief"]["historical_probe_replay"])
    twice, absent = (a["reason"] for a in a2["attempts"])
    assert twice.startswith("patch:edit 1: `old` occurs 2 times in the module, it must occur exactly once")
    # the rejection carries the WHOLE enclosing function (a +/-6 window is not enough to copy from)
    assert "add the surrounding lines" in twice and "line 30 is in function done, which reads" in twice
    assert "def done(self, env):" in twice and "line 46 is in function done, which reads" in twice
    assert absent.startswith("patch:edit 1: `old` occurs 0 times in the module")
    assert "line 34 is in class GrabStage, which reads" in absent   # no function encloses a class line
    assert "  34| class GrabStage:" in absent and '  47|         return "grab" in env.achieved' in absent
    returned_errors = [json.loads(r["messages"][1]["content"])["last_tool_result"] for r in a2["requests"][2:]]
    assert [value["data"]["error"]["message"] for value in returned_errors] == [twice, absent]
    # The repaired copy is bound for development; installation still needs its battery.
    assert (r2["proposer"], r2["tried"]["kind"], r2["tried"]["detail"]["to"]) == ("llm", "card", "patched")
    assert r2["tried"]["detail"]["module"] == MODULE and r2["tried"]["detail"]["edits"] == GOOD
    assert r2["tried"]["detail"]["match"] == ["lenient"]        # the round detail says HOW it matched
    assert (r2["before"], r2["after"], r2["accepted"], r2["published"]) == (0, 2, True, False)
    assert r2["learning"]["probes"][0]["scope"] == "probe"
    assert r2["learning"]["probes"][0]["accepted"] is False
    assert r2["learning"]["full_evaluations"] == 1 and r2["trial"]["seeds"] == [1, 2]
    assert runtime.sha[0] == runtime.sha[1] and "STOP = 0.65" in INSTALLED.read_text()
    cand = runtime.runs / "candidates" / "grab_stop"
    assert r2["tried"]["detail"]["path"] == str(cand)
    copy = (cand / "patch_stage.py").read_text()
    assert "\n    # patched by the proposer\n    STOP = 0.4\n" in copy   # re-indented to the file, not the answer
    assert 'PATCHED = "fakes.patch_stage"' in (cand / "__init__.py").read_text()
    assert '[executors.patched]\nskill = "grab"' in (cand / "manifest.toml").read_text()
    assert doc["applied"]["cards"]["patched"]["ref"] == "grab_stop:provider"
    assert "digest" not in r2["tried"]["detail"]  # development acceptance does not install
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




def test_recycle_cans_initial_request_keeps_catalog_addressable_within_byte_budget(tmp_path):
    from test_evolve_llm_e2e import _fake, recycle_cans_projection
    proj, before = recycle_cans_projection()
    proj["log_excerpt"] = [f"seed 4243 task.fault {json.dumps({'k': 'x' * 380})}"] * evolve_llm.MAX_LOG_LINES
    proj["history"] = [{"round": r, "proposer": "llm", "tried": {"kind": "tunables", "node": "drop-can1", "detail": {}},
                        "before": 0, "after": 0, "published": False,
                        "per_seed": [{"seed": s, "success": False, "first_death": "drop-can1", "failure_mode": "reach_stall"}
                                     for s in (4243, 4244)]} for r in range(1, 9)]
    ep = _fake(tmp_path, {"op": "stop", "args": {"reason": "Only inspect the request budget in this fixture."}})
    _, row = evolve_llm.llm_propose(ep, proj, before, 1, tmp_path / "audit", agent_tools={
        "projection": lambda policy=None: proj,
        "trial": lambda *args: pytest.fail("This model requested no trial."),
        "choose": lambda *args: pytest.fail("This model requested no selection.")})
    audit = json.loads((tmp_path / "audit" / "round-1.json").read_text())
    assert row["calls"] == 1 and row["stop_reason"] == "model_stop"
    assert audit["requests"][0]["input_bytes"] <= row["budget"]["limits"]["max_request_bytes"]
    b = audit["brief"]
    assert set(b["driver_index"]) == set(proj["drivers"])
    assert "source_index" not in b and b["history_index"]["local_rounds"] == 8
    catalog = inspect_evidence(proj, {"view": "catalog"}, max_bytes=200000)
    assert catalog["complete"] and catalog["data"]["source_index"]
    assert catalog["sha"] == b["catalog_ref"]["sha"]
    assert "history" not in b and "log_excerpt" not in b
    assert not {"module_sources", "scripted_driver_source", "functions"} & b.keys()


def test_apply_edits_replaces_an_exact_snippet_and_names_the_count_and_the_neighbourhood():
    src = "a\nb\n\nc = 1\nd\nc = 1\n"
    modes = []
    assert evolve_llm.apply_edits(src, [{"old": "b\n\nc = 1", "new": "b\n\nc = 2"}], modes) == "a\nb\n\nc = 2\nd\nc = 1\n"
    assert modes == ["exact"]
    with pytest.raises(ValueError, match="edit 1: `old` occurs 2 times"):
        evolve_llm.apply_edits(src, [{"old": "c = 1", "new": "c = 2"}])
    with pytest.raises(ValueError, match="edit 2: `old` occurs 0 times"):
        evolve_llm.apply_edits(src, [{"old": "a\n", "new": "z\n"}, {"old": "q = 9", "new": "q = 8"}])
    with pytest.raises(ValueError, match="changes nothing"):
        evolve_llm.apply_edits(src, [{"old": "d", "new": "d"}])
    with pytest.raises(ValueError, match="edit 1 must be"):
        evolve_llm.apply_edits(src, [{"new": "d"}])
    # the 0-count message points at the first line that DOES occur, with real line numbers
    why = str(pytest.raises(ValueError, evolve_llm.apply_edits, src, [{"old": "b\nq = 9", "new": "x"}]).value)
    assert "the module around line 2 reads:" in why and "   2| b" in why


def test_a_retyped_snippet_matches_leniently_exactly_once_and_reports_the_mode(tmp_path):
    """4 of the last 6 live rounds died on "`old` occurs 0 times": the model retypes the
    snippet at its own indentation. Exact first, then leading-indent / trailing-space blind --
    still EXACTLY one hit -- and `new` shifted to the file's columns."""
    src = "class C:\n    def f(self):\n        x = 1   \n        return x\n"
    modes = []
    out = evolve_llm.apply_edits(src, [{"old": "x = 1\nreturn x", "new": "x = 2\nreturn x"}], modes)
    assert out == "class C:\n    def f(self):\n        x = 2\n        return x\n" and modes == ["lenient"]
    modes.clear()
    assert evolve_llm.apply_edits(src, [{"old": "        x = 1   ", "new": "        x = 3"}], modes)
    assert modes == ["exact"]
    # two lenient hits are still ambiguous: the count comes back, not a guess
    two = "def f():\n    x = 1\n\ndef g():\n        x = 1\n"
    why = str(pytest.raises(ValueError, evolve_llm.apply_edits, two, [{"old": "x = 1", "new": "x = 2"}]).value)
    assert "occurs 2 times" in why and "not even ignoring indentation" in why
    # a miss hands back the WHOLE enclosing function, not a +/-6 window
    why = str(pytest.raises(ValueError, evolve_llm.apply_edits, src, [{"old": "  x = 9", "new": "  x = 8"}]).value)
    assert "occurs 0 times" in why and "inspect" in why and "symbol" in why
    why = str(pytest.raises(ValueError, evolve_llm.apply_edits, src, [{"old": "return x\nq = 9", "new": "z"}]).value)
    assert ("line 4 is in function f, which reads (copy `old` out of THIS text, "
            "WITHOUT the `NNNN| ` line-number prefixes):") in why
    assert "   2|     def f(self):\n   3|         x = 1   \n   4|         return x" in why
    # copied out of `functions` but sent against the wrong module (live round 102): the
    # refusal names the module the snippet is actually in, and the ids default
    pay = {"module": "scripts.evolve", "edits": [{"old": "SCORE_DEF = (", "new": "SCORE_DEF  = ("}]}
    why = evolve_llm.write_patch(pay, {"modules": ["scripts.evolve", "scripts.evolve_llm"]}, 7,
                                 root=tmp_path)
    assert "occurs 0 times" in why and "occurs EXACTLY ONCE in scripts.evolve_llm" in why
    assert pay["name"] == pay["to"] == "patch_r7"


def test_an_already_accepted_edit_is_refused_because_it_is_this_rounds_baseline(tmp_path):
    """An accepted change IS the baseline the round starts from: re-proposing it changes
    nothing. The refusal names the round that accepted it."""
    from test_evolve_llm_e2e import _repeat_proj

    proj, _ = _repeat_proj([])
    proj["drivers"]["grab-0"]["executor"] = "patched"
    proj["accepted_stack"] = [{"round": 3, "kind": "card", "node": "grab-0",
                               "detail": {"to": "patched", "module": MODULE, "edits": GOOD}}]
    same = {"node": "grab-0", "name": "again", "module": MODULE, "to": "patched", "edits": [dict(GOOD[0])]}
    assert evolve_llm._accepted_repeat(proj, same).startswith("this exact edit is already active on node grab-0 (round 3)")
    assert evolve_llm._accepted_repeat(proj, {**same, "edits": TWICE}) is None


def test_a_unified_diff_is_still_accepted_where_edits_would_go(tmp_path):
    pay = {"name": "grab_diff", "module": MODULE, "to": "patched", "diff": DIFF}
    why = evolve_llm.write_patch(pay, {"modules": [MODULE]}, 0, tmp_path)
    assert why.startswith("patch:hunk 1 does not apply")      # the diff path, not the edits path
    assert evolve_llm.write_patch({**pay, "diff": None}, {"modules": [MODULE]}, 0, tmp_path) \
        .startswith("patch:payload needs `edits`")


def test_state_init_names_where_new_state_belongs():
    """The rules tell a patch to initialise new state in the class's EXISTING constructor /
    reset: the materials name those methods and what they already set."""
    from fakes.patch_stage import Driver, GrabStage

    assert evolve_llm._state_init([GrabStage, Driver]) == {
        f"{MODULE}:GrabStage.__init__": ["target"],
        f"{MODULE}:Driver.__init__": ["_cap", "_env", "_ex", "_native", "_stage", "n"]}


def test_runtime_error_history_is_evidence_without_a_hardcoded_next_strategy():
    from test_evolve_llm_e2e import _repeat_proj

    hist = [{"round": r, "tried": {"kind": "card", "node": "grab-0", "detail": {}},
             "trial_evidence": "seed 1 raised AttributeError: '_last_d' at grab-0"} for r in (1, 2, 3)]
    proj, _ = _repeat_proj(hist)
    b = inspect_evidence(proj, {"view": "history"})["data"]
    assert [r["trial_evidence"] for r in b["history"]] == [r["trial_evidence"] for r in hist]
    assert not {"repeat_failure", "stuck", "untried", "exhausted", "target"} & b.keys()


def test_an_already_run_patch_is_refused_with_what_that_round_measured():
    """The gap that ate a 588-round campaign: a patch that APPLIES, RUNS and merely fails to
    raise the score is in no reject list, and the model never sees its own past code -- so it
    re-derived the same drop-point clamp for ~300 rounds. The key is the patched MODULE's AST
    dump, not its text (the re-wording is why: 363 patch rounds, 319 distinct files, 215
    distinct ASTs), and the refusal carries the verdict."""
    hist = [{"round": 12, "after": 0, "experiments": {"before": "fixture-baseline"}, "verdict": "focused trial: drop-can1 passed on no seed of its cluster",
             "tried": {"kind": "card", "node": "drop-can1",
                       "detail": {"module": "m", "patch_sha": "deadbeefdeadbeef"}}}]
    proj = {"history": hist, "experiment_id": "fixture-baseline"}
    ran = evolve_llm._ran_patches(proj, "drop-can1")
    assert ran["deadbeefdeadbeef"].startswith("round 12: focused trial: drop-can1")
    assert evolve_llm._ran_patches(proj, "other-node") == {}
    assert evolve_llm._ran_patches({**proj, "experiment_id": "new-baseline"}, "drop-can1") == {}
    assert evolve_llm._ran_patches({**proj, "history": [{**hist[0], "after": None}]}, "drop-can1") == {}


def test_a_patch_to_a_module_the_stage_never_inherits_from_is_refused_by_name():
    """171 of the campaign's 363 patch rounds were mechanically no-ops: a stuck round widens
    `modules` to the whole pipeline, but PATCH_CARD.make_stage() only swaps classes the target
    stage's MRO contains -- nav-can1's comes from stage_extras + drivers, so its 153
    recycle_driver patches (plus 11 planner ones, and 7 planner patches judged on drop-can1)
    installed a card that ran the stock code, doctor-clean, one full suite each."""
    from harness.manifest import discover
    from scripts import harness_runtime as hr
    binding = discover().task_bindings["recycle_cans"]
    nodes = ("nav-can1", "grasp-can1", "carry-can1", "drop-can1")
    before = {"count": 0, "seeds": {"4243": {"success": False, "first_death": "nav-can1", "trail": [],
                                             "nodes": {n: {"skill": n.replace("-", "_"), "success": False,
                                                           "executor": "scripted"} for n in nodes}}}}
    fd = evolve_llm._driver(before, hr._binding_records(binding), "robocasa", "scripted", binding, "nav-can1")
    assert fd["stage_modules"] == ["plugins.embodiment_robocasa.drivers",
                                   "plugins.embodiment_robocasa.stage_extras"]
    fd["modules"] = [*fd["modules"], "plugins.embodiment_robocasa.recycle_driver"]   # what stuck widening does
    pay = {"name": "p", "to": "p", "module": "plugins.embodiment_robocasa.recycle_driver",
           "edits": [{"old": "x", "new": "y"}]}
    why = evolve_llm.write_patch(pay, fd, 1)
    assert why.startswith("patch:") and "stage_extras" in why and "nav-can1" in why
    # the same stage's own module gets as far as the edit itself
    assert "cannot change node" not in (evolve_llm.write_patch(
        {**pay, "module": "plugins.embodiment_robocasa.stage_extras"}, fd, 1) or "")


def test_a_new_parameter_value_needs_no_task_specific_geometry_exception():
    """Trying either sign is not exhausting a continuous parameter domain."""
    from test_evolve_llm_e2e import _parameter_history_proj
    proj, before = _parameter_history_proj()
    answer = lambda to: {"kind": "tunables", "layer": "parameter", "rationale": "new value",
                         "payload": {"node": "grab-0", "ref": "test_evolve_e2e:policy_provider", "path": ["tunables", "hover_dz"], "to": to}}
    assert proj["this_round"]["per_seed"] == []  # no geometry or benchmark special case
    assert evolve_llm._try(answer(0.188), proj, before, 1)["detail"]["to"] == 0.188
    proj["history"].append({"round": 5, "after": 0, "experiments": {"before": proj["experiment_id"]}, "tried": {"kind": "tunables", "node": "grab-0",
        "detail": {"ref": "test_evolve_e2e:policy_provider", "path": ["tunables", "hover_dz"], "from": 0.1, "to": 0.188}}})
    with pytest.raises(ValueError, match="same baseline"):
        evolve_llm._try(answer(0.188), proj, before, 2)
    assert evolve_llm._try(answer(0.21), proj, before, 2)["detail"]["to"] == 0.21
    seen = set()
    evolve_llm._try(answer(0.3), proj, before, 2, seen=seen)
    evolve_llm._try(answer(0.15), proj, before, 2, seen=seen)
    with pytest.raises(ValueError, match="already"):
        evolve_llm._try(answer(0.3), proj, before, 2, seen=seen)
    assert "exhausted" not in compact_brief(proj)


def _two_death_proj():
    """The live shape: one seed dies at the head node the rotation picked, the other at the
    node the model actually diagnoses. ``drivers`` carries both (rsi_projection builds it)."""
    nodes = {n: {"skill": s, "executor": "scripted", "success": n == "reach-0"}
             for n, s in (("reach-0", "reach"), ("grab-0", "grab"))}
    before = {"count": 0, "seeds": {"1": {"first_death": "reach-0", "nodes": nodes},
                                    "2": {"first_death": "grab-0", "nodes": nodes}}}
    drv = lambda node, task: {"node": node, "skill": task, "task": task, "executor": "scripted",
                              "embodiment": EMB, "modules": [MODULE], "executors": {"scripted": {}},
                              "tunables": {"ref": "fakes.patch_stage:policy_provider",
                                           "path": ["tunables"], "values": {}, "hints": {}}}
    proj = {"first_death": drv("reach-0", "reach"),          # the rotation head, NOT the answer's node
            "drivers": {"reach-0": drv("reach-0", "reach"), "grab-0": drv("grab-0", "grab")},
            "history": [], "experiment_id": "fixture-baseline", "this_round": {"per_seed": []}}
    return proj, before


def test_a_payload_node_off_the_rotation_head_is_what_gets_materialised_and_mounted(tmp_path, monkeypatch):
    """The half of the live campaign that could not be executed at all: with two death nodes
    the rotation put nav-can1 at the head while the model kept diagnosing drop-can1, and the
    patch was materialised + mounted against the HEAD -- 210 of those 252 rounds installed a
    drop fix as nav's executor, so the drop segment ran the stock card either way. payload.node
    already decided the judgement; now it decides the skill/task stamped into the candidate and
    the key apply() writes."""
    from scripts import evolve

    monkeypatch.setattr(evolve_llm, "CANDIDATES_ROOT", tmp_path / "cands")
    proj, before = _two_death_proj()
    ans = {"kind": "patch", "payload": {"name": "grab_stop_node", "module": MODULE, "to": "patched",
                                        "node": "grab-0", "edits": GOOD},
           "summary": "STOP 太大。", "rationale": "grab never closes"}
    tried = evolve_llm._try(ans, proj, before, 9)
    assert tried["node"] == "grab-0" and tried["detail"]["skill"] == "grab"
    cand = tmp_path / "cands" / "grab_stop_node"
    assert '[executors.patched]\nskill = "grab"' in (cand / "manifest.toml").read_text()
    assert 'TASK = "grab"' in (cand / "__init__.py").read_text()      # the head would have said "reach"
    applied = {"executors": {"reach-0": "scripted", "grab-0": "scripted"}, "tunables": {}, "cards": {}}
    assert evolve.apply(tried, applied)["executors"] == {"reach-0": "scripted", "grab-0": "patched"}


def test_the_capability_space_and_executor_checks_follow_the_answered_node():
    """An off-head diagnosis sees its own knobs and executor history."""
    proj, before = _two_death_proj()
    proj["drivers"]["reach-0"]["tunables"] |= {"ref": "reach:p", "values": {"standoff": 1.0}}
    proj["drivers"]["grab-0"]["tunables"] |= {"ref": "grab:p", "values": {"grip": 1.0}}
    for n in proj["drivers"]:
        proj["drivers"][n]["executors"] = {"scripted": {}, "vla": {}}
    proj["first_death"] = proj["drivers"]["reach-0"]
    proj["history"] = [{"round": 1, "after": 0, "experiments": {"before": proj["experiment_id"]},
                        "tried": {"kind": "executor", "node": "reach-0", "detail": {"to": "vla"}}}]
    for node, knob in (("grab-0", "grip"), ("reach-0", "standoff")):
        fd = proj["drivers"][node]
        assert knob in fd["tunables"]["values"] and "vla" in fd["executors"]
        no = {"kind": "none", "rationale": "insufficient evidence", "payload": {"node": node}}
        assert evolve_llm._try(no, proj, before, 2)["kind"] == "none"
    ans = lambda node: {"kind": "executor", "payload": {"node": node, "to": "vla"},
                        "rationale": "test another bound executor"}
    assert evolve_llm._try(ans("grab-0"), proj, before, 2)["node"] == "grab-0"
    with pytest.raises(ValueError, match="already measured"):
        evolve_llm._try(ans("reach-0"), proj, before, 2)


def test_the_patch_sha_a_try_stamps_is_what_the_next_round_refuses(tmp_path, monkeypatch):
    """The gate above only bites if ``patch_sha`` actually reaches ``tried.detail`` -- the
    payload keys ``from_proposal`` copies do not include it, so for one whole campaign the
    stamp existed and the history never carried it (``_ran_patches`` returned {} on 363 patch
    rounds). Drives the real line: _try -> detail -> index_row -> history -> the refusal.
    The key is the AST dump (``_patch_key``), so the SECOND answer here differs only in a
    comment -- the campaign re-worded the same change every round and a text sha saw 319
    distinct modules in 363 patch rounds against the AST key's 215."""
    from scripts import evolve

    monkeypatch.setattr(evolve_llm, "CANDIDATES_ROOT", tmp_path / "cands")
    proj, before = _two_death_proj()
    ans = {"kind": "patch", "summary": "STOP 太大。", "rationale": "grab never closes",
           "payload": {"name": "p1", "module": MODULE, "to": "p1", "node": "grab-0", "edits": GOOD}}
    tried = evolve_llm._try(ans, proj, before, 9)
    sha = tried["detail"]["patch_sha"]
    assert sha and len(sha) == 16
    row = evolve.index_row({"round": 9, "tried": tried, "before": 0, "after": 0})
    assert row["tried"]["detail"]["patch_sha"] == sha          # survives the index row...
    proj["history"] = [{"round": 9, "after": 0, "experiments": {"before": proj["experiment_id"]}, "verdict": "it did not raise the score",
                        "tried": {"kind": row["tried"]["kind"], "node": row["tried"]["node"],
                                  "detail": {"patch_sha": row["tried"]["detail"]["patch_sha"]}}}]
    assert evolve_llm._ran_patches(proj, "grab-0") == {sha: "round 9: it did not raise the score"}
    reworded = [{"old": GOOD[0]["old"], "new": "# a different comment entirely\nSTOP = 0.4"}]
    ans2 = {**ans, "payload": {**ans["payload"], "name": "p2", "to": "p2", "edits": reworded}}
    with pytest.raises(ValueError) as e:
        evolve_llm._try(ans2, proj, before, 10)
    assert "same node and baseline: round 9: it did not raise the score" in str(e.value)


def test_patching_a_base_preserves_zero_argument_super_in_real_cross_module_subclasses(tmp_path, monkeypatch):
    """pack_lunch round 65 failed while binding a generated base-module patch.

    Rebuilding subclasses must rebind their __class__ closures. Merely copying
    their methods onto new bases breaks real zero-argument super() calls.
    """
    package = tmp_path / "super_chain_fixture"
    package.mkdir()
    (package / "base.py").write_text('''
class Base:
    VALUE = 1

    def __init__(self):
        self.initialized = ["base"]

    def act(self, env, obs):
        return (self.VALUE,)

    def done(self, env):
        return False

    @classmethod
    def ancestry(cls):
        return [cls.__name__, "base"]

    @property
    def label(self):
        return "base"
''')
    (package / "middle.py").write_text('''
from super_chain_fixture.base import Base

class Middle(Base):
    def __init__(self):
        super().__init__()
        self.initialized.append("middle")

    def act(self, env, obs):
        return (super().act(env, obs)[0] + 10,)

    @classmethod
    def ancestry(cls):
        return super().ancestry() + ["middle"]

    @property
    def label(self):
        return super().label + "/middle"
''')
    (package / "leaf.py").write_text('''
from super_chain_fixture.middle import Middle

class Leaf(Middle):
    def __init__(self):
        super().__init__()
        self.initialized.append("leaf")

    def act(self, env, obs):
        return (super().act(env, obs)[0] + 100,)

    @classmethod
    def ancestry(cls):
        return super().ancestry() + ["leaf"]

    @property
    def label(self):
        return super().label + "/leaf"
''')
    (package / "__init__.py").write_text('''
from super_chain_fixture import leaf

_STAGES = {"grab": (lambda: leaf.Leaf(),)}
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    installed = importlib.import_module("super_chain_fixture.leaf")
    original = installed.Leaf
    old_sources = {path: path.read_bytes() for path in package.glob("*.py")}
    module = "super_chain_fixture.base"
    payload = {"name": "super_chain_candidate", "module": module, "to": "super_chain",
               "edits": [{"old": "    VALUE = 1", "new": "    VALUE = 2"}]}
    driver = {"node": "grab-0", "skill": "grab", "task": "grab", "embodiment": EMB,
              "modules": [module], "stage_modules": [module],
              "tunables": {"ref": "super_chain_fixture:provider", "values": {}}}
    assert evolve_llm.write_patch(payload, driver, 65, tmp_path / "candidates") is None
    candidate = importlib.import_module("super_chain_candidate")
    executor = candidate.provider().make_driver(None)
    executor.bind(object())  # The original failure occurs here, before any simulator step.
    assert executor.act({}) == (112,)
    stage = executor._stage
    assert stage.initialized == ["base", "middle", "leaf"]
    assert stage.ancestry() == ["Leaf", "base", "middle", "leaf"]
    assert stage.label == "base/middle/leaf"
    assert installed.Leaf is original and original().act(None, {}) == (111,)
    assert all(path.read_bytes() == source for path, source in old_sources.items())
