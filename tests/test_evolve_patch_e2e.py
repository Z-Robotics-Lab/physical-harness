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

TWICE = [{"old": "        return (0.0,)", "new": "        return (0.1,)"}]        # 2 places
ABSENT = [{"old": "class GrabStage:\n    STOP = 0.99", "new": "class GrabStage:\n    STOP = 0.4"}]
# RETYPED out of `functions` at the wrong indentation (what the live model does): the lenient
# match still lands it, and `new` is re-indented to the file's own columns.
GOOD = [{"old": "# the loaded standoff: the scripted value never closes the grab\nSTOP = 0.65",
         "new": "# patched by the proposer\nSTOP = 0.4"}]
DIFF = "--- a/patch_stage.py\n+++ b/patch_stage.py\n@@ -1,3 +1,3 @@\n class GrabStage:\n-    STOP = 0.99\n+    STOP = 0.4\n"


def _patch(edits=None, diff=None):
    pay = {"name": "grab_stop", "module": MODULE, "to": "patched"}
    pay["edits" if edits is not None else "diff"] = edits if edits is not None else diff
    return {"kind": "patch", "payload": pay, "summary": "把 STOP 调小。", "rationale": "the standoff never closes"}


CANNED = [
    {"decision": "tunables", "payload": {"ref": "fakes.patch_stage:policy_provider", "path": ["stall_k"], "to": 28},
     "summary": "两颗种子都死在 grab-0。", "rationale": "先试 knob"},          # round 1: one call
    {"decision": "patch", "summary": "STOP 太大。", "rationale": "grab never closes"},   # round 2 call 1: no payload
    _patch(TWICE),                                                                     # call 2: `old` twice
    _patch(ABSENT),                                                                    # repair 1: `old` absent
    _patch(GOOD),                                                                      # repair 2: applies, wins
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


def test_patch_edits_land_on_a_copy_after_two_steps_and_two_repairs(runtime):
    doc = json.loads(runtime.campaign.read_text())
    r1, r2 = doc["rounds"][:2]
    audits = {p.name: json.loads(p.read_text()) for p in (runtime.campaign.parent / "llm").glob("round-*.json")}
    a1, a2 = audits["round-1.json"], audits["round-2.json"]
    # round 1: a decision with its payload is one call; the brief carries no code material
    assert (r1["proposer"], r1["tried"]["kind"], r1["published"]) == ("llm", "tunables", False)
    assert a1["calls"] == 1 and a1["attempts"] == [] and [m["role"] for m in a1["messages"]] == ["system", "user"]
    assert "Materials" not in a1["messages"][1]["content"] and "reference_card" not in a1["brief"]
    assert a1["brief"]["first_death"]["modules"] == [MODULE] and "scripted_driver_source" in a1["materials"]
    # the materials carry the module's REAL text, numbered, no elision: an edit is copied out of it
    src = a1["materials"]["module_sources"][MODULE]
    assert src.startswith("# file: ") and "patch_stage.py\n" in src.split("\n")[0] + "\n"
    assert "  36|     STOP = 0.65" in src and a1["brief"]["first_death"]["modules_full"] == [MODULE]
    assert INSTALLED.read_text().split("\n")[35] == "    STOP = 0.65"     # 1-based line 36, verbatim
    # round 2: call 2 = [system, materials, brief, assistant, ask]; both bad edits' text went back
    assert a2["calls"] == 4 and len(a2["attempts"]) == 2
    roles = [m["role"] for m in a2["messages"]]
    assert roles == ["system", "user", "user", "assistant", "user"] + ["assistant", "user"] * 2
    assert a2["messages"][1]["content"].startswith("Materials (static):") and "executor_contract" in a2["messages"][1]["content"]
    assert a2["messages"][2]["content"].startswith("Round input:") and a2["messages"][4]["content"].startswith("You decided patch")
    twice, absent = (a["reason"] for a in a2["attempts"])
    assert twice.startswith("patch:edit 1: `old` occurs 2 times in the module, it must occur exactly once")
    # the rejection carries the WHOLE enclosing function (a +/-6 window is not enough to copy from)
    assert "add the surrounding lines" in twice and "line 28 is in function act, which reads" in twice
    assert "  26|     def act(self, env, obs):\n  27|         env.reached = True\n  28|         return (0.0,)" in twice
    assert "line 44 is in function act, which reads" in twice
    assert absent.startswith("patch:edit 1: `old` occurs 0 times in the module")
    assert "line 34 is in class GrabStage, which reads" in absent   # no function encloses a class line
    assert "  34| class GrabStage:" in absent and "  47|         return bool(getattr(env, \"grabbed\", False))" in absent
    assert twice in a2["messages"][6]["content"] and absent in a2["messages"][8]["content"]
    # the repaired edits: applied on a copy, the installed file untouched, the card bound and published
    assert (r2["proposer"], r2["tried"]["kind"], r2["tried"]["detail"]["to"]) == ("llm", "card", "patched")
    assert r2["tried"]["detail"]["module"] == MODULE and r2["tried"]["detail"]["edits"] == GOOD
    assert r2["tried"]["detail"]["match"] == ["lenient"]        # the round detail says HOW it matched
    assert (r2["before"], r2["after"], r2["published"]) == (0, 2, True)
    assert runtime.sha[0] == runtime.sha[1] and "STOP = 0.65" in INSTALLED.read_text()
    cand = runtime.runs / "candidates" / "grab_stop"
    assert r2["tried"]["detail"]["path"] == str(cand)
    copy = (cand / "patch_stage.py").read_text()
    assert "\n    # patched by the proposer\n    STOP = 0.4\n" in copy   # re-indented to the file, not the answer
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
    assert "occurs 0 times" in why and 'functions["<module>:<Class>.<method>"]' in why
    why = str(pytest.raises(ValueError, evolve_llm.apply_edits, src, [{"old": "return x\nq = 9", "new": "z"}]).value)
    assert "line 4 is in function f, which reads (copy `old` out of THIS text):" in why
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
    # scripts/evolve.py keeps the edits under `detail`; a flat row works too
    proj["accepted_stack"] = {"changes": [{"round": 3, "kind": "card",
                                           "detail": {"module": MODULE, "edits": GOOD}}]}
    same = {"name": "again", "module": MODULE, "to": "patched", "edits": [dict(GOOD[0])]}
    assert evolve_llm._accepted_repeat(proj, same).startswith("this edit is ALREADY ACCEPTED (round 3)")
    assert evolve_llm._accepted_repeat(proj, {**same, "edits": TWICE}) is None


def test_a_unified_diff_is_still_accepted_where_edits_would_go(tmp_path):
    pay = {"name": "grab_diff", "module": MODULE, "to": "patched", "diff": DIFF}
    why = evolve_llm.write_patch(pay, {"modules": [MODULE]}, 0, tmp_path)
    assert why.startswith("patch:hunk 1 does not apply")      # the diff path, not the edits path
    assert evolve_llm.write_patch({**pay, "diff": None}, {"modules": [MODULE]}, 0, tmp_path) \
        .startswith("patch:payload needs `edits`")


def test_an_answer_identical_to_a_rejected_one_costs_no_attempt_and_ends_the_round(tmp_path):
    """Rounds 56-66 of the live campaign: the model resent the byte-identical answer three
    times, each one paying for a full attempt. Now the repeat is named, then taken as none."""
    from test_evolve_llm_e2e import _fake, _repeat_proj

    proj, before = _repeat_proj([{"round": 1, "tried": {"kind": "tunables", "node": "grab-0", "detail": {
        "ref": "r", "path": ["tunables", "hover_dz"], "from": 0.10, "to": 0.15}}}])
    same = {"kind": "tunables", "payload": {"ref": "r", "path": ["tunables", "hover_dz"], "to": 0.20},
            "summary": "再放大一点。", "rationale": "同一个方向"}
    ep = _fake(tmp_path, [same, same, same], name="same.json")
    tried, _ = evolve_llm.llm_propose(ep, proj, before, 2, tmp_path / "llm")
    audit = json.loads((tmp_path / "llm" / "round-2.json").read_text())
    assert len(audit["attempts"]) == 1 and len(audit["repeats"]) == 2   # the repeats are not attempts
    assert audit["repeats"][0]["reason"] == audit["attempts"][0]["reason"]
    nag = audit["messages"][5]["content"]
    assert nag.startswith("你重复了上一条被拒的回答（ValueError: hover_dz up was already tried")
    assert "必须换一个做法：改别的地方，或改用 executor/card/patch/none。" in nag
    assert tried["kind"] == "none" and tried["detail"]["reason"] == "llm: repeated the same rejected answer"
    assert tried["detail"]["needs"] == ["proposal", "llm: repeated the same rejected answer"]


def test_a_patch_written_blind_is_rejected_with_the_module_text_attached(tmp_path):
    """Round 70 of the live campaign: the model answered patch WITH a payload on call 1, so
    the two-step never fired and it was told to copy `old` out of material it had never been
    shown. The material now rides that rejection."""
    from test_evolve_llm_e2e import _fake, _repeat_proj

    proj, before = _repeat_proj([])
    proj["first_death"]["modules"] = [MODULE]
    proj["module_sources"], _ = evolve_llm._module_sources([MODULE], MODULE)
    ep = _fake(tmp_path, [_patch(ABSENT),   # invented, with no source in front of it
                          {"kind": "executor", "payload": {"to": "alt"}, "summary": "换。", "rationale": "-"}],
               name="blind.json")
    tried, _ = evolve_llm.llm_propose(ep, proj, before, 2, tmp_path / "llm")
    audit = json.loads((tmp_path / "llm" / "round-2.json").read_text())
    assert audit["calls"] == 2 and len(audit["attempts"]) == 1
    assert [m["role"] for m in audit["messages"]] == ["system", "user", "user", "assistant", "user"]
    mat = audit["messages"][1]["content"]           # inserted after the system message, once
    assert mat.startswith("Materials (static):") and "  36|     STOP = 0.65" in mat
    assert audit["messages"][4]["content"].startswith("Your proposal was rejected")
    assert tried["kind"] == "executor" and tried["detail"]["to"] == "alt"
