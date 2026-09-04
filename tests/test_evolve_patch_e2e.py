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
import importlib
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
    assert a2["messages"][4]["content"].startswith("You decided patch")
    # the brief LEADS with what round 1's knob actually did in the simulator: nothing
    assert a2["messages"][2]["content"].startswith(
        "你上一轮的补丁跑了：种子 1 在 grab-0 跑到第 8 步，与基线逐步完全相同＝你的改动没有生效")
    assert "\n\nRound input:\n" in a2["messages"][2]["content"]
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
    assert nag.startswith("你重复了一条已经被拒的回答（ValueError: hover_dz up was already tried")
    assert "必须换一个做法：改别的地方，或改用 executor/card/patch/none。" in nag
    assert tried["kind"] == "none" and tried["detail"]["reason"] == "llm: repeated the same rejected answer"
    assert tried["detail"]["needs"] == ["proposal", "llm: repeated the same rejected answer"]


def test_a_patch_written_blind_is_rejected_with_the_module_text_attached(tmp_path):
    """Round 70 of the live campaign: the model answered patch WITH a payload on call 1, so
    the two-step never fired and it was told to copy `old` out of material it had never been
    shown -- and that rejection cost one of the three attempts (108 times, 18.6M tokens).
    A patch answer now takes the material FIRST whether or not it brought a payload: the
    blind one is dropped for free and call 2 answers with the source in front of it."""
    from test_evolve_llm_e2e import _fake, _repeat_proj

    proj, before = _repeat_proj([])
    proj["first_death"]["modules"] = [MODULE]
    proj["module_sources"], _ = evolve_llm._module_sources([MODULE], MODULE)
    ep = _fake(tmp_path, [_patch(ABSENT),   # invented, with no source in front of it
                          {"kind": "executor", "payload": {"to": "alt"}, "summary": "换。", "rationale": "-"}],
               name="blind.json")
    tried, _ = evolve_llm.llm_propose(ep, proj, before, 2, tmp_path / "llm")
    audit = json.loads((tmp_path / "llm" / "round-2.json").read_text())
    assert audit["calls"] == 2 and audit["attempts"] == []   # the blind patch cost no attempt
    assert [m["role"] for m in audit["messages"]] == ["system", "user", "user", "assistant", "user"]
    mat = audit["messages"][1]["content"]           # inserted after the system message, once
    assert mat.startswith("Materials (static):") and "  36|     STOP = 0.65" in mat
    assert audit["messages"][4]["content"].startswith("You decided patch:")
    assert tried["kind"] == "executor" and tried["detail"]["to"] == "alt"


UNINIT = ("from harness.skill_executor import InprocExecutor\n\n\n"
          "class _E(InprocExecutor):\n    def bind(self, env, target=None):\n        pass\n\n"
          "    def act(self, obs):\n        return (0.0,) if self._last_d else (0.1,)\n\n\n"
          "class _P:\n    def make_driver(self, spec):\n        return _E()\n\n\n"
          "def provider(**params):\n    return _P()\n")


def _preflight(tried):
    """What scripts/evolve.py's preflight does: the static self-check first (a finding is
    raised where the doctor's are, so it rides the repair loop), then the seed."""
    from scripts import evolve

    if why := evolve.self_check(tried):
        raise evolve.SelfCheckError(why)


def _card(name, code, to="llm"):
    from test_evolve_llm_e2e import MANIFEST

    return {"kind": "card", "payload": {"name": name, "to": to, "ref": f"{name}:provider",
            "files": {"manifest.toml": MANIFEST.format(to=to, name=name), "__init__.py": code}},
            "summary": "写一个执行器。", "rationale": "-", "layer": "state"}


ELSEWHERE = {"kind": "executor", "payload": {"to": "alt"}, "summary": "换执行器。", "rationale": "-",
             "layer": "plan"}


def test_the_self_check_finding_comes_back_with_the_candidates_own_numbered_source(tmp_path, monkeypatch):
    """Rounds 104 and 108 wrote code reading state nothing assigns: the finding is what the
    model must see, together with the line of ITS OWN candidate copy that reads it."""
    from test_evolve_llm_e2e import _fake, _repeat_proj

    monkeypatch.setattr(evolve_llm, "CANDIDATES_ROOT", tmp_path / "cands")
    proj, before = _repeat_proj([])
    ep = _fake(tmp_path, [_card("grab_uninit", UNINIT), ELSEWHERE], name="uninit.json")
    tried, _ = evolve_llm.llm_propose(ep, proj, before, 4, tmp_path / "llm", preflight=_preflight)
    a4 = json.loads((tmp_path / "llm" / "round-4.json").read_text())
    why = a4["attempts"][0]["reason"]
    assert why.startswith("doctor:self-check (static, no sim): __init__.py: _E reads self._last_d")
    assert "__init__.py -- YOUR OWN code, around line 9 (fix THIS, not the installed source" in why
    assert "   9|         return (0.0,) if self._last_d else (0.1,)" in why
    assert why in a4["messages"][4]["content"]      # the repair message, verbatim
    assert (tried["kind"], tried["detail"]["to"]) == ("executor", "alt")
    assert a4["attempts"][0]["sha"]                 # remembered for the next rounds


def test_the_same_rejected_answer_a_later_round_is_refused_with_the_round_it_failed_in(tmp_path, monkeypatch):
    """Across ROUNDS, not only within one: round 108 re-sent round 104's patch and the
    harness paid for it a second time."""
    from test_evolve_llm_e2e import _fake, _repeat_proj

    monkeypatch.setattr(evolve_llm, "CANDIDATES_ROOT", tmp_path / "cands")
    proj, before = _repeat_proj([])
    bad = _card("grab_uninit", UNINIT)
    evolve_llm.llm_propose(_fake(tmp_path, [bad, ELSEWHERE], name="r4.json"), proj, before, 4,
                           tmp_path / "llm", preflight=_preflight)
    tried, _ = evolve_llm.llm_propose(_fake(tmp_path, [bad, bad], name="r5.json"), proj, before, 5,
                                      tmp_path / "llm", preflight=_preflight)
    a5 = json.loads((tmp_path / "llm" / "round-5.json").read_text())
    assert a5["attempts"] == []      # never doctored, never preflighted, no simulator
    assert a5["repeats"][0]["reason"].startswith("第 4 轮已经提过同一条回答并被拒：doctor:self-check")
    assert "第 4 轮已经提过同一条回答并被拒" in a5["messages"][3]["content"]
    assert tried["detail"]["reason"] == evolve_llm.REPEATED


def test_a_raising_trial_comes_back_with_the_exception_and_the_candidates_own_numbered_source(tmp_path, monkeypatch):
    """A trial that raises inside the model's own code: the traceback AND the numbered lines
    of the candidate copy, so the repair is to that code and not to the installed source."""
    from test_evolve_llm_e2e import _EXEC, _fake, _repeat_proj

    monkeypatch.setattr(evolve_llm, "CANDIDATES_ROOT", tmp_path / "cands")
    code = _EXEC.format(act="(0.0,)") + ('\n\ndef boom():\n    d = 0.0\n    raise AttributeError('
                                         '"\'GrabStage\' object has no attribute \'_last_d\'")\n')
    proj, before = _repeat_proj([])
    ep = _fake(tmp_path, [_card("grab_boom", code), ELSEWHERE], name="boom.json")
    tried, _ = evolve_llm.llm_propose(ep, proj, before, 6, tmp_path / "llm",
                                      preflight=lambda t: importlib.import_module("grab_boom").boom())
    why = json.loads((tmp_path / "llm" / "round-6.json").read_text())["attempts"][0]["reason"]
    assert why.startswith("preflight: the trial raised on seed")
    assert "AttributeError: 'GrabStage' object has no attribute '_last_d'" in why
    assert "__init__.py -- YOUR OWN code, around line" in why
    assert '|     raise AttributeError("\'GrabStage\' object has no attribute \'_last_d\'")' in why
    assert (tried["kind"], tried["detail"]["to"]) == ("executor", "alt")


def test_state_init_names_where_new_state_belongs():
    """The rules tell a patch to initialise new state in the class's EXISTING constructor /
    reset: the materials name those methods and what they already set."""
    from fakes.patch_stage import Driver, GrabStage

    assert evolve_llm._state_init([GrabStage, Driver]) == {
        f"{MODULE}:GrabStage.__init__": ["target"],
        f"{MODULE}:Driver.__init__": ["_cap", "_env", "_ex", "_native", "_stage", "n"]}


def test_the_brief_leads_with_what_the_last_change_actually_did_in_the_simulator(tmp_path):
    """The model could not see whether its own patch did anything: trial_evidence is the
    first thing it reads, and the rules carry the patch checklist."""
    from test_evolve_llm_e2e import _fake, _repeat_proj

    ev = "种子 4243 在 drop-can1 抛 AttributeError: '_last_d'（recycle_driver.py:212）"
    line = evolve_llm._trial_evidence({"last_outcome": {"round": 104, "trial_evidence": ev}}, [])
    assert line == f"你上一轮的补丁跑了：{ev}"
    assert evolve_llm._trial_evidence({}, [{"trial_evidence": ["跑到第 41 步", "d_eef 最小 0.57→0.55"]}]) \
        == "你上一轮的补丁跑了：跑到第 41 步；d_eef 最小 0.57→0.55"
    proj, before = _repeat_proj([])
    proj["trial_evidence"] = line
    assert evolve_llm.brief(proj)["trial_evidence"] == line
    ok = {"kind": "executor", "payload": {"to": "alt"}, "summary": "换。", "rationale": "-"}
    evolve_llm.llm_propose(_fake(tmp_path, [ok], name="ev.json"), proj, before, 7, tmp_path / "llm")
    msg = json.loads((tmp_path / "llm" / "round-7.json").read_text())["messages"][1]["content"]
    assert msg.startswith(line + "\n\nRound input:")
    for must in ("补丁自检清单", "state_init", "old != new", "最近 5 条**跨轮**记着", "repeat_failure"):
        assert must in evolve_llm._RULES


def test_three_runtime_error_rounds_on_one_node_ask_for_a_smaller_self_contained_change():
    from test_evolve_llm_e2e import _repeat_proj

    hist = [{"round": r, "tried": {"kind": "card", "node": "grab-0", "detail": {}},
             "trial_evidence": "种子 1 在 grab-0 抛 AttributeError: '_last_d'"} for r in (1, 2, 3)]
    proj, _ = _repeat_proj(hist)
    b = evolve_llm.brief(proj)
    assert "grab-0 连续 3 轮死在运行时错误" in b["repeat_failure"]
    assert "一个守卫" in b["repeat_failure"] and "另一个节点" in b["repeat_failure"]
    proj, _ = _repeat_proj(hist[:2])   # two in a row is not yet a loop
    assert "repeat_failure" not in evolve_llm.brief(proj)


def test_an_already_run_patch_is_refused_with_what_that_round_measured():
    """The gap that ate a 588-round campaign: a patch that APPLIES, RUNS and merely fails to
    raise the score is in no reject list, and the model never sees its own past code -- so it
    re-derived the same drop-point clamp for ~300 rounds. The key is the patched MODULE's AST
    dump, not its text (the re-wording is why: 363 patch rounds, 319 distinct files, 215
    distinct ASTs), and the refusal carries the verdict."""
    hist = [{"round": 12, "verdict": "focused trial: drop-can1 passed on no seed of its cluster",
             "tried": {"kind": "card", "node": "drop-can1",
                       "detail": {"module": "m", "patch_sha": "deadbeefdeadbeef"}}}]
    ran = evolve_llm._ran_patches({"history": hist})
    assert ran["deadbeefdeadbeef"].startswith("round 12: focused trial: drop-can1")
    assert evolve_llm._ran_patches({"history": [{"round": 1, "tried": {"detail": {}}}]}) == {}


# The live 4243 numbers: the drop point, the base the carry leg parked, and the dock that
# leg drove at (the carry row's trace_end.target), verbatim off round 200's 4243 shard.
# THREE different distances live here and the brief used to blur them: carry_stop 0.65 is the
# COMMAND, the carry leg realised 0.644, and the base_nudge recovery then pulled the base in
# to 0.596. The point is 1.028 from the base against a 0.664 reach.
_GEO = {"d_base_point": 1.028, "reach_max": 0.664,
        "base": [1.45, -1.728, -0.473], "point": [0.488, -1.364, 0.97]}
_UP = {"node": "carry-can1", "trace_end": {"target": [0.86, -1.815, 0.0], "d_base_target": 0.644}}
_DRV = {"drop-can1": {"node": "drop-can1",
                      "stage_modules": ["plugins.embodiment_robocasa.recycle_driver",
                                        "plugins.embodiment_robocasa.stage_extras"],
                      "tunables": {"ref": "r", "values": {"carry_stop": 0.65, "nudge_max": 0.15}}}}


def _wall_proj(head="drop-can1"):
    trail = [{"id": "drop-can1", "ok": False, "geometry": dict(_GEO), "upstream": dict(_UP)}]
    return {"first_death": {"node": head}, "drivers": {k: dict(v) for k, v in _DRV.items()},
            "this_round": {"per_seed": [{"seed": 4243, "trail": trail}]}}


def test_the_reach_wall_names_only_levers_that_exist_and_says_how_far_the_dock_must_move():
    """A drop point farther from the base than the arm reaches cannot be fixed inside an
    arm-only stage, and clamping it into reach moves it off the fixture the predicate scores
    (both measured on recycle_cans/4243). The wall must (1) show on every round, not only the
    ones the rotation put on this node -- it fired on 505 of 588 live rounds, silent only on
    1..83 where no geometry rode the trail yet -- (2) name ONLY levers that this module's own
    gates let through: not "insert a navigation node" (no answer kind edits the plan graph)
    and not a patch to drivers.VCAP (``drivers`` is not in drop-can1's stage_modules, so
    write_patch refuses it on sight), and (3) keep the three distances APART -- the knob
    command, what the leg realised, where the base ended up -- so "carry_stop <= 0.134" (which
    is 0.054 short of the mark) becomes "carry_stop ~ 0.188", with whether the leg can even
    deliver it said out loud."""
    proj = _wall_proj()
    wall = evolve_llm._reach_wall(proj)
    assert "1.028" in wall and "0.664" in wall and "0.364" in wall
    assert "carry-can1" in wall and "placed" in wall
    assert "导航节点" not in wall                       # (c) was never a lever: deleted
    assert "0.134" in wall and "0.596" in wall         # the standoff bound, and the standoff
    assert "0.644" in wall and "0.65" in wall          # what the leg realised vs what it was told
    assert "0.188" in wall                             # ...so THIS is the number to set it to
    assert "0.6–0.8" in wall                           # and the measured reason it may not stick
    assert "nudge_max" in wall and "base_nudge" in wall and "0.15" in wall
    assert "stage_modules" in wall and "驳回" in wall    # (b)'s VCAP half is named as a NON-lever
    assert all(k in wall for k in evolve_llm.WALL_KNOBS)   # what _try reopens is what it names
    # the rotation head is the OTHER node half the time; the wall still reports its own row
    assert evolve_llm._reach_wall(_wall_proj(head="nav-can1")) == wall
    proj["this_round"]["per_seed"][0]["trail"][0]["geometry"] = {"d_base_point": 0.4, "reach_max": 0.664}
    assert evolve_llm._reach_wall(proj) is None         # inside reach: no wall


def test_the_dock_bound_is_the_band_between_both_roots():
    """``|dock + s*u - point| <= reach_max`` is a quadratic: the standoffs that reach are the
    CLOSED INTERVAL between its roots, and a standoff under the lower root has driven PAST the
    point and is out of reach on the other side. On the live 4243 row the lower root is -0.739
    (behind the dock) so only the upper one binds -- but the docstring's promise ("solved") has
    to hold when it does not, so the second root is returned and the brief says "between"."""
    assert evolve_llm._dock_bound(_GEO, _UP) == (-0.739, 0.134, 0.596)
    # synthetic ray with the point in FRONT of the dock, where both roots bind:
    # dock (0,0) -> base (2,0), point (1,0), reach 0.3 -> the band is [0.7, 1.3]
    geo = {"d_base_point": 1.0, "reach_max": 0.3, "base": [2.0, 0.0, 0.0], "point": [1.0, 0.0, 0.0]}
    up = {"node": "carry-can1", "trace_end": {"target": [0.0, 0.0, 0.0], "d_base_target": 2.0}}
    assert evolve_llm._dock_bound(geo, up) == (0.7, 1.3, 2.0)
    wall = evolve_llm._reach_wall({**_wall_proj(), "this_round": {"per_seed": [
        {"seed": 4243, "trail": [{"id": "drop-can1", "ok": False, "geometry": geo, "upstream": up}]}]}})
    assert "落在 0.7–1.3 m 之间" in wall          # not "<= 1.3": below 0.7 is out of reach again
    assert evolve_llm._dock_bound({**_GEO, "reach_max": 0.1}, _UP) is None   # never reaches


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


def test_the_parameter_layer_reopens_at_the_reach_wall():
    """A geometric gap closes on a NUMBER (the upstream standoff, nudge_max) and nothing else,
    so refusing the parameter layer there hands the model a brief that points at a knob with
    one hand and forbids it with the other -- the state the campaign sat in from r55 on. BOTH
    gates have to give way: reopening the LAYER and then refusing the two knobs on ``seen``
    only moves the contradiction one gate down. The campaign spent carry_stop (r46 0.7, r52
    0.5) and nudge_max (r45 0.2, r51 0.1) blind in its first 55 rounds, before any geometry
    rode the brief, and named neither knob in the 505 rounds that had the numbers."""
    proj = _wall_proj()
    knobs = {"tunables": {"ref": "r", "path": [], "values": {"carry_stop": 0.65, "drop_dz": 0.05}},
             "executors": {"scripted": {}}}
    proj["first_death"] |= knobs        # what the brief reads
    proj["drivers"]["drop-can1"] |= knobs   # ...and what _try answers against (payload.node)
    # the campaign's own history, the way _tried_pairs reads it: every knob x direction spent
    proj["history"] = [{"round": i, "tried": {"kind": "tunables", "node": "drop-can1",
                                              "detail": {"ref": "r", "path": [k], "from": v, "to": t}}}
                       for i, (k, v, t) in enumerate([("carry_stop", 0.65, 0.7), ("carry_stop", 0.65, 0.5),
                                                      ("drop_dz", 0.05, 0.08), ("drop_dz", 0.05, 0.02)], 1)]
    ans = lambda knob, to: {"kind": "tunables", "layer": "parameter", "rationale": "close the gap",
                            "payload": {"ref": "r", "path": [knob], "to": to}}
    before = {"count": 0, "seeds": {"4243": {"first_death": "drop-can1", "nodes": {
        "drop-can1": {"skill": "drop_can1", "executor": "scripted", "success": False}}}}}
    assert evolve_llm._exhausted(proj, evolve_llm._untried(proj, evolve_llm._tried_pairs(proj)))
    # the wall's own knob goes all the way through BOTH gates, spent or not
    tried = evolve_llm._try(ans("carry_stop", 0.188), proj, before, 1, None)
    assert tried["kind"] == "tunables" and tried["detail"]["to"] == 0.188
    # ...and only it: a knob the wall does not name is still refused on the campaign's history
    # (a NEW number, so it is the direction gate talking, not the exact-value one below)
    with pytest.raises(ValueError, match="was already tried"):
        evolve_llm._try(ans("drop_dz", 0.09), proj, before, 1, None)
    # ...and the wall's own knob is answerable but not REPEATABLE: the wall hands the model a
    # number, so a direction is not an answer there -- but the same number twice buys nothing
    # and costs a whole suite, and the wall stands in 505 of the campaign's 588 rounds.
    proj["history"].append({"round": 5, "tried": {"kind": "tunables", "node": "drop-can1",
                                                  "detail": {"ref": "r", "path": ["carry_stop"],
                                                             "from": 0.65, "to": 0.188}}})
    with pytest.raises(ValueError, match="already set carry_stop to 0.188"):
        evolve_llm._try(ans("carry_stop", 0.188), proj, before, 1, None)
    tried = evolve_llm._try(ans("carry_stop", 0.21), proj, before, 1, None)   # a new number: open
    assert tried["kind"] == "tunables" and tried["detail"]["to"] == 0.21
    # ...and the round still cannot repeat ITSELF, wall or no wall
    mine: set = set()
    evolve_llm._try(ans("carry_stop", 0.30), proj, before, 1, None, seen=mine)
    with pytest.raises(ValueError, match="was already tried"):
        evolve_llm._try(ans("carry_stop", 0.15), proj, before, 1, None, seen=mine)
    # the brief says the same thing the gate does, instead of "the layer is CLOSED" alone
    ex = evolve_llm.brief(proj)["exhausted"]
    assert "parameter layer is CLOSED" in ex and "carry_stop / nudge_max 仍然可以再答一次" in ex
    proj["this_round"]["per_seed"] = []    # no wall: the layer is closed as before
    with pytest.raises(ValueError) as e:
        evolve_llm._try(ans("carry_stop", 0.33), proj, before, 1, None)
    assert "parameter layer is CLOSED" in str(e.value)
    assert evolve_llm.WALL_OPEN not in evolve_llm.brief(proj)["exhausted"]


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
            "history": [], "this_round": {"per_seed": []}}
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
    tried = evolve_llm._try(ans, proj, before, 9, None)
    assert tried["node"] == "grab-0" and tried["detail"]["skill"] == "grab"
    cand = tmp_path / "cands" / "grab_stop_node"
    assert '[executors.patched]\nskill = "grab"' in (cand / "manifest.toml").read_text()
    assert 'TASK = "grab"' in (cand / "__init__.py").read_text()      # the head would have said "reach"
    applied = {"executors": {"reach-0": "scripted", "grab-0": "scripted"}, "tunables": {}, "cards": {}}
    assert evolve.apply(tried, applied)["executors"] == {"reach-0": "scripted", "grab-0": "patched"}


def test_the_refusals_measure_the_answer_against_the_node_it_names():
    """``fd`` followed payload.node; the tables it is CHECKED against stayed on the rotation
    head. So a ``none`` on the answered node was refused with the head's knobs ("these are
    still untried on grab-0" listing reach-0's), an executor already tried on the head
    counted as tried on the other node, and ``exhausted`` closed the parameter layer on the
    wrong one. 86 of the live campaign's 568 shard rounds answered off the head."""
    proj, before = _two_death_proj()
    proj["drivers"]["reach-0"]["tunables"] |= {"ref": "reach:p", "values": {"standoff": 1.0}}
    proj["drivers"]["grab-0"]["tunables"] |= {"ref": "grab:p", "values": {"grip": 1.0}}
    for n in proj["drivers"]:
        proj["drivers"][n]["executors"] = {"scripted": {}, "vla": {}}
    proj["first_death"] = proj["drivers"]["reach-0"]                  # the rotation head
    proj["history"] = [{"round": 1, "tried": {"kind": "card", "node": "reach-0", "detail": {"to": "vla"}}}]
    none = lambda node: {"kind": "none", "summary": "-", "rationale": "-", "payload": {"node": node}}
    with pytest.raises(ValueError) as e:
        evolve_llm._try(none("grab-0"), proj, before, 2, None)
    why = str(e.value)
    assert "untried on grab-0" in why and "tunables grip down" in why and "standoff" not in why
    assert "executor vla" in why          # tried on reach-0, never on grab-0
    with pytest.raises(ValueError) as e:
        evolve_llm._try(none("reach-0"), proj, before, 2, None)
    why = str(e.value)
    assert "untried on reach-0" in why and "tunables standoff down" in why and "grip" not in why
    assert "executor" not in why          # the head's only other executor is the one it tried


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
    tried = evolve_llm._try(ans, proj, before, 9, None)
    sha = tried["detail"]["patch_sha"]
    assert sha and len(sha) == 16
    row = evolve.index_row({"round": 9, "tried": tried, "before": 0, "after": 0})
    assert row["tried"]["detail"]["patch_sha"] == sha          # survives the index row...
    proj["history"] = [{"round": 9, "verdict": "it did not raise the score",
                        "tried": {"kind": row["tried"]["kind"], "node": row["tried"]["node"],
                                  "detail": {"patch_sha": row["tried"]["detail"]["patch_sha"]}}}]
    assert evolve_llm._ran_patches(proj) == {sha: "round 9: it did not raise the score"}
    reworded = [{"old": GOOD[0]["old"], "new": "# a different comment entirely\nSTOP = 0.4"}]
    ans2 = {**ans, "payload": {**ans["payload"], "name": "p2", "to": "p2", "edits": reworded}}
    with pytest.raises(ValueError) as e:
        evolve_llm._try(ans2, proj, before, 10, None)
    assert "ALREADY RAN -- round 9: it did not raise the score" in str(e.value)
