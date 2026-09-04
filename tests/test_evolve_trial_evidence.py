"""Base lane: the model SEES ITS OWN CODE RUN.

Three things this covers, all measured failures of the recycle_cans campaign (110
rounds, nothing accepted): a patch that raised at runtime left only ``repr(exc)`` in a
reason string (round 104: ``self._last_d`` used and never initialised), a patch that
ran told the model nothing about what its code did in the simulator, and an edit whose
``old == new`` burned a round. Here: ``evolve.trial_evidence`` (the per-seed diff against
the baseline seed + the whole exception), ``evolve.self_check`` (the static read before
the sim) and one real end-to-end round pair over the fake stage card.
"""

from __future__ import annotations

import json

import pytest
from test_evolve_e2e import _CARD as _E2E_CARD
from test_evolve_patch_e2e import GOOD, MODULE
from test_mission_e2e import _Runtime

from scripts import evolve, evolve_llm

# ── trial_evidence: the diff against the baseline seed ────────────────────────────


def _row(step, phase, d_eef, d_base, base):
    return {"step": step, "phase": phase, "d_eef": d_eef, "d_base": d_base, "base": [*base, 0.0]}


def _suite(seed, node, series, steps, ok=False):
    return {"count": int(ok), "seeds": {str(seed): {"success": ok, "trail": [
        {"id": node, "ok": ok, "steps": steps, "kind": "segment",
         "trace": {"end": {"d_eef_target": series[-1]["d_eef"], "d_base_target": series[-1]["d_base"]},
                   "series": series}}]}}}


BASE = [_row(i, "approach", round(1.0 - 0.01 * i, 2), 1.03, (1.45, -1.73)) for i in range(1, 11)]
TRIAL = BASE[:3] + [_row(i, "drive", round(0.9 - 0.05 * i, 2), round(1.03 - 0.06 * i, 2),
                         (1.45 - 0.05 * i, -1.73)) for i in range(4, 13)]


def test_the_diff_says_what_the_trials_own_code_did_differently():
    ev = evolve.trial_evidence(_suite(1, "drop-can1", BASE, 40),
                               _suite(1, "drop-can1", TRIAL, 61), "drop-can1", [1])
    assert ev["node"] == "drop-can1" and ev["exception"] is None
    seed, = ev["seeds"]
    assert seed["seed"] == 1 and seed["trace"]["series"] is TRIAL   # the per-step evidence rides along
    # no failure_mode_* here: neither of these rows CARRIES the key, and a row nobody
    # measured must not turn into "无" downstream (see the absence test below).
    assert seed["diff"] == {
        "phase_changed": ["approach", "drive"], "first_divergent_step": 4, "base_moved": True,
        "d_eef_min_before": 0.9, "d_eef_min_after": 0.3, "d_base_min_before": 1.03,
        "d_base_min_after": 0.31, "steps_before": 40, "steps_after": 61, "ok_after": False}


def test_a_failure_mode_rides_only_when_that_side_actually_reported_one():
    """``failure_mode_after`` is the causal reading the distances never gave -- and the
    easiest one to fake. A candidate executor that reports no failure_mode leaves NO key
    (D.merge_executor_diagnostics), and the absence has to survive to _trial_line, which
    renders it 测不到. Over evolve-recycle_cans' 588 rounds the judged node read None on
    364 of the 365 candidate trial rows, 347 of them at the segment cap, while the same
    two nodes' scripted baseline rows carried the stall in 580 rounds: an unconditional
    key told the model "failure_mode reach_stall→无" in essentially every candidate round."""
    def suite(fm):
        s = _suite(1, "drop-can1", BASE, 40)
        row = s["seeds"]["1"]["trail"][0]
        if fm is not ...:
            row["failure_mode"] = fm
        return s

    scripted, silent = suite("reach_stall"), suite(...)
    answered = evolve.trial_evidence(scripted, suite(None), "drop-can1", [1])["seeds"][0]["diff"]
    assert answered["failure_mode_before"] == "reach_stall"   # an explicit None IS a reading
    assert "failure_mode_after" in answered and answered["failure_mode_after"] is None
    mute = evolve.trial_evidence(scripted, silent, "drop-can1", [1])["seeds"][0]["diff"]
    assert mute["failure_mode_before"] == "reach_stall" and "failure_mode_after" not in mute
    line = evolve_llm._trial_line({"node": "drop-can1", "exception": None,
                                   "seeds": [{"seed": 1, "diff": mute}]})
    assert "本轮测不到（执行器没交回这个读数）" in line and "→无" not in line


def test_a_trial_that_changed_nothing_says_so_and_a_missing_node_is_not_invented():
    ev = evolve.trial_evidence(_suite(1, "n", BASE, 40), _suite(1, "n", BASE, 40), "n", [1, 2])
    a, b = ev["seeds"]
    assert a["diff"]["phase_changed"] == [] and a["diff"]["first_divergent_step"] is None
    assert a["diff"]["base_moved"] is False and a["diff"]["steps_after"] == 40
    assert b["seed"] == 2 and "trace" not in b   # seed 2 never ran the node
    assert b["diff"]["steps_after"] is None and b["diff"]["base_moved"] is None


def test_the_summary_last_outcome_carries_drops_the_series_but_keeps_the_numbers():
    ev = evolve.trial_evidence(_suite(1, "n", BASE, 40), _suite(1, "n", TRIAL, 61), "n", [1],
                               exc={"type": "AttributeError", "message": "no _last_d"})
    s = evolve._evidence_summary(ev)
    assert s["exception"]["type"] == "AttributeError" and s["node"] == "n"
    assert s["seeds"] == [{"seed": 1, **ev["seeds"][0]["diff"]}]
    assert "trace" not in s["seeds"][0] and len(json.dumps(s)) < 600


# ── self_check: the static read before the simulator ──────────────────────────────

GOOD_SRC = """
class Drv:
    STOP = 0.4

    def __init__(self):
        self.n = 0
        setattr(self, "late", 1)

    def act(self, obs):
        self.n += 1
        return self.STOP + self.late + self._helper()

    def _helper(self):
        return 0
"""
BAD_SRC = GOOD_SRC.replace("self.STOP + self.late", "self.STOP + self._last_d")
CALL_SRC = GOOD_SRC.replace("self._helper()", "self._replan()")
BASED_SRC = "import unknown_thing_xyz\n\n\nclass Drv(unknown_thing_xyz.Base):\n    def act(self):\n        return self._last_d\n"


def _cand(tmp_path, src, name="drivers.py"):
    (tmp_path / name).write_text(src)
    return {"detail": {"path": str(tmp_path)}}


def test_self_check_refuses_state_the_class_never_initialises(tmp_path):
    assert evolve.self_check(_cand(tmp_path, GOOD_SRC)) is None
    why = evolve.self_check(_cand(tmp_path, BAD_SRC))
    assert why.startswith("doctor:self-check") and "Drv reads self._last_d" in why
    assert "drivers.py" in why and "never assigns" in why
    assert "self._replan" in evolve.self_check(_cand(tmp_path, CALL_SRC))   # a method that does not exist


def test_self_check_claims_nothing_about_a_class_whose_base_it_cannot_import(tmp_path):
    """No guessing: an unresolvable base could supply anything, so the class is skipped."""
    assert evolve.self_check(_cand(tmp_path, BASED_SRC)) is None


def test_self_check_refuses_an_edit_that_changes_nothing_and_a_file_that_does_not_parse(tmp_path):
    same = {"detail": {"path": str(tmp_path), "edits": [{"old": "x", "new": "y"}, {"old": "d", "new": "d"}]}}
    (tmp_path / "drivers.py").write_text(GOOD_SRC)
    assert evolve.self_check(same) == "doctor:self-check: edit 2 changes nothing: old == new"
    assert "does not parse" in evolve.self_check(_cand(tmp_path, "class :\n"))


# ── end to end: the round row and last_outcome carry it ───────────────────────────

_CARD = _E2E_CARD.replace("test_evolve_e2e:env_provider", "fakes.patch_stage:env_provider") \
    .replace("test_evolve_e2e:", "fakes.patch_stage:") \
    .replace("[task_bindings.e2e_evolve]", "[task_bindings.e2e_evidence]")
TASK = "e2e_evidence"
HIT = "        if self.STOP < 0.5:"


def _patch(name, new):
    return {"decision": "patch", "summary": "试一下。", "rationale": "grab never closes",
            "payload": {"name": name, "module": MODULE, "to": "patched",
                        "edits": [{"old": HIT, "new": new}]}}


#: A patch decision with no payload: call 1 of every patch round now, since the brief
#: carries no source and the answer must be written against the material (call 2).
_ASK = {"decision": "patch", "summary": "先要源码。", "rationale": "grab never closes"}

CANNED = [
    _ASK,   # round 1 call 1: the two-step answers with the material, no attempt spent
    # round 1, attempt 1: state the class never initialises -- refused with NO simulator
    _patch("uninit", "        if self._last_d < 0.5:"),
    # attempt 2: passes the doctor and the self-check, then raises inside act on the preflight seed
    _patch("raiser", '        raise RuntimeError("act blew up")\n' + HIT),
    # attempt 3 (the last): an honest none -- the round ends with the raise as its finding
    {"decision": "none", "summary": "先停。", "rationale": "两次都失败了，先停下"},
    _ASK,   # round 2 call 1
    # round 2: the real fix -- it runs, wins, and the evidence says what it did
    {"decision": "patch", "summary": "把 STOP 调小。", "rationale": "the standoff never closes",
     "payload": {"name": "grab_stop", "module": MODULE, "to": "patched", "edits": GOOD}},
]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    rt = _Runtime(runs, card=_CARD, canned=CANNED, mode="evolution",
                  env={"PH_CANDIDATES_ROOT": str(runs / "candidates")})
    rt.campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    try:
        rt.run({"kind": "evolve", "task": TASK, "seeds": [1, 2], "rounds": 1, "arm": "auto"})
        yield rt
    finally:
        rt.stop()


def test_a_raise_inside_the_candidate_is_the_rounds_finding_not_a_repr(runtime):
    doc = json.loads(runtime.campaign.read_text())
    r1 = doc["rounds"][0]
    audit = json.loads((runtime.campaign.parent / "llm" / "round-1.json").read_text())
    # the static rejection rode the repair loop, and it cost no simulator seed (whichever
    # door fired: write_patch checks a patch first, evolve.self_check is the last gate)
    assert "self._last_d" in audit["attempts"][0]["reason"]   # whichever door names it
    assert "RuntimeError" in audit["attempts"][1]["reason"] and "act blew up" in audit["attempts"][1]["reason"]
    exc = r1["trial_evidence"]["exception"]
    assert (exc["type"], r1["tried"]["kind"], r1["accepted"]) == ("RuntimeError", "none", False)
    assert exc["message"] == "act blew up" and exc["line"] > 0
    assert exc["file"].endswith("patch_stage.py") and any("act blew up" in t for t in exc["traceback"])
    assert 0 < len(exc["traceback"]) <= 15
    # and the NEXT round's model actually read it: last_outcome carried it into the brief
    nxt = json.loads((runtime.campaign.parent / "llm" / "round-2.json").read_text())
    assert nxt["brief"]["last_outcome"]["trial_evidence"]["exception"]["type"] == "RuntimeError"
    assert "act blew up" in json.dumps(nxt["brief"]["last_outcome"]["trial_evidence"])


def test_a_candidate_that_runs_reports_the_before_after_diff(runtime):
    doc = json.loads(runtime.campaign.read_text())
    r2 = doc["rounds"][1]
    ev = r2["trial_evidence"]
    assert (r2["tried"]["kind"], r2["accepted"], r2["before"], r2["after"]) == ("card", True, 0, 2)
    assert ev["exception"] is None and ev["node"] == r2["tried"]["node"]
    assert [s["seed"] for s in ev["seeds"]] == [1, 2]
    for s in ev["seeds"]:   # the fake stage carries no per-step series: the steps still diff
        # and no failure_mode either side -- it never seals one, so no key is invented
        assert set(s["diff"]) == {"phase_changed", "first_divergent_step", "base_moved",
                                  "d_eef_min_before", "d_eef_min_after", "d_base_min_before",
                                  "d_base_min_after", "steps_before", "steps_after", "ok_after"}
        assert s["diff"]["steps_before"] and s["diff"]["steps_after"]
    assert doc["last_outcome"]["trial_evidence"]["seeds"][0]["steps_after"]
