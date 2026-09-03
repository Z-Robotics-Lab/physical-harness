"""The LLM proposer inside the evolve loop, end to end through the REAL runtime +
scripts/evolve.py on the test_evolve_e2e task card, the model_endpoint FAKE answering
a fixed sequence (PH_MODEL_ENDPOINT_FAKE): round 1 a tunables answer becomes the
round's try (proposer "llm", summary kept); round 2 a card answer -- a tiny
code-as-policy candidate for the fake embodiment -- passes the doctor, is mounted for
the trial suite and wins; round 3 an unparseable answer falls back to the rules with
the reason; round 4 a card that fails the doctor is recorded honestly, files kept."""

from __future__ import annotations

import json

import pytest
from test_evolve_e2e import _CARD, EMB, TASK
from test_mission_e2e import _Runtime, _kinds

from board import store as bs

GOOD = ("class _P:\n    def make_driver(self, spec):\n        return object()\n\n\n"
        "def provider(**params):\n    return _P()\n")
BAD_MANIFEST = (f'needs_sim = false\n[executors.bad]\nskill = "grab"\nembodiment = "{EMB}"\n'
                'ref = "grab_bad:provider"\ntransport = "inproc"\n')
CANNED = [
    {"kind": "tunables", "payload": {"ref": "test_evolve_e2e:policy_provider",
                                     "path": ["tunables", "stall_k"], "to": 28},
     "summary": "两颗种子都死在 grab-0。", "rationale": "先把 stall_k 调低"},
    {"kind": "card", "payload": {"name": "grab_llm", "to": "llm", "ref": "grab_llm:provider",
                                 "files": {"manifest.toml": "needs_sim = false\n", "__init__.py": GOOD}},
     "summary": "调参无效，写一个新的 grab 执行器。", "rationale": "code-as-policy"},
    "not json at all",
    {"kind": "card", "payload": {"name": "grab_bad", "to": "bad", "ref": "grab_bad:provider",
                                 "files": {"manifest.toml": BAD_MANIFEST, "__init__.py": "def provider(:\n"}},
     "summary": "再试一个执行器。", "rationale": "broken on purpose"},
]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    rt = _Runtime(runs, card=_CARD, canned=CANNED, mode="evolution",
                  env={"PH_CANDIDATES_ROOT": str(runs / "candidates")})
    rt.campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    try:
        rt.rows = rt.run({"kind": "evolve", "task": TASK, "seeds": [1, 2], "rounds": 4, "arm": "auto"})[1]
        yield rt
    finally:
        rt.stop()


def test_llm_answers_drive_the_rounds_and_fall_back_honestly(runtime):
    doc = json.loads(runtime.campaign.read_text())
    r1, r2, r3, r4 = doc["rounds"]
    # round 1: the tunables answer is the try; summary / provenance on the row, never the key
    assert (r1["proposer"], r1["tried"]["kind"], r1["tried"]["node"]) == ("llm", "tunables", "grab-0")
    assert (r1["tried"]["detail"]["path"], r1["tried"]["detail"]["to"]) == (["tunables", "stall_k"], 28)
    assert r1["llm"]["summary"] == "两颗种子都死在 grab-0。" and r1["llm"]["rationale"] == "先把 stall_k 调低"
    assert len(r1["llm"]["prompt_sha"]) == 64 and len(r1["llm"]["raw_sha"]) == 64
    assert r1["llm"]["model"].startswith("fake(") and r1["llm"]["reason"] is None
    assert (r1["before"], r1["after"], r1["published"]) == (0, 0, False)
    # round 2: the model's candidate card passed the doctor, was mounted, and won
    assert (r2["proposer"], r2["tried"]["kind"], r2["tried"]["detail"]["to"]) == ("llm", "card", "llm")
    assert r2["tried"]["detail"]["path"] == str(runtime.runs / "candidates" / "grab_llm")
    assert (r2["before"], r2["after"], r2["published"]) == (0, 2, True)
    assert doc["applied"]["cards"]["llm"]["ref"] == "grab_llm:provider"
    rec = json.loads((runtime.session / "skills" / f"{r2['tried']['detail']['digest']}.json").read_text())
    assert rec["bindings"][EMB]["policies"]["llm"]["ref"] == "grab_llm:provider"
    # round 3: not JSON -> the rules proposer, with the reason
    assert r3["proposer"] == "rules" and r3["tried"]["kind"] == "none"
    assert "no JSON object" in r3["llm"]["reason"] and r3["llm"]["summary"] is None
    assert r3["llm"]["raw_sha"] and r3["llm"]["prompt_sha"]
    # round 4: a card the doctor rejects is an honest none; the files stay for the operator
    assert r4["proposer"] == "llm" and r4["tried"]["kind"] == "none"
    assert r4["tried"]["detail"]["reason"].startswith("doctor:executor:bad SyntaxError")
    assert (runtime.runs / "candidates" / "grab_bad" / "__init__.py").read_text() == "def provider(:\n"
    assert r4["llm"]["summary"] == "再试一个执行器。"
    # audit: the raw answer per round, outside the chain
    audits = {p.name: json.loads(p.read_text()) for p in (runtime.campaign.parent / "llm").glob("round-*.json")}
    assert set(audits) == {f"round-{r}.json" for r in (1, 2, 3, 4)}
    assert audits["round-3.json"]["raw"] == "not json at all"
    assert "first_death" in audits["round-1.json"]["messages"][1]["content"]
    # the round rows ride rsi_step / rsi_series; the live log said what the LLM was doing
    steps = _kinds(runtime.rows, "rsi_step")
    assert [s["proposer"] for s in steps] == ["llm", "llm", "rules", "llm"]
    assert steps[0]["llm"] == r1["llm"]
    assert [s["proposer"] for s in bs.rsi_series(runtime.session, TASK)] == ["llm", "llm", "rules", "llm"]
    assert any(m["text"].startswith("LLM 分析第") for m in doc["live"]["messages"])


def test_parse_unwraps_a_payload_nested_under_its_kind():
    """Seen live from DeepSeek: ``{"payload": {"executor": {"to": "alt"}}}``."""
    from scripts.evolve_llm import _parse
    ans = _parse(json.dumps({"kind": "executor", "payload": {"executor": {"to": "alt"}}, "summary": "换"}))
    assert ans["payload"] == {"to": "alt"} and ans["rationale"] == ""
    with pytest.raises(ValueError, match="kind must be"):
        _parse(json.dumps({"goal": "a planner reply", "nodes": []}))
