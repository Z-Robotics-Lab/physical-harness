"""The proposals inbox end to end: ``submit_proposal`` / ``proposals`` byte-equal
across the three faces, the REAL runtime + REAL scripts/evolve.py consuming a
``tunables`` proposal (round 1: the override reaches the driver, 0/2 -> 2/2,
accepted for development, sealed ``rsi_proposal_applied``) then a ``card`` proposal (round 2: the
candidate dir mounted through PH_PLUGINS_EXTRA, its executor forced), and
plugin_doctor GREEN on a candidate card directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_mission_e2e import SESSION, _kinds, _Runtime

from board import mcp_server as ms
from board import store as bs
from board import storecli
from harness import fakes, protocol
from harness.manifest import discover, mount_params
from harness.skill_executor import InprocExecutor
from harness.skill_library import segment_specs
from scripts import evolve, plugin_doctor

EMB = "test_proposals_e2e:env_provider"
POLICY = "test_proposals_e2e:policy_provider"
CAND = "test_proposals_e2e:cand_provider"

RECORDS = {
    "reach": {"id": "reach", "name": "reach", "kind": "segment", "args": {},
              "bindings": {EMB: {"task": "reach"}}},
    "grab": {"id": "grab", "name": "grab", "kind": "segment", "args": {},
             "bindings": {EMB: {"policies": {"scripted": {"task": "grab"}}}}},
}
CATALOGUE = {"reach": {}, "grab": {}}
ORACLES = ("seg_ok",)
EPISODE = {"task": "reach", "horizon": 20}
SEGMENT_SPECS = segment_specs(
    {k: protocol.SkillRecordV0.from_dict(v) for k, v in RECORDS.items()}, EMB)

_OBS = {"robot0_gripper_qpos": [0.03, -0.03], "robot0_gripper_qvel": [0.0, 0.0],
        "robot0_joint_vel": [0.0] * 7, "robot0_eef_pos": [0.0, 0.0, 1.0],
        "cubeA_pos": [0.0, 0.0, 0.1]}


class _Handle(fakes._FakeEnvHandle):
    def reset(self):
        super().reset()
        self.achieved = set()
        return dict(_OBS)

    def step(self, action):
        assert len(action) == 1
        self.t += 1
        if action[0] == 1.0:
            self.achieved.add("reach")
        elif action[0] == 2.0 and "reach" in self.achieved:
            self.achieved.add("grab")
        return dict(_OBS), 0.0, False, {}


class _Env:
    def make_env(self, spec):
        return _Handle()

    def tasks(self):
        return ("reach", "grab")

    def object_key(self, spec):
        raise AssertionError("heterogeneous segment path never reads object_key")

    def success(self, obs, spec, start_z):
        return True

    def terminal_success(self, obs, spec, start_z, env=None):
        return {"reach", "grab"} <= env.achieved


class _Driver(InprocExecutor):
    """grab succeeds only when the ``grip`` tunable (a proposal's override, read
    through manifest.mount_params) is >= 1, or under a non-scripted executor."""
    STEPS = 4
    n = 0
    ok = True
    task = "grab"
    executor = None

    @property
    def exhausted(self):
        return self.n >= self.STEPS

    def observe_once(self, obs):
        pass

    def on_handback(self):
        pass

    def act(self, obs):
        self.n += 1
        if self.executor is not None:
            return self.executor.act(obs)
        return (1.0,) if self.task == "reach" else (2.0,) if self.ok else (0.0,)

    def enter_segment(self, env, spec, executor=None):
        self.n = 0
        self.task = spec.task
        self.executor = executor
        grip = (mount_params(POLICY).get("tunables") or {}).get("grip", 0)
        self.ok = spec.task != "grab" or executor is not None or grip >= 1

    def segment_success(self, env):
        return self.task in env.achieved


class _Policy:
    def make_driver(self, spec):
        return _Driver()


class _Planner:
    identity = "proposals_e2e:fixed"

    def plan(self, brief):
        return {"goal": "reach then grab",
                "nodes": [{"id": "reach-0", "skill": "reach", "kind": "segment", "args": {}, "after": []},
                          {"id": "grab-0", "skill": "grab", "kind": "segment", "args": {},
                           "after": ["reach-0"]}],
                "verify": [{"after": "reach-0", "predicate": "seg_ok"},
                           {"after": "grab-0", "predicate": "seg_ok"}],
                "rationale": "fixed"}


def env_provider():
    return _Env()


def policy_provider(**params):  # mounted under its card's params (mount_params)
    return _Policy()


def cand_provider(**params):
    return _Policy()


def planner_provider():
    return _Planner()


_CARD = f"""
[task_bindings.e2e_prop]
env = "{EMB}"
policy = "{POLICY}"
planner = "test_proposals_e2e:planner_provider"
catalogue = "test_proposals_e2e:CATALOGUE"
records = "test_proposals_e2e:RECORDS"
oracles = "test_proposals_e2e:ORACLES"
episodic = true
episode = "test_proposals_e2e:EPISODE"
segment_specs = "test_proposals_e2e:SEGMENT_SPECS"
max_replans = 1
"""

#: A candidate card: disabled (no mount collision in the fold), one provides row
#: (so discover() proves it was mounted), doctor-able through its mounts table.
_CANDIDATE = f"""
enabled = false
[mounts]
"policy.driver" = "{CAND}"
[[provides]]
kind = "skill"
ref = "{CAND}"
name = "cand"
"""

TASK = "e2e_prop"


def _candidate(root: Path) -> Path:
    card = root / "plugins" / "candidates" / "cand_fixture"
    card.mkdir(parents=True, exist_ok=True)
    (card / "manifest.toml").write_text(_CANDIDATE)
    return card


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    params = runs / "plugins" / "params"
    params.mkdir(parents=True)
    (params / "manifest.toml").write_text(
        'enabled = false\n[mounts."policy.driver"]\n' + f'ref = "{POLICY}"\n'
        '[mounts."policy.driver".params.tunables]\ngrip = 0.0\n')
    rt = _Runtime(runs, card=_CARD, mode="evolution")
    rt.campaign = rt.session / "campaigns" / f"evolve-{TASK}" / "campaign.json"
    rt.candidate = _candidate(rt.runs)
    yield rt
    rt.stop()


def _raw(task, kind, payload, note="") -> str:
    return json.dumps({"task": task, "kind": kind, "payload": payload, "note": note})


def test_three_faces_agree_on_submit_and_list(runtime, capsys):
    sd, runs = runtime.session, runtime.runs
    ms.configure(runs)
    base = ["--runs", str(runs), "--session", SESSION]
    raw = _raw("other_task", "executor", {"to": "alt", "node": "grab-0"}, "faces")
    lib = bs.submit_proposal(sd, raw)
    code = storecli.main(["submit_proposal", "--brief", raw] + base)
    cli = json.loads(capsys.readouterr().out)
    mcp = ms.submit_proposal(json.loads(raw))
    assert code == 0 and all(r["inbox"] == str(sd / "proposals") for r in (lib, cli, mcp))
    ids = [lib["submitted"], cli["submitted"], mcp["submitted"]]
    assert all((sd / "proposals" / f"{i}.json").is_file() for i in ids)
    code = storecli.main(["proposals"] + base)
    out = capsys.readouterr().out.rstrip("\n")
    assert code == 0 and out == json.dumps(bs.proposals(sd)) == json.dumps(ms.proposals())
    rows = bs.proposals(sd)
    assert [r["id"] for r in rows] == ids   # submission order
    assert rows[0] == {"id": ids[0], "task": "other_task", "kind": "executor",
                       "payload": {"to": "alt", "node": "grab-0"}, "note": "faces", "applied": None}
    # the shape gate, same answer on every face
    for bad in ('{"task":"t","kind":"magic","payload":{}}', '{"task":"t","kind":"card"}',
                '{"task":"t","kind":"card","payload":{},"extra":1}', "not json"):
        assert "error" in bs.submit_proposal(sd, bad), bad
    assert ms.submit_proposal({"task": "t", "kind": "card"})["error"] == "proposal needs payload: object"
    assert ms.proposals("../x") == {"error": "unknown session"}
    assert storecli.main(["submit_proposal"] + base) == 3   # needs --brief
    capsys.readouterr()
    assert len(bs.proposals(sd)) == 3


@pytest.fixture(scope="module")
def two_rounds(runtime):
    sd = runtime.session
    t = bs.submit_proposal(sd, _raw(TASK, "tunables", {"node": "grab-0", "ref": POLICY, "path": ["tunables", "grip"],
                                                        "to": 1}, "grip harder"))["submitted"]
    c = bs.submit_proposal(sd, _raw(TASK, "card", {"path": str(runtime.candidate), "to": "cand",
                                                    "ref": CAND, "node": "grab-0"}, "try cand"))["submitted"]
    name, rows = runtime.run({"kind": "evolve", "task": TASK, "seeds": [1, 2],
                              "rounds": 2, "confirm_seeds": 0})
    return t, c, name, rows


def test_evolve_consumes_a_tunables_then_a_card_proposal(runtime, two_rounds):
    t, c, name, rows = two_rounds
    doc = json.loads(runtime.campaign.read_text())
    assert doc["status"] == "done" and doc["cursor"] == 2
    r1, r2 = doc["rounds"]
    # round 1: the tunables proposal, applied instead of the built-in proposer; the
    # override changed world state (0/2 -> 2/2), earning development acceptance.
    assert r1["tried"]["kind"] == "tunables" and r1["tried"]["node"] == "grab-0"
    d = r1["tried"]["detail"]
    assert (d["proposal"], d["note"], d["ref"], d["path"], d["to"]) == \
        (t, "grip harder", POLICY, ["tunables", "grip"], 1)
    assert (d["skill"], d["executor"]) == ("grab", "scripted")
    assert (r1["before"], r1["after"], r1["accepted"], r1["published"]) == (0, 2, True, False)
    assert r1["proposal"] == {"id": t, "kind": "tunables", "note": "grip harder"}
    assert doc["applied"]["tunables"] == {POLICY: {"tunables": {"grip": 1}}}
    assert "digest" not in d  # no installed record without the verification battery
    assert r1["evaluation"]["installation"]["status"] == "not_evaluated"
    # round 2: the card proposal -- candidate mounted, executor forced, suite ran
    # (2/2 stays 2/2: consumed, not published, no error)
    assert r2["tried"]["kind"] == "card" and r2["tried"]["detail"]["proposal"] == c
    assert r2["tried"]["detail"]["to"] == "cand" and "error" not in r2["tried"]["detail"]
    assert (r2["before"], r2["after"], r2["published"]) == (2, 2, False)
    assert r2["proposal"]["id"] == c and doc["applied"].get("cards", {}) == {}
    # the inbox shows both consumed, in their rounds
    applied = {p["id"]: p["applied"] for p in bs.proposals(runtime.session)}
    assert applied[t]["round"] == 1 and applied[c]["round"] == 2
    # sealed: rsi_proposal_applied right before each round's rsi_step
    kinds = [(r["kind"], r["data"].get("round")) for r in rows
             if r["kind"] in ("rsi_proposal_applied", "rsi_step")]
    assert kinds == [("rsi_proposal_applied", 1), ("rsi_step", 1),
                     ("rsi_proposal_applied", 2), ("rsi_step", 2)]
    sealed = _kinds(rows, "rsi_proposal_applied")
    assert [(s["id"], s["kind"], s["note"], s["brief"], s["task"]) for s in sealed] == \
        [(t, "tunables", "grip harder", name, TASK), (c, "card", "try cand", name, TASK)]
    assert not _kinds(rows, "runtime.task_error")


def test_card_root_mounts_and_accepted_overlay_retains_binding_without_installing(runtime, tmp_path, monkeypatch):
    # PH_PLUGINS_EXTRA may name ONE card dir (plugins/candidates/<name>)
    monkeypatch.setenv("PH_PLUGINS_EXTRA", str(runtime.candidate))
    prov = [p for p in discover().provides if p["plugin"] == "cand_fixture"]
    assert prov and prov[0]["kind"] == "skill" and prov[0]["name"] == "cand"
    skills = tmp_path / "skills"
    skills.mkdir()
    original = skills / "grab.json"
    original.write_text(json.dumps(RECORDS["grab"]))
    installed_before = original.read_bytes()
    tried = {"kind": "card", "node": "grab-0",
             "detail": {"skill": "grab", "executor": "scripted", "to": "cand", "ref": CAND,
                        "path": str(runtime.candidate), "params": {"k": 1}}}
    applied = {"executors": {}, "tunables": {POLICY: {"tunables": {"grip": 1}}}, "cards": {}}
    accepted = evolve.apply(tried, applied)
    assert accepted["executors"] == {"grab-0": "cand"}
    assert accepted["tunables"] == applied["tunables"]
    assert accepted["cards"]["cand"] == {
        "skill": "grab", "path": str(runtime.candidate), "ref": CAND,
        "params": {"k": 1}, "transport": "inproc",
        "artifact_sha": evolve.evolve_llm.candidate_digest(runtime.candidate),
    }
    assert applied["executors"] == {} and applied["cards"] == {}
    assert list(skills.iterdir()) == [original] and original.read_bytes() == installed_before
    # an incomplete proposal is an honest none, never a crash
    before = {"seeds": {"1": {"first_death": "grab-0",
                              "nodes": {"grab-0": {"skill": "grab", "success": False, "executor": "scripted"}}}}}
    none = evolve.from_proposal({"id": "p", "kind": "card", "note": "", "payload": {"node": "grab-0", "to": "x"}}, before)
    assert none["kind"] == "none" and "lacks" in none["detail"]["reason"] and none["detail"]["proposal"] == "p"


def test_doctor_is_green_on_a_candidate_card(tmp_path, capsys):
    card = _candidate(tmp_path)
    rep = plugin_doctor.check(card)
    assert rep.green, [(r.name, r.status, r.detail) for r in rep.results]
    assert {(r.tier, r.status) for r in rep.results} == {("A", "PASS"), ("B", "PASS")}
    assert plugin_doctor.main([str(card)]) == 0
    assert "cand_fixture" in capsys.readouterr().out
    assert not os.path.exists(Path("plugins") / "candidates" / "cand_fixture")   # never in the repo
