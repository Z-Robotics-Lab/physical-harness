"""Everything the evolve loop grows without a bound, bounded: the sharded campaign
store, the pruned llm audits, the candidate-card GC and the disk guard. The live
recycle_cans campaign reached 42 MB of campaign.json, 145 MB of audits in 482 files
and 318 candidate dirs on a root filesystem with 26 GB left -- and the board's faces
died on Node's 1 MB pipe, so the operator saw no chart after 490 rounds. No simulator:
synthesised rounds in the exact shape scripts/evolve.py writes."""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts import evolve

NODES = ("nav-can1", "grasp-can1", "carry-can1", "drop-can1")


def _seed_row(seed: int, dead: str, ok_upto: int) -> dict:
    """One per_seed row in the real shape -- the ``nodes`` trail is the bulk (19 MB of
    the live 42 MB) and the one thing an index row drops."""
    return {"seed": seed, "success": False, "first_death": dead, "failure_mode": "reach_stall",
            "elapsed_s": 12.5, "tunables_sha": "a" * 64,
            "nodes": [{"id": n, "ok": i < ok_upto, "steps": 100, "task": "recycle_cans",
                       "failure_mode": None if i < ok_upto else "reach_stall",
                       "after": {"pose": [0.1] * 7}, "kind": "segment",
                       "trace_end": [{"eef": [0.1, 0.2, 0.3], "step": s} for s in range(20)]}
                      for i, n in enumerate(NODES)]}


def _round(no: int, node: str = "drop-can1", accepted: bool = False) -> dict:
    rows = [_seed_row(4243, node, 3), _seed_row(4244, node, 3)]
    return {"round": no, "tried": {"kind": "tunables", "node": node,
                                   "detail": {"skill": "drop_can1", "ref": "m:provider",
                                              "path": ["tunables", "hover_dz"], "from": 1.0,
                                              "to": 1.3, "layer": "parameter",
                                              "reason": "x" * 2000, "edits": ["y" * 4000]}},
            "before": 0, "after": 0, "best": 0, "parent": 0, "layer": "parameter",
            "notes": "z" * 900, "outcome": "same", "accepted": accepted, "published": False,
            "accepted_reason": "score [0, 11, 0] -> [0, 11, 0]", "before_score": [0, 11, 0],
            "after_score": [0, 11, 0], "usage": {"llm_tokens": None, "sim_s": 30.0},
            "proposer": "llm", "needs": [], "confirm": None, "trial": None, "stuck": None,
            "regression": None, "burned": [], "suite_sha": "b" * 64, "proposal": None,
            "trial_evidence": {"node": node, "seeds": [{"seed": 4243, "diff": {}}]},
            "ts": 1.0 * no, "per_seed": rows, "after_seeds": rows,
            "media": [f"media/recycle_cans/4243/r{no}.mp4"],
            "media_dropped": {f"4243/{node}": {"reason": "no clip", "keyframes": []}},
            "llm": {"model": "deepseek", "prompt_sha": "c" * 64, "raw_sha": "d" * 64,
                    "summary": "s" * 400, "rationale": "r" * 3000, "reason": None}}


def _doc(n: int) -> dict:
    return {"task": "recycle_cans", "session": "s", "seeds": [4243, 4244], "arm": "scripted",
            "rounds": [_round(i, accepted=i == 7) for i in range(1, n + 1)],
            "best": 0, "cursor": n, "status": "running",
            "applied": {"executors": {}, "tunables": {}, "cards": {}}, "accepted_stack": []}


def _store(tmp_path, doc: dict) -> evolve.EvolveStore:
    st = evolve.EvolveStore(tmp_path, "recycle_cans")
    st.dir.mkdir(parents=True)
    st.path.write_text(json.dumps(doc, indent=1, sort_keys=True))
    return st


# ── 1. sharding: small file, same answers ────────────────────────────────────────

def test_a_500_round_campaign_stays_small_and_every_history_answer_is_unchanged(tmp_path):
    full = _doc(500)
    before = json.loads(json.dumps(full))          # the pre-sharding answers, kept whole
    st = _store(tmp_path, full)
    doc = st.load()

    assert st.path.stat().st_size < 2 * 1024 ** 2, st.path.stat().st_size
    assert len(list(st.rounds_dir.glob("*.json"))) == 500 - evolve.ROUNDS_KEPT
    assert len(doc["rounds"]) == 500 and doc["rounds"][-1] == before["rounds"][-1]

    hist, hist0 = doc["rounds"], before["rounds"]
    suite = {"count": 0, "seeds": {"4243": {"first_death": "drop-can1", "failure_mode": "reach_stall",
                                            "success": False, "trail": _seed_row(4243, "drop-can1", 3)["nodes"]},
                                   "4244": {"first_death": "nav-can1", "failure_mode": "reach_stall",
                                            "success": False, "trail": _seed_row(4244, "nav-can1", 1)["nodes"]}}}
    # every function that reads history, off the index instead of the whole rounds list
    assert evolve.death_nodes(suite, hist) == evolve.death_nodes(suite, hist0)
    assert evolve._first_death(suite, hist) == evolve._first_death(suite, hist0)
    assert evolve.stuck_on("drop-can1", hist) == evolve.stuck_on("drop-can1", hist0)
    assert evolve.cluster_seeds(hist, suite, "drop-can1") == evolve.cluster_seeds(hist0, suite, "drop-can1")
    assert evolve.focus_seeds(hist, suite, "drop-can1", [4243, 4244]) \
        == evolve.focus_seeds(hist0, suite, "drop-can1", [4243, 4244])
    assert evolve.regression(hist, suite, suite, "drop-can1") == evolve.regression(hist0, suite, suite, "drop-can1")
    # scripts.evolve.propose's two history reads: executors already tried on the node,
    # and the (knob, direction) tunables steps already spent
    spent = lambda h: ({r["tried"]["detail"].get("to") for r in h
                        if r["tried"]["kind"] in ("executor", "card")},
                       {(r["tried"]["detail"]["path"][-1], r["tried"]["detail"]["to"] > r["tried"]["detail"]["from"])
                        for r in h if r["tried"]["kind"] == "tunables"})
    assert spent(hist) == spent(hist0) == (set(), {("hover_dz", True)})
    # the round loop's own three reads
    assert sum(r["tried"]["kind"] != "none" for r in hist) == 500
    assert max((x["round"] for x in hist if x.get("accepted") or x.get("published")), default=0) == 7
    assert sum((r.get("usage") or {}).get("sim_s") or 0 for r in hist) == 15000.0
    # the origin cluster is scored off round 1's trails EVERY round: they stay inline
    assert evolve.cluster_seeds(hist, suite, "drop-can1") == [4243, 4244]
    assert doc["rounds"][0]["per_seed"][0]["nodes"] and not doc["rounds"][1]["per_seed"][0].get("nodes")


def test_the_index_carries_the_chart_and_the_heat_strip_without_the_trails(tmp_path):
    """Item 5: the RSI page must draw 500 rounds off campaign.json alone."""
    doc = _store(tmp_path, _doc(60)).load()
    r = doc["rounds"][5]
    assert r["sharded"] and (r["tried_kind"], r["node"]) == ("tunables", "drop-can1")
    assert (r["before_score"], r["after_score"], r["outcome"]) == ([0, 11, 0], [0, 11, 0], "same")
    assert (r["accepted"], r["published"]) == (False, False) and r["usage"]["sim_s"] == 30.0
    assert r["node_rate"] == {"before": 0.75, "after": 0.75}       # 3 of 4 nodes ok
    assert r["by_task"] == {"recycle_cans": {"before": 0.0, "after": 0.0}}
    assert [s["seed"] for s in r["per_seed"]] == [4243, 4244] and "nodes" not in r["per_seed"][0]
    # and the bulk really is gone
    assert "media" not in r and "rationale" not in r["llm"] and "edits" not in r["tried"]["detail"]
    st_full = evolve.EvolveStore(tmp_path, "recycle_cans").round(6)
    assert st_full["media"] and st_full["tried"]["detail"]["edits"]   # all of it, in the shard


# ── 2. migration of a legacy file ────────────────────────────────────────────────

def test_migration_keeps_a_bak_is_byte_faithful_for_the_window_and_is_idempotent(tmp_path):
    legacy = _doc(64)
    st = _store(tmp_path, legacy)
    raw = st.path.read_bytes()

    doc = st.load()
    bak = st.path.with_suffix(".json.bak")
    assert bak.read_bytes() == raw                                     # the original, untouched
    assert doc["rounds"][-evolve.ROUNDS_KEPT:] == legacy["rounds"][-evolve.ROUNDS_KEPT:]
    assert {k: doc[k] for k in ("task", "session", "seeds", "arm", "best", "cursor",
                                "status", "applied", "accepted_stack")} \
        == {k: legacy[k] for k in ("task", "session", "seeds", "arm", "best", "cursor",
                                   "status", "applied", "accepted_stack")}
    for i in range(1, 65 - evolve.ROUNDS_KEPT):
        assert st.round(i) == legacy["rounds"][i - 1]                  # the whole row, in its shard

    # idempotent: a second load re-shards nothing, rewrites no shard, keeps the first .bak
    on_disk, shards = st.path.read_bytes(), {p: p.read_bytes() for p in st.rounds_dir.iterdir()}
    again = evolve.EvolveStore(tmp_path, "recycle_cans")
    assert again.load() == doc
    assert again.path.read_bytes() == on_disk and bak.read_bytes() == raw
    assert {p: p.read_bytes() for p in again.rounds_dir.iterdir()} == shards


def test_a_short_campaign_is_never_sharded(tmp_path):
    st = _store(tmp_path, _doc(evolve.ROUNDS_KEPT))
    assert st.load()["rounds"] == _doc(evolve.ROUNDS_KEPT)["rounds"]
    assert not st.rounds_dir.exists() and not st.path.with_suffix(".json.bak").exists()


# ── 3. the audits shrink, never vanish ───────────────────────────────────────────

def _audit(d, r: int) -> None:
    (d / f"round-{r}.json").write_text(json.dumps(
        {"round": r, "model": "deepseek", "prompt_sha": "c" * 64, "raw_sha": "d" * 64,
         "summary": "摘要", "rationale": "理由", "reason": None, "calls": 2,
         "usage": {"prompt": 100, "completion": 20}, "messages": [{"role": "user", "content": "m" * 90000}],
         "materials": {"src": "s" * 60000}, "raw": "raw" * 1000, "repeats": [],
         "attempts": [{"raw": "x" * 5000, "sha": f"sha{r}", "reason": "doctor: no", "usage": {}}],
         "tried": {"kind": "card", "node": "drop-can1",
                   "detail": {"to": "cand", "layer": "policy", "edits": ["e" * 4000]}}}))


def test_old_audits_shrink_to_their_summary_and_none_is_deleted(tmp_path):
    for r in range(1, 61):
        _audit(tmp_path, r)
    assert len(evolve.prune_audits(tmp_path, keep=50, dry_run=True)) == 10 and \
        json.loads((tmp_path / "round-1.json").read_text()).get("pruned") is None
    log = evolve.prune_audits(tmp_path, keep=50)
    assert len(log) == 10 and len(list(tmp_path.glob("round-*.json"))) == 60   # never deleted
    old = json.loads((tmp_path / "round-1.json").read_text())
    assert old["pruned"] and old["round"] == 1 and old["decision"] == "card"
    assert (old["layer"], old["attempt_count"], old["summary"]) == ("policy", 1, "摘要")
    assert old["rationale"] == "理由" and old["usage"] == {"prompt": 100, "completion": 20}
    assert (old["prompt_sha"], old["raw_sha"]) == ("c" * 64, "d" * 64)
    assert old["attempts"] == [{"sha": "sha1", "reason": "doctor: no"}]   # _prior_rejects still reads it
    assert not {"messages", "materials", "raw", "repeats"} & set(old)
    assert "edits" not in old["tried"]["detail"]
    assert (tmp_path / "round-1.json").stat().st_size < 1500
    assert json.loads((tmp_path / "round-60.json").read_text())["messages"]   # the newest are whole
    assert evolve.prune_audits(tmp_path, keep=50) == []                      # idempotent


# ── 4. the candidate GC ──────────────────────────────────────────────────────────

@pytest.fixture
def cards(tmp_path):
    """A candidates tree with one TRACKED (hand-written) card, one card an accepted
    stack still stands on, and 30 model-written ones."""
    root = tmp_path / "plugins" / "candidates"
    root.mkdir(parents=True)
    for i, name in enumerate(["hand_written", "on_the_stack", "in_applied",
                              *(f"patch_r{i}" for i in range(30))]):
        (root / name).mkdir()
        (root / name / "__init__.py").write_text(f"# {name}\n")
        for f in (root / name).iterdir():
            import os
            os.utime(f, (1000 + i, 1000 + i))
        import os
        os.utime(root / name, (1000 + i, 1000 + i))
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-f", "plugins/candidates/hand_written"], cwd=tmp_path, check=True)
    camp = tmp_path / "runs" / "session-x" / "campaigns" / "evolve-recycle_cans"
    camp.mkdir(parents=True)
    (camp / "campaign.json").write_text(json.dumps({
        "applied": {"cards": {"in_applied": {"ref": "plugins.candidates.in_applied:provider"}}},
        "accepted_stack": [{"round": 3, "detail": {"path": str(root / "on_the_stack")}}],
        "rounds": []}))
    return root, tmp_path / "runs"


def test_the_gc_dry_run_never_touches_a_tracked_or_referenced_card(cards):
    root, runs = cards
    log = evolve.gc_candidates(root, runs, dry_run=True)
    gone = {line.split()[2] for line in log}
    assert not gone & {"hand_written", "on_the_stack", "in_applied"}
    assert gone == {f"patch_r{i}" for i in range(10)}       # 30 model-written, newest 20 kept
    assert len(list(root.iterdir())) == 33                  # dry run deleted nothing

    assert evolve.gc_candidates(root, runs) == log
    assert sorted(d.name for d in root.iterdir()) == \
        sorted(["hand_written", "on_the_stack", "in_applied", *(f"patch_r{i}" for i in range(10, 30))])
    assert evolve.gc_candidates(root, runs) == []           # nothing left to take


def test_the_gc_deletes_nothing_when_git_cannot_say_what_is_hand_written(tmp_path):
    root = tmp_path / "candidates"
    (root / "patch_r1").mkdir(parents=True)
    log = evolve.gc_candidates(root, tmp_path / "runs", keep=0)
    assert len(log) == 1 and "nothing deleted" in log[0] and (root / "patch_r1").is_dir()


# ── 5. the disk guard ────────────────────────────────────────────────────────────

def test_the_disk_guard_fires_on_free_space_and_on_the_campaign_size(tmp_path, monkeypatch):
    import shutil
    camp = tmp_path / "campaigns" / "evolve-recycle_cans"
    camp.mkdir(parents=True)
    (camp / "campaign.json").write_text("{}")
    free = [50 * 1024 ** 3]
    monkeypatch.setattr(shutil, "disk_usage", lambda p: type("U", (), {"free": free[0]})())
    assert evolve.disk_guard(tmp_path, camp) is None

    free[0] = 4 * 1024 ** 3
    msg = evolve.disk_guard(tmp_path, camp)
    assert "4.3 GB" in msg and "5 GB" in msg and "paused_disk" in msg

    free[0] = 50 * 1024 ** 3
    monkeypatch.setattr(evolve, "MAX_CAMPAIGN_BYTES", 1)
    msg = evolve.disk_guard(tmp_path, camp)
    assert "evolve-recycle_cans" in msg and "paused_disk" in msg


def test_the_round_pass_never_gcs_the_repos_cards_from_a_scratch_session(tmp_path, monkeypatch):
    """The card store is repo-global, the reference set comes out of campaigns: a loop on a
    scratch ``runs/`` (every e2e test, and this box's own test lane) must GC nothing --
    it cannot see the campaigns the repo's cards belong to."""
    called = []
    monkeypatch.setattr(evolve, "gc_candidates", lambda *a, **k: called.append(a) or [])
    st = evolve.EvolveStore(tmp_path / "runs" / "session-x", "recycle_cans")
    (st.dir / "llm").mkdir(parents=True)
    assert evolve.maintain(st, tmp_path / "runs" / "session-x") == [] and called == []


def test_the_loop_stops_at_paused_disk_before_it_spends_a_single_seed(tmp_path, monkeypatch):
    """The guard sits at the top of the round, ahead of the baseline suite: nothing is
    run, the status and the round row say why, and the process exits nonzero so the
    brief fails loudly instead of filling the disk."""
    monkeypatch.setattr(evolve, "disk_guard", lambda *a: "磁盘只剩 1.0 GB…paused_disk")
    monkeypatch.setattr(evolve, "run_suite", lambda *a, **k: pytest.fail("a seed was spent"))
    rc = evolve.main(["--mode", "evolution", "--task", "recycle_cans", "--session", str(tmp_path),
                      "--skills-root", str(tmp_path / "skills"), "--rounds", "1"])
    doc = evolve.EvolveStore(tmp_path, "recycle_cans").load()
    assert rc == 4 and doc["status"] == "paused_disk"
    r = doc["rounds"][-1]
    assert (r["round"], r["tried"]["kind"], r["accepted_reason"]) == (1, "none", "paused_disk")
    assert "paused_disk" in r["paused_disk"] and r["needs"] == ["disk"]
    assert doc["live"]["phase"] == "paused_disk" and doc["live"]["message"] == r["paused_disk"]


# ── 6. the faces over a SHARDED campaign ─────────────────────────────────────────
#
# The whole point of sharding is a console that draws; every one of these assertions
# was red when the store started sharding and the faces still read campaign.json only.

def test_the_faces_still_serve_a_round_the_store_sharded(tmp_path):
    import board.store as bs
    raw = tmp_path / "raw"
    (raw / "campaigns" / "evolve-recycle_cans").mkdir(parents=True)
    (raw / "campaigns" / "evolve-recycle_cans" / "campaign.json").write_text(json.dumps(_doc(60)))
    _store(tmp_path, _doc(60)).load()                     # shards in place

    # the chart: same numbers for every round, sharded or not -- not a run of nulls
    sharded, plain = bs.rsi_series(tmp_path, "recycle_cans"), bs.rsi_series(raw, "recycle_cans")
    assert len(sharded) == len(plain) == 60
    keys = ("round", "node_rate", "by_task", "before", "after", "accepted", "usage")
    assert [{k: r[k] for k in keys} for r in sharded] == [{k: r[k] for k in keys} for r in plain]
    assert all(r["node_rate"]["before"] == 0.75 and r["by_task"] for r in sharded)

    # the round card: the FULL row, read out of the shard
    row = bs.rsi_run(tmp_path, "recycle_cans", 6)["rounds"][0]
    assert row["per_seed"][0]["nodes"] and row["trial_evidence"] and row["media"]
    assert row["llm"]["rationale"] and row["tried"]["detail"]["edits"]
    assert bs.rsi_run(tmp_path, "recycle_cans", 999)["rounds"] == []
    # ... and the tail stays compact
    assert "per_seed" not in bs.rsi_run(tmp_path, "recycle_cans")["rounds"][-1]

    # the failure keyframes
    assert bs.rsi_frames(tmp_path, "recycle_cans", 6) == bs.rsi_frames(raw, "recycle_cans", 6)
    assert bs.rsi_frames(tmp_path, "recycle_cans", 6)["media"] == ["media/recycle_cans/4243/r6.mp4"]


def test_a_repair_node_never_counts_toward_the_chart():
    """A ``recover-<node>`` exists only because <node> failed: counting it made the
    chart paint a fix as a fall (by_task['recover'] 1.0 -> 0.5) on precisely the
    rounds the robot got further -- the same dishonesty evolve.score had."""
    import board.store as bs

    def trail(*rows):
        return [{"seed": 1, "nodes": [{"id": i, "ok": o, "task": t, "kind": k}
                                      for i, o, t, k in rows]}]
    plain = trail(("nav", True, "nav", "segment"), ("drop", False, "drop", "segment"))
    repaired = trail(("nav", True, "nav", "segment"), ("drop", False, "drop", "segment"),
                     ("recover-drop", True, "recover", "recovery"))
    assert bs._rates(plain) == bs._rates(repaired) == (0.5, {"nav": 1.0, "drop": 0.0})
    # the naming convention alone is enough, kind or no kind
    assert bs._rates(trail(("nav", True, "nav", "segment"),
                           ("recover-nav", True, "recover", None))) == (1.0, {"nav": 1.0})
