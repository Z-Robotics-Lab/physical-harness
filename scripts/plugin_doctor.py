#!/usr/bin/env python3
"""plugin_doctor: a card's 体检 (health check), mode-agnostic (GOAL v4.1 W2).

    PYTHONPATH=. .venv/bin/python scripts/plugin_doctor.py plugins/embodiment_robosuite

Two tiers, both cloned from machinery that already exists so the doctor invents
no new checking:

- Tier A (mount-time shape): parse the manifest, refuse ``actuation:real``, and
  for every declared mount ``load_provider`` it and mount it on a
  ``Kernel(CAPABILITIES)`` -- ``Kernel.provide`` IS the ``isinstance`` gate
  against the capability's declared contract, so a wrong-shaped provider fails
  here, exactly where the runtime would fail it.

- Tier B (one canonical fake-episode smoke per capability KIND): a kind->smoke
  dispatch table runs each mounted provider through its contract method(s) on a
  canonical episode. ``harness.fakes`` stands in for everything the card does not
  itself provide (an obs when the card mounts no env, say). Per-kind determinism
  policy: a percept/planner STAND-IN must be a pure function of its inputs
  (``required`` -- a sealed replay can't reproduce otherwise), while an LLM
  reasoner is validated for SHAPE only (``untrusted`` -- qwen is non-deterministic
  by nature; run the smoke once, never diff it).

``needs_sim`` gates the real-sim tier: a card that needs the simulator (its
third-party deps unimportable here) has its Tier A/B skipped, not failed -- the
base machine simply can't run its sim smokes. Everything else runs on the base
tier (fakes + determinism), so 体检 splits base/plugin the way W6 双层测试 asks.

Base-clean: imports only ``harness.*`` + stdlib at module load; a card's
providers (and any simulator they drag in) are imported lazily via
``load_provider`` at check time, so importing the doctor never imports a card.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from tomllib import loads as _toml
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.definitions import CAPABILITIES
from harness.kernel import Kernel
from harness.manifest import card_mounts
from harness.registry import load_provider
from harness.spec import EpisodeSpec


class DoctorSkip(Exception):
    """A smoke that cannot run here but is not a failure -- reported SKIP, loudly.

    The qwen reasoner card raises this when its endpoint is unreachable (GPU
    held): Tier A still validated the shape, and the untrusted-determinism policy
    never diffs an LLM anyway, so an absent model is a skip, not a red -- exactly
    how scripts/round25_rerun degrades its qwen arm.
    """


#: Determinism policy per capability KIND. Absent = shape-only (run once).
_DETERMINISM = {
    "percept.model": "required",       # a percept stand-in must be pure in (seed, draw)
    "task.planner": "required",        # a plan is a symbol graph, same brief -> same graph
    "reasoner.proposer": "untrusted",  # an LLM reasoner: validate the shape, never diff it
    "model.endpoint": "untrusted",     # a chat transport: probe + shape, never diff
}


# ── the canonical episode the smokes run against ────────────────────────────

@dataclass(frozen=True)
class _Ctx:
    spec: object
    obs: Mapping
    brief: dict


def _build_ctx(providers: dict, needs_sim: bool) -> _Ctx:
    """One episode's stand-in inputs: a real EpisodeSpec + the card's own env obs
    when it needs the sim, else a fake spec + the null env's canned obs. The obs
    is sourced from the mounted env if the card declares one (so a real percept
    reads the real key), otherwise from ``harness.fakes`` -- the doctor's
    stand-in for what the card does not provide."""
    if needs_sim:
        spec: object = EpisodeSpec(seed=90001, task="lift")
    else:
        spec = SimpleNamespace(seed=90001, task="fake", horizon=5)
    env = providers.get("embodiment.env")
    if env is not None:
        # The canonical spec asks for "lift"; a card that does not offer it (a
        # robocasa card whose only task is "kitchen_thaw") gets its OWN first
        # task instead, so the smoke builds a world the card can actually make.
        # robosuite keeps "lift" (it IS in its table) -- byte-identical there.
        known = env.tasks()
        if needs_sim and known and getattr(spec, "task", None) not in known:
            spec = EpisodeSpec(seed=spec.seed, task=known[0])
        handle = env.make_env(spec)
        obs = handle.reset()
        getattr(handle, "close", lambda: None)()
    else:
        obs = {"object": (0.0, 0.0, 0.0)}
    brief = {"task": getattr(spec, "task", "fake"), "scene": {}, "catalogue": []}
    return _Ctx(spec, obs, brief)


# ── per-kind smokes (cloned from the plugins' own self-checks) ───────────────
# Each returns a JSON-comparable value so a determinism-required kind can be
# diffed across two runs; shape assertions live inside.

def _smoke_env(prov, ctx: _Ctx):
    assert isinstance(prov.tasks(), tuple), "tasks() must return a tuple"
    assert isinstance(prov.object_key(ctx.spec), str), "object_key must return a str"
    assert isinstance(ctx.obs, Mapping) and ctx.obs, "make_env().reset() gave no obs"
    return "ok"


def _smoke_policy(prov, ctx: _Ctx):
    # A remote-transport policy (policy_vla_remote) exposes `available()` --
    # a TCP probe of its server. Nothing listening is a skip, not a red: Tier A
    # validated the shape, and the live inference needs the model-side process.
    probe = getattr(prov, "available", None)
    if probe is not None and not probe():
        raise DoctorSkip("policy server unreachable -- probed and skipped "
                         "(remote transport: shape validated Tier A)")
    action = tuple(prov.make_driver(ctx.spec).act(ctx.obs))
    assert len(action) > 0, "driver.act returned an empty action"
    return "ok"


def _smoke_percept(prov, ctx: _Ctx):
    est = prov.object_estimate(ctx.obs, ctx.spec, 0.02, 0)
    return [float(v) for v in list(est)]


def _smoke_planner(prov, ctx: _Ctx):
    # _smoke_reasoner's precedent: a model-backed planner exposes `available()`;
    # when its endpoint is down, skip loudly rather than red (shape passed Tier A).
    probe = getattr(prov, "available", None)
    if probe is not None and not probe():
        raise DoctorSkip("planner endpoint unreachable -- probed and skipped "
                         "(shape validated Tier A, live plan not run)")
    plan = prov.plan(ctx.brief)
    assert plan.get("goal") is not None and plan.get("nodes"), "plan lacks goal/nodes"
    return json.dumps(plan, sort_keys=True, default=str)


def _smoke_reasoner(prov, ctx: _Ctx):
    # A model-backed reasoner exposes `available()` (probes its endpoint); when it
    # is down, skip loudly rather than red -- the shape passed Tier A and the
    # untrusted policy never exercises the live proposal as a gate anyway.
    probe = getattr(prov, "available", None)
    if probe is not None and not probe():
        raise DoctorSkip("reasoner endpoint unreachable -- probed and skipped "
                         "(untrusted: shape validated Tier A, live proposal not run)")
    out = prov.propose(ctx.brief)
    assert isinstance(out, Mapping), "propose must return a Mapping"
    return "ok"  # untrusted: shape only, never diffed


def _smoke_model_endpoint(prov, ctx: _Ctx):
    # Same stance as _smoke_reasoner: probe first, SKIP when no endpoint is up
    # (GPU held, no API key) -- the shape passed Tier A and an LLM reply is
    # never diffed anyway.
    if not prov.available():
        raise DoctorSkip("model endpoint unreachable -- probed and skipped "
                         "(untrusted: shape validated Tier A, live chat not run)")
    out = prov.chat([{"role": "user", "content": "Reply with the single word: ok"}],
                    max_tokens=8, temperature=0.0)
    assert isinstance(out, str) and out, "chat must return a non-empty str"
    return "ok"  # untrusted: shape only, never diffed


def _smoke_skill(prov, ctx: _Ctx):
    rec = {"kind": "doctor_probe", "value": 1}
    digest = prov.publish(rec)
    assert digest == prov.publish(rec), "publish is not content-addressed (idempotent)"
    assert isinstance(prov.skills(), tuple), "skills() must return a tuple"
    return "ok"


def _smoke_scene(prov, ctx: _Ctx):
    snap = prov.snapshot(ctx.obs)
    assert isinstance(snap, Mapping) and ("nodes" in snap or "objects" in snap), \
        "snapshot must return a scene mapping"
    return "ok"


_SMOKES = {
    "embodiment.env": _smoke_env,
    "policy.driver": _smoke_policy,
    "percept.model": _smoke_percept,
    "task.planner": _smoke_planner,
    "reasoner.proposer": _smoke_reasoner,
    "model.endpoint": _smoke_model_endpoint,
    "graph.skill": _smoke_skill,
    "graph.scene": _smoke_scene,
}


# ── report ──────────────────────────────────────────────────────────────────

@dataclass
class Result:
    tier: str      # "A" | "B"
    name: str
    status: str    # "PASS" | "FAIL" | "SKIP"
    detail: str = ""


@dataclass
class Report:
    plugin: str
    results: list = field(default_factory=list)

    @property
    def green(self) -> bool:
        return all(r.status != "FAIL" for r in self.results)

    def add(self, tier: str, name: str, status: str, detail: str = "") -> None:
        self.results.append(Result(tier, name, status, detail))


def check(plugin_dir: str | Path) -> Report:
    """Run Tier A + Tier B on one card directory and return its report card."""
    plugin_dir = Path(plugin_dir)
    rep = Report(plugin_dir.name)
    data = _toml((plugin_dir / "manifest.toml").read_text())

    if data.get("actuation", "sim") == "real":
        rep.add("A", "actuation", "FAIL",
                "actuation:real refused -- a real actuator needs a separate "
                "authenticated runtime, never the sim doctor")
        return rep

    needs_sim = bool(data.get("needs_sim", False))
    if needs_sim:
        missing = [r for r in data.get("third_party", ())
                   if importlib.util.find_spec(r.split(".")[0]) is None]
        if missing:
            rep.add("B", "needs_sim", "SKIP",
                    f"gated: simulator absent here ({missing}) -- run on a sim machine")
            return rep

    # Tier A for task bindings: every ref a binding declares must load and pass
    # the SAME contract gates the runtime applies at dispatch -- policy/planner
    # through Kernel.provide (isinstance against PolicyFactory/TaskPlanner),
    # catalogue/oracles as importable attributes. GOAL v4.2's fixed principle is
    # "不合格在 mount 报错而非任务中失败"; a binding with a dead ref must redden
    # HERE, not when the resident runtime resolves it mid-brief.
    for task, binding in data.get("task_bindings", {}).items():
        bk = Kernel(CAPABILITIES)
        try:
            bk.provide("policy.driver", load_provider(binding["policy"], {}),
                       ref=binding["policy"], params={})
            bk.provide("task.planner", load_provider(binding["planner"], {}),
                       ref=binding["planner"], params={})
            for attr_key in ("catalogue", "oracles"):
                module_name, attr = binding[attr_key].split(":", 1)
                getattr(importlib.import_module(module_name), attr)
            # Heterogeneous-mission binding kind: a PREDICATES table (predicate
            # name -> "module:factory" ref) for perceive/decide/verify nodes.
            # Every ref must load_provider-resolve HERE -- a dead predicate ref is
            # a malformed declaration that must redden at mount, not mid-brief when
            # a kindful node reaches for its oracle. Symmetric with the policy/
            # planner load above (GOAL v4.2: 不合格在 mount 报错而非任务中失败).
            if "predicates" in binding:
                mod, attr = binding["predicates"].split(":", 1)
                table = getattr(importlib.import_module(mod), attr)
                for ref in table.values():
                    load_provider(ref)
            # Persistent-episode binding kind (M7): the ONE-episode block and the
            # per-sub-goal segment_specs each may name a `stages` "module:factory"
            # ref. Resolve every one HERE so a dead segment-stages ref reddens at
            # mount, not when the runner opens the world -- same stance as the
            # predicate refs above (GOAL v4.2: 不合格在 mount 报错而非任务中失败).
            if binding.get("episodic"):
                epmod, epattr = binding["episode"].split(":", 1)
                episode = getattr(importlib.import_module(epmod), epattr)
                smod, sattr = binding["segment_specs"].split(":", 1)
                seg_specs = getattr(importlib.import_module(smod), sattr)
                stages_refs = [episode.get("stages")]
                stages_refs += [s.get("stages") for s in seg_specs.values()]
                for ref in stages_refs:
                    if ref is not None:
                        load_provider(ref)
        except Exception as exc:  # noqa: BLE001 -- any dead ref is a red
            rep.add("A", f"task:{task}", "FAIL", f"{type(exc).__name__}: {exc}")
            continue
        rep.add("A", f"task:{task}", "PASS", binding["planner"])

    # Tier A for a candidate card's ``[executors.<key>]``: the ref the suite would
    # mount must load HERE (an LLM-written card reddens at the doctor, never mid-suite).
    for key, spec in data.get("executors", {}).items():
        try:
            load_provider(spec["ref"])
            rep.add("A", f"executor:{key}", "PASS", spec["ref"])
        except Exception as exc:  # noqa: BLE001 -- a dead ref / bad code is a red
            rep.add("A", f"executor:{key}", "FAIL", f"{type(exc).__name__}: {exc}")

    mounts = card_mounts(data)
    if not mounts:
        if not data.get("task_bindings") and not data.get("executors"):
            rep.add("A", "mounts", "SKIP", "claim-only card: nothing executable to check")
        return rep

    # Tier A: load + isinstance, reusing Kernel.provide's contract gate.
    kernel = Kernel(CAPABILITIES)
    providers: dict[str, object] = {}
    for m in mounts:
        try:
            prov = load_provider(m.provider, m.params)
            kernel.provide(m.capability, prov, ref=m.provider, params=dict(m.params))
        except Exception as exc:  # noqa: BLE001 -- any load/shape failure is a red
            rep.add("A", m.capability, "FAIL", f"{type(exc).__name__}: {exc}")
            continue
        providers[m.capability] = prov
        rep.add("A", m.capability, "PASS", m.provider)

    # Tier B: one canonical fake-episode smoke per mounted, conforming kind.
    ctx = _build_ctx(providers, needs_sim)
    for cap, prov in providers.items():
        smoke = _SMOKES.get(cap)
        if smoke is None:  # e.g. exec.rollouts / ground_truth: no episode smoke
            continue
        policy = _DETERMINISM.get(cap)
        # Non-determinism exemption (the _smoke_reasoner precedent, generalised):
        # a determinism-required provider that EXPLICITLY declares
        # ``deterministic = False`` (an LLM planner) is validated for shape only,
        # never double-run-diffed -- the untrusted policy, opted into loudly.
        if policy == "required" and getattr(prov, "deterministic", True) is False:
            policy = "exempt"
        try:
            out = smoke(prov, ctx)
            if policy == "required":
                again = smoke(prov, ctx)
                if out != again:
                    rep.add("B", cap, "FAIL",
                            f"non-deterministic (policy: required): {out!r} != {again!r}")
                    continue
            rep.add("B", cap, "PASS",
                    "deterministic" if policy == "required"
                    else "shape validated, not diffed (provider declares "
                         "deterministic=False)" if policy == "exempt"
                    else (policy or "shape"))
        except DoctorSkip as skip:
            rep.add("B", cap, "SKIP", str(skip))
        except Exception as exc:  # noqa: BLE001 -- a smoke that raises is a red
            rep.add("B", cap, "FAIL", f"{type(exc).__name__}: {exc}")
    return rep


def verify(store: str | Path) -> Report:
    """验货: read a published skill store and pass/fail the acceptance claim.

    The counterpart to 体检 (``check``): 体检 is mode-agnostic and asks "does this
    card mount and behave"; 验货 asks "did an evolution-mode campaign actually
    earn a promotable skill here" and is the execution-mode admission ticket
    (GOAL v4.1). No new statistics -- it only READS the ``SkillRecord``s the
    existing evidence machine published to ``store`` (a flat ``<digest>.json``
    graph.skill root -- ``rt.skills_root``, or an acceptance campaign's
    ``<out>/skills``) and folds their fields into four checks: a prereg hash is
    present, at least one skill was promoted (a record IS a promotion --
    ``workload.run`` publishes only promoted generations), the held-out judgement
    was established (``heldout_judgement_established`` True, i.e. p<alpha &
    fixed>broken on the held-out block), and the transfer ablation was run.
    """
    store = Path(store)
    rep = Report(f"{store.name} (验货)")
    records = [json.loads(f.read_text()) for f in sorted(store.glob("*.json"))]
    rep.add("V", "promoted skill", "PASS" if records else "FAIL",
            f"{len(records)} published skill record(s)")
    rep.add("V", "prereg_sha", "PASS" if records and all(r.get("prereg_sha") for r in records)
            else "FAIL", "content hash present on every record")
    established = [r for r in records if r.get("heldout_judgement_established") is True]
    rep.add("V", "heldout judgement", "PASS" if established else "FAIL",
            "established (p<alpha & fixed>broken on held-out)")
    ablation = [r for r in records if (r.get("bundle_evidence") or {}).get("ablation")]
    rep.add("V", "ablation", "PASS" if ablation else "FAIL", "transfer ablation curve present")
    return rep


def verify_claim(plugin_dir: str | Path) -> Report:
    """验货 a skill card's manifest CLAIM against the sealed store it names.

    The card-oriented counterpart to ``verify(store)``: a skill card declares its
    sealed evidence in ``[claim.sealed]`` -- the ``store`` its records live in, the
    exact SkillRecord ``skills`` digests it commits to, ``heldout_judgement_established``,
    and the headline ``rescore_blocks``. This resolves the store, reuses ``verify``
    for the four record checks (promotion / prereg_sha / held-out judgement /
    ablation), then adds the claim-specific pins: the declared digests must be
    EXACTLY the store's skill filenames (a content commitment -- a swapped or added
    sealed record breaks the claim), and each cited rescore block must be present
    as a sealed store. So `--verify-claim plugins/task` is the execution-mode
    admission ticket read straight off the card, no store path to remember.
    """
    plugin_dir = Path(plugin_dir)
    data = _toml((plugin_dir / "manifest.toml").read_text())
    sealed = (data.get("claim") or {}).get("sealed")
    if not sealed:
        rep = Report(f"{plugin_dir.name} (验货 claim)")
        rep.add("V", "claim.sealed", "FAIL",
                "manifest has no [claim.sealed] table to verify")
        return rep

    store = (REPO_ROOT / sealed["store"]).resolve()
    rep = verify(store / "skills")
    rep.plugin = f"{plugin_dir.name} (验货 claim: {sealed['store']})"

    # Content commitment: the claim pins EXACTLY these sealed digests. A record
    # swapped out (or an extra one slipped in) breaks set-equality, so the claim
    # can never silently drift off the evidence it was written against.
    on_disk = {f.stem for f in (store / "skills").glob("*.json")}
    declared = set(sealed.get("skills", ()))
    rep.add("V", "sealed digests",
            "PASS" if declared and declared == on_disk else "FAIL",
            f"claim pins {sorted(d[:12] for d in declared)} vs store "
            f"{sorted(d[:12] for d in on_disk)}")

    # The card asserts the held-out judgement is established; verify() above
    # confirmed it on the records themselves.
    rep.add("V", "claim established",
            "PASS" if sealed.get("heldout_judgement_established") is True else "FAIL",
            "manifest claims heldout_judgement_established")

    # The headline rescore blocks the claim cites must exist as sealed stores.
    for rb in sealed.get("rescore_blocks", ()):
        rep.add("V", f"rescore {rb}",
                "PASS" if (REPO_ROOT / rb / "index.jsonl").exists() else "FAIL",
                "sealed heldout_rescore block present")
    return rep


def _fmt(rep: Report) -> str:
    lines = [f"plugin_doctor: {rep.plugin} -- {'GREEN' if rep.green else 'RED'}"]
    for r in rep.results:
        lines.append(f"  [{r.status:4}] tier {r.tier} {r.name}: {r.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plugin_dir", nargs="?",
                    help="the card directory to 体检 (holds manifest.toml)")
    ap.add_argument("--verify", metavar="STORE",
                    help="验货: pass/fail a published skill store instead of 体检 a card")
    ap.add_argument("--verify-claim", metavar="DIR", dest="verify_claim",
                    help="验货 a skill card's [claim.sealed] against the store it names")
    args = ap.parse_args(argv)
    if args.verify_claim is not None:
        rep = verify_claim(args.verify_claim)
    elif args.verify is not None:
        rep = verify(args.verify)
    elif args.plugin_dir is not None:
        rep = check(args.plugin_dir)
    else:
        ap.error("give a card directory to 体检, --verify STORE, or --verify-claim DIR")
    print(_fmt(rep))
    return 0 if rep.green else 1


if __name__ == "__main__":
    sys.exit(main())
