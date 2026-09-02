"""Plugin self-registration: ``plugins/*/manifest.toml`` parsed as DATA.

A manifest is TOML, read and never imported -- so a card declaring its mounts,
task bindings, campaigns and third-party surface stays declarative in exactly
the way a hand-written profile was (``tests/test_boundaries.py``). Discovery is a
directory scan; the registry is the UNION of every installed manifest, and any
collision across manifests -- a duplicate capability mount, a duplicate task, a
duplicate campaign -- is LOUD, because two cards silently fighting over one seam
is the failure the fold exists to catch.

The union is what makes "adding a task = installing a plugin dir" true: a brief
still names only a task STRING, and the runtime resolves task->policy through the
manifests present at boot. A manifest declaring ``actuation = "real"`` is refused
here -- the sim runtime never mounts a real actuator; a real card rides a
separate authenticated runtime (GOAL v4.2).

Imports only stdlib + ``harness.config`` (never a plugin), so ``base_profile``
folds it while ``harness`` stays plugin-free (``tests/test_kernel.py``).
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from harness.config import Mount

#: The one plugin cage the scan walks. A card is a directory with a manifest.
PLUGINS_ROOT = Path(__file__).resolve().parent.parent / "plugins"


@dataclass(frozen=True)
class Registry:
    """The folded union of every installed manifest.

    ``mounts`` feed ``base_profile``; ``task_bindings`` and ``campaigns`` feed the
    runtime's server-side selectors (the tables that used to be welded into
    ``harness_runtime``); ``third_party`` feeds the plugin-boundary AST test.
    """

    mounts: tuple[Mount, ...] = ()
    task_bindings: dict[str, dict] = field(default_factory=dict)
    campaigns: dict[str, str] = field(default_factory=dict)
    third_party: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Named overlay bundles a card owns: an alternate mount set (a second robot,
    #: a real-world scene bridge) layered over ``base_profile`` by ``profiles.bundle``.
    #: The wiring lives in the card's manifest, not welded into ``profiles``.
    bundles: dict[str, tuple[Mount, ...]] = field(default_factory=dict)
    #: Recovery repair shapes a card declares (``[recoveries.<name>] ref = "module:attr"``):
    #: strategy name -> (declaring plugin dir, ref). ``plugins/rsi/repertoire.py``
    #: resolves the refs; the fold here stays data-only like everything else.
    recoveries: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: ``[[provides]]`` entries a card declares (``{kind, ref, name?, ...}``),
    #: each stamped with its ``plugin``. Kinds: PROVIDES_KINDS. Predicates carry
    #: ``reads`` (sigma keys) and optional ``args``; ``harness.predicates`` folds
    #: them into PredicateRecords keyed by name, bindings per card.
    provides: tuple[dict, ...] = ()
    #: ``[benchmarks.<name>]`` pure-data suite cards (tasks, arms, max_replans):
    #: the runtime's ``suite`` brief selector, folded exactly like campaigns.
    benchmarks: dict[str, dict] = field(default_factory=dict)
    #: ``[executors.<key>] skill = ..., embodiment = ..., ref = ..., transport?``:
    #: a card (typically a candidate mounted through PH_PLUGINS_EXTRA) binding an
    #: executor key onto a skill record for as long as it is mounted --
    #: ``harness.skill_library.bind_executors`` folds each into
    #: ``bindings.<emb>.policies.<key>``; the record file itself is untouched.
    executors: tuple[dict, ...] = ()


PROVIDES_KINDS = frozenset({"embodiment", "predicate", "recovery", "skill", "planner",
                            "executor"})


def card_provides(data: dict, plugin: str) -> list[dict]:
    """Every ``[[provides]]`` entry of a SINGLE manifest, shape-checked (fail loud)."""
    out, seen = [], set()
    for raw in data.get("provides", []):
        kind, ref = raw.get("kind"), raw.get("ref")
        if kind not in PROVIDES_KINDS:
            raise ValueError(f"plugin {plugin!r}: provides kind {kind!r} not in "
                             f"{sorted(PROVIDES_KINDS)}")
        if not isinstance(ref, str) or ":" not in ref:
            raise ValueError(f"plugin {plugin!r}: provides ref must be 'module:attr', got {ref!r}")
        name = raw.get("name", ref.rpartition(":")[2])
        entry = {"kind": kind, "ref": ref, "name": name, "plugin": plugin}
        if kind == "predicate":
            reads, args = raw.get("reads"), raw.get("args", [])
            if not isinstance(reads, list) or not all(isinstance(k, str) for k in reads):
                raise ValueError(f"plugin {plugin!r}: predicate {name!r} needs reads = [str, ...]")
            if not all(isinstance(a, str) for a in args):
                raise ValueError(f"plugin {plugin!r}: predicate {name!r} args must be strings")
            entry.update(reads=tuple(reads), args=tuple(args))
        if kind == "executor":  # which wire the executor speaks (skill_executor.TRANSPORTS)
            entry["transport"] = raw.get("transport", "inproc")
        if (kind, name) in seen:
            raise ValueError(f"plugin {plugin!r} provides {kind} {name!r} twice")
        seen.add((kind, name))
        out.append(entry)
    return out


def _load(root: Path) -> list[tuple[str, dict]]:
    """(plugin dir name, parsed manifest), in a deterministic scan order. A root
    that is itself ONE card (``plugins/candidates/<name>`` via PH_PLUGINS_EXTRA)
    loads as that single card."""
    if (root / "manifest.toml").is_file():
        return [(root.name, tomllib.loads((root / "manifest.toml").read_text()))]
    return [(mf.parent.name, tomllib.loads(mf.read_text()))
            for mf in sorted(root.glob("*/manifest.toml"))]


def card_mounts(data: dict) -> list[Mount]:
    """Every Mount a SINGLE manifest declares.

    ``discover`` folds many manifests (unioning, loud on collision);
    ``plugin_doctor`` checks one card in isolation. Both share this per-manifest
    ``cap -> ref (+params)`` extraction so the mount shape has one reader.
    """
    out = []
    for cap, spec in data.get("mounts", {}).items():
        ref = spec if isinstance(spec, str) else spec["ref"]
        params = {} if isinstance(spec, str) else dict(spec.get("params", {}))
        out.append(Mount(cap, ref, params))
    return out


def mount_params(ref: str, root: Path = PLUGINS_ROOT) -> dict:
    """The params the card declaring provider ``ref`` mounts it with, enabled or
    not: a segment an arm routes to an unmounted card's provider still runs under
    that card's declared contract. ``{}`` when no card mounts the ref."""
    cards = _load(root)
    params: dict = next((dict(m.params) for _, data in cards
                         for m in card_mounts(data) if m.provider == ref), {})
    # The hosting card's top-level ``[tunables]`` table reaches EVERY provider it
    # hosts (``plugins.<card>.<mod>:factory``) as ``params["tunables"]`` -- one
    # shared read, never a per-mount copy; an explicit mount table wins.
    card = ref.partition(":")[0].split(".")[1] if ref.startswith("plugins.") else ""
    tun = next((data.get("tunables") for name, data in cards if name == card), None)
    if tun and "tunables" not in params:
        params["tunables"] = dict(tun)
    hints = next((data.get("tunable_hints") for name, data in cards if name == card), None)
    if hints and "tunable_hints" not in params:   # failure_mode -> knobs to perturb first
        params["tunable_hints"] = {k: list(v) for k, v in hints.items()}
    # PH_MOUNT_PARAMS_OVERRIDE: {ref: {param: value}} -- an evolve trial's tunables
    # perturbation reaching the driver (scripts/evolve.py); one level of nesting
    # merges ([tunables] tables), anything else replaces.
    for k, v in json.loads(os.environ.get("PH_MOUNT_PARAMS_OVERRIDE") or "{}").get(ref, {}).items():
        params[k] = {**params[k], **v} if isinstance(v, dict) and isinstance(params.get(k), dict) else v
    return params


def card_bundles(data: dict) -> dict[str, list[Mount]]:
    """Every named ``[bundles.<name>]`` overlay a SINGLE manifest declares.

    A bundle table maps ``capability -> ref (+params)`` exactly like ``mounts``
    (same string-or-``{ref, params}`` shape), so ``profiles.bundle(name)`` can
    layer it over ``base_profile`` -- the reader lives here so bundle wiring has
    one shape reader, the same as ``card_mounts``."""
    out: dict[str, list[Mount]] = {}
    for name, table in data.get("bundles", {}).items():
        mounts = []
        for cap, spec in table.items():
            ref = spec if isinstance(spec, str) else spec["ref"]
            params = {} if isinstance(spec, str) else dict(spec.get("params", {}))
            mounts.append(Mount(cap, ref, params))
        out[name] = mounts
    return out


def _claim(registry: dict, owner: dict, key, value, *, kind: str, plugin: str) -> None:
    """Record key->value in one namespace, refusing a second claimant loudly.

    ``owner`` is per-namespace (capabilities, tasks and campaigns are independent
    name spaces), so a task and a campaign may legitimately share a name -- only a
    second card claiming the SAME kind of thing is a collision.
    """
    if key in owner:
        raise ValueError(
            f"duplicate {kind} {key!r}: declared by both {owner[key]!r} and "
            f"{plugin!r} -- two cards cannot claim one {kind}")
    owner[key] = plugin
    registry[key] = value


def discover(root: Path = PLUGINS_ROOT) -> Registry:
    """Fold every ``<root>/*/manifest.toml`` into one registry (union, loud on collision)."""
    mounts: dict[str, Mount] = {}
    task_bindings: dict[str, dict] = {}
    campaigns: dict[str, str] = {}
    third_party: dict[str, tuple[str, ...]] = {}
    bundles: dict[str, tuple[Mount, ...]] = {}
    recoveries: dict[str, tuple[str, str]] = {}
    provides: list[dict] = []
    cap_owner: dict = {}
    task_owner: dict = {}
    camp_owner: dict = {}
    bundle_owner: dict = {}
    recov_owner: dict = {}
    benchmarks: dict[str, dict] = {}
    bench_owner: dict = {}
    executors: list[dict] = []

    # PH_PLUGINS_EXTRA: colon-separated extra card roots folded after ``root``
    # (a test's tmp card beside the installed ones; same collision rules).
    extra = [Path(r) for r in os.environ.get("PH_PLUGINS_EXTRA", "").split(":") if r]
    for plugin, data in [c for r in (root, *extra) for c in _load(r)]:
        if data.get("actuation", "sim") == "real":
            raise ValueError(
                f"plugin {plugin!r} declares actuation:real; the sim runtime "
                "refuses it (a real actuator needs a separate authenticated runtime)")
        # third_party OWNERSHIP is declared even for a disabled card: its .py files
        # still sit in plugins/ and test_boundaries scans them (AST, not import)
        # regardless of enabled, so an inactive card that imports robocasa must
        # still be allowed to. This is ABOVE the enabled gate for that reason; it
        # feeds no mount, so the base_profile sha (folded over mounts only) is
        # untouched -- the robocasa card sits inactive yet owns robocasa/robosuite.
        if "third_party" in data:
            third_party[plugin] = tuple(data["third_party"])
        # Recovery OWNERSHIP is likewise declared even for a disabled card, and for
        # the same reason: a second-simulator card is enabled = false PERMANENTLY
        # (it re-claims embodiment.env and activates per session through a mission
        # binding), so gating its repair shapes on `enabled` would mean a card that
        # can never contribute one -- and RSI would report "nothing to work with"
        # for an embodiment that plainly has primitives. Like third_party this feeds
        # no mount, so the base_profile sha is untouched.
        for name, spec in data.get("recoveries", {}).items():
            ref = spec if isinstance(spec, str) else spec["ref"]
            _claim(recoveries, recov_owner, name, (plugin, ref),
                   kind="recovery", plugin=plugin)
        # provides is per-card capability ownership (a predicate binding is keyed
        # by the declaring card), so it folds above the enabled gate like recoveries.
        provides += card_provides(data, plugin)
        # ``enabled = false`` = installed but inactive: doctor it (card_mounts
        # reads the file directly, ignoring this), then flip it on and disable the
        # incumbent. Lets an ALTERNATIVE provider for an already-claimed seam --
        # the qwen reasoner card beside plugins/reasoner (R7) -- sit in the cage
        # without tripping the duplicate-capability guard, and keeps it out of the
        # folded plan so the base sha is untouched.
        if not data.get("enabled", True):
            continue
        for m in card_mounts(data):
            _claim(mounts, cap_owner, m.capability, m,
                   kind="capability", plugin=plugin)
        for task, binding in data.get("task_bindings", {}).items():
            _claim(task_bindings, task_owner, task, dict(binding),
                   kind="task", plugin=plugin)
        for name, script in data.get("campaigns", {}).items():
            _claim(campaigns, camp_owner, name, script,
                   kind="campaign", plugin=plugin)
        for name, spec in data.get("benchmarks", {}).items():
            _claim(benchmarks, bench_owner, name, dict(spec),
                   kind="benchmark", plugin=plugin)
        for name, mounts_ in card_bundles(data).items():
            _claim(bundles, bundle_owner, name, tuple(mounts_),
                   kind="bundle", plugin=plugin)
        for key, spec in data.get("executors", {}).items():
            executors.append({"key": key, "skill": spec["skill"], "embodiment": spec["embodiment"],
                              "ref": spec["ref"], "transport": spec.get("transport", "inproc"),
                              "plugin": plugin})

    return Registry(tuple(mounts.values()), task_bindings, campaigns,
                    third_party, bundles, recoveries, tuple(provides), benchmarks,
                    tuple(executors))
