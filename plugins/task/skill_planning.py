"""Natural-language task -> skill plan, over the skill graph and the task bindings.

The closed loop this module owns (everything else is reused, nothing forked):

    instruction
      -> retrieval        harness.unified_skill_graph (IS_A subtree, progressive disclosure)
                          + the instruction-driven task bindings in the manifest union
      -> one CHANNEL      the graph vocabulary (planning-only) OR one task binding's
                          card-authored catalogue (executable through the runtime)
      -> DeepSeek         the binding's own TaskPlanner (plugins.planner_vlm by ref):
                          strict-JSON {goal, nodes[], verify[]}, one re-ask on bad JSON
      -> validate         plugins.task.validate.validate_plan -- the SAME gate the
                          runtime applies before dispatch, not a copy
      -> expand           HAS_STAGE / DECOMPOSES_TO, server-side, recursive
      -> bindings         a leaf is bound only when a task binding declares its NAME

Two things are kept apart on purpose. A skill that EXISTS in the graph
(``executable: true`` in the annotator's ontology) and a skill that is BOUND to a
policy/driver in this repository are different facts; ``executable`` in the
response is true only when every leaf is bound, and a plan with any unbound leaf
is ``planning_only`` with the gaps listed. No execution happens here at all:
``verify_plan_record`` hands the board the brief to drop, and the resident
runtime re-plans and re-validates from that brief as the sole authority.

Boundaries: this module imports ``harness`` and its own package; the planner and
the graph card's vocabulary are reached by ref string (``harness.registry``),
never by a sibling import (``tests/test_boundaries.py``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from harness.manifest import Registry, discover
from harness.registry import load_attr, load_provider
from harness.unified_skill_graph import (
    Leaf,
    SkillGraphError,
    UnifiedSkillGraph,
    load_graph,
    tokens,
)
from plugins.task.validate import validate_plan

#: The DeepSeek-backed TaskPlanner card, resolved by ref (the §1b crossing).
PLANNER_REF = "plugins.planner_vlm:provider"
#: The graph-vocabulary card, resolved by ref: arg schema, symbolic oracle, aliases.
GRAPH_CARD = "plugins.skill_graph_robocasa"
#: Mirrors scripts/harness_runtime._MAX_INSTRUCTION_CHARS: a preview accepts
#: exactly the instruction the runtime would.
MAX_INSTRUCTION_CHARS = 4000
#: Scratch seed (CLAUDE.md: 42xxxx never burns the ledger) for previews/submits.
SCRATCH_SEED = 424242
#: The kitchen runtime: both the graph and every RoboCasa task route there.
DEFAULT_SESSION = "session-robocasa"
#: How many graph skills retrieval seeds with before the taxonomy closure.
RETRIEVAL_K = 8
#: Planning budget handed to the model for a graph-only preview (no binding
#: declares one). Task channels use their binding's max_actuations.
GRAPH_PREVIEW_BUDGET = 24
#: Reply cap for a planning preview, over planner_vlm's own 2048 default.
#: A preview asks for the WHOLE graph in one reply, and the largest installed
#: catalogue (pack_all_robocasa: four skills over four objects) is a 16-node,
#: 16-verify document; a reasoning model spends part of this budget thinking
#: before it writes any of it. Measured: at 2048 deepseek-v4-pro truncated that
#: plan at 646 characters and both the first reply and the one re-ask came back
#: unparseable, so the preview reported `rejected` for a plan the model could
#: state perfectly well. The planner card keeps its own default for the runtime
#: path, which plans one graph per replan under the same cap.
PREVIEW_MAX_TOKENS = 8192

STATUS_EXECUTABLE = "executable"
STATUS_PLANNING_ONLY = "planning_only"
STATUS_REJECTED = "rejected"
STATUS_NO_MATCH = "no_match"


class PlanningError(ValueError):
    """A request this module refuses before any model call (bad instruction,
    unknown channel). Never raised for a model's mistake -- that is ``rejected``."""


# --- channels -----------------------------------------------------------------


@dataclass(frozen=True)
class Channel:
    """One planning vocabulary the instruction may route to."""

    id: str
    kind: str                       # "graph" | "task"
    task: str | None                # the manifest task name for kind == "task"
    binding: Mapping[str, Any] | None
    vocabulary: frozenset[str]      # retrieval tokens
    planner_ref: str

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "task": self.task,
                "planner": self.planner_ref}


_PLANNER_MARK: dict[str, bool] = {}


def _instruction_driven(planner_ref: str) -> bool:
    """A binding is natural-language plannable iff its planner is the
    non-deterministic kind (``deterministic = False`` -- the same marker
    plugin_doctor reads). A table planner ignores the instruction, so offering
    it would preview a plan the runtime would never run."""
    if planner_ref not in _PLANNER_MARK:
        try:
            prov = load_provider(planner_ref, {})
            _PLANNER_MARK[planner_ref] = getattr(prov, "deterministic", True) is False
        except Exception:  # noqa: BLE001 -- an unimportable card is not a channel
            _PLANNER_MARK[planner_ref] = False
    return _PLANNER_MARK[planner_ref]


def _binding_vocabulary(binding: Mapping[str, Any]) -> frozenset[str]:
    words: list[str] = []
    for key in ("default_instruction",):
        if key in binding:
            words += tokens(str(load_attr(binding[key])))
    catalogue = load_attr(binding["catalogue"])
    for skill in catalogue:
        words += tokens(skill.replace("_", " "))
    if "skill_docs" in binding:
        for doc in load_attr(binding["skill_docs"]).values():
            words += tokens(str(doc.get("description", "")))
    if "planning_context" in binding:
        ctx = load_attr(binding["planning_context"])
        for key in ("scene", "benchmark"):
            if ctx.get(key):
                words += tokens(str(ctx[key]))
        for key in ("objects", "receptacles"):
            for item in ctx.get(key) or ():
                words += tokens(str(item))
    return frozenset(words)


def task_channels(registry: Registry | None = None) -> list[Channel]:
    """Every task binding whose planner is instruction-driven, name-sorted."""
    reg = registry or discover()
    out = []
    for task, binding in sorted(reg.task_bindings.items()):
        if "catalogue" not in binding or "planner" not in binding:
            continue
        if not _instruction_driven(binding["planner"]):
            continue
        try:
            vocab = _binding_vocabulary(binding)
        except Exception:  # noqa: BLE001, S112 -- a card whose data refs will not import here is not a channel
            continue
        out.append(Channel(task, "task", task, binding, vocab, binding["planner"]))
    return out


def graph_channel(graph: UnifiedSkillGraph) -> Channel:
    card = load_attr(f"{GRAPH_CARD}:CHANNEL")
    return Channel(card, "graph", None, None, frozenset(), PLANNER_REF)


def route(instruction: str, graph: UnifiedSkillGraph, channels: Sequence[Channel],
          pinned: str = "auto") -> tuple[Channel, dict]:
    """Pick the channel: the most distinct instruction tokens matched; ties go
    to an executable (task) channel, then to the id. ``pinned`` names one
    channel outright. Returns the channel plus the scoring evidence."""
    q = sorted(set(tokens(instruction)))
    retrieval = graph.retrieve_subtree(instruction, RETRIEVAL_K)
    graph_matched = sorted({t for s in retrieval["seeds"] for t in s["matched"]})
    rows = []
    for ch in channels:
        if ch.kind == "graph":
            matched = graph_matched
        else:
            matched = [t for t in q if t in ch.vocabulary]
        rows.append({"id": ch.id, "kind": ch.kind, "task": ch.task,
                     "score": len(matched), "matched": matched})
    rows.sort(key=lambda r: (-r["score"], 0 if r["kind"] == "task" else 1, r["id"]))
    by_id = {ch.id: ch for ch in channels}
    if pinned != "auto":
        if pinned not in by_id:
            raise PlanningError(f"unknown channel {pinned!r}; known: {sorted(by_id)}")
        chosen = by_id[pinned]
    else:
        chosen = by_id[rows[0]["id"]] if rows else None
    if chosen is None:
        raise PlanningError("no planning channel is installed")
    return chosen, {"query_tokens": q, "candidates": rows, "pinned": pinned != "auto",
                    "graph_retrieval": retrieval}


# --- bindings -----------------------------------------------------------------


def binding_index(registry: Registry) -> dict[str, list[str]]:
    """skill NAME -> the tasks whose catalogue binds it. The only way a graph
    skill becomes executable is a card declaring that exact name here."""
    index: dict[str, list[str]] = {}
    for task, binding in sorted(registry.task_bindings.items()):
        if "catalogue" not in binding:
            continue
        try:
            catalogue = load_attr(binding["catalogue"])
        except Exception:  # noqa: BLE001, S112 -- unimportable here (a sim venv's card) binds nothing here
            continue
        for skill in catalogue:
            index.setdefault(skill, []).append(task)
    return index


def skill_library_snapshot(registry: Registry | None = None,
                           graph: UnifiedSkillGraph | None = None) -> dict:
    """Union the annotation taxonomy with installed runtime task catalogues.

    ``bound`` means an exact runtime catalogue entry exists. Canonical aliases
    are shown as implementation candidates but never promoted to direct graph
    bindings, because the runtime cannot dispatch them by the graph name.
    """
    reg = registry or discover()
    g = graph or load_graph()
    taxonomy = g.library_snapshot()
    exact = binding_index(reg)
    aliases = dict(load_attr(f"{GRAPH_CARD}:LIBRARY_TO_CANONICAL"))
    runtime: dict[str, dict[str, Any]] = {}
    for task, binding in sorted(reg.task_bindings.items()):
        if "catalogue" not in binding:
            continue
        try:
            catalogue = load_attr(binding["catalogue"])
            docs = (load_attr(binding["skill_docs"])
                    if "skill_docs" in binding else {})
        except Exception:  # noqa: BLE001, S112 -- unavailable cards bind nothing
            continue
        for name, schema in catalogue.items():
            row = runtime.setdefault(name, {
                "name": name, "canonical": aliases.get(name), "bindings": [],
            })
            doc = docs.get(name) if isinstance(docs, Mapping) else None
            row["bindings"].append({
                "task": task,
                "policy": binding.get("policy"),
                "args": {arg: typ.__name__ for arg, typ in schema.items()},
                "description": doc.get("description") if isinstance(doc, Mapping) else None,
            })
    by_canonical: dict[str, list[str]] = {}
    for name, row in runtime.items():
        canonical = row.get("canonical")
        if canonical:
            by_canonical.setdefault(canonical, []).append(name)
    graph_skills = 0
    direct_bound = 0
    for row in taxonomy["nodes"]:
        if row["kind"] not in ("observed_skill", "canonical_skill"):
            continue
        graph_skills += 1
        tasks = list(exact.get(row["name"], ()))
        row["bound"] = bool(tasks)
        row["binding_tasks"] = tasks
        row["implementation_candidates"] = sorted(by_canonical.get(row["name"], ()))
        direct_bound += int(bool(tasks))
    runtime_rows = sorted(runtime.values(), key=lambda row: row["name"])
    return {
        "schema_version": "1.0",
        "graph": taxonomy,
        "runtime_skills": runtime_rows,
        "summary": {
            "graph_skills": graph_skills,
            "graph_directly_bound": direct_bound,
            "graph_unbound": graph_skills - direct_bound,
            "runtime_skills": len(runtime_rows),
            "task_bindings": len(reg.task_bindings),
        },
        "provenance": g.provenance(),
        "semantics": {
            "bound": "exact installed task catalogue entry with a policy/driver",
            "graph_executable": "ontology annotation only; not an execution binding",
            "implementation_candidates": "runtime skills mapped to the same canonical concept; not a direct graph binding",
            "tree": "IS_A taxonomy",
            "recipes": "DECOMPOSES_TO authored composition; stages come from HAS_STAGE",
        },
    }


def _embodiment_of(binding: Mapping[str, Any], ctx: Mapping[str, Any]) -> str | None:
    if ctx.get("benchmark"):
        return str(ctx["benchmark"])
    env = binding.get("env")
    if isinstance(env, str) and "embodiment_" in env:
        return env.split(":")[0].rsplit("embodiment_", 1)[-1]
    return None


# --- the plan -----------------------------------------------------------------


def _bounded_instruction(instruction: Any) -> str:
    if (not isinstance(instruction, str) or not instruction.strip()
            or len(instruction) > MAX_INSTRUCTION_CHARS):
        raise PlanningError(
            "instruction must be a non-empty string of at most "
            f"{MAX_INSTRUCTION_CHARS} characters")
    return instruction.strip()


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _type_names(catalogue: Mapping[str, Mapping[str, type]]) -> dict[str, dict[str, str]]:
    return {s: {a: t.__name__ for a, t in schema.items()} for s, schema in catalogue.items()}


def _graph_vocabulary(graph: UnifiedSkillGraph, retrieval: dict, bound: Mapping[str, list[str]]):
    """The compact graph catalogue for the model: the retrieved skills, their
    arg schema (card-authored), one card each, and binding availability."""
    rows = list(retrieval["skills"])
    if not rows:
        # A pinned graph channel with no retrieval hit: offer the canonical
        # interfaces (the taxonomy's leaf concepts), never the whole graph.
        rows = [{"name": n.name, "kind": n.kind, "score": 0, "matched": [],
                 "why": ["fallback: canonical interface"],
                 "taxonomy_path": list(graph.taxonomy_path(n.name))}
                for n in graph.skills() if n.kind == "canonical_skill"]
        retrieval = {**retrieval, "skills": rows}
    names = [row["name"] for row in rows]
    catalogue = load_attr(f"{GRAPH_CARD}:catalogue_for")(graph, names)
    oracles = tuple(load_attr(f"{GRAPH_CARD}:ORACLES"))
    docs = {name: graph.describe(name) for name in names}
    availability = {name: {"bound": name in bound, "tasks": list(bound.get(name, ()))}
                    for name in names}
    taxonomy = {"categories": retrieval["categories"],
                "skills": [{"name": r["name"], "graph_kind": r["kind"],
                            "taxonomy_path": r["taxonomy_path"],
                            "stages": docs[r["name"]]["stages"],
                            "decomposition": docs[r["name"]]["decomposition"]}
                           for r in retrieval["skills"]]}
    return catalogue, oracles, docs, availability, taxonomy


def _task_vocabulary(binding: Mapping[str, Any]):
    catalogue = dict(load_attr(binding["catalogue"]))
    oracles = tuple(load_attr(binding["oracles"]))
    docs = dict(load_attr(binding["skill_docs"])) if "skill_docs" in binding else {}
    ctx = dict(load_attr(binding["planning_context"])) if "planning_context" in binding else {}
    default = str(load_attr(binding["default_instruction"])) if "default_instruction" in binding else None
    return catalogue, oracles, docs, ctx, default


def _leaf_row(node: Mapping, leaf: Leaf, bound: Mapping[str, list[str]],
              binding_info: Mapping | None) -> dict:
    is_bound = binding_info is not None
    reason = None
    if not is_bound:
        by_name = bound.get(leaf.skill) or (bound.get(leaf.canonical) if leaf.canonical else None)
        reason = (f"no task binding declares skill {leaf.skill!r}"
                  + (f" (canonical {leaf.canonical!r})" if leaf.canonical else "")
                  + "; it exists only as a RoboCasa annotation"
                  + (f"; a same-named binding exists in {by_name}" if by_name else ""))
    return {"node": node["id"], **leaf.as_dict(), "args": dict(node.get("args") or {}),
            "bound": is_bound, "binding": dict(binding_info) if binding_info else None,
            "reason": reason}


def expand_plan(plan: Mapping, channel: Channel, graph: UnifiedSkillGraph,
                bound: Mapping[str, list[str]], *, expand: bool = True) -> dict:
    """Server-side expansion of a VALIDATED plan into its leaf chain.

    Graph channel: each node's skill expands by HAS_STAGE / DECOMPOSES_TO
    (recursive), and every leaf is checked against the binding index by NAME.
    Task channel: each node IS a bound library skill; its leaf is the segment
    the embodiment binding maps it to (``task_template`` with the node args).
    Returns ``{nodes, chain, terminal}`` where ``chain`` is the flat leaf order
    (a leaf's ``after`` names the leaves it follows) and ``terminal`` is the
    graph's terminal marker.
    """
    aliases = load_attr(f"{GRAPH_CARD}:LIBRARY_TO_CANONICAL")
    seg_specs: Mapping[str, Any] = {}
    embodiment = None
    if channel.kind == "task" and channel.binding is not None:
        if "segment_specs" in channel.binding:
            seg_specs = load_attr(channel.binding["segment_specs"])
        ctx = (load_attr(channel.binding["planning_context"])
               if "planning_context" in channel.binding else {})
        embodiment = _embodiment_of(channel.binding, ctx)
    nodes_out: list[dict] = []
    chain: list[dict] = []
    last_leaf_of: dict[str, list[str]] = {}
    for node in plan["nodes"]:
        skill = node["skill"]
        leaves: list[Leaf] = []
        info: dict | None = None
        taxonomy_path: list[str] = []
        stages: list[dict] = []
        decomposition: list[dict] = []
        if channel.kind == "graph":
            taxonomy_path = list(graph.taxonomy_path(skill))
            stages = [dict(s) for s in graph.stages(skill)]
            decomposition = [{"skill": part, "taxonomy_path": list(graph.taxonomy_path(part)),
                              "bound": part in bound}
                             for part in graph.decomposition(skill)]
            leaves = list(graph.expand(skill)) if expand else [
                Leaf(skill, None, graph.canonical_parent(skill), "NODE",
                     tuple(taxonomy_path))]
        else:
            canonical = aliases.get(skill)
            if canonical and graph.has(canonical):
                taxonomy_path = list(graph.taxonomy_path(canonical))
            spec = seg_specs.get(skill) or {}
            template = spec.get("task_template")
            try:
                stage = template.format(**node["args"]) if template else None
            except (KeyError, IndexError):
                stage = template
            info = {"task": channel.task, "policy": channel.binding["policy"],
                    "embodiment": embodiment, "task_template": template}
            leaves = [Leaf(skill, stage, canonical, "TASK_BINDING", tuple(taxonomy_path))]
        rows = []
        for leaf in leaves:
            leaf_info = info if channel.kind == "task" else (
                {"tasks": bound[leaf.skill]} if leaf.skill in bound else None)
            row = _leaf_row(node, leaf, bound, leaf_info)
            # execution order: the previous leaf of this node, else the last
            # leaves of the node's `after` parents
            prev = [rows[-1]["label"]] if rows else [
                lbl for parent in node["after"] for lbl in last_leaf_of.get(parent, ())]
            row["after"] = prev
            rows.append(row)
        last_leaf_of[node["id"]] = [rows[-1]["label"]] if rows else []
        chain.extend(rows)
        nodes_out.append({"id": node["id"], "skill": skill,
                          "kind": node.get("kind", "manipulate"),
                          "args": dict(node["args"]), "after": list(node["after"]),
                          "taxonomy_path": taxonomy_path, "stages": stages,
                          "decomposition": decomposition,
                          "leaves": [r["label"] for r in rows],
                          "bound": all(r["bound"] for r in rows)})
    return {"nodes": nodes_out, "chain": chain, "terminal": "done"}


def _planner_for(channel: Channel, planner=None, planner_params: Mapping | None = None):
    """The channel's planner, or the caller's stand-in (a test's fake endpoint).

    ``max_tokens`` defaults to PREVIEW_MAX_TOKENS rather than the card's own
    value: a preview must fit the whole graph in one reply. An explicit
    ``max_tokens`` in ``planner_params`` still wins.
    """
    if planner is not None:
        return planner
    params = dict(planner_params or {})
    params.setdefault("max_tokens", PREVIEW_MAX_TOKENS)
    return load_provider(channel.planner_ref, params)


def plan_skill_task(instruction: str, *, session: str = DEFAULT_SESSION,
                    expand: bool = True, channel: str = "auto",
                    seed: int = SCRATCH_SEED, planner=None,
                    planner_params: Mapping | None = None,
                    graph: UnifiedSkillGraph | None = None,
                    registry: Registry | None = None) -> dict:
    """The read-only planning call behind the ``plan_skill_task`` faces.

    Writes nothing anywhere. ``planner`` / ``planner_params`` exist so a test
    can hand in a fake endpoint (never the real API); the default is the
    channel's own planner ref with its own defaults (DeepSeek, key from env or
    the console credential store -- never from here).
    """
    instruction = _bounded_instruction(instruction)
    g = graph or load_graph()
    reg = registry or discover()
    channels = [graph_channel(g), *task_channels(reg)]
    chosen, routing = route(instruction, g, channels, channel)
    bound = binding_index(reg)
    base = {
        "instruction": instruction, "session": session, "seed": seed,
        "channel": {**chosen.as_dict(), "score": next(
            (r["score"] for r in routing["candidates"] if r["id"] == chosen.id), 0),
            "matched": next((r["matched"] for r in routing["candidates"]
                             if r["id"] == chosen.id), [])},
        "retrieval": {"query_tokens": routing["query_tokens"],
                      "candidates": routing["candidates"], "pinned": routing["pinned"],
                      "graph_seeds": routing["graph_retrieval"]["seeds"],
                      "graph_categories": routing["graph_retrieval"]["categories"],
                      "graph_total_skills": routing["graph_retrieval"]["total_skills"]},
        "graph_provenance": g.provenance(),
        "planner": {"ref": chosen.planner_ref},
    }
    if base["channel"]["score"] == 0 and not routing["pinned"]:
        return {**base, "status": STATUS_NO_MATCH, "goal": None,
                "selected_catalogue": None, "composite_plan": None,
                "expanded_plan": None, "executable": False, "missing_bindings": [],
                "unbound_oracles": [], "validation": {
                    "ok": False, "message": "no installed skill vocabulary matches the "
                    "instruction; nothing was sent to the model"}}

    # -- the compact catalogue for THIS channel --------------------------------
    planning_context: dict = {}
    default_instruction = None
    taxonomy: dict | None = None
    if chosen.kind == "graph":
        catalogue, oracles, docs, availability, taxonomy = _graph_vocabulary(
            g, routing["graph_retrieval"], bound)
        budget = GRAPH_PREVIEW_BUDGET
        task_name = f"skill_graph:{chosen.id}"
    else:
        catalogue, oracles, docs, planning_context, default_instruction = _task_vocabulary(
            chosen.binding)
        availability = {name: {"bound": True, "tasks": [chosen.task]} for name in catalogue}
        budget = int(chosen.binding.get("max_actuations", GRAPH_PREVIEW_BUDGET))
        task_name = chosen.task
    catalogue_digest = _digest({"channel": chosen.id, "catalogue": _type_names(catalogue),
                                "oracles": list(oracles), "graph": g.sha256})
    selected = {
        "channel": chosen.id, "size": len(catalogue),
        "graph_total_skills": len(g.skills()),
        "oracles": list(oracles), "catalogue_digest": catalogue_digest,
        "skills": [{"name": name, "args": _type_names({name: schema})[name],
                    "bound": availability[name]["bound"],
                    "tasks": availability[name]["tasks"],
                    "doc": docs.get(name, {})}
                   for name, schema in catalogue.items()],
    }
    if taxonomy is not None:
        selected["taxonomy"] = taxonomy

    # -- the model (strict JSON, one re-ask inside the planner) -----------------
    brief = {"task": task_name, "seed": seed, "instruction": instruction,
             "catalogue": catalogue, "oracles": oracles, "skill_docs": docs,
             "planning_context": planning_context, "scene": {}, "budget": budget,
             "catalogue_digest": catalogue_digest,
             "binding_availability": availability}
    if default_instruction:
        brief["default_instruction"] = default_instruction
    if taxonomy is not None:
        brief["taxonomy"] = taxonomy
    plan = _planner_for(chosen, planner, planner_params).plan(brief)
    plan = json.loads(json.dumps(plan, sort_keys=True, default=str))  # pure JSON
    goal = plan.get("goal") if isinstance(plan, Mapping) else None

    # -- the validator: the runtime's gate, unchanged --------------------------
    ok, msg = validate_plan(plan, catalogue, oracles, requirements=planning_context or None)
    validation = {"ok": ok, "message": msg, "validator": "plugins.task.validate.validate_plan",
                  "unparseable": isinstance(goal, str) and "unparseable" in goal
                  and not plan.get("nodes")}
    if not ok:
        return {**base, "status": STATUS_REJECTED, "goal": goal,
                "selected_catalogue": selected, "composite_plan": {
                    "plan_id": None, "channel": chosen.id, "task": chosen.task,
                    "instruction": instruction, "plan": plan},
                "expanded_plan": None, "executable": False, "missing_bindings": [],
                "unbound_oracles": [] if chosen.kind == "task" else list(oracles),
                "validation": validation}

    record = {"plan_id": _digest({"channel": chosen.id, "task": chosen.task,
                                  "instruction": instruction, "plan": plan}),
              "channel": chosen.id, "task": chosen.task, "instruction": instruction,
              "plan": plan}
    expanded = expand_plan(plan, chosen, g, bound, expand=expand)
    missing = [{k: row[k] for k in ("node", "label", "skill", "stage", "canonical", "reason")}
               for row in expanded["chain"] if not row["bound"]]
    unbound_oracles = [] if chosen.kind == "task" else list(oracles)
    executable = (chosen.kind == "task" and not missing and not unbound_oracles)
    return {**base,
            "status": STATUS_EXECUTABLE if executable else STATUS_PLANNING_ONLY,
            "goal": goal, "selected_catalogue": selected, "composite_plan": record,
            "expanded_plan": expanded, "executable": executable,
            "missing_bindings": missing, "unbound_oracles": unbound_oracles,
            "validation": validation}


# --- the execution boundary --------------------------------------------------


def verify_plan_record(record: Mapping, *, seed: int = SCRATCH_SEED,
                       max_replans: int | None = None, max_actuations: int | None = None,
                       graph: UnifiedSkillGraph | None = None,
                       registry: Registry | None = None) -> dict:
    """Re-verify a plan record the board is asked to execute, from scratch.

    Trusts nothing in ``record`` beyond its shape: the channel must be a TASK
    channel installed right now, the plan must pass ``validate_plan`` against
    that task's current catalogue/oracles/planning_context, and every leaf must
    be bound. Returns ``{ok, status, brief, plan_id, ...}``; ``brief`` is the
    selector+budgets the board may drop (``{"kind":"task","task",...,
    "instruction","seed"[, budgets]}``) and is present ONLY when ``ok``.
    A planning-only or rejected record is refused here, before any drop.
    """
    if not isinstance(record, Mapping):
        return {"ok": False, "status": STATUS_REJECTED, "error": "plan record must be an object"}
    channel_id = record.get("channel")
    plan = record.get("plan")
    try:
        instruction = _bounded_instruction(record.get("instruction"))
    except PlanningError as exc:
        return {"ok": False, "status": STATUS_REJECTED, "error": str(exc)}
    if not isinstance(plan, Mapping):
        return {"ok": False, "status": STATUS_REJECTED, "error": "plan record has no plan object"}
    g = graph or load_graph()
    reg = registry or discover()
    channels = {ch.id: ch for ch in task_channels(reg)}
    graph_id = graph_channel(g).id
    if channel_id == graph_id:
        return {"ok": False, "status": STATUS_PLANNING_ONLY,
                "error": f"channel {graph_id!r} is planning-only: RoboCasa annotation "
                         "skills have no policy/driver binding in this repository"}
    chosen = channels.get(channel_id)
    if chosen is None:
        return {"ok": False, "status": STATUS_REJECTED,
                "error": f"unknown or non-executable channel {channel_id!r}; "
                         f"executable channels: {sorted(channels)}"}
    catalogue, oracles, _docs, planning_context, _default = _task_vocabulary(chosen.binding)
    ok, msg = validate_plan(plan, catalogue, oracles, requirements=planning_context or None)
    if not ok:
        return {"ok": False, "status": STATUS_REJECTED, "error": f"validate_plan: {msg}"}
    expanded = expand_plan(plan, chosen, g, binding_index(reg))
    missing = [row["label"] for row in expanded["chain"] if not row["bound"]]
    if missing:
        return {"ok": False, "status": STATUS_PLANNING_ONLY,
                "error": f"unbound leaves: {missing}", "missing_bindings": missing}
    plan_id = _digest({"channel": chosen.id, "task": chosen.task,
                       "instruction": instruction, "plan": plan})
    brief: dict[str, Any] = {"kind": "task", "task": chosen.task,
                             "instruction": instruction, "seed": int(seed)}
    if max_replans is not None:
        brief["max_replans"] = int(max_replans)
    if max_actuations is not None:
        brief["max_actuations"] = int(max_actuations)
    return {"ok": True, "status": STATUS_EXECUTABLE, "task": chosen.task,
            "plan_id": plan_id, "brief": brief,
            "execution_note": ("the resident runtime re-plans from this brief with the "
                               "same planner and validator; the previewed graph is "
                               "advisory, the runtime's sealed plan is the evidence")}


__all__ = ["Channel", "PlanningError", "SkillGraphError", "binding_index", "expand_plan",
           "graph_channel", "plan_skill_task", "route", "task_channels",
           "verify_plan_record"]
