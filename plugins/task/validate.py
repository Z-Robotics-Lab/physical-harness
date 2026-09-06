"""Plan validation: the untrusted-planner boundary, harness-native.

Same stance as ``governor.proposer.parse_proposal``: every branch is a refusal
a real planner can trigger, and the message is written to be folded straight
back into the next brief. Stdlib-only on purpose — the AST boundary test
forbids reaching zos or sibling plugins, so the boundary is re-authored here,
never reused from zos/graph.py.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from harness.protocol import (TYPES, ExecutionGraph, Node, SkillRecordV0, Task,
                              replan_monotone, validate_graph)

_TYPE_NAME = {t: n for n, t in TYPES.items()}     # python type -> TYPES name

#: The exact key set every node must carry: the 4-key graph dialect zos's
#: cockpit renders is the same JSON this validator admits.
_NODE_KEYS = frozenset({"id", "skill", "args", "after"})
#: Admitted beside them: ``kind`` (dispatch), ``task`` (which declared task the
#: node serves in a composed graph) and ``on_fail`` (protocol Node, inert today).
_NODE_OPTIONAL = frozenset({"kind", "task", "on_fail", "executor"})

#: The node KINDS the loop can dispatch. This frozenset is the single source of
#: truth for the NAMES; ``workload._KIND_HANDLERS`` maps exactly these to their
#: generic handlers and self-checks coverage against this set at import. A node
#: with no ``kind`` defaults to ``"manipulate"`` -- so existing cards (no kind on
#: any node) validate byte-identically, sealed plan shas unmoved.
#: ``recovery``: a planner-inserted repair (``protocol.insert_recovery``); its
#: ``skill`` names an embodiment card's ``[recoveries.*]`` strategy, not a
#: catalogue skill, and it carries no args.
NODE_KINDS = frozenset({"manipulate", "segment", "perceive", "decide", "verify", "recovery"})


def validate_plan(plan: Mapping, catalogue: Mapping[str, Mapping[str, type]],
                  oracles: Collection[str],
                  done: Collection[Mapping] = (),
                  requirements: Mapping | None = None) -> tuple[bool, str]:
    """``(ok, message)`` — fail-first, message names the offender.

    ``catalogue`` maps skill name -> {arg name: required python type}; it is
    authored by the skill side, never by the planner, which only selects and
    parameterizes. ``oracles`` are the verify predicates a plan may name.

    ``done`` is the workload's ledger of completed node declarations, including
    dispatch kind, task, executor and dependencies. A replan keeps their original
    execution identity and ordered prefix: the runner skips these past actions,
    so moving them cannot make their effects happen again.

    ``requirements`` is optional task-authored grounding. An ``objects`` list
    plus ``required_per_object_order`` requires exactly one call to each named
    skill for every object, with dependency ancestry preserving that order.
    ``target_by_object`` fixes any target argument carried by those calls.
    """
    if not isinstance(plan, Mapping):
        return False, f"plan must be a JSON object, got {type(plan).__name__}"
    goal = plan.get("goal")
    if not isinstance(goal, str) or not goal:
        return False, "plan.goal must be a non-empty string"
    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False, "plan.nodes must be a non-empty list of skill calls"
    # A composed (mission) graph declares its tasks with GROUNDED goal preds;
    # nodes then carry ``task``. Covered judges those goals in the protocol gate.
    tasks = plan.get("tasks")
    if tasks is not None:
        if not isinstance(tasks, list) or not tasks:
            return False, "plan.tasks must be a non-empty list of {id, goal}"
        for i, t in enumerate(tasks):
            if (not isinstance(t, Mapping) or not isinstance(t.get("id"), str)
                    or not t["id"] or not isinstance(t.get("goal"), list)
                    or not all(isinstance(g, str) and g for g in t["goal"])):
                return False, f"tasks[{i}] must be {{id: str, goal: [pred ref strings]}}"
    ids: list[str] = []
    for i, node in enumerate(nodes):
        keys = set(node) if isinstance(node, Mapping) else set()
        if not isinstance(node, Mapping) or keys - _NODE_OPTIONAL != _NODE_KEYS:
            return False, (f"node[{i}] must be an object with exactly "
                           f"{sorted(_NODE_KEYS)} (optional {sorted(_NODE_OPTIONAL)})")
        nid = node["id"]
        if not isinstance(nid, str) or not nid:
            return False, f"node[{i}].id must be a non-empty string"
        if nid in ids:
            return False, f"duplicate node id {nid!r}"
        kind = node.get("kind", "manipulate")
        if kind not in NODE_KINDS:
            return False, (f"node {nid!r} declares unknown kind {kind!r}; "
                           f"known kinds: {sorted(NODE_KINDS)}")
        skill = node["skill"]
        if skill not in catalogue and kind != "recovery":
            return False, (f"node {nid!r} names unknown skill {skill!r}; "
                           f"catalogue is {sorted(catalogue)}")
        if not isinstance(node["args"], Mapping):
            return False, f"node {nid!r}.args must be an object"
        after = node["after"]
        # EARLIER ids only: admits exactly the topologically ordered DAGs, so
        # execution order is the list order — determinism for free.
        if not isinstance(after, list) or not all(a in ids for a in after):
            return False, (f"node {nid!r}.after must list ids of earlier nodes; "
                           f"earlier ids: {ids}")
        ids.append(nid)
    # Typed / Grounded / Supported / Covered: protocol.validate_graph is the one
    # legality judge; the catalogue is lifted to arg-only records (no contract
    # predicates), so today's callers get exactly the Typed check.
    records = {name: SkillRecordV0(id=name, name=name,
                                   args={k: _TYPE_NAME[t] for k, t in schema.items()})
               for name, schema in catalogue.items()}
    for n in nodes:  # a recovery strategy is an arg-less record; the loop resolves the name
        if n.get("kind") == "recovery":
            records.setdefault(n["skill"], SkillRecordV0(id=n["skill"], name=n["skill"], args={}))
    # Goals and executors stripped here: Covered and Bound need contract-bearing
    # records (bindings), which the workload's protocol gate (_graph_problems) supplies.
    typed = {**plan, "tasks": [{"id": t["id"], "goal": []} for t in tasks or ()],
             "nodes": [{k: v for k, v in n.items() if k != "executor"} for n in plan["nodes"]]}
    ok, problems = validate_graph(plan_to_graph(typed), records, (), ())
    if not ok:
        return False, "; ".join(problems)
    verify = plan.get("verify")
    if not isinstance(verify, list) or not verify:
        return False, "plan.verify must be a non-empty list: an unverified plan is vacuous"
    for i, v in enumerate(verify):
        if not isinstance(v, Mapping) or v.get("after") not in ids:
            return False, f"verify[{i}].after must name a node id from {ids}"
        pred = v.get("predicate")
        if pred not in oracles:
            return False, (f"verify[{i}] names unknown predicate {pred!r}; "
                           f"declared oracles: {sorted(oracles)}")
    # Verify coverage: every manipulate/segment node must be gated by a machine
    # check — a verify-list edge after it, or a verify-KIND successor node.
    # Without this a planner can emit six action nodes and one verify edge and
    # the five misses fail silently (audit oracles before trusting them).
    covered = {v["after"] for v in verify}
    for node in nodes:
        if node.get("kind", "manipulate") == "verify":
            covered.update(node["after"])
    for node in nodes:
        kind = node.get("kind", "manipulate")
        if kind in ("manipulate", "segment") and node["id"] not in covered:
            return False, (f"node {node['id']!r} (kind {kind!r}) is not covered "
                           "by any verify: add a verify entry after it or a "
                           "verify-kind successor node")
    if requirements:
        objects = requirements.get("objects")
        required_order = requirements.get("required_per_object_order")
        if objects is not None or required_order is not None:
            if (not isinstance(objects, list) or not objects
                    or not all(isinstance(obj, str) and obj for obj in objects)):
                return False, "planning_context.objects must be a non-empty string list"
            if (not isinstance(required_order, list) or not required_order
                    or not all(isinstance(skill, str) and skill
                               for skill in required_order)):
                return False, ("planning_context.required_per_object_order must "
                               "be a non-empty string list")
            ancestors: dict[str, set[str]] = {}
            for node in nodes:
                direct = set(node["after"])
                ancestors[node["id"]] = direct.union(
                    *(ancestors[parent] for parent in direct)) if direct else set()
            known_objects = set(objects)
            for node in nodes:
                if (node["skill"] in required_order
                        and node["args"].get("object") not in known_objects):
                    return False, (f"node {node['id']!r} applies required skill "
                                   f"{node['skill']!r} to undeclared object "
                                   f"{node['args'].get('object')!r}")
            for obj in objects:
                previous: Mapping | None = None
                for skill in required_order:
                    calls = [node for node in nodes
                             if node["skill"] == skill
                             and node["args"].get("object") == obj]
                    if len(calls) != 1:
                        return False, (f"object {obj!r} requires exactly one "
                                       f"{skill!r} call, got {len(calls)}")
                    current = calls[0]
                    if (previous is not None
                            and previous["id"] not in ancestors[current["id"]]):
                        return False, (f"object {obj!r} requires {skill!r} after "
                                       f"{previous['skill']!r}")
                    previous = current
            target_by_object = requirements.get("target_by_object")
            if target_by_object is not None:
                if not isinstance(target_by_object, Mapping):
                    return False, "planning_context.target_by_object must be an object"
                for node in nodes:
                    args = node["args"]
                    obj = args.get("object")
                    if obj in known_objects and "target" in args:
                        expected = target_by_object.get(obj)
                        if args["target"] != expected:
                            return False, (f"node {node['id']!r} targets "
                                           f"{args['target']!r} for object {obj!r}; "
                                           f"expected {expected!r}")
    # Replan stability: every completed node must reappear byte-identical.
    if done:
        old = {"mission": goal, "nodes": [{"task": "main", **d} for d in done]}
        ok, problems = replan_monotone(old, plan_to_graph(plan), [d["id"] for d in done])
        if not ok:
            return False, "; ".join(problems) + "; done nodes must be preserved verbatim"
    return True, ""


def plan_to_graph(plan: Mapping) -> ExecutionGraph:
    """Today's ``{goal, nodes[{id, skill, args, after}], verify}`` plan as a protocol
    ExecutionGraph. A plan with no ``tasks`` gets one implicit task ``main`` that
    every node belongs to; its goal is empty (the verify list is the gate today)."""
    tasks = tuple(Task(id=t["id"], goal=tuple(t.get("goal", ())))
                  for t in plan.get("tasks") or ()) or (Task(id="main", goal=()),)
    nodes = tuple(Node(id=n["id"], task=n.get("task", tasks[0].id), skill=n["skill"],
                       args=dict(n["args"]), after=tuple(n["after"]),
                       on_fail=dict(n.get("on_fail") or {}), executor=n.get("executor"),
                       kind=n.get("kind", "manipulate"))
                  for n in plan["nodes"])
    return ExecutionGraph(mission=plan["goal"], seed=int(plan.get("seed", 0)),
                          tasks=tasks, nodes=nodes)
