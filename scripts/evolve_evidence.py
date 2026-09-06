"""Bounded, read-only views of evidence already available to the RSI controller.

Pages retain source identities. Static parameter references are observations of
code, not a complete dependency analysis or a proposed intervention direction.
"""
from __future__ import annotations

import ast
import copy
import importlib
import inspect
import json
from pathlib import Path
from collections import OrderedDict

from harness.protocol import content_id


def encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str,
                      separators=(",", ":")).encode("utf-8")


def request_messages(body, system, max_bytes):
    """Send a self-contained decision, with stable catalog detail addressable.

    The working cache and audit remain intact. If these reference-based
    reductions cannot fit, the caller's existing budget guard still stops.
    """
    def messages(value):
        return [{"role": "system", "content": system},
                {"role": "user", "content": encoded(value).decode("utf-8")}]

    body = copy.deepcopy(body)
    state = body.get("state") or {}
    if "driver_index" in state and "capabilities" not in state:
        catalog = {key: state[key] for key in ("driver_index", "source_index") if key in state}
        state["catalog_ref"] = {"view": "catalog", "policy_id": state.get("policy_id"),
                                "sha": content_id(catalog)}
        capabilities, drivers = {}, {}
        for node, driver in state["driver_index"].items():
            capability = {key: driver.get(key, {} if key == "tunables" else [])
                          for key in ("tunables", "executors", "modules")}
            key = next((key for key, value in capabilities.items() if value == capability), None)
            if key is None:
                key = f"c{len(capabilities)}"
                capabilities[key] = capability
            drivers[node] = {"skill": driver.get("skill"), "executor": driver.get("executor"), "capability": key}
        state["driver_index"], state["capabilities"] = drivers, capabilities
        state.pop("source_index", None)
        state.pop("source_index_format", None)
    if state.get("cycle_context"):
        state["cycle_context"].pop("memo", None)  # Already carried once as body.memo.
    working_ids = {p.get('policy_id') for p in state.get('working_policies', [])}
    state['historical_probe_replay'] = [p for p in state.get('historical_probe_replay', [])
                                       if p.get('policy_id') not in working_ids]
    latest = body.get("last_tool_result")
    if latest is not None:
        body["retained_evidence"] = [page for page in body.get("retained_evidence", [])
                                     if EvidenceWorkingSet.key(page) != EvidenceWorkingSet.key(latest)]
    result = messages(body)
    if len(encoded(result)) <= max_bytes:
        return body, result
    for page in body.get("retained_evidence", []):
        if page.get("view") == "source":
            data = page.get("data") or {}
            read = {"view": "source", "policy_id": page.get("policy_id"),
                    "cursor": page.get("cursor", 0)}
            if data.get("symbol"):
                read["symbol"] = data["symbol"]
            else:
                read.update({k: data[k] for k in ("module", "start", "end") if k in data})
                read["node"] = page.get("node")
            page["data"] = {"omitted_for_request_budget": True, "read": read}
    result = messages(body)
    if len(encoded(result)) <= max_bytes:
        return body, result
    state = body["state"]
    for observation in state.get("observations", []):
        for node in observation.get("nodes", []):
            if node.get("motion") is not None:
                node.pop("motion")
                node["motion_ref"] = {"view": "trace", "node": node["node"],
                                      "seed": observation.get("seed"),
                                      "policy_id": state.get("policy_id")}
    result = messages(body)
    if len(encoded(result)) <= max_bytes:
        return body, result
    for policy in state.get('working_policies', []):
        measurements = policy.get('measurements') or []
        if not any('gains' in row or 'regressions' in row for row in measurements):
            continue
        policy['measurements_ref'] = {'view': 'history', 'policy_id': policy['policy_id']}
        policy['measurements'] = [{**{k: row[k] for k in ('seed', 'comparable', 'cost') if k in row},
            **{kind + '_count': len(row[kind]) if isinstance(row.get(kind), list) else None
               for kind in ('gains', 'regressions')}} for row in measurements]
    result = messages(body)
    if len(encoded(result)) <= max_bytes:
        return body, result
    for policy in state.get('working_policies', []):
        if not policy.get('tried'):
            continue
        policy['intervention_ref'] = {'view': 'history', 'policy_id': policy['policy_id']}
        policy['tried'] = {k: v for k, v in policy['tried'].items() if k != 'detail'}
    return body, messages(body)


def _definitions(source):
    rows = []
    def visit(items, prefix=""):
        for item in items:
            if not isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            symbol = prefix + item.name
            rows.append({"symbol": symbol, "kind": "class" if isinstance(item, ast.ClassDef) else "function",
                         "start": min([item.lineno, *(d.lineno for d in item.decorator_list)]),
                         "end": item.end_lineno})
            visit(item.body, symbol + ".")
    visit(ast.parse(source).body)
    return rows


def bound_class_index(classes):
    result, modules = [], {}
    for cls in classes:
        try:
            if cls.__module__ not in modules:
                modules[cls.__module__] = _definitions(Path(inspect.getsourcefile(cls)).read_text())
            rows = modules[cls.__module__]
            declaration = next(r for r in rows if r["kind"] == "class" and r["symbol"] == cls.__qualname__)
        except (OSError, TypeError, StopIteration, SyntaxError):
            continue
        prefix = cls.__qualname__ + "."
        result.append({"id": f"{cls.__module__}:{cls.__qualname__}", "module": cls.__module__,
                       "class": cls.__qualname__, "start": declaration["start"], "end": declaration["end"],
                       "methods": [{"id": f"{cls.__module__}:{r['symbol']}", "start": r["start"], "end": r["end"]}
                                   for r in rows if r["kind"] == "function" and r["symbol"].startswith(prefix)
                                   and "." not in r["symbol"][len(prefix):]]})
    return result


def _replay_summary(rows, current_contract_sha=None):
    summaries = []
    for row in (rows or [])[-6:]:
        tried, comparison = row.get("tried") or {}, row.get("comparison") or {}
        error = row.get("error")
        error_text = (f"{error.get('type', 'Error')}: {error.get('message', '')}"
                      if isinstance(error, dict) else str(error)) if error else None
        summaries.append({"round": row.get("round"), "task": row.get("task"),
                          "contract_sha": row.get("contract_sha"), "seeds": row.get("seeds"),
                          "transfer_only": not row.get("contract_sha") or row["contract_sha"] != current_contract_sha,
                          "kind": tried.get("kind"), "node": tried.get("node"),
                          "intervention": {k: v for k, v in (tried.get("detail") or {}).items()
                                           if k in ("ref", "path", "from", "to", "module", "patch_sha", "artifact_sha", "graph_sha")},
                          "sample_count": row.get("sample_count"), "parent_id": row.get("parent_id"),
                          "reason": str(comparison.get("reason") or "")[:240],
                          "gains": len(comparison.get("gains") or []),
                          "regressions": len(comparison.get("regressions") or []),
                          "error": error_text[:160] if error_text else None,
                          "measurement_sha": row.get("measurement_sha"), "policy_id": row.get("policy_id")})
    return summaries


def compact_brief(proj: dict) -> dict:
    drivers = proj.get("drivers") or {}
    observations = []
    seeds = (proj.get("this_round") or {}).get("per_seed") or []
    motion = {(str(row.get("seed")), row.get("node")):
              {"coverage": row.get("coverage"),
               "channels": [{"channel": finding.get("channel"), "evidence": finding.get("evidence")}
                            for finding in row.get("findings") or []]}
              for row in (proj.get("diagnosis") or {}).get("traces") or []}
    for seed in seeds:
        observations.append({k: seed[k] for k in ("seed", "success", "first_death", "failure_mode")
                             if k in seed} | {
            "nodes": [{"node": node.get("id"), "ok": node.get("ok"), "steps": node.get("steps"),
                       "motion": motion.get((str(seed.get("seed")), node.get("id")))}
                      for node in seed.get("trail") or []
                      if node.get("ok") is not None or node.get("steps") is not None
                      or (str(seed.get("seed")), node.get("id")) in motion]})
        observations[-1]["unobserved_nodes"] = len(seed.get("trail") or []) - len(observations[-1]["nodes"])
    readings = [s.get("evaluation") or {} for s in seeds]
    counts = {key: (sum(r[key] for r in readings)
                    if readings and all(type(r.get(key)) is int for r in readings) else None)
              for key in ("passed", "observed", "total")}
    declared = proj.get("declared_development_seeds", proj.get("development_seeds"))
    if declared is None:
        bounds = proj.get("seeds") or []
        declared = (list(range(bounds[0], bounds[1] + 1))
                    if len(bounds) == 2 and all(type(s) is int for s in bounds)
                    else bounds or [row.get("seed") for row in observations])
    return {"task": proj.get("task"), "round": proj.get("round"),
            "policy_id": proj.get("policy_id"), "incumbent_policy_id": proj.get("incumbent_policy_id", proj.get("policy_id")),
            "experiment_id": proj.get("experiment_id"),
            "evaluation": {"contract_sha": (proj.get("evaluation_contract") or {}).get("sha"),
                           "objectives_ref": {"view": "evaluation", "policy_id": proj.get("policy_id")},
                           "condition_count": len((proj.get("evaluation_contract") or {}).get("obligations") or []),
                           "episodes": (proj.get("this_round") or {}).get("seeds_total"),
                           "successes": (proj.get("this_round") or {}).get("count"), **counts,
                           "progress": counts["passed"] / counts["total"] if counts["total"] else None},
            "driver_index": {node: {**{k: fd.get(k) for k in ("skill", "executor", "modules")},
                                    "bound_classes": [c["id"] for c in fd.get("bound_classes") or []],
                                    "tunables": (fd.get("tunables") or {}).get("values") or {},
                                    "executors": sorted(fd.get("executors") or {})}
                             for node, fd in sorted(drivers.items())},
            "source_index": {c["id"]: c for fd in drivers.values() for c in fd.get("bound_classes") or []},
            "development_seeds": declared, "probe_budget": proj.get("probe_budget"),
            "evaluation_budget": proj.get("evaluation_budget"),
            "development_cost": proj.get("development_cost"),
            "cycle_context": {key: copy.deepcopy(value) for key, value in (proj.get("cycle_context") or {}).items()
                              if key in ("previous_round", "cycle_outcome", "stop_reason", "reason", "memo", "continuous", "cycle",
                                         "cycles_without_update", "cycles_without_sample", "cycles_without_full_evaluation")},
            "workspace": proj.get("workspace"),
            "working_policies": proj.get("working_policies") or [],
            "history_index": {"local_rounds": len(proj.get("history") or []),
                              "other_task_records": len((proj.get("experience") or {}).get("retrieved") or [])},
            "historical_probe_replay": _replay_summary(proj.get("learning_replay"),
                                                       (proj.get("evaluation_contract") or {}).get("sha")),
            "observations": observations,
            "last_outcome": ({k: v for k, v in proj["last_outcome"].items()
                              if k in ("round", "outcome", "accepted", "accepted_reason", "before_score", "after_score")}
                             if isinstance(proj.get("last_outcome"), dict) else proj.get("last_outcome"))}


class EvidenceWorkingSet:
    """A byte-bounded LRU of unique tool pages, with the latest trial preferred.

    Complete requests and responses are recorded elsewhere. This only selects
    previously observed pages to re-present; it never invents evidence or actions.
    """
    def __init__(self, max_bytes=8000):
        self.max_bytes = max_bytes
        self.pages = OrderedDict()
        self.latest = self.trial = None

    @staticmethod
    def key(page):
        return tuple(page.get(k) for k in ("policy_id", "view", "node", "sha", "cursor", "next"))

    def page_limit(self, max_page_bytes, *, trial=False):
        reserve = len(encoded(self.pages[self.trial])) if not trial and self.trial in self.pages else 0
        return max(512, min(max_page_bytes, self.max_bytes - reserve - 128))

    def snapshot(self):
        return {"last_tool_result": self.pages.get(self.latest),
                "retained_evidence": [page for key, page in self.pages.items() if key != self.latest]}

    def retain_sources(self, project):
        """Retain literal code only while the owning policy and source still match.

        Dynamic observations/errors belong to their decision, not the next cycle.
        Source membership and content are rechecked; permission is not persisted.
        """
        kept = []
        for value in self.pages.values():
            data = value.get('data')
            if value.get('view') != 'source' or not value.get('code_read') or not isinstance(data, dict):
                continue
            try:
                proj = project(value.get('policy_id'))
                owners = source_owners(proj, data['module'])
                if not owners or owners != data.get('source_refs'):
                    continue
                driver = proj['drivers'][next(iter(owners))]
                source = _sources(driver).get(data['module'])
                if source is not None and content_id(source) == data.get('source_sha'):
                    kept.append(value)
            except (KeyError, ValueError, TypeError, ImportError, OSError):
                continue
        self.pages.clear()
        self.latest = self.trial = None
        for value in kept:
            self.add(value)

    def add(self, page, *, trial=False):
        # Reject before evicting useful evidence or changing the latest reward.
        if len(encoded({"last_tool_result": page, "retained_evidence": []})) > self.max_bytes:
            raise ValueError("latest tool page exceeds the evidence working-set budget")
        key = self.key(page)
        self.pages.pop(key, None)
        self.pages[key] = page
        self.latest = key
        if trial:
            self.trial = key
        while len(encoded(self.snapshot())) > self.max_bytes:
            victims = [k for k in self.pages if k not in (self.latest, self.trial)]
            if not victims:
                victims = [k for k in self.pages if k != self.latest]
            del self.pages[victims[0]]


def page(value, *, view: str, node=None, policy_id=None, cursor=0, limit=None,
         max_bytes=8000, metadata=None) -> dict:
    cursor = int(cursor or 0)
    if cursor < 0:
        raise ValueError("cursor must be nonnegative")
    cap = min(max_bytes, int(limit or max_bytes))
    if cap < 512:
        raise ValueError("page limit must be at least 512 bytes")
    base = {"view": view, "node": node, "policy_id": policy_id,
            "sha": content_id(value), "cursor": cursor, "next": None, "complete": True, **(metadata or {})}
    whole = {**base, "data": value}
    if cursor == 0 and len(encoded(whole)) <= cap:
        return whole
    text = encoded(value).decode("utf-8")
    if cursor >= len(text) and text:
        raise ValueError("cursor exceeds this evidence page; restart at cursor=0")
    end = min(len(text), cursor + cap)
    while end > cursor:
        result = {**base, "data": {"encoding": "json_fragment", "text": text[cursor:end]},
                  "next": end if end < len(text) else None, "complete": end == len(text)}
        if len(encoded(result)) <= cap:
            return result
        end = cursor + max(0, (end - cursor) * 3 // 4)
    raise ValueError("page metadata exceeds the tool byte budget")


def _code_page(value, *, node, policy_id, args, max_bytes):
    cursor = int(args.get("cursor") or 0)
    cap = min(max_bytes, int(args.get("limit") or max_bytes))
    code = value["code"]
    if cursor < 0 or cursor >= len(code):
        raise ValueError("source code cursor must identify a character in the selected symbol/range")
    if cap < 512:
        raise ValueError("page limit must be at least 512 bytes")
    end = min(len(code), cursor + cap)
    while end > cursor:
        result = {"view": "source", "node": node, "policy_id": policy_id, "sha": content_id(value),
                  "cursor": cursor, "next": end if end < len(code) else None,
                  "complete": end == len(code), "code_read": True,
                  "data": {**value, "code": code[cursor:end],
                           "page_start_line": value["start"] + code[:cursor].count("\n")}}
        if len(encoded(result)) <= cap:
            return result
        end = cursor + (end - cursor) * 3 // 4
    raise ValueError("source page metadata exceeds the tool byte budget")


def _sources(fd):
    modules = fd.get("modules") or []
    if not modules and fd.get("source_ref"):
        modules = [fd["source_ref"].partition(":")[0]]
    out = {}
    for module in modules:
        try:
            path = inspect.getsourcefile(importlib.import_module(module))
            if path:
                out[module] = Path(path).read_text()
        except (ImportError, OSError, TypeError):
            continue
    return out


def source_owners(proj, module, node=None):
    return {key: fd.get("source_ref") for key, fd in (proj.get("drivers") or {}).items()
            if (node is None or key == node) and module in (fd.get("modules") or
                ([fd["source_ref"].partition(":")[0]] if fd.get("source_ref") else []))}


def compact_command(command):
    if not isinstance(command, dict):
        return None
    def small(value, key=None):
        if key in ("files", "edits", "diff", "graph"):
            return {"sha": content_id(value), "omitted_from_reminder": True}
        if isinstance(value, dict):
            return {k: small(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [small(v) for v in value[:12]]
        return value[:300] if isinstance(value, str) else value
    result = {"op": command.get("op"), "args": small(command.get("args") or {})}
    if len(encoded(result)) > 2000:
        return {"op": command.get("op"), "args_sha": content_id(command.get("args")),
                "args_preview": encoded(result["args"]).decode("utf-8")[:400], "truncated": True}
    return result


def _parameter(fd, name):
    values = (fd.get("tunables") or {}).get("values") or {}
    if name is not None and (not isinstance(name, str) or not name):
        raise ValueError("parameter must be a nonempty code identifier or manifest name")
    if name is None:
        return {"parameter": None, "declared_tunable": bool(values),
                "available_tunables": sorted(values), "effective": values,
                "analysis": "Parameter directory only. Name a parameter to inspect its source consumers.",
                "references": [], "static_definitions": [], "complete_dependency_analysis": False}
    names = [name] if name else sorted(values)
    references, definitions = [], []
    bound_classes = fd.get("bound_classes") or []
    allowed_classes = {(row.get("module"), row.get("class"))
                       for row in bound_classes if isinstance(row, dict)}
    for module, source in _sources(fd).items():
        lines, tree = source.splitlines(keepends=True), ast.parse(source)
        # Lexical class scope resolves Class.attr / self.attr in the source. It
        # does not establish which value an instance has after execution.
        scopes = {}
        def scope(item, classes=()):
            scopes[item] = classes
            for child in ast.iter_child_nodes(item):
                scope(child, (*classes, item.name) if isinstance(item, ast.ClassDef) else classes)
        scope(tree)
        def allowed(item):
            classes = scopes[item]
            return not bound_classes or not classes or (module, ".".join(classes)) in allowed_classes
        for key in names:
            hits, aliases = set(), set()
            def matches(item):
                if not isinstance(item, (ast.Name, ast.Attribute)):
                    return False
                spelling = ast.unparse(item)
                if spelling == key:
                    return True
                if "." not in key:
                    return (item.id if isinstance(item, ast.Name) else item.attr) == key
                prefix = ".".join(scopes[item])
                if key.startswith(("self.", "cls.")) and key.count(".") == 1:
                    attribute = key.split(".", 1)[1]
                    if isinstance(item, ast.Name):
                        return bool(prefix) and item.id == attribute
                    return (isinstance(item.value, ast.Name) and item.value.id in ("self", "cls")
                            and item.attr == attribute)
                if isinstance(item, ast.Name):
                    return bool(prefix) and f"{prefix}.{item.id}" == key
                if isinstance(item.value, ast.Name) and item.value.id in ("self", "cls"):
                    return bool(prefix) and f"{prefix}.{item.attr}" == key
                return False

            for item in ast.walk(tree):
                if not allowed(item):
                    continue
                if matches(item) or isinstance(item, ast.Constant) and item.value == key:
                    hits.add(item.lineno)
                if isinstance(item, (ast.Assign, ast.AnnAssign)) and item.value is not None:
                    targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                    literal_lookup = any(isinstance(n, ast.Constant) and n.value == key
                                         for n in ast.walk(item.value))
                    for target in targets:
                        if not isinstance(target, (ast.Name, ast.Attribute)):
                            continue
                        if literal_lookup:
                            aliases.add(ast.unparse(target))
                        if not (literal_lookup or matches(target)):
                            continue
                        definition = {"parameter": key, "module": module, "source_sha": content_id(source),
                                      "target": ast.unparse(target), "start": item.lineno, "end": item.end_lineno,
                                      "expression": ast.get_source_segment(source, item.value),
                                      "code": "".join(lines[item.lineno - 1:item.end_lineno]), "literal_known": False}
                        try:
                            literal = ast.literal_eval(item.value)
                            # The evidence protocol is JSON. Unsupported static
                            # literals remain visible as source, never repr values.
                            json.dumps(literal, allow_nan=False, sort_keys=True)
                            definition.update(literal_value=literal, literal_known=True)
                        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                            pass
                        definitions.append(definition)
            for item in ast.walk(tree):
                if allowed(item) and isinstance(item, (ast.Name, ast.Attribute)) and ast.unparse(item) in aliases:
                    hits.add(item.lineno)
            windows = []
            for line in sorted(hits):
                start, end = max(1, line - 2), min(len(lines), line + 3)
                if windows and start <= windows[-1][1]:
                    windows[-1][1] = max(windows[-1][1], end)
                    windows[-1][2].append(line)
                else:
                    windows.append([start, end, [line]])
            for start, end, matched_lines in windows:
                references.append({"parameter": key, "module": module, "source_sha": content_id(source),
                                   "line": matched_lines[0], "lines": matched_lines,
                                   "start": start, "end": end,
                                   "code": "".join(lines[start - 1:end]), "aliases": sorted(aliases)})
    declared = name in values if name is not None else bool(values)
    return {"parameter": name, "declared_tunable": declared, "available_tunables": sorted(values),
            "effective": {key: values[key] for key in names} if name is None or declared else None,
            "references": references, "static_definitions": definitions,
            "analysis": ("Bound-class lexical scope plus module-level definitions/functions. " if bound_classes
                         else "Binding classes unavailable; module-wide static references. ") +
                        "Static names, attributes, literals and direct assignments; class scope is lexical, "
                        "not runtime resolution. Only manifest values are effective tunables; reading other "
                        "identifiers grants no tunables edit permission. Dynamic dependencies may be unresolved",
            "complete_dependency_analysis": False, "unresolved": not bool(references)}


def inspect_evidence(proj: dict, args: dict, max_bytes=8000) -> dict:
    view, node = args.get("view"), args.get("node")
    fd = (proj.get("drivers") or {}).get(node)
    if (view in ("node", "parameter", "trace") or view in ("source", "catalog") and node is not None) and fd is None:
        raise ValueError(f"inspect.args.node is required for view {view!r} and must identify an executed driver "
                         f"in driver_index: {sorted(proj.get('drivers') or {})}; a module is not a node")
    if view == "catalog":
        state = compact_brief(proj)
        value = {key: state[key] for key in ("driver_index", "source_index")}
        if node is not None:
            value['driver_index'] = {node: state['driver_index'][node]}
            value['source_index'] = {c['id']: c for c in fd.get('bound_classes') or []}
    elif view == "node":
        value = {"driver": fd, "diagnosis": [f for f in (proj.get("diagnosis") or {}).get("findings") or []
                                             if f.get("node") == node]}
    elif view == "parameter":
        value = _parameter(fd, args.get("parameter"))
    elif view == "source":
        module, symbol = args.get("module"), args.get("symbol")
        if symbol and ("start" in args or "end" in args):
            raise ValueError("source symbol uses cursor=next for character continuation; omit start/end. "
                             "For absolute module line ranges, omit symbol instead.")
        if isinstance(symbol, str) and ":" in symbol:
            qualified_module, symbol = symbol.split(":", 1)
            if module is not None and module != qualified_module:
                raise ValueError("symbol module must agree with args.module")
            module = qualified_module
        if node is None and not module:
            raise ValueError("inspect source: args.module from installed driver modules or args.node from driver_index is required")
        sources = _sources(fd) if fd else {m: src for driver in (proj.get("drivers") or {}).values()
                                           for m, src in _sources(driver).items()}
        if module is None:
            value = {"modules": {m: {"source_sha": content_id(s), "lines": len(s.splitlines()),
                                     "owner_nodes": list(source_owners(proj, m)),
                                     "source_refs": source_owners(proj, m),
                                     "symbols": [{"id": f"{m}:{r['symbol']}", **r} for r in _definitions(s)]}
                                 for m, s in sources.items()}, "code_read": False}
        else:
            if module not in sources:
                raise ValueError(f"module must be one of {sorted(sources)}; inspect catalog with the target node for exact source IDs")
            source = sources[module]
            lines = source.splitlines(keepends=True)
            definitions = _definitions(source)
            value = {"module": module, "source_sha": content_id(source),
                     "owner_nodes": list(source_owners(proj, module)), "source_refs": source_owners(proj, module)}
            if not symbol and "start" not in args and "end" not in args:
                value.update(symbols=[{"id": f"{module}:{r['symbol']}", **r} for r in definitions],
                             lines=len(lines), code_read=False)
            else:
                start, end = int(args.get("start") or 1), int(args.get("end") or len(lines))
                if symbol:
                    matches = [r for r in definitions if r["symbol"] == symbol]
                    if not matches and "." not in symbol:
                        matches = [r for r in definitions if r["symbol"].rsplit(".", 1)[-1] == symbol]
                    if len(matches) != 1:
                        raise ValueError("symbol must identify one definition; inspect catalog with the target node and copy an exact source_index method id, or use explicit start/end")
                    start, end = matches[0]["start"], matches[0]["end"]
                    symbol = f"{module}:{matches[0]['symbol']}"
                if start < 1 or start > end or end > len(lines):
                    raise ValueError("source start/end exceeds module")
                value.update(symbol=symbol, start=start, end=end,
                             code="".join(lines[start - 1:end]), code_read=True)
        if value.get("code_read"):
            return _code_page(value, node=node, policy_id=args.get("policy_id") or proj.get("policy_id"),
                              args=args, max_bytes=max_bytes)
        return page(value, view=view, node=node, policy_id=args.get("policy_id") or proj.get("policy_id"),
                    cursor=args.get("cursor"), limit=args.get("limit"), max_bytes=max_bytes,
                    metadata={"code_read": False})
    elif view == "trace":
        value = [{"seed": row.get("seed"), "nodes": [n for n in row.get("trail") or [] if n.get("id") == node]}
                 for row in (proj.get("this_round") or {}).get("per_seed") or []
                 if args.get("seed") is None or str(row.get("seed")) == str(args["seed"])]
    elif view == "plan":
        value = proj.get("plan_space")
    elif view == "history":
        value = {"history": proj.get("history"), "experience": proj.get("experience"),
                 "trial_evidence": proj.get("trial_evidence"), "learning_replay": proj.get("learning_replay"),
                 "working_policy": next((p for p in proj.get('working_policies', [])
                                         if p['policy_id'] == proj.get('policy_id')), None)}
    elif view == "evaluation":
        value = {"contract": proj.get("evaluation_contract"), "diagnosis": proj.get("diagnosis"),
                 "score_definition": proj.get("score_definition"),
                 "per_seed": [{k: row[k] for k in ("seed", "evaluation", "verification_observations",
                                                   "terminal_observation") if k in row}
                              for row in (proj.get("this_round") or {}).get("per_seed") or []
                              if args.get("seed") is None or str(row.get("seed")) == str(args["seed"])]}
    else:
        raise ValueError("view must be catalog/node/source/parameter/trace/plan/history/evaluation")
    return page(value, view=view, node=node, policy_id=args.get("policy_id") or proj.get("policy_id"),
                cursor=args.get("cursor"), limit=args.get("limit"), max_bytes=max_bytes)
