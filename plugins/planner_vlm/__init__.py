"""planner_vlm card: a VLM-backed ``harness.contracts.TaskPlanner`` (plan §1).

The empty VLM seat filled: the provider prompts a model with the SkillRecord
projection (``harness.protocol.vlm_projection``: skill cards, facts, objects,
done, fault, output schema) plus goal/oracles/scene/budget and parses a
strict-JSON graph ``{goal, nodes[], verify[], rationale}``; the returned dict
carries ``planner: {provider, endpoint, prompt_sha}`` for task.plan. Everything the model may NOT do is enforced
OUTSIDE this card, by ``plugins.task.validate.validate_plan`` -- this card never
pre-repairs a graph, never invents one silently: an unparseable reply (after
one re-ask carrying the parse error) returns a graph the validator is
guaranteed to refuse, so the failure rides the existing ``invalid_plan``
fold-back channel like every other planner refusal.

The model transport is the ONE model seam (§1b): the provider resolves
``plugins.model_endpoint:provider`` by registry ref string -- the same
sanctioned crossing SKILL_SPECS ``stages`` refs and mission PREDICATES refs use
(tests/test_boundaries.py forbids a sibling import; harness.registry is the
door). The model_endpoint card stays ``enabled = false``: it is consumed by
ref, not by kernel mount, so the folded base plan sha (sealed in
runs/round25-rerun) is untouched.

``deterministic = False`` is the loud opt-in to plugin_doctor's exemption
(shape validated, never double-run-diffed) -- an LLM planner cannot promise the
byte-identical replay a table planner does. What CAN be promised is
generate-once-then-frozen (§1 hardening item 4): the first graph emitted for a
(endpoint, task, seed, fault) key is frozen in a process-lifetime cache, so a
same-process replay -- a calibration re-run above all -- mounts the byte-same
graph. Cross-process freezing rides the existing plan-sha entry in the sealed
log; no second sealing mechanism here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from harness import opstream, protocol
from harness.registry import load_provider

#: Process-lifetime frozen-graph cache: canonical-JSON key -> canonical-JSON
#: graph. Module-level on purpose (a fresh kernel mount builds a fresh provider
#: per run; the freeze must survive that within one process). The key carries
#: the endpoint identity + task + seed + fault, so two runs of the same
#: (task, seed) at the same fold point replay one byte-identical graph, while a
#: replan (different fault) generates fresh.
#: ponytail: unbounded dict, process lifetime; add eviction if a resident
#: runtime ever plans enough distinct (task, seed) pairs for it to matter.
_FROZEN: dict[str, str] = {}

#: The stack_vlm binding's planner-facing vocabulary, card-authored like every
#: catalogue (the planner only selects and parameterizes). Deliberately ONLY
#: the skills the binding's mounted policy can drive: the binding's policy is
#: the stack scripted driver, and offering "pick" here let the live model emit
#: a pick node whose pickcan env the stack driver crashes on (KeyError
#: cubeB_pos, observed) -- a planner may not be offered vocabulary its channel
#: cannot execute. The deterministic stack binding shares the same effective
#: surface (its planner only ever emits "stack"), so the A/B stays fair.
CATALOGUE: dict[str, dict[str, type]] = {"stack": {"object": str, "target": str}}

#: The verify predicates a stack_vlm plan may name -- stack's terminal oracle.
ORACLES: tuple[str, ...] = ("stack_success",)

_RULES = """You are a robot task planner. Reply with ONE JSON object and nothing else \
(no prose, no code fences), exactly the shape of planning input's output_schema.
Hard rules -- a violation gets the whole plan rejected:
- Select skills ONLY from skills, passing EXACTLY the declared args with the declared \
types. Never invent a skill, an arg, or a predicate.
- Supported: every predicate in a node's requires must be in facts or in the ensures of \
a node it is (transitively) after, and not clobbered in between. Grounded: every object \
argument names an entry of objects (or one produced by an earlier node's ensures).
- Ground every object and target in planning_context. If planning_context declares a \
target_by_object mapping, use that exact target for each object.
- "after" lists only ids of nodes EARLIER in the list (the list is the execution order).
- nodes must be NON-EMPTY (an empty plan is always rejected): plan the minimal \
graph that achieves the goal with the listed skills.
- Every action node must be covered by at least one verify entry ({"after": its id, \
"predicate": <a declared oracle>}); verify must be non-empty.
- On a replan (fault and completed_nodes are given): keep every completed node in the \
plan with its id, skill and args EXACTLY as listed -- byte-identical -- and only \
re-plan the remaining work.
- If fault.kind is "no_progress": the graph you proposed last (its graph_sha is in \
fault) was ALREADY tried against this fault. Change the failed node's args or executor, \
or add a recovery node before it; repeating that graph ends the task.
- executor: when the input's arm is "auto", pick one per node from that skill's \
executors (a key), preferring the one with measured evidence (a non-null interval) and \
the higher lower bound; otherwise omit executor. Never name a key the skill lacks.
- rationale: one short paragraph on why the graph is legal (which facts / ensures \
support each requires) and sufficient.
- If taxonomy is present, a catalogue skill listed with stages or a decomposition is a \
COMPOSITE: select it whole; the server expands it from the skill graph. Never hand-expand \
it into stages, and never emit a stage name as a skill.
- If binding_availability is present, prefer skills marked bound:true whenever they can \
achieve the goal; a plan that uses an unbound skill is planning-only and will not run."""

_DECOMPOSE_RULES = """You are a robot mission decomposer. Reply with ONE JSON object and \
nothing else (no prose, no code fences), exactly the shape of the input's output_schema.
Hard rules -- a violation gets the whole decomposition rejected:
- Split the mission into an ORDERED list of tasks. Each entry names a task from \
known_tasks (its description says what it does) and lists the GROUNDED goal predicates \
that task must make true, using ONLY predicate names from predicates and ONLY object \
names from objects.
- Never invent a task, a predicate, or an object. Give every entry a unique id.
- rationale: one short paragraph on why these tasks in this order achieve the mission."""


class VlmPlanner:
    """Layer 1 ``harness.contracts.TaskPlanner`` over the model.endpoint seam."""

    #: The plugin_doctor exemption marker: shape validated, never diffed.
    deterministic = False

    def __init__(self, *, endpoint: str = "plugins.model_endpoint:provider",
                 endpoint_params: Mapping[str, Any] | None = None,
                 max_tokens: int = 2048) -> None:
        self._endpoint_ref = endpoint
        # Match the 3080 console's checked-in provider/model route. The endpoint
        # resolves DEEPSEEK_API_KEY from env first, then the console-owned DSH
        # credential store; the secret never enters this config or a plan hash.
        self._endpoint_params = dict(endpoint_params or {
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
            "model": "deepseek-v4-pro",
        })
        self._max_tokens = max_tokens
        self._ep = None            # the ModelEndpoint, resolved lazily by ref
        #: last successfully parsed plan, so a fault's ``nodes_done`` ids can be
        #: echoed back as full {id, skill, args} rows (in-run state, like
        #: clear_workspace's skip set).
        self._last_plan: Mapping | None = None

    # -- transport ----------------------------------------------------------
    def _endpoint(self):
        if self._ep is None:
            self._ep = load_provider(self._endpoint_ref, self._endpoint_params)
        return self._ep

    def available(self) -> bool:
        """The doctor/consumer probe: is the model endpoint answering at all?"""
        return self._endpoint().available()

    @property
    def identity(self) -> str:
        return f"planner_vlm({self._endpoint_ref})"

    # -- prompting ----------------------------------------------------------
    def _payload(self, brief: Mapping) -> dict:
        catalogue = brief.get("catalogue") or {}
        if not isinstance(catalogue, Mapping):  # the doctor's canned brief carries []
            catalogue = {}
        # Records: the card's protocol dicts when the brief carries them, else
        # lifted from the catalogue + the card's skill_docs (planner_docs shape).
        docs = brief.get("skill_docs") or {}
        records = brief.get("records") or {
            skill: {"id": skill, "name": skill, "kind": docs.get(skill, {}).get("kind", "segment"),
                    "args": {a: protocol.TYPES_BY_PY.get(t, "str") for a, t in schema.items()},
                    **{k: docs.get(skill, {}).get(k, ()) for k in ("requires", "ensures", "clobbers")},
                    "description": docs.get(skill, {}).get("description", "")}
            for skill, schema in catalogue.items()}
        fault = brief.get("fault")
        done_ids = tuple((fault or {}).get("nodes_done", ()))
        completed = [n for n in (self._last_plan or {}).get("nodes", ())
                     if n["id"] in done_ids]
        proj = protocol.vlm_projection(
            records, brief.get("facts") or (), brief.get("objects") or (),
            done_ids, fault, show_evidence=bool(brief.get("show_evidence")))
        payload = {
            **proj,
            "goal": (brief.get("instruction")
                     or brief.get("default_instruction")
                     or brief.get("task")),
            "planning_context": brief.get("planning_context") or {},
            "oracles": list(brief.get("oracles") or ()),
            "scene": brief.get("scene") or {},
            "budget": brief.get("budget"),
            "arm": brief.get("arm", "scripted"),
            "completed_nodes": [{"id": n["id"], "skill": n["skill"],
                                 "args": dict(n["args"])} for n in completed],
        }
        # Skill-graph planning (plugins/task/skill_planning) threads two more
        # server-authored context blocks; they are ABSENT from every other task
        # brief, so those payloads stay byte-identical.
        for key in ("taxonomy", "binding_availability"):
            if brief.get(key):
                payload[key] = brief[key]
        return payload

    @staticmethod
    def _parse(text: str, key: str = "nodes") -> Mapping:
        """Strict-JSON extraction: the reply as-is, or the outermost {...} slice
        (tolerates fences / a thinking preamble, repairs nothing inside).
        ``key`` is the one structural key the reply must carry."""
        try:
            plan = json.loads(text)
        except ValueError:
            # The FIRST balanced JSON object in the reply (raw_decode ignores
            # trailing prose / a second object -- observed live from Qwen).
            start = text.find("{")
            if start < 0:
                raise ValueError("reply contains no JSON object")
            plan, _ = json.JSONDecoder().raw_decode(text[start:])
        if not isinstance(plan, Mapping):
            raise ValueError(f"reply is JSON but not an object: {type(plan).__name__}")
        if key not in plan:
            # Structural, not semantic: a JSON object that is not even
            # plan-shaped (observed live: the model echoing the input payload
            # back) earns the re-ask; everything plan-shaped goes to the
            # validator untouched.
            raise ValueError(f"reply is a JSON object but not a plan (no {key!r} key)")
        return plan

    # -- mission decomposition ---------------------------------------------
    def decompose(self, brief: Mapping) -> dict:
        """Mission -> ``{tasks: [{id, task?, goal}], rationale, prompt_sha}`` over
        ``protocol.mission_projection(known_tasks, predicates, objects)``.
        Refuses (ValueError, the sealed refusal) a reply that is not JSON, names
        an unknown task, or grounds a goal in a predicate outside the catalogue;
        the model never invents vocabulary."""
        proj = protocol.mission_projection(
            brief["known_tasks"], brief["predicates"], brief.get("objects") or ())
        messages = [{"role": "system", "content": _DECOMPOSE_RULES},
                    {"role": "user", "content":
                     f"Mission: {brief['mission']}\n\nDecomposition input:\n"
                     + json.dumps(proj, sort_keys=True)
                     + "\n\nOutput ONLY the decomposition JSON object now."}]
        prompt_sha = protocol.content_id(messages)
        reply = self._endpoint().chat(messages, temperature=0.0,
                                      max_tokens=self._max_tokens,
                                      response_format={"type": "json_object"})
        out = self._parse(reply, key="tasks")
        tasks = out["tasks"]
        known = {t["task"] for t in proj["known_tasks"]}
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("decomposition.tasks must be a non-empty list")
        ids: set[str] = set()
        for i, t in enumerate(tasks):
            if not isinstance(t, Mapping) or not isinstance(t.get("id"), str) or t["id"] in ids:
                raise ValueError(f"decomposition.tasks[{i}] needs a unique string id")
            ids.add(t["id"])
            if t.get("task") is not None and t["task"] not in known:
                raise ValueError(f"decomposition.tasks[{i}] names unknown task "
                                 f"{t['task']!r}; known: {sorted(known)}")
            goal = t.get("goal")
            if not isinstance(goal, list) or not goal:
                raise ValueError(f"decomposition.tasks[{i}].goal must be a non-empty list")
            for g in goal:
                name = protocol.parse_pred_ref(g)[0]
                if name not in proj["predicates"]:
                    raise ValueError(f"decomposition.tasks[{i}] goal {g!r} names unknown "
                                     f"predicate {name!r}; catalogue: {sorted(proj['predicates'])}")
        return {"tasks": [{"id": t["id"], "task": t.get("task"),
                           "goal": [protocol.pred_ref_str(g) for g in t["goal"]]}
                          for t in tasks],
                "rationale": str(out.get("rationale", "")), "prompt_sha": prompt_sha}

    def _generate(self, brief: Mapping) -> tuple[Mapping, str]:
        messages = [{"role": "system", "content": _RULES},
                    {"role": "user", "content":
                     "Planning input:\n"
                     + json.dumps(self._payload(brief), sort_keys=True)
                     + "\n\nOutput ONLY the plan JSON object for this input now."}]
        prompt_sha = protocol.content_id(messages)
        json_mode = {"type": "json_object"}
        reply = self._endpoint().chat(messages, temperature=0.0,
                                      max_tokens=self._max_tokens,
                                      response_format=json_mode)
        try:
            return self._parse(reply), prompt_sha
        except ValueError as first:
            # ONE re-ask, carrying the parse error verbatim.
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content":
                          f"Your reply failed strict JSON parsing: {first}. "
                          "Reply again with ONLY the JSON object."}]
            retry = self._endpoint().chat(messages, temperature=0.0,
                                          max_tokens=self._max_tokens,
                                          response_format=json_mode)
            try:
                return self._parse(retry), prompt_sha
            except ValueError as second:
                # NEVER silently invent a graph: return one validate_plan is
                # guaranteed to refuse (empty nodes), so the failure folds back
                # through the loop's invalid_plan channel. The parse error rides
                # the goal string into the plan_built operational event.
                opstream.emit("planner_vlm_unparseable", error=str(second))
                return {"goal": f"planner_vlm: model reply unparseable "
                                f"after one retry ({second})",
                        "nodes": [], "verify": []}, prompt_sha

    # -- the seam -----------------------------------------------------------
    def plan(self, brief: Mapping) -> Mapping:
        key = json.dumps([self._endpoint_ref,
                          dict(sorted(self._endpoint_params.items())),
                          brief.get("task"), brief.get("seed"),
                          brief.get("instruction"),
                          brief.get("default_instruction"),
                          brief.get("planning_context"),
                          brief.get("fault"),
                          # skill-graph briefs: a changed catalogue (retrieval or
                          # graph file) is a fresh generation, same instruction
                          brief.get("catalogue_digest")], sort_keys=True, default=str)
        frozen = _FROZEN.get(key)
        if frozen is None:
            plan, prompt_sha = self._generate(brief)
            # provenance the workload seals in task.plan: WHICH endpoint saw
            # WHICH prompt bytes (the prompt is a pure function of the brief).
            plan = {**plan, "planner": {"provider": self._endpoint_ref,
                                        "endpoint": self._endpoint().identity,
                                        "prompt_sha": prompt_sha}}
            # canonical byte form (the planner_stack round-trip stance): only
            # pure JSON types leave this seam, byte-stable on replay.
            frozen = _FROZEN[key] = json.dumps(plan, sort_keys=True)
        out = json.loads(frozen)
        if out.get("nodes"):
            self._last_plan = out
        return out


def provider(**params: Any) -> VlmPlanner:
    return VlmPlanner(**params)
