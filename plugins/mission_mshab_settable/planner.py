"""Card vocabulary for the VLM-planned mshab set_table chain.

The planner provider wraps planner_vlm BY REF (harness.registry -- cards never
import siblings) with the local sglang endpoint. Everything else is pure data:
the catalogue types the four drivable skills, SEGMENT_SPECS re-tasks each
segment to ``chain-<skill>.<target>`` (the ChainDriver's dispatch key) and
grounds the argument vocabulary, PLANNING_CONTEXT states scene FACTS without
giving the order away -- inferring "open the fridge before picking from it"
is exactly the planning the VLM is asked to do.
"""

from __future__ import annotations

from typing import Any

from harness.registry import load_provider

CATALOGUE: dict[str, dict[str, type]] = {
    "navigate": {"target": str},
    "open": {"articulation": str},
    "pick": {"object": str},
    "place": {"object": str},
}
ORACLES: tuple[str, ...] = ("segment_success",)

SKILL_DOCS: dict[str, dict[str, Any]] = {
    "navigate": {"description": "Drive the mobile base to the named target and "
                                "settle within manipulation range.",
                 "kind": "segment", "arguments": {"target": "str"},
                 "requires": [], "ensures": ["near({target})"], "clobbers": []},
    "open": {"description": "Open the named articulation with the arm. Needed "
                            "before reaching anything shut inside it.",
             "kind": "segment", "arguments": {"articulation": "str"},
             "requires": ["near({articulation})"],
             "ensures": ["open({articulation})"], "clobbers": []},
    "pick": {"description": "Grasp the named object and hold it. The object "
                            "must be reachable, not shut inside anything.",
             "kind": "segment", "arguments": {"object": "str"},
             "requires": ["near({object})"],
             "ensures": ["holding({object})"], "clobbers": []},
    "place": {"description": "Put the held object down at its goal location "
                             "(navigate there first).",
              "kind": "segment", "arguments": {"object": "str"},
              "requires": ["holding({object})", "near(dining_table)"],
              "ensures": ["at_goal({object})"], "clobbers": []},
}

PLANNING_CONTEXT: dict[str, Any] = {
    "benchmark": "mshab (ManiSkill-HAB set_table)",
    "scene": "ReplicaCAD apartment, official set_table task plan 0",
    "objects": ["013_apple"],
    # validate_plan grounding: exactly one pick then one place per object,
    # dependency-ordered. States nothing about the fridge -- inferring
    # open-before-pick stays the VLM's job.
    "required_per_object_order": ["pick", "place"],
    "articulations": ["fridge"],
    "navigate_targets": ["fridge", "013_apple", "dining_table"],
    "facts": [
        "the apple (013_apple) starts INSIDE the CLOSED fridge",
        "the apple's goal location is on the dining table",
        "the robot starts away from the fridge",
        "every manipulation needs a navigate to its target first",
    ],
    "unavailable_skills": ["close"],
    "notes": "One skill call per graph node; every node needs a verify entry "
             "with predicate segment_success.",
}

DEFAULT_INSTRUCTION = (
    "Fetch the apple from the fridge and set it on the dining table. Design "
    "the full skill graph yourself from the scene facts."
)

#: The ONE episode: the SequentialTask-v0 chain env. Horizon covers the
#: per-segment caps (3x500 navigate + 3x200 manipulation = 2100) with slack --
#: the kitchen_thaw c3 lesson: a horizon equal to the cap sum dies on the clock.
EPISODE: dict[str, Any] = {"task": "mshab_settable_chain", "horizon": 9000}  # 3 real-driving navs (~1200 steps each) + manipulation + dock overhead

#: Segment re-tasking: the ChainDriver parses ``chain-<skill>.<target>`` and
#: the allowed_args tables are the static grounding boundary (an object with
#: no binding fails before any actuation).
SEGMENT_SPECS: dict[str, dict[str, Any]] = {
    "navigate": {"task_template": "chain-navigate.{target}",
                 "allowed_args": {"target": ("fridge", "013_apple", "dining_table")}},
    "open": {"task_template": "chain-open.{articulation}",
             "allowed_args": {"articulation": ("fridge",)}},
    "pick": {"task_template": "chain-pick.{object}",
             "allowed_args": {"object": ("013_apple",)}},
    "place": {"task_template": "chain-place.{object}",
              "allowed_args": {"object": ("013_apple",)}},
}


# ---- the FULL official episode (sequential plan 0, all 16 subtasks): two
# ---- objects, two containers, close-after-delivery. Same scene, same env
# ---- authority; the graph the VLM must design roughly doubles.

CATALOGUE_FULL: dict[str, dict[str, type]] = {
    **CATALOGUE,
    "close": {"articulation": str},
}

SKILL_DOCS_FULL: dict[str, dict[str, Any]] = {
    **SKILL_DOCS,
    "close": {"description": "Close the named articulation with the arm. Only "
                             "sensible once everything needed from inside it "
                             "has been taken out and delivered.",
              "kind": "segment", "arguments": {"articulation": "str"},
              "requires": ["near({articulation})", "open({articulation})"],
              "ensures": ["closed({articulation})"],
              "clobbers": ["open({articulation})"]},
}

PLANNING_CONTEXT_FULL: dict[str, Any] = {
    "benchmark": "mshab (ManiSkill-HAB set_table, full episode)",
    "scene": "ReplicaCAD apartment, official set_table sequential plan 0",
    "objects": ["024_bowl", "013_apple"],
    "required_per_object_order": ["pick", "place"],
    "articulations": ["kitchen_counter", "fridge"],
    "navigate_targets": ["kitchen_counter", "024_bowl", "fridge",
                         "013_apple", "dining_table"],
    "facts": [
        "the bowl (024_bowl) starts INSIDE a CLOSED drawer of the kitchen_counter",
        "the apple (013_apple) starts INSIDE the CLOSED fridge",
        "both goal locations are on the dining table",
        "every manipulation needs a navigate to its target first",
        "each container must be CLOSED again once the item it held has been "
        "delivered to the table",
        # env grounding, not planning help: the baked official plan runs the
        # bowl arc first -- an apple-first graph would fail its first segment.
        "the environment enforces this global order: the bowl is delivered and "
        "its drawer closed BEFORE the fridge is touched for the apple",
    ],
    "unavailable_skills": [],
    "notes": "One skill call per graph node; every node needs a verify entry "
             "with predicate segment_success.",
}

DEFAULT_INSTRUCTION_FULL = (
    "Set the table: fetch the bowl from the kitchen drawer and the apple from "
    "the fridge, put both on the dining table, and leave both containers "
    "closed. Design the full skill graph yourself from the scene facts."
)

#: 16 subtasks: 8 navigates (two of them long real-driving legs) + 4
#: manipulations + 2 open + 2 close, plus dock-search overhead billed to the
#: env clock -- the kitchen_thaw c3 lesson says give real slack.
EPISODE_FULL: dict[str, Any] = {"task": "mshab_settable_full_chain",
                                "horizon": 24000}

SEGMENT_SPECS_FULL: dict[str, dict[str, Any]] = {
    "navigate": {"task_template": "chain-navigate.{target}",
                 "allowed_args": {"target": ("kitchen_counter", "024_bowl",
                                             "fridge", "013_apple",
                                             "dining_table")}},
    "open": {"task_template": "chain-open.{articulation}",
             "allowed_args": {"articulation": ("kitchen_counter", "fridge")}},
    "close": {"task_template": "chain-close.{articulation}",
              "allowed_args": {"articulation": ("kitchen_counter", "fridge")}},
    "pick": {"task_template": "chain-pick.{object}",
             "allowed_args": {"object": ("024_bowl", "013_apple")}},
    "place": {"task_template": "chain-place.{object}",
              "allowed_args": {"object": ("024_bowl", "013_apple")}},
}


class _SegmentStamp:
    """Stamp ``kind="segment"`` on every catalogue node the VLM emits.

    Every skill this card declares IS a persistent-episode segment, but the
    model is not asked to (and reliably does not) echo a ``kind`` field, and a
    kindless node defaults to manipulate -- straight into the robosuite
    SKILL_SPECS refusal. The card knows its own dispatch; the model does not
    have to."""

    def __init__(self, inner: Any):
        self._inner = inner

    def plan(self, brief):
        out = dict(self._inner.plan(brief))
        nodes = [dict(n) for n in out.get("nodes") or ()]
        for n in nodes:
            if n.get("skill") in CATALOGUE_FULL:   # superset of CATALOGUE
                n.setdefault("kind", "segment")
        out["nodes"] = nodes
        return out

    def __getattr__(self, name: str) -> Any:  # identity/available/deterministic
        return getattr(self._inner, name)


def provider(**params: Any):
    """planner_vlm over the LOCAL llama.cpp endpoint (model discovered from
    the server; QWEN38_API_KEY is a placeholder credential, the local server
    checks nothing). Explicit params win, so an A/B against another endpoint
    is a binding edit away."""
    merged: dict[str, Any] = {
        "endpoint_params": {"base_url": "http://127.0.0.1:30001/v1",
                            "api_key_env": "QWEN38_API_KEY", "model": None,
                            # IQ4 27B on the Vulkan llama.cpp build generates
                            # ~43 tok/s; the 60s default cuts a full plan off
                            # mid-JSON.
                            "timeout": 600.0},
        # Thinking is disabled server-side (--chat-template-kwargs); 4096
        # keeps headroom for the full six-node JSON even if a preamble leaks.
        "max_tokens": 4096,
    }
    merged.update(params)
    return _SegmentStamp(load_provider("plugins.planner_vlm:provider", merged))


def provider_full(**params: Any):
    """Same endpoint; a ~16-node graph JSON with verify entries runs well past
    the 6-node budget, so double max_tokens (measured: 6 nodes ~1.6k tokens)."""
    params.setdefault("max_tokens", 8192)
    return provider(**params)
