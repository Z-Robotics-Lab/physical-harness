"""skill_graph_robocasa card: planner vocabulary over the unified RoboCasa skill graph.

The graph (``harness.unified_skill_graph``) knows WHICH skills exist, how they
classify (IS_A), stage (HAS_STAGE/REALIZES) and decompose (DECOMPOSES_TO). It
does not know what ARGUMENTS a planner may pass or which verify predicate a
graph-only plan may name -- those are card-authored here, the same way every
other planner catalogue is authored on the skill side and never by the planner.

Nothing in this module is an execution binding. ``plugins/task/skill_planning``
resolves bindings from task manifests; a skill named here and nowhere else is
planning-only, and the response says so.
"""

from __future__ import annotations

#: The channel id the planning pipeline routes graph-only planning through.
CHANNEL = "robocasa_skill_graph"

#: Argument schema per CANONICAL graph skill ({arg: required type}). Observed
#: skills (CoffeeSetupMug, OpenFridge, ...) take no arguments: their annotation
#: instructions already bind the object and fixture ("pick up the mug from the
#: cabinet"), so a planner selects them whole. A canonical skill absent from this
#: table takes ``{"object": str}``.
CANONICAL_ARGS: dict[str, dict[str, type]] = {
    "Pick": {"object": str},
    "Place": {"object": str, "target": str},
    "Transport": {"object": str, "target": str},
    "TransferObject": {"object": str, "target": str},
    "Open": {"target": str},
    "Close": {"target": str},
    "Slide": {"target": str},
    "PressButton": {"target": str},
    "TurnControl": {"target": str},
    "Wait": {"target": str},
    "Scrub": {"object": str},
    "Stir": {"object": str},
    "Tilt": {"object": str},
    "NavigateToObject": {"object": str},
    "NavigateToFixture": {"target": str},
    "OtherManipulation": {"object": str},
}
DEFAULT_CANONICAL_ARGS: dict[str, type] = {"object": str}

#: The ONE verify predicate a graph-only plan may name. It is SYMBOLIC: "the
#: annotation stage for this node is complete". No executable oracle evaluates
#: it -- which is exactly why a plan over this vocabulary cannot be executable.
ORACLES: tuple[str, ...] = ("annotation_complete",)

#: Display aliases: static skill-library name -> canonical graph interface, so a
#: task-channel plan (pack_all_robocasa's ``pick``) can show its taxonomy path.
#: A display alias is NOT a binding in the other direction: graph ``Pick`` is not
#: executable because library ``pick`` is bound inside one task's scene.
LIBRARY_TO_CANONICAL: dict[str, str] = {
    # Both survive the synonym fold: `grasp` is the folded family name the
    # robocasa missions bind, `pick` is still bound by the robosuite tasks.
    "grasp": "Pick",
    "pick": "Pick",
    "place": "Place",
    "place_on": "Place",
    "carry": "Transport",
    "navigate": "NavigateToObject",
}


def catalogue_for(graph, names) -> dict[str, dict[str, type]]:
    """The strict-JSON catalogue for a retrieved skill set: observed skills take
    no args, canonical skills take their authored schema."""
    out: dict[str, dict[str, type]] = {}
    for name in names:
        node = graph.node(name)
        if node.kind == "canonical_skill":
            out[name] = dict(CANONICAL_ARGS.get(name, DEFAULT_CANONICAL_ARGS))
        else:
            out[name] = {}
    return out
