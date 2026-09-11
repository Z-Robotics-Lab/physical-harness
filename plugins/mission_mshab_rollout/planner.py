"""A deterministic one-node planner for the mshab rollout missions.

Table lookup like skill_toy's ToyPlanner, keyed on the four mshab_* task names:
one `rollout` SEGMENT node driving the persistent episode, one verify edge
naming its own terminal (`segment_success` -- the loop reads the node's sealed
boolean, no second oracle dialect). CATALOGUE/ORACLES/EPISODE_*/SEGMENT_SPECS
are the card-authored vocabulary the manifest points the runtime at by ref.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

#: The one skill these missions decompose to. No args: the task name on the
#: brief already picks the subtask env, the seed picks the spawn.
CATALOGUE: dict[str, dict[str, type]] = {"rollout": {}}
ORACLES: tuple[str, ...] = ("segment_success",)

#: No per-sub-goal spec override: the ONE episode spec drives the one segment.
#: (episodic bindings must name a segment_specs ref; empty is the honest value.)
SEGMENT_SPECS: dict[str, dict] = {}

#: Rollout length in env steps: the episode horizon IS the segment step budget.
_HORIZON = 400

TASKS = ("mshab_pick", "mshab_place", "mshab_open", "mshab_close")

EPISODE_PICK: dict[str, Any] = {"task": "mshab_pick", "horizon": _HORIZON}
EPISODE_PLACE: dict[str, Any] = {"task": "mshab_place", "horizon": _HORIZON}
EPISODE_OPEN: dict[str, Any] = {"task": "mshab_open", "horizon": _HORIZON}
EPISODE_CLOSE: dict[str, Any] = {"task": "mshab_close", "horizon": _HORIZON}


class MshabRolloutPlanner:
    def plan(self, brief: Mapping) -> Mapping:
        task = brief.get("task")
        if task not in TASKS:
            raise ValueError(
                f"MshabRolloutPlanner only plans {TASKS}, got {task!r}")
        # Round-trip through sorted JSON, same as ToyPlanner: the emitted
        # mapping is exactly its canonical byte form.
        return json.loads(json.dumps({
            "goal": f"mshab rollout: {task}",
            "nodes": [{"id": "roll-0", "kind": "segment", "skill": "rollout",
                       "args": {}, "after": []}],
            "verify": [{"after": "roll-0", "predicate": "segment_success"}],
        }, sort_keys=True))

    @property
    def identity(self) -> str:
        return "mshab_rollout_planner@v1"


def provider(**params: Any) -> MshabRolloutPlanner:
    return MshabRolloutPlanner(**params)
