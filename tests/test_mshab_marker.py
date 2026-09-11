"""mshab marker self-proof.

Runs only in the mshab venv (`pytest -m mshab`, the ManiSkill-HAB checkout's
own .venv -- see plugins/embodiment_mshab/env.py docstring). In the harness
.venv mani_skill is unimportable, so the conftest hook auto-skips this -- the
extra base-lane skip captured in docs/project-documentation.md §3.
"""
import pytest


@pytest.mark.mshab
def test_mshab_subtask_envs_register():
    import gymnasium as gym

    import mshab.envs  # noqa: F401  registers [Name]SubtaskTrain-v0

    ids = set(gym.registry)
    assert {"PickSubtaskTrain-v0", "PlaceSubtaskTrain-v0",
            "OpenSubtaskTrain-v0", "CloseSubtaskTrain-v0"} <= ids


@pytest.mark.mshab
def test_mshab_card_task_table():
    from harness.spec import EpisodeSpec

    import plugins.embodiment_mshab as card

    p = card.provider()
    assert p.tasks() == ("mshab_close", "mshab_open", "mshab_pick", "mshab_place")
    with pytest.raises(KeyError):
        p.object_key(EpisodeSpec(task="mshab_pick", seed=424242))
