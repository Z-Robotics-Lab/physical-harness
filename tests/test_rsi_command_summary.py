"""The command summary face: one round's agent activity as counts, never its prompts."""
import json

from board.store import rsi_command_summary


def test_summary_reads_the_sealed_llm_block_and_nothing_else(tmp_path):
    directory = tmp_path / "campaigns" / "evolve-test"
    directory.mkdir(parents=True)
    llm = {"status": "finished", "calls": 5, "actions": {"read": 2, "edit": 1, "run": 1, "finish": 1},
           "errors": ["error: `old` must occur exactly once"], "summary": "PRIVATE"}
    (directory / "campaign.json").write_text(json.dumps({"cursor": 2, "rounds": [
        {"round": 1, "llm": llm}, {"round": 2, "llm": None}]}))
    assert rsi_command_summary(tmp_path, "test", 1) == {
        "round": 1, "status": "finished", "calls": 5, "actions": llm["actions"], "errors": llm["errors"]}
    assert rsi_command_summary(tmp_path, "test") is None      # round 2 (cursor) sealed no llm block
    assert rsi_command_summary(tmp_path, "absent") is None
