"""Diagnostic projections expose counts, not experiment or model payloads."""
import json

from board.store import rsi_command_summary


def test_legacy_command_summary_matches_row_identity_and_omits_raw_material(tmp_path):
    directory = tmp_path / 'campaigns' / 'evolve-test'
    (directory / 'llm').mkdir(parents=True)
    identity = {'prompt_sha': 'p', 'raw_sha': 'r'}
    (directory / 'campaign.json').write_text(json.dumps({'cursor': 1, 'rounds': [{'round': 1, 'llm': identity}]}))
    audit = {**identity, 'raw': 'RAW_PRIVATE_MATERIAL', 'events': [
        {'type': 'command', 'command': {'op': 'inspect', 'args': {'source': 'RAW_PRIVATE_MATERIAL'}}},
        {'view': 'error', 'result': {'error': {'type': 'ValueError', 'message': 'invalid module'},
                                   'previous_command': {'op': 'inspect'}}}],
        'requests': [{'messages': [{'role': 'user', 'content': json.dumps({'phase': 'selection', 'raw': 'RAW_PRIVATE_MATERIAL'})}]}]}
    path = directory / 'llm' / 'round-1.json'
    path.write_text(json.dumps(audit))
    summary = rsi_command_summary(tmp_path, 'test')
    assert summary['source'] == 'audit_summary' and summary['selection_calls'] == 1
    assert summary['requested'] == {'inspect': 1} and summary['errors'][0]['count'] == 1
    assert 'RAW_PRIVATE_MATERIAL' not in json.dumps(summary)
    audit['prompt_sha'] = 'mismatched'
    path.write_text(json.dumps(audit))
    assert rsi_command_summary(tmp_path, 'test') is None


def test_sealed_decision_flow_needs_no_audit_file(tmp_path):
    directory = tmp_path / 'campaigns' / 'evolve-test'
    directory.mkdir(parents=True)
    flow = {'requested': {'trial': 1}, 'executed': {'trial': 1}, 'errors': [], 'selection_calls': 0}
    (directory / 'campaign.json').write_text(json.dumps({'cursor': 1, 'rounds': [{'round': 1, 'llm': {'decision_flow': flow}}]}))
    assert rsi_command_summary(tmp_path, 'test') == {'round': 1, 'source': 'sealed_row', **flow}
