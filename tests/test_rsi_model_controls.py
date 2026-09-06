"""Operator model/effort choices reach the real request and sealed runtime row."""
import json

import pytest

from board import store
from plugins.model_endpoint import OpenAICompatEndpoint, reasoning_options
from scripts import evolve_llm
from test_evolve_agent_tools import _Endpoint, _projection, _tools
from test_evolve_observation_loop import STOP


@pytest.mark.parametrize('effort', ['off', 'low', 'high', 'max'])
def test_loop_dispatches_selected_model_and_effort_with_original_output_cap(tmp_path, monkeypatch, effort):
    endpoint = _Endpoint([STOP])
    selected = []
    def factory(model=None):
        selected.append(model)
        return endpoint
    monkeypatch.setattr(evolve_llm, 'endpoint', factory)
    projection, before = _projection()
    tools, _ = _tools(projection)
    _, row = evolve_llm.llm_propose(None, projection, before, 1, tmp_path,
        agent_tools=tools, model='operator-selected-model', effort=effort, max_tokens=137)
    assert selected == ['operator-selected-model']
    assert row['requested_model'] == 'operator-selected-model' and row['effort'] == effort
    _, options = endpoint.requests[0]
    assert options['max_tokens'] == 137
    assert options['thinking']['type'] == ('disabled' if effort == 'off' else 'enabled')
    assert options.get('reasoning_effort') == (None if effort == 'off' else effort)
    audit = json.loads((tmp_path / 'round-1.json').read_text())
    assert audit['requests'][0]['options'] == options
    assert audit['effort'] == effort and audit['requested_model'] == 'operator-selected-model'


@pytest.mark.parametrize('model,effort', [('', 'off'), (' x', 'off'), (['model'], 'off'),
                                        ('x\ny', 'off'), ('x', 'invented'), ('x', None)])
def test_invalid_selection_is_rejected_before_a_request(model, effort):
    with pytest.raises(ValueError):
        evolve_llm.model_request_config(model, effort)


def test_effort_declaration_cannot_override_model_messages_or_budget():
    with pytest.raises(ValueError, match='only configure'):
        reasoning_options('high', {'high': {'model': 'other', 'max_tokens': 999999}})


def test_model_catalog_reads_only_ids_and_retains_default_if_discovery_fails(monkeypatch):
    calls = []
    def get_json(self, url, timeout):
        calls.append((url, timeout))
        return {'data': [{'id': 'model-b', 'secret': 'DO_NOT_EXPOSE'}, {'id': 'model-a'}]}
    monkeypatch.setattr(OpenAICompatEndpoint, '_get_json', get_json)
    result = store.rsi_model_options()
    assert result['default_effort'] == 'off'
    assert result['efforts'] == ['off', 'low', 'high', 'max']
    assert {r['id'] for r in result['models']} == {result['default_model'], 'model-a', 'model-b'}
    assert 'DO_NOT_EXPOSE' not in json.dumps(result) and calls[0][0].endswith('/models')
    assert calls[0][1] == 3.0
    def fail(*args):
        raise OSError('DO_NOT_EXPOSE credential details')
    monkeypatch.setattr(OpenAICompatEndpoint, '_get_json', fail)
    failed = store.rsi_model_options()
    assert failed['models'] == [{'id': failed['default_model']}]
    assert failed['error'] == 'Model discovery failed (OSError)'


def test_explicit_selection_survives_runtime_subprocess_and_next_brief_defaults(tmp_path):
    from test_evolve_e2e import _Runtime, _CARD, TASK, LLM_NONE, _doc
    rt = _Runtime(tmp_path, card=_CARD, canned=LLM_NONE, mode='evolution')
    rt.campaign = rt.session / 'campaigns' / f'evolve-{TASK}' / 'campaign.json'
    try:
        rt.run({'kind': 'evolve', 'task': TASK, 'rounds': 1, 'seeds': [1, 2],
                'llm_model': 'operator-selected-model', 'llm_effort': 'high'})
        doc = _doc(rt)
        assert doc['llm_config'] == {'model': 'operator-selected-model', 'effort': 'high'}
        row = store.rsi_run(rt.session, TASK, 1)['rounds'][0]
        assert row['llm']['requested_model'] == 'operator-selected-model'
        assert row['llm']['effort'] == 'high'
        assert store.rsi_campaigns(rt.session)[0]['llm_config'] == doc['llm_config']
        rt.run({'kind': 'evolve', 'task': TASK, 'rounds': 1, 'seeds': [1, 2]})
        assert _doc(rt)['llm_config']['effort'] == 'off'
        assert _doc(rt)['llm_config']['model'] != 'operator-selected-model'
        assert store.rsi_run(rt.session, TASK, 1)['rounds'][0]['llm']['effort'] == 'high'
    finally:
        rt.stop()
