"""Code discovery and reuse must leave a feasible path from reading to sampling."""
import copy
import json

import pytest
from test_evolve_agent_tools import _Endpoint, _projection, _tools
from test_evolve_observation_loop import STOP

from scripts import evolve_evidence, evolve_llm
from scripts.evolve_evidence import EvidenceWorkingSet, bound_class_index, inspect_evidence


def test_node_catalog_exposes_only_its_bound_classes_with_readable_method_ids():
    from test_evolve_e2e import _Driver, _AltExecutor
    proj, _ = _projection()
    driver = proj['drivers']['grab-0']
    driver['bound_classes'] = bound_class_index([_Driver])
    proj['drivers']['other'] = {**copy.deepcopy(driver), 'bound_classes': bound_class_index([_AltExecutor])}
    page = inspect_evidence(proj, {'view': 'catalog', 'node': 'grab-0'})
    assert page['complete'] and set(page['data']['driver_index']) == {'grab-0'}
    assert set(page['data']['source_index']) == {'test_evolve_e2e:_Driver'}
    method = page['data']['source_index']['test_evolve_e2e:_Driver']['methods'][0]['id']
    source = inspect_evidence(proj, {'view': 'source', 'symbol': method, 'node': 'grab-0'})
    assert source['code_read'] and source['data']['code']


def test_invalid_source_request_does_not_consume_successful_read_allowance(tmp_path):
    proj, before = _projection()
    tools, _ = _tools(proj)
    endpoint = _Endpoint([
        {'op': 'inspect', 'args': {'view': 'source', 'node': 'grab-0', 'module': 'invented.module'}},
        {'op': 'inspect', 'args': {'view': 'source', 'symbol': 'test_evolve_e2e:_Driver.act'}}, STOP])
    _, row = evolve_llm.llm_propose(endpoint, proj, before, 1, tmp_path, agent_tools=tools,
                                   budget={'max_read_calls': 1})
    bodies = [json.loads(messages[1]['content']) for messages, _ in endpoint.requests]
    assert [b['read_calls_left'] for b in bodies] == [1, 1, 0]
    assert row['evidence_reads'] == 1 and row['calls'] == 3
    assert row['decision_flow']['requested'] == {'inspect': 2, 'stop': 1}
    assert row['decision_flow']['executed'] == {'inspect': 1, 'stop': 1}
    assert row['decision_flow']['errors'][0]['op'] == 'inspect'


def test_scheduler_carries_literal_source_into_next_cycle_without_another_read(tmp_path, monkeypatch):
    from test_evolve_agent_runtime import _run
    from test_evolve_cycles import CycleModel, stop
    model = CycleModel([
        {'op': 'inspect', 'args': {'view': 'source', 'node': 'grab-0',
                                 'symbol': 'test_evolve_agent_runtime:_ConjunctionDriver.act'}},
        stop(), stop()])
    _, doc = _run(tmp_path, monkeypatch, model, continuous=True, rounds=2, max_calls=2)
    assert [r['llm']['evidence_reads'] for r in doc['rounds']] == [1, 0]
    body = json.loads(model.requests[2][1]['content'])
    pages = [body.get('last_tool_result'), *body.get('retained_evidence', [])]
    assert any(p and p.get('code_read') and p['data'].get('code') for p in pages)
    assert doc['rounds'][1]['usage']['episode_attempts'] == 0


@pytest.mark.parametrize('changed', [False, True])
def test_next_cycle_reuses_visible_code_but_never_stale_source_permission(tmp_path, monkeypatch, changed):
    proj, before = _projection()
    tools, calls = _tools(proj)
    evidence = EvidenceWorkingSet()
    first = _Endpoint([{'op': 'inspect', 'args': {'view': 'source', 'symbol': 'test_evolve_e2e:_Driver.act'}}, STOP])
    evolve_llm.llm_propose(first, proj, before, 1, tmp_path, agent_tools=tools, evidence=evidence)
    if changed:
        sources = evolve_evidence._sources
        monkeypatch.setattr(evolve_evidence, '_sources', lambda fd: {m: s + '\n# changed' for m, s in sources(fd).items()})
    # Isolate the inspection gate; compiler/executor contracts have their own tests.
    monkeypatch.setattr(evolve_llm, '_try', lambda *a, **k: {'kind': 'executor', 'node': 'grab-0', 'detail': {'to': 'alt'}})
    second = _Endpoint([{'op': 'trial', 'args': {'kind': 'patch', 'payload': {
        'node': 'grab-0', 'module': 'test_evolve_e2e', 'edits': [{'old': 'old', 'new': 'new'}]}}}, STOP])
    _, row = evolve_llm.llm_propose(second, proj, before, 2, tmp_path, agent_tools=tools, evidence=evidence)
    body = json.loads(second.requests[0][0][1]['content'])
    pages = [body.get('last_tool_result'), *body.get('retained_evidence', [])]
    assert any(p and p.get('code_read') for p in pages) is not changed
    assert row['evidence_reads'] == 0
    assert len(calls) == (0 if changed else 1)
    if changed:
        assert 'requires_inspection' in row['decision_flow']['errors'][0]['message']
