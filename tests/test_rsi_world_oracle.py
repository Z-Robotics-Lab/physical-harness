"""The workload marks a verify that read a controller's self-report (``blocked_reads``);
its execution verdict is unchanged, its world-only reading stays None."""
import importlib
from types import SimpleNamespace

import pytest

from plugins.task import workload


@pytest.mark.parametrize('module_name,item,facts_key', [
    ('plugins.mission_recycle_cans.planner', 'can1', 'cans'),
    ('plugins.mission_pack_lunch.planner', 'hot1', 'objects'),
])
def test_actual_grasp_predicate_keeps_execution_semantics_without_rewarding_self_report(
        monkeypatch, module_name, item, facts_key):
    module = importlib.import_module(module_name)
    calls = []

    def contact(env):
        calls.append(env)
        return True

    monkeypatch.setattr(module, 'load_provider', lambda *args, **kwargs: contact)
    monkeypatch.setattr(module, '_obj_z', lambda *args: 1.0)  # no world motion
    ref = f'{module_name}:v_grasped_{item}'
    node = {'id': f'grasped-{item}', 'kind': 'verify', 'skill': f'v_grasped_{item}', 'args': {}}
    for controller_claim in (False, True):
        ctx = workload.NodeCtx(
            seed=1, env_ref='test:world', policy_ref='test:controller', skills=(),
            episode=SimpleNamespace(env=SimpleNamespace(), obs={}),
            nodes_out={'survey': {'success': True, 'facts': {facts_key: {item: [0, 0, 1.0]}}},
                       f'grasp-{item}': {'success': controller_claim}},
            predicates={node['skill']: ref},
            _provenance={'survey': {'kind': 'perceive', 'clean': True, 'blocked_reads': []},
                         f'grasp-{item}': {'kind': 'segment', 'clean': False, 'blocked_reads': []}})
        result = workload._verify(node, ctx)
        assert result['success'] is controller_claim  # the installed execution contract is unchanged
        assert result['verification_success'] is None
        assert any(f'grasp-{item}' in path for path in result['blocked_reads'])
    assert len(calls) == 2  # one predicate call for each world, never a second scoring run


def test_actual_world_only_predicate_remains_measurable(monkeypatch):
    module = importlib.import_module('plugins.mission_recycle_cans.planner')
    calls = []

    def near(env):
        calls.append(env)
        return True

    monkeypatch.setattr(module, 'load_provider', lambda *args, **kwargs: near)
    node = {'id': 'at-can1', 'kind': 'verify', 'skill': 'v_at_can1', 'args': {}}
    ctx = workload.NodeCtx(seed=1, env_ref='test:world', policy_ref='test:controller', skills=(),
                           episode=SimpleNamespace(env=SimpleNamespace()), nodes_out={},
                           predicates={node['skill']: 'plugins.mission_recycle_cans.planner:v_at_can1'})
    result = workload._verify(node, ctx)
    assert result['success'] is True and result['verification_success'] is True
    assert result['blocked_reads'] == [] and len(calls) == 1
