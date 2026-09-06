"""The real plan proposal channel changes actions while its objective stays fixed."""
import json

import pytest
from test_evolve_e2e import _CARD, TASK, _Planner
from test_mission_e2e import _Runtime


@pytest.mark.parametrize('skill,executor,accepted', [('grab', 'alt', True), ('reach', None, False)])
def test_inserted_action_is_executed_but_earns_only_world_verified_progress(
        tmp_path, skill, executor, accepted):
    graph = _Planner().plan({})
    inserted = {'id': 'prepare', 'skill': skill, 'kind': 'segment', 'args': {}, 'after': ['reach-0']}
    if executor:
        inserted['executor'] = executor
    graph['nodes'].insert(1, inserted)
    graph['nodes'][2]['after'] = ['prepare']
    graph['verify'].append({'after': 'prepare', 'predicate': 'seg_ok'})
    answer = {'kind': 'plan', 'payload': {'graph': graph},
              'summary': 'Insert a supported preparation action before the failed segment.',
              'rationale': 'Test the extra action against the same independent world objective.'}
    rt = _Runtime(tmp_path, card=_CARD, canned=[answer], mode='evolution')
    try:
        rt.run({'kind': 'evolve', 'task': TASK, 'seeds': [1, 2], 'rounds': 1,
                'confirm_seeds': 0})
        path = rt.session / 'campaigns' / f'evolve-{TASK}' / 'campaign.json'
        campaign = json.loads(path.read_text())
        row = campaign['rounds'][0]
        assert row['proposer'] == 'llm' and row['llm']['status'] == 'proposed'
        assert row['tried']['kind'] == 'plan' and row['trial']['scope'] == 'full'
        assert row['accepted'] is accepted
        assert row['published'] is False
        assert row['evaluation']['before']['obligations'] == row['evaluation']['after']['obligations'] == 1
        assert row['evaluation']['after']['progress'] == (1.0 if accepted else 0.0)
        assert all(any(n['id'] == 'prepare' for n in seed['nodes']) for seed in row['after_seeds'])
        assert row['evaluation']['installation']['status'] == 'not_evaluated'
    finally:
        rt.stop()
