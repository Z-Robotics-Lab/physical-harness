"""The model sees the same frozen conditions and unknowns that judge its trial."""
import json

from harness.manifest import discover
from plugins.rsi import evaluation
from scripts import evolve, evolve_llm
from scripts.evolve_evidence import compact_brief, inspect_evidence
from scripts import harness_runtime as hr
from test_evolve_rotation import _seed


def test_fixed_ruler_and_dependency_failures_reach_the_model_brief():
    binding = discover().task_bindings['recycle_cans']
    node = {'id': 'grasped-can1', 'kind': 'verify', 'skill': 'v_grasped_can1', 'args': {}}
    ref = 'plugins.mission_recycle_cans.planner:v_grasped_can1'
    contract = evaluation.compile_contract({'nodes': [node]}, task='recycle_cans',
                                          predicates={node['skill']: ref},
                                          identity={'sources': {'large-installed-source-table': 'omitted'}})
    observation = {'node': node, 'authority': 'predicate', 'source': ref, 'success': None,
                   'evidence_policy': evaluation.EVIDENCE_POLICY,
                   'blocked_reads': ["ctx.nodes_out['grasp-can1']['success']"]}
    reading = evaluation.evaluate(contract, [observation])
    before = {'count': 0, 'seeds': {'4243': {**_seed('drop-can1'), 'evaluation': reading,
                                           'verification_observations': [observation]}}}
    doc = {'task': 'recycle_cans', 'seeds': [4243, 4243], 'cursor': 0, 'rounds': [],
           'applied': {}, 'evaluation_contract': contract}
    projection = evolve_llm.rsi_projection(doc, before, hr._binding_records(binding),
                                          'robocasa', 'scripted', binding, [])
    projection["diagnosis"] = evolve._diagnose_suite(before)
    brief = compact_brief(projection)
    assert brief['evaluation']['contract_sha'] == contract['sha']
    assert brief['evaluation']['passed'] == 0
    evidence = inspect_evidence(projection, {'view': 'evaluation'})['data']
    assert evidence['contract']['obligations'] == contract['obligations']
    assert 'context' not in evidence['contract']
    seed = evidence['per_seed'][0]
    assert seed['evaluation']['vector'] == reading['vector']
    assert seed['verification_observations'] == [observation]
    assert 'blocked_reads' in json.dumps(evidence) and reading['passed'] == 0
