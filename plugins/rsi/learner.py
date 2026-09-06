"""Measured exploration of program policies, separate from development acceptance.

The model chooses inspections, experiments and parents. This class owns identities,
bounded environment sampling and receipts. A probe never updates the incumbent.
"""
from __future__ import annotations

import copy
import traceback
from typing import Callable

from harness.config import sha_json
from plugins.rsi import evaluation

METHOD = 'online_program_policy_v1'
WORKSPACE_LIMIT = 12


def intervention_summary(tried: dict) -> dict:
    """Exact small edits; large programs retain their content identity."""
    detail = tried.get('detail') or {}
    result = {k: copy.deepcopy(v) for k, v in detail.items()
              if k in ('ref', 'path', 'from', 'to', 'module', 'patch_sha', 'artifact_sha', 'graph_sha')}
    if 'graph' in detail:
        result['graph_sha'] = sha_json(detail['graph'])
    return {'kind': tried['kind'], 'node': tried.get('node'), 'detail': result}


def _failure(exc: Exception) -> dict:
    frames = traceback.extract_tb(exc.__traceback__)
    last = frames[-1] if frames else None
    return {'type': type(exc).__name__, 'message': str(exc)[:2000],
            'file': last.filename if last else None, 'line': last.lineno if last else None,
            'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)).splitlines()[-15:]}


def _subset(suite: dict, seed: int) -> dict:
    rows = {k: v for k, v in suite['seeds'].items() if str(k) == str(seed)}
    if len(rows) != 1:
        raise ValueError(f'no unique baseline observation for seed {seed}')
    return {**suite, 'seeds': rows}


def _behavior(before: dict, after: dict) -> list[dict]:
    """Compare retained observations, without equating step counts with trajectories."""
    result = []
    for seed, new in after.get('seeds', {}).items():
        old = next((v for k, v in before.get('seeds', {}).items() if str(k) == str(seed)), {})
        def nodes(row):
            values = {k: dict(v) for k, v in row.get('nodes', {}).items()}
            for item in row.get('trail') or []:
                values[item['id']] = {**values.get(item['id'], {}), **item}
            return values
        left, right = nodes(old), nodes(new)
        for node in sorted(set(left) | set(right)):
            a, b = left.get(node, {}), right.get(node, {})
            ta, tb = a.get('trace'), b.get('trace')
            # A trace must contain actual sampled series to support this equality.
            comparable = (isinstance(ta, dict) and isinstance(tb, dict)
                          and bool(ta.get('series')) and bool(tb.get('series')))
            result.append({'seed': int(seed), 'node': node,
                           'before_steps': a.get('steps'), 'after_steps': b.get('steps'),
                           'before_ok': a.get('ok', a.get('success')),
                           'after_ok': b.get('ok', b.get('success')),
                           'sampled_trace_equal': ta == tb if comparable else None})
    return result


class ProgramLearner:
    """A bounded measured workspace around an immutable incumbent policy.

    ``run`` always starts episodes from reset. A working branch is deliberately
    retained across learning cycles, not a saved simulator/controller checkpoint.
    Only the caller may accept a fully evaluated selection into its campaign.
    """

    def __init__(self, *, applied: dict, baseline: dict, contract: dict, seeds: list[int],
                 project: Callable, validate: Callable, apply: Callable, run: Callable,
                 max_probes: int = 3, observations: Callable | None = None):
        if len(seeds) != 2 or seeds[1] < seeds[0] or max_probes < 0:
            raise ValueError('invalid development seed range or probe budget')
        self.contract, self.seeds = copy.deepcopy(contract), list(seeds)
        self.baseline = baseline
        self._project, self._validate, self._apply, self._run = project, validate, apply, run
        self._observations = observations or (lambda suite: copy.deepcopy(suite['seeds']))
        self.max_probes = max_probes
        self.initial_id = self._identity(applied)
        self.policies = {self.initial_id: {'overlay': copy.deepcopy(applied), 'suite': baseline,
                                         'tried': None, 'parent_id': None}}
        self.probes: list[dict] = []
        self._measured: list[dict] = []
        self._sampled_suites: dict[tuple, dict] = {}
        self.cache_hits = 0
        self.selected_id = None
        self.selected_overlay = None
        self.selected_suite = None
        self.full_calls = 0
        self.selection_receipt = None

    def begin_cycle(self, *, max_probes: int, project=None, run=None):
        """Renew allowances without discarding neutral experiments or their parents.

        Keep recent complete ancestry paths, then useful prefixes if a path exceeds
        capacity. Retention is storage management, never a reward-based selection.
        The outer scheduler creates a new workspace when the incumbent changes.
        """
        if max_probes < 0:
            raise ValueError('invalid probe budget')
        keep = {self.initial_id}
        for key in reversed(self.policies):
            path = []
            while key not in keep:
                path.append(key)
                key = self.policies[key]['parent_id']
            room = WORKSPACE_LIMIT + 1 - len(keep)
            keep.update(list(reversed(path))[:room])
            if len(keep) == WORKSPACE_LIMIT + 1:
                break
        self.policies = {key: state for key, state in self.policies.items() if key in keep}
        # Successful measurements only: a transient exception is not a cached world.
        self._measured = [self._receipt(p) for p in [*self._measured, *self.probes]
                          if p['policy_id'] in keep and not p.get('error')][-48:]
        measured_keys = {(p['policy_id'], p['seeds'][0]) for p in self._measured}
        self._sampled_suites = {k: v for k, v in self._sampled_suites.items() if k in measured_keys}
        self.probes = []
        self.max_probes, self.cache_hits, self.full_calls = max_probes, 0, 0
        self.selected_id = self.selected_overlay = self.selected_suite = None
        self.selection_receipt = None
        if project is not None:
            self._project = project
        if run is not None:
            self._run = run

    def _identity(self, overlay: dict) -> str:
        return sha_json({'representation': 'program_overlay', 'evaluator': self.contract['sha'],
                         'overlay': overlay})

    def _policy(self, policy_id):
        key = self.initial_id if policy_id is None else policy_id
        if not isinstance(key, str) or key not in self.policies:
            raise ValueError('unknown policy_id; select an incumbent or measured working policy')
        return key, self.policies[key]

    def projection(self, policy_id=None) -> dict:
        key, state = self._policy(policy_id)
        view = self._project(state['overlay'], state['suite'])
        measurements = {}
        for probe in [*self._measured, *self.probes]:
            if probe.get('error') is not None or 'comparison' not in probe:
                continue
            comparison = probe['comparison']
            comparable = comparison['comparable']
            seed = probe['seeds'][0]
            measurements.setdefault(probe['policy_id'], {})[seed] = {
                'seed': seed, 'comparable': comparable,
                **{kind: [{k: change[k] for k in ('obligation', 'before', 'after')}
                          for change in comparison[kind]] if comparable else None
                   for kind in ('gains', 'regressions')},
                'reason': comparison['reason'], 'cost': copy.deepcopy(probe.get('cost')),
            }
        return {**view, 'policy_id': key, 'incumbent_policy_id': self.initial_id,
                'working_policies': [{'policy_id': p, 'parent_id': s['parent_id'],
                                      'scope': 'incumbent' if p == self.initial_id else 'probe',
                                      'comparison_to': self.initial_id,
                                      'acceptance_scope': 'not_evaluated',
                                      'measurements': list(measurements.get(p, {}).values()),
                                      'tried': None if s['tried'] is None else intervention_summary(s['tried'])}
                                     for p, s in self.policies.items()],
                'workspace': {'retained_candidates': len(self.policies) - 1,
                              'cycle_retention_limit': WORKSPACE_LIMIT,
                              'cached_probes': self.cache_hits},
                'probe_budget': {'limit': self.max_probes, 'used': len(self.probes)},
                'evaluation_budget': {'limit': 1, 'used': self.full_calls,
                                      'candidate_episodes': self.seeds[1] - self.seeds[0] + 1,
                                      'baseline_reused': True},
                'feedback': self._receipt(self.probes[-1]) if self.probes else None}

    @staticmethod
    def _receipt(receipt: dict) -> dict:
        # Full observations remain in the sealed exploration report and are read
        # through an explicit evidence request, never returned in every prompt.
        value = copy.deepcopy({k: v for k, v in receipt.items() if k not in ('observations', 'overlay')})
        behavior = value.get('behavior', [])
        value['behavior'] = [row for row in behavior if any(row.get(key) is not None for key in
                            ('before_steps', 'after_steps', 'before_ok', 'after_ok', 'sampled_trace_equal'))]
        value['unobserved_behavior_rows'] = (receipt.get('unobserved_behavior_rows', 0)
                                            + len(behavior) - len(value['behavior']))
        tried = value.get('tried')
        if tried:
            value['tried'] = intervention_summary(tried)
        return value

    def observations(self, policy_id=None) -> dict:
        return self._policy(policy_id)[1]['suite']

    def _measure(self, seeds, overlay, scope):
        outcome = self._run(seeds, overlay, scope)
        rows = outcome.get('seeds') or {}
        expected = {str(seed) for seed in range(seeds[0], seeds[1] + 1)}
        if len(rows) != len(expected) or {str(seed) for seed in rows} != expected:
            raise ValueError(f'{scope} did not produce the entire paired seed set')
        return outcome

    def _validate_ancestry(self, key):
        ancestry = []
        while key != self.initial_id:
            atom = self.policies[key]
            ancestry.append(atom)
            key = atom['parent_id']
        for atom in reversed(ancestry):
            self._validate(atom['tried'], self.projection(atom['parent_id']))

    def trial(self, tried: dict, parent_policy_id=None, seed=None) -> dict:
        if self.selected_id is not None:
            raise ValueError('a full selection already ended this exploration session')
        parent_id, parent = self._policy(parent_policy_id)
        if seed is None:
            seed = self.seeds[0]
        if isinstance(seed, bool) or not isinstance(seed, int) or not self.seeds[0] <= seed <= self.seeds[1]:
            raise ValueError('probe seed must belong to the declared development seed range')
        if tried.get('kind') == 'none':
            raise ValueError('a probe requires an executable candidate')
        candidate = copy.deepcopy(tried)
        self._validate_ancestry(parent_id)
        self._validate(candidate, self.projection(parent_id))
        overlay = self._apply(candidate, parent['overlay'])
        policy_id = self._identity(overlay)
        if policy_id == parent_id:
            raise ValueError('candidate does not change its parent program policy')
        duplicate = next((p for p in [*self._measured, *self.probes] if p['policy_id'] == policy_id
                          and p['seeds'] == [seed] and not p.get('error')), None)
        if duplicate is not None:
            self.cache_hits += 1
            if policy_id != self.initial_id:
                self.policies[policy_id]['suite'] = self._sampled_suites[(policy_id, seed)]
            return {**self._receipt(duplicate), 'cached': True}
        if len(self.probes) >= self.max_probes:
            raise ValueError('probe budget exhausted; choose a measured policy or stop')
        receipt = {'policy_id': policy_id, 'parent_id': parent_id, 'scope': 'probe',
                   'seeds': [seed], 'tried': candidate, 'accepted': False,
                   'acceptance_scope': 'not_evaluated', 'error': None}
        # Reserve before executing: a simulator exception still spends this probe.
        self.probes.append(receipt)
        try:
            outcome = self._measure([seed, seed], overlay, 'probe')
            incumbent = _subset(self.baseline, seed)
            comparison = evaluation.compare(incumbent, outcome, self.contract)
            receipt.update(evaluation=evaluation.summary(outcome, self.contract),
                           baseline_evaluation=evaluation.summary(incumbent, self.contract),
                           comparison={k: comparison[k] for k in ('accepted', 'comparable', 'gains', 'regressions', 'reason')},
                           comparison_to=self.initial_id,
                           behavior=_behavior(incumbent, outcome),
                           failures=[{'seed': int(k), 'node': v.get('first_death'),
                                      'failure_mode': v.get('failure_mode')}
                                     for k, v in outcome['seeds'].items()],
                           measurement_sha=outcome.get('sha'),
                           experiment_id=outcome.get('experiment_id'),
                           observations=self._observations(outcome), overlay=copy.deepcopy(overlay),
                           cost={'episodes': len(outcome['seeds']), 'sim_s': outcome.get('elapsed_s')})
            # A reversible edit can revisit an existing content identity. Keep
            # its original ancestry immutable; rewriting it creates cycles and
            # can even overwrite the incumbent with a one-seed probe.
            self.policies.setdefault(policy_id, {'overlay': overlay, 'suite': outcome, 'tried': candidate,
                                                 'parent_id': parent_id})
            if policy_id != self.initial_id:
                self.policies[policy_id]['suite'] = outcome
            self._sampled_suites[(policy_id, seed)] = outcome
        except Exception as exc:
            receipt['error'] = _failure(exc)
            raise
        return self._receipt(receipt)

    def choose(self, policy_id: str) -> dict:
        key, state = self._policy(policy_id)
        if key == self.initial_id:
            raise ValueError('use stop to retain the incumbent without a candidate')
        if self.full_calls:
            raise ValueError('full evaluation budget exhausted for this decision session')
        self.full_calls += 1
        # Validate the selected atom against its actual parent before using its
        # composed overlay. Prior validated atoms are content-addressed policies.
        self.selection_receipt = {'policy_id': key, 'parent_id': state['parent_id'],
                                  'scope': 'full', 'seeds': list(range(self.seeds[0], self.seeds[1] + 1)),
                                  'error': None}
        try:
            self._validate_ancestry(key)
            outcome = self._measure(self.seeds, state['overlay'], 'full')
            if {str(k) for k in outcome.get('seeds', {})} != {str(k) for k in self.baseline.get('seeds', {})}:
                raise ValueError('full selection did not produce the entire paired seed set')
            self.selection_receipt.update(measurement_sha=outcome.get('sha'),
                                          experiment_id=outcome.get('experiment_id'),
                                          cost={'episodes': len(outcome['seeds']), 'sim_s': outcome.get('elapsed_s')})
        except Exception as exc:
            self.selection_receipt['error'] = _failure(exc)
            raise
        self.selected_id = key
        self.selected_overlay = copy.deepcopy(state['overlay'])
        self.selected_suite = outcome
        return copy.deepcopy(state['tried'])

    def report(self) -> dict:
        return {'method': METHOD, 'representation': 'program_overlay',
                'incumbent_policy_id': self.initial_id, 'selected_policy_id': self.selected_id,
                'probes': copy.deepcopy(self.probes), 'probe_limit': self.max_probes,
                'cached_probes': self.cache_hits,
                'retained_candidates': len(self.policies) - 1,
                'selection': copy.deepcopy(self.selection_receipt),
                'full_evaluations': self.full_calls,
                'limits': 'Bounded working branches survive cycles within this run and incumbent; no model weights are trained.'}
