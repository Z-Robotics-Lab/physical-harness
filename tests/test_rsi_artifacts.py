"""Candidate import names and experiment identities cannot silently change content."""
import pytest

from scripts import evolve, evolve_llm


def test_candidate_name_cannot_be_overwritten_even_after_rejection(tmp_path):
    target = tmp_path / 'candidate'
    original = {'manifest.toml': 'enabled = false\n', '__init__.py': 'VALUE = 1\n'}
    assert evolve_llm._write_files_once(target, original) is None
    assert evolve_llm._write_files_once(target, original) is None
    digest = evolve_llm.candidate_digest(target)
    reason = evolve_llm._write_files_once(target, {**original, '__init__.py': 'VALUE = 2\n'})
    assert 'immutable' in reason
    assert evolve_llm.candidate_digest(target) == digest
    assert (target / '__init__.py').read_text() == 'VALUE = 1\n'


def test_candidate_digest_covers_data_files_but_not_interpreter_cache(tmp_path):
    (tmp_path / '__init__.py').write_text('')
    (tmp_path / 'policy.json').write_text('{"gain":1}')
    original = evolve_llm.candidate_digest(tmp_path)
    cache = tmp_path / '__pycache__'
    cache.mkdir()
    (cache / 'model.pyc').write_bytes(b'cache')
    assert evolve_llm.candidate_digest(tmp_path) == original
    (tmp_path / 'policy.json').write_text('{"gain":2}')
    assert evolve_llm.candidate_digest(tmp_path) != original


def test_trial_refuses_an_accepted_artifact_changed_on_disk_before_mount(tmp_path, monkeypatch):
    (tmp_path / '__init__.py').write_text('')
    digest = evolve_llm.candidate_digest(tmp_path)
    (tmp_path / '__init__.py').write_text('CHANGED = True\n')
    monkeypatch.setenv(evolve.OVERRIDE_ENV, '{}')
    with pytest.raises(ValueError, match='source changed'):
        evolve.run_suite('test', {}, [1, 1], 'auto', tmp_path / 'skills', {
            'executors': {}, 'tunables': {},
            'cards': {'candidate': {'path': str(tmp_path), 'artifact_sha': digest}}})
