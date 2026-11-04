"""One source-bound evaluation entry for the five released task contracts.

This module dispatches existing contracts; it does not introduce a shared rubric
or change any task's scoring formula.
"""
from __future__ import annotations

SUPPORTED_TASKS = ('T1', 'T2', 'T3', 'T4', 'T5')


def _adapter(task):
    if task in ('T2', 'T4'):
        from . import frozen_entry
        return frozen_entry
    if task in ('T1', 'T3', 'T5'):
        from . import verifiable_entry
        return verifiable_entry
    raise ValueError('unsupported_verifiable_task:' + str(task))


def get_catalog():
    from .frozen_judge import get_catalog as frozen_catalog
    from .verifiable_entry import get_catalog as other_catalog
    return {**frozen_catalog(), **other_catalog()}


def manifest():
    from .frozen_judge import manifest as frozen_manifest
    from .verifiable_entry import manifest as other_manifest
    from .full_final import WINDOW_POLICY, WINDOW_TASKS
    return {'release_revision': 'dev490-priority-repair-20261029-rc5',
            'window_policy': WINDOW_POLICY, 'window_tasks': list(WINDOW_TASKS),
            'contracts': {'T2_T4': frozen_manifest(), 'T1_T3_T5': other_manifest()},
            'catalog': get_catalog()}


def compile_spec(task, scenario_key, full_messages, model, api_base='', **kwargs):
    _adapter(task)  # Reject unsupported tasks before dispatch.
    from .full_final import compile_spec as compile_full_final
    return compile_full_final(task, scenario_key, full_messages, model, api_base, **kwargs)


def compile_legacy_spec(task, scenario_key, full_messages, model, api_base='', **kwargs):
    """Explicit historical compiler for archived window comparisons only."""
    runtime_profile = kwargs.pop('runtime_profile', None)
    if task in ('T2', 'T4'):
        from .frozen_judge import compile_spec as compiler
    else:
        compiler = _adapter(task).compile_legacy_spec
    from .runtime_profile import apply_runtime_profile
    return apply_runtime_profile(compiler(task, scenario_key, full_messages, model,
                                         api_base, **kwargs), full_messages, runtime_profile)


def verify_source(scenario_key, document):
    task = scenario_key.split('/')[0]
    if task in ('T2', 'T4'):
        from .frozen_judge import verify_source as verify_frozen
        return verify_frozen(scenario_key, document)
    return _adapter(task).verify_source(scenario_key, document)


def load_source(task, source_path):
    return _adapter(task).load_source(task, source_path)


def judge_once(task, scenario_key, full_messages, model, **kwargs):
    return _adapter(task).judge_once(task, scenario_key, full_messages, model, **kwargs)


def _incomplete_simulation_result(task, simulation, *, source_path=None, scenario_key=None):
    """Reject incomplete runs at every public evaluation entry, without calls."""
    from .judge_rules import resolve_task_type
    from .base import EvalResult, ScoreBreakdown
    task_type = resolve_task_type(task_type_attr=getattr(task.scenario, 'task_type', None),
                                 scenario_id=task.scenario.scenario_id)
    # Only actual completed conversations may be counted as benchmark judgments.
    # Direct judge_once remains available for explicitly supplied fixed inputs.
    termination = getattr(simulation, 'termination_reason', None)
    reason = getattr(termination, 'value', termination)
    if reason != 'success':
        row = dict(task=task_type, scenario_key=scenario_key or task.scenario.scenario_id,
                   verdict='INVALID', decision_status='INVALID', scoreable=False,
                   passed=None, raw_output='', call_count=0,
                   errors=['incomplete_conversation:' + str(reason)])
        # Preserve trustworthy identity for zero-call runtime failures as well.
        try:
            key, _ = load_source(task_type, source_path)
            if scenario_key is not None and scenario_key != key:
                raise ValueError('scenario_identity_source_mismatch')
            if task.scenario.scenario_id not in (key, key.split('/', 1)[1], key.rsplit('/', 1)[1]):
                raise ValueError('scenario_identity_source_mismatch')
            entry = get_catalog()[key]
            row.update(scenario_key=key, profile=entry['profile'],
                       source_content_sha256=entry['source_content_sha256'])
            if task_type in ('T2', 'T4'):
                from .frozen_judge import revision
            else:
                revision = _adapter(task_type).manifest()['revision']
            if task_type != 'T2':
                from .full_final import WINDOW_POLICY
                row.update(revision=WINDOW_POLICY, window_policy=WINDOW_POLICY,
                           contract_revision=revision)
            else:
                row['revision'] = revision
        except (OSError, ValueError, KeyError):
            pass
        return EvalResult(task_id=task.task_id, scenario_id=task.scenario.scenario_id,
            success=False, scores=ScoreBreakdown(correctness=None, overall=0),
            turn_count=simulation.turn_count, evaluator_type='incomplete_conversation',
            feedback='INVALID', metadata={'judge': row, 'decision_status': 'INVALID',
            'scoreable': False, 'legacy_success_is_pass_indicator': True})
    return None


def evaluate_simulation(task, simulation, model, *, source_path=None, scenario_key=None, runtime_profile=None):
    from .judge_rules import resolve_task_type
    task_type = resolve_task_type(task_type_attr=getattr(task.scenario, 'task_type', None),
                                 scenario_id=task.scenario.scenario_id)
    incomplete = _incomplete_simulation_result(task, simulation, source_path=source_path,
                                             scenario_key=scenario_key)
    if incomplete is not None:
        return incomplete
    return _adapter(task_type).evaluate_simulation(
        task, simulation, model, source_path=source_path, scenario_key=scenario_key, runtime_profile=runtime_profile)
