"""Single-call production adapter for the frozen T2/T4 judge.

Compilation and parsing live in ``frozen_judge``. This adapter owns only the
transport boundary and status accounting shared by CLI and benchmark paths.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

DECISIONS = ('PASS', 'FAIL', 'U', 'NA', 'INVALID')


def _core():
    from benchmark.evaluator import frozen_judge
    return frozen_judge


def default_transport(request):
    from benchmark.utils import llm_utils
    from benchmark.user.first_pass import CallBudget
    evidence = {'attempts': 0, 'raw_output': ''}
    budget = CallBudget(limit=1)
    budget.reject_invalid_responses = True
    try:
        explicit = {k: v for k, v in request.items()
                    if v is not None and not (k == 'api_base' and not v)}
        model, kwargs = llm_utils._route_model(**explicit)
        # A frozen explicit request wins over route defaults. Credentials and
        # provider configuration still come from the existing route resolver.
        kwargs.update({k: v for k, v in explicit.items() if k != 'model'})
        from .runtime_profile import FLASH_K1, PROFILES
        flash = PROFILES[FLASH_K1]
        if (request.get('model') in flash['models']
                and request.get('reasoning_effort') == 'high'
                and request.get('stream') is False
                and 'temperature' not in request and 'thinking' not in request):
            # Routing may add sampling/thinking defaults or force streaming.
            # The frozen Flash profile requires these fields to stay omitted.
            wire_fields = ('model', 'messages', 'temperature', 'max_tokens',
                           'max_completion_tokens', 'reasoning_effort', 'response_format',
                           'top_p', 'seed', 'stream', 'thinking', 'stream_options',
                           'timeout', 'num_retries', 'max_retries')
            for key in wire_fields:
                if key != 'model' and key not in explicit:
                    kwargs.pop(key, None)
            kwargs.pop('_force_stream', None)
            if kwargs.get('extra_body'):
                kwargs['extra_body'] = {k: v for k, v in kwargs['extra_body'].items()
                                        if k not in wire_fields and k != 'enable_thinking'}
        # OpenAI-compatible routers accept these extension fields on the wire;
        # LiteLLM's provider validator requires them in extra_body.
        if 'thinking' in kwargs:
            kwargs['extra_body'] = {**(kwargs.get('extra_body') or {}),
                                    'thinking': kwargs.pop('thinking')}
        if request.get('stream') is False:
            kwargs['extra_body'] = {**(kwargs.get('extra_body') or {}), 'stream': False}
        if urlparse(kwargs.get('api_base') or '').hostname in ('127.0.0.1', 'localhost', '::1'):
            # The local development router is intentionally unauthenticated;
            # the OpenAI SDK still requires a nonempty placeholder.
            kwargs['api_key'] = 'local-router-no-auth'
        kwargs['num_retries'] = 0
        kwargs.pop('max_retries', None)
        # The route may contain a recording gateway with bounded upstream
        # retries. Its client deadline must cover that configured transport.
        from benchmark.utils.llm_routes import get_model_config
        route_timeout = get_model_config(request['model']).get('timeout')
        if isinstance(route_timeout, (int, float)):
            kwargs['timeout'] = max(kwargs.get('timeout') or 0, route_timeout)
        evidence['actual_request'] = {
            k: v for k, v in {'model': model, **kwargs}.items()
            if k in ('model', 'messages', 'temperature', 'max_tokens', 'response_format',
                     'timeout', 'extra_body', 'reasoning_effort', 'num_retries')}
        # Never include credentials (including arbitrary extra_body entries).
        if 'extra_body' in evidence['actual_request']:
            extra = evidence['actual_request']['extra_body'] or {}
            evidence['actual_request']['extra_body'] = {
                k: extra[k] for k in ('enable_thinking', 'thinking', 'stream') if k in extra}
        response = llm_utils._completion_with_retry(
            model=model, max_retries=0, _first_pass_stage=evidence,
            _first_pass_budget=budget, **kwargs)
        choice = response.choices[0]
        evidence.update(raw_output=choice.message.content or '',
                        response_model=getattr(response, 'model', None),
                        finish_reason=getattr(choice, 'finish_reason', None),
                        request_id=getattr(response, 'id', None),
                        reasoning_content=getattr(choice.message, 'reasoning_content', None))
        if evidence['finish_reason'] == 'length':
            evidence['transport_error'] = 'truncated_response'
    except Exception as exc:
        evidence['transport_error'] = type(exc).__name__
    return evidence


def _invalid(task, scenario_key, error, *, raw='', call_count=0, **metadata):
    return dict(task=task, scenario_key=scenario_key, status='invalid',
                decision_status='INVALID', verdict='INVALID', scoreable=False,
                passed=None, raw_output=raw, call_count=call_count,
                errors=[error], **metadata)


def judge_once(task, scenario_key, full_messages, model, *, source_document=None,
               api_base='', transport=None, runtime_profile=None):
    """Compile, make exactly one call, and parse without repair or retry.

    ``source_document`` is the original scenario JSON when supplied by a
    benchmark loader. Standalone callers may select the packaged, hash-checked
    source by its full ``T2/v2/T2_026`` identity instead.
    """
    preflight_metadata = {}
    try:
        core = _core()
        if source_document is not None:
            core.verify_source(scenario_key, source_document)
        # Preserve the verified source contract even when the input cannot be
        # compiled (e.g. the frozen T4 context limit). Resume must keep these
        # INVALID positions, not quietly rerun them. Never attach this identity
        # before an explicitly supplied source has passed verification.
        catalog_entry = core.get_catalog().get(scenario_key)
        if catalog_entry is not None and catalog_entry.get('task') == task:
            preflight_metadata = dict(revision=core.revision,
                profile=catalog_entry['profile'], provisional=catalog_entry.get('provisional'),
                source_content_sha256=catalog_entry['source_content_sha256'])
            if task == 'T4':
                from .full_final import WINDOW_POLICY
                preflight_metadata.update(revision=WINDOW_POLICY, window_policy=WINDOW_POLICY,
                                          contract_revision=core.revision)
        if not model:
            raise ValueError('judge_model_required')
        from .task_entry import compile_spec
        spec = compile_spec(task, scenario_key, full_messages, model, api_base=api_base,
                            runtime_profile=runtime_profile)
    except Exception as exc:
        return _invalid(task, scenario_key, f'preflight:{type(exc).__name__}:{exc}',
                        **preflight_metadata)
    metadata = {
        'revision': spec.get('revision', spec.get('frozen_revision')),
        'profile': spec.get('profile', spec.get('frozen_profile')),
        'provisional': spec.get('provisional'),
        'source_content_sha256': spec.get('source_content_sha256'),
        'request_hash': spec.get('request_hash'),
        'spec_hash': hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True,
                                              default=str).encode()).hexdigest(),
    }
    metadata['runtime_identity'] = spec['runtime_identity']
    for key in ('window_policy', 'contract_revision', 'window_map'):
        if key in spec:
            metadata[key] = spec[key]
    raw = ''
    diagnostics = {}
    try:
        response = (transport or default_transport)(spec['request'])
        if isinstance(response, str):
            raw = response
        elif isinstance(response, dict):
            raw = response.get('raw_output', '')
            diagnostics = {k: v for k, v in response.items() if k != 'raw_output'}
            if response.get('transport_error'):
                return _invalid(task, scenario_key, 'transport:' + response['transport_error'],
                                raw=raw, call_count=response.get('attempts', 1),
                                transport_diagnostics=diagnostics, **metadata)
        else:
            raise TypeError('transport_requires_text_or_raw_output_mapping')
        if not isinstance(raw, str):
            raise TypeError('raw_output_requires_string')
    except Exception as exc:
        return _invalid(task, scenario_key, f'transport:{type(exc).__name__}', raw=raw,
                        call_count=1, transport_diagnostics=diagnostics, **metadata)
    try:
        parsed = core.parse_response(raw, spec)
        status = parsed.get('verdict')
        if status not in DECISIONS or parsed.get('scoreable') is False:
            # NA/U may deliberately be unscoreable but remain distinguishable.
            status = status if status in ('NA', 'U') else 'INVALID'
        return {**parsed, **metadata, 'task': task, 'scenario_key': scenario_key,
                'decision_status': status, 'verdict': status,
                'scoreable': status in ('PASS', 'FAIL'),
                'passed': status == 'PASS' if status in ('PASS', 'FAIL') else None,
                'raw_output': raw, 'call_count': 1,
                'transport_diagnostics': diagnostics}
    except Exception as exc:
        return _invalid(task, scenario_key, f'parse:{type(exc).__name__}', raw=raw,
                        call_count=1, transport_diagnostics=diagnostics, **metadata)


def summarize(rows):
    """Keep all attempted/planned rows, with abstentions separate from FAIL."""
    counts = dict.fromkeys(DECISIONS, 0)
    for row in rows:
        counts[row.get('decision_status') if row.get('decision_status') in counts
               else 'INVALID'] += 1
    total = len(rows)
    scoreable = counts['PASS'] + counts['FAIL']
    return dict(total=total, decision_counts=counts, scoreable=scoreable,
                scoreable_coverage=scoreable / total if total else None,
                pass_rate_over_total=counts['PASS'] / total if total else None,
                pass_rate_over_scoreable=counts['PASS'] / scoreable if scoreable else None)


def load_source(task, source_path):
    """Resolve a real file to exactly one frozen full identity by source hash."""
    if not source_path:
        raise ValueError('original_scenario_source_path_required')
    document = json.loads(Path(source_path).read_text(encoding='utf-8'))
    digest = hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    matches = [key for key, row in _core().get_catalog().items()
               if key.startswith(task + '/') and row.get('source_content_sha256') == digest]
    if len(matches) != 1:
        raise ValueError('unknown_or_ambiguous_scenario_source')
    return matches[0], document


def evaluate_simulation(task, simulation, model, *, source_path=None, scenario_key=None, runtime_profile=None):
    """Adapt the production result without flattening U/NA into agent FAIL."""
    from .task_entry import _incomplete_simulation_result
    incomplete = _incomplete_simulation_result(task, simulation, source_path=source_path,
                                             scenario_key=scenario_key)
    if incomplete is not None:
        return incomplete
    from benchmark.evaluator.base import EvalResult, ScoreBreakdown
    from benchmark.evaluator.judge_rules import resolve_task_type
    scenario = task.scenario
    task_type = resolve_task_type(task_type_attr=getattr(scenario, 'task_type', None),
                                  scenario_id=scenario.scenario_id)
    messages = [{'role': getattr(m.role, 'value', m.role), 'content': m.content}
                for m in simulation.conversation.messages]
    try:
        key, source = load_source(task_type, source_path)
        if scenario_key is not None and scenario_key != key:
            raise ValueError('loaded_scenario_source_identity_changed')
        if scenario.scenario_id not in (key, key.split('/', 1)[1], key.rsplit('/', 1)[1]):
            raise ValueError('scenario_identity_source_mismatch')
        row = judge_once(task_type, key, messages, model, source_document=source, runtime_profile=runtime_profile)
    except Exception as exc:
        row = _invalid(task_type, scenario.scenario_id,
                       f'preflight:{type(exc).__name__}:{exc}')
    return EvalResult(
        task_id=task.task_id, scenario_id=scenario.scenario_id,
        success=row['passed'] is True,
        scores=ScoreBreakdown(correctness=float(row['passed']) if row['scoreable'] else None,
                              overall=float(row['passed'] is True)),
        turn_count=simulation.turn_count, evaluator_type='frozen_t2_t4',
        feedback=row['decision_status'], metadata={'judge': row,
            'scoreable': row['scoreable'], 'decision_status': row['decision_status'],
            'legacy_success_is_pass_indicator': True})


def evaluation_rows(results):
    """Extract statuses for reports while preserving other tasks' old results."""
    return [((r.metadata or {}).get('judge') or
             {'decision_status': 'PASS' if r.success else 'FAIL'}) for r in results]


def judgment_fields(result):
    """Fields to retain in every legacy JSON serialization of EvalResult."""
    row = ((result.metadata or {}).get('judge') if result is not None else None)
    return ({'judge': row, 'decision_status': row['decision_status'],
             'scoreable': row['scoreable']} if row is not None else {})
