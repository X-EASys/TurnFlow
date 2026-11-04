"""Versioned call parameters; frozen judge semantics and parsers stay intact."""
from copy import deepcopy
import hashlib
import json

DEFAULT = 'frozen_default'
OUTPUT = 'glm52_judge_output_v1'
PAPER_K1 = 'teamrouter_gpt54mini_judge_k1_v1'
FLASH_K1 = 'deepseek_v4_flash_judge_k1_v1'
PROFILES = {DEFAULT: {'name': DEFAULT, 'version': 1, 'overrides': {}},
            OUTPUT: {'name': OUTPUT, 'version': 1, 'tasks': ['T17', 'T20'],
                     'models': ['glm-5.2', 'zai/glm-5.2'],
                     'overrides': {'max_tokens': 16384},
                     'thinking': {'type': 'disabled'}},
            PAPER_K1: {'name': PAPER_K1, 'version': 1,
                       'tasks': ['T16', 'T17', 'T18', 'T19', 'T20'],
                       'models': ['paper-judge', 'gpt-5.4-mini'],
                       'overrides': {'max_tokens': 16384},
                       'wire_defaults': {'stream': False},
                       'sampling_policy': 'retain frozen task parameters; route adaptation requires sealed evidence',
                       'judge_calls_per_conversation': 1},
            FLASH_K1: {'name': FLASH_K1, 'version': 1,
                       'tasks': ['T16', 'T17', 'T18', 'T19', 'T20'],
                       'models': ['paper-judge', 'deepseek-v4-flash', 'openai/deepseek-v4-flash'],
                       'overrides': {'max_tokens': 16384, 'reasoning_effort': 'high',
                                     'stream': False, 'timeout': 330},
                       'omit_fields': ['temperature', 'thinking'],
                       'wire_defaults': {'stream': False},
                       'judge_calls_per_conversation': 1,
                       'evidence': '2026-10-13-full-final-window; no new semantic change'}}


def effective_wire(request, profile=DEFAULT):
    fields = ('model', 'messages', 'temperature', 'max_tokens', 'response_format',
              'top_p', 'seed', 'reasoning_effort', 'stream', 'thinking')
    wire = {k: v for k, v in request.items() if k in fields and v is not None}
    if profile in (PAPER_K1, FLASH_K1):
        wire.pop('thinking', None)
        wire.update(PROFILES[profile]['wire_defaults'])
        for field in PROFILES[profile].get('omit_fields', []):
            wire.pop(field, None)
    else:
        wire.setdefault('thinking', {'type': 'disabled'})
        wire.setdefault('stream', False)
    return wire


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def apply_runtime_profile(spec, full_messages, profile=None):
    name = profile or DEFAULT
    if name not in PROFILES:
        raise ValueError('unknown_judge_runtime_profile')
    result = deepcopy(spec)
    effective = name
    if name == OUTPUT and (spec['task'] not in PROFILES[name]['tasks'] or
                           spec['request']['model'] not in PROFILES[name]['models']):
        effective = DEFAULT
    request = result['request']
    if name in (PAPER_K1, FLASH_K1) and (spec['task'] not in PROFILES[name]['tasks'] or
                             request['model'] not in PROFILES[name]['models']):
        raise ValueError('paper_k1_judge_task_or_model_mismatch')
    if effective in (OUTPUT, PAPER_K1, FLASH_K1):
        request.update(PROFILES[effective]['overrides'])
        if effective in (PAPER_K1, FLASH_K1):
            request.pop('thinking', None)
            request.update(PROFILES[effective]['wire_defaults'])
            for field in PROFILES[effective].get('omit_fields', []):
                request.pop(field, None)
        result['frozen_request_hash'] = spec.get('request_hash')
        result['request_hash'] = digest(request)
        if 'expected_request_body_hash' in result:
            result['frozen_expected_request_body_hash'] = spec['expected_request_body_hash']
            result['expected_request_body_hash'] = digest({k: v for k, v in request.items()
                if k not in ('api_base', 'timeout', 'num_retries', 'max_retries')})
    # api_base is transport configuration, not a model request field. The
    # upstream wire includes the existing default thinking and stream fields.
    wire = effective_wire(request, effective)
    identity = dict(schema='judge-runtime-identity-v1', profile=effective,
        profile_sha256=digest(PROFILES[effective]),
        effective_request_sha256=digest(wire), conversation_sha256=digest(full_messages),
        semantic_revision=spec.get('revision', spec.get('frozen_revision')))
    result['runtime_identity'] = identity
    result['effective_request_hash'] = identity['effective_request_sha256']
    return result
