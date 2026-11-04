"""Versioned, source-bound T2/T4 judge compilation and parsing.

No model call, retry, credential, experiment file, or reference label is needed
here. The catalog selects one previously validated profile per scenario.
"""
import copy
import json
from functools import lru_cache
from importlib.resources import files

from .. import shared_rubric
from . import _program, _scope, _scope_tolerance
from ._contracts import digest

revision = 't2-t4-priority-repaired-20261029-v3'
_parser = _scope_tolerance.make_parser(_scope.make_parser(_program))


@lru_cache(maxsize=1)
def _resources():
    root = files(__package__).joinpath('resources')
    catalog = json.loads(root.joinpath('manifest.json').read_text(encoding='utf-8'))
    templates = json.loads(root.joinpath('templates.json').read_text(encoding='utf-8'))
    if catalog['revision'] != revision or digest(templates) != catalog['templates_sha256']:
        raise ValueError('frozen_bundle_integrity_mismatch')
    if set(templates) != set(catalog['catalog']):
        raise ValueError('frozen_bundle_catalog_mismatch')
    for key, template in templates.items():
        if digest(template) != catalog['catalog'][key]['template_sha256']:
            raise ValueError('frozen_template_integrity_mismatch:' + key)
    return catalog, templates


def manifest():
    return copy.deepcopy(_resources()[0])


def get_catalog():
    return copy.deepcopy(_resources()[0]['catalog'])


def available_scenarios(task=None):
    if task not in (None, 'T2', 'T4'):
        raise ValueError('unsupported_task')
    return sorted(key for key, entry in _resources()[0]['catalog'].items() if task is None or entry['task'] == task)


def verify_source(scenario_key, source_document):
    entry = _resources()[0]['catalog'].get(scenario_key)
    if entry is None:
        raise ValueError('unknown_frozen_scenario:' + str(scenario_key))
    if not isinstance(source_document, dict) or digest(source_document) != entry['source_content_sha256']:
        raise ValueError('frozen_source_content_mismatch:' + scenario_key)
    return True


def compile_spec(task, scenario_key, full_messages, model, api_base='', input_id='runtime', profile=None):
    """Fill the frozen scenario template with the actual final-turn conversation."""
    catalog, templates = _resources()
    entry = catalog['catalog'].get(scenario_key)
    if entry is None or entry['task'] != task:
        raise ValueError('unknown_frozen_scenario_or_task:' + str(scenario_key))
    if profile is not None and profile != entry['profile']:
        raise ValueError('unvalidated_profile_override')
    if not isinstance(full_messages, list) or not full_messages:
        raise ValueError('final_assistant_message_required')
    if any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant') or not isinstance(m.get('content'), str) for m in full_messages):
        raise ValueError('invalid_conversation_message')
    if full_messages[-1]['role'] != 'assistant':
        raise ValueError('final_assistant_message_required')
    if not isinstance(model, str) or not model:
        raise ValueError('model_required')
    messages = copy.deepcopy(full_messages)
    final_reply = messages[-1]['content']
    if task == 'T4' and len(final_reply) > 5000:
        raise ValueError('T4_final_reply_exceeds_frozen_5000_character_window')
    history = '\n'.join(m['role'] + ': ' + m['content'] for m in messages[:-1])
    template = templates[scenario_key]
    spec = copy.deepcopy(template['spec'])
    if task == 'T2':
        data = copy.deepcopy(template['user_data'])
        paragraphs = shared_rubric.paragraphs(messages)
        prefix = f'M{len(messages)-1:03d}.'
        data['applicability_input']['history_paragraphs'] = [p for p in paragraphs if not p['id'].startswith(prefix)]
        checks_key = 'checks_input' if entry['profile'] == 'scope' else 'checks_input_only_if_applicable'
        data[checks_key]['final_reply_paragraphs'] = [p for p in paragraphs if p['id'].startswith(prefix)]
        user = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        # Preserve exactly the inherited 1500+1500 history window and markers.
        visible = shared_rubric.b0_messages(final_reply, history, {})[1]['content']
        window = visible.split('## CONVERSATION\n', 1)[1].split('## QUESTIONS:\n', 1)[0]
        user = template['user_prefix'] + '## CONVERSATION\n' + window + template['user_suffix']
    request = copy.deepcopy(template['request_options'])
    request.update(model=model, api_base=api_base or '', messages=[{'role': 'system', 'content': template['system_prompt']}, {'role': 'user', 'content': user}])
    spec.update(input_id=input_id, fixture={'full_messages': messages, 'final_reply': final_reply, 'conv_text': history},
                full_messages=messages, request=request, request_hash=digest(request),
                expected_request_body_hash=digest({k:v for k,v in request.items() if k not in ('api_base', 'timeout', 'num_retries', 'max_retries')}),
                frozen_revision=revision, frozen_profile=entry['profile'], source_content_sha256=entry['source_content_sha256'],
                provisional=entry['provisional'])
    return spec


def compile_request(scenario_key, full_messages, model, api_base=None):
    return compile_spec(scenario_key.split('/')[0], scenario_key, full_messages, model, api_base or '')


def parse_response(raw, spec):
    """Validate consumed facts, apply registered formulas, retain U/NA/invalid."""
    result = _parser(raw, spec)
    historical = result.pop('posthoc_auxiliary_tolerance', None)
    if historical is not None:
        historical.pop('not_formal_result', None)
        historical['policy_revision'] = revision
        result['scope_auxiliary_tolerance'] = historical
    return result
