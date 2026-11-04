"""Official full-final windows; frozen rubrics, parsers and defaults stay intact.

The window policy retains the inherited contracts. Compilation and parsing
perform no transport calls.
"""
from copy import deepcopy

from . import frozen_judge, shared_rubric, verifiable_entry
from .runtime_profile import apply_runtime_profile, digest


WINDOW_POLICY = 'full-final-v1'
WINDOW_TASKS = ('T16', 'T18', 'T19', 'T20')


def _validate_messages(messages):
    # Validate the real input before any local rendering placeholder is used.
    if not isinstance(messages, list) or not messages:
        raise ValueError('final_assistant_message_required')
    if any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant')
           or not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('invalid_conversation_message')
    if messages[-1]['role'] != 'assistant':
        raise ValueError('final_assistant_message_required')


def _t19_prompt(scenario_key, history, reply):
    # Render a local skeleton using the inherited wrapper. Split only this
    # controlled text, never conversation text that may contain its delimiters.
    history_slot, reply_slot = '__full_final_history__', '__full_final_reply__'
    skeleton = shared_rubric.b0_messages(reply_slot, history_slot, {})[1]['content']
    window = skeleton.split('## CONVERSATION\n', 1)[1].split('## QUESTIONS:\n', 1)[0]
    before, rest = window.split(history_slot)
    middle, after = rest.split(reply_slot)
    visible_history = history if len(history) <= 3000 else history[:1500] + '\n...\n' + history[-1500:]
    # _resources verifies the packaged template hashes before returning them.
    template = frozen_judge._resources()[1][scenario_key]
    return (template['user_prefix'] + '## CONVERSATION\n' + before + visible_history
            + middle + reply + after + template['user_suffix'])


def compile_spec(task, scenario_key, full_messages, model, api_base='', **kwargs):
    """Compile a complete final reply while retaining each task's history view.

    T17 is delegated unchanged. Other tasks retain ordinary short requests and
    carry an explicit window revision plus their inherited contract revision.
    T19 also repairs the inherited delimiter split for short text containing
    prompt markers, which could otherwise hide part or all of the final reply.
    """
    runtime_profile = kwargs.pop('runtime_profile', None)
    if task == 'T17':
        return apply_runtime_profile(frozen_judge.compile_spec(
            task, scenario_key, full_messages, model, api_base, **kwargs),
            full_messages, runtime_profile)
    _validate_messages(full_messages)
    reply = full_messages[-1]['content']
    limit = 6000 if task == 'T20' else 5000
    compiler = frozen_judge.compile_spec if task == 'T19' else verifiable_entry.compile_legacy_spec
    # The frozen T19 guard remains intact. Its short empty render supplies only
    # source-bound contract metadata; all returned conversation identity is real.
    render_messages = deepcopy(full_messages)
    if task == 'T19' and len(reply) > limit:
        render_messages[-1]['content'] = ''
    spec = compiler(task, scenario_key, render_messages, model, api_base, **kwargs)
    if task == 'T19':
        history = '\n'.join(m['role'] + ': ' + m['content'] for m in full_messages[:-1])
        spec['full_messages'] = deepcopy(full_messages)
        spec['fixture'] = {'full_messages': deepcopy(full_messages),
                           'final_reply': reply, 'conv_text': history}
        spec['window_map'] = {
            'history_kept': ([[0, 1500], [len(history)-1500, len(history)]]
                             if len(history) > 3000 else [[0, len(history)]]),
            'history_dropped': [1500, len(history)-1500] if len(history) > 3000 else None,
        }
    else:
        history = spec['conv_text']
    if task == 'T19':
        spec['request']['messages'][-1]['content'] = _t19_prompt(scenario_key, history, reply)
    elif len(reply) > limit:
        # Choose a slot absent from the existing request, including source
        # questions/history. Substitute by position once, before user text
        # can introduce a second occurrence of that slot.
        # Numeric alternatives stay short even when the input contains a very
        # long run of a marker suffix; the frozen renderer must not clip them.
        slot_index = 0
        slot = '__full_final_reply_0__'
        while any(slot in message['content'] for message in spec['request']['messages']):
            slot_index += 1
            slot = f'__full_final_reply_{slot_index}__'
        render_messages[-1]['content'] = slot
        skeleton = compiler(task, scenario_key, render_messages, model, api_base, **kwargs)
        prompt = skeleton['request']['messages'][-1]['content']
        if prompt.count(slot) != 1:
            raise ValueError('full_final_render_slot_not_unique')
        before, after = prompt.split(slot)
        spec['request']['messages'][-1]['content'] = before + reply + after
    spec.update(full_reply=reply, reply_visible=reply, conv_text=history)
    spec['window_map'].update(reply_kept=[0, len(reply)], reply_dropped=None)
    spec['contract_revision'] = spec.get('revision', spec.get('frozen_revision'))
    spec.update(revision=WINDOW_POLICY, window_policy=WINDOW_POLICY)
    spec['request_hash'] = digest(spec['request'])
    if 'expected_request_body_hash' in spec:
        spec['expected_request_body_hash'] = digest({k: v for k, v in spec['request'].items()
            if k not in ('api_base', 'timeout', 'num_retries', 'max_retries')})
    result = apply_runtime_profile(spec, full_messages, runtime_profile)
    result['runtime_identity'].update(window_policy=WINDOW_POLICY,
                                      contract_revision=spec['contract_revision'])
    return result


def parse_response(raw, spec):
    """Use the original task parser and reducer, including their abstentions."""
    parser = frozen_judge if spec['task'] in ('T17', 'T19') else verifiable_entry
    return parser.parse_response(raw, spec)
