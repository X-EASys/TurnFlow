"""Current-turn material for the first-pass input experiment.

Scenario-wide domain, plot overviews, ground truth and inferred answers are not
speech sources. Actual user history is retained verbatim, including updates.
"""
from benchmark.data_model.message import UserMessage, AssistantMessage


PREFERENCE_GENERATION_TEMPLATE = "PREFERENCE ALREADY EXPRESSED — GENERATION BOUNDARY, NOT NEW SPEECH:\nThe earlier user message expressing this scenario's preference was:\n{source_user_message}\n\nDo not proactively restate or strengthen the preference expressed in that message.\nDo not add it as a method, channel, location, timing, or payment requirement for\nthe current item or request. Express THIS TURN and its supplied facts naturally.\nPreserve requirements explicitly supplied in THIS TURN.\nIf THIS TURN explicitly calls for restating or updating that preference, or the\nassistant directly asks about the past habit, answer that point from disclosed\nfacts without turning the habit into a new requirement for the current task.\nDo not state that the preference is wrong, identify exceptions, or supply a solution."


def preference_generation_state(scenario, aim, messages):
    """Identify emitted source speech only; this is not semantic verification."""
    config = (scenario.metadata or {}).get('simulator_no_repeat')
    if not config:
        return 'unconfigured', None
    source_id = config.get('preference_source_aim_id')
    if not source_id or not config.get('preference_text'):
        raise ValueError('Incomplete simulator_no_repeat binding')
    matches = [a for a in scenario.state_queue.aim_sequence.aims if a.aim_id == source_id]
    if len(matches) != 1:
        raise ValueError('Preference source aim must exist uniquely')
    current_id = getattr(aim, 'aim_id', None)
    if current_id is None:
        return 'opening', None
    if current_id == source_id:
        return 'source_turn', None
    for message in reversed(messages):
        metadata = getattr(message, 'metadata', None) or {}
        if (isinstance(message, UserMessage) and message.content.strip()
                and metadata.get('aim_id') == source_id
                and metadata.get('delivery_status') == 'emitted'):
            return 'source_turn_emitted', message
    return 'missing_emitted_source_metadata', None


def preference_generation_note(scenario, aim, messages) -> str:
    """Return the frozen generation boundary, quoting actual emitted speech."""
    _, message = preference_generation_state(scenario, aim, messages)
    return PREFERENCE_GENERATION_TEMPLATE.format(source_user_message=message.content) if message else ''


def current_payload(scenario, aim):
    sq = scenario.state_queue
    sources = {'flow_info': sq.flow_info, 'constraints': sq.constraints,
               'distractor': getattr(sq, 'distractor', None) or []}
    result = []
    seen = set()
    for source, keys in [('flow_info', aim.required_info_keys),
                         ('constraints', aim.required_constraint_keys)]:
        for key in keys:
            order = [source] if source == 'constraints' else ['flow_info', 'distractor', 'constraints']
            for origin in order:
                items = [item for item in sources[origin] if
                         (item.get('key') if isinstance(item, dict) else item.key) == key]
                if len(items) > 1:
                    raise ValueError(f'Ambiguous {origin} key: {key}')
                if not items:
                    continue
                item = items[0]
                identity = f'{origin}.{key}'
                if identity not in seen:
                    if isinstance(item, dict):
                        value, exact = item['value'], False
                    else:
                        value = item.get_effective_surface() if hasattr(item, 'get_effective_surface') else item.value
                        exact = bool(getattr(item, 'exact_disclosure', False))
                        if getattr(item, 'delivery_mode', '') in ('inferential', 'scattered'):
                            exact = False
                    result.append((identity, value, exact))
                    seen.add(identity)
                break
            else:
                raise ValueError(f'Missing required {source} key: {key}')
    return result


def build_turn_input(scenario, aim, messages, payload):
    # Personality is the declared speaking style, not the full hidden-state plot.
    style = scenario.personality or 'Natural and conversational'
    history = '\n'.join(f'User: {m.content}' for m in messages if isinstance(m, UserMessage))
    assistant = next((m.content for m in reversed(messages) if isinstance(m, AssistantMessage)), '')
    # No keyword coverage, planned disclosure, or assistant assertions become user facts.
    material = '\n'.join(f'- {value}' for _, value, _ in payload)
    intent = aim.description or ''
    context = (f'SPEAKER: {style}\n\n'
               f'CONTEXT — actual prior user messages:\n{history or "(first user turn)"}\n\n'
               f'THIS TURN:\n{intent}\n{material}\n\n'
               f'ASSISTANT — quotation to respond to, not established user facts:\n{assistant or "(no assistant message yet)"}\n\n'
               'BOUNDARIES: Preserve the people, timing, facts and intent in this material. '
               'Natural paraphrase, reported speech, proposals and the stated beliefs are allowed. '
               'Do not invent personal facts or future events. Assistant questions and examples '
               'do not establish user facts. Answer questions only from THIS TURN or actual prior '
               'user statements; if the requested details are absent, leave those questions '
               'unanswered and express the current intent without guessing or filling blanks. '
               'Leave requested judgments to the assistant.')
    # Fail explicitly instead of silently losing introductions or updates in long histories.
    if len(context) > 48000:
        raise ValueError('Turn input exceeds reviewed context budget; prepare a verified long-history checkpoint')
    return {'text': context, 'style': style, 'history': history, 'assistant': assistant,
            'generator_note': preference_generation_note(scenario, aim, messages),
            'intent': intent, 'truncated': False, 'characters': len(context)}


def behavior_instruction(aim, turn_input):
    mode = getattr(aim, 'aim_mode', 'disclosure')
    intent_type = getattr(getattr(aim, 'aim_type', None), 'value', '')
    if getattr(aim, 'is_final', False) or intent_type == 'probe':
        return ('Present only the current request naturally. Do not recap prior constraints, '
                'hint at a conflict, supply the requested judgment, or preface it with agreement.')
    modes = {
        'pushback': 'Express disagreement naturally, preserving the current facts.',
        'cooperative': 'Express agreement naturally, preserving the current facts.',
        'withhold': 'Deflect the question and express only the current intent.',
        'ambiguous': 'Preserve the current reference ambiguity; do not resolve it yourself.',
    }
    if mode in modes:
        return modes[mode]
    if not turn_input['assistant']:
        return 'Start the conversation directly; there is no prior assistant reply to acknowledge.'
    return 'Respond naturally to what the assistant actually said; acknowledgment is optional.'
