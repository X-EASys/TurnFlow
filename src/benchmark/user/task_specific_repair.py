"""Experimental exact-source GEN routing; production defaults are untouched.

Bindings are prepared offline, not inferred from generated output. The caller
must supply the source file's verified hash and full catalog key.
"""
from copy import deepcopy
import hashlib
import json

from benchmark.user.source_boundary import released_source_material
from benchmark.user.typed_generation_boundary import validated_current_source, BUDGET_TEMPLATE, _render_payload

REVISION = 'task-specific-repair-20261011-v1'
COMMON = """Write one natural USER message as the person requesting help. Express the
current source intent and public material, preserving people, quantities and
strength. Do not solve the task or supply the assistant's requested judgment.
Preserve plans, wishes, hypotheses, approximate values and completed actions as
such. Current explicit source updates take precedence over earlier source.
Dialogue is context for references and continuity, not independent permission
to add facts or commitments. Earlier allowed source is not a checklist or proof
that the assistant knows it; do not newly disclose omitted earlier material or
recap constraints to remind the assistant. Ordinary paraphrase, natural responses
and correct simple arithmetic remain allowed. Output only the user's message."""
S = {
'T17': """Cooperation is tone, not approval of a new decision. Without current source
authorization, do not agree to remove or replace features, reduce scope, add
personnel or change execution responsibility on the assistant's suggestion.
Feasibility discussion, noting a suggestion, ordinary implementation details and
source-authorized phasing remain allowed. Preserve qualifications such as 'if
possible'. A question about changing a requirement is not an approved change.""",
'T20': """Cooperation is tone, not approval of a new decision. Without current source
authorization, do not adopt suggested changes to shipping frequency, services,
channels, delivery time or workload. Distinguish frequency of changing product
varieties from frequency of deliveries. Discussion and source-authorized updates
remain allowed; do not restate all earlier requirements.""",
'T18': BUDGET_TEMPLATE,
}
F = {
'T17': """Your task is to express the current material (or current request on a final
turn). Responding to the assistant is optional; you need not answer every
question. Answer naturally when source supports it; otherwise you may skip the
question and continue the intent. Do not fill in project funds, personnel,
experience, authority or completed preparation. Missing information is neither
'no' nor 'I do not know'. Do not invent negative facts, a knowledge state, or
commitments changing resources, permissions or task scope. Ordinary prospective
preparation that does not alter requirements remains allowed.""",
'T20': """Your task is to express the current material (or current request on a final
turn). Responding to the assistant is optional; you need not answer every
question. Answer naturally when source supports it; otherwise you may skip the
question and continue the intent. Do not assign business/personal background,
country, capacity, qualifications, existing orders or experience from assistant
examples. Missing information is neither 'no' nor 'I do not know'. Do not invent
negative facts, a knowledge state, or commitments changing resources, permissions
or task scope. Preserve source-authorized completed purchases and subscriptions
as completed. Ordinary prospective preparation remains allowed.""",
}
C = """Express this self-contained current material naturally as the user. Preserve
its meaning, people, quantities, qualifications and action status. Do not add
personal circumstances, negative facts, knowledge states or decisions to make it
sound complete. No fixed transition or statement of ignorance is required."""


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def render(scenario, *, source_key, source_sha256, stage, aim, history,
           baseline, arm, registrations):
    """No-op outside an exact registered task/source/scenario/aim/role/arm."""
    result = deepcopy(baseline)
    task = source_key.split('/')[0]
    if stage != 'GENERATE' or task not in {'T17', 'T18', 'T20'} or arm == 'B0':
        return result
    matches = [r for r in registrations if r['source_key'] == source_key
               and r['source_sha256'] == source_sha256
               and r['scenario_sha256'] == digest(scenario.model_dump(mode='json'))
               and r['aim_id'] == getattr(aim, 'aim_id', None) and arm in r['arms']]
    if not matches:
        return result
    if len(matches) != 1:
        raise ValueError('ambiguous_task_repair_registration')
    bound = matches[0]
    role = 'final' if aim.is_final else 'material'
    if bound['task'] != task or bound['turn_role'] != role:
        raise ValueError('task_repair_role_mismatch')
    current = validated_current_source(scenario, aim, baseline['messages'])
    if arm == 'C' and (not bound['self_contained'] or role != 'material' or task == 'T18'):
        raise ValueError('task_repair_ineligible_C')
    if arm not in {'S', 'F', 'C', 'FS'} or (task == 'T18' and arm != 'S'):
        raise ValueError('task_repair_invalid_arm')
    # Style is source metadata, not generated speech or assistant inference.
    style = bound['speaker']
    notes = []
    if arm in {'S', 'FS'}:
        notes.append(S[task].format(current_intent=current['current_intent'],
                                   current_payload=_render_payload(current['current_payload'])))
    if arm in {'F', 'FS'}:
        notes.append(F[task])
    if arm == 'C':
        notes.append(C)
    content = f"TURN ROLE: {role}\nSPEAKER: {style}\n\n" + '\n\n'.join(notes)
    content += '\n\nCURRENT SOURCE:\n' + json.dumps(current, ensure_ascii=False)
    if arm != 'C':
        released = {r['aim_id']: r for r in released_source_material(scenario, aim, history)}
        references = []
        for pointer in bound['dependencies']:
            if pointer['aim_id'] == 'OPENING':
                anchors = [m for m in history if m.role == 'user' and m.content.strip()
                           and (m.metadata or {}).get('is_anchor')
                           and (m.metadata or {}).get('anchor_guidance') == scenario.state_queue.anchor]
                if not anchors:
                    raise ValueError('task_repair_opening_not_captured')
                references.append(dict(aim_id='OPENING', source_intent=scenario.state_queue.anchor, source_payload=[]))
                continue
            if pointer['aim_id'] not in released:
                raise ValueError('task_repair_dependency_not_emitted')
            entry = released[pointer['aim_id']]
            keys = pointer['payload_keys']
            payload = [p for p in entry['source_payload'] if p[0] in keys]
            if set(keys) != {p[0] for p in payload}:
                raise ValueError('task_repair_dependency_not_public')
            references.append(dict(aim_id=entry['aim_id'],
                source_intent=entry['source_intent'] if pointer['include_intent'] else None,
                source_payload=payload))
        content += '\n\nEARLIER ALLOWED SOURCE (not necessarily expressed; reference only):\n' + json.dumps(references, ensure_ascii=False)
        content += '\n\nACTUAL DIALOGUE CONTEXT:\n' + json.dumps([
            {'role': m.role, 'content': m.content} for m in history], ensure_ascii=False)
    content += '\nOutput only the user message.'
    result['messages'] = [{'role': 'system', 'content': COMMON}, {'role': 'user', 'content': content}]
    return result
