"""Source-bound, opt-in midturn experiment. No production hook or output guard."""
from copy import deepcopy

from benchmark.user.preference_prompt_options import _captured_context
from benchmark.user.typed_generation_boundary import (
    _render_payload, bind_preference_source, validated_current_source,
)
from benchmark.user.source_boundary import _source_aims

SYSTEM = """Write one natural message as the user seeking help.
CURRENT SOURCE is the authority for this turn's new facts, choices and requirements.
Preserve its people, amounts, timing, uncertainty and degree of commitment.
Any supplied dialogue is reference context, not authority to change the current
source. Follow SPEAKER STYLE in phrasing only. A brief natural acknowledgment is
optional; it must not replace the current message or add a current requirement.
Do not correct the user's stated belief, reveal a hidden conflict, invent personal
facts, or supply a solution to the current task. Output only the user's message."""

ANCHOR = """THIS TURN'S ROLE: disclose the supplied habit or belief.
Express the original preference at its original strength. Do not soften it by
adding willingness to change, exceptions or a retraction. Equally, do not extend
the habit into a new instruction that the current task must follow it, a refusal
to consider other approaches, or a demand for a plan restricted to that habit.
Such a present-task requirement is allowed only if CURRENT SOURCE itself states
it. You may acknowledge the preceding reply, then disclose this habit; you do
not need to add another request to make the message complete."""

MATERIAL = """THIS TURN'S ROLE: disclose the supplied item or situation.
Preserve the current facts and any explicit conditions. Do not turn this disclosure
into a request restricted by the earlier preference below. Do not repeat that
preference, invoke the usual approach, or add its method, location or timing as a
condition on this item or situation. This does not remove any condition actually
stated in CURRENT SOURCE. Briefly acknowledging earlier advice is optional; it
must not add a condition to this turn. Leave open choices open.

ORIGINAL EARLIER PREFERENCE — not an additional condition on this disclosure:
{preference_source}"""

REFERENCE = """REFERENCE ONLY — actual prior user messages, retained verbatim:
{history}

REFERENCE ONLY — latest assistant quotation, retained verbatim:
{assistant}

"""

TEMPLATE = """SPEAKER STYLE:
{speaker}

{reference}{role_instruction}

CURRENT SOURCE — complete content to express this turn:
{current_intent}
Current public payload:
{current_payload}

Output only the user's message."""


def build_midturn_messages(scenario, aim, history, baseline_messages, *, role, arm):
    """Experiment binds role before calls; variables are validated original text.

    R1 and R2 differ only in the two reference blocks. All source people and
    pronouns remain untouched; no inferred reference summary is synthesized.
    R2 is an input ablation, not a claim that references never require history.
    """
    if arm not in {'B0','R1','R2'} or role not in {'anchor','material'}:
        raise ValueError('unknown_midturn_arm_or_role')
    source = validated_current_source(scenario, aim, baseline_messages)
    aims, current = _source_aims(scenario, aim)
    binding = (scenario.metadata or {}).get('simulator_no_repeat') or {}
    anchor_id, preference = binding.get('preference_source_aim_id'), binding.get('preference_text')
    anchors = [a for a in aims if a.aim_id == anchor_id]
    if len(anchors)!=1 or preference != anchors[0].description:
        raise ValueError('original_preference_binding_mismatch')
    if aim.is_final or current is None:
        raise ValueError('requires_nonfinal_source')
    if role == 'anchor':
        if aim.aim_id != anchor_id:
            raise ValueError('anchor_role_requires_original_anchor')
        instruction = ANCHOR
    else:
        released, _ = bind_preference_source(scenario, aim, history)
        if released != preference:
            raise ValueError('material_requires_source_captured_released_preference')
        instruction = MATERIAL.format(preference_source=preference)
    if arm == 'B0':
        return deepcopy(baseline_messages)
    context = _captured_context(baseline_messages[1]['content'])
    content = TEMPLATE.format(speaker=context['speaker'],
        reference=REFERENCE.format(**context) if arm=='R1' else '',
        role_instruction=instruction, current_intent=source['current_intent'],
        current_payload=_render_payload(source['current_payload']) or '(no separate payload)')
    return [{'role':'system','content':SYSTEM},{'role':'user','content':content}]
