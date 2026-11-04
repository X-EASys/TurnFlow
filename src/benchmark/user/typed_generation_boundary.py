"""Opt-in fixed generation templates with source-bound scenario variables.

These candidates are for frozen-input experiments only. They do not change the
production simulator, inspect generated answers, add a judge or retry a model.
The experiment binds a turn to a kind before generation; there is deliberately
no runtime semantic classifier. Every variable is original source material,
not an agent-written instruction or an inferred answer.
"""
from __future__ import annotations

from copy import deepcopy

from benchmark.data_model.message import UserMessage
from benchmark.user.source_boundary import (
    _public_turn_block,
    _source_aims,
    _validated_public_material,
)
from benchmark.user.turn_input import preference_generation_note


BUDGET_TEMPLATE = """CURRENT FINANCIAL MATERIAL — GENERATION INSTRUCTION, NOT NEW SPEECH:
Perform the current source intent below. Express the supplied financial amounts,
who contributes them, their timing, their original degree of certainty, and any
supplied circumstances naturally. Preserve distinctions such as a maximum,
an approximate amount, and an explicit willingness or decision.
If the current source asks for a choice or a budget update, express that exact
choice or update, even when it conflicts with earlier material. Do not replace
it with what you think is a better or more affordable choice.
Do not add willingness to contribute extra money, stretch a budget, accept an
overrun, or regard a more expensive choice as affordable unless that commitment
is expressly supplied in the current source. Do not turn an approximate amount
into a strict cap. Do not invent a financial justification, personal motivation,
or trade-off to make the statement sound complete.
An assistant question can guide a natural transition but does not authorize
an extra commitment, choice, or reason. Acknowledge it briefly if useful and
then express the current intent. Ordinary paraphrase and correct arithmetic
are allowed; do not add an unsolicited solution or signal a hidden conflict.

Current source intent:
{current_intent}
Current public payload:
{current_payload}"""

PREFERENCE_FINAL_TEMPLATE = """FINAL OPEN REQUEST — GENERATION INSTRUCTION, NOT NEW SPEECH:
Express only the current source request and its supplied details below, in
natural conversational language.
The original previously released preference is:
{preference_source}

Do not mention that preference in this final request. Do not repeat it directly,
paraphrase it, or indirectly import it through expressions such as the user's
usual way or prior approach. In particular, do not turn the preference into a
required method, channel, location, timing, payment method, or other restriction
on the current request. The earlier preference remains in the conversation;
there is no need to restate it to make the request natural or complete.
Do not announce that the preference is withdrawn or wrong, identify an exception,
hint at a conflict, suggest an alternative, or supply the assistant's answer.
This boundary does not erase requirements explicitly present in the current
source request. Preserve those requirements at their original strength.

Current source intent:
{current_intent}
Current public payload:
{current_payload}"""

_CURRENT_PREFERENCE_HEADER = "PREFERENCE ALREADY EXPRESSED — GENERATION BOUNDARY, NOT NEW SPEECH:"
_OUTPUT_SUFFIX = "\nOutput only the user's message."
_KINDS = {"budget_material", "preference_final"}


def _frozen_messages(messages):
    if (not isinstance(messages, list) or len(messages) != 2
            or messages[0].get("role") != "system"
            or messages[1].get("role") != "user"
            or not all(isinstance(m.get("content"), str) for m in messages)):
        raise ValueError("typed_boundary_requires_frozen_v3_messages")


def validated_current_source(scenario, aim, baseline_messages):
    """Bind original intent and only the captured, authorized public subset.

    This never inserts the full constraint snapshot, derived values, hidden
    state, future aims, success criteria or ground truth. Captured public
    material must match the registered source before it becomes a variable.
    """
    _frozen_messages(baseline_messages)
    aims, current = _source_aims(scenario, aim)
    if current is None:
        raise ValueError("typed_boundary_requires_source_aim")
    source_aim = aims[current]
    return {
        "current_intent": source_aim.description or "",
        "current_payload": _validated_public_material(
            scenario, source_aim, baseline_messages[1]["content"]
        ),
    }


def bind_preference_source(scenario, aim, history):
    """Return (original previously released preference, existing P1 note).

    No preference is released merely because it exists somewhere in a scenario.
    The source aim must precede this aim, and an emitted user message must have
    captured the corresponding original source intent/public payload. Actual
    generated wording is not promoted to a source variable. Missing bindings or
    unreached source turns return no anchor; corrupt source captures raise.
    """
    aims, current = _source_aims(scenario, aim)
    config = (scenario.metadata or {}).get("simulator_no_repeat")
    if current is None or not config:
        return None, ""
    source_id = config.get("preference_source_aim_id")
    preference = config.get("preference_text")
    indices = [i for i, item in enumerate(aims) if item.aim_id == source_id]
    if (len(indices) != 1 or not isinstance(preference, str)
            or not preference.strip()):
        raise ValueError("typed_boundary_incomplete_preference_binding")
    source_index = indices[0]
    source_aim = aims[source_index]
    if preference != source_aim.description:
        raise ValueError("typed_boundary_preference_source_mismatch")
    if current <= source_index:
        return None, ""
    released = False
    for message in history:
        metadata = getattr(message, "metadata", None) or {}
        if (not isinstance(message, UserMessage) or not message.content.strip()
                or metadata.get("aim_id") != source_id
                or metadata.get("delivery_status") != "emitted"):
            continue
        captured = metadata.get("turn_input")
        if (not isinstance(captured, dict)
                or captured.get("intent") != source_aim.description
                or not isinstance(captured.get("text"), str)):
            raise ValueError("typed_boundary_preference_capture_missing_or_mismatched")
        _validated_public_material(scenario, source_aim, captured["text"])
        released = True
    if not released:
        return None, ""
    return preference, preference_generation_note(scenario, aim, history)


def _render_payload(payload):
    if not isinstance(payload, (list, tuple)):
        raise ValueError("typed_boundary_invalid_current_payload")
    values = []
    for item in payload:
        if (not isinstance(item, (list, tuple)) or len(item) != 3
                or not isinstance(item[0], str) or not isinstance(item[1], str)
                or not isinstance(item[2], bool)):
            raise ValueError("typed_boundary_invalid_current_payload")
        values.append(f"- {item[1]}")
    return "\n".join(values)


def build_typed_messages(
    baseline_messages,
    *,
    kind,
    current_intent,
    current_payload,
    active=True,
    preference_source=None,
    old_preference_note="",
):
    """Apply one fixed template without changing the captured history.

    Use ``validated_current_source`` and ``bind_preference_source`` to obtain
    the variables. The low-level builder additionally requires the intent and
    payload to reconstruct the exact captured THIS TURN block. It accepts no
    free-form case-specific guidance.

    ``active`` is a preregistered binding, not a semantic prediction. In the
    preference-final experiment it is true only for a final open request that
    does not itself restate or update the preference. A source-authorized
    restatement/update or a nonfinal turn is an exact no-op. The budget template
    remains applicable to an explicit current choice/update and preserves it.
    """
    if kind not in _KINDS:
        raise ValueError("typed_boundary_unknown_kind")
    if not isinstance(active, bool):
        raise ValueError("typed_boundary_activation_must_be_boolean")
    result = deepcopy(baseline_messages)
    if not active:
        return result
    _frozen_messages(result)
    if not isinstance(current_intent, str):
        raise ValueError("typed_boundary_invalid_current_intent")
    material = _render_payload(current_payload)
    expected = current_intent + "\n" + material
    if _public_turn_block(result[1]["content"]) != expected:
        raise ValueError("typed_boundary_current_source_capture_mismatch")
    if not result[1]["content"].endswith(_OUTPUT_SUFFIX):
        raise ValueError("typed_boundary_unrecognized_output_suffix")
    variables = {
        "current_intent": current_intent,
        "current_payload": material or "(no separate payload)",
    }
    if kind == "budget_material":
        if preference_source is not None or old_preference_note:
            raise ValueError("typed_boundary_budget_does_not_accept_preference_variables")
        note = BUDGET_TEMPLATE.format(**variables)
    else:
        if not isinstance(preference_source, str) or not preference_source.strip():
            raise ValueError("typed_boundary_preference_requires_released_source")
        content = result[1]["content"]
        if old_preference_note:
            old = "\n\n" + old_preference_note
            if (not old_preference_note.startswith(_CURRENT_PREFERENCE_HEADER)
                    or content.count(old) != 1):
                raise ValueError("typed_boundary_unrecognized_existing_preference_note")
            content = content.replace(old, "", 1)
        if _CURRENT_PREFERENCE_HEADER in content:
            raise ValueError("typed_boundary_existing_preference_note_not_removed")
        result[1]["content"] = content
        note = PREFERENCE_FINAL_TEMPLATE.format(
            **variables, preference_source=preference_source
        )
    result[1]["content"] = (
        result[1]["content"][:-len(_OUTPUT_SUFFIX)] + "\n\n" + note + _OUTPUT_SUFFIX
    )
    return result
