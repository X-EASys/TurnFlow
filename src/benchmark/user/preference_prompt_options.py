"""Frozen opt-in preference-final prompt alternatives; no production imports.

Every candidate uses the same source-bound variables. O1/O2 retain captured
reference text. O3 is explicitly an input ablation: O2's final-only contract
without prior-user history or the latest assistant quotation in its generation
request. Actual stored conversations are untouched. None adds a model call,
classifies output or retries generation.
"""
from __future__ import annotations

from copy import deepcopy

from benchmark.user.typed_generation_boundary import (
    PREFERENCE_FINAL_TEMPLATE,
    _render_payload,
    bind_preference_source,
    build_typed_messages,
    validated_current_source,
)


OPTIONS = ("B0", "T1", "O1", "O2", "O3")

O1_TEMPLATE = """PRESERVE THE OPEN SCOPE OF THIS REQUEST:
The current source below is complete. Express that request naturally, preserving
what it leaves unspecified. If it asks where, how, when, or which, keep that
choice open; do not fill it with a location, method, time, or choice from history.
Use historical messages only to understand necessary references, not to supply
additional conditions. Keep every requirement that the current source actually
states, at its original strength.
The following previously expressed preference is historical context, not a
condition to add to this request:
{preference_source}
Do not repeat it, refer to the user's usual approach, or use it to narrow the
request. Do not retract the preference, announce a conflict, or suggest a solution.

Complete current request source:
{current_intent}
Current public payload:
{current_payload}"""

O2_SYSTEM = """Write one natural message as the user making a final request.
The CURRENT SPEECH SOURCE fully specifies what to express. Preserve its people,
facts, uncertainty, requirements and open choices. Use any supplied earlier
dialogue only to understand necessary references; it cannot supply requirements.
Assistant questions and suggestions do not create another task for this turn.
Follow SPEAKER STYLE in phrasing only. Do not solve the task, explain a conflict,
withdraw a preference, recap history, or include these generation instructions.
Output only the user's message."""

O2_TEMPLATE = """SPEAKER STYLE:
{speaker}

REFERENCE ONLY — actual prior user messages, retained verbatim:
{history}

REFERENCE ONLY — latest assistant quotation, retained verbatim:
{assistant}

HISTORICAL PREFERENCE — excluded from the current request's content and scope:
{preference_source}

CURRENT SPEECH SOURCE — complete content to express:
{current_intent}
Current public payload:
{current_payload}

Make the request naturally. Keep unspecified choices unspecified. Do not add
the historical preference directly, indirectly, or as a restriction on an open
choice. Preserve restrictions explicitly stated in the current speech source.
Output only the user's message."""

# This pair differs from O2 only by omitting two reference blocks. It does not
# manufacture a resolved-reference summary or replace source pronouns. Whether
# that omission is acceptable is an experimental question, not assumed here.
O3_SYSTEM = O2_SYSTEM
O3_TEMPLATE = O2_TEMPLATE.replace(
    "REFERENCE ONLY — actual prior user messages, retained verbatim:\n"
    "{history}\n\n"
    "REFERENCE ONLY — latest assistant quotation, retained verbatim:\n"
    "{assistant}\n\n", "",
)

_SPEAKER = "SPEAKER: "
_HISTORY = "\n\nCONTEXT — actual prior user messages:\n"
_CURRENT = "\n\nTHIS TURN:\n"
_ASSISTANT = "\n\nASSISTANT — quotation to respond to, not established user facts:\n"
_BOUNDARIES = "\n\nBOUNDARIES: "


def _captured_context(content):
    """Extract captured reference values without inventing a history summary."""
    markers = (_SPEAKER, _HISTORY, _CURRENT, _ASSISTANT, _BOUNDARIES)
    if any(content.count(marker) != 1 for marker in markers):
        raise ValueError("preference_options_ambiguous_context_capture")
    offsets = [content.index(marker) for marker in markers]
    if offsets != sorted(offsets):
        raise ValueError("preference_options_misordered_context_capture")
    return {
        "speaker": content[offsets[0] + len(_SPEAKER):offsets[1]],
        "history": content[offsets[1] + len(_HISTORY):offsets[2]],
        "assistant": content[offsets[3] + len(_ASSISTANT):offsets[4]],
    }


def build_preference_option(
    baseline_messages, *, option, current_intent, current_payload,
    preference_source=None, old_preference_note="", active=True,
):
    """Render frozen B0/T1/O1/O2/O3 with validated original-source variables.

    Activation is a preregistered input binding, never a generated semantic
    decision. Nonfinal turns and explicit source-authorized preference updates
    use ``active=False`` and preserve the original request exactly. Use the
    source-binding wrapper below rather than supplying unverified variables.
    """
    if option not in OPTIONS:
        raise ValueError("preference_options_unknown_option")
    if not isinstance(active, bool):
        raise ValueError("preference_options_activation_must_be_boolean")
    if option == "B0" or not active:
        return deepcopy(baseline_messages)
    # Reuse T1's exact current-capture check and original P1 removal validation.
    result = build_typed_messages(
        baseline_messages, kind="preference_final", current_intent=current_intent,
        current_payload=current_payload, preference_source=preference_source,
        old_preference_note=old_preference_note,
    )
    if option == "T1":
        return result
    variables = {
        "current_intent": current_intent,
        "current_payload": _render_payload(current_payload) or "(no separate payload)",
        "preference_source": preference_source,
    }
    if option in {"O2", "O3"}:
        context = _captured_context(baseline_messages[1]["content"])
        result[0]["content"] = O2_SYSTEM
        template = O2_TEMPLATE if option == "O2" else O3_TEMPLATE
        result[1]["content"] = template.format(**variables, **context)
        return result
    old_note = PREFERENCE_FINAL_TEMPLATE.format(**variables)
    if result[1]["content"].count(old_note) != 1:
        raise ValueError("preference_options_ambiguous_t1_note")
    result[1]["content"] = result[1]["content"].replace(
        old_note, O1_TEMPLATE.format(**variables), 1,
    )
    return result


def build_candidate_messages(
    scenario, aim, messages, variant="B0", baseline_messages=None, *, active=True,
):
    """Bind original intent/payload/released preference, then render a candidate."""
    if baseline_messages is None:
        raise ValueError("preference_options_requires_captured_baseline")
    if variant not in OPTIONS:
        raise ValueError("preference_options_unknown_option")
    if not isinstance(active, bool):
        raise ValueError("preference_options_activation_must_be_boolean")
    if variant == "B0" or not active:
        return deepcopy(baseline_messages)
    source = validated_current_source(scenario, aim, baseline_messages)
    preference, note = bind_preference_source(scenario, aim, messages)
    return build_preference_option(
        baseline_messages, option=variant, **source,
        preference_source=preference, old_preference_note=note,
    )
