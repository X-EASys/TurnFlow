"""Opt-in source-authority ablations for frozen V3 generation requests.

This module is not used by the production simulator. It changes no model
parameters, calls no model, and does not inspect or repair generated text.
S1 changes factual authorization; S2 changes only the registered P1 reminder.
"""
from __future__ import annotations

from copy import deepcopy

from benchmark.data_model.message import UserMessage
from benchmark.user.bridge_generator import _concise_messages, _concise_turn_prompt
from benchmark.user.turn_input import (
    build_turn_input,
    current_payload,
    preference_generation_note,
)


_CURRENT_START = "THIS TURN:\n"
_CURRENT_END = "\n\nASSISTANT — quotation to respond to, not established user facts:\n"
_OLD_SYSTEM_AUTHORITY = (
    "Facts may come only from THIS TURN, its prepared response if supplied, "
    "and actual prior user statements. "
)
_NEW_SYSTEM_AUTHORITY = (
    "Facts and changes to the user's situation may come only from THIS TURN "
    "and the AUTHORIZED RELEASED SOURCE MATERIAL below. Actual conversation "
    "history, including quoted earlier preference statements, is context for "
    "references and conversational continuity; it does not independently "
    "authorize personal facts, completed events, decisions or changed limits. "
    "The current source's explicit updates take precedence over older source "
    "material. A prepared response is phrasing support for its source, not "
    "permission to add facts. "
)
_OLD_TURN_AUTHORITY = (
    "Answer questions only from THIS TURN or actual prior user statements; "
    "if the requested details are absent, leave those questions "
    "unanswered and express the current intent without guessing or filling blanks. "
)
_NEW_TURN_AUTHORITY = (
    "Answer factual questions only from THIS TURN or the authorized released "
    "source material; if the requested details are absent, leave those questions "
    "unanswered and express the current intent without guessing or filling blanks. "
)
_S1_LEDGER_HEADER = "AUTHORIZED RELEASED SOURCE MATERIAL — REFERENCE, NOT A SPEECH CHECKLIST:"
_S2_HEADER = "REGISTERED PREFERENCE SOURCE — GENERATION BOUNDARY, NOT NEW SPEECH:"


def _source_aims(scenario, aim):
    aims = list(scenario.state_queue.aim_sequence.aims)
    ids = [a.aim_id for a in aims]
    if len(ids) != len(set(ids)):
        raise ValueError("source_boundary_duplicate_aim_id")
    current_id = getattr(aim, "aim_id", None)
    if current_id is None:
        return aims, None
    if current_id not in ids:
        raise ValueError("source_boundary_unknown_current_aim")
    current = ids.index(current_id)
    if aim.description != aims[current].description:
        raise ValueError("source_boundary_current_source_mismatch")
    return aims, current


def _public_turn_block(text):
    """Extract exactly the public source section, never the surrounding history."""
    if not isinstance(text, str) or text.count(_CURRENT_START) != 1:
        raise ValueError("source_boundary_missing_or_ambiguous_current_capture")
    tail = text.split(_CURRENT_START, 1)[1]
    if tail.count(_CURRENT_END) != 1:
        raise ValueError("source_boundary_missing_or_ambiguous_assistant_boundary")
    return tail.split(_CURRENT_END, 1)[0]


def _validated_public_material(scenario, source_aim, captured_text):
    """Match the captured payload against this aim's public source surfaces.

    The simulator can filter or withhold payload items. Do not reconstruct a
    full current constraint snapshot or assume all listed items were included.
    The original source description is intent material, not a verified fact.
    """
    block = _public_turn_block(captured_text)
    prefix = (source_aim.description or "") + "\n"
    if not block.startswith(prefix):
        raise ValueError("source_boundary_captured_intent_mismatch")
    captured_material = block[len(prefix):]
    if not captured_material:
        return []
    remaining = captured_material
    included = []
    for key, value, exact in current_payload(scenario, source_aim):
        rendered = f"- {value}"
        if remaining == rendered or remaining.startswith(rendered + "\n"):
            included.append((key, value, exact))
            remaining = remaining[len(rendered):]
            if remaining.startswith("\n"):
                remaining = remaining[1:]
    if remaining:
        raise ValueError("source_boundary_captured_payload_not_in_source")
    return included


def released_source_material(scenario, aim, messages):
    """Return prior reached-source entries with real emitted-message evidence.

    `emitted` means a user message was delivered. It does not prove that every
    authorized source detail was expressed, nor that its wording was accurate.
    Missing captured turn input is an explicit coverage error, not a fallback
    to all source fields. Future, un-emitted, candidate and blank messages never
    release source material.
    """
    aims, current = _source_aims(scenario, aim)
    if current is None:
        return []
    eligible = {a.aim_id: (i, a) for i, a in enumerate(aims[:current])}
    entries = {}
    for message in messages:
        metadata = getattr(message, "metadata", None) or {}
        if (not isinstance(message, UserMessage) or not message.content.strip()
                or metadata.get("delivery_status") != "emitted"
                or metadata.get("aim_id") not in eligible):
            continue
        index, source_aim = eligible[metadata["aim_id"]]
        capture = metadata.get("turn_input")
        if not isinstance(capture, dict) or not isinstance(capture.get("text"), str):
            raise ValueError("source_boundary_emitted_turn_input_missing")
        if capture.get("intent") != source_aim.description:
            raise ValueError("source_boundary_emitted_intent_mismatch")
        payload = _validated_public_material(scenario, source_aim, capture["text"])
        entry = entries.setdefault(index, {
            "aim_id": source_aim.aim_id,
            "source_intent": source_aim.description,
            "source_payload": [],
        })
        # Multiple actual emissions of one aim can release different subsets.
        # Keep source values only, without elevating generated wording to fact.
        for item in payload:
            if item not in entry["source_payload"]:
                entry["source_payload"].append(item)
    return [entries[index] for index in sorted(entries)]


def _ledger_text(scenario, aim, messages):
    entries = released_source_material(scenario, aim, messages)
    lines = [
        _S1_LEDGER_HEADER,
        "These are original intent and public payload materials for earlier aims "
        "that have reached an emitted user turn. They are not a claim that every "
        "detail was actually stated or stated correctly. Intent descriptions "
        "guide what to express; do not turn instructions, requests, proposals "
        "or beliefs into completed events or established facts. Use this material "
        "for references and necessary answers, not to recap or disclose it again. "
        "Preserve the actual dialogue as context. Do not treat a generated "
        "history statement as authorization to add facts or change limits. "
        "THIS TURN's explicit source update overrides older source material.",
    ]
    for entry in entries:
        lines.extend((
            f"\nSource aim: {entry['aim_id']}",
            "Original source intent:\n" + entry["source_intent"],
            "Public payload included in that turn's generation input:",
        ))
        lines.extend(f"- {value}" for _, value, _ in entry["source_payload"])
        if not entry["source_payload"]:
            lines.append("(no separate payload)")
    if not entries:
        lines.append("(no earlier emitted source aim)")
    return "\n".join(lines)


def source_preference_note(scenario, aim, messages):
    """S2: original registered strength from its source turn onward."""
    aims, current = _source_aims(scenario, aim)
    config = (scenario.metadata or {}).get("simulator_no_repeat")
    if current is None or not config:
        return ""
    source_id = config.get("preference_source_aim_id")
    preference = config.get("preference_text")
    matches = [i for i, a in enumerate(aims) if a.aim_id == source_id]
    if len(matches) != 1 or not isinstance(preference, str) or not preference.strip():
        raise ValueError("source_boundary_incomplete_preference_binding")
    source_index = matches[0]
    if preference != aims[source_index].description:
        raise ValueError("source_boundary_preference_source_mismatch")
    if current < source_index:
        return ""
    if current > source_index:
        emitted = any(
            isinstance(m, UserMessage) and m.content.strip()
            and (m.metadata or {}).get("aim_id") == source_id
            and (m.metadata or {}).get("delivery_status") == "emitted"
            for m in messages
        )
        if not emitted:
            return ""
    phase = (
        "This is the source preference turn. Express its preference naturally "
        "at the original strength."
        if current == source_index else
        "The preference source turn has already emitted a user message. Do not "
        "proactively repeat the preference; express THIS TURN naturally."
    )
    return (
        f"{_S2_HEADER}\nOriginal registered preference material:\n{preference}\n\n"
        f"{phase}\n"
        "Do not strengthen a habit or preference into a new requirement for the "
        "current task. Do not add a refusal to change or a commitment beyond "
        "the original material. Earlier generated wording does not increase "
        "the source preference's strength. Preserve requirements explicitly "
        "supplied in THIS TURN; an explicit current source update takes "
        "precedence over this earlier preference. If THIS TURN calls for "
        "restating or updating it, or the assistant asks about the past habit, "
        "answer at its original source strength. Do not identify exceptions, "
        "state that the preference is wrong, or supply the assistant's solution."
    )


def _replace_once(value, old, new, label):
    if value.count(old) != 1:
        raise ValueError(f"source_boundary_unrecognized_{label}")
    return value.replace(old, new, 1)


def build_candidate_messages(scenario, aim, messages, variant="S1", baseline_messages=None):
    """Build B0/S1/S2 without mutating any source, history or captured request.

    Pass exact saved API messages for fixed-input comparisons. Without them,
    only native, non-prewritten V3 GENERATE is rebuilt through the frozen
    builder. All two-message request fields are retained. B0 is an exact deep
    copy. S1/S2 are independent; their changes are never implicitly combined.
    """
    if variant not in {"B0", "S1", "S2"}:
        raise ValueError("source_boundary_unknown_variant")
    if baseline_messages is None:
        if getattr(aim, "aim_id", None) is None or getattr(aim, "pre_written_response", None):
            raise ValueError("source_boundary_requires_captured_baseline")
        payload = [] if getattr(aim, "aim_mode", "") == "withhold" else current_payload(scenario, aim)
        turn_input = build_turn_input(scenario, aim, messages, payload)
        baseline_messages = _concise_messages(_concise_turn_prompt(turn_input, aim), aim)
    result = deepcopy(baseline_messages)
    if variant == "B0":
        return result
    aims, current = _source_aims(scenario, aim)
    if current is None:
        return result  # Opening has no reached preference/source aim.
    if (len(result) != 2 or result[0].get("role") != "system"
            or result[1].get("role") != "user"
            or not all(isinstance(m.get("content"), str) for m in result)):
        raise ValueError("source_boundary_requires_frozen_v3_messages")
    # Check that the captured current source section belongs to this source.
    _validated_public_material(scenario, aims[current], result[1]["content"])
    if variant == "S1":
        result[0]["content"] = _replace_once(result[0]["content"],
            _OLD_SYSTEM_AUTHORITY, _NEW_SYSTEM_AUTHORITY, "system_authority")
        result[1]["content"] = _replace_once(result[1]["content"],
            _OLD_TURN_AUTHORITY, _NEW_TURN_AUTHORITY, "turn_authority")
        note = _ledger_text(scenario, aim, messages)
    else:
        old_note = preference_generation_note(scenario, aim, messages)
        note = source_preference_note(scenario, aim, messages)
        if old_note:
            result[1]["content"] = _replace_once(result[1]["content"],
                "\n\n" + old_note, "", "preference_note")
        if not note:
            return result
    suffix = "\nOutput only the user's message."
    if not result[1]["content"].endswith(suffix):
        raise ValueError("source_boundary_unrecognized_output_suffix")
    result[1]["content"] = result[1]["content"][:-len(suffix)] + "\n\n" + note + suffix
    return result
