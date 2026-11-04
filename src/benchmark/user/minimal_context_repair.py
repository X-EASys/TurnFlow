"""Offline-bound D/R GEN candidates; no production route or model calls.

D adds a deferral boundary to S17. R supplies current public source plus only
registered earlier source references, without generated dialogue. Registrations
use the task_specific_repair shape; R does not require ``self_contained``.
Legacy arms delegate to their original renderer without rewriting the request.
"""
from __future__ import annotations

from copy import deepcopy
import json

from benchmark.data_model.message import UserMessage
from benchmark.user import task_specific_repair as legacy
from benchmark.user.source_boundary import released_source_material
from benchmark.user.typed_generation_boundary import validated_current_source


REVISION = "minimal-context-repair-20261011-v1"

D = legacy.S["T17"] + """

Deferring an existing requested feature also changes its scope or timing. Unless
the current source authorizes that change, do not agree to postpone it, park it,
leave it for a later phase, revisit it someday, or use a manual substitute now
with automation only later. Keeping a future wish does not authorize removing
the feature from the requested plan. You may acknowledge or discuss a suggestion
without accepting the change on the user's behalf.
This does not make every feature mandatory or due immediately. Preserve the
source's own optionality, 'if possible', ordering, phases and explicit choices.
When the current source itself authorizes a deferral or replacement, express it
at its stated strength rather than restoring the earlier requirement."""

R = """Express the current material naturally as the user. The selected earlier
source references are internal support for necessary referents and dependencies,
not a speech checklist or a claim that every detail was previously said. Use
only the referenced original fields needed to interpret the current material;
do not recap unrelated earlier information or newly disclose omitted details.
Do not invent personal circumstances, completed preparation, negative facts,
knowledge states or decisions to make the message sound complete. Missing
information does not mean 'no' or 'I do not know'. Preserve explicit current
updates, tentative ideas, ordinary implementation proposals and correct simple
arithmetic. No fixed transition or statement of ignorance is required."""


def _references(scenario, aim, history, dependencies):
    """Select original fields only after validating real emission/capture.

    An emitted turn releases its captured public subset for reference; it does
    not prove the assistant heard every field. Unrelated history is neither
    serialized nor required to validate these independently bound references.
    """
    source_ids = {p["aim_id"] for p in dependencies if p["aim_id"] != "OPENING"}
    relevant_history = [
        message for message in history
        if (getattr(message, "metadata", None) or {}).get("aim_id") in source_ids
    ]
    released = {
        entry["aim_id"]: entry
        for entry in released_source_material(scenario, aim, relevant_history)
    } if source_ids else {}
    references = []
    for pointer in dependencies:
        if pointer["aim_id"] == "OPENING":
            # Frozen production openings predate delivery_status; their actual
            # UserMessage, is_anchor and matching capture identify the emission.
            # An explicit candidate/withheld status never qualifies.
            anchors = [
                message for message in history
                if isinstance(message, UserMessage) and message.content.strip()
                and (message.metadata or {}).get("is_anchor")
                and (message.metadata or {}).get("anchor_guidance")
                == scenario.state_queue.anchor
                and (message.metadata or {}).get("delivery_status", "emitted")
                == "emitted"
            ]
            if not anchors:
                raise ValueError("minimal_context_opening_not_captured")
            if pointer.get("payload_keys"):
                raise ValueError("minimal_context_dependency_not_public")
            references.append({
                "aim_id": "OPENING",
                "source_intent": scenario.state_queue.anchor,
                "source_payload": [],
            })
            continue
        if pointer["aim_id"] not in released:
            raise ValueError("minimal_context_dependency_not_emitted")
        entry = released[pointer["aim_id"]]
        keys = pointer["payload_keys"]
        payload = [item for item in entry["source_payload"] if item[0] in keys]
        if set(keys) != {item[0] for item in payload}:
            raise ValueError("minimal_context_dependency_not_public")
        references.append({
            "aim_id": entry["aim_id"],
            "source_intent": entry["source_intent"] if pointer["include_intent"] else None,
            "source_payload": payload,
        })
    return references


def render(scenario, *, source_key, source_sha256, stage, aim, history,
           baseline, arm, registrations):
    """Return a new request only for an exact preregistered eligible GEN turn.

    D is T17-only. R is T17/T20 material-only, including statically bound
    history-dependent material. Its source references use the old dependency
    pointers, but neither prior generated user text nor assistant text is sent.
    Everything outside the selected route is an exact deep-copy no-op.
    """
    result = deepcopy(baseline)
    task = source_key.split("/")[0]
    if stage != "GENERATE" or task not in {"T17", "T20"} or arm == "B0":
        return result
    if arm in {"S", "F", "C", "FS"}:
        return legacy.render(
            scenario, source_key=source_key, source_sha256=source_sha256,
            stage=stage, aim=aim, history=history, baseline=baseline,
            arm=arm, registrations=registrations,
        )
    if arm == "D" and task != "T17":
        return result
    if arm == "R" and (aim is None or aim.is_final):
        return result
    matches = [
        registration for registration in registrations
        if registration["source_key"] == source_key
        and registration["source_sha256"] == source_sha256
        and registration["scenario_sha256"] == legacy.digest(scenario.model_dump(mode="json"))
        and registration["aim_id"] == getattr(aim, "aim_id", None)
        and arm in registration["arms"]
    ]
    if not matches:
        return result
    if len(matches) != 1:
        raise ValueError("minimal_context_ambiguous_registration")
    bound = matches[0]
    role = "final" if aim.is_final else "material"
    if bound["task"] != task or bound["turn_role"] != role:
        raise ValueError("minimal_context_role_mismatch")
    if arm not in {"D", "R"}:
        raise ValueError("minimal_context_invalid_arm")
    current = validated_current_source(scenario, aim, baseline["messages"])
    references = _references(scenario, aim, history, bound["dependencies"])
    notes = [D if task == "T17" else legacy.S["T20"]]
    if arm == "R":
        notes.append(R)
    content = f"TURN ROLE: {role}\nSPEAKER: {bound['speaker']}\n\n"
    content += "\n\n".join(notes)
    content += "\n\nCURRENT SOURCE:\n" + json.dumps(current, ensure_ascii=False)
    content += (
        "\n\nEARLIER ALLOWED SOURCE (not necessarily expressed; reference only):\n"
        + json.dumps(references, ensure_ascii=False)
    )
    if arm == "D":
        content += "\n\nACTUAL DIALOGUE CONTEXT:\n" + json.dumps([
            {"role": message.role, "content": message.content} for message in history
        ], ensure_ascii=False)
    content += "\nOutput only the user message."
    result["messages"] = [
        {"role": "system", "content": legacy.COMMON},
        {"role": "user", "content": content},
    ]
    return result
