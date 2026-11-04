"""Opt-in opening experiment; no production registration or output guard.

The caller must supply the opening selected by the existing simulator, including
its initial_statement precedence. This module never reads a scenario or history.
"""

SYSTEM = """Write one natural opening message as the user seeking help.
CURRENT OPENING supplies all content; SPEAKER STYLE changes phrasing only.
Keep the same level of detail: a general introduction stays general, while any
listed items, activities or concrete requirements stay explicit.
Preserve the supplied people, relationships, quantities, timing, conditions,
uncertainty and requests. Do not add examples, list members, background facts,
preferences or requirements that CURRENT OPENING does not supply.
A general introduction is already complete; it does not need an invented
example or an extra request. Do not diagnose an unstated problem or solve the
task. Output only the user's message."""


def build_opening_messages(*, opening_intent: str, speaker: str) -> list[dict]:
    """Build the single O1 candidate from two verbatim, authorized variables."""
    if not isinstance(opening_intent, str) or not opening_intent.strip():
        raise ValueError("opening_intent_required")
    if not isinstance(speaker, str) or not speaker.strip():
        raise ValueError("speaker_style_required")
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"SPEAKER STYLE:\n{speaker}\n\nCURRENT OPENING:\n{opening_intent}",
        },
    ]
