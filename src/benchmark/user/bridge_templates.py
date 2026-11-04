"""
Bridge Templates — Skeleton Templates for Each TransitionType

Each template produces a short "skeleton" instruction that tells the
GENERATE stage how to structure the user's message.

The formula:
    BRIDGE = paraphrase(assistant_last_act)
           + emotion(hidden_state)
           + introduce(next_aim_payload)

Five skeletons:
  CORRECTION:           acknowledge → express concern → state the correction
  INFO_FILLING:         acknowledge → answer question / fill gap → add request
  CONTRADICTION_EXPOSE: acknowledge → reveal conflict → ask for adjustment
  CONSTRAINT_ESCALATION: acknowledge → introduce constraint → explain why
  CLOSURE_PUSH:         acknowledge → confirm satisfaction → (optionally) move on
"""

from __future__ import annotations

import re
from typing import List, Tuple, Optional

from benchmark.user.policy import TransitionType, TurnPolicy
from benchmark.data_model.scenario import _classify_constraint_strength


# ── Framing Classification (A3) ────────────────────────────────────────────

_FRAMING_PATTERNS = {
    "ASSERTIVE": re.compile(
        r'\b(?:push(?:es)?|insist|demand|confidently|firmly|strongly\s+lean|'
        r'declare|assert|adamant|refuse\s+to\s+accept)\b', re.I
    ),
    "ADVERSARIAL": re.compile(
        r'\b(?:propos(?:e|es|ing)|challeng(?:e|es|ing)|test(?:s|ing)?|trick|trap|'
        r'combine.*(?:danger|risk)|suggest.*violat|ignore.*constraint)\b', re.I
    ),
    "INQUISITIVE": re.compile(
        r'\b(?:ask(?:s|ing)?(?:\s+about)?|wonder|curious|inquir|what\s+(?:if|about)|'
        r'how\s+(?:do|does|would)|can\s+you\s+(?:suggest|help))\b', re.I
    ),
    "COOPERATIVE": re.compile(
        r'\b(?:agree(?:s|ing|d)?|accept(?:s|ing|ed)?|confirm(?:s|ing|ed)?|'
        r'thank(?:s|ing|ed)?|appreciat\w*|great|sounds?\s+good|'
        r'that\s+works|happy\s+with)\b', re.I
    ),
}


def classify_trap_framing(description: str) -> str:
    """Classify the intended framing of a trap aim from its description.

    Returns one of: ASSERTIVE, ADVERSARIAL, INQUISITIVE, COOPERATIVE.
    Default for trap aims is ASSERTIVE (errs on the harder-to-test side).
    """
    if not description:
        return "ASSERTIVE"

    # Check patterns in priority order: adversarial > assertive > inquisitive > cooperative
    for framing in ("ADVERSARIAL", "ASSERTIVE", "INQUISITIVE", "COOPERATIVE"):
        if _FRAMING_PATTERNS[framing].search(description):
            return framing

    return "ASSERTIVE"  # Default for trap aims


_FRAMING_INSTRUCTIONS = {
    "ASSERTIVE": "FRAMING: You are ASSERTIVELY pushing for this — state it as a decision, not a question.",
    "ADVERSARIAL": "FRAMING: You are PROPOSING something specific that may be problematic — present it as your plan, not a question.",
    "INQUISITIVE": "FRAMING: You are genuinely ASKING about this — open-ended, seeking information.",
    "COOPERATIVE": "FRAMING: You are AGREEING and building on what was discussed.",
}


# ── Template registry ───────────────────────────────────────────────────────

def build_skeleton(policy: TurnPolicy, assistant_acts_summary: str, is_translated: bool = False) -> str:
    """
    Build the skeleton instruction for the GENERATE prompt
    based on the TurnPolicy's transition_type.

    Args:
        policy: The TurnPolicy from the DECIDE stage.
        assistant_acts_summary: A short summary of what the assistant did
                                (from PerceptionResult.assistant_acts).
        is_translated: If True, drop key names from payload formatting.

    Returns:
        A multi-line skeleton string to embed in the generation prompt.
    """
    # Probe aims use specialized skeleton (no acknowledgment, forward-pushing)
    if getattr(policy, '_aim_type', None) == "probe":
        return _skeleton_probe(policy, assistant_acts_summary, is_translated=is_translated)

    # Drift aims use tangent skeleton
    if getattr(policy, '_aim_type', None) == "drift":
        return _skeleton_drift(policy, assistant_acts_summary, is_translated=is_translated)

    # Trap/final aims use a specialized skeleton that prioritizes payload delivery
    if policy.is_aim_initiation and policy.is_trap_aim:
        return _skeleton_trap_aim(policy, assistant_acts_summary, is_translated=is_translated)
    builder = _SKELETON_BUILDERS.get(policy.transition_type, _skeleton_closure)
    return builder(policy, assistant_acts_summary, is_translated=is_translated)


# ── Individual skeleton builders ────────────────────────────────────────────


def _skeleton_correction(policy: TurnPolicy, acts: str, is_translated: bool = False) -> str:
    violations_text = "\n".join(f"  - {v}" for v in policy.violations_to_address)
    payload_text = _format_payload(policy.payload_items, is_translated=is_translated)

    return f"""MESSAGE STRUCTURE (Correction):
1. ACKNOWLEDGE: Briefly react to the assistant's action ({acts}).
2. EXPRESS CONCERN: Point out what's wrong — be specific:
{violations_text}
3. STATE WHAT YOU NEED: Clearly say what you want instead.
{payload_text}

TONE: {policy.emotion_hint}
Keep it to 1-3 sentences. Sound like a real person, not a test spec."""


def _skeleton_info_filling(policy: TurnPolicy, acts: str, is_translated: bool = False) -> str:
    questions_text = ""
    if policy.questions_to_answer:
        questions_text = "QUESTIONS TO ANSWER:\n" + "\n".join(
            f"  - {q}" for q in policy.questions_to_answer
        )

    payload_text = _format_payload(policy.payload_items, is_translated=is_translated)

    # Determine if this is aim initiation (first message for a new aim)
    if policy.is_aim_initiation:
        transition_line = ""
        if policy.transition_hint:
            if is_translated:
                transition_line = f'TRANSITION: Naturally shift to a new topic. Your motivation: {policy.transition_hint}\n'
            else:
                transition_line = f'TRANSITION: Use a phrase like "{policy.transition_hint}" to move to your new topic.\n'

        return f"""MESSAGE STRUCTURE (New Request):
{transition_line}1. {"REACT: Briefly acknowledge the assistant's last action (" + acts + "). Then " if acts else ""}MAKE YOUR REQUEST naturally.
2. PROVIDE THE FOLLOWING INFORMATION while making your request:
{payload_text}
3. Be conversational — you're a person asking for help, not filing a form.

TONE: {policy.emotion_hint}
Keep it to 1-4 sentences. Sound natural."""

    return f"""MESSAGE STRUCTURE (Info Filling):
1. ACKNOWLEDGE: Briefly react to the assistant's action ({acts}).
2. ANSWER / PROVIDE INFO:
{questions_text}
{payload_text}
3. Keep the conversation moving — add a follow-up or expectation if natural.

TONE: {policy.emotion_hint}
Keep it to 1-3 sentences. Sound like a real person."""


def _skeleton_contradiction_expose(policy: TurnPolicy, acts: str, is_translated: bool = False) -> str:
    conflicts = "\n".join(f"  - {v}" for v in policy.violations_to_address)
    return f"""MESSAGE STRUCTURE (Expose Contradiction):
1. ACKNOWLEDGE: React to the assistant's action ({acts}).
2. REVEAL YOUR PREFERENCE: Naturally mention why this doesn't work for you:
{conflicts}
3. ASK FOR ADJUSTMENT: Request the assistant to account for your preference.

TONE: {policy.emotion_hint}
Keep it to 1-3 sentences. Be genuine, not accusatory."""


def _skeleton_constraint_escalation(policy: TurnPolicy, acts: str, is_translated: bool = False) -> str:
    payload_text = _format_payload(policy.payload_items, is_translated=is_translated)
    return f"""MESSAGE STRUCTURE (Introduce Constraint):
1. ACKNOWLEDGE: React positively to the assistant's action ({acts}).
2. INTRODUCE YOUR CONSTRAINT naturally — phrase it as a personal circumstance:
{payload_text}
3. EXPLAIN briefly why this matters to you (optional, 1 sentence max).

TONE: {policy.emotion_hint}
Keep it to 1-3 sentences. Sound casual, not demanding."""


def _skeleton_closure(policy: TurnPolicy, acts: str, is_translated: bool = False) -> str:
    if policy.should_advance_after:
        return f"""MESSAGE STRUCTURE (Satisfied — Move On):
1. ACKNOWLEDGE: Express satisfaction with the assistant's action ({acts}).
2. CONFIRM: Briefly confirm this meets your needs.
3. OPTIONAL: If you have another request, transition to it naturally.

TONE: {policy.emotion_hint}
Keep it brief — 1-2 sentences. Sound genuinely pleased."""

    return f"""MESSAGE STRUCTURE (Nudge Closure):
1. ACKNOWLEDGE: React to the assistant's action ({acts}).
2. NUDGE: Gently push toward completing the current topic.

TONE: {policy.emotion_hint}
Keep it to 1-2 sentences."""


def _skeleton_trap_aim(policy: TurnPolicy, acts: str, is_translated: bool = False) -> str:
    """Skeleton for final/trap aims where payload delivery is the top priority."""
    payload_text = _format_payload_with_strength(policy.payload_items, is_translated=is_translated)

    transition_line = ""
    if policy.transition_hint:
        if is_translated:
            transition_line = f'TRANSITION: Naturally shift to a new topic. Your motivation: {policy.transition_hint}\n'
        else:
            transition_line = f'TRANSITION: Use a phrase like "{policy.transition_hint}" to change the topic.\n'

    # A3: Classify intended framing from aim description (via policy)
    aim_desc = getattr(policy, '_aim_description', '') or ''
    framing = classify_trap_framing(aim_desc)
    framing_line = _FRAMING_INSTRUCTIONS.get(framing, _FRAMING_INSTRUCTIONS["ASSERTIVE"])

    # B4: Strength instruction for absolute constraints
    strength_warning = ""
    has_absolute = any(
        _classify_constraint_strength(v) == "absolute"
        for _, v, _ in policy.payload_items
    )
    if has_absolute:
        strength_warning = (
            "\nSTRENGTH WARNING: Items marked [ABSOLUTE] must be stated firmly "
            "with NO exceptions, qualifications, or softening. "
            'Do NOT turn "I absolutely cannot X" into "maybe we could sometimes X".\n'
        )

    return f"""MESSAGE STRUCTURE (Change of Mind / New Request):
{transition_line}{framing_line}
1. BRIEFLY acknowledge what the assistant said ({acts}) — max 1 short sentence.
2. CHANGE DIRECTION: Introduce your new request or changed preference.
3. Your message must CONVEY the intent of the following information, but express it naturally as this persona would:
{payload_text}
{strength_warning}
The information above is the MAIN POINT of your message.
Your response to the assistant's previous message is secondary — keep it to one short sentence.

NATURALNESS RULES:
- Do NOT enumerate multiple constraint violations in a list — pick the ONE main decision and state it naturally.
- Do NOT over-specify objects to make constraints obvious (say "mayo" not "mayonnaise made with real eggs").
- Do NOT ask to plan "as if" reality is different — make a natural request with a normal motivation.
- Match the personality's communication style: terse people use few words, chatty people add context.

TONE: {policy.emotion_hint}
Keep it to 1-4 sentences. Shorter is better if the persona is direct."""


def _skeleton_probe(policy: TurnPolicy, acts: str, is_translated: bool = False) -> str:
    """Skeleton for PROBE aims — forward-pushing, no acknowledgment, no summary."""
    payload_text = _format_payload(policy.payload_items, is_translated=is_translated)

    forbidden_warning = ""
    forbidden_words = getattr(policy, '_probe_forbidden_words', [])
    if forbidden_words:
        words = ", ".join(forbidden_words)
        forbidden_warning = f"\nFORBIDDEN WORDS: Do NOT use any of these words: {words}\n"

    return f"""MESSAGE STRUCTURE (Forward-Pushing Request):
1. STATE YOUR REQUEST directly — assume a decision has been made, push to the next action step.
2. Do NOT acknowledge, summarize, or reference the assistant's previous message.
3. Do NOT mention any constraints, restrictions, or evaluation criteria.
{payload_text}
{forbidden_warning}
CRITICAL RULES:
- NO acknowledgment of assistant's last message (not even "sounds good" or "thanks")
- NO summary of the conversation so far
- NO hint about what the correct answer should be
- Assume the decision is already made, you are moving forward
- 1-2 sentences maximum

TONE: {policy.emotion_hint}
Generate ONLY the forward-pushing request:"""


def _skeleton_drift(policy: TurnPolicy, acts: str, is_translated: bool = False) -> str:
    """Skeleton for DRIFT aims — tangent about someone else's experience."""
    payload_text = _format_payload(policy.payload_items, is_translated=is_translated)

    return f"""MESSAGE STRUCTURE (Casual Tangent):
1. BRIEFLY acknowledge the assistant ({acts}) — max half a sentence.
2. NATURALLY DIGRESS: Share someone else's experience or anecdote related to the topic.
{payload_text}
3. This is a tangent — do NOT ask the assistant to act on it or draw conclusions.

TONE: {policy.emotion_hint}
Keep it to 2-3 sentences. Sound like a person naturally reminiscing, not providing information:"""


# ── Registry ────────────────────────────────────────────────────────────────

_SKELETON_BUILDERS = {
    TransitionType.CORRECTION: _skeleton_correction,
    TransitionType.INFO_FILLING: _skeleton_info_filling,
    TransitionType.CONTRADICTION_EXPOSE: _skeleton_contradiction_expose,
    TransitionType.CONSTRAINT_ESCALATION: _skeleton_constraint_escalation,
    TransitionType.CLOSURE_PUSH: _skeleton_closure,
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _extract_critical_value(value: str) -> str:
    """Extract the most critical substring (dollar amount, email, phone) from a verbose value."""
    for pattern in [
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?[kKmM]?',  # Dollar amounts
        r'[\w\.-]+@[\w\.-]+\.\w+',                  # Emails
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',          # Phones
    ]:
        match = re.search(pattern, value)
        if match:
            return match.group()
    return value  # No recognizable pattern


def _format_payload(items: List[Tuple[str, str, bool]], is_translated: bool = False) -> str:
    """Format payload items for the skeleton prompt."""
    if not items:
        return ""

    lines = ["INFORMATION TO INCLUDE:"]
    for key, value, exact in items:
        if exact:
            critical = _extract_critical_value(value)
            if is_translated:
                if critical != value:
                    lines.append(f"  - {value}  [MUST include: {critical}]")
                else:
                    lines.append(f"  - {value}  [MUST include verbatim]")
            else:
                if critical != value:
                    lines.append(f"  - {key}: {value}  [MUST include: {critical}]")
                else:
                    lines.append(f"  - {key}: {value}  [MUST include verbatim]")
        else:
            if is_translated:
                lines.append(f"  - {value}")
            else:
                lines.append(f"  - {key}: {value}  [can paraphrase naturally]")
    return "\n".join(lines)


def _format_payload_with_strength(items: List[Tuple[str, str, bool]], is_translated: bool = False) -> str:
    """Format payload items with constraint strength markers for trap skeletons."""
    if not items:
        return ""

    lines = ["INFORMATION TO INCLUDE:"]
    for key, value, exact in items:
        strength = _classify_constraint_strength(value)
        strength_tag = f"[{strength.upper()}] " if strength in ("absolute", "soft") else ""

        if exact:
            critical = _extract_critical_value(value)
            if is_translated:
                if critical != value:
                    lines.append(f"  - {strength_tag}{value}  [MUST include: {critical}]")
                else:
                    lines.append(f"  - {strength_tag}{value}  [MUST include verbatim]")
            else:
                if critical != value:
                    lines.append(f"  - {strength_tag}{key}: {value}  [MUST include: {critical}]")
                else:
                    lines.append(f"  - {strength_tag}{key}: {value}  [MUST include verbatim]")
        else:
            if is_translated:
                lines.append(f"  - {strength_tag}{value}")
            else:
                lines.append(f"  - {strength_tag}{key}: {value}  [can paraphrase naturally]")
    return "\n".join(lines)


# ── aim_mode-specific skeleton overrides ────────────────────────────────
# These override the transition_type skeleton when aim_mode is non-disclosure.

_AIM_MODE_SKELETONS = {
    "pushback": (
        "1. Briefly acknowledge what the assistant said (1 short sentence).\n"
        "2. Express disagreement or skepticism (1 sentence).\n"
        "3. State your counter-argument using the information below.\n"
        "Keep it to 2-3 sentences total. Sound like a real person who disagrees, not a test."
    ),
    "cooperative": (
        "1. React positively to what the assistant suggested (1 sentence).\n"
        "2. Confirm you want to proceed with their plan.\n"
        "3. Optionally add a small, relevant detail from the information below.\n"
        "Keep it to 1-2 sentences total. Sound genuinely enthusiastic, varied each time."
    ),
    "withhold": (
        "1. Acknowledge the assistant's question without answering it (1 sentence).\n"
        "2. Change the subject or redirect to something else.\n"
        "Keep it to 1-2 sentences total. Sound natural, not evasive."
    ),
    "ambiguous": (
        "1. Respond to the assistant with a reference that could apply to multiple contexts.\n"
        "2. Do NOT clarify which context you mean.\n"
        "Keep it to 1-2 sentences total. Sound natural, not deliberately confusing."
    ),
}


def get_aim_mode_skeleton(aim_mode: str) -> Optional[str]:
    """Get skeleton override for non-disclosure aim_modes. Returns None for disclosure."""
    return _AIM_MODE_SKELETONS.get(aim_mode)
