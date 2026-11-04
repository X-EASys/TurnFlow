"""
TRANSLATE Stage — Stage 2.5 of the BRIDGE Pipeline

Converts raw orchestration metadata (aim descriptions, constraint values,
flow_info items) into sanitized first-person inputs for the GENERATE stage.

This creates an information barrier: the GENERATE LLM never sees aim IDs,
success criteria, evaluation text, or third-person descriptions. It only
sees what a real human would know and say.

Two output zones:
  Zone 1 (context): First-person personality, goal, emotional state
  Zone 2 (mandatory): Pre-built sentences the user MUST include in their message
"""

from __future__ import annotations

import re
import json as _json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from benchmark.user.first_pass import call_stage, repair
from loguru import logger

from benchmark.data_model.scenario import _classify_constraint_strength


@dataclass
class TranslateResult:
    """Output of the TRANSLATE stage."""

    # Zone 1: What a real person would know/feel (context for the LLM)
    zone1_context: str = ""

    # Zone 2: Sentences the person MUST say (deterministic content)
    zone2_mandatory_sentences: List[str] = field(default_factory=list)

    # Generation guidance
    framing_instruction: str = ""   # e.g. "state firmly" vs "ask casually"
    emotion_hint: str = ""          # e.g. "mildly frustrated"
    acknowledgment_hint: str = ""   # brief hint for reacting to agent's last message


# ── Perspective conversion patterns ──────────────────────────────────────

_PERSPECTIVE_PATTERNS = [
    (re.compile(r'\bThe user\b', re.I), 'I'),
    (re.compile(r'\bthe user\b'), 'I'),
    (re.compile(r'\bUser\b'), 'I'),
    (re.compile(r'\bShe\b'), 'I'),
    (re.compile(r'\bshe\b'), 'I'),
    (re.compile(r'\bHe\b'), 'I'),
    (re.compile(r'\bhe\b'), 'I'),
    (re.compile(r'\bThey\b'), 'I'),
    (re.compile(r'\bthey\b'), 'I'),
    (re.compile(r'\btheir\b'), 'my'),
    (re.compile(r'\bTheir\b'), 'My'),
    (re.compile(r'\bHer\b'), 'My'),
    (re.compile(r'\bher\b'), 'my'),
    (re.compile(r'\bHis\b'), 'My'),
    (re.compile(r'\bhis\b'), 'my'),
]

# Patterns that indicate evaluator/agent-prescriptive language
_AGENT_PRESCRIPTIVE = re.compile(
    r'\b(?:the agent must|agent should|agent is expected to|must be evaluated|'
    r'success_criteria|reasoning_hops|evaluation_focus|core_capability)\b', re.I
)

# Strength-based sentence starters
_STRENGTH_TEMPLATES = {
    "absolute": [
        "I absolutely cannot {}.",
        "This is non-negotiable: {}.",
        "No exceptions — {}.",
    ],
    "strong": [
        "It's really important that {}.",
        "I need to make sure {}.",
        "I'm firm on this — {}.",
    ],
    "soft": [
        "I'd prefer if {}.",
        "If possible, {}.",
        "Ideally, {}.",
    ],
}

# Metadata patterns that must never appear in Zone 1
_ZONE1_FORBIDDEN = re.compile(
    r'\b(?:aim_\d|aim_id|success_criteria|evaluation_focus|core_capability|'
    r'required_elements|forbidden_elements|ground_truth|task_type_name|'
    r'reasoning_hops|difficulty_level|factor_instantiation)\b', re.I
)

_TRANSLATE_PROMPT = """You are converting internal scenario data into natural first-person speech for a user simulator.

TASK: Rewrite the following orchestration data as if YOU are the person described.

YOUR PERSONALITY: {personality}
DOMAIN: {domain}
EMOTIONAL STATE: {emotion_hint}

WHAT YOU WANT THIS TURN (aim context):
{aim_context}

INFORMATION YOU MUST CONVEY (payload items):
{payload_text}

OUTPUT FORMAT (JSON):
{{
  "zone1_context": "1-2 sentence first-person description of who you are, what you want, and how you feel right now. NO aim IDs, NO evaluation language, NO third-person references.",
  "zone2_sentences": ["One natural first-person sentence per payload item. Preserve exact numbers/names. Use 'I' not 'the user'."],
  "framing": "How to deliver these (e.g., 'state firmly', 'mention casually', 'demand clearly')",
  "acknowledgment_hint": "Brief hint for how to react to the assistant's last response"
}}

RULES:
- Write as the user speaking in first-person ("I", "my", "me") for your OWN thoughts, actions, and feelings
- When a payload item mentions another person by name (e.g., "Lao Zhang finished...", "The client said..."), you are RELAYING what that person said or did — keep THEM in third person. Say "Lao Zhang told me he finished..." or "The client said they'll send...", NEVER absorb their actions into "I" or "my"
- NEVER use third-person for the USER ("the user", "she wants") — but DO keep OTHER named people in third person
- NEVER include aim IDs, success criteria, or evaluation text
- PRESERVE exact dollar amounts, dates, percentages, and names verbatim
- Each zone2 sentence must be self-contained and natural-sounding
- Keep zone1_context under 40 words
- Keep each zone2 sentence under 30 words"""


def _to_first_person(text: str) -> str:
    """Convert third-person or agent-prescriptive text to first-person user speech."""
    result = text
    for pattern, replacement in _PERSPECTIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    # Handle "The agent must avoid X" → "I need to avoid X"
    result = re.sub(
        r'\b[Tt]he agent (?:must|should|is expected to)\s+',
        'I need to ',
        result,
    )
    # Handle "must not exceed" → "I can't go over"
    result = re.sub(r'\bmust not exceed\b', "I can't go over", result)
    # Fix verb conjugation after She/He → I replacement
    # "I has" → "I have", "I refuses" → "I refuse", "I does" → "I do"
    result = re.sub(r'\bI has\b', 'I have', result)
    result = re.sub(r'\bI does\b', 'I do', result)
    result = re.sub(r'\bI (\w+)s\b', _fix_verb_conjugation, result)
    return result


def _fix_verb_conjugation(match: re.Match) -> str:
    """Fix third-person verbs after pronoun replacement (I refuses → I refuse)."""
    verb_with_s = match.group(1) + "s"
    # Don't fix words that naturally end in 's' (analysis, this, is, was, etc.)
    # Only fix clear third-person-singular forms
    non_verbs = {
        "this", "is", "was", "his", "us", "its", "yes", "bus",
        "plus", "gas", "thus", "axis", "basis", "analysis",
        "always", "includes", "provides",
    }
    if verb_with_s.lower() in non_verbs:
        return f"I {verb_with_s}"
    # Common third-person singular patterns
    if re.match(r'^(refuse|want|need|prefer|like|hate|love|take|make|use|require|insist|demand|expect|plan|avoid|wish|request)s$', verb_with_s, re.I):
        return f"I {match.group(1)}"
    # For other verbs ending in -es (e.g. "refuses" → "refuse")
    base = match.group(1)
    if base.endswith("e") and len(base) > 2:
        # "refuses" matched as "refuse" + "s" — this is correct already
        return f"I {base}"
    return f"I {verb_with_s}"  # Keep as-is if unsure


def translate_payload_items(
    items: List[Tuple[str, str, bool]],
) -> List[str]:
    """
    Convert payload item tuples into first-person natural sentences.

    This is the deterministic (no-LLM) path. Each (key, value, exact) tuple
    becomes one sentence. Exact items preserve the literal value. Non-exact
    items are rewritten using strength-appropriate templates.

    Args:
        items: List of (key, value, exact_disclosure) tuples from DECIDE.

    Returns:
        List of first-person sentences, one per item.
    """
    sentences = []
    for key, value, exact in items:
        # Step 1: Convert to first-person
        fp_value = _to_first_person(value)

        # Step 2: Strip evaluator language entirely
        if _AGENT_PRESCRIPTIVE.search(fp_value):
            # Extract the core action after prescriptive phrasing
            fp_value = re.sub(
                r'^.*?(?:must|should|is expected to)\s+', '', fp_value, count=1
            )
            fp_value = f"I need to {fp_value}" if fp_value else value

        # Step 3: For exact items, ensure the critical value is preserved
        if exact:
            # The sentence must contain the literal value
            if fp_value != value and not _contains_critical_value(fp_value, value):
                # Fallback: use the raw value in a simple frame
                sentences.append(f"Just to be clear: {_to_first_person(value)}")
            else:
                sentences.append(fp_value)
        else:
            # Non-exact: apply strength-based template
            strength = _classify_constraint_strength(value)
            templates = _STRENGTH_TEMPLATES.get(strength, _STRENGTH_TEMPLATES["strong"])
            # Deterministic selection based on key hash
            template = templates[hash(key) % len(templates)]

            # Extract the core content (strip leading articles/pronouns)
            core = re.sub(r'^(?:I |my |she |he |they |the user )', '', fp_value, flags=re.I).strip()
            if core:
                sentences.append(template.format(core))
            else:
                sentences.append(fp_value)

    return sentences


def _contains_critical_value(text: str, original: str) -> bool:
    """Check if text still contains critical numeric/named values from original."""
    # Extract dollar amounts, percentages, dates
    patterns = [
        re.compile(r'\$[\d,]+(?:\.\d+)?'),
        re.compile(r'\d+\s*%'),
        re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
    ]
    for p in patterns:
        originals = p.findall(original)
        for val in originals:
            if val not in text:
                return False
    return True


def _translate_simplified(aim, policy, personality, domain) -> TranslateResult:
    """Simplified translate path for pre-translated scenarios.

    Only applies: perspective patterns, critical value checks.
    Skips: _AGENT_PRESCRIPTIVE, _ZONE1_FORBIDDEN, _STRENGTH_TEMPLATES,
    agent-prescriptive rewriting in _to_first_person().
    """
    payload_items = []
    for key, value, exact in getattr(policy, "payload_items", []):
        # Only apply perspective fix (she->I, they->I)
        cleaned = value
        for pattern, replacement in _PERSPECTIVE_PATTERNS:
            cleaned = pattern.sub(replacement, cleaned)
        payload_items.append((key, cleaned, exact))

    # Build minimal zone1 from aim description
    desc = getattr(aim, "description", "")
    goal = getattr(aim, "conversation_goal", "")
    zone1 = f"You are: {personality}. Domain: {domain}."
    if goal:
        zone1 += f" Your current goal: {goal}"

    return TranslateResult(
        zone1_context=zone1,
        zone2_mandatory_sentences=[item[1] for item in payload_items if item[2]],
        framing_instruction="State naturally",
        emotion_hint=getattr(policy, "emotion_hint", "calm and friendly"),
        acknowledgment_hint="Respond naturally to what the assistant said.",
    )


def translate_for_aim(
    aim,
    policy,
    personality: str,
    domain: str,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    is_translated: bool = False,
    trace=None,
    turn_input=None,
    aim_mode: str = "disclosure",  # v6.0: Tier 3 aim mode
) -> TranslateResult:
    """
    Translate raw orchestration data into sanitized Zone 1 + Zone 2 inputs.

    Runs once per aim. Uses LLM for natural rephrasing when available,
    falls back to rule-based translation otherwise.

    Args:
        aim: Current aim being pursued.
        policy: TurnPolicy from DECIDE with payload_items.
        personality: User personality description.
        domain: Scenario domain.
        llm: Model name (None = rule-based only).
        llm_args: Extra kwargs for LLM call.
        is_translated: If True, use simplified translate path for pre-translated scenarios.

    Returns:
        TranslateResult with Zone 1 context and Zone 2 mandatory sentences.
    """
    if is_translated:
        # Pre-translated data: only apply lightweight perspective fix
        # and critical value preservation. Skip all overfitted patterns.
        return _translate_simplified(aim, policy, personality, domain)

    # Always build rule-based sentences as fallback
    rule_sentences = translate_payload_items(policy.payload_items)

    if not llm:
        return _build_rule_based_result(aim, policy, personality, rule_sentences)

    # Build aim context for the TRANSLATE prompt (stripped of eval metadata)
    aim_context = turn_input["text"] if turn_input is not None else _build_aim_context(aim, policy)

    # v6.0: aim_mode-specific framing for TRANSLATE
    mode_framing = ""
    mode_ack = ""
    if aim_mode == "pushback":
        mode_framing = "TONE: The user DISAGREES with the agent. Frame all sentences as firm pushback. The user is skeptical and unconvinced."
        mode_ack = "Express skepticism or disagreement about what the agent said"
    elif aim_mode == "cooperative":
        mode_framing = "TONE: The user AGREES with the agent. Frame all sentences as enthusiastic confirmation. The user is happy and supportive."
        mode_ack = "Express enthusiasm about what the agent said"
    elif aim_mode == "withhold":
        mode_framing = "TONE: The user is DEFLECTING. They want to avoid the agent's question and change the subject."
        mode_ack = "Respond naturally but dodge the topic the agent asked about"
    elif aim_mode == "ambiguous":
        mode_framing = "TONE: The user makes an AMBIGUOUS reference that could apply to multiple contexts discussed earlier. Do NOT clarify."
        mode_ack = "Respond with a reference that could apply to multiple things discussed"

    # Build payload text
    payload_lines = []
    for key, value, exact in policy.payload_items:
        tag = "[EXACT — preserve verbatim]" if exact else "[can rephrase naturally]"
        payload_lines.append(f"- {key}: {value}  {tag}")
    payload_text = "\n".join(payload_lines) if payload_lines else "(no payload this turn)"

    prompt = _TRANSLATE_PROMPT.format(
        personality=personality,
        domain=domain,
        emotion_hint=getattr(policy, 'emotion_hint', '') or "neutral",
        aim_context=aim_context,
        payload_text=payload_text,
    )

    try:
        from benchmark.utils.llm_utils import generate_chat_completion

        response = call_stage(trace, "TRANSLATE", generate_chat_completion,
            model=llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=16384,
            **(llm_args or {}),
        )
        parsed = _parse_translate_response(response)

        # Post-validate: strip any leaked metadata from zone1
        zone1 = parsed.get("zone1_context", "")
        if (trace is None or trace.content_repair) and _ZONE1_FORBIDDEN.search(zone1):
            logger.warning(f"TRANSLATE zone1 contained metadata, stripping: {zone1[:80]}")
            cleaned = _ZONE1_FORBIDDEN.sub("", zone1).strip()
            zone1 = trace.change("translate_metadata_strip", zone1, cleaned) if trace is not None else cleaned

        zone2 = parsed.get("zone2_sentences", [])

        # Verify exact values survived translation
        zone2 = repair(trace, "translate_exact_values", _verify_exact_values, zone2, policy.payload_items, rule_sentences)

        return TranslateResult(
            zone1_context=zone1,
            zone2_mandatory_sentences=zone2,
            framing_instruction=mode_framing or parsed.get("framing", ""),
            emotion_hint=getattr(policy, 'emotion_hint', '') or "",
            acknowledgment_hint=mode_ack or parsed.get("acknowledgment_hint", ""),
        )

    except Exception as e:
        logger.error(f"TRANSLATE LLM failed ({e}), no fallback — propagating error")
        raise RuntimeError(f"TRANSLATE LLM failed after retries: {e}") from e


def _build_aim_context(aim, policy) -> str:
    """Extract user-relevant context from aim, stripping evaluation metadata."""
    parts = []
    if hasattr(aim, 'conversation_goal') and aim.conversation_goal:
        parts.append(f"Goal: {_to_first_person(aim.conversation_goal)}")
    if hasattr(aim, 'description') and aim.description:
        # Strip evaluation-focused text from description
        desc = aim.description
        desc = re.sub(r'(?:testing whether|evaluating if|this tests).*$', '', desc, flags=re.I)
        desc = _to_first_person(desc)
        if desc.strip():
            parts.append(f"What to do: {desc.strip()}")
    if hasattr(policy, 'transition_hint') and policy.transition_hint:
        parts.append(f"Transition: {_to_first_person(policy.transition_hint)}")
    return "\n".join(parts) if parts else "Continue the conversation naturally."


def _build_rule_based_result(
    aim, policy, personality: str, sentences: List[str]
) -> TranslateResult:
    """Build TranslateResult without LLM, using rule-based sentences."""
    goal = _to_first_person(aim.conversation_goal or "") if hasattr(aim, 'conversation_goal') and aim.conversation_goal else ""
    zone1 = f"I'm a {personality} looking for help with {goal}." if goal else f"I'm a {personality}."
    # Truncate zone1 to keep it brief
    if len(zone1) > 200:
        zone1 = zone1[:200].rsplit(" ", 1)[0] + "."

    framing = ""
    if getattr(policy, 'is_trap_aim', False):
        from benchmark.user.bridge_templates import classify_trap_framing
        framing_type = classify_trap_framing(getattr(policy, '_aim_description', ''))
        framing_map = {
            "ASSERTIVE": "State this firmly as a decision, not a question.",
            "ADVERSARIAL": "Present this as a challenge or test — push back on prior advice.",
            "INQUISITIVE": "Ask about this openly, as a genuine question.",
            "COOPERATIVE": "Mention this agreeably while moving to your new topic.",
        }
        framing = framing_map.get(framing_type, framing_map["ASSERTIVE"])

    return TranslateResult(
        zone1_context=zone1,
        zone2_mandatory_sentences=sentences,
        framing_instruction=framing,
        emotion_hint=getattr(policy, 'emotion_hint', '') or "",
        acknowledgment_hint="",
    )


def _parse_translate_response(response: str) -> dict:
    """Parse LLM JSON response, handling markdown fences."""
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        # Try to extract JSON from mixed output
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return _json.loads(match.group())
        raise


def _verify_exact_values(
    zone2: List[str],
    payload_items: List[Tuple[str, str, bool]],
    fallback_sentences: List[str],
) -> List[str]:
    """Ensure exact payload values survived LLM translation. Replace with fallback if not."""
    exact_indices = [i for i, (_, _, exact) in enumerate(payload_items) if exact]
    for i in exact_indices:
        if i >= len(zone2) or i >= len(fallback_sentences):
            continue
        key, value, _ = payload_items[i]
        if not _contains_critical_value(zone2[i], value):
            logger.warning(
                f"TRANSLATE lost exact value for '{key}', using rule-based fallback"
            )
            zone2[i] = fallback_sentences[i]
    return zone2
