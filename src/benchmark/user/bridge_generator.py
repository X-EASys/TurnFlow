"""
Bridge Generator — Stage 3 of the BRIDGE Pipeline

GENERATE: Produce the final user message via a slim LLM prompt
composed from a skeleton template + payload + emotion / style directives.

The prompt is intentionally SHORT (~20 lines) so the LLM focuses on
natural phrasing rather than being overwhelmed by instructions.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple, TYPE_CHECKING

from benchmark.user.first_pass import call_stage, repair
from loguru import logger

from benchmark.data_model.aim import Aim, AimType
from benchmark.data_model.message import AnyMessage, UserMessage, AssistantMessage
from benchmark.user.perception import PerceptionResult
from benchmark.user.policy import TurnPolicy, TransitionType
from benchmark.user.bridge_templates import build_skeleton
from benchmark.utils.llm_utils import generate_chat_completion

if TYPE_CHECKING:
    from benchmark.user.translate import TranslateResult


_DEGENERATE_PATTERNS = [
    "could you please try that again",
    "i think something was missing",
    "i'm not sure what you mean",
    "could you help me with this",
    "let me try again",
    "i'll try to be more clear",
]


def _is_degenerate_response(content: str) -> bool:
    """Check if GENERATE produced a generic non-answer instead of payload content."""
    lower = content.lower().strip()
    return any(lower.startswith(p) for p in _DEGENERATE_PATTERNS)


# ── Generation Prompt ───────────────────────────────────────────────────────

_GENERATION_PROMPT = """You are simulating a real person talking to an AI assistant.

PERSONALITY: {personality}
DOMAIN: {domain}
{conversation_goal_line}{hidden_state_prompt}

{task_specific_prompt}

{skeleton}

{trigger_block}

{weave_instruction}

IMPORTANT: All descriptions about "you" in YOUR INNER STATE are behavioral guides.
Express them naturally in first person. Never say "User prefers" or "They want" -- say "I prefer" or "I want".
However, when relaying information about OTHER people (mentioned by name), keep them in third person. Say "Lao Zhang told me..." not "I finished my medicine". Never absorb another person's attributes into your own "I".

CRITICAL CONSTRAINT: You MUST respond to what the agent ACTUALLY said in its last message.
NEVER invent information the agent did not provide. NEVER claim the agent said something it did not say.
If the agent provided a solution, acknowledge it naturally. Do NOT pretend a different version was given.

RECENT CONVERSATION:
{recent_history}

{exact_warning}

Generate ONLY the user's message (1-4 sentences, natural and conversational):"""


_GENERATION_PROMPT_V2 = """You are a real person talking to an AI assistant.

{zone1_context}

{framing_instruction}

RESPONSE STRUCTURE:
1. {acknowledgment_hint}
2. YOU MUST INCLUDE ALL of the following in your message — these are the MAIN POINT:
{zone2_block}

CRITICAL RULES:
- Write in first person ("I", "my", "me") for YOUR OWN thoughts and actions
- When the mandatory content mentions another person by name (e.g., "Lao Zhang", "Alex", "the client"), you are RELAYING their info — keep them in third person. Say "Lao Zhang told me he..." or "the client said they...", NEVER convert their attributes to "I" or "my"
- 1-4 sentences, natural and conversational
- The mandatory sentences above are MORE IMPORTANT than responding to the assistant
- If you must choose between acknowledging the assistant and including mandatory content, ALWAYS include the mandatory content
- NEVER invent information the assistant did not provide

RECENT CONVERSATION:
{recent_history}

Generate ONLY the user's message:"""


def _concise_messages(prompt, aim=None):
    final = bool(getattr(aim, "is_final", False) or
                 getattr(getattr(aim, "aim_type", None), "value", "") == "probe")
    instruction = (
        "Write one natural USER message expressing the supplied THIS TURN intent. "
        "Facts may come only from THIS TURN, its prepared response if supplied, and actual prior user statements. "
        "Assistant questions, examples and suggestions are quoted context, not user facts. "
        "If requested details are unavailable, leave those questions unanswered and "
        "continue the current intent; never fill blanks with plausible personal details. "
        "Speak as the person requesting help, not as the assistant offering help. "
        "Do not solve the user's task. Output only the user's message.")
    if final:
        instruction += (
            " For this final request, use ONLY THIS TURN and its supplied payload in the output. "
            "Earlier history is reference context only: do not repeat earlier people, conditions, "
            "restrictions or updates unless they are explicitly part of THIS TURN. "
            "Do not identify conflicts, give hints, or restate the facts the assistant needs to recall.")
    return [{"role": "system", "content": instruction},
        {"role": "user", "content": prompt}]


def _concise_turn_prompt(turn_input, aim, translation=None):
    from benchmark.user.turn_input import behavior_instruction
    translated = ""
    if translation is not None:
        translated = "\nPhrasing notes for this turn:\n" + "\n".join(
            [translation.zone1_context, *translation.zone2_mandatory_sentences])
    note = turn_input.get("generator_note", "")
    boundary = "\n\n" + note if note else ""
    return ("Write the next message as the user. Preserve the meaning and people; natural "
            "paraphrase and reported speech are welcome. Use enough sentences for the current intent. "
            "Use history to understand references; do not restate prior constraints unless the "
            "current intent or an actual assistant question requires them. Keep speaking as the "
            "person seeking help, rather than adopting the assistant\'s questions or role.\n\n"
            + turn_input["text"] + translated + "\n" + behavior_instruction(aim, turn_input)
            + boundary + "\nOutput only the user's message.")


# ── Public API ──────────────────────────────────────────────────────────────


def generate_bridge_message(
    policy: TurnPolicy,
    perception: PerceptionResult,
    aim: Aim,
    personality: str,
    domain: str,
    messages: List[AnyMessage],
    hidden_state_prompt: str = "",
    task_specific_prompt: str = "",
    llm: str = "gpt-4o-mini",
    llm_args: Optional[dict] = None,
    temperature: float = 0.7,
    translate_result: Optional["TranslateResult"] = None,
    is_translated: bool = False,
    trace=None,
    turn_input=None,
    concise_prompt=False,
    messages_renderer=None,
) -> str:
    """
    Generate the user's message using the BRIDGE formula.

    When translate_result is provided (v2 path), uses the Zone 1/Zone 2
    prompt structure from the TRANSLATE stage. Otherwise falls back to
    the original v1 prompt.

    Args:
        policy: TurnPolicy from the DECIDE stage.
        perception: PerceptionResult from the PERCEIVE stage.
        aim: Current Aim being pursued.
        personality: Scenario personality string.
        domain: Scenario domain string.
        messages: Full conversation history.
        hidden_state_prompt: Pre-built hidden state prompt section.
        task_specific_prompt: Task-specific additions from controller.
        llm: Model name for generation.
        llm_args: Extra kwargs.
        temperature: Generation temperature.
        translate_result: Output of the TRANSLATE stage (v2 path).

    Returns:
        The generated user message string.
    """
    # V2 path: use TRANSLATE result for prompt construction
    if translate_result is not None:
        return _generate_v2(
            messages_renderer=messages_renderer,
            trace=trace,
            turn_input=turn_input,
            concise_prompt=concise_prompt,
            translate_result=translate_result,
            perception=perception,
            aim=aim,
            messages=messages,
            hidden_state_prompt=hidden_state_prompt,
            llm=llm,
            llm_args=llm_args,
            temperature=temperature,
            policy=policy,
        )

    # V1 path (original): skeleton-based prompt
    return _generate_v1(
        messages_renderer=messages_renderer,
        trace=trace,
        turn_input=turn_input,
        concise_prompt=concise_prompt,
        policy=policy,
        perception=perception,
        aim=aim,
        personality=personality,
        domain=domain,
        messages=messages,
        hidden_state_prompt=hidden_state_prompt,
        task_specific_prompt=task_specific_prompt,
        llm=llm,
        llm_args=llm_args,
        temperature=temperature,
        is_translated=is_translated,
    )


def _generate_v1(
    policy: TurnPolicy,
    perception: PerceptionResult,
    aim: Aim,
    personality: str,
    domain: str,
    messages: List[AnyMessage],
    hidden_state_prompt: str = "",
    task_specific_prompt: str = "",
    llm: str = "gpt-4o-mini",
    llm_args: Optional[dict] = None,
    temperature: float = 0.7,
    is_translated: bool = False,
    trace=None,
    turn_input=None,
    concise_prompt=False,
    messages_renderer=None,
) -> str:
    """Original v1 generation path using skeleton-based prompt."""
    if turn_input is not None:
        from copy import deepcopy
        aim = deepcopy(aim)
        aim.conversation_goal = ""
        hidden_state_prompt = turn_input["text"]
        personality = turn_input["style"]
        domain = "current conversation"
    # aim_mode is now handled in generate_bridge_message() before V1/V2 branch.

    # Summarize assistant's actions for the skeleton
    acts_summary = ", ".join(perception.assistant_acts[:3]) if perception.assistant_acts else "responded"

    # Build the skeleton from the template
    skeleton = build_skeleton(policy, acts_summary, is_translated=is_translated)

    # v6.0: aim_mode skeleton override
    from benchmark.user.bridge_templates import get_aim_mode_skeleton
    aim_mode = getattr(aim, 'aim_mode', 'disclosure') or 'disclosure'
    mode_skeleton = get_aim_mode_skeleton(aim_mode)
    if mode_skeleton:
        skeleton = mode_skeleton

    # Build trigger block for latent preferences
    trigger_block = ""
    if policy.trigger_instructions:
        trigger_block = "\n".join(policy.trigger_instructions)

    # Exact-item warning
    exact_items = [(k, v) for k, v, exact in policy.payload_items if exact]
    exact_warning = ""
    if is_translated:
        # For translated scenarios: exact flag is off on values, but we still
        # need to ensure critical numbers/measurements survive paraphrasing.
        # Use regex extraction for the inline prompt warning.
        critical_tokens = _extract_critical_tokens_regex(policy.payload_items)
        if critical_tokens:
            exact_warning = (
                "⚠️ IMPORTANT: Your message MUST include these specific values:\n"
                + "\n".join(f"  - {t}" for t in critical_tokens)
            )
    elif exact_items:
        exact_warning = (
            "⚠️ CRITICAL: Include these values VERBATIM (numbers, emails, names):\n"
            + "\n".join(f"  - {k}: {v}" for k, v in exact_items)
        )

    # Recent history
    recent = _format_recent(messages, last_n=3) if turn_input is None else turn_input["text"]

    # Build conversation goal line (provides high-level context to the generator)
    conversation_goal_line = ""
    if hasattr(aim, "conversation_goal") and aim.conversation_goal:
        conversation_goal_line = f"YOUR CURRENT GOAL: {aim.conversation_goal}\n"

    # Weave instruction for pre-translated scenarios
    weave_instruction = ""
    if is_translated:
        weave_instruction = (
            "STYLE: The information points above are already written in your voice. "
            "Weave them together naturally into your response — don't list them as separate items. "
            "They should flow as one coherent thought."
        )

    prompt = _GENERATION_PROMPT.format(
        personality=personality,
        domain=domain,
        conversation_goal_line=conversation_goal_line,
        hidden_state_prompt=hidden_state_prompt,
        task_specific_prompt=task_specific_prompt,
        skeleton=skeleton,
        trigger_block=trigger_block,
        weave_instruction=weave_instruction,
        recent_history=recent,
        exact_warning=exact_warning,
    )

    if concise_prompt and turn_input is not None:
        prompt = _concise_turn_prompt(turn_input, aim)

    request_messages = (_concise_messages(prompt, aim) if concise_prompt and turn_input is not None
                        else [{"role": "user", "content": prompt}])
    if messages_renderer is not None:
        request_messages = messages_renderer(request_messages, aim=aim, history=messages)

    try:
        response = call_stage(trace, "GENERATE", generate_chat_completion,
            model=llm,
            messages=request_messages,
            temperature=temperature,
            max_tokens=16384,
            **(llm_args or {}),
        )

        content = response.strip()

        # Strip wrapping quotes
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]

        # v6.0+: Degenerate LLM output — fail loudly. Removed deterministic
        # _fallback_message substitution because it masked real LLM failures
        # and produced template-only user messages that polluted sim results.
        if not content or _is_degenerate_response(content):
            raise RuntimeError(
                "Bridge V1 generation produced degenerate output (empty or trivial). "
                "Refusing to fall back to template — fix the LLM call instead."
            )

        # Validate exact items are present (only for non-translated)
        if not is_translated:
            content = repair(trace, "exact_info", _ensure_exact_info, content, exact_items, llm=llm, trace=trace)

        # For translated scenarios: LLM-based verification of critical tokens
        if is_translated:
            content = repair(trace, "_verify_critical_tokens_llm", _verify_critical_tokens_llm,
                content, policy.payload_items, llm=llm, llm_args=llm_args, trace=trace
            )

        # Verify payload delivery semantically for critical aims (trap/final)
        content = repair(trace, "_verify_payload_semantics", _verify_payload_semantics,
            content, policy.payload_items, aim, llm=llm, llm_args=llm_args, trace=trace
        )

        return content

    except Exception as exc:
        logger.error(f"Bridge V1 generation failed: {exc}, no fallback — propagating error")
        raise RuntimeError(f"Bridge V1 generation failed after retries: {exc}") from exc


def _generate_v2(
    translate_result: "TranslateResult",
    perception: PerceptionResult,
    aim: Aim,
    messages: List[AnyMessage],
    hidden_state_prompt: str = "",
    llm: str = "gpt-4o-mini",
    llm_args: Optional[dict] = None,
    temperature: float = 0.7,
    policy: Optional[TurnPolicy] = None,
    trace=None,
    turn_input=None,
    concise_prompt=False,
    messages_renderer=None,
) -> str:
    """V2 generation path using TRANSLATE Zone 1 + Zone 2."""
    recent = _format_recent(messages, last_n=3) if turn_input is None else turn_input["text"]

    # Build zone2 block — numbered mandatory sentences
    zone2_lines = []
    for i, sentence in enumerate(translate_result.zone2_mandatory_sentences, 1):
        zone2_lines.append(f"   {i}. {sentence}")
    zone2_block = "\n".join(zone2_lines) if zone2_lines else "   (no specific content required)"

    # Build acknowledgment hint
    ack_hint = translate_result.acknowledgment_hint or (
        "Briefly acknowledge the assistant's last response"
    )

    # Build zone1 with hidden state appended
    zone1 = translate_result.zone1_context
    if hidden_state_prompt:
        zone1 = f"{zone1}\n\n{hidden_state_prompt}"

    prompt = _GENERATION_PROMPT_V2.format(
        zone1_context=zone1,
        framing_instruction=translate_result.framing_instruction or "",
        acknowledgment_hint=ack_hint,
        zone2_block=zone2_block,
        recent_history=recent,
    )

    if concise_prompt and turn_input is not None:
        prompt = _concise_turn_prompt(turn_input, aim, translate_result)

    request_messages = (_concise_messages(prompt, aim) if concise_prompt and turn_input is not None
                        else [{"role": "user", "content": prompt}])
    if messages_renderer is not None:
        request_messages = messages_renderer(request_messages, aim=aim, history=messages)

    try:
        response = call_stage(trace, "GENERATE", generate_chat_completion,
            model=llm,
            messages=request_messages,
            temperature=temperature,
            max_tokens=16384,
            **(llm_args or {}),
        )

        content = response.strip()

        # Strip wrapping quotes
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]

        # v6.0+: Degenerate LLM output (empty/etc) — fail loudly instead of
        # silently substituting zone2 template. Template fallback masked real
        # LLM failures and produced runs that looked successful but had
        # template-only user messages contaminating downstream stats.
        if not content or _is_degenerate_response(content) and translate_result.zone2_mandatory_sentences:
            raise RuntimeError(
                "Bridge V2 generation produced degenerate output (empty or trivial). "
                "Refusing to fall back to zone2 template — fix the LLM call instead."
            )

        # Verify mandatory sentences were included
        content = repair(trace, "_verify_mandatory_sentences", _verify_mandatory_sentences,
            content,
            translate_result.zone2_mandatory_sentences,
            llm=llm,
            llm_args=llm_args,
            trace=trace,
        )

        return content

    except Exception as exc:
        logger.error(f"Bridge V2 generation failed: {exc}, no fallback — propagating error")
        raise RuntimeError(f"Bridge V2 generation failed after retries: {exc}") from exc


# ── Helpers ─────────────────────────────────────────────────────────────────


def _verify_mandatory_sentences(
    content: str,
    mandatory: List[str],
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    trace=None,
) -> str:
    """Check that each mandatory Zone 2 sentence was semantically included.

    Uses keyword overlap for fast check, then LLM retry if items are missing.
    """
    if not mandatory:
        return content

    content_lower = content.lower()
    missing = []
    for sentence in mandatory:
        # Extract significant words (>3 chars) from the mandatory sentence
        words = [w for w in re.sub(r'[^\w\s]', '', sentence.lower()).split() if len(w) > 3]
        if not words:
            continue
        matches = sum(1 for w in words if w in content_lower)
        if matches / len(words) < 0.3:
            missing.append(sentence)

    if not missing:
        return content

    logger.warning(
        f"V2 mandatory sentences missing ({len(missing)}/{len(mandatory)}), retrying"
    )

    if not llm:
        raise RuntimeError(
            f"V2 mandatory sentences missing ({len(missing)}/{len(mandatory)}) "
            f"and no LLM available for retry. Refusing to fall back to "
            f"sentence-append template — fix the upstream LLM call."
        )

    missing_block = "\n".join(f"- {s}" for s in missing)
    retry_prompt = (
        f"The following user message is missing required content.\n\n"
        f"ORIGINAL MESSAGE: {content}\n\n"
        f"MISSING CONTENT THAT MUST BE INCLUDED:\n{missing_block}\n\n"
        f"Rewrite the message to naturally include ALL missing content. "
        f"Keep the same tone and style. 1-4 sentences.\n"
        f"Output ONLY the rewritten message:"
    )
    revised = call_stage(trace, "CONTENT_REPAIR", generate_chat_completion,
        model=llm,
        messages=[{"role": "user", "content": retry_prompt}],
        temperature=0.5,
        max_tokens=16384,
        **(llm_args or {}),
    )
    if revised and len(revised.strip()) > 10:
        return revised.strip()

    raise RuntimeError(
        "V2 mandatory sentence retry returned empty/trivial output. "
        "Refusing to fall back to sentence-append template."
    )


def _extract_critical_tokens_regex(
    payload_items: List[Tuple[str, str, bool]],
) -> List[str]:
    """Extract critical numeric tokens from payload items using regex.

    Fast, deterministic extraction of values that must survive paraphrasing:
    dollar amounts, percentages, measurements, durations, product names.
    """
    tokens = []
    seen = set()
    for _key, value, _exact in payload_items:
        # Dollar amounts: $8, $350, $1,800.50
        for m in re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', value):
            if m not in seen:
                tokens.append(m)
                seen.add(m)
        # Percentages: 7%, 0.5%, 2.5%
        for m in re.findall(r'\d+(?:\.\d+)?\s*%', value):
            if m not in seen:
                tokens.append(m)
                seen.add(m)
        # Measurements: 30ml, 240ml, 50mg
        for m in re.findall(r'\d+\s*(?:ml|mg|oz|g)\b', value, re.I):
            if m not in seen:
                tokens.append(m)
                seen.add(m)
        # Duration: 6 weeks, 30 days, 8 weeks pregnant
        for m in re.findall(r'\d+\s+(?:weeks?|days?|months?)\b', value, re.I):
            if m not in seen:
                tokens.append(m)
                seen.add(m)
        # Specific product names (multi-word capitalized, >5 chars)
        for m in re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', value):
            if m not in seen and len(m) > 5:
                tokens.append(m)
                seen.add(m)
    return tokens


def _verify_critical_tokens_llm(
    content: str,
    payload_items: List[Tuple[str, str, bool]],
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    trace=None,
) -> str:
    """LLM-based verification that critical factual tokens survived paraphrasing.

    After generation, checks whether all important numbers, product names,
    and measurements from payload items appear in the output. If missing,
    asks the LLM to revise the message to include them.
    """
    if not llm or not payload_items:
        return content

    # Build a concise summary of critical facts from all payload items
    facts_lines = []
    for key, value, _exact in payload_items:
        # Extract just the factual core (numbers, products, specifics)
        facts = []
        facts.extend(re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', value))
        facts.extend(re.findall(r'\d+(?:\.\d+)?\s*%', value))
        facts.extend(re.findall(r'\d+\s*(?:ml|mg|oz|g)\b', value, re.I))
        facts.extend(re.findall(r'\d+\s+(?:weeks?|days?|months?)\b', value, re.I))
        if facts:
            facts_lines.append(f"  - {', '.join(facts)} (from: {value[:80]})")

    if not facts_lines:
        return content

    verify_prompt = f"""Check if this user message includes all the critical factual values listed below.

MESSAGE: "{content}"

CRITICAL VALUES THAT MUST APPEAR:
{chr(10).join(facts_lines)}

If ALL values are present (exact numbers, not rounded), respond with just: PASS
If any values are MISSING, respond with a revised message that naturally includes the missing values while keeping the same tone and length. Output ONLY the revised message."""

    try:
        response = call_stage(trace, "CONTENT_REPAIR", generate_chat_completion,
            model=llm,
            messages=[{"role": "user", "content": verify_prompt}],
            temperature=0.3,
            max_tokens=16384,
            **(llm_args or {}),
        )
        result = response.strip()
        if result.upper().startswith("PASS"):
            return content
        # LLM returned a revised message
        if len(result) > 10:
            logger.info("LLM critical token verification: revised message to include missing values")
            return result
    except Exception as exc:
        logger.warning(f"LLM critical token verification failed: {exc}")

    return content


def _format_recent(messages: List[AnyMessage], last_n: int = 3) -> str:
    recent = messages[-last_n:] if len(messages) > last_n else messages
    parts: List[str] = []
    for msg in recent:
        if isinstance(msg, UserMessage):
            parts.append(f"User: {msg.content}")
        elif isinstance(msg, AssistantMessage):
            text = msg.content or "[no content]"
            if len(text) > 300:
                text = text[:300] + "..."
            parts.append(f"Assistant: {text}")
    return "\n".join(parts) if parts else "(No previous conversation)"


def _ensure_exact_info(content: str, exact_items: List[Tuple[str, str]],
                       llm: Optional[str] = None, trace=None) -> str:
    """Ensure critical exact values (emails, phones, dollar amounts) are present.

    Uses LLM re-generation when available, with regex-extracted fallback.
    """
    if not exact_items:
        return content

    content_lower = content.lower()
    missing: List[Tuple[str, str]] = []  # (key, regex-extracted critical value)

    for key, value in exact_items:
        # Check emails
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', value)
        if emails and not any(e.lower() in content_lower for e in emails):
            missing.append((key, emails[0]))
            continue

        # Check phone numbers
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', value)
        if phones and not any(
            p.replace('-', '').replace('.', '') in content.replace('-', '').replace('.', '')
            for p in phones
        ):
            missing.append((key, phones[0]))
            continue

        # Check dollar amounts
        dollars = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?[kKmM]?', value)
        if dollars and not any(d.lower() in content_lower for d in dollars):
            missing.append((key, dollars[0]))
            continue

        # Check dates
        dates = re.findall(
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},? \d{4})\b', value
        )
        if dates and not any(d.lower() in content_lower for d in dates):
            missing.append((key, dates[0]))
            continue

    if not missing:
        return content

    logger.warning(
        f"Exact items missing, attempting re-generation: {[k for k, _ in missing]}"
    )

    if not llm:
        raise RuntimeError(
            f"Exact items missing ({[k for k, _ in missing]}) and no LLM "
            "available for revision. Refusing to fall back to regex-appendix "
            "template — fix the upstream LLM call."
        )

    missing_desc = "\n".join(f"- {k}: {v}" for k, v in missing)
    revision_prompt = (
        "The following message was generated but is missing critical information.\n"
        "Revise to naturally include the missing items without changing tone or meaning.\n\n"
        f"ORIGINAL MESSAGE: {content}\n\n"
        f"MISSING INFORMATION:\n{missing_desc}\n\n"
        "Revised message (1-4 sentences, same tone):"
    )
    revised = call_stage(trace, "CONTENT_REPAIR", generate_chat_completion,
        model=llm,
        messages=[{"role": "user", "content": revision_prompt}],
        temperature=0.5,
        max_tokens=16384,
    )
    if revised and len(revised.strip()) > 10:
        return revised.strip()

    raise RuntimeError(
        "Payload item revision returned empty/trivial output. "
        "Refusing to fall back to regex-appendix template."
    )


def _verify_payload_semantics(
    content: str,
    payload_items: List[Tuple[str, str, bool]],
    aim: Aim,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    trace=None,
) -> str:
    """Verify payload items were semantically delivered in the generated message.

    Uses an LLM to check whether each non-exact payload item was conveyed
    with correct values, framing, and specificity. If items fail verification,
    retries generation with targeted corrections.

    Only runs for trap/final aims where payload delivery is critical.
    Falls back to keyword-based check if no LLM is available.
    """
    if not payload_items:
        return content

    # Only verify for trap/final aims
    if not getattr(aim, 'is_final', False):
        return content

    # Collect non-exact items to verify (exact items handled by _ensure_exact_info)
    items_to_verify = [(k, v) for k, v, exact in payload_items if not exact]
    if not items_to_verify:
        return content

    # If no LLM available, fall back to keyword check
    if not llm:
        return _keyword_payload_check(content, items_to_verify)

    # Build verification prompt
    items_text = "\n".join(f"  {i+1}. [{k}]: {v}" for i, (k, v) in enumerate(items_to_verify))

    verify_prompt = f"""Check whether a generated user message correctly conveys each required payload item.

GENERATED MESSAGE:
"{content}"

REQUIRED PAYLOAD ITEMS:
{items_text}

For each item, check:
1. NUMERIC VALUES: Are all numbers, prices, dates, measurements exact matches? ($350 ≠ $250)
2. FRAMING/INTENT: Does the message convey the same intent? (proposing X ≠ asking about X)
3. SPECIFICITY: Are specific details preserved? ("BP + retinol same night" ≠ "incorporating BP")

Respond with a JSON object:
{{"items": [
  {{"key": "<key>", "verdict": "PASS" or "FAIL", "reason": "<brief reason if FAIL>"}}
]}}

Be STRICT about numeric accuracy and framing. Even small number differences ($250 vs $350) are FAIL."""

    try:
        import json
        raw = call_stage(trace, "CONTENT_REPAIR", generate_chat_completion,
            model=llm,
            messages=[{"role": "user", "content": verify_prompt}],
            temperature=0.0,
            max_tokens=16384,
            **(llm_args or {}),
        )

        # Parse JSON from response (handle code fences)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        result = json.loads(raw)
        items_result = result.get("items", [])

        failed_items = [
            item for item in items_result
            if item.get("verdict", "").upper() == "FAIL"
        ]

        if not failed_items:
            logger.debug("Semantic payload verification: all items PASS")
            return content

        # Build correction prompt for failed items
        corrections = "\n".join(
            f"  - [{item['key']}]: {item.get('reason', 'incorrect')}. "
            f"Correct value: {dict(items_to_verify).get(item['key'], '?')}"
            for item in failed_items
        )

        logger.warning(
            f"Semantic payload verification: {len(failed_items)} item(s) FAIL — "
            f"{[item['key'] for item in failed_items]}. Retrying."
        )

        retry_prompt = (
            f"The following user message has INCORRECT information that must be fixed:\n\n"
            f"ORIGINAL MESSAGE: {content}\n\n"
            f"ERRORS TO FIX:\n{corrections}\n\n"
            f"Rewrite the message fixing ONLY the errors above. Keep the same tone, "
            f"style, and length. Do NOT add new information or change anything else.\n"
            f"Output ONLY the corrected message (1-4 sentences):"
        )

        retry_response = call_stage(trace, "CONTENT_REPAIR", generate_chat_completion,
            model=llm,
            messages=[{"role": "user", "content": retry_prompt}],
            temperature=0.5,
            max_tokens=16384,
            **(llm_args or {}),
        )
        result_text = retry_response.strip()
        if result_text.startswith('"') and result_text.endswith('"'):
            result_text = result_text[1:-1]
        if result_text and len(result_text) > 10:
            logger.info(f"Semantic retry successful, corrected {len(failed_items)} item(s)")
            return result_text

        raise RuntimeError(
            "Semantic payload retry returned empty/trivial output. "
            "Refusing to silently keep original (incorrect) content."
        )

    except Exception as exc:
        raise RuntimeError(
            f"Semantic payload verification failed ({exc}). Refusing to "
            "silently keep original (incorrect) content."
        ) from exc


def _keyword_payload_check(content: str, items: List[Tuple[str, str]]) -> str:
    """Keyword-based check when LLM is not available — fail loudly if missing.

    Previously this returned `content` unchanged with only a warning, which
    silently produced sim results that looked complete but had missing
    payload items. Now raises so callers fix the LLM/route instead of
    accepting a degraded run.
    """
    content_lower = content.lower()
    missing = []
    for key, value in items:
        words = [w for w in value.lower().split() if len(w) > 3]
        if words:
            matches = sum(1 for w in words if w in content_lower)
            if matches / len(words) < 0.3:
                missing.append((key, value))
    if missing:
        raise RuntimeError(
            f"Keyword payload check: {len(missing)} items below 30% threshold "
            f"({[k for k, _ in missing]}) and no LLM available to revise. "
            "Refusing to silently keep incomplete content."
        )
    return content
