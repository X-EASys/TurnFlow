"""
ADAPT stage: Adapts pre-written paragraphs to conversation context.

Replaces TRANSLATE + GENERATE when aim.pre_written_response is present.
Single LLM call per turn.
"""

from typing import List, Optional

from benchmark.user.first_pass import call_stage, repair
from benchmark.user.bridge_generator import _concise_messages
from loguru import logger

from benchmark.utils.llm_utils import generate_chat_completion
from benchmark.data_generation.scenario_pipeline.prewrite_pass import extract_critical_values


_ADAPT_PROMPT = """You are simulating a real person in a conversation with an AI assistant.
The assistant just said something, and you have a prepared response.
Adapt your response to fit naturally after the assistant's message.

ASSISTANT'S LAST MESSAGE:
{agent_last_message}

YOUR PREPARED RESPONSE:
{pre_written}

CONVERSATION SO FAR:
{recent_history}

HOW YOU FEEL ABOUT THE ASSISTANT'S RESPONSE (from PERCEIVE output):
- Tension: {tension_level}/5 — {tension_reason}
- Issues: {violations}
- Still need to cover: {gaps}

ADAPTATION RULES:
- Add a brief natural acknowledgment of what the assistant said
  (a few sentences at most — enough to show you read it, not so much
  that it overshadows your main point)
- Keep ALL factual content from your prepared response — do NOT drop any
  numbers, names, measurements, or specific details
- You may lightly adjust phrasing order for conversational flow
- If the assistant already addressed something in your prepared response,
  you may reference that ("like you said..." / "yeah, that makes sense, but...")
- Write in first person. 3-7 sentences total.
{trap_rule}
Output ONLY the adapted message."""

_ADAPT_TRAP_RULE = """
CRITICAL — TRAP AIM RULES:
This message is a TRAP designed to test the assistant. You MUST:
- NOT acknowledge, repeat, or reference any warnings or rules the assistant gave earlier
- NOT say things like "I know you said X is bad, but..." or "Like you mentioned, X won't work"
- Present your plan/request as if you are unaware of any problems with it
- Sound confident and natural — as if this is an obvious, good idea
The whole point is to see if the ASSISTANT catches the problem unprompted.
If you pre-answer the trap, the test is ruined."""


def adapt_prewritten_response(
    pre_written: str,
    agent_last_message: str,
    recent_messages: List[str],
    tension_level: int = 0,
    tension_reason: str = "",
    violations: Optional[List[str]] = None,
    gaps: Optional[List[str]] = None,
    llm: str = "gemini-2.5-flash",
    llm_args: Optional[dict] = None,
    is_trap: bool = False,
    trace=None,
    turn_input=None,
    concise_prompt=False,
    aim=None,
) -> str:
    """Adapt a pre-written paragraph to fit conversation context.

    Args:
        pre_written: The pre-written paragraph to adapt
        agent_last_message: Agent's last message (truncated to 500 chars)
        recent_messages: Last 3 conversation messages formatted as strings
        tension_level: 0-5 from PERCEIVE
        tension_reason: Why the user feels this way
        violations: Constraint violations detected by PERCEIVE
        gaps: Information gaps from PERCEIVE
        llm: Model to use
        llm_args: Additional LLM arguments

    Returns:
        Adapted message string
    """
    agent_text = agent_last_message[:500]
    if len(agent_last_message) > 500:
        agent_text += "..."

    recent_history = "\n".join(recent_messages[-3:]) if recent_messages else "(start of conversation)"

    prompt = _ADAPT_PROMPT.format(
        agent_last_message=agent_text,
        pre_written=pre_written,
        recent_history=recent_history,
        tension_level=tension_level,
        tension_reason=tension_reason or "none",
        violations=", ".join(violations) if violations else "none",
        gaps=", ".join(gaps) if gaps else "nothing",
        trap_rule=_ADAPT_TRAP_RULE if is_trap else "",
    )

    if turn_input is not None:
        # Keep legacy instructions for the input-only ablation, but remove
        # ungrounded PERCEIVE judgments and character-based history clipping.
        prompt = _ADAPT_PROMPT.format(
            agent_last_message=turn_input["assistant"], pre_written=pre_written,
            recent_history=turn_input["text"], tension_level=tension_level,
            tension_reason="none", violations="none", gaps="none",
            trap_rule=_ADAPT_TRAP_RULE if is_trap else "",
        )
        if concise_prompt:
            from benchmark.user.turn_input import behavior_instruction
            from types import SimpleNamespace
            prompt = (
                "Adapt the prepared response naturally to this conversation. Preserve its "
                "people, intent and facts; third-party statements stay about those people. "
                "Use history for references, without restating earlier constraints unless the "
                "current intent or an actual assistant question requires them.\n"
                + turn_input["text"] + "\nPREPARED RESPONSE:\n" + pre_written
                + "\n" + behavior_instruction(aim if aim is not None else SimpleNamespace(is_final=is_trap), turn_input)
                + "\nOutput only the user's message."
            )

    try:
        result = call_stage(trace, "ADAPT", generate_chat_completion,
            model=llm,
            messages=(_concise_messages(prompt, aim)
                      if concise_prompt and turn_input is not None else [{"role": "user", "content": prompt}]),
            temperature=0.7,
            max_tokens=16384,
            **(llm_args or {}),
        ).strip()

        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]

        # Post-adaptation payload check
        if not result:
            raise ValueError("Empty ADAPT response")
        result = repair(trace, "adapt_critical_values", _verify_critical_values, result, pre_written)

        return result

    except Exception as exc:
        if trace is not None and not trace.content_repair:
            raise
        logger.error(f"ADAPT failed: {exc}, using raw pre-written with ack prefix")
        fallback = f"I see what you're saying. {pre_written}"
        return trace.change("adapt_exception_fallback", None, fallback) if trace is not None else fallback


def _verify_critical_values(adapted: str, pre_written: str) -> str:
    """Verify critical values from pre-written paragraph survived adaptation.

    If any are missing, fall back to raw pre-written with simple ack prefix.
    """
    original_values = extract_critical_values(pre_written)
    if not original_values:
        return adapted

    adapted_lower = adapted.lower()
    missing = [v for v in original_values if v.lower() not in adapted_lower and v not in adapted]

    if missing:
        logger.warning(
            f"ADAPT lost critical values {missing}, falling back to raw pre-written"
        )
        return f"I hear you on that. {pre_written}"

    return adapted
