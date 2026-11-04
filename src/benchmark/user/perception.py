"""
Perception Module — Stage 1 of the BRIDGE Pipeline

PERCEIVE: Analyze the assistant's last message against the current aim's
success criteria to produce a structured PerceptionResult.

Uses a single LLM call (gpt-4o-mini, temperature=0.0) with strict JSON
output to extract:
  - What the assistant DID (acts)
  - Which success criteria are met / unmet
  - Whether there are violations of forbidden elements
  - Tension level (0-5) indicating user frustration
  - Whether the aim should advance
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from benchmark.user.first_pass import call_stage, repair
from loguru import logger

from benchmark.data_model.aim import Aim, AimStatus
from benchmark.data_model.message import AnyMessage, AssistantMessage, UserMessage
from benchmark.data_model.scenario import HiddenStateItem
from benchmark.utils.llm_utils import generate_chat_completion


PERCEPTION_PARSER_POLICY = "strict-json-string-crlf-v1"


# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class PerceptionResult:
    """Structured output of the PERCEIVE stage."""

    # What the assistant did in its last message
    assistant_acts: List[str] = field(default_factory=list)

    # Questions the assistant asked the user
    agent_asked_questions: List[str] = field(default_factory=list)

    # Success-criteria that are already met
    satisfied_criteria: List[str] = field(default_factory=list)

    # Success-criteria that are NOT yet met
    unsatisfied_criteria: List[str] = field(default_factory=list)

    # Information gaps — things the user still needs to tell the agent
    gaps: List[str] = field(default_factory=list)

    # Violations — things the agent did that conflict with user constraints
    violations: List[str] = field(default_factory=list)

    # Tension level (0 = calm, 5 = very frustrated)
    tension_level: int = 0

    # Human-readable reason for the tension level
    tension_reason: str = ""

    # Whether the aim should be considered satisfied and advanced
    aim_should_advance: bool = False

    schema_warnings: List[dict] = field(default_factory=list)

    # Raw LLM response (for debugging)
    raw_response: str = ""

    # Equivalent JSON formatting recovery, separate from content/schema changes.
    format_recoveries: List[dict] = field(default_factory=list)


# ── Perception Prompt ───────────────────────────────────────────────────────

_PERCEPTION_PROMPT = """You are an objective evaluator. Analyze the assistant's last message
relative to the user's current goal and success criteria.

## USER'S CURRENT GOAL
{aim_description}

## SUCCESS CRITERIA
{success_criteria}

## FORBIDDEN / CONSTRAINTS
{forbidden_elements}

## ASSISTANT'S LAST MESSAGE
{assistant_message}

## RECENT CONVERSATION (last 4 turns)
{recent_history}

## INSTRUCTIONS
Produce a JSON object with EXACTLY these keys:
{{
  "assistant_acts": ["<what the assistant DID, 1-3 short phrases>"],
  "agent_asked_questions": ["<questions the assistant asked, if any>"],
  "satisfied_criteria": ["<criteria that ARE met>"],
  "unsatisfied_criteria": ["<criteria that are NOT met>"],
  "gaps": ["<info the user still needs to provide>"],
  "violations": ["<things that conflict with user constraints, if any>"],
  "tension_level": <0-5 integer>,
  "tension_reason": "<why this tension level>",
  "aim_should_advance": <true if ALL success criteria are met, else false>
}}

IMPORTANT:
- All list fields must be arrays of strings, never objects. tension_level must be an integer
  from 0 to 5 (not boolean); aim_should_advance must be a JSON boolean, not a string.
  Missing fields default to empty arrays, tension_level=0, tension_reason="",
  and aim_should_advance=false. tension_reason must be a string.
- Be strict: only mark aim_should_advance=true if ALL criteria are genuinely satisfied.
- If the assistant asked a question, note it in agent_asked_questions.
- Return ONLY the JSON object, no other text."""


# ── Public API ──────────────────────────────────────────────────────────────


def perceive(
    aim: Aim,
    assistant_message: str,
    messages: List[AnyMessage],
    hidden_state: Optional[List[HiddenStateItem]] = None,
    constraints_text: str = "",
    llm: str = "gpt-4o-mini",
    llm_args: Optional[dict] = None,
    trace=None,
    turn_input=None,
    diagnostic_acts: bool = False,
) -> PerceptionResult:
    """
    Run the PERCEIVE stage: analyse the assistant's last response.

    Args:
        aim: The current Aim being pursued.
        assistant_message: The assistant's last message content.
        messages: Full conversation history (for context window).
        hidden_state: Optional hidden-state items (for constraint awareness).
        constraints_text: Pre-formatted string of user constraints.
        llm: Model identifier for the evaluation call.
        llm_args: Extra kwargs forwarded to the LLM.

    Returns:
        A populated PerceptionResult.
    """
    if not assistant_message.strip():
        # Edge case: empty assistant response → nothing satisfied
        return PerceptionResult(
            unsatisfied_criteria=[aim.success_criteria],
            tension_level=1,
            tension_reason="Assistant gave an empty response",
        )

    # Build forbidden-elements text from constraints + hidden-state
    forbidden = constraints_text
    if hidden_state:
        pref_items = [h for h in hidden_state if h.category.is_preference]
        if pref_items:
            forbidden += "\nUser latent preferences (may conflict):\n"
            for h in pref_items:
                forbidden += f"- {h.value}\n"

    # Format recent history
    recent = _format_recent(messages, last_n=4)

    prompt = _PERCEPTION_PROMPT.format(
        aim_description=aim.description or "",
        success_criteria=aim.success_criteria or "",
        forbidden_elements=forbidden or "(none)",
        assistant_message=assistant_message,
        recent_history=recent,
    )

    if turn_input is not None:
        prompt = (
            "Describe the assistant's conversational response using only the material below. "
            "Do not solve the user's task or infer undisclosed facts. Return JSON with "
            "assistant_acts (array of strings, never objects), agent_asked_questions (array of strings), tension_level (integer 0-5, not boolean), "
            "and aim_should_advance (JSON boolean, not a string). Missing fields default to empty arrays, 0 and false respectively.\n\n" + turn_input["text"]
        )

    phase = "provider"
    try:
        raw = call_stage(trace, "PERCEIVE", generate_chat_completion,
            model=llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16384,
            **(llm_args or {}),
        )
        phase = "schema"
        result = _parse_perception(raw, strict=trace is not None or turn_input is not None,
                                   diagnostic_acts=diagnostic_acts)
        if trace is not None:
            trace.stages[-1]["schema_warnings"] = result.schema_warnings
            trace.stages[-1]["format_recoveries"] = result.format_recoveries
        return result
    except Exception as exc:
        if trace is not None:
            trace.stages[-1]["error"] = type(exc).__name__
            trace.stages[-1]["error_phase"] = phase
        if trace is not None or turn_input is not None:
            raise
        logger.error(f"Perception LLM call failed: {exc}")
        # Conservative fallback — assume nothing satisfied
        return PerceptionResult(
            unsatisfied_criteria=[aim.success_criteria],
            tension_level=0,
            tension_reason="Perception failed, conservative fallback",
            raw_response=str(exc),
        )


# ── Internal helpers ────────────────────────────────────────────────────────


def normalize_literal_newlines(raw: str) -> tuple[str, list[dict]]:
    """Escape only unescaped CR/LF inside JSON strings, retaining source offsets."""
    inside = escaped = False
    output, changes = [], []
    for offset, char in enumerate(raw):
        if inside and not escaped and char in '\r\n':
            replacement = '\\r' if char == '\r' else '\\n'
            output.append(replacement)
            changes.append(dict(offset=offset, original=char, replacement=replacement))
            continue
        output.append(char)
        if escaped:
            escaped = False
        elif inside and char == '\\':
            escaped = True
        elif char == '"':
            inside = not inside
    return ''.join(output), changes


def _perception_json_text(raw: str) -> str:
    """Retain the existing whitespace and markdown-fence handling."""
    raw = raw.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)
    return raw


def _parse_perception(raw: str, strict: bool = False, diagnostic_acts: bool = False) -> PerceptionResult:
    """Parse PERCEIVE JSON; strict mode recovers only literal string CR/LF.

    Recovery is a local formatting operation, followed by the same JSON and
    schema validation. The original response and each replacement remain visible.
    """
    original_raw = raw
    raw = _perception_json_text(raw)
    format_recoveries = []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        if strict:
            normalized, changes = normalize_literal_newlines(original_raw)
            if not changes:
                raise
            data = json.loads(_perception_json_text(normalized))
            format_recoveries.append(dict(kind='perceive-literal-newline-v1',
                raw_response=original_raw, normalized_text=normalized, changes=changes))
        else:
            logger.warning(f"Perception JSON parse failed, raw={raw[:200]}")
            return PerceptionResult(
                tension_reason="Could not parse perception JSON",
                raw_response=raw,
            )

    if not isinstance(data, dict):
        raise ValueError("PERCEIVE schema: expected JSON object")
    warnings = []
    for name in ("assistant_acts", "agent_asked_questions", "satisfied_criteria",
                 "unsatisfied_criteria", "gaps", "violations"):
        value = data.get(name, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            if name == "assistant_acts" and diagnostic_acts:
                warnings.append(dict(field=name, issue="expected array of strings", available=False))
                data[name] = []
            else:
                raise ValueError(f"PERCEIVE schema: {name} must be an array of strings")
    tension = data.get("tension_level", 0)
    if type(tension) is not int or not 0 <= tension <= 5:
        raise ValueError("PERCEIVE schema: tension_level must be integer 0-5")
    if type(data.get("aim_should_advance", False)) is not bool:
        raise ValueError("PERCEIVE schema: aim_should_advance must be boolean")
    if not isinstance(data.get("tension_reason", ""), str):
        raise ValueError("PERCEIVE schema: tension_reason must be string")

    return PerceptionResult(
        assistant_acts=data.get("assistant_acts", []),
        agent_asked_questions=data.get("agent_asked_questions", []),
        satisfied_criteria=data.get("satisfied_criteria", []),
        unsatisfied_criteria=data.get("unsatisfied_criteria", []),
        gaps=data.get("gaps", []),
        violations=data.get("violations", []),
        tension_level=data.get("tension_level", 0),
        tension_reason=data.get("tension_reason", ""),
        aim_should_advance=data.get("aim_should_advance", False),
        raw_response=original_raw,
        schema_warnings=warnings,
        format_recoveries=format_recoveries,
    )


def _format_recent(messages: List[AnyMessage], last_n: int = 4) -> str:
    """Format the last N messages for the perception prompt."""
    recent = messages[-last_n:] if len(messages) > last_n else messages
    parts: List[str] = []
    for msg in recent:
        if isinstance(msg, UserMessage):
            parts.append(f"User: {msg.content}")
        elif isinstance(msg, AssistantMessage):
            text = msg.content or "[no content]"
            if len(text) > 400:
                text = text[:400] + "..."
            parts.append(f"Assistant: {text}")
    return "\n".join(parts) if parts else "(no history)"
