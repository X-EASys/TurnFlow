"""
Policy Module — Stage 2 of the BRIDGE Pipeline

DECIDE: Pure-rule policy engine (no LLM call).

Given a PerceptionResult + current Aim + state-tracker context,
produces a TurnPolicy that tells the GENERATE stage exactly what to do.

Priority order (first match wins):
  1. CORRECTION        — violations detected
  2. INFO_FILLING      — agent asked questions / info gaps exist
  3. CONTRADICTION_EXPOSE — agent output conflicts with user constraints
  4. CONSTRAINT_ESCALATION — time to reveal a constraint
  5. CLOSURE_PUSH      — aim nearly satisfied, nudge toward completion
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple

from loguru import logger

_INVALID_HINT_RE = re.compile(
    r'\b(she|he|the user|user will|user should|should shift|'
    r'must now|will follow up|will introduce|will describe|'
    r'will ask|will share|will mention)\b', re.IGNORECASE
)

from benchmark.data_model.aim import Aim, AimType
from benchmark.data_model.scenario import HiddenStateItem
from benchmark.user.perception import PerceptionResult


# ── Transition Type ─────────────────────────────────────────────────────────


class TransitionType(str, Enum):
    """The kind of conversational move the user should make."""

    CORRECTION = "correction"
    """Agent did something wrong → user corrects / complains."""

    INFO_FILLING = "info_filling"
    """Agent asked a question or there are info gaps → user provides info."""

    CONTRADICTION_EXPOSE = "contradiction_expose"
    """Agent's output contradicts user constraint → user exposes the conflict."""

    CONSTRAINT_ESCALATION = "constraint_escalation"
    """Time to mention a new constraint the user hasn't shared yet."""

    CLOSURE_PUSH = "closure_push"
    """Aim is nearly / fully satisfied → user wraps up or moves on."""


# ── Turn Policy ─────────────────────────────────────────────────────────────


@dataclass
class TurnPolicy:
    """The output of the DECIDE stage — a fully deterministic action plan."""

    # Which transition template to use
    transition_type: TransitionType = TransitionType.CLOSURE_PUSH

    # The payload the user should deliver (info items, corrections, etc.)
    payload_items: List[Tuple[str, str, bool]] = field(default_factory=list)
    """List of (key, value, exact_disclosure) tuples to include in the message."""

    # Questions from the agent that the user should answer
    questions_to_answer: List[str] = field(default_factory=list)

    # Violations to address
    violations_to_address: List[str] = field(default_factory=list)

    # Constraint keys being escalated this turn
    constraint_keys_to_reveal: List[str] = field(default_factory=list)

    # Emotional tone hint (derived from tension_level)
    emotion_hint: str = "neutral"

    # Whether this is the FIRST message for a new aim
    is_aim_initiation: bool = False

    # Whether this is a trap/final aim (payload delivery is critical)
    is_trap_aim: bool = False

    # Whether to advance to next aim after generating this message
    should_advance_after: bool = False

    # Optional transition hint from the aim definition
    transition_hint: str = ""

    # Latent preference trigger instructions (from hidden state)
    trigger_instructions: List[str] = field(default_factory=list)

    # Debug: reason for choosing this transition type
    decision_reason: str = ""

    # Aim description — used by trap skeleton for framing classification (A3)
    _aim_description: str = ""

    # Aim type — used by skeleton builder for PROBE/DRIFT selection (T16/T17/T18)
    _aim_type: Optional[str] = None

    # Probe forbidden words — passed to PROBE skeleton
    _probe_forbidden_words: List[str] = field(default_factory=list)


# ── Emotion mapping ─────────────────────────────────────────────────────────

_TENSION_TO_EMOTION = {
    0: "calm and friendly",
    1: "slightly curious",
    2: "mildly concerned",
    3: "noticeably frustrated",
    4: "quite irritated",
    5: "very upset and impatient",
}


# ── Public API ──────────────────────────────────────────────────────────────


def decide_policy(
    perception: PerceptionResult,
    aim: Aim,
    info_items: List[Tuple[str, str, bool]],
    is_new_aim: bool = False,
    hidden_state: Optional[List[HiddenStateItem]] = None,
    trigger_instructions: Optional[List[str]] = None,
) -> TurnPolicy:
    """
    Pure-rule policy engine. No LLM call — deterministic.

    Args:
        perception: Output of the PERCEIVE stage.
        aim: Current Aim being pursued.
        info_items: Available (key, value, exact) info items for this aim.
        is_new_aim: True if the user is starting a brand-new aim.
        hidden_state: Optional hidden-state items.
        trigger_instructions: Latent preference triggers from perception.

    Returns:
        A fully populated TurnPolicy.
    """
    policy = TurnPolicy()
    policy.trigger_instructions = trigger_instructions or []
    policy.is_aim_initiation = is_new_aim
    policy.is_trap_aim = getattr(aim, 'is_final', False)
    policy._aim_description = getattr(aim, 'description', '') or ''

    # Emotion from tension
    policy.emotion_hint = _TENSION_TO_EMOTION.get(
        perception.tension_level, "neutral"
    )

    # Transition hint from aim (with validation for Bug 2.2+2.6)
    if hasattr(aim, "transition_hint") and aim.transition_hint:
        if not _INVALID_HINT_RE.search(aim.transition_hint):
            policy.transition_hint = aim.transition_hint
        else:
            logger.warning(f"Rejected invalid transition hint: {aim.transition_hint[:60]}...")

    # ── Killer aim_type overrides (T16/T17/T18) ──────────────────────────
    aim_type_val = getattr(aim, 'aim_type', None)

    # For special aim types (INFO_SCATTER/DRIFT/PROBE), reset emotion to neutral.
    # These types bypass the general CORRECTION cascade, so tension-based emotion
    # has no feedback loop and can only escalate — conflicting with hidden_state
    # communication_style and causing probe answer leakage.
    if aim_type_val in (AimType.DRIFT, AimType.PROBE, AimType.INFO_SCATTER):
        policy.emotion_hint = "neutral"

    if aim_type_val == AimType.DRIFT:
        policy.transition_type = TransitionType.INFO_FILLING
        policy.is_aim_initiation = True
        drift_config = getattr(aim, 'drift_config', None)
        if drift_config:
            drift_text = f"{drift_config.drift_subject}: {drift_config.drift_topic}"
            policy.payload_items = [("drift", drift_text, False)]
        policy.decision_reason = f"DRIFT aim — tangent about {getattr(drift_config, 'drift_subject', 'someone') if drift_config else 'someone'}"
        policy._aim_type = "drift"
        return policy

    if aim_type_val == AimType.PROBE:
        policy.transition_type = TransitionType.CLOSURE_PUSH
        policy.is_aim_initiation = True
        policy.is_trap_aim = True
        policy.payload_items = info_items
        probe_config = getattr(aim, 'probe_config', None)
        if probe_config:
            policy._probe_forbidden_words = probe_config.forbidden_words
        else:
            policy._probe_forbidden_words = []
        policy.decision_reason = "PROBE aim — forward-pushing evaluation trigger"
        policy._aim_type = "probe"
        return policy

    if aim_type_val == AimType.INFO_SCATTER:
        policy.transition_type = TransitionType.INFO_FILLING
        policy.is_aim_initiation = True
        policy.payload_items = info_items
        if perception.agent_asked_questions:
            policy.questions_to_answer = perception.agent_asked_questions
        policy.decision_reason = "INFO_SCATTER aim — deliver scattered info piece"
        policy._aim_type = "info_scatter"
        return policy

    # ── aim_mode override for Tier 3 types ──────────────────────────────
    aim_mode = getattr(aim, 'aim_mode', 'disclosure') or 'disclosure'
    if aim_mode == "pushback":
        policy.transition_type = TransitionType.CORRECTION
        policy.decision_reason = "Pushback aim — user disagrees with agent"
        policy.payload_items = info_items  # v6.0: include payload for TRANSLATE
        if perception.agent_asked_questions:
            policy.questions_to_answer = perception.agent_asked_questions
        return policy
    elif aim_mode == "cooperative":
        policy.transition_type = TransitionType.CLOSURE_PUSH
        policy.decision_reason = "Cooperative aim — user confirms agent's plan"
        policy.payload_items = info_items  # v6.0: include payload for TRANSLATE
        if perception.agent_asked_questions:
            policy.questions_to_answer = perception.agent_asked_questions
        return policy
    elif aim_mode == "withhold":
        policy.transition_type = TransitionType.INFO_FILLING
        policy.decision_reason = "Withhold aim — user avoids revealing key info"
        # Deliberately empty payload — withhold mode does NOT disclose
        if perception.agent_asked_questions:
            policy.questions_to_answer = perception.agent_asked_questions
        return policy
    elif aim_mode == "ambiguous":
        policy.transition_type = TransitionType.INFO_FILLING
        policy.decision_reason = "Ambiguous aim — user provides ambiguous reference"
        policy.payload_items = info_items  # v6.0: include payload
        if perception.agent_asked_questions:
            policy.questions_to_answer = perception.agent_asked_questions
        return policy

    # ── Priority 0: Aim initiation ──────────────────────────────────────
    if is_new_aim:
        policy.transition_type = TransitionType.INFO_FILLING
        policy.payload_items = info_items  # disclose all planned info for this aim
        # Even for a new aim, if perception detected agent questions from
        # the previous turn, include them so the user naturally answers
        # while introducing the new aim.
        if perception.agent_asked_questions:
            policy.questions_to_answer = perception.agent_asked_questions
        policy.decision_reason = "New aim — provide initial info and make request"
        return policy

    # ── Priority 1: CORRECTION — violations exist ───────────────────────
    if perception.violations:
        policy.transition_type = TransitionType.CORRECTION
        policy.violations_to_address = perception.violations
        # Still provide info if agent also asked questions
        if perception.agent_asked_questions:
            policy.questions_to_answer = perception.agent_asked_questions
            policy.payload_items = info_items[:1]  # answer one question at most
        policy.decision_reason = (
            f"Violations detected: {perception.violations[:2]}"
        )
        return policy

    # ── Priority 2: INFO_FILLING — agent asked questions / gaps exist ───
    if perception.agent_asked_questions or perception.gaps:
        policy.transition_type = TransitionType.INFO_FILLING
        policy.questions_to_answer = perception.agent_asked_questions
        # Provide up to 2 info items to fill gaps
        policy.payload_items = info_items[:2]
        policy.decision_reason = (
            f"Agent questions: {perception.agent_asked_questions[:2]}, "
            f"Gaps: {perception.gaps[:2]}"
        )
        return policy

    # ── Priority 3: CONTRADICTION_EXPOSE ────────────────────────────────
    # Check if hidden-state preferences conflict with what the agent said
    if hidden_state:
        conflicting = _detect_preference_conflicts(perception, hidden_state)
        if conflicting:
            policy.transition_type = TransitionType.CONTRADICTION_EXPOSE
            policy.violations_to_address = conflicting
            policy.decision_reason = (
                f"Latent preference conflict: {conflicting[:2]}"
            )
            return policy

    # ── Priority 4: CONSTRAINT_ESCALATION ───────────────────────────────
    # If there are still info items to disclose (constraints / info not yet told)
    if info_items:
        policy.transition_type = TransitionType.CONSTRAINT_ESCALATION
        policy.payload_items = info_items[:1]  # reveal one constraint
        policy.constraint_keys_to_reveal = [info_items[0][0]]
        policy.decision_reason = (
            f"Escalating constraint: {info_items[0][0]}"
        )
        return policy

    # ── Priority 5: CLOSURE_PUSH ────────────────────────────────────────
    if perception.aim_should_advance:
        policy.transition_type = TransitionType.CLOSURE_PUSH
        policy.should_advance_after = True
        policy.decision_reason = "All criteria satisfied — closure"
        return policy

    # Default: closure push even if not fully satisfied (to avoid loops)
    policy.transition_type = TransitionType.CLOSURE_PUSH
    policy.decision_reason = "No actionable items remain — nudging closure"
    return policy


# ── Internal helpers ────────────────────────────────────────────────────────


def _detect_preference_conflicts(
    perception: PerceptionResult,
    hidden_state: List[HiddenStateItem],
) -> List[str]:
    """
    Check if the assistant's acts conflict with the user's latent preferences.
    Returns a list of conflict description strings.
    """
    conflicts: List[str] = []
    pref_items = [h for h in hidden_state if h.category.is_preference]

    if not pref_items:
        return conflicts

    # Simple keyword overlap check between assistant acts and preference triggers
    for h in pref_items:
        if h.disclosed_during_conversation:
            continue
        if not h.trigger_condition:
            continue

        trigger_lower = h.trigger_condition.lower()
        trigger_words = [w for w in trigger_lower.split() if len(w) > 3]
        if not trigger_words:
            continue

        # Check against assistant acts
        for act in perception.assistant_acts:
            act_lower = act.lower()
            match_count = sum(1 for w in trigger_words if w in act_lower)
            if trigger_words and match_count / len(trigger_words) >= 0.4:
                conflicts.append(
                    f"Assistant's action '{act}' conflicts with user preference: {h.value}"
                )
                break

    return conflicts
