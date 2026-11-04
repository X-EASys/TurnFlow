"""
Aim-Driven User Simulator

A sophisticated user simulator that executes multi-aim, multi-task scenarios.
Instead of pre-defining each turn's content, this simulator:

1. Works through a sequence of Aims (user goals/tasks)
2. For each Aim: provides relevant info → makes request → evaluates response
3. Transitions to next Aim when current is satisfied or times out
4. Uses LLM to dynamically generate natural user messages

This enables complex scenarios like document editing where users
progressively modify content through multiple related requests.

Task-Specific Behavior Support (v2.0):
- T1 (Self-Reference): Vanilla behavior with history validation
- T3 (Proactivity): INFORMATION_HIDING - only reveal info when asked
- T5 (Negative Constraint): Vanilla with constraint violation detection
- T6 (Ambiguity Resolution): HIDDEN_INTENT_LOCK - reveal intent only when clarified
- T8 (Reference Resolution): ENTITY_MASKING - use references instead of entity names
"""

import re
from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Any
from loguru import logger

from benchmark.data_model.scenario import (
    Scenario, StateQueue,
    HiddenStateItem, HiddenStateCategory, OverlapMode,
    _classify_constraint_strength,
)
from benchmark.data_model.aim import (
    Aim, AimSequence, AimStatus, AimType,
    ENABLE_CLARIFICATION_HINTS,  # v2.1: Global config for clarification behavior
)
from benchmark.data_model.message import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    AnyMessage,
    MessageRole,
)
from benchmark.user.base import BaseUser, UserState, UserContext
from benchmark.user.user_simulator import UserSimulator
from benchmark.user.state_tracker import StateTracker
from benchmark.utils.llm_utils import generate_chat_completion
from benchmark.user.prompts import build_hidden_state_prompt_section
from benchmark.user.task_specific_controls import (
    TaskSpecificController,
    TaskControlConfig,
    SimulatorControlType,
    create_controller_from_scenario,
)

# v5.0: BRIDGE pipeline imports
from benchmark.user.perception import perceive, PerceptionResult
from benchmark.user.policy import decide_policy, TurnPolicy, TransitionType
from benchmark.user.bridge_generator import generate_bridge_message
from benchmark.user.translate import translate_for_aim, TranslateResult


# ── Metadata leak prevention patterns ──────────────────────────────────────
# Detect evaluator/pipeline language that should never appear in user messages

_METADATA_VALUE_PATTERNS = [
    # Agent-prescriptive language (evaluation criteria, not user speech)
    re.compile(r'\b(?:the\s+)?agent\s+(?:must|should|needs?\s+to|is\s+expected\s+to)\b', re.I),
    re.compile(r'\b(?:the\s+)?agent\s+explicitly\s+(?:states?|mentions?|confirms?|recommends?)\b', re.I),
    re.compile(r'\bmust\s+(?:warn|avoid|recommend|check|verify|confirm)\b', re.I),
    # Pipeline parameter references
    re.compile(r'\b(?:reasoning_hops|difficulty_level|complexity_budget|evaluation_focus)\b', re.I),
    re.compile(r'\b(?:required_elements?|forbidden_elements?|success_criteria)\b', re.I),
    re.compile(r'\b(?:core_capability_tested|constraint_satisfaction|info_extraction)\b', re.I),
    # Trailing "key: number" patterns like "reasoning_hops_required: 4"
    re.compile(r'\b\w+_(?:required|level|budget|weight|score)\s*:\s*\d+'),
]

_PIPELINE_KEY_PATTERNS = [
    re.compile(r'^(?:agent_|eval_|scoring_|validation_|hidden_state_)'),
    re.compile(r'_(?:required|expected|criteria|threshold|weight)$'),
]

# ── A1: Auto-promote numeric values to exact_disclosure ───────────────────
# Dollar, percentage, date, or measurement patterns that indicate exact values
_EXACT_PROMOTION_PATTERNS = [
    re.compile(r'\$\d'),                              # Dollar amounts: $350, $1,800
    re.compile(r'\d+\s*%'),                           # Percentages: 15%, 20 %
    re.compile(r'\d+\s*(?:USD|EUR|GBP|CAD|AUD)', re.I),  # Currency with code: 1800 USD
    re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),    # Dates: 6/15/2026, 15-06-26
    re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}', re.I),  # June 15
    re.compile(r'\b\d{1,2}(?:st|nd|rd|th)\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)', re.I),  # 15th of June
    re.compile(r'\d+\s*(?:ml|mg|oz|kg|lb|g|cm|mm|m|ft|in)\b', re.I),  # Measurements: 30ml, 50mg
]


def _should_promote_to_exact(value: str) -> bool:
    """Check if a value contains numeric/date/measurement content that should be exact."""
    for pattern in _EXACT_PROMOTION_PATTERNS:
        if pattern.search(value):
            return True
    return False


# Strength-aware rewrite templates for constraint sanitization
_STRENGTH_REWRITE_PREFIXES = {
    "absolute": [
        "I absolutely cannot {}",
        "This is non-negotiable — {}",
        "No exceptions on this: {}",
    ],
    "strong": [
        "It's really important that {}",
        "I need to make sure {}",
        "I'm firm on this — {}",
    ],
    "soft": [
        "I'd prefer if {}",
        "If possible, {}",
        "Ideally, {}",
    ],
}


class AimStateTracker:
    """
    Tracks the state of aim execution
    """
    
    def __init__(self):
        self.started_aims: set = set()
        self.aim_entry_turns: Dict[str, int] = {}  # aim_id -> turn when entered
    
    def mark_aim_started(self, aim_id: str, turn: int) -> None:
        """Mark an aim as started"""
        self.started_aims.add(aim_id)
        self.aim_entry_turns[aim_id] = turn
    
    def is_new_aim(self, aim_id: str) -> bool:
        """Check if this aim hasn't been started yet"""
        return aim_id not in self.started_aims
    
    def reset(self) -> None:
        """Reset tracker state"""
        self.started_aims.clear()
        self.aim_entry_turns.clear()


class AimDrivenUserSimulator(UserSimulator):
    """
    User Simulator driven by Aim sequences.
    
    This simulator extends the base UserSimulator to support multi-aim
    scenarios where the user has multiple sequential goals to accomplish.
    
    Task-Specific Behavior (v2.0):
    - Integrates TaskSpecificController for T1, T3, T5, T6, T8 support
    - Modifies prompt generation based on task type
    - Enforces task-specific constraints on user behavior
    
    State Machine:
    
    ┌─────────────────────────────────────────────────────────────┐
    │   NEW AIM                                                    │
    │   ├── Disclose required info (one or more items)            │
    │   └── Make request based on aim description                  │
    │                                                              │
    │   AGENT RESPONDS                                             │
    │   ├── Evaluate: Does response satisfy success_criteria?      │
    │   │   ├── YES → Mark SATISFIED, advance to next aim         │
    │   │   └── NO  → Provide clarification or more details       │
    │   │                                                          │
    │   TURN LIMIT                                                 │
    │   └── If turns_spent >= max_turns → Mark TIMEOUT, advance   │
    │                                                              │
    │   COMPLETION                                                 │
    │   └── All aims processed → End conversation                  │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        scenario: Optional[Scenario] = None,
        llm: str = "gpt-4o-mini",
        llm_args: Optional[dict] = None,
        enable_state_tracking: bool = True,
        temperature: float = 0.7,
        strict_single_info_per_turn: bool = False,
        evaluation_llm: Optional[str] = None,
        task_controller: Optional[TaskSpecificController] = None,
        trace=None,
        experiment_variant="V0",
        perception_llm_args=None,
    ):
        """
        Initialize the aim-driven user simulator.
        
        Args:
            scenario: The scenario to simulate
            llm: LLM model for response generation
            llm_args: Additional LLM arguments
            enable_state_tracking: Whether to use state tracker
            temperature: Temperature for generation
            strict_single_info_per_turn: Whether to limit info disclosure
            evaluation_llm: LLM for aim satisfaction evaluation (defaults to same as llm)
            task_controller: Optional TaskSpecificController for task-type-specific behavior
        """
        # Initialize aim-specific attributes BEFORE calling super().__init__
        # because super().__init__ may call set_scenario
        if experiment_variant not in ("V0", "V1", "V2", "V3"):
            raise ValueError("Unknown simulator experiment variant")
        self.experiment_variant = experiment_variant
        if experiment_variant != "V0":
            from benchmark.user.first_pass import FirstPassTrace
            trace = trace if trace is not None else FirstPassTrace()
            if trace.content_repair:
                raise ValueError("Candidate experiments require content_repair=off")
        self.trace = trace
        self.perception_llm_args = {**(llm_args or {}), **(perception_llm_args or {})}
        self._controller_template = deepcopy(task_controller)
        self.evaluation_llm = evaluation_llm or llm
        self.aim_state_tracker = AimStateTracker()
        self._aim_sequence: Optional[AimSequence] = None
        
        # Task-specific controller (v2.0)
        self._task_controller: Optional[TaskSpecificController] = task_controller
        
        # Call parent __init__ WITHOUT scenario first
        super().__init__(
            scenario=None,  # Don't pass scenario here
            llm=llm,
            llm_args=llm_args,
            enable_state_tracking=enable_state_tracking,
            temperature=temperature,
            strict_single_info_per_turn=strict_single_info_per_turn,
        )
        
        # Now set scenario using our own set_scenario method
        if scenario:
            self.set_scenario(scenario)
    
    def set_scenario(self, scenario: Scenario) -> None:
        """Set the scenario and initialize aim sequence"""
        self._scenario_template = deepcopy(scenario)
        scenario = deepcopy(scenario)
        self.scenario = scenario
        UserSimulator.reset(self)
        self.state_tracker = StateTracker(scenario)
        self._system_prompt = None
        self._is_translated = bool(
            (getattr(scenario, "metadata", None) or {}).get("translated", False)
            if scenario else False
        )

        # Extract aim sequence from state_queue if present
        if hasattr(scenario.state_queue, 'aim_sequence') and scenario.state_queue.aim_sequence:
            self._aim_sequence = scenario.state_queue.aim_sequence
            self._aim_sequence.reset()
        else:
            self._aim_sequence = None
        
        self.aim_state_tracker.reset()
        
        # v4.0: Extract hidden state for persistent persona injection
        self._hidden_state: List[HiddenStateItem] = []
        if hasattr(scenario.state_queue, 'hidden_state') and scenario.state_queue.hidden_state:
            self._hidden_state = list(scenario.state_queue.hidden_state)
        
        # Pre-build hidden state prompt section (reused every turn)
        self._hidden_state_prompt: str = build_hidden_state_prompt_section(
            self._hidden_state if self._hidden_state else None
        )
        
        # Initialize task-specific controller (v2.0)
        self._task_controller = deepcopy(self._controller_template)
        if self._task_controller is None:
            self._task_controller = create_controller_from_scenario(scenario)
        if self._task_controller:
            self._task_controller.reset()
            # v4.0: Pass hidden state items to task controller for task-specific integration
            if self._hidden_state:
                self._task_controller.set_hidden_state(self._hidden_state)
            logger.info(f"Task controller initialized for task type: {self._task_controller.get_task_type()}")
    
    def reset(self) -> None:
        """Reset for a new conversation"""
        if self.scenario:
            self.set_scenario(self._scenario_template)
        else:
            super().reset()

    @property
    def task_controller(self) -> Optional[TaskSpecificController]:
        """Get the task-specific controller"""
        return self._task_controller
    
    @property
    def aim_sequence(self) -> Optional[AimSequence]:
        """Get the aim sequence"""
        return self._aim_sequence
    
    @property
    def is_aim_driven(self) -> bool:
        """Check if this scenario uses aim-driven mode"""
        return self._aim_sequence is not None and len(self._aim_sequence.aims) > 0
    
    def step(
        self,
        messages: List[AnyMessage],
    ) -> Tuple[UserMessage, UserState]:
        """
        Generate the next user message.
        
        If scenario has aim_sequence, uses aim-driven logic.
        Otherwise, falls back to standard info-driven behavior.
        
        Args:
            messages: Conversation history
            
        Returns:
            Tuple of (UserMessage, UserState)
        """
        if not self.is_aim_driven:
            # Fall back to standard behavior
            return super().step(messages)
        
        # Aim-driven mode
        # Copy related mutable state together to preserve controller/hidden-state aliases.
        names = ("scenario", "context", "_aim_sequence", "aim_state_tracker", "_hidden_state",
                 "_task_controller", "state_tracker", "_anchor_revealed")
        snapshot = deepcopy({name: getattr(self, name) for name in names})
        stage_start = len(self.trace.stages) if self.trace is not None else 0
        try:
            return self._aim_driven_step(messages)
        except Exception as exc:
            if self.trace is not None and not any(
                stage["error"] for stage in self.trace.stages[stage_start:]
            ):
                self.trace.execution_errors.append(dict(
                    phase="aim_driven_step", error=type(exc).__name__,
                    stage_start=stage_start, stage_end=len(self.trace.stages)))
            for name, value in snapshot.items():
                setattr(self, name, value)
            raise
    
    def _aim_driven_step(
        self,
        messages: List[AnyMessage],
    ) -> Tuple[UserMessage, UserState]:
        """
        Execute aim-driven step logic using the BRIDGE pipeline.

        v5.0: Refactored to use 3-stage pipeline:
            PERCEIVE → DECIDE → GENERATE

        Args:
            messages: Conversation history

        Returns:
            Tuple of (UserMessage, UserState)
        """
        if self._aim_sequence is None:
            raise ValueError("Aim sequence not set")

        # Check total turn limit
        if self._aim_sequence.is_total_timeout():
            logger.info("Total turn limit reached, ending conversation")
            return self._generate_timeout_end_message()

        # Get current aim
        current_aim = self._aim_sequence.get_current_aim()

        if current_aim is None:
            # All aims completed
            logger.info("All aims completed")
            return self._generate_satisfaction_message()

        # Check if this is a new aim
        is_new = self.aim_state_tracker.is_new_aim(current_aim.aim_id)

        if is_new:
            # Mark aim started
            current_aim.status = AimStatus.IN_PROGRESS
            self.aim_state_tracker.mark_aim_started(
                current_aim.aim_id, self.context.current_turn
            )

        scoped = self.experiment_variant != "V0"
        turn_input = None
        if scoped:
            from benchmark.user.turn_input import current_payload, build_turn_input
            scoped_payload = current_payload(self.scenario, current_aim)
            if getattr(current_aim, "aim_mode", "") == "withhold":
                scoped_payload = []
            if self._task_controller:
                scoped_payload = [item for item in scoped_payload if
                    self._task_controller.should_allow_info_reveal(
                        item[0].split(".", 1)[-1], self._get_last_agent_message(messages))]
            turn_input = build_turn_input(self.scenario, current_aim, messages, scoped_payload)

        # ── STAGE 1: PERCEIVE ───────────────────────────────────────────
        last_agent_msg = self._get_last_agent_message(messages)

        if is_new and not last_agent_msg:
            # Very first message of conversation — synthesize a minimal perception
            perception = PerceptionResult(
                assistant_acts=[],
                aim_should_advance=False,
            )
        else:
            # Always run full perception when there IS an agent message,
            # even for new aims — we need to understand agent questions,
            # tension level, and potential hidden state triggers.
            constraints_text = "" if scoped else self._build_constraints_text()
            perception = perceive(
                trace=self.trace,
                turn_input=turn_input,
                diagnostic_acts=self.experiment_variant == "V3",
                aim=current_aim,
                assistant_message=last_agent_msg,
                messages=messages,
                hidden_state=None if scoped else (self._hidden_state or None),
                constraints_text=constraints_text,
                llm=self.evaluation_llm,
                llm_args=self.perception_llm_args,
            )

        logger.debug(
            f"PERCEIVE aim='{current_aim.aim_id}' → advance={perception.aim_should_advance}, "
            f"tension={perception.tension_level}, violations={len(perception.violations)}"
        )

        # ── Handle aim satisfaction (from perception) ───────────────────
        # In one-turn-per-aim design, we don't advance early based on
        # perception. Each aim gets exactly 1 user turn. Advance happens
        # AFTER generating the message (see post-generate section below).
        # But if the agent FULLY satisfied a continuing aim, we can skip.
        if not is_new and perception.aim_should_advance:
            logger.info(f"Aim '{current_aim.aim_id}' satisfied by agent, advancing")
            next_aim = self._aim_sequence.advance_to_next_aim(AimStatus.SATISFIED)

            if next_aim is None:
                return self._generate_satisfaction_message()

            # Recurse into next aim as a new aim
            return self._aim_driven_step(messages)

        # ── Handle aim timeout ──────────────────────────────────────────
        if not is_new:
            current_aim.increment_turn()
            if current_aim.is_timeout():
                logger.info(f"Aim '{current_aim.aim_id}' timeout, forcing advance")
                next_aim = self._aim_sequence.advance_to_next_aim(AimStatus.TIMEOUT)
                if next_aim is None:
                    return self._generate_satisfaction_message()
                return self._aim_driven_step(messages)

        # ── Increment counters ──────────────────────────────────────────
        self.context.increment_turn()
        self._aim_sequence.increment_total_turns()
        if self._task_controller:
            self._task_controller.increment_turn()

        # ── Collect info items to disclose ──────────────────────────────
        info_items = scoped_payload if scoped else self._get_info_for_aim(current_aim)

        # Apply task-specific filtering
        if self._task_controller:
            filtered = []
            for key, value, exact in info_items:
                if self._task_controller.should_allow_info_reveal(key, last_agent_msg):
                    filtered.append((key, value, exact))
                else:
                    logger.debug(f"Task controller blocked info reveal: {key}")
            info_items = filtered

        # NOTE: Disclosure marking moved to AFTER generation (v6.0 fix).
        # Previously, items were marked as disclosed before GENERATE ran,
        # causing false-disclosure: items appeared "disclosed" even when
        # the LLM dropped them from the output.
        if is_new:
            current_aim.request_made = True

        # ── Check latent preference triggers ────────────────────────────
        trigger_instructions: List[str] = []
        if last_agent_msg and not scoped:
            trigger_instructions = self._check_latent_preference_triggers(
                last_agent_msg, messages
            )

        # ── STAGE 2: DECIDE ─────────────────────────────────────────────
        policy = decide_policy(
            perception=perception,
            aim=current_aim,
            info_items=info_items,
            is_new_aim=is_new,
            hidden_state=None if scoped else (self._hidden_state or None),
            trigger_instructions=trigger_instructions,
        )

        if scoped:
            # Free-text judgments are diagnostic, not a source for the user's speech.
            policy.questions_to_answer = []
            policy.violations_to_address = []
            policy.trigger_instructions = []
            policy.transition_hint = ""
            policy._probe_forbidden_words = []

        logger.debug(
            f"DECIDE aim='{current_aim.aim_id}' → type={policy.transition_type.value}, "
            f"reason='{policy.decision_reason}'"
        )

        # ── STAGE 2.75: ADAPT (pre-written paragraph) ──────────────────
        tr = None       # Initialize for metadata block (only set in TRANSLATE branch)
        skeleton = None  # Initialize for metadata block (only set in GENERATE branch)
        if current_aim.pre_written_response:
            from benchmark.user.adapt import adapt_prewritten_response

            # Format recent messages for context
            recent_formatted = []
            for msg in messages[-3:]:
                if hasattr(msg, 'content'):
                    role = "User" if isinstance(msg, UserMessage) else "Assistant"
                    text = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                    recent_formatted.append(f"{role}: {text}")

            agent_last = ""
            if messages and hasattr(messages[-1], 'content'):
                agent_last = messages[-1].content

            content = adapt_prewritten_response(
                trace=self.trace,
                turn_input=turn_input,
                concise_prompt=self.experiment_variant in ("V2", "V3"),
                aim=current_aim,
                pre_written=current_aim.pre_written_response,
                agent_last_message=agent_last,
                recent_messages=recent_formatted,
                tension_level=getattr(perception, 'tension_level', 0),
                tension_reason=getattr(perception, 'tension_reason', ''),
                violations=getattr(perception, 'violations', []),
                gaps=getattr(perception, 'gaps', []),
                llm=self.llm,
                llm_args=self.llm_args,
                is_trap=getattr(current_aim, 'is_final', False),
            )

            logger.info(f"Used ADAPT stage for aim '{current_aim.aim_id}' "
                       f"({len(content)} chars)")
        else:
            # ── STAGE 2.5: TRANSLATE ─────────────────────────────────────
            if self._is_translated:
                # Pre-translated scenarios: skip runtime translate entirely.
                # Values are already first-person from Stage 2.5 (translate_pass.py).
                tr = None
            elif self.experiment_variant == "V3":
                tr = TranslateResult(
                    zone1_context="",
                    zone2_mandatory_sentences=[value for _, value, _ in info_items],
                )
            else:
                tr = translate_for_aim(
                    trace=self.trace,
                    turn_input=turn_input,
                    aim=current_aim,
                    policy=policy,
                    personality=self.scenario.personality if self.scenario else "Friendly",
                    domain="current conversation" if scoped else (self.scenario.domain if self.scenario else "general"),
                    llm=self.evaluation_llm,
                    llm_args=self.llm_args,
                    is_translated=False,
                    aim_mode=getattr(current_aim, 'aim_mode', 'disclosure'),  # v6.0
                )

            if tr is not None:
                logger.debug(
                    f"TRANSLATE aim='{current_aim.aim_id}' → "
                    f"zone1_len={len(tr.zone1_context)}, "
                    f"zone2_count={len(tr.zone2_mandatory_sentences)}, "
                    f"framing='{tr.framing_instruction[:50]}'"
                )
            else:
                logger.debug(
                    f"TRANSLATE aim='{current_aim.aim_id}' → skipped (pre-translated)"
                )

            # ── STAGE 3: GENERATE ────────────────────────────────────────
            inherited_context = self._aim_sequence.get_inherited_context(current_aim)

            # Pre-build the skeleton for diagnostics (also used inside generate_bridge_message)
            from benchmark.user.bridge_templates import build_skeleton as _build_skeleton
            acts_summary = "responded" if self.experiment_variant == "V3" else (
                ", ".join(perception.assistant_acts[:3]) if perception.assistant_acts else "responded")
            skeleton = _build_skeleton(policy, acts_summary, is_translated=self._is_translated)

            # Get task-specific prompt additions
            task_specific_prompt = ""
            if self._task_controller and not scoped:
                task_specific_prompt = self._task_controller.get_task_prompt_additions()

            content = generate_bridge_message(
                messages_renderer=self._render_generation_messages,
                trace=self.trace,
                turn_input=turn_input,
                concise_prompt=self.experiment_variant in ("V2", "V3"),
                policy=policy,
                perception=PerceptionResult(tension_level=perception.tension_level) if scoped else perception,
                aim=current_aim,
                personality=self.scenario.personality if self.scenario else "Friendly",
                domain="current conversation" if scoped else (self.scenario.domain if self.scenario else "general"),
                messages=messages,
                hidden_state_prompt="" if scoped else (self._hidden_state_prompt or ""),
                task_specific_prompt=task_specific_prompt,
                llm=self.llm,
                llm_args=self.llm_args,
                temperature=self.temperature,
                translate_result=tr,
                is_translated=self._is_translated,
            )

        # Apply task-specific post-processing
        if self._task_controller:
            from benchmark.user.first_pass import repair
            content = repair(self.trace, "controller_postprocess",
                             self._task_controller.process_user_message, content, messages)

        # Keyword overlap is diagnostic only: it cannot establish who said what.
        disclosure_hints = []
        content_lower = content.lower()
        for info_key, value, _ in info_items:
            words = [w for w in value.lower().split() if len(w) > 3]
            ratio = sum(w in content_lower for w in words) / len(words) if words else 0.0
            disclosure_hints.append({"key": info_key, "keyword_overlap": ratio})

        # ── Build metadata (v5.1: comprehensive BRIDGE diagnostics) ───
        metadata = {
            # ── Aim state ──
            "experiment_variant": self.experiment_variant,
            "turn_input": turn_input,
            "delivery_status": "emitted",
            "semantic_success": "unreviewed",
            "disclosure_tracking": "keyword_hint_only",
            "disclosure_hints": disclosure_hints,
            "aim_id": current_aim.aim_id,
            "aim_type": current_aim.aim_type.value,
            "aim_description": current_aim.description or "",
            "aim_success_criteria": current_aim.success_criteria or "",
            "aim_turns_spent": current_aim.turns_spent,
            "aim_max_turns": current_aim.max_turns,
            "is_new_aim": is_new,
            "is_final_aim": current_aim.is_final,
            # ── Stage 1: PERCEIVE ──
            "perception": {
                "assistant_acts": perception.assistant_acts,
                "schema_warnings": perception.schema_warnings,
                "format_recoveries": perception.format_recoveries,
                "raw_response": perception.raw_response,
                "agent_asked_questions": perception.agent_asked_questions,
                "satisfied_criteria": perception.satisfied_criteria,
                "unsatisfied_criteria": perception.unsatisfied_criteria,
                "gaps": perception.gaps,
                "violations": perception.violations,
                "tension_level": perception.tension_level,
                "tension_reason": perception.tension_reason,
                "aim_should_advance": perception.aim_should_advance,
            },
            # ── Stage 2: DECIDE ──
            "policy": {
                "transition_type": policy.transition_type.value,
                "decision_reason": policy.decision_reason,
                "emotion_hint": policy.emotion_hint,
                "payload_items": [
                    {"key": k, "value": v, "exact": e}
                    for k, v, e in policy.payload_items
                ],
                "questions_to_answer": policy.questions_to_answer,
                "violations_to_address": policy.violations_to_address,
                "constraint_keys_to_reveal": policy.constraint_keys_to_reveal,
                "is_aim_initiation": policy.is_aim_initiation,
                "should_advance_after": policy.should_advance_after,
                "transition_hint": policy.transition_hint,
                "trigger_instructions": policy.trigger_instructions,
            },
            # ── Stage 2.5: TRANSLATE / ADAPT ──
            "translate": {
                "zone1_context": tr.zone1_context if tr else None,
                "zone2_mandatory_sentences": tr.zone2_mandatory_sentences if tr else None,
                "framing_instruction": tr.framing_instruction if tr else None,
                "emotion_hint": tr.emotion_hint if tr else None,
                "acknowledgment_hint": tr.acknowledgment_hint if tr else None,
                "skipped": tr is None or self.experiment_variant == "V3",
                "used_adapt": bool(current_aim.pre_written_response),
            },
            # ── Stage 3: GENERATE context ──
            "generation": {
                "skeleton_template": skeleton,
                "exact_items_count": len(
                    [(k, v) for k, v, e in policy.payload_items if e]
                ),
                "hidden_state_prompt_length": len(self._hidden_state_prompt or ""),
                "trigger_block": "\n".join(policy.trigger_instructions) if policy.trigger_instructions else "",
            },
            # ── Hidden state status ──
            "hidden_state_status": [
                {
                    "key": hs.key,
                    "category": hs.category.value,
                    "disclosed": hs.disclosed_during_conversation,
                    "trigger_condition": hs.trigger_condition or "",
                    "trigger_keywords": hs.trigger_keywords if hasattr(hs, 'trigger_keywords') and hs.trigger_keywords else [],
                }
                for hs in self._hidden_state
            ] if self._hidden_state else [],
            # ── Global progress ──
            "global_turn": self.context.current_turn,
            "total_aim_turns": self._aim_sequence.total_turns_spent if self._aim_sequence else 0,
        }

        message = UserMessage(
            content=content,
            turn_idx=self.context.current_turn,
            metadata=metadata,
        )

        logger.debug(
            f"GENERATE aim='{current_aim.aim_id}' [{policy.transition_type.value}]: "
            f"{content[:100]}..."
        )

        # ── Post-generate: ONE-TURN-PER-AIM — always advance ────────────
        # Each aim = exactly 1 user turn. After generating the message,
        # unconditionally advance to the next aim.
        next_aim = self._aim_sequence.advance_to_next_aim(AimStatus.SATISFIED)

        # If this was the final aim (no next aim), signal conversation complete
        if next_aim is None and current_aim.is_final:
            return message, UserState.SATISFIED

        return message, UserState.ACTIVE
    
    def _build_constraints_text(self) -> str:
        """Build a text summary of user constraints for perception."""
        if not self.scenario:
            return ""
        sq = self.scenario.state_queue
        lines: List[str] = []
        for item in sq.constraints:
            if hasattr(item, 'value'):
                lines.append(f"- {item.value}")
        return "\n".join(lines) if lines else "(no constraints)"

    def _handle_new_aim(
        self,
        aim: Aim,
        messages: List[AnyMessage],
    ) -> Tuple[UserMessage, UserState]:
        """
        Handle entering a new aim.
        
        DEPRECATED in v5.0: The BRIDGE pipeline in _aim_driven_step now handles
        new aims directly. This method is kept for backward compatibility but
        delegates to _aim_driven_step.
        """
        # The new pipeline handles everything — just re-enter the main loop
        # (aim_state_tracker.is_new_aim will still return True)
        return self._aim_driven_step(messages)
    
    def _get_info_for_aim(self, aim: Aim) -> List[Tuple[str, str, bool]]:
        """
        Get information items to disclose for an aim.
        
        v2.1 Enhancement: Filters out already-disclosed constraints for memory-testing
        task types (T1, T5) to prevent the simulator from repeating them.
        
        Args:
            aim: The aim to get info for
            
        Returns:
            List of (key, value, exact_disclosure) tuples for info to disclose.
            exact_disclosure=True means the value MUST be included verbatim (numbers, emails, names).
        """
        if not self.scenario:
            return []
        
        sq = self.scenario.state_queue
        flow_info = sq.flow_info
        constraints = sq.constraints
        distractors = getattr(sq, 'distractor', None) or []
        
        result = []
        
        # Get undisclosed required info
        undisclosed_keys = aim.get_undisclosed_info_keys()
        
        # v6.0: Build set of keys required by this specific aim
        aim_required_keys = set(aim.required_info_keys) | set(aim.required_constraint_keys)

        for key in undisclosed_keys:
            # v2.1: Check if task controller blocks repetition
            # v6.0: Pass whether this key is required by the current aim
            if self._task_controller:
                if not self._task_controller.should_repeat_constraint_in_message(
                    key, is_required_by_current_aim=(key in aim_required_keys)
                ):
                    logger.info(f"Skipping already-disclosed constraint: {key}")
                    continue
            
            found = False
            # Search in flow_info
            for item in flow_info:
                if hasattr(item, 'key') and item.key == key:
                    exact = getattr(item, 'exact_disclosure', False)
                    # For inferential items, use surface_form and disable exact_disclosure
                    effective_value = item.get_effective_surface() if hasattr(item, 'get_effective_surface') else item.value
                    if hasattr(item, 'delivery_mode') and item.delivery_mode == 'inferential':
                        exact = False
                    result.append((key, effective_value, exact))
                    found = True
                    break

            if not found:
                # Search in constraints
                for item in constraints:
                    if hasattr(item, 'key') and item.key == key:
                        exact = getattr(item, 'exact_disclosure', False)
                        effective_value = item.get_effective_surface() if hasattr(item, 'get_effective_surface') else item.value
                        if hasattr(item, 'delivery_mode') and item.delivery_mode == 'inferential':
                            exact = False
                        result.append((key, effective_value, exact))
                        found = True
                        break
            
            if not found:
                # Search in distractors (trap/hazard items the user should mention)
                for item in distractors:
                    item_key = item.key if hasattr(item, 'key') else item.get('key', '')
                    if item_key == key:
                        item_value = item.value if hasattr(item, 'value') else item.get('value', '')
                        result.append((key, item_value, False))
                        logger.debug(f"Found info key '{key}' in distractors")
                        break
        
        # Also add constraint info if needed
        for key in aim.required_constraint_keys:
            # v2.1: Check if task controller blocks repetition
            # v6.0: required_constraint_keys are always required by the current aim
            if self._task_controller:
                if not self._task_controller.should_repeat_constraint_in_message(
                    key, is_required_by_current_aim=True
                ):
                    logger.info(f"Skipping already-disclosed required constraint: {key}")
                    continue
            
            if key not in [k for k, _, _ in result]:
                for item in constraints:
                    if hasattr(item, 'key') and item.key == key:
                        exact = getattr(item, 'exact_disclosure', False)
                        # For inferential items, use surface_form and disable exact_disclosure
                        effective_value = item.get_effective_surface() if hasattr(item, 'get_effective_surface') else item.value
                        if hasattr(item, 'delivery_mode') and item.delivery_mode == 'inferential':
                            exact = False
                        result.append((key, effective_value, exact))
                        break
        
        # Sanitize payload values to prevent metadata leakage
        sanitized = []
        for key, value, exact in result:
            if self._is_translated:
                cleaned = value  # Pre-translated, skip sanitization
                # For translated scenarios: values are already first-person natural
                # text. Forcing exact reproduction causes verbatim paragraph copying.
                # Critical numbers (%, $, ml) are preserved via exact_warning separately.
                exact = False
            else:
                cleaned = self._sanitize_payload_value(key, value)
                # A1: Auto-promote numeric/date/measurement values to exact_disclosure
                if cleaned is not None and not exact and _should_promote_to_exact(cleaned):
                    logger.debug(f"Auto-promoting '{key}' to exact_disclosure (contains numeric/date value)")
                    exact = True
            if cleaned is not None:
                sanitized.append((key, cleaned, exact))
        return sanitized

    def _sanitize_payload_value(self, key: str, value: str) -> Optional[str]:
        """Sanitize a payload value to remove evaluator/pipeline metadata.

        Returns sanitized value, or None if the entire value is metadata
        (should be dropped from the payload).
        """
        if value is None:
            logger.warning(f"Dropping payload item with None value: {key}")
            return None

        # Check if the key itself is a pipeline metadata key
        for pattern in _PIPELINE_KEY_PATTERNS:
            if pattern.search(key):
                logger.warning(f"Dropping pipeline metadata key from payload: {key}")
                return None

        # Check value for metadata patterns
        for pattern in _METADATA_VALUE_PATTERNS:
            if pattern.search(value):
                logger.warning(
                    f"Metadata detected in payload value for '{key}': "
                    f"'{value[:80]}...' — sanitizing"
                )
                # Determine constraint strength for appropriate rewriting
                strength = _classify_constraint_strength(value)

                # Extract the core action from agent-prescriptive language
                # Match patterns like "The agent must avoid X" or "The agent should recommend Y"
                core_match = re.search(
                    r'\b[Tt]he\s+agent\s+(?:must|should|needs?\s+to|is\s+expected\s+to)\s+(.+?)(?:\.|$)',
                    value,
                )
                if core_match:
                    core_action = core_match.group(1).strip()
                    # Pick a template based on strength (deterministic per key)
                    templates = _STRENGTH_REWRITE_PREFIXES.get(strength, _STRENGTH_REWRITE_PREFIXES["strong"])
                    template = templates[hash(key) % len(templates)]
                    value = template.format(core_action)
                else:
                    # Fallback: generic rewrites for partial matches
                    value = re.sub(
                        r'\b[Tt]he\s+agent\s+(?:must|should)\s+avoid\s+(\w+ing)',
                        r"I want to make sure we avoid \1",
                        value,
                    )
                    value = re.sub(
                        r'\b[Tt]he\s+agent\s+(?:must|should)\s+(\w+)',
                        r'I need you to \1',
                        value,
                    )
                value = re.sub(
                    r'\b[Aa]gent\s+explicitly\s+(?:states?|mentions?|confirms?)',
                    'you mention',
                    value,
                )
                value = re.sub(
                    r'\b[Mm]ust\s+warn\s+(?:the\s+)?(?:user\s+)?(?:about\s+)?',
                    'I should know about ',
                    value,
                )
                # Remove raw pipeline parameter references entirely
                value = re.sub(
                    r'\b\w+_(?:required|level|budget|weight|score)\s*:\s*\d+\s*,?\s*',
                    '',
                    value,
                )
                value = re.sub(
                    r'\b(?:required_elements?|forbidden_elements?|success_criteria'
                    r'|core_capability_tested|constraint_satisfaction|evaluation_focus'
                    r'|reasoning_hops|difficulty_level|complexity_budget)\b',
                    '',
                    value,
                )
                # Clean up artifacts
                value = re.sub(r'\s+', ' ', value).strip()
                value = re.sub(r'^[,;.\s]+', '', value).strip()

                if not value or len(value) < 10:
                    logger.warning(f"Payload '{key}' reduced to nothing after sanitization, dropping")
                    return None
                break  # Only need one pass

        return value

    def _generate_aim_initiation_message(
        self,
        aim: Aim,
        info_to_disclose: List[Tuple[str, str, bool]],
        inherited_context: Dict[str, Any],
        messages: List[AnyMessage],
    ) -> str:
        """
        Generate the user message for starting a new aim.

        DEPRECATED in v5.0: Kept for backward compatibility.
        The BRIDGE pipeline (_aim_driven_step) now handles generation
        via generate_bridge_message(). This method is only called if
        external code references it directly.
        """
        # Build a minimal perception + policy and delegate to bridge generator
        perception = PerceptionResult(
            assistant_acts=["responded"],
            aim_should_advance=False,
        )
        policy = TurnPolicy(
            transition_type=TransitionType.INFO_FILLING,
            payload_items=info_to_disclose,
            is_aim_initiation=True,
            emotion_hint="calm and friendly",
            transition_hint=getattr(aim, 'transition_hint', '') or '',
        )

        task_specific_prompt = ""
        if self._task_controller:
            task_specific_prompt = self._task_controller.get_task_prompt_additions()

        content = generate_bridge_message(
            policy=policy,
            perception=perception,
            aim=aim,
            personality=self.scenario.personality if self.scenario else "Friendly",
            domain=self.scenario.domain if self.scenario else "general",
            messages=messages,
            hidden_state_prompt=self._hidden_state_prompt or "",
            task_specific_prompt=task_specific_prompt,
            llm=self.llm,
            llm_args=self.llm_args,
            temperature=self.temperature,
        )

        # Apply task-specific post-processing
        if self._task_controller:
            content = self._task_controller.process_user_message(content, messages)

        return content
    
    def _render_generation_messages(self, messages, *, aim, history):
        """Instance extension point; baseline simulators retain their request."""
        return messages

    def _get_aim_type_guidance(self, aim_type: AimType) -> str:
        """Get type-specific guidance for message generation"""
        guidance_map = {
            AimType.ASK_ADVICE: "STYLE: Ask for advice or tips naturally. Example: 'Can you give me some tips on...'",
            AimType.ASK_INFORMATION: "STYLE: Request specific information. Example: 'Can you tell me about...'",
            AimType.CREATE_DOCUMENT: "STYLE: Request creating a document. Example: 'Can you write a [format] that includes...'",
            AimType.ADD_CONTENT: "STYLE: Request adding content. Example: 'Can you also add/include...'",
            AimType.REMOVE_CONTENT: "STYLE: Request removing content. Example: 'Can you remove/take out...'",
            AimType.EDIT_CONTENT: "STYLE: Request editing. Example: 'Can you edit this to...'",
            AimType.FORMAT_CHANGE: "STYLE: Request format change. Example: 'Can you change this into [format] format?'",
            AimType.PERSPECTIVE_CHANGE: "STYLE: Request perspective change. Example: 'Can you rewrite this from [person]'s perspective?'",
            AimType.RECIPIENT_CHANGE: "STYLE: Request changing recipient. Example: 'Can you rewrite this for [new recipient]?'",
            AimType.STRUCTURE_CHANGE: "STYLE: Request structural changes. Example: 'Can you reorganize this to put [X] at [location]?'",
            AimType.KEEP_AND_MODIFY: "STYLE: Keep some, change others. Example: 'Keep [these parts] but [change these]'",
            AimType.MERGE_CONTENT: "STYLE: Merge content. Example: 'Can you combine this with the previous...'",
            AimType.CONFIRM: "STYLE: Confirm or verify. Example: 'That looks good! Just to confirm...'",
            AimType.FINAL_REQUEST: "STYLE: Make final modifications. Example: 'One last thing, can you also...'",
        }
        return guidance_map.get(aim_type, "STYLE: Make a clear, natural request")
    
    # ========================================================================
    # v3.0: Context-Aware Conversation Methods (DEPRECATED by BRIDGE v5.0)
    # Kept as stubs for backward compatibility.
    # ========================================================================

    def _build_reaction_section(
        self,
        aim: Aim,
        last_agent_msg: str,
        messages: List[AnyMessage],
    ) -> str:
        """DEPRECATED v5.0: Replaced by BRIDGE templates. Returns empty."""
        return ""

    def _classify_agent_response(self, agent_msg: str) -> str:
        """DEPRECATED v5.0: Replaced by perceive(). Returns 'informative'."""
        return "informative"

    def _build_transition_section(
        self,
        aim: Aim,
        messages: List[AnyMessage],
    ) -> str:
        """DEPRECATED v5.0: Replaced by BRIDGE templates. Returns empty."""
        return ""
    
    def _evaluate_aim_satisfaction(
        self,
        aim: Aim,
        agent_response: str,
        messages: List[AnyMessage],
    ) -> bool:
        """
        Evaluate if the agent's response satisfies the current aim.

        DEPRECATED in v5.0: The BRIDGE pipeline now uses perceive() for this.
        Kept for backward compatibility / direct invocation.
        """
        constraints_text = self._build_constraints_text()
        perception = perceive(
            aim=aim,
            assistant_message=agent_response,
            messages=messages,
            hidden_state=self._hidden_state if self._hidden_state else None,
            constraints_text=constraints_text,
            llm=self.evaluation_llm,
            llm_args=self.llm_args,
        )
        return perception.aim_should_advance
    
    def _check_latent_preference_triggers(
        self,
        agent_response: str,
        messages: List[AnyMessage],
    ) -> List[str]:
        """
        Check if any latent preference hidden states should be triggered
        based on the agent's response content.
        
        Uses trigger_keywords (explicit keyword list) first, falls back to
        trigger_condition word matching if keywords are not available.
        
        Returns a list of trigger instruction strings to inject into
        the next user message prompt.
        """
        if not self._hidden_state:
            return []
        
        triggered = []
        response_lower = agent_response.lower()
        
        for hs in self._hidden_state:
            if not hs.category.is_preference:
                continue
            if hs.disclosed_during_conversation:
                # Already triggered previously
                continue
            
            matched = False
            
            # Strategy 1: Use explicit trigger_keywords if available
            if hasattr(hs, 'trigger_keywords') and hs.trigger_keywords:
                for keyword in hs.trigger_keywords:
                    if keyword.lower() in response_lower:
                        matched = True
                        logger.info(
                            f"Latent preference triggered via keyword: "
                            f"{hs.key} (keyword='{keyword}')"
                        )
                        break
            
            # Strategy 2: Fall back to trigger_condition word matching
            if not matched and hs.trigger_condition:
                trigger_lower = hs.trigger_condition.lower()
                trigger_words = [w for w in trigger_lower.split() if len(w) > 3]
                match_count = sum(1 for w in trigger_words if w in response_lower)
                
                if trigger_words and match_count / len(trigger_words) >= 0.4:
                    matched = True
                    logger.info(
                        f"Latent preference triggered via condition: "
                        f"{hs.key} (matched {match_count}/{len(trigger_words)} words)"
                    )
            
            if matched:
                effect = hs.behavioral_effect or f"Mention your preference: {hs.value}"
                triggered.append(
                    f"⚡ TRIGGERED PREFERENCE: The agent's response relates to '{hs.key}'. "
                    f"You should now naturally reveal: {effect}"
                )
        
        return triggered

    def _provide_clarification_or_detail(
        self,
        aim: Aim,
        messages: List[AnyMessage],
    ) -> Tuple[UserMessage, UserState]:
        """
        Provide clarification or additional details when aim not satisfied.

        DEPRECATED in v5.0: The BRIDGE pipeline in _aim_driven_step handles
        continuation turns directly via perceive → decide → generate.
        Kept for backward compatibility.
        """
        return self._aim_driven_step(messages)

    def _generate_llm_clarification(
        self,
        aim: Aim,
        messages: List[AnyMessage],
        trigger_instructions: Optional[List[str]] = None,
    ) -> str:
        """
        Generate clarification message using LLM.

        DEPRECATED in v5.0: Replaced by generate_bridge_message().
        Kept for backward compatibility.
        """
        # Build a minimal perception + policy and delegate to bridge generator
        last_agent_msg = self._get_last_agent_message(messages)
        info_items = self._get_info_for_aim(aim)

        perception = PerceptionResult(
            assistant_acts=["responded"],
            aim_should_advance=False,
        )
        policy = TurnPolicy(
            transition_type=TransitionType.INFO_FILLING,
            payload_items=info_items[:2],
            trigger_instructions=trigger_instructions or [],
            emotion_hint="calm and friendly",
        )

        task_specific_prompt = ""
        if self._task_controller:
            task_specific_prompt = self._task_controller.get_task_prompt_additions()

        content = generate_bridge_message(
            policy=policy,
            perception=perception,
            aim=aim,
            personality=self.scenario.personality if self.scenario else "Friendly",
            domain=self.scenario.domain if self.scenario else "general",
            messages=messages,
            hidden_state_prompt=self._hidden_state_prompt or "",
            task_specific_prompt=task_specific_prompt,
            llm=self.llm,
            llm_args=self.llm_args,
            temperature=self.temperature,
        )

        if self._task_controller:
            content = self._task_controller.process_user_message(content, messages)

        return content

    def _get_last_agent_message(self, messages: List[AnyMessage]) -> str:
        """Extract the last agent message content"""
        for msg in reversed(messages):
            if isinstance(msg, AssistantMessage) and msg.content:
                return msg.content
        return ""
    
    def _format_recent_history(self, messages: List[AnyMessage], last_n: int = 4) -> str:
        """Format recent conversation history for prompt"""
        recent = messages[-last_n:] if len(messages) > last_n else messages
        
        formatted = []
        for msg in recent:
            if isinstance(msg, UserMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AssistantMessage):
                content = msg.content or "[no content]"
                # Truncate long responses
                if len(content) > 300:
                    content = content[:300] + "..."
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted) if formatted else "(No previous conversation)"
    
    def _generate_satisfaction_message(self) -> Tuple[UserMessage, UserState]:
        """Generate message when all aims completed"""
        self.context.increment_turn()
        
        content = self._generate_natural_satisfaction()
        
        message = UserMessage(
            content=content,
            turn_idx=self.context.current_turn,
            metadata={
                "is_completion": True,
                "all_aims_done": True,
            }
        )
        
        return message, UserState.SATISFIED
    
    def _generate_natural_satisfaction(self) -> str:
        """Generate a natural satisfaction message"""
        prompt = f"""You are a user who has completed all their tasks with an AI assistant.

PERSONALITY: {self.scenario.personality if self.scenario else 'Friendly'}
{self._hidden_state_prompt if self._hidden_state_prompt else ''}

Generate a brief, natural message expressing satisfaction and ending the conversation.
Examples:
- "Perfect, that's exactly what I needed! Thanks so much!"
- "Great, this looks good. Thanks for your help!"
- "Awesome, I really appreciate it. That's all I needed!"

Generate ONLY the user's message:"""

        # No try/except: if the LLM fails here, raise loudly so the
        # offending route/quota issue gets surfaced. Previous code substituted
        # a canned "Perfect, that's everything I needed..." string, which made
        # broken runs look successful.
        response = generate_chat_completion(
            model=self.llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=16384,
            **(self.llm_args or {}),
        )
        return response.strip()
    
    def _generate_timeout_end_message(self) -> Tuple[UserMessage, UserState]:
        """Generate message when total turn limit reached"""
        self.context.increment_turn()
        
        message = UserMessage(
            content="I think that's enough for now, thanks for your help!",
            turn_idx=self.context.current_turn,
            metadata={
                "is_timeout": True,
                "total_turns_limit": True,
            }
        )
        
        return message, UserState.SATISFIED
    
    def get_aim_progress(self) -> Optional[Dict[str, Any]]:
        """Get current aim execution progress"""
        if self._aim_sequence:
            return self._aim_sequence.get_progress_summary()
        return None
