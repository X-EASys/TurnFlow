"""
Aim Data Model

Defines the structure for multi-aim, multi-task scenarios where:
- Each Aim represents a user goal/task to accomplish
- Aims are executed sequentially with dependencies
- LLM dynamically generates user messages to achieve each aim
- State transitions occur when aims are satisfied or timeout

This enables complex multi-turn conversations like document editing,
where users progressively modify content through multiple requests.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field


class AimStatus(str, Enum):
    """Status of an Aim during execution"""
    PENDING = "pending"           # Waiting to start
    IN_PROGRESS = "in_progress"   # Currently being worked on
    SATISFIED = "satisfied"       # Successfully completed
    TIMEOUT = "timeout"           # Max turns reached, forced transition
    SKIPPED = "skipped"           # Skipped due to dependency failure


class AimType(str, Enum):
    """Types of user aims/tasks"""
    # === Information & Advice ===
    ASK_ADVICE = "ask_advice"                       # Request tips/suggestions
    ASK_INFORMATION = "ask_information"             # Request specific information
    
    # === Document Creation ===
    CREATE_DOCUMENT = "create_document"             # Create new document (email, post, etc.)
    
    # === Document Modification ===
    ADD_CONTENT = "add_content"                     # Add new content to existing document
    REMOVE_CONTENT = "remove_content"               # Remove specific content
    EDIT_CONTENT = "edit_content"                   # Modify existing content
    
    # === Structural Changes ===
    FORMAT_CHANGE = "format_change"                 # Change document format
    PERSPECTIVE_CHANGE = "perspective_change"       # Change author/viewpoint
    RECIPIENT_CHANGE = "recipient_change"           # Change target audience
    STRUCTURE_CHANGE = "structure_change"           # Change document structure
    
    # === Composite Operations ===
    KEEP_AND_MODIFY = "keep_and_modify"             # Keep some, change others
    MERGE_CONTENT = "merge_content"                 # Merge from multiple sources
    
    # === Verification ===
    CONFIRM = "confirm"                             # Confirm/verify result
    FINAL_REQUEST = "final_request"                 # Final modification (evaluation point)
    
    # === T3/T5 Task Types (v2.0) ===
    PROVIDE_INFO = "provide_info"                   # User provides info (T3: only if asked)
    SET_RULE = "set_rule"                           # User sets a rule/constraint (T5)
    RESPOND_TO_AGENT = "respond_to_agent"           # User responds to Agent's output (T1/T4: callback)
    
    # === Distractor/Filler Tasks ===
    CASUAL_CHAT = "casual_chat"                     # Casual conversation filler (distractor)

    # === Killer Scenario Types (T1/T2/T3) ===
    DRIFT = "drift"                   # Deliberate noise about someone else's experience
    PROBE = "probe"                   # Forward-pushing implicit evaluation trigger
    INFO_SCATTER = "info_scatter"     # Deliver one piece of scattered information


# === Global Configuration ===
# Set to 1 to force all aims to have max 1 turn (for testing)
# Set to None to use the aim's configured max_turns
FORCE_MAX_TURNS_PER_AIM: int | None = 1  # Simplified testing: 1 turn per aim

# Set to False to disable clarification_hints entirely
# When FORCE_MAX_TURNS_PER_AIM=1, clarification is not needed since each aim has only 1 turn
# Set to True to allow normal clarification behavior
ENABLE_CLARIFICATION_HINTS: bool = False  # Disabled for simplified testing


class DriftConfig(BaseModel):
    """Configuration for a DRIFT aim — deliberate tangent about someone else."""
    drift_subject: str = Field(
        ..., description="Who the drift is about (e.g., 'colleague Xiao Li')"
    )
    drift_topic: str = Field(
        ..., description="What about them (e.g., 'she took class A last year and loved it')"
    )
    target_option: str = Field(
        default="",
        description="Which option this creates recency bias toward (e.g., 'class A')"
    )


class ProbeConfig(BaseModel):
    """Configuration for a PROBE aim — forward-pushing evaluation trigger."""
    probe_style: str = Field(
        default="forward_pushing",
        description="Style: 'forward_pushing' or 'per_item_verify'"
    )
    forbidden_words: List[str] = Field(
        default_factory=list,
        description="Words that must NOT appear in the probe message"
    )
    no_acknowledge: bool = Field(
        default=True,
        description="If True, suppress acknowledgment of agent's previous message"
    )


class Aim(BaseModel):
    """
    Single Aim/Task Definition

    An Aim represents one user goal in a multi-step conversation.
    The User Simulator will attempt to achieve this aim through
    providing information and making requests.
    
    Execution Flow:
    1. Enter new Aim → Provide related information
    2. Make request based on aim description
    3. Evaluate if Agent response satisfies success_criteria
    4. If satisfied → Move to next Aim
    5. If not satisfied → Provide clarification (up to max_turns)
    6. If timeout → Force transition to next Aim
    """
    
    # === Identity ===
    aim_id: str = Field(
        ..., 
        description="Unique identifier for this aim"
    )
    aim_type: AimType = Field(
        ..., 
        description="Type of aim/task"
    )
    
    # === Description ===
    description: str = Field(
        ..., 
        description="What the user wants to achieve (for LLM to generate request)"
    )
    success_criteria: str = Field(
        ..., 
        description="How to determine if aim is satisfied"
    )
    
    # === Associated Information ===
    required_info_keys: List[str] = Field(
        default_factory=list,
        description="Keys of flow_info items needed for this aim"
    )
    required_constraint_keys: List[str] = Field(
        default_factory=list,
        description="Keys of constraints to apply in this aim (user's actual constraints like budget, allergies)"
    )
    
    # === Expected Agent Output (Evaluation Only) ===
    expected_agent_output: Dict[str, str] = Field(
        default_factory=dict,
        description="Information the Agent should provide at this aim. Used ONLY for evaluation - never disclosed to User Simulator or Agent. Keys are identifiers, values are expected content. Example: {'festival_timing': 'The Tancho Festival is held in late February'}"
    )
    
    # === Dependencies ===
    depends_on_aims: List[str] = Field(
        default_factory=list,
        description="Aim IDs that must be completed first (hard dependency)"
    )
    inherits_info_from: List[str] = Field(
        default_factory=list,
        description="Aim IDs whose context/results are used by this aim"
    )
    
    # === Execution Control ===
    max_turns: int = Field(
        default=3,
        description="Maximum turns to attempt before timeout"
    )
    min_info_before_request: int = Field(
        default=1,
        description="Minimum info items to disclose before making the request"
    )
    is_final: bool = Field(
        default=False,
        description="Whether this is the final aim (evaluation point)"
    )
    
    # === Optional Guidance ===
    request_template: Optional[str] = Field(
        default=None,
        description="Optional template for generating the request (LLM will naturalize)"
    )
    clarification_hints: Optional[List[str]] = Field(
        default=None,
        description="Hints for what to clarify if aim is not satisfied"
    )
    
    # === v3.0: Context-Aware Conversation Fields ===
    transition_hint: Optional[str] = Field(
        default=None,
        description="How to naturally transition from the previous aim. Example: 'Oh that reminds me...' or 'Actually, there's another thing...'"
    )
    reaction_templates: Optional[Dict[str, str]] = Field(
        default=None,
        description="Templates for reacting to different agent response types. Keys: 'positive', 'negative', 'clarifying', 'partial'. Values are natural reaction phrases."
    )
    context_summary: Optional[str] = Field(
        default=None,
        description="Summary of context from previous aims, helps maintain conversation continuity"
    )
    conversation_goal: Optional[str] = Field(
        default=None,
        description="High-level conversational goal, more natural than technical description. Example: 'Find out about travel options while mentioning budget concerns'"
    )
    pre_written_response: Optional[str] = Field(
        default=None,
        description="Pre-written user response paragraph for this aim. "
                    "Generated and validated at scenario creation time. "
                    "Adapted to conversation context at runtime."
    )
    aim_mode: str = Field(
        default="disclosure",
        description="Mode for this aim: 'disclosure' (default, reveal info), 'cooperative' (confirm agent plan), "
                    "'pushback' (disagree with agent), 'withhold' (deliberately withhold info), 'ambiguous' (ambiguous reference)"
    )

    # === Killer Scenario Configs (T1/T2/T3) ===
    drift_config: Optional[DriftConfig] = Field(
        default=None,
        description="Configuration for DRIFT aims — defines who/what the tangent is about"
    )
    probe_config: Optional[ProbeConfig] = Field(
        default=None,
        description="Configuration for PROBE aims — defines probe style and constraints"
    )

    # === Composite Task Type Phase ===
    phase: Optional[str] = Field(
        default=None,
        description="Phase this aim belongs to: 'phase_1', 'phase_2', or None (single-type)"
    )
    is_phase_trap: Optional[bool] = Field(
        default=None,
        description="True if this aim is the trap aim for its phase"
    )
    phase_task_type: Optional[str] = Field(
        default=None,
        description="Task type governing this aim's phase (e.g., 'T9' for Phase 1 of T9+T10)"
    )

    # === Runtime State (not serialized) ===
    status: AimStatus = Field(
        default=AimStatus.PENDING,
        exclude=True
    )
    turns_spent: int = Field(
        default=0,
        exclude=True
    )
    info_disclosed: List[str] = Field(
        default_factory=list,
        exclude=True
    )
    request_made: bool = Field(
        default=False,
        exclude=True
    )
    
    def reset_runtime_state(self) -> None:
        """Reset runtime state for new conversation"""
        self.status = AimStatus.PENDING
        self.turns_spent = 0
        self.info_disclosed = []
        self.request_made = False
    
    def get_undisclosed_info_keys(self) -> List[str]:
        """Get info keys not yet disclosed"""
        return [k for k in self.required_info_keys if k not in self.info_disclosed]
    
    def has_pending_info(self) -> bool:
        """Check if there's still info to disclose"""
        return len(self.get_undisclosed_info_keys()) > 0
    
    def can_make_request(self) -> bool:
        """Check if enough info disclosed to make request"""
        disclosed_count = len(self.info_disclosed)
        return disclosed_count >= self.min_info_before_request or not self.has_pending_info()
    
    def increment_turn(self) -> None:
        """Increment turn counter"""
        self.turns_spent += 1
    
    def is_timeout(self) -> bool:
        """Check if max turns reached.
        
        Uses FORCE_MAX_TURNS_PER_AIM if set, otherwise uses the aim's configured max_turns.
        """
        effective_max_turns = FORCE_MAX_TURNS_PER_AIM if FORCE_MAX_TURNS_PER_AIM is not None else self.max_turns
        return self.turns_spent >= effective_max_turns
    
    def get_effective_max_turns(self) -> int:
        """Get the effective max turns (considering global override)."""
        return FORCE_MAX_TURNS_PER_AIM if FORCE_MAX_TURNS_PER_AIM is not None else self.max_turns

    def mark_info_disclosed(self, info_key: str) -> None:
        """Mark an info item as disclosed"""
        if info_key not in self.info_disclosed:
            self.info_disclosed.append(info_key)
    
    def has_expected_agent_output(self) -> bool:
        """Check if this aim has expected agent output for evaluation"""
        return bool(self.expected_agent_output)
    
    def get_expected_agent_output_items(self) -> List[Tuple[str, str]]:
        """
        Get expected agent output as list of (key, value) tuples.
        Used ONLY for evaluation - never disclosed to User Simulator.
        
        Returns:
            List of (key, expected_value) tuples for evaluation
        """
        return list(self.expected_agent_output.items())


class AimSequence(BaseModel):
    """
    Sequence of Aims for Multi-Task Scenarios
    
    Defines the ordered list of aims the user will try to accomplish.
    The User Simulator executes aims sequentially, with dynamic
    state transitions based on satisfaction evaluation.
    
    Key Features:
    - Sequential execution with dependencies
    - Information inheritance between aims
    - Timeout-based forced transitions
    - Final aim marking for evaluation
    """
    
    aims: List[Aim] = Field(
        ..., 
        description="Ordered list of aims to execute"
    )
    
    # === Evaluation Config ===
    evaluation_aim_id: Optional[str] = Field(
        default=None,
        description="Aim ID used for final evaluation (defaults to last aim)"
    )
    
    # === Global Limits ===
    total_max_turns: int = Field(
        default=15,
        description="Maximum total turns for entire conversation"
    )
    
    # === Runtime State ===
    current_aim_index: int = Field(
        default=0,
        exclude=True
    )
    total_turns_spent: int = Field(
        default=0,
        exclude=True
    )
    
    def model_post_init(self, __context) -> None:
        """Post-initialization setup"""
        # Set evaluation_aim_id to last aim if not specified
        if self.evaluation_aim_id is None and self.aims:
            # Find the aim marked as final, or use the last one
            for aim in reversed(self.aims):
                if aim.is_final:
                    self.evaluation_aim_id = aim.aim_id
                    break
            if self.evaluation_aim_id is None:
                self.evaluation_aim_id = self.aims[-1].aim_id
                self.aims[-1].is_final = True
    
    def reset(self) -> None:
        """Reset all runtime state for new conversation"""
        self.current_aim_index = 0
        self.total_turns_spent = 0
        for aim in self.aims:
            aim.reset_runtime_state()
    
    def get_current_aim(self) -> Optional[Aim]:
        """Get the current aim being worked on"""
        if 0 <= self.current_aim_index < len(self.aims):
            return self.aims[self.current_aim_index]
        return None
    
    def get_aim_by_id(self, aim_id: str) -> Optional[Aim]:
        """Get aim by its ID"""
        for aim in self.aims:
            if aim.aim_id == aim_id:
                return aim
        return None
    
    def advance_to_next_aim(self, reason: AimStatus = AimStatus.SATISFIED) -> Optional[Aim]:
        """
        Advance to the next aim
        
        Args:
            reason: Why we're advancing (SATISFIED or TIMEOUT)
            
        Returns:
            The next aim, or None if all aims completed
        """
        current = self.get_current_aim()
        if current:
            current.status = reason
        
        self.current_aim_index += 1
        
        next_aim = self.get_current_aim()
        if next_aim:
            next_aim.status = AimStatus.IN_PROGRESS
        
        return next_aim
    
    def is_complete(self) -> bool:
        """Check if all aims have been processed"""
        return self.current_aim_index >= len(self.aims)
    
    def is_total_timeout(self) -> bool:
        """Check if total turn limit reached"""
        return self.total_turns_spent >= self.total_max_turns
    
    def increment_total_turns(self) -> None:
        """Increment total turn counter"""
        self.total_turns_spent += 1
    
    def get_completed_aims(self) -> List[Aim]:
        """Get list of completed aims"""
        return [aim for aim in self.aims 
                if aim.status in (AimStatus.SATISFIED, AimStatus.TIMEOUT)]
    
    def get_evaluation_aim(self) -> Optional[Aim]:
        """Get the aim used for evaluation"""
        if self.evaluation_aim_id:
            return self.get_aim_by_id(self.evaluation_aim_id)
        return self.aims[-1] if self.aims else None
    
    def get_inherited_context(self, aim: Aim) -> Dict[str, Any]:
        """
        Get context inherited from previous aims
        
        Args:
            aim: The aim requesting inherited context
            
        Returns:
            Dict with inherited information
        """
        context = {
            "completed_aims": [],
            "inherited_info_keys": [],
        }
        
        for inherit_aim_id in aim.inherits_info_from:
            inherit_aim = self.get_aim_by_id(inherit_aim_id)
            if inherit_aim and inherit_aim.status in (AimStatus.SATISFIED, AimStatus.TIMEOUT):
                context["completed_aims"].append({
                    "aim_id": inherit_aim_id,
                    "description": inherit_aim.description,
                    "info_disclosed": inherit_aim.info_disclosed.copy(),
                })
                context["inherited_info_keys"].extend(inherit_aim.info_disclosed)
        
        # Remove duplicates while preserving order
        seen = set()
        context["inherited_info_keys"] = [
            k for k in context["inherited_info_keys"] 
            if not (k in seen or seen.add(k))
        ]
        
        return context
    
    def check_dependencies_met(self, aim: Aim) -> bool:
        """
        Check if all dependencies for an aim are met
        
        Args:
            aim: The aim to check dependencies for
            
        Returns:
            True if all dependencies are satisfied
        """
        for dep_aim_id in aim.depends_on_aims:
            dep_aim = self.get_aim_by_id(dep_aim_id)
            if not dep_aim or dep_aim.status not in (AimStatus.SATISFIED, AimStatus.TIMEOUT):
                return False
        return True
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of aim execution progress"""
        return {
            "total_aims": len(self.aims),
            "current_index": self.current_aim_index,
            "current_aim": self.get_current_aim().aim_id if self.get_current_aim() else None,
            "completed": len(self.get_completed_aims()),
            "total_turns_spent": self.total_turns_spent,
            "is_complete": self.is_complete(),
            "aims_status": [
                {"id": aim.aim_id, "status": aim.status.value, "turns": aim.turns_spent}
                for aim in self.aims
            ]
        }
    
    def get_all_expected_agent_outputs(self) -> Dict[str, Dict[str, str]]:
        """
        Get all expected agent outputs from all aims.
        Used ONLY for evaluation - never disclosed to User Simulator.
        
        Returns:
            Dict mapping aim_id to expected_agent_output dict
            Example: {
                "aim_2_festival_info": {
                    "festival_timing": "The festival is in late February"
                }
            }
        """
        outputs = {}
        for aim in self.aims:
            if aim.has_expected_agent_output():
                outputs[aim.aim_id] = aim.expected_agent_output.copy()
        return outputs
