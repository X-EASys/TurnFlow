"""
Scenario Data Model

Defines the structure for user scenarios including:
- StateQueue: The information queue that user reveals progressively
- Scenario: Complete scenario definition with personality and domain
- AnchorConfig: Configuration for anchor revelation strategy (vague/disguised)
"""

from enum import Enum
from typing import Optional, List, Union, Tuple, Dict, Any, TYPE_CHECKING
import json as _json
import re as _re
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from benchmark.data_model.aim import AimSequence


# =============================================================================
# HIDDEN STATE — Enums and Models
# =============================================================================

class HiddenStateCategory(str, Enum):
    """Categories of hidden state information.
    
    Supports two naming conventions:
    - Legacy:   persona_style / latent_preference / internal_knowledge
    - Pipeline: identity / preference / communication_style
    
    The mapping is:
        identity           ↔ persona_style        (WHO the user is)
        preference         ↔ latent_preference     (persistent biases)
        communication_style ↔ internal_knowledge   ... NO — communication_style
                             is about HOW the user talks, which maps to persona_style.
    
    Revised semantic mapping (pipeline → prompt section):
        identity            → 🎭 Communication Style section (persona — who you are)
        preference          → 🔒 Latent Preference section  (persistent bias)
        communication_style → 🎭 Communication Style section (how you talk)
    
    Both old and new values are kept so existing data files remain loadable.
    """
    # --- Legacy values (backward-compatible) ---
    PERSONA_STYLE = "persona_style"
    LATENT_PREFERENCE = "latent_preference"
    INTERNAL_KNOWLEDGE = "internal_knowledge"

    # --- Pipeline-generated values ---
    IDENTITY = "identity"                        # Who the user IS → maps to persona section
    PREFERENCE = "preference"                    # Persistent bias → maps to latent section
    COMMUNICATION_STYLE = "communication_style"  # How the user talks → maps to persona section

    # ----- helpers -----
    @property
    def is_persona(self) -> bool:
        """True for categories that belong to the persona / communication-style prompt section."""
        return self in (
            HiddenStateCategory.PERSONA_STYLE,
            HiddenStateCategory.IDENTITY,
            HiddenStateCategory.COMMUNICATION_STYLE,
        )

    @property
    def is_preference(self) -> bool:
        """True for categories that belong to the latent-preference prompt section."""
        return self in (
            HiddenStateCategory.LATENT_PREFERENCE,
            HiddenStateCategory.PREFERENCE,
        )

    @property
    def is_knowledge(self) -> bool:
        """True for categories that belong to the internal-knowledge prompt section."""
        return self == HiddenStateCategory.INTERNAL_KNOWLEDGE


class OverlapMode(str, Enum):
    """How a hidden state item relates to flow_info/constraint."""
    NONE = "none"                    # Pure hidden, not in flow_info
    ALSO_DISCLOSED = "also_disclosed"  # Also exists in flow_info (user says it once + behaves)
    CONDITIONAL = "conditional"      # Hidden by default, becomes disclosed if triggered


class HiddenStateItem(BaseModel):
    """
    A single hidden state item — information the user knows but doesn't
    (or doesn't always) explicitly tell the agent.
    
    Hidden state drives the user simulator's BEHAVIOR rather than
    controlling WHEN information is disclosed.
    """
    key: str = Field(..., description="Unique identifier, e.g. 'communication_style'")
    category: HiddenStateCategory = Field(..., description="One of the three sub-types")
    value: str = Field(..., description="Content, e.g. 'terse, no pleasantries'")
    description: str = Field(..., description="Human-readable explanation")
    
    # === Overlap with flow_info/constraint ===
    overlap_mode: OverlapMode = Field(
        default=OverlapMode.NONE,
        description="Relationship to flow_info/constraint"
    )
    linked_flow_info_key: Optional[str] = Field(
        default=None,
        description="Key of the related flow_info item (for also_disclosed/conditional)"
    )
    linked_constraint_key: Optional[str] = Field(
        default=None,
        description="Key of the related constraint item"
    )
    
    # === persona_style specific ===
    style_enforcement: Optional[str] = Field(
        default=None,
        description="Programmatic rule, e.g. 'max_words_per_turn:30'"
    )

    @field_validator('style_enforcement', mode='before')
    @classmethod
    def coerce_style_enforcement(cls, v):
        """Coerce non-string LLM outputs (dict) to str."""
        if v is None:
            return v
        if isinstance(v, dict):
            return _json.dumps(v)
        if not isinstance(v, str):
            return str(v)
        return v
    
    # === latent_preference specific ===
    trigger_condition: Optional[str] = Field(
        default=None,
        description="When to disclose, e.g. 'agent asks about budget'"
    )
    trigger_keywords: Optional[List[str]] = Field(
        default=None,
        description="Explicit keywords that trigger this preference, e.g. ['budget', 'price', 'cost']"
    )
    disclosed_during_conversation: bool = Field(
        default=False,
        description="Runtime flag: whether trigger was hit and info was disclosed"
    )
    
    # === internal_knowledge specific ===
    behavioral_effect: Optional[str] = Field(
        default=None,
        description="How this affects user reactions, e.g. 'reject product X'"
    )
    
    # === Evaluation ===
    evaluation_weight: str = Field(
        default="soft",
        description="'soft' = bonus/no-penalty | 'hard' = must-comply"
    )


class AnchorDisguiseStrategy(str, Enum):
    """Anchor disguise strategies for challenging LLM evaluation"""
    DIRECT = "direct"                        # No disguise, but still vague/life-contextualized
    DELAYED_REVEAL = "delayed_reveal"        # True anchor revealed after several turns
    CONTEXT_SWITCH = "context_switch"        # Conversation appears about one thing, switches to real need
    DISTRACTOR_FIRST = "distractor_first"    # Open with distractor topic, real need emerges
    EMBEDDED_PROBLEM = "embedded_problem"    # Technical problem hidden in life scenario


class AnchorConfig(BaseModel):
    """
    Anchor Configuration - Supports vague/life-contextualized and disguised anchor strategies.
    
    ALL anchors (both initial_statement and true_anchor) must be:
    - Life-contextualized: Sound like a real person talking about their life
    - Vague: Not directly reveal the underlying problem type (math, logic, time, etc.)
    - Emotionally engaging: Create personal connection before technical details
    
    The disguised anchor strategy adds an additional layer where the true need
    is revealed after several turns of conversation.
    """
    
    # === Basic fields (required for all scenarios) ===
    initial_statement: str = Field(
        ...,
        description="User's opening statement - must be vague/life-contextualized, NOT revealing problem type"
    )
    underlying_problem_type: str = Field(
        default="general",
        description="The actual problem type (for internal tracking): math, time, logic, decision, geography, etc."
    )
    vagueness_strategy: str = Field(
        default="",
        description="How the technical problem is wrapped in a life scenario"
    )
    
    # === Disguised anchor fields (optional) ===
    is_disguised: bool = Field(
        default=False,
        description="Whether true anchor is hidden and revealed later"
    )
    true_anchor: Optional[str] = Field(
        default=None,
        description="The real need (revealed later) - also must be vague/life-contextualized"
    )
    reveal_turn: Optional[int] = Field(
        default=None,
        description="Turn number to reveal true_anchor if agent hasn't triggered it earlier"
    )
    reveal_trigger_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords that trigger early reveal of true_anchor when agent mentions them"
    )
    disguise_strategy: Optional[AnchorDisguiseStrategy] = Field(
        default=None,
        description="The strategy used for disguising the anchor"
    )
    disguise_rationale: Optional[str] = Field(
        default=None,
        description="Explanation of why/how the anchor is disguised"
    )
    false_anchor_topic: Optional[str] = Field(
        default=None,
        description="What the initial conversation appears to be about (for disguised mode)"
    )
    
    def get_effective_anchor(self) -> str:
        """Get the anchor to use at Turn 0 (always initial_statement)"""
        return self.initial_statement
    
    def get_true_anchor(self) -> str:
        """Get the true anchor (same as initial if not disguised)"""
        if self.is_disguised and self.true_anchor:
            return self.true_anchor
        return self.initial_statement
    
    def should_reveal_now(self, current_turn: int, agent_message: str = "") -> bool:
        """Check if true anchor should be revealed based on turn or trigger keywords"""
        if not self.is_disguised:
            return False  # Not disguised, no reveal needed
        
        # Check if reached reveal turn
        if self.reveal_turn and current_turn >= self.reveal_turn:
            return True
        
        # Check trigger keywords in agent message
        if self.reveal_trigger_keywords and agent_message:
            agent_lower = agent_message.lower()
            for keyword in self.reveal_trigger_keywords:
                if keyword.lower() in agent_lower:
                    return True
        
        return False


class PersonalityType(str, Enum):
    """Predefined personality types for user simulation"""
    INDECISIVE_CHATTY = "Indecisive but chatty"
    BUSY_STRESSED = "Busy and slightly stressed"
    PATIENT_DETAILED = "Patient and detail-oriented"
    IMPATIENT_DIRECT = "Impatient and direct"
    FRIENDLY_COOPERATIVE = "Friendly and cooperative"
    SKEPTICAL_CAUTIOUS = "Skeptical and cautious"
    CONFUSED_NEEDS_HELP = "Confused and needs guidance"
    TECH_SAVVY = "Tech-savvy and efficient"
    ELDERLY_PATIENT = "Elderly and needs patience"
    CUSTOM = "Custom"


# ── Constraint Strength Classification ──────────────────────────────────

_ABSOLUTE_PATTERNS = [
    _re.compile(r'\b(?:never|absolutely|prohibited|forbidden|banned|allergy|allergic|anaphyla|fatal|dangerous|toxic|lethal|illegal|medical)\b', _re.I),
    _re.compile(r'\bno\s+(?:\w+\s+){0,2}(?:under\s+any|whatsoever|at\s+all|ever)\b', _re.I),
    _re.compile(r'\b(?:must\s+not|cannot|can\s*not)\b', _re.I),
    _re.compile(r'\bnon-negotiable\b', _re.I),
]

_SOFT_PATTERNS = [
    _re.compile(r'\b(?:prefer|ideally|if\s+possible|nice\s+to\s+have|would\s+like|optional|flexible)\b', _re.I),
]


def _classify_constraint_strength(value: str) -> str:
    """Auto-classify constraint strength from value text.

    Returns 'absolute', 'strong', or 'soft'.
    Default is 'strong' if no pattern matches.
    """
    for pattern in _ABSOLUTE_PATTERNS:
        if pattern.search(value):
            return "absolute"
    for pattern in _SOFT_PATTERNS:
        if pattern.search(value):
            return "soft"
    return "strong"


class InfoItem(BaseModel):
    """Structured information item with key, value, and disclosure status"""
    key: str = Field(..., description="Key/identifier for this info")
    value: str = Field(..., description="The actual information content")

    @field_validator('value', mode='before')
    @classmethod
    def coerce_value_to_str(cls, v):
        """Coerce non-string LLM outputs (list, int, dict) to str."""
        if isinstance(v, list):
            return ', '.join(str(item) for item in v)
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, dict):
            return _json.dumps(v)
        return v
    disclosed: bool = Field(default=False, description="Whether this info has been disclosed")
    exact_disclosure: bool = Field(
        default=False,
        description="If True, the exact value (especially numbers, emails, names) MUST be included verbatim in the user message. "
                    "If False, the information can be paraphrased or naturally incorporated."
    )
    role: str = Field(
        default="user_input",
        description="Role: user_input (user tells agent), reference (agent derives), or context (supplementary)"
    )
    strength: str = Field(
        default="",
        description="Constraint strength: 'absolute' (non-negotiable, safety/medical/legal), "
                    "'strong' (important rule, rare exceptions), 'soft' (preference, flexible). "
                    "Empty string means auto-classify from value text."
    )
    check_values: List[str] = Field(
        default_factory=list,
        description="Structural values (dollar amounts, percentages, measurements, durations) "
                    "that must appear verbatim in the user's message."
    )

    # === Killer scenario fields (T1/T2/T3) ===
    surface_form: Optional[str] = Field(
        default=None,
        description="What the user actually says (e.g., 'I am Ningxia Hui ethnicity'). "
                    "When set with delivery_mode='inferential', the simulator uses this instead of value."
    )
    inference_chain: Optional[List[str]] = Field(
        default=None,
        description="Reasoning steps from surface_form to value "
                    "(e.g., ['Hui → Muslim', 'Muslim → halal', 'halal → no pork'])."
    )
    delivery_mode: str = Field(
        default="direct",
        description="How the simulator delivers this info: "
                    "'direct' (state value as-is), "
                    "'inferential' (use surface_form, never mention value), "
                    "'scattered' (split across multiple turns)."
    )

    def get_strength(self) -> str:
        """Get constraint strength, auto-classifying from value text if not set."""
        if self.strength and self.strength in ("absolute", "strong", "soft"):
            return self.strength
        return _classify_constraint_strength(self.value)

    def get_effective_surface(self) -> str:
        """Return the text the simulator should use when delivering this info.

        For inferential/scattered items where surface_form is set, returns
        surface_form so the simulator never directly states the underlying value.
        For direct items (or any item without a surface_form), returns value.
        """
        if self.delivery_mode in ("inferential", "scattered") and self.surface_form:
            return self.surface_form
        return self.value

    def __str__(self) -> str:
        return self.value


# Type alias for flexible info format
InfoType = Union[str, InfoItem]


class StateQueue(BaseModel):
    """
    State Queue: The information stack that the user holds and reveals progressively.
    
    This is the core data structure for controlling information flow in the conversation.
    Supports both simple string format and structured InfoItem format.
    
    Attributes:
        anchor: The initial statement that starts the conversation (Turn 0) - LEGACY, use anchor_config
        anchor_config: New structured anchor configuration with vagueness and disguise support
        flow_info: List of information pieces to be revealed progressively
        constraints: Hard constraints that must be enforced (rejection triggers)
        distractor: Optional off-topic content to add naturalness
        soft_preferences: Optional soft preferences (nice to have, but not required)
    """
    # Legacy anchor field - kept for backward compatibility
    anchor: Optional[str] = Field(
        default=None,
        description="LEGACY: The initial anchor statement. Use anchor_config for new scenarios."
    )
    
    # New structured anchor configuration
    anchor_config: Optional[AnchorConfig] = Field(
        default=None,
        description="Structured anchor configuration with vagueness and disguise support"
    )
    
    flow_info: List[InfoType] = Field(
        default_factory=list,
        description="Information pieces to reveal progressively during conversation"
    )
    constraints: List[InfoType] = Field(
        default_factory=list,
        description="Hard constraints that must be enforced - triggers rejection if violated"
    )
    distractor: Optional[Union[str, List[InfoType]]] = Field(
        default=None,
        description="Off-topic content to naturally mention mid-conversation"
    )
    soft_preferences: Optional[List[str]] = Field(
        default=None,
        description="Soft preferences that are nice to have but not deal-breakers"
    )
    
    # === Multi-Aim Support ===
    aim_sequence: Optional["AimSequence"] = Field(
        default=None,
        description="Sequence of aims for multi-task scenarios (optional)"
    )
    
    # === Task-Specific Configuration (v2.0) ===
    task_config: Optional[dict] = Field(
        default=None,
        description="Task-specific configuration for User Simulator behavior control. "
                    "Legacy controller configuration; unused by the five-task release protocol."
    )
    
    # === Hidden State (v3.0) ===
    hidden_state: Optional[List[HiddenStateItem]] = Field(
        default=None,
        description="Information user knows but doesn't explicitly tell agent. "
                    "Drives user behavior (style, latent preferences, internal knowledge)."
    )
    
    def model_post_init(self, __context) -> None:
        """Post-initialization to ensure anchor compatibility"""
        # If we have legacy anchor but no anchor_config, create one
        if self.anchor and not self.anchor_config:
            object.__setattr__(self, 'anchor_config', AnchorConfig(
                initial_statement=self.anchor,
                underlying_problem_type="general",
                vagueness_strategy="legacy_format",
                is_disguised=False
            ))
        # If we have anchor_config but no legacy anchor, set it for compatibility
        elif self.anchor_config and not self.anchor:
            object.__setattr__(self, 'anchor', self.anchor_config.initial_statement)
    
    def get_anchor(self) -> str:
        """Get the anchor statement (Turn 0 opening)"""
        if self.anchor_config:
            return self.anchor_config.get_effective_anchor()
        return self.anchor or ""
    
    def get_anchor_config(self) -> AnchorConfig:
        """Get the anchor configuration, creating from legacy if needed"""
        if self.anchor_config:
            return self.anchor_config
        # Create from legacy anchor
        return AnchorConfig(
            initial_statement=self.anchor or "",
            underlying_problem_type="general",
            vagueness_strategy="legacy_format",
            is_disguised=False
        )
    
    def is_anchor_disguised(self) -> bool:
        """Check if anchor uses disguised strategy"""
        if self.anchor_config:
            return self.anchor_config.is_disguised
        return False
    
    def _get_value(self, item: InfoType) -> str:
        """Extract string value from either string or InfoItem"""
        if isinstance(item, str):
            return item
        return item.value
    
    def get_flow_info_values(self) -> List[str]:
        """Get flow_info as list of strings"""
        return [self._get_value(item) for item in self.flow_info]
    
    def get_constraint_values(self) -> List[str]:
        """Get constraints as list of strings"""
        return [self._get_value(item) for item in self.constraints]
    
    def get_distractor_value(self) -> Optional[str]:
        """Get distractor as single string (legacy compatibility)"""
        if self.distractor is None:
            return None
        if isinstance(self.distractor, str):
            return self.distractor
        if isinstance(self.distractor, list):
            return " ".join(self._get_value(item) for item in self.distractor)
        return str(self.distractor)
    
    def get_distractor_values(self) -> List[str]:
        """Get distractors as list of strings"""
        if self.distractor is None:
            return []
        if isinstance(self.distractor, str):
            return [self.distractor]
        if isinstance(self.distractor, list):
            return [self._get_value(item) for item in self.distractor]
        return [str(self.distractor)]

    def get_all_info_count(self) -> int:
        """Get total count of information pieces to reveal"""
        return len(self.flow_info)
    
    def get_constraint_count(self) -> int:
        """Get total count of hard constraints"""
        return len(self.constraints)
    
    def is_aim_driven(self) -> bool:
        """Check if this state queue uses aim-driven mode"""
        return self.aim_sequence is not None and len(self.aim_sequence.aims) > 0

    # =========================================================================
    # Hidden State Helpers (v3.0)
    # =========================================================================

    def get_hidden_state_by_category(
        self, category: HiddenStateCategory
    ) -> List[HiddenStateItem]:
        """Get hidden state items filtered by category."""
        if not self.hidden_state:
            return []
        return [h for h in self.hidden_state if h.category == category]

    def get_persona_styles(self) -> List[HiddenStateItem]:
        """All hidden state items that belong to the persona section.
        
        Matches both legacy 'persona_style' and pipeline 'identity'/'communication_style'.
        """
        if not self.hidden_state:
            return []
        return [h for h in self.hidden_state if h.category.is_persona]

    def get_latent_preferences(self) -> List[HiddenStateItem]:
        """All hidden state items that belong to the latent-preference section.
        
        Matches both legacy 'latent_preference' and pipeline 'preference'.
        """
        if not self.hidden_state:
            return []
        return [h for h in self.hidden_state if h.category.is_preference]

    def get_internal_knowledge(self) -> List[HiddenStateItem]:
        """All hidden state items that belong to the internal-knowledge section.
        
        Matches legacy 'internal_knowledge'.
        """
        if not self.hidden_state:
            return []
        return [h for h in self.hidden_state if h.category.is_knowledge]

    def get_hidden_state_values(self) -> List[str]:
        """Get all hidden state values as plain strings."""
        if not self.hidden_state:
            return []
        return [h.value for h in self.hidden_state]

    def get_disclosed_hidden_items(self) -> List[HiddenStateItem]:
        """Get conditional hidden items that have been disclosed during conversation."""
        if not self.hidden_state:
            return []
        return [
            h for h in self.hidden_state
            if h.overlap_mode == OverlapMode.CONDITIONAL
            and h.disclosed_during_conversation
        ]

    def get_explicit_flow_info_values(self) -> List[str]:
        """
        Get flow_info values for evaluation, overlap-aware.
        
        - Items NOT linked to any hidden_state → always included (unchanged).
        - Items linked to a hidden_state with overlap_mode='also_disclosed'
          → included (user did say it).
        - Items linked to a hidden_state with overlap_mode='conditional'
          AND disclosed_during_conversation=True → included (agent was told).
        - Items linked to a hidden_state with overlap_mode='none' are
          impossible by definition (none means no flow_info linkage).
        
        Effectively this returns ALL flow_info unless a hidden_state
        explicitly removes something (which the current design does not do —
        hidden_state only ADDS behavior, it does not delete flow_info).
        """
        if not self.hidden_state:
            # No hidden state → return everything (backward compatible)
            return self.get_flow_info_values()

        # Build a set of flow_info keys whose evaluation should be skipped
        # because they are ONLY reachable through a none-overlap hidden item.
        # In practice this set is empty unless future code adds such links.
        skip_keys: set = set()
        for h in self.hidden_state:
            if (
                h.overlap_mode == OverlapMode.NONE
                and h.linked_flow_info_key is not None
            ):
                skip_keys.add(h.linked_flow_info_key)

        result = []
        for item in self.flow_info:
            if isinstance(item, str):
                result.append(item)
            else:
                if item.key not in skip_keys:
                    result.append(item.value)
        return result

    def get_evaluation_relevant_info(self) -> Tuple[List[str], List[str]]:
        """
        PRIMARY method for evaluators.

        Returns (hard_eval_info, soft_eval_info):
          - hard: info the agent was explicitly told → must respond correctly
          - soft: info the agent was NOT told → bonus only

        Role-aware filtering (applied to flow_info items):
          - role == "user_input" (or missing/legacy) → hard requirement
          - role == "reference" → soft (agent derives this, not told by user)
          - role == "context" → soft (supplementary enrichment)
        """
        # Partition flow_info by role
        hard_flow: List[str] = []
        soft_flow: List[str] = []
        for item in self.flow_info:
            if isinstance(item, str):
                # Legacy string items → hard (backward compatible)
                hard_flow.append(item)
            else:
                role = getattr(item, "role", "user_input")
                if role == "user_input":
                    hard_flow.append(item.value)
                else:
                    # "reference" or "context" → soft
                    soft_flow.append(item.value)

        if not self.hidden_state:
            return hard_flow, soft_flow

        # With hidden_state: filter hard_flow through overlap-aware skip_keys
        skip_keys: set = set()
        for h in self.hidden_state:
            if (
                h.overlap_mode == OverlapMode.NONE
                and h.linked_flow_info_key is not None
            ):
                skip_keys.add(h.linked_flow_info_key)

        if skip_keys:
            # Remove skipped keys from hard_flow, but only for user_input items
            filtered_hard: List[str] = []
            for item in self.flow_info:
                if isinstance(item, str):
                    filtered_hard.append(item)
                else:
                    role = getattr(item, "role", "user_input")
                    if role == "user_input" and item.key not in skip_keys:
                        filtered_hard.append(item.value)
            hard_flow = filtered_hard

        hard_info = list(hard_flow)
        soft_info = list(soft_flow)

        for h in self.hidden_state:
            if h.overlap_mode == OverlapMode.ALSO_DISCLOSED:
                # The linked flow_info is already in hard_info.
                # The behavioral aspect is a soft bonus.
                soft_info.append(h.value)
            elif h.overlap_mode == OverlapMode.CONDITIONAL:
                if h.disclosed_during_conversation:
                    # Trigger was hit → agent was told → hard
                    hard_info.append(h.value)
                else:
                    # Never triggered → agent was never told → soft bonus
                    soft_info.append(h.value)
            elif h.overlap_mode == OverlapMode.NONE:
                # Pure hidden → soft bonus only
                soft_info.append(h.value)

        return hard_info, soft_info


class GroundTruthAnswer(BaseModel):
    """
    Ground Truth Answer for verifiable evaluation.
    
    This ensures that if all flow_info is collected and constraints are satisfied,
    the correct answer can be deterministically derived WITHOUT external lookup.
    
    Design Principles:
    1. The answer should be COMMON KNOWLEDGE (no internet search needed)
    2. The answer should be DETERMINISTIC given the constraints
    3. The answer should be EASILY VERIFIABLE with simple string matching
    
    Example Good Scenarios:
    - Math/Logic: "Given budget $100 and item costs $30 each, max items = 3"
    - Time/Date: "Meeting at 3pm for 2 hours ends at 5pm"
    - Geography: "Capital of France is Paris"
    - Simple Facts: "Water boils at 100°C at sea level"
    
    Example Bad Scenarios:
    - Product recommendations (requires real product database)
    - Current prices (changes over time)
    - Personal preferences (subjective)
    """
    model_config = {"extra": "allow"}  # Allow extra fields for converted data
    
    answer: str = Field(
        ...,
        description="The correct answer that should be derived from flow_info + constraints"
    )

    @field_validator('answer', mode='before')
    @classmethod
    def coerce_answer_to_str(cls, v):
        """Coerce non-string LLM outputs (dict with steps, list) to str."""
        if isinstance(v, dict):
            if 'steps' in v:
                return ' '.join(str(s) for s in v['steps'])
            return _json.dumps(v)
        if isinstance(v, list):
            return ' '.join(str(item) for item in v)
        if not isinstance(v, str):
            return str(v)
        return v

    answer_type: str = Field(
        default="exact",
        description="Type of answer matching: 'exact', 'contains', 'numeric', 'choice', 'recommendation', 'constraint_compliance', 'conditional_result'"
    )
    acceptable_variants: Optional[List[str]] = Field(
        default=None,
        description="List of acceptable answer variants (for 'contains' or 'choice' types)"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Explanation of how the answer is derived from flow_info + constraints"
    )
    required_elements: Optional[List[str]] = Field(
        default=None,
        description="List of elements that MUST appear in the answer for it to be correct"
    )
    forbidden_elements: Optional[List[str]] = Field(
        default=None,
        description="List of elements that must NOT be recommended (e.g., rejected candidates, allergens)"
    )
    # v4.0: Hidden state evaluation criteria (bonus scoring)
    hidden_state_bonus_criteria: Optional[List[str]] = Field(
        default=None,
        description="Bonus criteria for agent proactively discovering hidden user state"
    )
    hidden_state_max_bonus: float = Field(
        default=0.1,
        description="Maximum bonus points for hidden state awareness (default 0.1)"
    )
    # Fix 12: Sourced reasoning — each claim cites its source key for auditability
    sourced_reasoning: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of claim objects with source attribution. "
                    "Each entry: {claim, source_key, source_type, [rule_statement]}"
    )
    # Fields for converted trajectories (from external benchmarks)
    evaluation_criteria_text: Optional[str] = Field(
        default=None,
        description="A single sentence summarizing what we're checking for"
    )


class UserIntentionType(str, Enum):
    """Types of user intention relative to ground truth"""
    CONFLICTING = "conflicting"       # 50%: User intention conflicts with ground_truth in subtle ways
    ALIGNED = "aligned"               # 25%: User intention matches ground_truth exactly
    UNCERTAIN = "uncertain"           # 25%: User doesn't know what the correct answer should be


class UserIntentionResult(BaseModel):
    """
    User's Internal Intention/Desire - What the user secretly wants.
    
    This represents the user's TRUE internal preference, which may conflict with
    the objective ground_truth answer. This creates realistic scenarios where:
    
    1. CONFLICTING (50%): User's desire subtly conflicts with constraints/requirements.
       Example (Time/Physical Conflict):
       - Scenario: Planning travel itinerary
       - Constraint: Must arrive at airport by 14:00 (1-hour drive). Current time: 12:30.
       - Distractor: "There's a famous burger place on the way, 40-min queue, haven't been in ages"
       - User Intention: Really wants to stop for burgers, insists "I'll eat quickly"
       - Ground Truth: 12:30 + 40min(queue) + eating + 60min(drive) > 14:00, physically impossible
       - AI Response: Calculate time, firmly decline burger stop, explain time constraint
       - Evaluation: Did Agent reject the burger stop suggestion? (Yes/No)
       
    2. ALIGNED (25%): User genuinely wants what the ground_truth suggests.
       Example: User wants help calculating their budget and truly wants the correct answer.
       
    3. UNCERTAIN (25%): User doesn't actually know what the correct answer is.
       Example: User is asking because they genuinely don't know the capital of a country.
    
    This adds psychological realism to evaluations - a good AI assistant should:
    - For CONFLICTING: Calculate/verify constraints, firmly but politely decline impossible requests
    - For ALIGNED: Simply provide the correct answer
    - For UNCERTAIN: Provide clear, educational explanation
    """
    intention_type: UserIntentionType = Field(
        ...,
        description="Type of user intention relative to ground_truth"
    )
    
    internal_desire: Optional[str] = Field(
        default=None,
        description="What the user secretly wants (for CONFLICTING type)"
    )
    
    conflict_source: Optional[str] = Field(
        default=None,
        description="Which element conflicts: 'flow_info', 'constraint', or 'distractor'"
    )
    
    conflict_index: Optional[int] = Field(
        default=None,
        description="Index of the conflicting element"
    )
    
    conflict_reason: Optional[str] = Field(
        default=None,
        description="Why the user wants something different (emotional/situational reason)"
    )
    
    resolution_hint: Optional[str] = Field(
        default=None,
        description="How a good assistant should handle this situation"
    )
    
    def is_conflicting(self) -> bool:
        """Check if user intention conflicts with ground truth"""
        return self.intention_type == UserIntentionType.CONFLICTING
    
    def is_aligned(self) -> bool:
        """Check if user intention aligns with ground truth"""
        return self.intention_type == UserIntentionType.ALIGNED
    
    def is_uncertain(self) -> bool:
        """Check if user is uncertain about the answer"""
        return self.intention_type == UserIntentionType.UNCERTAIN


class PersonalityConfig(BaseModel):
    """Structured personality configuration"""
    type: str = Field(..., description="Personality type name")
    disclosure_style: str = Field(default="one_by_one", description="How info is disclosed")
    verbosity: str = Field(default="normal", description="Verbosity level")


class ExpectedTurnsConfig(BaseModel):
    """Expected turns range configuration"""
    min: int = Field(default=3, description="Minimum expected turns")
    max: int = Field(default=10, description="Maximum expected turns")


# Type aliases for flexible input formats
PersonalityInput = Union[PersonalityType, str, PersonalityConfig]
ExpectedTurnsInput = Union[int, ExpectedTurnsConfig]


class Scenario(BaseModel):
    """
    Complete Scenario Definition
    
    This is the primary configuration unit for data generation.
    Each scenario defines a complete user simulation case.
    
    Attributes:
        scenario_id: Unique identifier for the scenario
        domain: The domain of the scenario (shopping, scheduling, etc.)
        personality: User's personality type affecting conversation style
        custom_personality: Custom personality description if personality is CUSTOM
        state_queue: The StateQueue containing all information to manage
        language: Primary language for the scenario
        expected_turns: Estimated number of conversation turns
        difficulty: Difficulty level (1-5 or string like 'easy', 'medium', 'hard')
        tags: Optional tags for categorization
        metadata: Optional additional metadata
        base_instruction: Optional base instruction for the assistant
        verification_criteria: Optional verification criteria from JSON
    """
    scenario_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the scenario"
    )
    domain: str = Field(
        ...,
        description="Domain of the scenario (e.g., shopping, scheduling, travel)"
    )
    personality: PersonalityInput = Field(
        ...,
        description="User's personality type"
    )
    custom_personality: Optional[str] = Field(
        default=None,
        description="Custom personality description when personality is 'Custom'"
    )
    state_queue: StateQueue = Field(
        ...,
        description="The state queue containing information to reveal"
    )
    base_instruction: Optional[str] = Field(
        default=None,
        description="Base instruction for the assistant"
    )
    language: str = Field(
        default="en",
        description="Primary language code (zh, en, etc.)"
    )
    expected_turns: Optional[ExpectedTurnsInput] = Field(
        default=None,
        description="Estimated number of conversation turns"
    )
    difficulty: Union[int, str] = Field(
        default=3,
        description="Difficulty level from 1 (easy) to 5 (hard) or string"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tags for categorization"
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Additional metadata"
    )
    verification_criteria: Optional[dict] = Field(
        default=None,
        description="Verification criteria for evaluation"
    )
    ground_truth: Optional[GroundTruthAnswer] = Field(
        default=None,
        description="The ground truth answer for verifiable evaluation"
    )
    user_intention_result: Optional[UserIntentionResult] = Field(
        default=None,
        description="The user's internal intention/desire, which may conflict with ground_truth"
    )

    # === Composite Task Type Fields ===
    composite_type: Optional[str] = Field(
        default=None,
        description="Composite type ID (e.g., 'T9+T10') or None for single-type"
    )
    phase_boundary_aim_index: Optional[int] = Field(
        default=None,
        description="Aim index where Phase 2 starts (0-based)"
    )
    phase_1_task_type: Optional[str] = Field(
        default=None,
        description="Task type for Phase 1 (e.g., 'T9')"
    )
    phase_2_task_type: Optional[str] = Field(
        default=None,
        description="Task type for Phase 2 (e.g., 'T10')"
    )
    phase_ground_truths: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-phase ground truths: {'phase_1': {...}, 'phase_2': {...}}"
    )

    # === N-Phase Composite Fields (generalized) ===
    phase_boundaries: Optional[List[int]] = Field(
        default=None,
        description="List of aim indices where each phase starts (0-based)"
    )
    phase_task_types: Optional[List[str]] = Field(
        default=None,
        description="List of task types for each phase"
    )
    phase_weights: Optional[List[float]] = Field(
        default=None,
        description="Evaluation weights per phase"
    )

    def get_personality_description(self) -> str:
        """Get the personality description string"""
        if isinstance(self.personality, PersonalityConfig):
            return f"{self.personality.type} ({self.personality.disclosure_style}, {self.personality.verbosity})"
        elif self.personality == PersonalityType.CUSTOM and self.custom_personality:
            return self.custom_personality
        elif isinstance(self.personality, PersonalityType):
            return self.personality.value
        else:
            return str(self.personality)
    
    def get_difficulty_level(self) -> int:
        """Get difficulty as integer (1-5)"""
        if isinstance(self.difficulty, int):
            return self.difficulty
        difficulty_map = {"easy": 1, "medium": 3, "hard": 5}
        return difficulty_map.get(str(self.difficulty).lower(), 3)
    
    def get_expected_turns_range(self) -> Tuple[int, int]:
        """Get expected turns as (min, max) tuple"""
        if self.expected_turns is None:
            return (5, 10)
        if isinstance(self.expected_turns, int):
            return (max(3, self.expected_turns - 2), self.expected_turns + 2)
        if isinstance(self.expected_turns, ExpectedTurnsConfig):
            return (self.expected_turns.min, self.expected_turns.max)
        return (5, 10)
    
    def estimate_complexity(self) -> int:
        """
        Estimate scenario complexity based on information count.
        Returns a score from 1-10.
        """
        info_count = self.state_queue.get_all_info_count()
        constraint_count = self.state_queue.get_constraint_count()
        has_distractor = 1 if self.state_queue.distractor else 0
        
        # Weighted complexity score
        complexity = info_count * 2 + constraint_count * 3 + has_distractor
        return min(10, max(1, complexity))

    class Config:
        json_schema_extra = {
            "example": {
                "domain": "shopping",
                "personality": "Indecisive but chatty",
                "state_queue": {
                    "anchor": "I want to buy a birthday gift for my 6-year-old niece.",
                    "flow_info": [
                        "She's kind of a tomboy and loves science.",
                        "My budget is around 50 to 100 dollars."
                    ],
                    "constraints": [
                        "Definitely no clothes (I'm not sure about the size).",
                        "If it's a really cool space model, I can go slightly over budget."
                    ],
                    "distractor": "She looked so cute in that Elsa dress last year, though she's outgrown it now."
                }
            }
        }


# Deferred import to avoid circular dependency
# This allows AimSequence to be used in StateQueue while aim.py imports from scenario.py
def _rebuild_models():
    """Rebuild models to resolve forward references after all imports"""
    from benchmark.data_model.aim import AimSequence
    StateQueue.model_rebuild()

# Call rebuild when module is fully loaded
try:
    _rebuild_models()
except ImportError:
    # AimSequence might not be available yet during initial import
    pass
