"""
Task-Specific User Simulator Controls

This module implements specialized behavior controls for different task types (T1-T9).
Each task type has unique requirements for how the User Simulator should behave
to properly test specific LLM agent capabilities.

Task Categories:
- Retention (T1, T7, T8): Memory of conversation history
- Manipulation (T2, T5, T9): State and constraint management  
- Reasoning (T3, T4, T6): Proactive inference and conflict detection

Implementation Status:
✅ T1 (Self-Reference) - Vanilla prompt OK, with history validation
✅ T3 (Proactivity) - INFORMATION_HIDING: Only reveal info when explicitly asked
✅ T5 (Negative Constraint) - Vanilla prompt OK, with constraint detection
✅ T6 (Ambiguity Resolution) - HIDDEN_INTENT_LOCK: Hidden true intent
✅ T8 (Reference Resolution) - ENTITY_MASKING: Use references instead of names

🔲 T2 (State Update) - STATE_ANCHORING: Timing control for pivot
🔲 T4 (Conflict Detection) - IGNORANCE_MAINTENANCE: Pretend not to know conflict
🔲 T7 (Drip-Feed) - INSTRUCTION_QUEUE: One constraint per turn
🔲 T9 (Version Reversion) - VERSION_TAGGING: Track content versions

Author: Auto-generated for tau2-bench
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger
import re

# Import hidden state types for integration
try:
    from benchmark.data_model.scenario import HiddenStateItem, HiddenStateCategory
except ImportError:
    HiddenStateItem = None
    HiddenStateCategory = None


class TaskType(str, Enum):
    """Task type identifiers"""
    T1_SELF_REFERENCE = "T1"
    T2_STATE_UPDATE = "T2"
    T3_PROACTIVITY = "T3"
    T4_CONFLICT_DETECTION = "T4"
    T5_NEGATIVE_CONSTRAINT = "T5"
    T6_AMBIGUITY_RESOLUTION = "T6"
    T7_DRIP_FEED = "T7"
    T8_REFERENCE_RESOLUTION = "T8"
    T9_VERSION_REVERSION = "T9"


class SimulatorControlType(str, Enum):
    """Types of simulator control mechanisms"""
    GLOBAL_HISTORY_LOCK = "global_history_lock"       # T1: Validate history before callback
    INFORMATION_HIDING = "information_hiding"         # T3: Only reveal when asked
    NEGATIVE_LOGIC_FILTER = "negative_logic_filter"   # T5: Focus on absence
    HIDDEN_INTENT_LOCK = "hidden_intent_lock"         # T6: Hidden true intent
    ENTITY_MASKING = "entity_masking"                 # T8: Use references not names
    
    # Not yet implemented
    STATE_ANCHORING = "state_anchoring"               # T2: Track state before pivot
    IGNORANCE_MAINTENANCE = "ignorance_maintenance"   # T4: Pretend not to know
    INSTRUCTION_QUEUE = "instruction_queue"           # T7: Pop one per turn
    VERSION_TAGGING = "version_tagging"               # T9: Track versions


@dataclass
class TaskControlConfig:
    """Configuration for task-specific controls"""
    task_type: str
    control_type: SimulatorControlType
    
    # T3: Information hiding config
    hidden_info: Dict[str, str] = field(default_factory=dict)  # keyword -> info to reveal
    reveal_keywords: List[str] = field(default_factory=list)   # keywords that trigger reveal
    
    # T5: Negative constraint config  
    negative_constraints: List[str] = field(default_factory=list)  # things to exclude
    forbidden_patterns: List[str] = field(default_factory=list)    # regex patterns to detect violation
    
    # T6: Ambiguity resolution config
    true_intent: str = ""
    false_intents: List[str] = field(default_factory=list)
    ambiguous_instruction: str = ""
    
    # T8: Entity masking config
    entity_registry: Dict[str, Dict[str, str]] = field(default_factory=dict)  # entity_id -> {name, attributes}
    forbidden_entity_names: List[str] = field(default_factory=list)
    
    # T1: Self-reference config (for validation)
    tracked_agent_outputs: List[str] = field(default_factory=list)
    
    # =========================================================================
    # NEW: T2, T4, T7, T9 Configuration Fields
    # =========================================================================
    
    # T2: State update config
    state_a: Dict[str, Any] = field(default_factory=dict)  # Initial state content
    state_b: Dict[str, Any] = field(default_factory=dict)  # New state content
    pivot_turn: int = 0  # Turn number for pivot
    pivot_trigger: str = ""  # Trigger phrase for pivot
    contamination_keywords: List[str] = field(default_factory=list)  # Old state keywords to detect
    
    # T4: Conflict detection config
    agent_stated_fact: str = ""  # Fact the agent stated earlier
    user_conflicting_plan: str = ""  # User's plan that conflicts
    conflict_explanation: str = ""  # Explanation of the conflict
    acknowledgment_script: str = ""  # Script when agent catches conflict
    conflict_detection_keywords: List[str] = field(default_factory=list)
    
    # T7: Drip-feed config
    constraint_queue: List[str] = field(default_factory=list)  # Constraints to drip
    current_queue_index: int = 0  # Current position in queue
    final_aggregation_trigger: str = ""  # Trigger for final request
    drip_behavior: Dict[str, bool] = field(default_factory=dict)  # Behavior flags
    
    # T9: Version reversion config
    version_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # v1, v2, v3 -> content/characteristics
    target_version: str = ""  # Version to revert to
    forbidden_version_terms: List[str] = field(default_factory=list)  # Terms not allowed


class TaskSpecificController:
    """
    Controls User Simulator behavior based on task type.
    
    This controller modifies how the User Simulator generates responses
    to ensure proper testing of different LLM capabilities.
    
    Key Enhancement (v2.1): Constraint Disclosure Tracking
    - Tracks which constraints have been disclosed in the conversation
    - Prevents User Simulator from repeating disclosed constraints
    - Makes the simulation more realistic (real users don't constantly remind)
    """
    
    def __init__(self, config: Optional[TaskControlConfig] = None):
        self.config = config
        self._revealed_info: Set[str] = set()
        self._entity_references_used: List[str] = []
        self._agent_stated_facts: List[str] = []
        
        # v2.1: Constraint disclosure tracking
        self._disclosed_constraints: Dict[str, int] = {}  # constraint_key -> turn_disclosed
        self._current_turn: int = 0
        
        # v4.0: Hidden state integration
        self._hidden_state_items: List[Any] = []  # List[HiddenStateItem]
        
    def reset(self):
        """Reset controller state"""
        self._revealed_info.clear()
        self._entity_references_used.clear()
        self._agent_stated_facts.clear()
        self._disclosed_constraints.clear()
        self._current_turn = 0
        self._hidden_state_items.clear()
    
    # =========================================================================
    # Constraint Disclosure Tracking (v2.1)
    # =========================================================================
    
    def mark_constraint_disclosed(self, constraint_key: str, turn: int) -> None:
        """
        Mark a constraint as disclosed at a specific turn.
        
        Args:
            constraint_key: The key of the constraint
            turn: The turn number when disclosed
        """
        if constraint_key not in self._disclosed_constraints:
            self._disclosed_constraints[constraint_key] = turn
            logger.debug(f"Constraint '{constraint_key}' disclosed at turn {turn}")
    
    def is_constraint_disclosed(self, constraint_key: str) -> bool:
        """Check if a constraint has already been disclosed."""
        return constraint_key in self._disclosed_constraints
    
    def get_disclosed_constraints(self) -> Dict[str, int]:
        """Get all disclosed constraints with their disclosure turns."""
        return self._disclosed_constraints.copy()
    
    def should_repeat_constraint_in_message(
        self,
        constraint_key: str,
        is_required_by_current_aim: bool = False,
    ) -> bool:
        """
        Determine if a constraint should be repeated in the current message.

        v2.4: For ALL task types, constraints should NOT be repeated after
        initial disclosure. This creates more realistic user behavior where
        users mention things once and expect them to be remembered.

        v6.0: Exception for trap/final aims that explicitly require a
        constraint. If the aim's required_constraint_keys includes this key,
        it MUST be delivered regardless of prior disclosure — this is the
        whole point of trap aims (testing whether the agent remembers).

        Args:
            constraint_key: The key of the constraint
            is_required_by_current_aim: True if the current aim explicitly
                lists this constraint in required_constraint_keys

        Returns:
            False if constraint was disclosed and NOT required by current aim
        """
        if not self.config:
            return True  # No config = allow repetition

        # v2.4+v7.0: Constraints that were already disclosed should NOT be repeated.
        # Real users don't restate constraints every turn — they expect the agent to remember.
        # This applies even to trap/final aims: the whole point of a trap is testing
        # whether the agent remembers constraints WITHOUT the user reminding them.
        if self.is_constraint_disclosed(constraint_key):
            logger.debug(f"Blocking constraint repetition for '{constraint_key}' (already disclosed)")
            return False

        return True
    
    def increment_turn(self) -> None:
        """Increment the current turn counter."""
        self._current_turn += 1
    
    def get_task_type(self) -> Optional[str]:
        """Get the current task type"""
        return self.config.task_type if self.config else None
    
    # =========================================================================
    # Hidden State Integration (v4.0)
    # =========================================================================
    
    def set_hidden_state(self, hidden_state_items: List[Any]) -> None:
        """
        Set hidden state items from the scenario for task-specific integration.
        
        For T3 tasks: HIDDEN_LATENT items are merged into hidden_info for 
        INFORMATION_HIDING behavior.
        For T6 tasks: HIDDEN_KNOWLEDGE items reinforce HIDDEN_INTENT_LOCK behavior.
        
        Args:
            hidden_state_items: List of HiddenStateItem instances
        """
        self._hidden_state_items = hidden_state_items or []
        
        if not self.config or not self._hidden_state_items:
            return
        
        task_type = self.config.task_type.upper()
        
        if task_type == "T3" and HiddenStateCategory is not None:
            # Merge HIDDEN_LATENT / PREFERENCE items into T3's hidden_info dict
            for item in self._hidden_state_items:
                if item.category.is_preference:
                    if item.key not in self.config.hidden_info:
                        self.config.hidden_info[item.key] = item.value
                        logger.debug(f"T3: Merged latent preference '{item.key}' into hidden_info")
                    # Also add trigger keywords to reveal_keywords
                    if item.trigger_condition:
                        trigger_words = [w for w in item.trigger_condition.split() if len(w) > 3]
                        self.config.reveal_keywords.extend(trigger_words)
        
        elif task_type == "T6" and HiddenStateCategory is not None:
            # Merge INTERNAL_KNOWLEDGE items into T6's intent config
            for item in self._hidden_state_items:
                if item.category.is_knowledge:
                    # Reinforce the hidden intent with internal knowledge
                    if not self.config.true_intent and item.value:
                        self.config.true_intent = item.value
                        logger.debug(f"T6: Set true_intent from internal knowledge '{item.key}'")
    
    def get_hidden_state_items(self) -> List[Any]:
        """Get the current hidden state items."""
        return self._hidden_state_items
    
    # =========================================================================
    # Clarification Hint Filtering (v2.2, updated v2.4)
    # =========================================================================
    
    def filter_clarification_hint(
        self,
        hint: str,
        disclosed_constraint_keywords: List[str]
    ) -> str:
        """
        Filter a clarification hint to remove references to already-disclosed constraints.
        
        v2.4: This applies to ALL task types. Real users don't constantly
        remind agents about constraints they already mentioned.
        
        Args:
            hint: The original clarification hint
            disclosed_constraint_keywords: Keywords from disclosed constraints to filter out
            
        Returns:
            Filtered hint with constraint reminders removed
        """
        if not self.config:
            return hint
        
        # v2.4: Apply filtering to ALL task types (removed T1/T5 restriction)
        # Real users don't repeat themselves regardless of task type
        
        # Common reminder phrases that should be removed
        reminder_patterns = [
            r"[Rr]emember[,]?\s+(?:that\s+)?(?:they\s+)?(?:all\s+)?(?:need\s+to\s+be\s+)?",
            r"[Kk]eep(?:ing)?\s+in\s+mind\s+(?:that\s+)?(?:my\s+)?",
            r"[Aa]s\s+I\s+(?:mentioned|said)\s+(?:before|earlier)?[,]?\s*",
            r"[Dd]on'?t\s+forget\s+(?:that\s+)?(?:I\s+)?(?:need\s+)?",
            r"[Gg]iven\s+(?:my|that)\s+",
            r"[Ss]ince\s+I\s+(?:mentioned|said|have)\s+",
            r"[Mm]ake\s+sure\s+(?:to\s+)?(?:keep|remember)\s+",
        ]
        
        filtered_hint = hint
        
        # Remove reminder patterns followed by constraint keywords
        for pattern in reminder_patterns:
            for keyword in disclosed_constraint_keywords:
                # Build a regex that matches reminder + keyword context
                full_pattern = pattern + r"[^.!?]*" + re.escape(keyword.lower()) + r"[^.!?]*[.!?]?\s*"
                filtered_hint = re.sub(full_pattern, "", filtered_hint, flags=re.IGNORECASE)
        
        # Also remove standalone constraint mentions if they look like reminders
        for keyword in disclosed_constraint_keywords:
            # Remove sentences that contain the keyword and look like reminders
            sentence_with_reminder = r"[^.!?]*(?:[Rr]emember|[Kk]eep\s+in\s+mind|[Dd]on'?t\s+forget)[^.!?]*" + re.escape(keyword.lower()) + r"[^.!?]*[.!?]?\s*"
            filtered_hint = re.sub(sentence_with_reminder, "", filtered_hint, flags=re.IGNORECASE)
        
        # Clean up any double spaces
        filtered_hint = re.sub(r'\s+', ' ', filtered_hint).strip()
        
        if filtered_hint != hint:
            logger.info(f"Filtered clarification hint: '{hint[:50]}...' -> '{filtered_hint[:50]}...'")
        
        return filtered_hint
    
    def get_disclosed_constraint_keywords(self) -> List[str]:
        """
        Get keywords from disclosed constraints for filtering.
        
        Returns:
            List of keywords that should not be repeated
        """
        keywords = []
        
        # Add keys themselves
        keywords.extend(list(self._disclosed_constraints.keys()))
        
        # Add common constraint-related keywords from config
        if self.config:
            keywords.extend(self.config.negative_constraints)
            
        return keywords
    
    # =========================================================================
    # T1: Self-Reference - GLOBAL_HISTORY_LOCK
    # =========================================================================
    
    def t1_validate_callback_possible(
        self, 
        messages: List[Any],
        target_info: str
    ) -> bool:
        """
        T1: Validate that target information exists in conversation history
        before allowing a callback question.
        
        This prevents the simulator from asking about data that was never stated.
        
        Args:
            messages: Conversation history
            target_info: The information user wants to reference
            
        Returns:
            True if the information exists in history
        """
        history_text = " ".join([
            msg.content for msg in messages 
            if hasattr(msg, 'content') and msg.content
        ])
        
        # Check if any key terms from target_info appear in history
        # This is a simple heuristic - could be enhanced with semantic matching
        target_terms = re.findall(r'\b\w+\b', target_info.lower())
        history_lower = history_text.lower()
        
        matches = sum(1 for term in target_terms if term in history_lower)
        coverage = matches / len(target_terms) if target_terms else 0
        
        return coverage > 0.3  # At least 30% term coverage
    
    def t1_get_prompt_additions(self) -> str:
        """
        T1: Get additional prompt content for self-reference task.
        
        The user asks about something, gets an answer, then after DISTRACTIONS
        asks the agent to recall what it said earlier WITHOUT repeating the context.
        
        CRITICAL: Real users assume agents remember. They don't re-explain context.
        
        v2.1 Enhancement: Also includes NO_CONSTRAINT_REPEAT behavior.
        """
        # Get list of already disclosed constraints for context
        disclosed = list(self._disclosed_constraints.keys())
        disclosed_warning = ""
        if disclosed:
            disclosed_warning = f"""
⚠️ PREVIOUSLY DISCLOSED CONSTRAINTS (DO NOT REPEAT THESE):
{chr(10).join(['- ' + c for c in disclosed])}

You have ALREADY mentioned these constraints earlier in the conversation.
A normal human would NOT repeat them - they assume the agent remembers.
"""
        
        return f"""
## T1: Self-Reference Memory Test - BE REALISTIC

You are testing the agent's ability to remember its OWN earlier outputs.

{disclosed_warning}

⚠️⚠️⚠️ CRITICAL REALISTIC BEHAVIOR RULES ⚠️⚠️⚠️

1. **ASK FOR INFORMATION FIRST**
   - Request calculations, recommendations, dates, etc.
   - Let the agent provide detailed information
   
2. **GET DISTRACTED BY OTHER TOPICS**  
   - Talk about multiple unrelated things after getting the answer
   - This creates a "memory gap" before the callback
   - Don't stay focused on the original topic

3. **CALLBACK WITHOUT CONTEXT**
   - When asking the agent to recall earlier info, be CASUAL
   - Don't re-explain the context or what you're referring to
   - Assume the agent should remember

4. **NEVER REPEAT CONSTRAINTS OR CONTEXT**
   - If you mentioned an allergy, preference, or requirement earlier, DO NOT mention it again
   - Real users say things ONCE and expect them to be remembered
   - Repeating constraints defeats the purpose of the memory test

YOUR PERSONA: "The Casual User"
- You're having a natural conversation, not conducting a test
- You expect the agent to keep track of the conversation
- You don't re-explain things you've already discussed

❌ BAD BEHAVIOR (DO NOT DO):
- "Remember when I asked about [topic] and you said [details]? What was it again?"
- "Earlier we discussed [detailed context]. What did you conclude?"
- "Given my [allergy/constraint] that I mentioned before..." (repeating constraints!)
- "Since I have a [preference], can you recommend..." (reminding of disclosed info!)
- Re-explaining the original question before asking for recall

✅ GOOD BEHAVIOR (DO THIS):
- "What was that date again?"
- "How much was the total?"
- "Which one did you say was better?"
- "Can you remind me of that thing you mentioned?"
- "What would you recommend?" (without re-stating constraints)

EXAMPLE FLOW:
1. "I have a shellfish allergy. When is the Tancho festival in Hokkaido?"
2. Agent: "Late February to early March..."
3. [Several turns about other topics - lodging, food, transportation]
4. "Can you recommend some protein-rich seafood?" (NOT "...safe for my allergy")
5. "What dates were those again?" (NOT "what dates for the Tancho festival?")

The callback and recommendations should test if the agent tracks conversation context.
"""

    # =========================================================================
    # T3: Proactivity - INFORMATION_HIDING
    # =========================================================================
    
    def t3_should_reveal_info(
        self,
        agent_message: str,
        info_key: str
    ) -> bool:
        """
        T3: Check if agent's question triggers information reveal.
        
        Information is ONLY revealed if the agent explicitly asks about it.
        The user is passive and doesn't volunteer information.
        
        Args:
            agent_message: The agent's latest message
            info_key: The key of information to potentially reveal
            
        Returns:
            True if agent asked about this information
        """
        if not self.config:
            return False
            
        agent_lower = agent_message.lower()
        
        # Check if agent message contains any reveal keywords for this info
        for keyword in self.config.reveal_keywords:
            if keyword.lower() in agent_lower:
                # Check if this keyword maps to the info_key
                if info_key in self.config.hidden_info:
                    self._revealed_info.add(info_key)
                    return True
        
        # Also check for direct questions about the topic
        question_indicators = ["?", "what", "which", "how", "when", "where", "who", "do you", "can you tell"]
        has_question = any(ind in agent_lower for ind in question_indicators)
        
        if has_question and info_key.lower() in agent_lower:
            self._revealed_info.add(info_key)
            return True
            
        return False
    
    def t3_get_hidden_info_status(self) -> Dict[str, bool]:
        """Get status of which hidden info has been revealed"""
        if not self.config:
            return {}
        return {
            key: key in self._revealed_info 
            for key in self.config.hidden_info
        }
    
    def t3_get_prompt_additions(self) -> str:
        """
        T3: Get additional prompt content for proactivity task.
        
        User is PASSIVE and doesn't volunteer information.
        """
        hidden_list = "\n".join([
            f"- {key}: {value}" 
            for key, value in (self.config.hidden_info if self.config else {}).items()
        ])
        
        # Build list of forbidden values that should NEVER appear unless asked
        forbidden_values = []
        if self.config and self.config.hidden_info:
            for key, value in self.config.hidden_info.items():
                if key not in self._revealed_info:
                    # Extract key values (numbers, specific terms)
                    import re
                    numbers = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?', value)
                    forbidden_values.extend(numbers)
                    # Add specific terms
                    terms = re.findall(r'\b[A-Za-z]{4,}\b', value)
                    forbidden_values.extend([t for t in terms if len(t) > 4])
        
        forbidden_str = ", ".join(set(forbidden_values)) if forbidden_values else "N/A"
        
        return f"""
## T3: Proactivity Test - BE PASSIVE AND UNHELPFUL

You are testing the agent's ability to ASK QUESTIONS before giving recommendations.

⚠️⚠️⚠️ ABSOLUTE CRITICAL RULE ⚠️⚠️⚠️
DO NOT mention ANY of these specific values in your response:
{forbidden_str}

These are HIDDEN INFORMATION that you must NOT reveal unless the agent EXPLICITLY asks about them!

HIDDEN INFORMATION (DO NOT reveal unless agent asks SPECIFICALLY):
{hidden_list}

YOUR STRICT BEHAVIOR RULES:
1. **NEVER volunteer information** - If agent doesn't ask, you DON'T tell
2. **Be EXTREMELY vague** - Say things like "I'm not sure" or "I haven't really thought about it"
3. **Only answer DIRECT questions** - If agent asks "what's your budget?", THEN you can say the budget
4. **If agent gives generic advice** - Say "Hmm, I'm not sure that fits my situation" but DON'T explain why

YOUR PERSONA: "The Layman"
- You genuinely don't know what information is relevant
- You're not deliberately withholding - you just don't think to mention things
- You need the agent to ASK THE RIGHT QUESTIONS

EXAMPLE GOOD RESPONSES (when agent hasn't asked):
- "I'm not really sure what would work for me..."
- "Hmm, there are so many options, I don't know where to start"
- "Can you maybe ask me some questions to help narrow it down?"
- "I haven't really thought through all the details yet"

EXAMPLE BAD RESPONSES (revealing info without being asked):
- ❌ "I have a budget of $X" (agent didn't ask about budget!)
- ❌ "I'm looking for about X days" (agent didn't ask about duration!)
- ❌ "I prefer beaches" (agent didn't ask about preferences!)

Remember: You are TESTING if the agent is smart enough to ASK questions!
"""

    # =========================================================================
    # T5: Negative Constraint - NEGATIVE_LOGIC_FILTER
    # =========================================================================
    
    def t5_check_constraint_violated(self, agent_response: str) -> Tuple[bool, List[str]]:
        """
        T5: Check if agent's response violates any negative constraints.
        
        Args:
            agent_response: The agent's response to check
            
        Returns:
            Tuple of (is_violated, list of violations)
        """
        if not self.config:
            return False, []
            
        violations = []
        response_lower = agent_response.lower()
        
        for pattern in self.config.forbidden_patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                violations.append(f"Pattern '{pattern}' found in response")
        
        for constraint in self.config.negative_constraints:
            if constraint.lower() in response_lower:
                violations.append(f"Forbidden content '{constraint}' found")
        
        return len(violations) > 0, violations
    
    def t5_get_prompt_additions(self) -> str:
        """
        T5: Get additional prompt content for negative constraint task.
        
        User mentions what they DON'T want ONCE at the start, then moves on
        to other topics WITHOUT repeating the constraint. The challenge is
        whether the agent remembers when the final request is made.
        
        CRITICAL: Real users do NOT constantly remind agents about constraints.
        They mention it once, then expect the agent to remember.
        
        v2.1 Enhancement: Explicit constraint repeat blocking.
        """
        constraints = "\n".join([
            f"- {c}" 
            for c in (self.config.negative_constraints if self.config else [])
        ])
        
        # Get list of already disclosed constraints
        disclosed = list(self._disclosed_constraints.keys())
        disclosed_warning = ""
        if disclosed:
            disclosed_warning = f"""
🚫 CONSTRAINT REPETITION BLOCKER 🚫
These constraints have ALREADY been mentioned (turn numbers shown):
{chr(10).join([f'- {k} (disclosed at turn {v})' for k, v in self._disclosed_constraints.items()])}

YOU MUST NOT:
- Re-state these constraints in any form
- Hint at these constraints (e.g., "given what I said before...")
- Remind the agent about these constraints
- Use phrases like "as I mentioned", "remember I said", "keeping in mind my..."

The ENTIRE POINT of this test is to see if the agent remembers WITHOUT reminders!
"""
        
        return f"""
## T5: Negative Constraint Test - BE REALISTIC

You are testing the agent's ability to REMEMBER constraints mentioned earlier.

THE CONSTRAINTS (things to AVOID - but you mention these ONLY ONCE):
{constraints}

{disclosed_warning}

⚠️⚠️⚠️ CRITICAL REALISTIC BEHAVIOR RULES ⚠️⚠️⚠️

1. **MENTION CONSTRAINT ONLY ONCE AT THE START**
   - State the constraint naturally in your FIRST message
   - After that, NEVER mention or hint at the constraint again
   - This is how real users behave - they say something once and expect it to be remembered

2. **GET DISTRACTED BY OTHER TOPICS**
   - Move on to different subjects (drinks, events, logistics, etc.)
   - Talk about multiple unrelated things
   - Act like a normal person with many concerns, not just the constraint

3. **FINAL REQUEST WITHOUT REMINDER**
   - When making your final request (the "trap"), do NOT mention the constraint
   - Just ask for what you want naturally
   - The agent should remember without being reminded

4. **ONLY REACT IF VIOLATED**
   - If the agent violates the constraint in the final response, THEN express disappointment
   - But NOT before - you shouldn't be "monitoring" throughout

YOUR PERSONA: "The Normal User"
- You're a regular person, not a robot testing constraints
- You mention preferences once and expect them to be remembered
- You get distracted by other topics during the conversation
- You don't constantly remind people of your requirements

❌ BAD BEHAVIOR (DO NOT DO):
- "Remember, I mentioned earlier that..."
- "As I said before, I don't want..."
- "Just to remind you about the [constraint]..."
- "Make sure you don't forget about the [constraint]..."
- "Given my allergy/preference, can you..."
- "Since I need to avoid X, what about..."
- Mentioning the constraint multiple times
- Adding qualifiers like "safe for my allergy" or "that fits my requirements"

✅ GOOD BEHAVIOR (DO THIS):
- First message: "Tom has switched to vegetarian diet..." (constraint mentioned ONCE)
- Middle messages: "What drinks pair well?" "How about prep time?" (OTHER topics)
- Final message: "Can you suggest a menu for Tom's party?" (NO constraint reminder)
- If violated: "Wait, Tom is vegetarian - why suggest steak?"

ANOTHER EXAMPLE:
- First message: "I'm allergic to shellfish and nuts." (constraints mentioned ONCE)
- Middle messages: "What about cooking techniques?" "Any tips on presentation?"
- Final message: "Can you recommend some protein-rich seafood?" (NO "safe for my allergy")
- Agent should remember and avoid shellfish/nuts WITHOUT being reminded

This creates a realistic "memory trap" where the agent must track earlier info.
"""

    # =========================================================================
    # T6: Ambiguity Resolution - HIDDEN_INTENT_LOCK
    # =========================================================================
    
    def t6_check_agent_clarifies(self, agent_message: str) -> bool:
        """
        T6: Check if agent is asking for clarification.
        
        Args:
            agent_message: The agent's message
            
        Returns:
            True if agent is asking clarifying questions
        """
        clarification_indicators = [
            "could you clarify",
            "what do you mean",
            "can you be more specific",
            "do you mean",
            "which one",
            "are you referring to",
            "just to confirm",
            "to make sure i understand",
            "did you mean",
            "by that do you mean",
            "?",  # Any question
        ]
        
        agent_lower = agent_message.lower()
        return any(ind in agent_lower for ind in clarification_indicators)
    
    def t6_should_reveal_intent(self, agent_message: str) -> bool:
        """
        T6: Determine if user should reveal true intent.
        
        Only reveal if agent asks for clarification.
        """
        return self.t6_check_agent_clarifies(agent_message)
    
    def t6_get_prompt_additions(self) -> str:
        """
        T6: Get additional prompt content for ambiguity resolution task.
        """
        true_intent = self.config.true_intent if self.config else ""
        false_intents = self.config.false_intents if self.config else []
        ambiguous = self.config.ambiguous_instruction if self.config else ""
        
        false_list = "\n".join([f"- {fi}" for fi in false_intents])
        
        return f"""
## T6: Ambiguity Resolution Test

You are testing the agent's ability to DETECT AMBIGUITY and ASK for clarification.

YOUR AMBIGUOUS INSTRUCTION:
"{ambiguous}"

This instruction could mean multiple things:
{false_list}

YOUR TRUE INTENT (HIDDEN - only reveal if agent asks):
"{true_intent}"

YOUR BEHAVIOR:
1. Give the ambiguous instruction with confidence
2. If agent asks for clarification → Reveal your true intent
3. If agent guesses without asking → Express dissatisfaction if wrong

YOUR PERSONA: "The Trap Setter"
- You have a specific expectation but won't say it upfront
- You expect the agent to ask if unclear
- You'll judge harshly if agent guesses wrong

EXAMPLE RESPONSES:
- If agent clarifies: "Oh, I meant [true intent]. Thanks for asking!"
- If agent guesses wrong: "That's not what I meant at all..."
- If agent guesses right (lucky): "Yes, that's it!" (but note they didn't ask)
"""

    # =========================================================================
    # T8: Reference Resolution - ENTITY_MASKING
    # =========================================================================
    
    def t8_mask_entity_name(self, entity_name: str, position: int = 0) -> str:
        """
        T8: Generate a reference expression instead of using entity name.
        
        Args:
            entity_name: The entity name to mask
            position: Position in a list (1-indexed, 0 means use attribute reference)
            
        Returns:
            A reference expression (e.g., "the first one", "the cheap one")
        """
        if not self.config or entity_name not in self.config.entity_registry:
            return entity_name
            
        entity_info = self.config.entity_registry[entity_name]
        
        # Reference types to use
        if position > 0:
            # Positional reference
            ordinals = ["first", "second", "third", "fourth", "fifth"]
            if position <= len(ordinals):
                return f"the {ordinals[position-1]} one"
            return f"option {position}"
        
        # Attribute-based reference
        attributes = entity_info.get("attributes", {})
        if attributes:
            # Use the most distinctive attribute
            key, value = next(iter(attributes.items()))
            return f"the {value} one"
        
        # Pronominal reference
        return "that one"
    
    def t8_generate_reference(
        self, 
        entity_name: str,
        reference_type: str = "positional"
    ) -> str:
        """
        T8: Generate appropriate reference for an entity.
        
        Args:
            entity_name: The entity to reference
            reference_type: "positional", "attributive", or "pronominal"
            
        Returns:
            The reference expression
        """
        if reference_type == "pronominal":
            return "it"
        
        if not self.config or entity_name not in self.config.entity_registry:
            return entity_name
            
        entity_info = self.config.entity_registry.get(entity_name, {})
        
        if reference_type == "positional":
            position = entity_info.get("position", 1)
            ordinals = ["first", "second", "third", "fourth", "fifth"]
            if position <= len(ordinals):
                return f"the {ordinals[position-1]} one"
            return f"number {position}"
            
        elif reference_type == "attributive":
            attributes = entity_info.get("attributes", {})
            if attributes:
                key, value = next(iter(attributes.items()))
                return f"the {value} one"
        
        return "that one"
    
    def t8_is_forbidden_name(self, text: str) -> bool:
        """
        T8: Check if text contains forbidden entity names.
        """
        if not self.config:
            return False
            
        text_lower = text.lower()
        for name in self.config.forbidden_entity_names:
            if name.lower() in text_lower:
                return True
        return False
    
    def t8_get_prompt_additions(self) -> str:
        """
        T8: Get additional prompt content for reference resolution task.
        """
        if not self.config:
            return ""
            
        forbidden = ", ".join(self.config.forbidden_entity_names)
        entity_list = "\n".join([
            f"- Position {info.get('position', i+1)}: {name} ({info.get('attributes', {})})"
            for i, (name, info) in enumerate(self.config.entity_registry.items())
        ])
        
        return f"""
## T8: Reference Resolution Test - NEVER USE FULL NAMES

You are testing the agent's ability to track WHAT you're referring to.

⚠️ CRITICAL: After the agent lists options, you must NEVER use full entity names.

FORBIDDEN WORDS (do NOT say these): {forbidden}

ENTITY REGISTRY (you know this, but don't say the names!):
{entity_list}

REFERENCE TYPES TO USE:
1. POSITIONAL: "the first one", "the second option", "number 3"
2. ATTRIBUTIVE: "the cheap one", "the bigger one", "the red one"  
3. PRONOMINAL: "it", "that", "this one"

YOUR PERSONA: "The Lazy User"
- You hate repeating yourself
- You expect the agent to remember what you're talking about
- If asked to clarify, use a DIFFERENT reference type

EXAMPLE CONVERSATION:
Agent: "Here are 3 hotels: Hilton ($200), Marriott ($150), Motel 6 ($80)"
You: "Tell me more about the second one" [NOT "Tell me about the Marriott"]
Agent: "The Marriott has..."
You: "How far is it from downtown?" [using 'it' for Marriott]
You: "Compare it to the cheap one" [referring to Motel 6 by attribute]
"""

    # =========================================================================
    # T2: State Update - STATE_ANCHORING
    # =========================================================================
    
    def t2_get_current_phase(self) -> str:
        """
        T2: Determine if we're in pre-pivot or post-pivot phase.
        
        Returns:
            "pre_pivot" or "post_pivot"
        """
        if not self.config:
            return "pre_pivot"
        
        pivot_turn = self.config.pivot_turn
        if self._current_turn >= pivot_turn:
            return "post_pivot"
        return "pre_pivot"
    
    def t2_should_pivot(self) -> bool:
        """
        T2: Check if it's time to pivot to the new state.
        """
        if not self.config:
            return False
        return self._current_turn == self.config.pivot_turn
    
    def t2_check_contamination(self, agent_response: str) -> Tuple[bool, List[str]]:
        """
        T2: Check if agent's response contains contamination from old state.
        
        Args:
            agent_response: The agent's response to check
            
        Returns:
            Tuple of (is_contaminated, list of contamination keywords found)
        """
        if not self.config:
            return False, []
        
        contaminations = []
        response_lower = agent_response.lower()
        
        for keyword in self.config.contamination_keywords:
            if keyword.lower() in response_lower:
                contaminations.append(keyword)
        
        return len(contaminations) > 0, contaminations
    
    def t2_get_prompt_additions(self) -> str:
        """
        T2: Get additional prompt content for state update task.
        """
        if not self.config:
            return ""
        
        phase = self.t2_get_current_phase()
        
        state_a_str = "\n".join([f"- {k}: {v}" for k, v in self.config.state_a.items()])
        state_b_str = "\n".join([f"- {k}: {v}" for k, v in self.config.state_b.items()])
        pivot_trigger = self.config.pivot_trigger
        
        if phase == "pre_pivot":
            return f"""
## T2: State Update Test - PRE-PIVOT PHASE

You are in the INITIAL STATE (State A). Your requirements are:
{state_a_str}

YOUR BEHAVIOR:
1. Be clear and specific about State A requirements
2. Let the agent commit to State A
3. Ask follow-up questions about State A

YOUR PERSONA: "The Fickle Boss"
- You're decisive and clear (for now)
- You know what you want (State A)
- Build rapport with the agent about State A

DO NOT mention anything about State B yet!
"""
        else:
            return f"""
## T2: State Update Test - POST-PIVOT PHASE

⚠️⚠️⚠️ CRITICAL: You have COMPLETELY CHANGED YOUR MIND ⚠️⚠️⚠️

FORGET State A completely. Your NEW requirements are:
{state_b_str}

PIVOT PHRASE TO USE (if not already said):
"{pivot_trigger}"

YOUR BEHAVIOR:
1. COMPLETELY abandon State A - never mention it again
2. Be decisive and clear about State B
3. Act like State A never existed
4. If agent asks about State A, dismiss it completely

YOUR PERSONA: "The Fickle Boss"
- You've changed your mind, and that's final
- You don't want to hear about the old plan
- Be authoritative about the new direction

❌ DO NOT:
- Mention anything from State A
- Compare State A and State B
- Apologize for changing your mind
- Explain why you changed

✅ DO:
- Be confident about State B
- Act like State B was always the plan
- Expect agent to adapt immediately
"""

    # =========================================================================
    # T4: Conflict Detection - IGNORANCE_MAINTENANCE
    # =========================================================================
    
    def t4_detect_agent_catches_conflict(self, agent_message: str) -> bool:
        """
        T4: Check if the agent has caught the conflict.
        
        Args:
            agent_message: The agent's message
            
        Returns:
            True if agent appears to have caught the conflict
        """
        if not self.config:
            return False
        
        agent_lower = agent_message.lower()
        
        # Check for conflict detection keywords
        for keyword in self.config.conflict_detection_keywords:
            if keyword.lower() in agent_lower:
                return True
        
        # Also check for common conflict-catching phrases
        conflict_phrases = [
            "however",
            "but",
            "unfortunately",
            "i should mention",
            "keep in mind",
            "note that",
            "closed on",
            "not available",
            "won't be open",
            "isn't open"
        ]
        
        return any(phrase in agent_lower for phrase in conflict_phrases)
    
    def t4_get_acknowledgment_response(self) -> str:
        """
        T4: Get the acknowledgment script when agent catches conflict.
        """
        if not self.config or not self.config.acknowledgment_script:
            return "Oh! I completely forgot about that. Thanks for catching it!"
        return self.config.acknowledgment_script
    
    def t4_get_prompt_additions(self) -> str:
        """
        T4: Get additional prompt content for conflict detection task.
        """
        if not self.config:
            return ""
        
        agent_fact = self.config.agent_stated_fact
        user_plan = self.config.user_conflicting_plan
        acknowledgment = self.config.acknowledgment_script
        
        return f"""
## T4: Conflict Detection Test - MAINTAIN IGNORANCE

You are testing the agent's ability to CATCH CONFLICTS proactively.

THE CONFLICT:
- Agent stated earlier: "{agent_fact}"
- Your plan (which conflicts): "{user_plan}"

⚠️⚠️⚠️ CRITICAL BEHAVIOR RULES ⚠️⚠️⚠️

1. **PRETEND YOU DON'T KNOW ABOUT THE CONFLICT**
   - You genuinely believe your plan is fine
   - You are NOT testing the agent - you're just making plans
   - You forgot what the agent said earlier

2. **BE CONFIDENT ABOUT YOUR PLAN**
   - State your plan matter-of-factly
   - Don't ask "is this okay?" or "will this work?"
   - Assume it's a good plan

3. **DO NOT HINT AT THE CONFLICT**
   - Never say "I know you mentioned..." 
   - Never ask "Is X still closed/unavailable?"
   - Never reference the agent's earlier statement

4. **IF AGENT CATCHES THE CONFLICT:**
   - Act SURPRISED (you genuinely forgot)
   - Say something like: "{acknowledgment}"
   - Thank them for catching it

5. **IF AGENT DOES NOT CATCH IT:**
   - Continue with your flawed plan
   - Don't point out the conflict yourself
   - Let the conversation continue naturally

YOUR PERSONA: "The Confused User"
- You're confident but forgetful
- You trust the agent to catch mistakes
- You're NOT deliberately testing anyone

EXAMPLE BAD BEHAVIOR (DO NOT DO):
- "I want to visit X on Tuesday - is that still closed?"
- "You mentioned earlier that... but I was thinking..."
- "Just to check, is X available on that day?"

EXAMPLE GOOD BEHAVIOR:
- "Great! So I'll visit X on Tuesday morning, grab lunch, then explore."
- (Wait for agent to catch or not catch the conflict)
"""

    # =========================================================================
    # T7: Drip-Feed Accumulation - INSTRUCTION_QUEUE
    # =========================================================================
    
    def t7_get_next_constraint(self) -> Optional[str]:
        """
        T7: Get the next constraint from the queue.
        
        Returns:
            The next constraint to reveal, or None if queue is exhausted
        """
        if not self.config or not self.config.constraint_queue:
            return None
        
        idx = self.config.current_queue_index
        if idx < len(self.config.constraint_queue):
            constraint = self.config.constraint_queue[idx]
            self.config.current_queue_index += 1
            return constraint
        return None
    
    def t7_get_queue_status(self) -> Dict[str, Any]:
        """
        T7: Get current queue status.
        """
        if not self.config:
            return {"total": 0, "revealed": 0, "remaining": 0}
        
        total = len(self.config.constraint_queue)
        revealed = self.config.current_queue_index
        return {
            "total": total,
            "revealed": revealed,
            "remaining": total - revealed,
            "all_revealed": revealed >= total
        }
    
    def t7_is_final_request_turn(self) -> bool:
        """
        T7: Check if all constraints have been revealed (time for final request).
        """
        status = self.t7_get_queue_status()
        return status.get("all_revealed", False)
    
    def t7_get_prompt_additions(self) -> str:
        """
        T7: Get additional prompt content for drip-feed task.
        """
        if not self.config:
            return ""
        
        queue_status = self.t7_get_queue_status()
        total = queue_status["total"]
        revealed = queue_status["revealed"]
        remaining = queue_status["remaining"]
        
        # Build list of already revealed constraints
        revealed_constraints = self.config.constraint_queue[:revealed] if self.config.constraint_queue else []
        revealed_str = "\n".join([f"- {c} (ALREADY DISCLOSED - DO NOT REPEAT)" for c in revealed_constraints])
        
        # Next constraint to reveal (if any)
        next_constraint = ""
        if remaining > 0:
            next_idx = revealed
            next_constraint = self.config.constraint_queue[next_idx]
        
        if remaining > 0:
            return f"""
## T7: Drip-Feed Test - REVEAL ONE CONSTRAINT AT A TIME

You are testing the agent's ability to ACCUMULATE constraints revealed across turns.

QUEUE STATUS: {revealed}/{total} constraints revealed

ALREADY DISCLOSED (DO NOT REPEAT THESE):
{revealed_str if revealed_str else "(None yet)"}

🎯 THIS TURN: Reveal this ONE constraint naturally:
"{next_constraint}"

⚠️⚠️⚠️ CRITICAL BEHAVIOR RULES ⚠️⚠️⚠️

1. **REVEAL ONLY ONE NEW CONSTRAINT**
   - Mention "{next_constraint}" naturally
   - Do NOT reveal any other constraints yet
   - Do NOT summarize previous constraints

2. **NEVER REPEAT PREVIOUS CONSTRAINTS**
   - You mentioned those earlier
   - Assume the agent remembers
   - Real users don't constantly remind

3. **BE FRAGMENTED AND SCATTERED**
   - You're disorganized and remember things one at a time
   - You're NOT deliberately drip-feeding - you just forgot to mention it earlier

YOUR PERSONA: "The Fragmented User"
- You're scatterbrained and disorganized
- You remember requirements one at a time
- You assume the agent is keeping track

EXAMPLE GOOD BEHAVIOR:
Turn 1: "I need a recipe for dinner."
Turn 2: "Oh, it needs to be vegetarian." (ONE constraint)
Turn 3: "I just remembered - no gluten either." (ONE more, no repeat)
Turn 4: "Also, my friend has a nut allergy." (ONE more, no repeat)
"""
        else:
            return f"""
## T7: Drip-Feed Test - FINAL REQUEST (ALL CONSTRAINTS DISCLOSED)

All {total} constraints have been revealed across previous turns:
{revealed_str}

🎯 NOW: Make your FINAL REQUEST.

⚠️⚠️⚠️ CRITICAL: DO NOT SUMMARIZE THE CONSTRAINTS ⚠️⚠️⚠️

The agent should remember ALL constraints WITHOUT you reminding them.

YOUR REQUEST should be SIMPLE:
- "So what do you recommend?"
- "Can you give me a complete menu?"
- "What's your final suggestion?"

❌ DO NOT SAY:
- "Given all the requirements I mentioned..."
- "Remember, it needs to be vegetarian, gluten-free, and nut-free..."
- "Keeping in mind everything I said..."

✅ DO SAY:
- "Great, so what complete menu would you recommend?"
- "Can you put together a full recommendation for me?"

YOUR PERSONA: "The Expectant User"
- You've told them everything they need
- You expect them to remember without reminders
- You'll be disappointed if they forgot anything
"""

    # =========================================================================
    # T9: Version Reversion - VERSION_TAGGING
    # =========================================================================
    
    def t9_get_version_reference(self, version_id: str) -> str:
        """
        T9: Generate a characteristic-based reference for a version.
        
        Args:
            version_id: The version to reference (e.g., "v1")
            
        Returns:
            A reference based on characteristics (not version number)
        """
        if not self.config or version_id not in self.config.version_registry:
            return "the earlier version"
        
        version_info = self.config.version_registry[version_id]
        characteristics = version_info.get("characteristics", [])
        content_markers = version_info.get("content_markers", [])
        
        # Build natural reference from characteristics
        if content_markers:
            # Use a specific content marker
            marker = content_markers[0]
            return f"the version that started with '{marker}'"
        elif characteristics:
            # Use characteristics
            char = characteristics[0]
            return f"the more {char} version"
        
        return "that earlier version"
    
    def t9_is_forbidden_term(self, text: str) -> bool:
        """
        T9: Check if text contains forbidden version terms.
        """
        if not self.config:
            return False
        
        text_lower = text.lower()
        for term in self.config.forbidden_version_terms:
            if term.lower() in text_lower:
                return True
        return False
    
    def t9_get_prompt_additions(self) -> str:
        """
        T9: Get additional prompt content for version reversion task.
        """
        if not self.config:
            return ""
        
        target = self.config.target_version
        forbidden = ", ".join(self.config.forbidden_version_terms)
        
        # Get target version characteristics
        target_info = self.config.version_registry.get(target, {})
        target_characteristics = target_info.get("characteristics", [])
        target_markers = target_info.get("content_markers", [])
        
        characteristics_str = ", ".join(target_characteristics) if target_characteristics else "unknown"
        markers_str = ", ".join([f'"{m}"' for m in target_markers[:2]]) if target_markers else "unknown"
        
        # Build version registry display
        registry_str = "\n".join([
            f"- {vid}: {vinfo.get('characteristics', [])}"
            for vid, vinfo in self.config.version_registry.items()
        ])
        
        return f"""
## T9: Version Reversion Test - REFER BY CHARACTERISTICS

You are testing the agent's ability to identify versions by CHARACTERISTICS, not numbers.

VERSION REGISTRY (for your reference, but don't use version names!):
{registry_str}

TARGET VERSION TO REQUEST: {target}
- Characteristics: {characteristics_str}
- Content markers: {markers_str}

⚠️⚠️⚠️ FORBIDDEN PHRASES (NEVER SAY THESE) ⚠️⚠️⚠️
{forbidden}

YOUR BEHAVIOR:
1. You've gone through multiple edits and now want to go back
2. Describe the version you want by its CHARACTERISTICS
3. NEVER use version numbers or ordinal references

EXAMPLE GOOD REFERENCES:
- "Can you bring back the version that was more formal?"
- "I liked the one that started with 'Dear Ms. Johnson'"
- "Go back to the detailed version with specific dates"

EXAMPLE BAD REFERENCES (DO NOT USE):
- "Go back to the first version" ❌
- "I want version 1" ❌
- "Restore the original" ❌
- "The initial version" ❌

YOUR PERSONA: "The Regretful User"
- You've been through multiple edits
- You realize the earlier version was better
- You describe what you liked about it, not when you made it
"""

    # =========================================================================
    # Main Interface
    # =========================================================================
    
    def get_no_repeat_constraint_prompt(self) -> str:
        """
        Get a universal prompt addition that prevents constraint repetition.
        
        v2.4: This applies to ALL task types, not just T1/T5.
        Real users don't repeat themselves - they expect agents to remember.
        The prompt explicitly lists disclosed constraints and instructs
        the simulator not to repeat them.
        
        Returns:
            Prompt addition for no-repeat behavior, or empty string if no constraints disclosed
        """
        if not self._disclosed_constraints:
            return ""
        
        task_type = self.config.task_type.upper() if self.config else ""
        
        # v2.4: Apply to ALL task types (removed T1/T5 restriction)
        # Real users don't repeat themselves regardless of task type
        
        return f"""
## ⚠️ CONSTRAINT REPETITION PREVENTION (ACTIVE) ⚠️

The following information has ALREADY been disclosed in this conversation:
{chr(10).join([f'  • "{k}" (mentioned at turn {v})' for k, v in self._disclosed_constraints.items()])}

### STRICT RULES - YOU MUST FOLLOW THESE:

1. **DO NOT repeat any of the above information in your message**
   - Not directly, not indirectly, not as a reminder, not as context

2. **DO NOT use phrases that reference previous disclosures:**
   - ❌ "given my allergy..."
   - ❌ "since I mentioned..."
   - ❌ "keeping in mind my..."
   - ❌ "as I said before..."
   - ❌ "considering my preference..."
   - ❌ "safe for my [condition]..."
   - ❌ "that fits my requirements..."
   - ❌ "especially since..." (when followed by constraint reference)

3. **ASSUME the agent remembers everything**
   - Real humans don't repeat themselves constantly
   - If you already told someone you're allergic to shellfish, you don't say it again every time you ask for food recommendations

4. **Your message should be SIMPLE and NATURAL:**
   - ✅ "Can you recommend some protein-rich seafood?"
   - ✅ "What would you suggest for dinner?"
   - ✅ "Any good options you'd recommend?"
   - NOT: "Can you recommend some seafood that's safe for my allergy?"

### WHY THIS MATTERS:
This test evaluates whether the AI agent can REMEMBER constraints from earlier in the conversation.
If you keep reminding the agent, you're defeating the purpose of the test!
"""
    
    def get_task_prompt_additions(self) -> str:
        """
        Get task-specific prompt additions based on configured task type.
        
        Returns:
            Additional prompt content for the task type
        """
        if not self.config:
            return ""
            
        task_type = self.config.task_type.upper()
        
        # Get the base task-specific prompt
        base_prompt = ""
        if task_type == "T1":
            base_prompt = self.t1_get_prompt_additions()
        elif task_type == "T2":
            base_prompt = self.t2_get_prompt_additions()
        elif task_type == "T3":
            base_prompt = self.t3_get_prompt_additions()
        elif task_type == "T4":
            base_prompt = self.t4_get_prompt_additions()
        elif task_type == "T5":
            base_prompt = self.t5_get_prompt_additions()
        elif task_type == "T6":
            base_prompt = self.t6_get_prompt_additions()
        elif task_type == "T7":
            base_prompt = self.t7_get_prompt_additions()
        elif task_type == "T8":
            base_prompt = self.t8_get_prompt_additions()
        elif task_type == "T9":
            base_prompt = self.t9_get_prompt_additions()
        else:
            logger.warning(f"Task type {task_type} not yet implemented")
        
        # Add the no-repeat constraint prompt if applicable
        no_repeat_prompt = self.get_no_repeat_constraint_prompt()
        
        if no_repeat_prompt:
            return base_prompt + "\n" + no_repeat_prompt
        
        return base_prompt
    
    def should_allow_info_reveal(
        self,
        info_key: str,
        agent_message: str
    ) -> bool:
        """
        Check if information should be revealed based on task type rules.
        
        Args:
            info_key: The key of information to reveal
            agent_message: The agent's latest message
            
        Returns:
            True if information can be revealed
        """
        if not self.config:
            return True  # No restrictions without config
            
        task_type = self.config.task_type.upper()
        
        if task_type == "T3":
            # T3: Only reveal if agent asks
            return self.t3_should_reveal_info(agent_message, info_key)
        elif task_type == "T6":
            # T6: Reveal true intent only if agent clarifies
            if "true_intent" in info_key.lower():
                return self.t6_should_reveal_intent(agent_message)
        
        # Default: allow reveal
        return True
    
    def process_user_message(
        self,
        generated_message: str,
        messages: List[Any]
    ) -> str:
        """
        Post-process the generated user message based on task type.
        
        This can modify the message to enforce task-specific constraints.
        
        v2.4 Enhancement: Remove constraint reminders for ALL task types.
        Real users don't repeat themselves regardless of task type.
        
        Args:
            generated_message: The LLM-generated user message
            messages: Conversation history
            
        Returns:
            Processed message
        """
        if not self.config:
            return generated_message
            
        task_type = self.config.task_type.upper()
        processed = generated_message
        
        # v2.4: Remove constraint reminders for ALL task types (not just T1/T5)
        # Real users don't repeat themselves - this creates more realistic simulation
        if self._current_turn > 1:
            processed = self._remove_constraint_reminders(processed)
        
        if task_type == "T8":
            # T8: Replace any forbidden entity names with references
            for name in self.config.forbidden_entity_names:
                if name.lower() in processed.lower():
                    # Find position of this entity
                    position = 0
                    for entity_name, info in self.config.entity_registry.items():
                        if entity_name.lower() == name.lower():
                            position = info.get("position", 1)
                            break
                    reference = self.t8_mask_entity_name(name, position)
                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(name), re.IGNORECASE)
                    processed = pattern.sub(reference, processed)
                    logger.debug(f"T8: Masked '{name}' -> '{reference}'")
            return processed
        
        return processed
    
    def _remove_constraint_reminders(self, message: str) -> str:
        """
        Remove phrases that remind the agent about previously disclosed constraints.
        
        v2.4: This applies to ALL task types to create realistic user behavior.
        Real users don't constantly remind agents about their constraints.
        
        Args:
            message: The generated message
            
        Returns:
            Message with constraint reminders removed
        """
        if not self._disclosed_constraints:
            return message
        
        original_message = message
        
        # Get keywords from disclosed constraints
        disclosed_keywords = self.get_disclosed_constraint_keywords()
        
        # Patterns that indicate a constraint reminder
        reminder_phrases = [
            # Direct reminders
            (r"[Rr]emember(?:,)?\s+(?:that\s+)?(?:they\s+)?(?:all\s+)?(?:need\s+to\s+)?(?:be\s+)?", ""),
            (r"[Aa]s\s+I\s+(?:mentioned|said)\s+(?:before|earlier)?(?:,)?\s*", ""),
            (r"[Kk]eep(?:ing)?\s+in\s+mind\s+(?:that\s+)?(?:my\s+)?", ""),
            (r"[Dd]on'?t\s+forget\s+(?:that\s+)?(?:I\s+)?(?:need\s+)?", ""),
            (r"[Gg]iven\s+(?:my|that\s+I)\s+", ""),
            (r"[Ss]ince\s+I\s+(?:mentioned|said|have|need)\s+", "Since I need "),
            (r"[Mm]ake\s+sure\s+(?:to\s+)?(?:keep|remember)\s+", ""),
            (r"[Jj]ust\s+(?:to\s+)?remind\s+you(?:,)?\s*", ""),
            # Qualifier phrases before constraints
            (r"(?:that\s+is\s+)?safe\s+for\s+my\s+\w+(?:,)?\s*", ""),
            (r"(?:that\s+)?fits?\s+my\s+(?:requirements?|needs?|preferences?)(?:,)?\s*", ""),
            (r"(?:that\s+)?meets?\s+my\s+(?:requirements?|needs?|preferences?)(?:,)?\s*", ""),
            (r"(?:that\s+)?works?\s+with\s+my\s+(?:\w+)(?:,)?\s*", ""),
        ]
        
        # Apply reminder phrase removal
        for pattern, replacement in reminder_phrases:
            message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
        
        # Remove sentences that contain both a reminder phrase AND a disclosed keyword
        for keyword in disclosed_keywords:
            if len(keyword) < 3:  # Skip very short keywords
                continue
            # Pattern: sentence containing reminder + keyword
            sentence_pattern = r'[^.!?]*(?:' + '|'.join([
                r'[Rr]emember',
                r'[Aa]s\s+I\s+(?:mentioned|said)',
                r'[Kk]eep(?:ing)?\s+in\s+mind',
                r'[Dd]on\'?t\s+forget',
            ]) + r')[^.!?]*' + re.escape(keyword) + r'[^.!?]*[.!?]\s*'
            message = re.sub(sentence_pattern, "", message, flags=re.IGNORECASE)
        
        # Clean up double spaces and trim
        message = re.sub(r'\s+', ' ', message).strip()
        
        # Ensure the message is not empty
        if not message.strip():
            message = original_message  # Fallback to original if over-filtered
        
        if message != original_message:
            logger.info(f"Removed constraint reminders from message. Before: {len(original_message)} chars, After: {len(message)} chars")
        
        return message


def create_controller_from_scenario(scenario: Any) -> Optional[TaskSpecificController]:
    """
    Create a TaskSpecificController from a scenario's task configuration.
    
    Args:
        scenario: The Scenario object containing task_config
        
    Returns:
        Configured TaskSpecificController or None
    """
    if not hasattr(scenario, 'state_queue'):
        return None
        
    state_queue = scenario.state_queue
    
    # Check for task_config in state_queue
    if not hasattr(state_queue, 'task_config') or not state_queue.task_config:
        return None
        
    task_config = state_queue.task_config
    task_type = task_config.get("task_type", "").upper()
    
    if not task_type:
        return None
    
    # Build TaskControlConfig based on task type
    config = TaskControlConfig(
        task_type=task_type,
        control_type=SimulatorControlType.GLOBAL_HISTORY_LOCK,  # Default
    )
    
    # Extract task-specific configuration
    if task_type == "T3":
        config.control_type = SimulatorControlType.INFORMATION_HIDING
        config.hidden_info = task_config.get("hidden_info", {})
        config.reveal_keywords = task_config.get("reveal_keywords", [])
        
    elif task_type == "T5":
        config.control_type = SimulatorControlType.NEGATIVE_LOGIC_FILTER
        config.negative_constraints = task_config.get("negative_constraints", [])
        config.forbidden_patterns = task_config.get("forbidden_patterns", [])
        
    elif task_type == "T6":
        config.control_type = SimulatorControlType.HIDDEN_INTENT_LOCK
        config.true_intent = task_config.get("true_intent", "")
        config.false_intents = task_config.get("false_intents", [])
        config.ambiguous_instruction = task_config.get("ambiguous_instruction", "")
        
    elif task_type == "T8":
        config.control_type = SimulatorControlType.ENTITY_MASKING
        config.entity_registry = task_config.get("entity_registry", {})
        config.forbidden_entity_names = task_config.get("forbidden_entity_names", [])
    
    # =========================================================================
    # NEW: T2, T4, T7, T9 Configuration Parsing
    # =========================================================================
    
    elif task_type == "T2":
        config.control_type = SimulatorControlType.STATE_ANCHORING
        config.state_a = task_config.get("state_a", {})
        config.state_b = task_config.get("state_b", {})
        config.pivot_turn = task_config.get("pivot_turn", 3)
        config.pivot_trigger = task_config.get("pivot_trigger", "I've changed my mind")
        config.contamination_keywords = task_config.get("contamination_keywords", [])
        
    elif task_type == "T4":
        config.control_type = SimulatorControlType.IGNORANCE_MAINTENANCE
        config.agent_stated_fact = task_config.get("agent_stated_fact", "")
        config.user_conflicting_plan = task_config.get("user_conflicting_plan", "")
        config.conflict_explanation = task_config.get("conflict_explanation", "")
        config.acknowledgment_script = task_config.get("acknowledgment_script", "")
        config.conflict_detection_keywords = task_config.get("conflict_detection_keywords", [])
        
    elif task_type == "T7":
        config.control_type = SimulatorControlType.INSTRUCTION_QUEUE
        config.constraint_queue = task_config.get("constraint_queue", [])
        config.current_queue_index = 0  # Always start at 0
        config.final_aggregation_trigger = task_config.get("final_aggregation_trigger", "")
        config.drip_behavior = task_config.get("drip_behavior", {})
        
    elif task_type == "T9":
        config.control_type = SimulatorControlType.VERSION_TAGGING
        config.version_registry = task_config.get("version_registry", {})
        config.target_version = task_config.get("target_version", "")
        config.forbidden_version_terms = task_config.get("forbidden_version_terms", [])
    
    return TaskSpecificController(config)
