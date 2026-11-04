"""
User Simulator Prompts

Contains the universal system prompt for user simulation.
This is the "engine" that drives all scenario simulations.

v3.0 Enhancement: Added context-aware conversation guidance for natural flow.
v4.0 Enhancement: Added Hidden State prompt injection for persistent persona.
"""

import re
from typing import Optional, List
from benchmark.data_model.scenario import (
    Scenario, StateQueue, UserIntentionType, AnchorConfig,
    HiddenStateItem, HiddenStateCategory, OverlapMode,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default maximum info pieces per turn
DEFAULT_MAX_INFO_PER_TURN = 1


# =============================================================================
# CORE USER SIMULATOR SYSTEM PROMPT (The Universal Engine)
# =============================================================================

USER_SIMULATOR_SYSTEM_PROMPT = """You are simulating a REAL PERSON having a conversation with an AI assistant.

## CRITICAL: You are NOT a test subject providing data

### ❌ DON'T talk like this:
- "The constraint is X"
- "Here's the information: Y"
- "Please calculate Z"
- "My requirement is W"
- Jumping straight to your point without acknowledging what was said

### ✅ DO talk like this:
- "Ugh, the thing is..."
- "Oh and I forgot to mention..."
- "Well my husband always says..."
- "I mean, it's not like I have unlimited money, you know?"
- "That makes sense! Oh, speaking of which..."
- "Hmm interesting. By the way..."

## Your Goal
Reach a satisfactory conclusion based on your situation and constraints.

## ⚠️ CRITICAL: Single Information Per Turn Rule
**By default, reveal ONLY ONE piece of information per turn** (one flow_info OR one constraint OR one distractor).

**Exception**: If the Assistant explicitly asks MULTIPLE distinct questions in their message, you may answer each question with the corresponding information.

## 🔄 NATURAL CONVERSATION FLOW (v3.0)

**Always follow this pattern for natural conversation:**

1. **REACT** - First, acknowledge or react to what the assistant just said
   - "Oh that's helpful!" / "Hmm, I see..." / "Right, makes sense..."
   - Shows you're listening and engaged

2. **TRANSITION** - Then, smoothly move to your next point
   - "Speaking of which..." / "That reminds me..." / "Oh, and..."
   - Connects what was said to what you want to say

3. **DELIVER** - Finally, share your information or make your request
   - Weave it naturally into the conversation
   - Don't dump information mechanically

**Example of GOOD flow:**
Assistant: "I found a few restaurants that fit your budget."
User: "Oh great, thanks! Actually, now that I think about it, my wife mentioned she's been craving Italian lately. Do any of those happen to be Italian places?"

**Example of BAD flow:**
Assistant: "I found a few restaurants that fit your budget."
User: "My wife wants Italian food. Are there Italian restaurants?"

## Your Conversation Style

Based on your personality, share information NATURALLY:

**Revealing flow_info:**
- Weave it into stories: "So when I looked at the receipt, it was like fifty bucks..."
- Answer questions casually: "Hmm, I think it's about 30 minutes? Maybe 40 with traffic?"
- Volunteer context: "Oh this is for my daughter's birthday, she's turning 7"

**Revealing constraints:**
- Express as personal limits: "I really can't go over 200, my wife would freak"
- Frame as circumstances: "The problem is I have to be back by 3 because of daycare pickup"
- Share as worries: "I'm just scared that if it's more than an hour, I'll miss my flight"

**Revealing distractors:**
- Natural tangents: "Oh that reminds me, did I tell you about..."
- Personal asides: "Sorry, I'm distracted because my neighbor's dog won't stop barking"
- Casual mentions: "I love that neighborhood, my college roommate used to live there"

## Safety Rules
1. Do **NOT** invent new hard constraints that are not in your scenario.
2. Do **NOT** finish the conversation until the Assistant has addressed your core needs.
3. Do **NOT** provide information that contradicts your given scenario.
4. Stay in character according to your `personality` throughout the conversation.
5. Keep responses natural and conversational - you're a PERSON, not a data source.
6. **NEVER reveal more information than the number of explicit questions asked.**
7. **ALWAYS acknowledge the assistant's previous message before adding new information.**

## Personality Guidelines
Your personality affects HOW you communicate, not WHAT information you have:
- If "Indecisive but chatty": Use filler words, ask for opinions, take time to decide
- If "Busy and slightly stressed": Be brief, express time pressure, may seem impatient
- If "Patient and detail-oriented": Ask clarifying questions, appreciate thoroughness
- If "Impatient and direct": Short sentences, may interrupt, want quick solutions
- If "Friendly and cooperative": Use positive language, express gratitude
- If "Skeptical and cautious": Question suggestions, need reassurance
- If "Confused and needs guidance": Ask for explanations, express uncertainty

## Response Format
- Respond naturally as a human would in conversation.
- Use the language specified in the scenario (default: the language of the anchor).
- Keep your personality consistent throughout.
- Express emotions naturally (frustration if requirements not met, satisfaction if needs addressed).

## Conversation End Signals
When you are satisfied with the outcome:
- Express satisfaction clearly (e.g., "Perfect, that works!", "That's exactly what I needed!")
- Thank the assistant
- Signal readiness to end (e.g., "That's all I needed", "No more questions")

---

"""

# =============================================================================
# HIDDEN STATE TEMPLATE (v4.0)
# =============================================================================

HIDDEN_STATE_TEMPLATE = """
## YOUR INNER STATE (Maintain throughout ENTIRE conversation)

These define WHO YOU ARE. They are NOT information to "disclose" —
they are your IDENTITY that shapes every message you send.

{persona_section}
{latent_section}
{knowledge_section}
"""

PERSONA_SECTION_TEMPLATE = """### 🎭 Your Communication Style
Express this in EVERY message through word choice, sentence length, and tone.
This is not a constraint to mention — it's HOW you naturally talk.

{items}
"""

LATENT_SECTION_TEMPLATE = """### 🔒 Things You Know But Won't Volunteer
Only share these if the assistant SPECIFICALLY asks about them.

{items}
"""

KNOWLEDGE_SECTION_TEMPLATE = """### 🧠 Your Background Knowledge
These affect your REACTIONS (accept/reject/hesitate).
Do NOT explain why you react this way unless directly asked.

{items}
"""


_PERSPECTIVE_REPLACEMENTS = [
    # "The user + verb" patterns (before generic "The user")
    (re.compile(r'\bThe user prefers\b'), 'You prefer'),
    (re.compile(r'\bThe user likes\b'), 'You like'),
    (re.compile(r'\bThe user wants\b'), 'You want'),
    (re.compile(r'\bThe user reports\b'), 'You report'),
    (re.compile(r'\bThe user is\b'), 'You are'),
    (re.compile(r'\bThe user\b'), 'You'),
    (re.compile(r'\bthe user prefers\b'), 'you prefer'),
    (re.compile(r'\bthe user likes\b'), 'you like'),
    (re.compile(r'\bthe user wants\b'), 'you want'),
    (re.compile(r'\bthe user is\b'), 'you are'),
    (re.compile(r'\bthe user\b'), 'you'),
    # "User + verb" patterns (before generic "User")
    (re.compile(r'\bUser prefers\b'), 'You prefer'),
    (re.compile(r'\bUser likes\b'), 'You like'),
    (re.compile(r'\bUser wants\b'), 'You want'),
    (re.compile(r'\bUser reports\b'), 'You report'),
    (re.compile(r'\bUser is\b'), 'You are'),
    (re.compile(r'\bUser\b'), 'You'),
    (re.compile(r'\buser\b'), 'you'),
    # Third-person pronouns
    (re.compile(r'\bThey prefer\b'), 'You prefer'),
    (re.compile(r'\bthey prefer\b'), 'you prefer'),
]


def _convert_perspective(text: str) -> str:
    """Convert third-person hidden state descriptions to second person."""
    for pattern, replacement in _PERSPECTIVE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def build_hidden_state_prompt_section(
    hidden_state: Optional[List[HiddenStateItem]],
) -> str:
    """
    Build the hidden state prompt section for injection into every LLM call.
    
    This is the core mechanism for persistent persona — it produces a text
    block that should be prepended to every user-simulator prompt.
    
    Args:
        hidden_state: List of HiddenStateItem from scenario.state_queue
        
    Returns:
        Formatted prompt section string (empty if no hidden state)
    """
    if not hidden_state:
        return ""
    
    # Partition by category
    persona_items: List[HiddenStateItem] = []
    latent_items: List[HiddenStateItem] = []
    knowledge_items: List[HiddenStateItem] = []
    
    for h in hidden_state:
        if h.category.is_persona:
            persona_items.append(h)
        elif h.category.is_preference:
            # Only include items not yet disclosed
            if not h.disclosed_during_conversation:
                latent_items.append(h)
        elif h.category.is_knowledge:
            knowledge_items.append(h)
    
    # Build persona section
    persona_section = ""
    if persona_items:
        lines = []
        for item in persona_items:
            lines.append(f"- {_convert_perspective(item.value)}")
            if item.overlap_mode == OverlapMode.ALSO_DISCLOSED:
                lines.append(
                    "  (You will naturally mention this once in conversation, "
                    "but more importantly, show it in HOW you talk every turn)"
                )
            if item.style_enforcement:
                lines.append(f"  ⚠️ RULE: {item.style_enforcement}")
        persona_section = PERSONA_SECTION_TEMPLATE.format(items="\n".join(lines))
    
    # Build latent section
    latent_section = ""
    if latent_items:
        lines = []
        for item in latent_items:
            lines.append(f"- {_convert_perspective(item.value)}")
            if item.trigger_condition:
                lines.append(f"  → Only reveal if: {item.trigger_condition}")
        latent_section = LATENT_SECTION_TEMPLATE.format(items="\n".join(lines))
    
    # Build knowledge section
    knowledge_section = ""
    if knowledge_items:
        lines = []
        for item in knowledge_items:
            lines.append(f"- {_convert_perspective(item.value)}")
            if item.behavioral_effect:
                lines.append(f"  → Your reaction: {item.behavioral_effect}")
        knowledge_section = KNOWLEDGE_SECTION_TEMPLATE.format(items="\n".join(lines))
    
    # Only produce output if at least one section is populated
    if not (persona_section or latent_section or knowledge_section):
        return ""
    
    return HIDDEN_STATE_TEMPLATE.format(
        persona_section=persona_section.strip(),
        latent_section=latent_section.strip(),
        knowledge_section=knowledge_section.strip(),
    ).strip()


# =============================================================================
# SCENARIO TEMPLATE
# =============================================================================

SCENARIO_TEMPLATE = """
<scenario>
## Domain
{domain}

## Your Personality
{personality}
{hidden_state_section}

## Your Situation (State Queue)

{anchor_section}

### Information You Can Reveal (flow_info)
Reveal these progressively as the conversation develops (share like you're chatting with a friend):
{flow_info_formatted}

### Your Requirements (Constraints)
Mention each requirement ONCE when relevant - phrase as life circumstances, not technical requirements:
{constraints_formatted}

### Side Notes / Tangential Thoughts
Mention these naturally during the conversation as brief asides:
{distractor_formatted}
{user_intention_section}
</scenario>
"""

# =============================================================================
# ANCHOR STRATEGY TEMPLATES
# =============================================================================

ANCHOR_DIRECT_TEMPLATE = """
### Your Opening Statement (Turn 0)
{initial_statement}

This is your main concern from the start. Share it naturally, like you're telling a friend about your situation.
The challenge isn't in hiding your need - it's in the details that follow.
"""

ANCHOR_DISGUISED_TEMPLATE = """
### Your Conversation Strategy: LAYERED REVEAL

**Phase 1: Initial Focus (Turns 0 to {reveal_turn_minus_one})**
Your opening statement: "{initial_statement}"

This is what you start talking about. It's genuinely on your mind, so engage with it naturally.
The conversation might go in this direction for a bit - that's fine.

**Phase 2: Your Real Concern (Turn {reveal_turn} or earlier if triggered)**
What you actually need help with: "{true_anchor}"

This was in the back of your mind all along. It connects to the initial topic.

**How to Transition:**
- If the Assistant mentions any of these words: {reveal_trigger_keywords}
  → That's your cue! Naturally bring up your real concern.
- If you reach Turn {reveal_turn} without a trigger:
  → Transition naturally using phrases like:
    - "Oh, that actually reminds me of the main thing I'm stressed about..."
    - "Wait, so if that's true, then what about..."
    - "Hmm, that makes me think - the real issue is..."
    - "Actually, the reason I'm even thinking about this is..."

**Why This Feels Natural:**
{transition_rationale}

**IMPORTANT:** Don't force the transition. Let it flow naturally from the conversation.
"""

# =============================================================================
# USER INTENTION TEMPLATES
# =============================================================================

USER_INTENTION_CONFLICTING_TEMPLATE = """
## Something Else You'd Like To Do

Besides your main goal, there's something else you'd really like to do:

**What you want:** {internal_desire}
**Why:** {conflict_reason}

### How You Feel About This:
- You genuinely want BOTH: to achieve your main goal AND to do this.
- You don't realize (or haven't calculated) whether doing both is actually possible.
- You're hopeful and optimistic that somehow it can work out.
- You might naturally bring this up and ask if it's possible to fit it in.
- If the Assistant says it won't work, you might:
  - Ask "Are you sure?" or "What if I'm really quick?"
  - Try to negotiate or find a workaround
  - Express disappointment but eventually accept if the reasoning is solid
- If the Assistant easily agrees without checking, you'll happily go along with it.

**Your Mindset:** You're not trying to cause problems - you simply want both things and hope they can work together. You trust the Assistant to help figure out if it's feasible.
"""

USER_INTENTION_ALIGNED_TEMPLATE = """
## Your Intention
You genuinely want to get the correct answer to your question. You have no hidden agenda - 
simply seek accurate information or help with your stated need.
"""

USER_INTENTION_UNCERTAIN_TEMPLATE = """
## Your Intention
You genuinely don't know the answer to your question. You're asking because you need guidance.
Express genuine curiosity and appreciate clear explanations.
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_list_items(items: list, prefix: str = "- ") -> str:
    """Format a list of items with prefix"""
    if not items:
        return "(None)"
    return "\n".join(f"{prefix}{item}" for item in items)


def format_numbered_items(items: list) -> str:
    """Format a list of items with numbering"""
    if not items:
        return "(None)"
    return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))


def get_distractor_values(sq) -> list:
    """Get distractors as a list of strings"""
    distractor = sq.distractor
    
    if distractor is None:
        return []
    
    if isinstance(distractor, str):
        return [distractor] if distractor.strip() else []
    
    if isinstance(distractor, list):
        result = []
        for item in distractor:
            if hasattr(item, 'value'):
                result.append(str(item.value))
            else:
                result.append(str(item))
        return result
    
    return [str(distractor)]


def get_user_system_prompt(scenario: Scenario, max_info_per_turn: int = DEFAULT_MAX_INFO_PER_TURN) -> str:
    """
    Generate the complete system prompt for a user simulator.
    
    Combines the universal engine prompt with scenario-specific information,
    including user intention and anchor strategy if available.
    
    v4.0: Injects hidden_state section for persistent persona.
    
    Args:
        scenario: The scenario to simulate
        max_info_per_turn: Maximum number of info pieces to reveal per turn (default: 1)
        
    Returns:
        Complete system prompt string
    """
    sq = scenario.state_queue
    
    # Use the base prompt directly
    base_prompt = USER_SIMULATOR_SYSTEM_PROMPT
    
    # Format flow_info with numbering (overlap-aware: use explicit values)
    if hasattr(sq, 'get_explicit_flow_info_values'):
        flow_info_values = sq.get_explicit_flow_info_values()
    elif hasattr(sq, 'get_flow_info_values'):
        flow_info_values = sq.get_flow_info_values()
    else:
        flow_info_values = sq.flow_info
    flow_info_formatted = format_numbered_items(flow_info_values) if flow_info_values else "(No additional information to reveal)"
    
    # Format constraints with numbering
    constraint_values = sq.get_constraint_values() if hasattr(sq, 'get_constraint_values') else sq.constraints
    constraints_formatted = format_numbered_items(constraint_values) if constraint_values else "(No requirements)"
    
    # Format distractors with numbering
    distractor_values = get_distractor_values(sq)
    distractor_formatted = format_numbered_items(distractor_values) if distractor_values else "(No side notes)"
    
    # Format user intention section
    user_intention_section = _format_user_intention(scenario)
    
    # Format anchor section based on anchor_config
    anchor_section = _format_anchor_section(sq)
    
    # v4.0: Build hidden state section
    hidden_state_section = ""
    if hasattr(sq, 'hidden_state') and sq.hidden_state:
        hidden_state_section = "\n" + build_hidden_state_prompt_section(sq.hidden_state)
    
    # Build scenario section
    scenario_section = SCENARIO_TEMPLATE.format(
        domain=scenario.domain,
        personality=scenario.get_personality_description(),
        hidden_state_section=hidden_state_section,
        anchor_section=anchor_section,
        flow_info_formatted=flow_info_formatted,
        constraints_formatted=constraints_formatted,
        distractor_formatted=distractor_formatted,
        user_intention_section=user_intention_section,
    )
    
    # Combine with main prompt
    full_prompt = base_prompt + scenario_section
    
    return full_prompt.strip()


def _format_anchor_section(sq: StateQueue) -> str:
    """
    Format the anchor section based on anchor_config.
    
    Handles both:
    - Legacy format (simple anchor string)
    - New format (AnchorConfig with disguised anchor support)
    
    Args:
        sq: The StateQueue containing anchor configuration
        
    Returns:
        Formatted anchor section string
    """
    # Check if we have the new anchor_config
    anchor_config = getattr(sq, 'anchor_config', None)
    
    if anchor_config is None:
        # Legacy format - simple anchor
        anchor = sq.get_anchor() if hasattr(sq, 'get_anchor') else sq.anchor
        return ANCHOR_DIRECT_TEMPLATE.format(
            initial_statement=anchor or "(No opening statement provided)"
        )
    
    # New format with AnchorConfig
    if anchor_config.is_disguised and anchor_config.true_anchor:
        # Disguised anchor mode
        reveal_turn = anchor_config.reveal_turn or 3
        trigger_keywords = anchor_config.reveal_trigger_keywords or []
        
        return ANCHOR_DISGUISED_TEMPLATE.format(
            reveal_turn_minus_one=reveal_turn - 1,
            initial_statement=anchor_config.initial_statement,
            true_anchor=anchor_config.true_anchor,
            reveal_turn=reveal_turn,
            reveal_trigger_keywords=", ".join(trigger_keywords) if trigger_keywords else "(no specific triggers)",
            transition_rationale=anchor_config.disguise_rationale or "The topics are naturally connected"
        )
    else:
        # Direct mode (but still vague/life-contextualized)
        return ANCHOR_DIRECT_TEMPLATE.format(
            initial_statement=anchor_config.initial_statement
        )


def _format_user_intention(scenario: Scenario) -> str:
    """
    Format the user intention section based on intention type.
    
    Args:
        scenario: The scenario with optional user_intention_result
        
    Returns:
        Formatted user intention section string
    """
    uir = scenario.user_intention_result
    
    if uir is None:
        return ""  # No user intention specified
    
    if uir.intention_type == UserIntentionType.CONFLICTING:
        return USER_INTENTION_CONFLICTING_TEMPLATE.format(
            internal_desire=uir.internal_desire or "Something that conflicts with your constraints",
            conflict_reason=uir.conflict_reason or "You have a strong personal preference",
        )
    elif uir.intention_type == UserIntentionType.ALIGNED:
        return USER_INTENTION_ALIGNED_TEMPLATE
    elif uir.intention_type == UserIntentionType.UNCERTAIN:
        return USER_INTENTION_UNCERTAIN_TEMPLATE
    else:
        return ""


def get_turn_guidance(
    turn_number: int, 
    total_info_count: int, 
    last_agent_message: str = "",
    strict_single_info: bool = False,
) -> str:
    """
    Get turn-specific guidance for the user simulator.
    
    Args:
        turn_number: Current turn number (0-indexed)
        total_info_count: Total number of all info items
        last_agent_message: The last message from the agent (to analyze questions)
        strict_single_info: If True, always reveal exactly ONE piece of info per turn,
                           regardless of how many questions the agent asks.
        
    Returns:
        Turn-specific guidance string
    """
    if turn_number == 0:
        return "[Turn 0: State your anchor/opening statement clearly.]"
    
    # Build guidance based on strict_single_info mode
    guidance_parts = []
    
    if strict_single_info:
        # Strict mode: always exactly ONE info per turn
        guidance_parts.append(
            "[STRICT MODE: Reveal EXACTLY ONE piece of information this turn. "
            "Even if the Assistant asks multiple questions, only answer ONE of them. "
            "Choose the most relevant question to answer, and say something like "
            "'Let me focus on one thing at a time...' or 'Actually, first let me tell you about...']"
        )
    else:
        # Default mode: can reveal multiple info if agent asks multiple questions
        question_count = _count_questions_in_message(last_agent_message)
        
        if question_count == 0:
            guidance_parts.append(
                "[The Assistant didn't ask a specific question. "
                "You may proactively share ONE piece of relevant information to move the conversation forward.]"
            )
        elif question_count == 1:
            guidance_parts.append(
                "[The Assistant asked ONE question. "
                "Reveal exactly ONE piece of information to answer it.]"
            )
        else:
            guidance_parts.append(
                f"[The Assistant asked {question_count} questions. "
                f"You may reveal up to {question_count} pieces of information to answer them.]"
            )
    
    # Add end-of-conversation hint if most info revealed
    if turn_number > total_info_count:
        guidance_parts.append(
            "[Later turn: You should have revealed most information. Consider wrapping up if satisfied.]"
        )
    
    return "\n".join(guidance_parts)


def _count_questions_in_message(message: str) -> int:
    """
    Count the number of explicit questions in a message.
    
    Uses multiple heuristics:
    1. Question marks
    2. Question words at the start of sentences
    
    Args:
        message: The message to analyze
        
    Returns:
        Estimated number of questions
    """
    if not message:
        return 0
    
    # Count question marks as primary indicator
    question_mark_count = message.count("?")
    
    # If no question marks, check for implicit questions
    if question_mark_count == 0:
        # Check for question-like phrases without question marks
        question_starters = [
            "tell me", "let me know", "please share", "could you",
            "would you", "can you", "what about", "how about",
        ]
        implicit_count = 0
        message_lower = message.lower()
        for starter in question_starters:
            if starter in message_lower:
                implicit_count += 1
        return min(implicit_count, 1)  # At most 1 for implicit questions
    
    return question_mark_count
