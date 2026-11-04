"""
User Simulator

LLM-based user simulator that follows the universal simulation policy.
"""

from typing import List, Tuple, Optional
from benchmark.user.first_pass import call_stage
from loguru import logger

from benchmark.data_model.scenario import Scenario, AnchorConfig, AnchorDisguiseStrategy
from benchmark.data_model.message import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    AnyMessage,
    MessageRole,
)
from benchmark.user.base import BaseUser, UserState, UserContext
from benchmark.user.state_tracker import StateTracker
from benchmark.user.prompts import get_user_system_prompt, get_turn_guidance, DEFAULT_MAX_INFO_PER_TURN
from benchmark.utils.llm_utils import generate_chat_completion


# Special tokens/phrases for detecting user intent
SATISFACTION_INDICATORS = [
    "that's all", "thank you", "thanks", "perfect", "great",
    "sounds good", "that works", "i'm satisfied", "awesome",
    "that's perfect", "that helps", "all set", "good to go",
]

TRANSFER_INDICATORS = [
    "speak to a human", "human agent", "real person", "transfer me",
    "talk to someone", "real representative", "live agent",
]

OUT_OF_SCOPE_INDICATORS = [
    "never mind", "forget it", "can't help", "no use",
    "don't need it", "not helpful", "give up",
]


class UserSimulator(BaseUser):
    """
    LLM-powered user simulator.
    
    This simulator uses the universal policy prompt to drive natural,
    scenario-based conversations. It tracks state internally and
    generates responses using an LLM.
    
    Supports disguised anchor mode where the true anchor is revealed
    after a delay or when triggered by specific keywords.
    """
    
    def __init__(
        self,
        scenario: Optional[Scenario] = None,
        llm: str = "gpt-4o-mini",
        llm_args: Optional[dict] = None,
        enable_state_tracking: bool = True,
        temperature: float = 0.7,
        strict_single_info_per_turn: bool = False,
    ):
        """
        Initialize the user simulator.
        
        Args:
            scenario: The scenario to simulate
            llm: LLM model name
            llm_args: Additional LLM arguments
            enable_state_tracking: Whether to use state tracker
            temperature: Temperature for generation
            strict_single_info_per_turn: If True, always reveal exactly ONE piece of info per turn
                                         regardless of how many questions the agent asks.
                                         If False (default), can reveal multiple info if agent 
                                         asks multiple questions.
        """
        super().__init__(scenario=scenario, llm=llm, llm_args=llm_args)
        
        self.temperature = temperature
        self.enable_state_tracking = enable_state_tracking
        self.strict_single_info_per_turn = strict_single_info_per_turn
        
        # Initialize state tracker if scenario provided
        self.state_tracker: Optional[StateTracker] = None
        if scenario:
            self.state_tracker = StateTracker(scenario)
        
        # Cache system prompt
        self._system_prompt: Optional[str] = None
        
        # Anchor reveal state (for disguised anchors)
        self._anchor_revealed: bool = False
    
    def set_scenario(self, scenario: Scenario) -> None:
        """Set the scenario and reinitialize"""
        super().set_scenario(scenario)
        self.state_tracker = StateTracker(scenario)
        self._system_prompt = None  # Clear cached prompt
        self._anchor_revealed = False  # Reset anchor reveal state
    
    def reset(self) -> None:
        """Reset for a new conversation"""
        super().reset()
        if self.state_tracker:
            self.state_tracker.reset()
        self._anchor_revealed = False  # Reset anchor reveal state
    
    @property
    def system_prompt(self) -> str:
        """Get the system prompt (cached)"""
        if self._system_prompt is None:
            if self.scenario is None:
                raise ValueError("No scenario set for user simulator")
            self._system_prompt = get_user_system_prompt(self.scenario)
        return self._system_prompt
    
    def get_opening_message(self) -> UserMessage:
        """
        Generate the opening message (Turn 0).
        
        For Turn 0, we use the anchor configuration to guide LLM generation:
        - The anchor/initial_statement serves as guidance for what to talk about
        - LLM generates a natural, life-contextualized opening message
        
        The anchor_config determines how the conversation starts and when
        the true anchor is revealed (for disguised mode).
        """
        if self.scenario is None:
            raise ValueError("No scenario set")
        
        sq = self.scenario.state_queue
        anchor_config = sq.anchor_config
        
        # Determine the anchor guidance and disguise status
        if anchor_config is not None:
            anchor_guidance = anchor_config.initial_statement or sq.anchor
            is_disguised = anchor_config.is_disguised
            
        else:
            # Legacy mode: use anchor directly
            anchor_guidance = sq.anchor
            is_disguised = False
        
        # Generate natural opening message using LLM
        opening_content = self._generate_natural_opening(anchor_guidance, is_disguised)
        
        if not opening_content.strip():
            raise ValueError("Empty opening response")
        if not is_disguised:
            self._anchor_revealed = True

        message = UserMessage(
            content=opening_content,
            turn_idx=0,
            metadata={
                "is_anchor": not is_disguised,
                "is_initial_statement": True,
                "anchor_disguised": is_disguised,
                "anchor_guidance": anchor_guidance,  # Store original for debugging
            }
        )
        
        self.context.increment_turn()
        if self.state_tracker:
            self.state_tracker.increment_turn()
        
        return message
    
    def _render_opening_messages(self, messages, *, anchor_guidance, is_disguised):
        """Instance extension point; the default preserves the existing request."""
        return messages

    def _generate_natural_opening(self, anchor_guidance: str, is_disguised: bool) -> str:
        """
        Use LLM to generate a natural, life-contextualized opening message.
        
        Args:
            anchor_guidance: The topic/concern to address (may be descriptive)
            is_disguised: Whether this is a disguised anchor scenario
            
        Returns:
            A natural, conversational opening message
        """
        personality = self.scenario.personality if self.scenario else "Friendly and cooperative"
        variant = getattr(self, "experiment_variant", "V0")
        domain = "current conversation" if variant != "V0" else (self.scenario.domain if self.scenario else "general")
        
        # Check if the anchor_guidance is descriptive (third-person) rather than natural dialogue
        # Descriptive patterns indicate it needs to be converted to natural speech
        descriptive_indicators = [
            "User ", "user ", "The user", "the user",
            "Person ", "person ",
            "Customer ", "customer ",
            "describes ", "mentions ", "explains ", "asks ", "wants ",
            "is looking for", "is trying to", "needs to",
        ]
        is_descriptive = any(ind in anchor_guidance for ind in descriptive_indicators)
        
        # Check if the anchor_guidance already looks like natural dialogue
        # (contains first-person pronouns at the start, or casual expressions)
        natural_start_indicators = [
            "I ", "I'm ", "I've ", "I'd ", "I'll ",
            "My ", "my ",
            "Hey", "Hi", "Hello", "So ", "Ugh", "Oh", "Hmm", "Well",
        ]
        # looks_natural = any(anchor_guidance.strip().startswith(ind) for ind in natural_start_indicators)
        
        # # If it's descriptive OR doesn't look natural, generate a natural version
        # if is_descriptive or not looks_natural:
        #     # Need to generate natural opening via LLM
        #     pass  # Fall through to LLM generation
        # elif len(anchor_guidance) > 30:
        #     # Already looks natural and has enough content
        #     return anchor_guidance
        
        prompt = f"""You are simulating a real person starting a conversation with a customer service / AI assistant.

CONTEXT:
- Domain: {domain}
- Personality: {personality}
- Topic/Concern: {anchor_guidance}
- Mode: {"DISGUISED - don't directly state the technical problem" if is_disguised else "DIRECT - you can mention your concern"}

YOUR TASK:
Generate a NATURAL, CASUAL opening message as if you're a real person reaching out for help.

CRITICAL RULES:
1. Sound like a REAL PERSON, not a robot or test case
2. Use casual, conversational language
3. {"Do NOT directly state the technical problem - describe your life situation instead" if is_disguised else "You can mention what you need help with, but keep it conversational"}
4. Include some personality (based on: {personality})
5. Keep it 1-3 sentences - natural opening, not a wall of text
6. Use first-person perspective ("I", "my", "I'm")

EXAMPLES of natural openings:
- "Hey, so I've been having this weird issue with my car lately..."
- "Hi there! I'm hoping you can help me figure something out..."
- "Ugh, okay so this is kind of embarrassing but..."
- "I'm not sure if this is something you can help with, but..."

Generate ONLY the opening message, nothing else:"""

        if variant in ("V2", "V3"):
            from types import SimpleNamespace
            from benchmark.user.turn_input import build_turn_input
            from benchmark.user.bridge_generator import _concise_turn_prompt, _concise_messages
            opening_aim = SimpleNamespace(description=anchor_guidance, is_final=False)
            turn_input = build_turn_input(self.scenario, opening_aim, [], [])
            prompt = _concise_turn_prompt(turn_input, opening_aim)
            if is_disguised:
                prompt += "\nDescribe only the given initial situation; do not diagnose its underlying problem."

        opening_messages = (_concise_messages(prompt) if variant in ("V2", "V3")
                            else [{"role": "user", "content": prompt}])
        opening_messages = self._render_opening_messages(
            opening_messages, anchor_guidance=anchor_guidance, is_disguised=is_disguised)

        # Retry logic for server errors
        max_retries = 3
        base_delay = 30
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                response = call_stage(getattr(self, "trace", None), "OPENING", generate_chat_completion,
                    model=self.llm,
                    messages=opening_messages,
                    temperature=self.temperature,
                    max_tokens=16384,
                    **(self.llm_args or {}),
                )
                
                # Clean up the response
                opening = response.strip()
                
                # Remove any quotes if the LLM wrapped the response
                if opening.startswith('"') and opening.endswith('"'):
                    opening = opening[1:-1]
                if opening.startswith("'") and opening.endswith("'"):
                    opening = opening[1:-1]
                    
                return opening
                
            except Exception as e:
                if getattr(self, "trace", None) is not None:
                    raise
                last_error = e
                error_str = str(e).lower()
                
                # Check for retryable server errors
                retryable_indicators = [
                    "500", "502", "503", "504",
                    "internal server error", "internal error",
                    "service unavailable", "overload", 
                    "gateway timeout", "please retry",
                ]
                is_retryable = any(ind in error_str for ind in retryable_indicators)
                
                if is_retryable and attempt < max_retries:
                    import time
                    import random
                    delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay / 2)
                    logger.warning(
                        f"Failed to generate natural opening (attempt {attempt + 1}/{max_retries}): {e}\n"
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(f"Failed to generate natural opening: {e}, using anchor directly")
                    # Fallback to the anchor guidance if LLM fails
                    return anchor_guidance
        
        # Should not reach here, but just in case
        logger.warning(f"Max retries exceeded for natural opening: {last_error}, using anchor directly")
        return anchor_guidance
    
    def step(
        self,
        messages: List[AnyMessage],
    ) -> Tuple[UserMessage, UserState]:
        """
        Generate the next user message.
        
        For disguised anchor scenarios, this method also checks if it's time
        to reveal the true anchor based on:
        - Turn number (reveal_turn)
        - Trigger keywords in agent's response (reveal_trigger_keywords)
        
        Args:
            messages: Conversation history (should end with assistant message)
            
        Returns:
            Tuple of (UserMessage, UserState)
        """
        if self.scenario is None:
            raise ValueError("No scenario set")
        
        # Check for anchor reveal trigger before generating response
        should_reveal = self._check_anchor_reveal_trigger(messages)
        
        # Increment turn counter
        self.context.increment_turn()
        if self.state_tracker:
            self.state_tracker.increment_turn()
        
        # Build messages for LLM, with reveal hint if needed
        llm_messages = self._build_llm_messages(messages, should_reveal_anchor=should_reveal)
        
        # Generate response
        response_content = self._generate_response(llm_messages)
        
        # Detect user state from response
        new_state = self._detect_state(response_content)
        self.context.state = new_state
        
        # Create user message with metadata
        user_message = UserMessage(
            content=response_content,
            turn_idx=self.context.current_turn,
            metadata={
                "anchor_revealed_this_turn": should_reveal and not self._anchor_revealed,
            }
        )
        
        # Mark anchor as revealed if this turn triggers it
        if should_reveal:
            self._anchor_revealed = True
        
        # Update tracking (heuristic-based for now)
        self._update_tracking(response_content)
        
        logger.debug(f"User turn {self.context.current_turn}: {response_content[:100]}...")
        
        return user_message, new_state
    
    def _flip_message_role(self, msg: AnyMessage) -> Optional[dict]:
        """
        Flip message role for user simulation.
        
        Since we're simulating the USER, we need to flip the conversation:
        - Agent's messages (assistant) → become "user" (what the LLM sees as input)
        - User's messages → become "assistant" (what the LLM "previously said")
        - System messages → SKIP (Agent's system prompt should not be included)
        
        This way, the LLM continues as "assistant" but actually generates user responses.
        
        Returns:
            Flipped message dict, or None if the message should be skipped
        """
        if isinstance(msg, SystemMessage):
            # SKIP Agent's system messages - User Simulator has its own system prompt
            # This is important: we don't want Agent's instructions mixed with User's
            logger.debug(f"Skipping system message in history (Agent's system prompt)")
            return None
        elif isinstance(msg, AssistantMessage):
            # Agent's message → becomes user input to the LLM
            return {
                "role": "user",
                "content": msg.content or ""
            }
        elif isinstance(msg, UserMessage):
            # User's message → becomes what the LLM "said" before
            return {
                "role": "assistant", 
                "content": msg.content or ""
            }
        else:
            # Unknown message type - log warning and skip
            logger.warning(f"Unknown message type in history: {type(msg)}, skipping")
            return None
    
    def _build_llm_messages(self, history: List[AnyMessage], should_reveal_anchor: bool = False) -> List[dict]:
        """
        Build message list for LLM API call with role flipping.
        
        Following tau2-bench pattern:
        - User Simulator's system prompt (NOT Agent's) is added first
        - Conversation history has roles flipped:
          - Original assistant messages → user (the "input" to respond to)
          - Original user messages → assistant (what the LLM "previously said")
          - Original system messages → SKIPPED (Agent's system prompt is not relevant)
        - LLM generates as "assistant", which we then use as user response
        
        Important: Agent's system prompt and User Simulator's system prompt are DIFFERENT!
        - Agent's system prompt: Instructions for the agent to help users
        - User's system prompt: Instructions for simulating user behavior with scenario
        
        Args:
            history: Conversation history
            should_reveal_anchor: If True, add hint to reveal the true anchor this turn
        """
        llm_messages = []
        
        # USER SIMULATOR's system message with scenario (NOT Agent's!)
        llm_messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Extract the last agent message for question analysis
        last_agent_message = ""
        for msg in reversed(history):
            if isinstance(msg, AssistantMessage):
                last_agent_message = msg.content or ""
                break
        
        # Add anchor reveal guidance if it's time to reveal
        if should_reveal_anchor and not self._anchor_revealed:
            anchor_config = self.scenario.state_queue.anchor_config
            if anchor_config and anchor_config.is_disguised:
                reveal_hint = self._build_anchor_reveal_hint(anchor_config, last_agent_message)
                llm_messages.append({
                    "role": "system",
                    "content": reveal_hint
                })
        
        # Add turn guidance as system hint
        if self.state_tracker:
            # Get total info count
            flow_info_count = len(self.scenario.state_queue.flow_info)
            constraint_count = len(self.scenario.state_queue.constraints) if self.scenario.state_queue.constraints else 0
            distractor_count = len(self.scenario.state_queue.get_distractor_values())
            total_info_count = flow_info_count + constraint_count + distractor_count
            
            turn_guidance = get_turn_guidance(
                turn_number=self.context.current_turn,
                total_info_count=total_info_count,
                last_agent_message=last_agent_message,
                strict_single_info=self.strict_single_info_per_turn,
            )
            if turn_guidance:
                tracking_context = self.state_tracker.get_tracking_prompt_context()
                llm_messages.append({
                    "role": "system",
                    "content": f"{turn_guidance}\n{tracking_context}"
                })
        
        # Add conversation history WITH ROLE FLIPPING
        # Skip any system messages (Agent's system prompt should not be included)
        for msg in history:
            flipped = self._flip_message_role(msg)
            if flipped is not None:
                llm_messages.append(flipped)
        
        return llm_messages
    
    def _generate_response(self, messages: List[dict]) -> str:
        """Generate response using LLM"""
        try:
            response = generate_chat_completion(
                model=self.llm,
                messages=messages,
                temperature=self.temperature,
                **self.llm_args
            )
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback response
            return "Sorry, I didn't catch that. Could you repeat?"
    
    def _check_anchor_reveal_trigger(self, messages: List[AnyMessage]) -> bool:
        """
        Check if it's time to reveal the true anchor (for disguised anchor mode).
        
        The anchor should be revealed when:
        1. The current turn number >= reveal_turn (configured in anchor_config), OR
        2. The agent's response contains any of the reveal_trigger_keywords
        
        Args:
            messages: Conversation history
            
        Returns:
            True if the true anchor should be revealed this turn
        """
        if self._anchor_revealed:
            return False  # Already revealed
        
        if self.scenario is None:
            return False
        
        anchor_config = self.scenario.state_queue.anchor_config
        if anchor_config is None or not anchor_config.is_disguised:
            return False  # Not a disguised anchor scenario
        
        # Check turn-based reveal
        current_turn = self.context.current_turn + 1  # After increment
        reveal_turn = anchor_config.reveal_turn
        if reveal_turn is not None and current_turn >= reveal_turn:
            logger.debug(f"Anchor reveal triggered by turn number: {current_turn} >= {reveal_turn}")
            return True
        
        # Check keyword-based trigger
        trigger_keywords = anchor_config.reveal_trigger_keywords or []
        if trigger_keywords:
            # Get the last agent message
            for msg in reversed(messages):
                if isinstance(msg, AssistantMessage) and msg.content:
                    msg_lower = msg.content.lower()
                    for keyword in trigger_keywords:
                        if keyword.lower() in msg_lower:
                            logger.debug(f"Anchor reveal triggered by keyword: '{keyword}'")
                            return True
                    break  # Only check the last agent message
        
        return False
    
    def _build_anchor_reveal_hint(self, anchor_config: AnchorConfig, last_agent_message: str) -> str:
        """
        Build a hint for the LLM to naturally transition to the true anchor.
        
        This guides the user simulator to reveal the real request in a natural way,
        as if they just remembered or realized what they actually need.
        
        Args:
            anchor_config: The anchor configuration
            last_agent_message: The agent's last response
            
        Returns:
            A system hint for natural anchor revelation
        """
        true_anchor = anchor_config.true_anchor
        strategy = anchor_config.disguise_strategy
        
        # Map strategy to transition phrase
        # Strategies: direct, delayed_reveal, context_switch, distractor_first, embedded_problem
        if strategy == AnchorDisguiseStrategy.DELAYED_REVEAL:
            transition = "Now naturally transition to mentioning your actual need"
        elif strategy == AnchorDisguiseStrategy.CONTEXT_SWITCH:
            transition = "Now pivot from the current topic to what you actually need"
        elif strategy == AnchorDisguiseStrategy.DISTRACTOR_FIRST:
            transition = "Now move past the distractor topic and bring up your real concern"
        elif strategy == AnchorDisguiseStrategy.EMBEDDED_PROBLEM:
            transition = "Now let the real technical problem emerge from your life situation"
        else:  # DIRECT or default
            transition = "Now, as if you just remembered, bring up what you actually need"
        
        hint = f"""
=== IMPORTANT: ANCHOR REVEAL ===
{transition}. Your TRUE request is: "{true_anchor}"

Reveal this naturally in your next response. Don't make it feel forced.
Examples of natural transitions:
- "Oh, that reminds me, what I actually wanted to ask about was..."
- "Thanks for that! But actually, my main reason for calling was..."
- "Speaking of which, I also need help with something else..."
- "By the way, while I have you on the line, I was wondering..."
"""
        return hint
    
    def _detect_state(self, response: str) -> UserState:
        """
        Detect user state from response content.
        
        Args:
            response: The generated response
            
        Returns:
            Detected UserState
        """
        response_lower = response.lower()
        
        # Check for transfer request
        for indicator in TRANSFER_INDICATORS:
            if indicator in response_lower:
                return UserState.TRANSFER
        
        # Check for satisfaction
        satisfaction_count = sum(
            1 for indicator in SATISFACTION_INDICATORS
            if indicator in response_lower
        )
        if satisfaction_count >= 2:  # Multiple satisfaction indicators
            return UserState.SATISFIED
        
        # Check for abandonment
        for indicator in OUT_OF_SCOPE_INDICATORS:
            if indicator in response_lower:
                return UserState.OUT_OF_SCOPE
        
        return UserState.ACTIVE
    
    def _update_tracking(self, response: str) -> None:
        """
        Update state tracking based on response.
        
        This uses heuristics to detect what information was revealed.
        Tracks three types: flow_info, constraints, and distractors.
        A more sophisticated version could use LLM-based detection.
        """
        if not self.state_tracker or not self.scenario:
            return
        
        response_lower = response.lower()
        response_words = set(response_lower.split())
        sq = self.scenario.state_queue
        
        # 1. Check if any flow_info was mentioned
        flow_info_values = sq.get_flow_info_values() if hasattr(sq, 'get_flow_info_values') else sq.flow_info
        
        for i, info in enumerate(flow_info_values):
            if i in self.state_tracker.revealed_flow_info_indices:
                continue
            
            # Simple heuristic: check for key phrase overlap
            info_str = str(info)
            info_keywords = set(info_str.lower().split())
            
            # If significant overlap, consider it revealed
            overlap = info_keywords & response_words
            if len(overlap) >= min(3, max(1, len(info_keywords) // 2)):
                self.state_tracker.mark_info_revealed(i, trigger="detected")
        
        # 2. Check if any constraint was mentioned
        constraint_values = sq.get_constraint_values() if hasattr(sq, 'get_constraint_values') else []
        if hasattr(sq, 'constraint') and sq.constraint and not constraint_values:
            constraint_values = [sq.constraint]
        
        for i, constraint in enumerate(constraint_values):
            if i in self.state_tracker.revealed_constraint_indices:
                continue
            
            constraint_str = str(constraint)
            constraint_keywords = set(constraint_str.lower().split())
            
            overlap = constraint_keywords & response_words
            if len(overlap) >= min(3, max(1, len(constraint_keywords) // 2)):
                self.state_tracker.mark_constraint_revealed(i, trigger="detected")
        
        # 3. Check if any distractor (tangential thought) was mentioned
        distractor_values = sq.get_distractor_values() if hasattr(sq, 'get_distractor_values') else []
        if hasattr(sq, 'distractor') and sq.distractor and not distractor_values:
            # Legacy single distractor format
            distractor_values = [sq.distractor]
        
        for i, distractor in enumerate(distractor_values):
            if i in self.state_tracker.revealed_distractor_indices:
                continue
            
            distractor_str = str(distractor)
            distractor_keywords = set(distractor_str.lower().split())
            
            overlap = distractor_keywords & response_words
            if len(overlap) >= min(3, max(1, len(distractor_keywords) // 2)):
                self.state_tracker.mark_distractor_revealed(i, trigger="detected")
    
    def get_tracking_summary(self) -> dict:
        """Get summary of tracking state"""
        if self.state_tracker:
            return self.state_tracker.get_state_summary()
        return {"tracking_disabled": True}


class UserSimulatorWithTools(UserSimulator):
    """
    Extended user simulator that can use tools.
    
    Some scenarios may require the user to look up information
    (e.g., check their order number, verify their account).
    """
    
    def __init__(
        self,
        tools: Optional[List[dict]] = None,
        **kwargs
    ):
        """
        Initialize with tool capability.
        
        Args:
            tools: List of tool definitions (OpenAI format)
            **kwargs: Arguments passed to UserSimulator
        """
        super().__init__(**kwargs)
        self.tools = tools or []
    
    def step(
        self,
        messages: List[AnyMessage],
    ) -> Tuple[UserMessage, UserState]:
        """
        Generate next message, potentially using tools.
        
        For now, this is a placeholder. Full implementation would:
        1. Check if tool use is needed
        2. Execute tools if needed
        3. Incorporate tool results into response
        """
        # For now, delegate to parent
        return super().step(messages)
