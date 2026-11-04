"""Dialogue loop used by the released simulator; no dataset-generation pipeline."""
from __future__ import annotations
import time
from loguru import logger
from benchmark.data_model.scenario import Scenario
from benchmark.data_model.message import Conversation, UserMessage, AssistantMessage
from benchmark.data_model.simulation import SimulationResult, TerminationReason
from benchmark.agent.base import BaseAgent
from benchmark.user.user_simulator import UserSimulator

def _is_retryable_server_error(error: Exception) -> bool:
    """
    Check if the error is a retryable server-side error.
    
    Covers:
    - 500 Internal Server Error
    - 502 Bad Gateway
    - 503 Service Unavailable / Overloaded
    - 504 Gateway Timeout
    - Generic internal/temporary errors
    """
    error_str = str(error).lower()
    
    retryable_indicators = [
        # HTTP status codes
        "500", "502", "503", "504",
        # Error messages
        "internal server error", "internal error", "internalservererror",
        "bad gateway", "badgateway",
        "service unavailable", "service_unavailable", "overload", "overloaded",
        "gateway timeout", "gateway_timeout",
        "temporarily unavailable", "temporary error",
        "please retry", "try again",
        # Provider-specific
        "geminiexception", "internalserverexception",
    ]
    
    for indicator in retryable_indicators:
        if indicator in error_str:
            return True
    return False

def _call_with_retry(
    func,
    *args,
    base_delay: int = 30,
    max_delay: int = 300,
    max_retries: int = 5,
    logger_instance=None,
    error_context: str = "",
    **kwargs
):
    """
    Call a function with exponential backoff retry logic for server errors.
    
    Handles retryable errors including:
    - 500 Internal Server Error
    - 502 Bad Gateway  
    - 503 Service Unavailable / Overloaded
    - 504 Gateway Timeout
    
    Uses exponential backoff with jitter for optimal retry timing.
    
    Args:
        func: The function to call
        *args: Arguments to pass to the function
        base_delay: Base delay in seconds (default: 30)
        max_delay: Maximum delay between retries (default: 300s = 5min)
        max_retries: Maximum number of retry attempts (default: 5)
        logger_instance: Logger instance for logging
        error_context: Context string for error messages (e.g., "agent response")
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call
        
    Raises:
        The original exception if max retries exceeded or non-retryable error
    """
    import random
    
    log = logger_instance or logger
    context_str = f" ({error_context})" if error_context else ""
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            is_retryable = _is_retryable_server_error(e)
            
            if is_retryable and attempt < max_retries:
                # Exponential backoff with jitter: delay = base * 2^attempt + random(0, base/2)
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, base_delay / 2)
                wait_time = delay + jitter
                
                log.warning(
                    f"🔄 Retryable error{context_str} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}\n"
                    f"   Error: {str(e)[:200]}...\n"
                    f"   Waiting {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)
                continue
            else:
                # Non-retryable error or max retries exceeded
                if is_retryable:
                    log.error(
                        f"❌ Max retries ({max_retries}) exceeded{context_str}. "
                        f"Last error: {type(e).__name__}: {str(e)[:200]}"
                    )
                raise

def run_single_scenario(*args, **kwargs):
    """Preserve actual identity on every return and raised execution error."""
    from benchmark.user.simulator_identity import simulator_state
    simulator = kwargs.get('user_simulator') or args[2]
    try:
        result = _run_single_scenario(*args, **kwargs)
        result.final_state = dict(result.final_state or {}, **simulator_state(simulator))
        return result
    except Exception as exc:
        exc.simulator_state = simulator_state(simulator)
        raise

def _run_single_scenario(
    scenario: Scenario,
    agent: BaseAgent,
    user_simulator: UserSimulator,
    max_turns: int = 10,
    min_turns_after_info_complete: int = 3,
    max_consecutive_empty: int = 2,
    max_repetition_turns: int = 3,
    verbose: bool = True,
    *,
    strict_calls: bool = False,
    call_budget=None,
) -> SimulationResult:
    """
    Run a single scenario and return the result.
    
    IMPORTANT: The Agent LLM only sees conversation history (user/assistant messages).
    It does NOT have access to:
    - User Simulator's system prompt (scenario description)
    - Scenario state_queue (flow_info, constraints, distractor)
    
    Termination conditions (in order of priority):
    1. ERROR: Agent or User Simulator throws an error
    2. INFO_COMPLETE_AND_SATISFIED: All flow_info revealed + user satisfied
    3. REPETITION: Same response repeated max_repetition_turns times
    4. EMPTY_RESPONSE: max_consecutive_empty empty responses in a row
    5. MAX_TURNS: Reached max_turns limit
    
    Args:
        scenario: The scenario to run
        agent: The agent being evaluated
        user_simulator: The user simulator
        max_turns: Maximum number of turns
        min_turns_after_info_complete: Minimum turns after all info is revealed
        max_consecutive_empty: Maximum consecutive empty responses before termination
        max_repetition_turns: Maximum times the same response can repeat
        verbose: Whether to print progress
    
    Returns:
        SimulationResult with conversation and metrics
    """
    from benchmark.user.base import UserState
    strict_calls = strict_calls or (getattr(user_simulator, 'release_profile', {}).get('revision') not in (None, 'legacy'))
    trace = getattr(user_simulator, 'trace', None)
    if call_budget is None and trace is not None:
        call_budget = trace.budget
    if call_budget is not None and strict_calls:
        call_budget.reject_invalid_responses = True

    def agent_reply(messages, stage_name):
        if not strict_calls:
            return _call_with_retry(agent.respond, messages, base_delay=10, max_delay=60,
                                    max_retries=3, logger_instance=logger,
                                    error_context=stage_name)
        stage = dict(name=stage_name, attempts=0, raw_response=None, error=None)
        if trace is not None:
            trace.stages.append(stage)
        response = agent.respond(messages, _first_pass_stage=stage,
                                 _first_pass_budget=call_budget,
                                 max_retries=0, num_retries=0)
        stage['raw_response'] = response.content
        stage['finish_reason'] = response.finish_reason
        if response.finish_reason in ('error', 'length', 'content_filter'):
            stage['error'] = 'invalid_agent_response:' + response.finish_reason
            raise ValueError(stage['error'])
        if not response.content or not response.content.strip():
            stage['error'] = 'empty_agent_response'
            raise ValueError(stage['error'])
        return response
    
    # Reset for new scenario
    agent.reset()
    user_simulator.reset()
    user_simulator.set_scenario(scenario)

    # Wire scenario.base_instruction to agent system prompt so thinking models
    # (gpt-5, o-series, gemini-3) get a non-empty `instructions` field.
    if getattr(scenario, "base_instruction", None):
        agent.system_prompt = scenario.base_instruction
    
    # Initialize conversation
    # NOTE: Only messages added to `conversation` are visible to the Agent
    # The Agent does NOT see User Simulator's system prompt or scenario details
    conversation = Conversation()
    
    # Check if this is an aim-driven scenario
    is_aim_driven = scenario.state_queue.is_aim_driven()
    
    # Get total info count for tracking completion
    total_info_count = len(scenario.state_queue.flow_info)
    
    # Get initial user message (anchor) - already returns UserMessage
    initial_message = user_simulator.get_opening_message()
    
    # Validate initial message is not empty
    if not initial_message.content or not initial_message.content.strip():
        if strict_calls:
            return SimulationResult(task_id=f'task_{scenario.scenario_id}',
                scenario_id=scenario.scenario_id, conversation=conversation,
                termination_reason=TerminationReason.AGENT_ERROR, turn_count=0,
                final_state={'execution_error': 'empty_user_opening',
                             'release_profile': getattr(user_simulator, 'release_profile', {})})
        logger.warning("Initial message is empty, using anchor from scenario")
        initial_message = UserMessage(content=scenario.state_queue.anchor)
    
    conversation.add_message(initial_message)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.scenario_id}")
        print(f"Domain: {scenario.domain}")
        if is_aim_driven:
            aim_count = len(scenario.state_queue.aim_sequence.aims)
            eval_aim = scenario.state_queue.aim_sequence.evaluation_aim_id
            print(f"🎯 Mode: AIM-DRIVEN ({aim_count} aims)")
            print(f"   Evaluation Aim: {eval_aim}")
        else:
            print(f"📋 Mode: INFO-DRIVEN")
            print(f"   Info items to reveal: {total_info_count}")
        print(f"{'='*60}")
        print(f"\n[Turn 0] User: {initial_message.content[:100]}...")
    
    # State tracking
    termination_reason = TerminationReason.MAX_TURNS
    turn_count = 0
    current_state = UserState.ACTIVE
    
    # Termination tracking
    consecutive_empty_count = 0
    turns_after_info_complete = 0
    info_complete_at_turn = None
    last_responses = []  # For repetition detection
    
    for turn in range(max_turns):
        turn_count = turn + 1
        
        # =====================================================================
        # Agent Response
        # NOTE: agent.respond() only receives conversation.messages
        # This list contains ONLY UserMessage and AssistantMessage objects
        # The Agent has NO access to User Simulator's system prompt or scenario
        # =====================================================================
        try:
            agent_response = agent_reply(conversation.messages, f'AGENT_{turn_count}')
            agent_content = agent_response.content or ""
        except Exception as e:
            logger.error(f"Agent error after retries: {e}")
            agent_content = ""
            termination_reason = TerminationReason.AGENT_ERROR
            break
        
        # Check for empty agent response
        if not agent_content.strip():
            consecutive_empty_count += 1
            logger.warning(f"Agent returned empty response (count: {consecutive_empty_count})")
            if consecutive_empty_count >= max_consecutive_empty:
                termination_reason = TerminationReason.AGENT_ERROR
                if verbose:
                    print(f"\n[Turn {turn_count}] Agent: [EMPTY - terminating due to consecutive empty responses]")
                break
            # Add placeholder to keep conversation going
            agent_content = "I apologize, could you please repeat that?"
        else:
            consecutive_empty_count = 0
        
        conversation.add_message(AssistantMessage(content=agent_content))
        
        if verbose:
            print(f"\n[Turn {turn_count}] Agent: {agent_content[:100]}...")
        
        # Check for repetition
        last_responses.append(agent_content.strip().lower()[:200])
        if len(last_responses) > max_repetition_turns:
            last_responses.pop(0)
        if len(last_responses) >= max_repetition_turns and len(set(last_responses)) == 1:
            termination_reason = TerminationReason.AGENT_ERROR  # Repetition error
            if verbose:
                print(f"\n[Terminating: Agent repeating same response {max_repetition_turns} times]")
            break
        
        # =====================================================================
        # User Response
        # =====================================================================
        try:
            user_message, current_state = _call_with_retry(
                user_simulator.step,
                conversation.messages,
                base_delay=10,
                max_delay=60,
                max_retries=0 if strict_calls else 3,
                logger_instance=logger,
                error_context=f"user simulator turn {turn_count}"
            )
            user_content = user_message.content or ""
        except Exception as e:
            import traceback
            logger.error(f"User simulator error (no fallback): {e}\n{traceback.format_exc()}")
            termination_reason = TerminationReason.AGENT_ERROR
            if verbose:
                print(f"\n[Turn {turn_count}] User simulator failed — terminating task")
            break
        
        # Check for empty user response. Discard this trajectory immediately;
        # do NOT inject a "Could you give me more details?" placeholder that
        # would pollute the conversation log with template-only content.
        if not user_content.strip():
            consecutive_empty_count += 1
            logger.warning(
                f"User returned empty response (count: {consecutive_empty_count}) — "
                "terminating scenario (no template fallback)"
            )
            termination_reason = TerminationReason.AGENT_ERROR
            if verbose:
                print(f"\n[Turn {turn_count}] User: [EMPTY — terminating scenario, no fallback]")
            break
        else:
            consecutive_empty_count = 0
        
        conversation.add_message(user_message)
        
        if verbose:
            print(f"\n[Turn {turn_count}] User: {user_content[:100]}...")
        
        # =====================================================================
        # Check Info Completion Status
        # =====================================================================
        tracking = user_simulator.get_tracking_summary() if hasattr(user_simulator, 'get_tracking_summary') else {}
        # Use 'revealed_indices' key from StateTracker.get_state_summary()
        revealed_count = tracking.get('revealed_count', 0) if isinstance(tracking, dict) else 0
        
        if revealed_count >= total_info_count and info_complete_at_turn is None:
            info_complete_at_turn = turn_count
            if verbose:
                print(f"\n[Info Complete at turn {turn_count}]")
        
        if info_complete_at_turn is not None:
            turns_after_info_complete = turn_count - info_complete_at_turn
        
        # =====================================================================
        # Termination Checks
        # =====================================================================
        
        # For aim-driven scenarios: check if all aims are completed
        if is_aim_driven:
            aim_progress = user_simulator.get_aim_progress() if hasattr(user_simulator, 'get_aim_progress') else None
            if aim_progress and aim_progress.get('is_complete', False):
                termination_reason = TerminationReason.SUCCESS
                if verbose:
                    print(f"\n[Terminating: All aims completed (aim-driven)]")
                break
        
        # Check for satisfaction state
        if current_state == UserState.SATISFIED:
            if is_aim_driven:
                # For aim-driven scenarios: terminate immediately when user is satisfied
                # (aims completion is already checked above, SATISFIED means conversation should end)
                termination_reason = TerminationReason.SUCCESS
                if verbose:
                    print(f"\n[Terminating: User satisfied (aim-driven)]")
                break
            else:
                # For info-driven scenarios: also check info completion
                if info_complete_at_turn is not None and turns_after_info_complete >= min_turns_after_info_complete:
                    termination_reason = TerminationReason.SUCCESS
                    if verbose:
                        print(f"\n[Terminating: User satisfied after all info revealed]")
                    break
                elif verbose:
                    # User says satisfied but info not complete - don't terminate yet
                    print(f"\n[User says satisfied but info not complete: {revealed_count}/{total_info_count}]")
        
        # Strong satisfaction signals (user explicitly ending conversation)
        # For aim-driven: always honor these signals
        # For info-driven: only honor if ALL flow_info has been revealed
        strong_end_signals = [
            "that's all i need", "that's all", "thank you, goodbye", 
            "thanks, bye", "i'm satisfied", "this is exactly what i needed",
            "that solves my problem", "you've been very helpful, bye"
        ]
        content_lower = user_content.lower()
        if not (strict_calls and is_aim_driven) and any(signal in content_lower for signal in strong_end_signals):
            if is_aim_driven:
                # For aim-driven scenarios: honor strong end signals immediately
                termination_reason = TerminationReason.SUCCESS
                if verbose:
                    print(f"\n[Terminating: User explicitly ended conversation (aim-driven)]")
                break
            else:
                # For info-driven: MUST check that info is actually complete (not None) AND waited enough turns
                if info_complete_at_turn is not None and turns_after_info_complete >= min_turns_after_info_complete:
                    termination_reason = TerminationReason.SUCCESS
                    if verbose:
                        print(f"\n[Terminating: User explicitly ended conversation after all info revealed]")
                    break
                elif verbose:
                    print(f"\n[Ignoring end signal - info not complete yet: {revealed_count}/{total_info_count}]")
    
    # =====================================================================
    # Final Agent Turn After Aim Completion
    # All aim-driven scenarios end with the user's final message (trap).
    # The agent never gets to respond to the trap, but the evaluator
    # checks the last assistant message. Without this fix, the evaluator
    # judges the pre-trap response against ground truth that expects a
    # response TO the trap — causing all scenarios to fail.
    # =====================================================================
    if termination_reason == TerminationReason.SUCCESS and is_aim_driven:
        last_msg = conversation.messages[-1] if conversation.messages else None
        if last_msg and isinstance(last_msg, UserMessage):
            if verbose:
                print(f"\n[Final agent turn: responding to last user message]")
            try:
                agent_response = agent_reply(conversation.messages, 'AGENT_FINAL')
                final_content = agent_response.content or ""
                if final_content.strip():
                    conversation.add_message(AssistantMessage(content=final_content))
                    turn_count += 1
                    if verbose:
                        print(f"\n[Turn {turn_count}] Agent (final): {final_content[:100]}...")
            except Exception as e:
                logger.warning(f"Final agent turn failed: {e}")
                if strict_calls:
                    termination_reason = TerminationReason.AGENT_ERROR

    if strict_calls and is_aim_driven and termination_reason == TerminationReason.SUCCESS:
        progress = user_simulator.get_aim_progress()
        if (not progress.get('is_complete') or not conversation.messages or
                not isinstance(conversation.messages[-1], AssistantMessage)):
            termination_reason = TerminationReason.AGENT_ERROR

    # Final summary
    if verbose:
        tracking = user_simulator.get_tracking_summary() if hasattr(user_simulator, 'get_tracking_summary') else {}
        revealed_count = tracking.get('revealed_count', 0) if isinstance(tracking, dict) else 0
        print(f"\n{'='*60}")
        print(f"Conversation ended: {termination_reason.value}")
        print(f"Total turns: {turn_count}")
        
        # Show aim-specific progress for aim-driven scenarios
        if is_aim_driven and hasattr(user_simulator, 'get_aim_progress'):
            aim_progress = user_simulator.get_aim_progress()
            if aim_progress:
                print(f"🎯 Aim Progress:")
                print(f"   Completed: {aim_progress.get('completed', 0)}/{aim_progress.get('total_aims', 0)}")
                current_aim_id = aim_progress.get('current_aim')
                if current_aim_id:
                    print(f"   Current Aim: {current_aim_id}")
                print(f"   Aims Status:")
                for aim_status in aim_progress.get('aims_status', []):
                    status_emoji = "✓" if aim_status['status'] == 'satisfied' else "○"
                    print(f"     {status_emoji} {aim_status['id']}: {aim_status['status']} ({aim_status['turns']} turns)")
        else:
            print(f"Info revealed: {revealed_count}/{total_info_count}")
        print(f"{'='*60}")
    
    # Get final tracking summary
    tracking = user_simulator.get_tracking_summary() if hasattr(user_simulator, 'get_tracking_summary') else {}
    if strict_calls:
        tracking = dict(tracking or {})
        tracking['release_profile'] = getattr(user_simulator, 'release_profile', {})
        tracking['strict_calls'] = True
        tracking['aim_progress'] = user_simulator.get_aim_progress() if is_aim_driven else None
    
    return SimulationResult(
        task_id=f"task_{scenario.scenario_id}",
        scenario_id=scenario.scenario_id,
        conversation=conversation,
        termination_reason=termination_reason,
        turn_count=turn_count,
        final_state=tracking,
        disclosed_info=tracking.get('revealed_indices', []) if isinstance(tracking, dict) else []
    )
