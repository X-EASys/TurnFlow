"""
API Agent Implementation

Agent that uses external LLM APIs (OpenAI, Anthropic, etc.) via LiteLLM.
"""

import time
from typing import List, Optional, Dict, Any
from loguru import logger

from benchmark.agent.base import BaseAgent, AgentConfig, AgentResponse, AgentType
from benchmark.data_model.message import AnyMessage, MessageRole, SystemMessage
from benchmark.utils.llm_utils import generate_chat_completion


class APIAgent(BaseAgent):
    """
    Agent that uses external LLM APIs.
    
    Supports any model accessible through LiteLLM:
    - OpenAI: gpt-4, gpt-4o, gpt-3.5-turbo
    - Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku
    - Google: gemini-pro
    - And many more
    
    Example:
        ```python
        config = AgentConfig(
            name="my-gpt4-agent",
            model_id="gpt-4",
            api_key="sk-...",
            temperature=0.7
        )
        agent = APIAgent(config)
        response = agent.respond(messages)
        ```
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the API agent.
        
        Args:
            config: Agent configuration with API settings
        """
        config.agent_type = AgentType.API
        super().__init__(config)
        
        # Configure LiteLLM if API key provided
        if config.api_key:
            self._setup_api_key()
    
    def _setup_api_key(self) -> None:
        """Setup API key for the model.

        Note: API key is passed directly via kwargs from llm_routes,
        not through environment variables. This avoids race conditions
        when multiple agents run in parallel threads.
        """
        # No-op: key flows through llm_routes kwargs to litellm.completion(api_key=...)
        # Do NOT mutate os.environ here — it is process-global and not thread-safe.
        pass
    
    def _prepare_messages(
        self,
        messages: List[AnyMessage]
    ) -> List[Dict[str, str]]:
        """
        Convert messages to API format.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of messages in OpenAI format
        """
        api_messages = []
        
        # Add system prompt if set
        if self._system_prompt:
            api_messages.append({
                "role": "system",
                "content": self._system_prompt
            })
        
        # Convert each message
        for msg in messages:
            # Skip system messages if we already added system prompt
            if msg.role == MessageRole.SYSTEM and self._system_prompt:
                continue
            
            api_messages.append(msg.to_openai_format())
        
        return api_messages
    
    def respond(
        self,
        messages: List[AnyMessage],
        **kwargs
    ) -> AgentResponse:
        """
        Generate a response using the API.
        
        Args:
            messages: Conversation history
            **kwargs: Additional parameters to override config
            
        Returns:
            AgentResponse with generated content
        """
        start_time = time.time()
        
        # Prepare messages
        api_messages = self._prepare_messages(messages)
        
        # Merge config with overrides
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        
        # Prepare extra args
        extra_args = self.config.extra_params.copy()
        for key in ('_first_pass_stage', '_first_pass_budget', 'max_retries', 'num_retries'):
            if key in kwargs:
                extra_args[key] = kwargs[key]
        if self.config.base_url:
            extra_args["api_base"] = self.config.base_url
        
        try:
            # Call LLM with raw response
            response = generate_chat_completion(
                messages=api_messages,
                model=self.config.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                return_raw_response=True,  # Get full response object
                **extra_args
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract content - handle both object and string responses
            # Following tau2-bench pattern: check choices exists and is not empty
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                content = choice.message.content or ""
                finish_reason = choice.finish_reason or "stop"
                # Extract usage
                usage = None
                if hasattr(response, "usage") and response.usage:
                    usage = {
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(response.usage, 'total_tokens', 0),
                    }
                    self._total_tokens_used += usage["total_tokens"]
            else:
                # Fallback for string response or empty choices
                content = str(response) if response else ""
                finish_reason = "stop"
                usage = None
                logger.warning(f"Response has no valid choices, using fallback. Response type: {type(response)}")
            
            self._call_count += 1
            
            return AgentResponse(
                content=content,
                model_id=self.config.model_id,
                finish_reason=finish_reason,
                usage=usage,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            if kwargs.get('_first_pass_stage') is not None:
                # Release first-pass runs retain technical failure in the ledger;
                # provider exception text must never become assistant speech.
                raise
            logger.error(f"API call failed: {e}")

            # Fail-fast if account is out of credit — keep going would only fill
            # the conversation log with "[Error: ... 用户额度不足 ...]" entries
            # that the verifier judge cannot distinguish from a real refusal.
            from benchmark.utils.llm_utils import _fail_fast_on_quota_exhausted
            _fail_fast_on_quota_exhausted(e)

            # Return error response
            return AgentResponse(
                content=f"[Error: {str(e)}]",
                model_id=self.config.model_id,
                finish_reason="error",
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def respond_async(
        self,
        messages: List[AnyMessage],
        **kwargs
    ) -> AgentResponse:
        """
        Async version of respond.
        
        Args:
            messages: Conversation history
            **kwargs: Additional parameters
            
        Returns:
            AgentResponse
        """
        # For now, just wrap the sync version
        # TODO: Implement true async with aiohttp/httpx
        return self.respond(messages, **kwargs)
