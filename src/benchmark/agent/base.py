"""
Base Agent Interface

Abstract base class for all agents to be evaluated.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from loguru import logger

from benchmark.data_model.message import (
    Message,
    AnyMessage,
    AssistantMessage,
    MessageRole,
)


class AgentType(str, Enum):
    """Types of agents"""
    API = "api"
    LOCAL = "local"
    TOOL = "tool"
    CUSTOM = "custom"


class AgentConfig(BaseModel):
    """Configuration for an agent"""
    name: str = Field(..., description="Agent name for identification")
    agent_type: AgentType = Field(default=AgentType.API, description="Type of agent")
    model_id: str = Field(..., description="Model identifier (e.g., gpt-4, claude-3)")
    
    # API configuration
    api_key: Optional[str] = Field(default=None, description="API key")
    base_url: Optional[str] = Field(default=None, description="API base URL")
    
    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # System prompt
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt for the agent"
    )
    
    # Additional options
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model-specific parameters"
    )


class AgentResponse(BaseModel):
    """Response from an agent"""
    content: str = Field(..., description="Text content of the response")
    role: MessageRole = Field(default=MessageRole.ASSISTANT)
    
    # Tool-related fields
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tool calls made by the agent"
    )
    
    # Metadata
    model_id: Optional[str] = Field(default=None, description="Model that generated this")
    finish_reason: Optional[str] = Field(default=None, description="Why generation stopped")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage")
    latency_ms: Optional[float] = Field(default=None, description="Response latency")
    
    # Internal tracking
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw API response for debugging"
    )
    
    def to_message(self, turn_idx: Optional[int] = None) -> AssistantMessage:
        """Convert to AssistantMessage"""
        return AssistantMessage(
            content=self.content,
            turn_idx=turn_idx,
            metadata={
                "model_id": self.model_id,
                "finish_reason": self.finish_reason,
                "usage": self.usage,
                "latency_ms": self.latency_ms,
            }
        )


class BaseAgent(ABC):
    """
    Abstract base class for agents to be evaluated.
    
    All agent implementations must inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.name = config.name
        self.model_id = config.model_id
        self._system_prompt = config.system_prompt
        self._conversation_history: List[AnyMessage] = []
        self._total_tokens_used = 0
        self._call_count = 0
        
        logger.debug(f"Initialized agent: {self.name} ({self.model_id})")
    
    @property
    def system_prompt(self) -> Optional[str]:
        """Get the system prompt"""
        return self._system_prompt
    
    @system_prompt.setter
    def system_prompt(self, prompt: str) -> None:
        """Set the system prompt"""
        self._system_prompt = prompt
    
    @abstractmethod
    def respond(
        self,
        messages: List[AnyMessage],
        **kwargs
    ) -> AgentResponse:
        """
        Generate a response to the conversation.
        
        This is the core method that must be implemented by all agents.
        
        Args:
            messages: List of messages in the conversation
            **kwargs: Additional generation parameters
            
        Returns:
            AgentResponse with the generated content
        """
        pass
    
    def respond_to_message(
        self,
        message: Union[str, Message],
        **kwargs
    ) -> AgentResponse:
        """
        Convenience method to respond to a single message.
        
        Args:
            message: Either a string or Message object
            **kwargs: Additional parameters
            
        Returns:
            AgentResponse
        """
        from benchmark.data_model.message import UserMessage
        
        if isinstance(message, str):
            message = UserMessage(content=message)
        
        self._conversation_history.append(message)
        response = self.respond(self._conversation_history, **kwargs)
        
        # Add response to history
        self._conversation_history.append(response.to_message())
        
        return response
    
    def reset(self) -> None:
        """Reset the agent state for a new conversation"""
        self._conversation_history = []
        logger.debug(f"Reset agent: {self.name}")
    
    def get_conversation_history(self) -> List[AnyMessage]:
        """Get the conversation history"""
        return self._conversation_history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "total_tokens_used": self._total_tokens_used,
            "call_count": self._call_count,
            "conversation_length": len(self._conversation_history),
        }
    
    def get_final_action(self) -> Optional[Dict[str, Any]]:
        """
        Extract the final action from the conversation.
        
        This can be overridden to parse agent-specific action formats.
        
        Returns:
            Dictionary describing the final action, or None
        """
        if not self._conversation_history:
            return None
        
        # Get the last assistant message
        for msg in reversed(self._conversation_history):
            if msg.role == MessageRole.ASSISTANT:
                return {
                    "type": "text_response",
                    "content": msg.content,
                }
        
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, model={self.model_id})"
