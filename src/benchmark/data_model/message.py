"""
Message Data Model

Defines the message structures for multi-turn conversations.
"""

from enum import Enum
from typing import Optional, List, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role types"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """Represents a tool/function call made by the assistant"""
    id: str = Field(..., description="Unique ID for the tool call")
    name: str = Field(..., description="Name of the tool/function")
    arguments: dict = Field(default_factory=dict, description="Arguments for the tool call")


class ToolResult(BaseModel):
    """Result from a tool execution"""
    tool_call_id: str = Field(..., description="ID of the corresponding tool call")
    content: str = Field(..., description="Result content from the tool")
    is_error: bool = Field(default=False, description="Whether the tool execution resulted in an error")


class Message(BaseModel):
    """
    Base Message class for all message types
    
    Attributes:
        role: The role of the message sender
        content: The text content of the message
        timestamp: When the message was created
        turn_idx: The turn index in the conversation
        metadata: Additional metadata
    """
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Text content of the message")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")
    turn_idx: Optional[int] = Field(default=None, description="Turn index in conversation")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")

    def to_openai_format(self) -> dict:
        """Convert to OpenAI API message format"""
        return {
            "role": self.role.value,
            "content": self.content
        }


class SystemMessage(Message):
    """System message for setting context"""
    role: MessageRole = Field(default=MessageRole.SYSTEM, frozen=True)


class UserMessage(Message):
    """Message from the user (or user simulator)"""
    role: MessageRole = Field(default=MessageRole.USER, frozen=True)
    
    # User simulator specific fields
    revealed_info: Optional[List[str]] = Field(
        default=None,
        description="Information pieces revealed in this message"
    )
    constraint_triggered: Optional[str] = Field(
        default=None,
        description="Constraint that was triggered (if any)"
    )
    is_distractor: bool = Field(
        default=False,
        description="Whether this message contains distractor content"
    )


class AssistantMessage(Message):
    """Message from the assistant (agent being evaluated)"""
    role: MessageRole = Field(default=MessageRole.ASSISTANT, frozen=True)
    
    # Tool call related fields
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None,
        description="Tool calls made by the assistant"
    )
    
    # Cost tracking
    cost: Optional[float] = Field(default=None, description="API cost for this message")
    tokens_used: Optional[int] = Field(default=None, description="Tokens used")

    def has_tool_calls(self) -> bool:
        """Check if this message contains tool calls"""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    def to_openai_format(self) -> dict:
        """Convert to OpenAI API message format"""
        msg = {
            "role": self.role.value,
            "content": self.content
        }
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": str(tc.arguments)
                    }
                }
                for tc in self.tool_calls
            ]
        return msg


class ToolMessage(Message):
    """Message containing tool execution result"""
    role: MessageRole = Field(default=MessageRole.TOOL, frozen=True)
    tool_call_id: str = Field(..., description="ID of the tool call this responds to")
    tool_name: str = Field(..., description="Name of the tool")
    is_error: bool = Field(default=False, description="Whether tool execution failed")

    def to_openai_format(self) -> dict:
        """Convert to OpenAI API message format"""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content
        }


# Type alias for any message type
AnyMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


class Conversation(BaseModel):
    """
    Represents a complete conversation (list of messages)
    """
    messages: List[AnyMessage] = Field(default_factory=list)
    
    def add_message(self, message: AnyMessage) -> None:
        """Add a message to the conversation"""
        message.turn_idx = len(self.messages)
        message.timestamp = datetime.now()
        self.messages.append(message)
    
    def get_last_message(self) -> Optional[AnyMessage]:
        """Get the last message in the conversation"""
        return self.messages[-1] if self.messages else None
    
    def get_messages_by_role(self, role: MessageRole) -> List[AnyMessage]:
        """Get all messages from a specific role"""
        return [m for m in self.messages if m.role == role]
    
    def get_turn_count(self) -> int:
        """Get the number of user turns (excluding system messages)"""
        return len([m for m in self.messages if m.role in [MessageRole.USER, MessageRole.ASSISTANT]])
    
    def to_openai_format(self) -> List[dict]:
        """Convert entire conversation to OpenAI API format"""
        return [m.to_openai_format() for m in self.messages]
    
    def get_user_messages(self) -> List[UserMessage]:
        """Get all user messages"""
        return [m for m in self.messages if isinstance(m, UserMessage)]
    
    def get_assistant_messages(self) -> List[AssistantMessage]:
        """Get all assistant messages"""
        return [m for m in self.messages if isinstance(m, AssistantMessage)]
