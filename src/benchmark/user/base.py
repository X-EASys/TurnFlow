"""
Base User Interface

Defines the abstract base class for user implementations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Tuple, Any
from pydantic import BaseModel, Field

from benchmark.data_model.message import Message, UserMessage, AnyMessage
from benchmark.data_model.scenario import Scenario


class UserState(str, Enum):
    """Possible states of the user in conversation"""
    ACTIVE = "active"           # Normal conversation
    SATISFIED = "satisfied"     # User is satisfied, ready to end
    TRANSFER = "transfer"       # User wants to transfer to human
    OUT_OF_SCOPE = "out_of_scope"  # Request is out of scope
    ABANDONED = "abandoned"     # User gave up


class UserContext(BaseModel):
    """
    Context tracking for the user simulator
    """
    current_turn: int = Field(default=0, description="Current turn number")
    revealed_info_indices: List[int] = Field(
        default_factory=list,
        description="Indices of flow_info items revealed"
    )
    triggered_constraint_indices: List[int] = Field(
        default_factory=list,
        description="Indices of constraints that were triggered"
    )
    distractor_mentioned: bool = Field(
        default=False,
        description="Whether distractor has been mentioned"
    )
    state: UserState = Field(
        default=UserState.ACTIVE,
        description="Current user state"
    )
    
    def mark_info_revealed(self, index: int) -> None:
        """Mark a flow_info item as revealed"""
        if index not in self.revealed_info_indices:
            self.revealed_info_indices.append(index)
    
    def mark_constraint_triggered(self, index: int) -> None:
        """Mark a constraint as triggered"""
        if index not in self.triggered_constraint_indices:
            self.triggered_constraint_indices.append(index)
    
    def get_unrevealed_info_indices(self, total_count: int) -> List[int]:
        """Get indices of unrevealed flow_info items"""
        return [i for i in range(total_count) if i not in self.revealed_info_indices]
    
    def increment_turn(self) -> None:
        """Increment turn counter"""
        self.current_turn += 1


class BaseUser(ABC):
    """
    Abstract base class for user implementations.
    
    This defines the interface that all user simulators must implement.
    """
    
    def __init__(
        self,
        scenario: Optional[Scenario] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the base user.
        
        Args:
            scenario: The scenario to simulate
            llm: LLM model name for generation
            llm_args: Additional arguments for LLM
        """
        self.scenario = scenario
        self.llm = llm
        self.llm_args = llm_args or {}
        self.context = UserContext()
    
    @abstractmethod
    def step(
        self,
        messages: List[AnyMessage],
    ) -> Tuple[UserMessage, UserState]:
        """
        Generate the next user message given conversation history.
        
        Args:
            messages: Conversation history
            
        Returns:
            Tuple of (UserMessage, UserState)
        """
        pass
    
    @abstractmethod
    def get_opening_message(self) -> UserMessage:
        """
        Get the opening message (Turn 0) based on the anchor.
        
        Returns:
            UserMessage containing the anchor statement
        """
        pass
    
    def set_scenario(self, scenario: Scenario) -> None:
        """Set or update the scenario"""
        self.scenario = scenario
        self.reset()
    
    def reset(self) -> None:
        """Reset the user state for a new conversation"""
        self.context = UserContext()
    
    def get_state(self) -> UserState:
        """Get current user state"""
        return self.context.state
    
    def set_state(self, state: UserState) -> None:
        """Set user state"""
        self.context.state = state
    
    def get_context(self) -> UserContext:
        """Get the full user context"""
        return self.context
    
    @property
    def is_done(self) -> bool:
        """Check if user is done with conversation"""
        return self.context.state in [
            UserState.SATISFIED,
            UserState.TRANSFER,
            UserState.OUT_OF_SCOPE,
            UserState.ABANDONED,
        ]
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility"""
        # Subclasses can override if needed
        pass


class DummyUser(BaseUser):
    """
    A dummy user that returns predefined responses.
    Useful for testing and solo mode.
    """
    
    def __init__(
        self,
        responses: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.responses = responses or ["I need help."]
        self.response_index = 0
    
    def step(
        self,
        messages: List[AnyMessage],
    ) -> Tuple[UserMessage, UserState]:
        """Return the next predefined response"""
        if self.response_index >= len(self.responses):
            return UserMessage(content="Thanks, no more questions."), UserState.SATISFIED
        
        response = self.responses[self.response_index]
        self.response_index += 1
        self.context.increment_turn()
        
        return UserMessage(content=response), UserState.ACTIVE
    
    def get_opening_message(self) -> UserMessage:
        """Get the first response as opening"""
        if self.scenario and self.scenario.state_queue.anchor:
            return UserMessage(content=self.scenario.state_queue.anchor)
        return UserMessage(content=self.responses[0] if self.responses else "Hello")
    
    def reset(self) -> None:
        """Reset the dummy user"""
        super().reset()
        self.response_index = 0
