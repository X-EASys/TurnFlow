"""
Simulation Data Model

Defines structures for simulation runs and results.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from benchmark.data_model.message import AnyMessage, Conversation


class TerminationReason(str, Enum):
    """Reasons for simulation termination"""
    SUCCESS = "success"                    # User satisfied, task completed
    USER_ABANDON = "user_abandon"          # User gave up
    TRANSFER = "transfer"                  # User requested human transfer
    MAX_TURNS = "max_turns"                # Exceeded maximum turns
    MAX_ERRORS = "max_errors"              # Too many errors
    AGENT_ERROR = "agent_error"            # Agent encountered fatal error
    TIMEOUT = "timeout"                    # Simulation timed out
    OUT_OF_SCOPE = "out_of_scope"          # Request out of scope


class UserSimulatorState(BaseModel):
    """
    Tracks the state of the user simulator during conversation
    
    This tracks which information has been revealed and which constraints triggered.
    """
    revealed_flow_info: List[int] = Field(
        default_factory=list,
        description="Indices of flow_info items that have been revealed"
    )
    triggered_constraints: List[int] = Field(
        default_factory=list,
        description="Indices of constraints that have been triggered"
    )
    distractor_used: bool = Field(
        default=False,
        description="Whether the distractor has been mentioned"
    )
    current_turn: int = Field(
        default=0,
        description="Current turn number"
    )
    is_satisfied: bool = Field(
        default=False,
        description="Whether user is satisfied with the outcome"
    )
    wants_transfer: bool = Field(
        default=False,
        description="Whether user wants to transfer to human"
    )

    def mark_info_revealed(self, index: int) -> None:
        """Mark a flow_info item as revealed"""
        if index not in self.revealed_flow_info:
            self.revealed_flow_info.append(index)
    
    def mark_constraint_triggered(self, index: int) -> None:
        """Mark a constraint as triggered"""
        if index not in self.triggered_constraints:
            self.triggered_constraints.append(index)
    
    def get_unrevealed_info_count(self, total_info: int) -> int:
        """Get count of unrevealed information pieces"""
        return total_info - len(self.revealed_flow_info)
    
    def increment_turn(self) -> None:
        """Increment turn counter"""
        self.current_turn += 1


class RewardBreakdown(BaseModel):
    """Detailed breakdown of reward components"""
    info_gathering_reward: Optional[float] = Field(
        default=None,
        description="Reward for gathering required information"
    )
    constraint_respect_reward: Optional[float] = Field(
        default=None,
        description="Reward for respecting user constraints"
    )
    task_completion_reward: Optional[float] = Field(
        default=None,
        description="Reward for task completion"
    )
    communication_reward: Optional[float] = Field(
        default=None,
        description="Reward for communication quality"
    )
    efficiency_reward: Optional[float] = Field(
        default=None,
        description="Reward for conversation efficiency"
    )
    safety_reward: Optional[float] = Field(
        default=None,
        description="Reward for safety compliance"
    )


class RewardInfo(BaseModel):
    """
    Complete reward information for a simulation
    """
    reward: float = Field(..., ge=0.0, le=1.0, description="Total reward (0-1)")
    reward_breakdown: Optional[RewardBreakdown] = Field(
        default=None,
        description="Detailed reward breakdown"
    )
    passed_checks: List[str] = Field(
        default_factory=list,
        description="List of passed verification checks"
    )
    failed_checks: List[str] = Field(
        default_factory=list,
        description="List of failed verification checks"
    )
    info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional information"
    )

    def is_success(self, threshold: float = 1.0) -> bool:
        """Check if reward meets success threshold"""
        return self.reward >= threshold
    
    def is_partial_success(self, threshold: float = 0.5) -> bool:
        """Check if reward meets partial success threshold"""
        return self.reward >= threshold


class SimulationState(BaseModel):
    """Current state of a simulation"""
    user_state: UserSimulatorState = Field(
        default_factory=UserSimulatorState,
        description="User simulator state"
    )
    environment_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Environment state snapshot"
    )
    is_done: bool = Field(
        default=False,
        description="Whether simulation is complete"
    )
    termination_reason: Optional[TerminationReason] = Field(
        default=None,
        description="Reason for termination if done"
    )


class SimulationConfig(BaseModel):
    """Configuration for a simulation run"""
    max_turns: int = Field(default=20, description="Maximum conversation turns")
    timeout_seconds: float = Field(default=300.0, description="Timeout in seconds")
    user_llm: str = Field(default="gpt-4o-mini", description="User simulator LLM")
    user_temperature: float = Field(default=0.7, description="User simulator temperature")


class SimulationResult(BaseModel):
    """
    Result of a completed simulation.
    
    This is the main output of running a conversation between
    the user simulator and an agent.
    """
    task_id: str = Field(..., description="Task identifier")
    scenario_id: str = Field(..., description="Scenario identifier")
    
    # Conversation
    conversation: Conversation = Field(..., description="Complete conversation")
    
    # Termination
    termination_reason: TerminationReason = Field(
        ...,
        description="Why the simulation ended"
    )
    
    # Turn count
    turn_count: int = Field(default=0, description="Number of conversation turns")
    
    # User state
    user_state: Optional[UserSimulatorState] = Field(
        default=None,
        description="Final user simulator state"
    )
    final_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime mechanism revision, completion progress, and execution diagnostics"
    )
    
    # Timing
    start_time: Optional[datetime] = Field(default=None, description="Start time")
    end_time: Optional[datetime] = Field(default=None, description="End time")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )

    # === Composite Task Type ===
    phase_boundary: Optional[int] = Field(
        default=None,
        description="Aim index where Phase 2 starts (for composite scenarios)"
    )

    def get_duration_seconds(self) -> Optional[float]:
        """Get simulation duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class SimulationRun(BaseModel):

    """
    Complete record of a simulation run
    
    This captures everything about a single task execution.
    """
    run_id: str = Field(..., description="Unique run identifier")
    task_id: str = Field(..., description="Task identifier")
    
    # Timing
    started_at: Optional[datetime] = Field(default=None, description="Start time")
    ended_at: Optional[datetime] = Field(default=None, description="End time")
    
    # Conversation
    messages: List[AnyMessage] = Field(
        default_factory=list,
        description="Complete conversation trajectory"
    )
    
    # State
    final_state: Optional[SimulationState] = Field(
        default=None,
        description="Final simulation state"
    )
    termination_reason: Optional[TerminationReason] = Field(
        default=None,
        description="Why simulation ended"
    )
    
    # Results
    reward_info: Optional[RewardInfo] = Field(
        default=None,
        description="Evaluation reward"
    )
    
    # Metadata
    agent_llm: Optional[str] = Field(default=None, description="Agent LLM used")
    user_llm: Optional[str] = Field(default=None, description="User simulator LLM used")
    seed: Optional[int] = Field(default=None, description="Random seed")
    cost: Optional[float] = Field(default=None, description="Total API cost")
    error_count: int = Field(default=0, description="Number of errors during run")
    
    def get_turn_count(self) -> int:
        """Get number of conversation turns"""
        from benchmark.data_model.message import MessageRole
        return len([m for m in self.messages if m.role == MessageRole.USER])
    
    def get_duration_seconds(self) -> Optional[float]:
        """Get duration in seconds"""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None
    
    def is_success(self) -> bool:
        """Check if run was successful"""
        return (
            self.reward_info is not None 
            and self.reward_info.reward >= 1.0
            and self.termination_reason == TerminationReason.SUCCESS
        )
    
    def add_message(self, message: AnyMessage) -> None:
        """Add a message to the trajectory"""
        message.turn_idx = len(self.messages)
        message.timestamp = datetime.now()
        self.messages.append(message)


class SimulationBatch(BaseModel):
    """Collection of simulation runs, typically for a full evaluation"""
    batch_id: str = Field(..., description="Unique batch identifier")
    runs: List[SimulationRun] = Field(default_factory=list, description="List of runs")
    
    # Aggregated stats
    total_tasks: int = Field(default=0)
    completed_tasks: int = Field(default=0)
    successful_tasks: int = Field(default=0)
    
    # Config
    domain: Optional[str] = Field(default=None)
    agent_llm: Optional[str] = Field(default=None)
    user_llm: Optional[str] = Field(default=None)
    
    def add_run(self, run: SimulationRun) -> None:
        """Add a run to the batch"""
        self.runs.append(run)
        self.completed_tasks += 1
        if run.is_success():
            self.successful_tasks += 1
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.completed_tasks == 0:
            return 0.0
        return self.successful_tasks / self.completed_tasks
    
    def get_average_reward(self) -> float:
        """Calculate average reward across all runs"""
        rewards = [r.reward_info.reward for r in self.runs if r.reward_info]
        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards)
    
    def get_average_turns(self) -> float:
        """Calculate average turn count"""
        turns = [r.get_turn_count() for r in self.runs]
        if not turns:
            return 0.0
        return sum(turns) / len(turns)
