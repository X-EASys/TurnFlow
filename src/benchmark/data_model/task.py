"""
Task Data Model

Defines the structure for evaluation tasks.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from benchmark.data_model.scenario import Scenario, StateQueue
from benchmark.data_model.message import AnyMessage


class RewardType(str, Enum):
    """Types of reward components"""
    INFO_GATHERING = "info_gathering"      # Did agent gather all necessary info?
    CONSTRAINT_RESPECT = "constraint_respect"  # Did agent respect user constraints?
    TASK_COMPLETION = "task_completion"    # Was the task completed?
    COMMUNICATION = "communication"        # Quality of communication
    EFFICIENCY = "efficiency"              # Conversation efficiency
    SAFETY = "safety"                      # Safety checks


class EvaluationCriteria(BaseModel):
    """
    Criteria for evaluating a task completion
    
    Attributes:
        reward_basis: List of reward types to include in evaluation
        required_info_gathered: Information that must be gathered by agent
        constraint_violations_forbidden: Constraints that must not be violated
        expected_outcome: Expected outcome description
        max_turns: Maximum allowed turns (for efficiency)
        custom_checks: Custom check functions (as strings for now)
    """
    reward_basis: List[RewardType] = Field(
        default_factory=lambda: [
            RewardType.INFO_GATHERING,
            RewardType.CONSTRAINT_RESPECT,
            RewardType.TASK_COMPLETION
        ],
        description="Types of rewards to include in evaluation"
    )
    required_info_gathered: Optional[List[str]] = Field(
        default=None,
        description="Information that the agent should have gathered"
    )
    constraint_violations_forbidden: Optional[List[str]] = Field(
        default=None,
        description="Constraints that must not be violated"
    )
    expected_outcome: Optional[str] = Field(
        default=None,
        description="Description of the expected successful outcome"
    )
    max_turns: Optional[int] = Field(
        default=20,
        description="Maximum allowed conversation turns"
    )
    custom_checks: Optional[List[str]] = Field(
        default=None,
        description="Custom verification checks"
    )


class TaskMetadata(BaseModel):
    """Metadata for a task"""
    created_at: Optional[str] = None
    created_by: Optional[str] = None
    version: str = "1.0"
    source: Optional[str] = None
    notes: Optional[str] = None


class Task(BaseModel):
    """
    Complete Task Definition
    
    A Task combines a Scenario with evaluation criteria to form
    a complete test case for the benchmark.
    
    Attributes:
        task_id: Unique identifier for the task
        scenario: The user scenario for this task
        evaluation_criteria: Criteria for evaluating success
        initial_state: Optional initial conversation state
        metadata: Additional metadata
    """
    task_id: str = Field(..., description="Unique task identifier")
    scenario: Scenario = Field(..., description="The user scenario")
    evaluation_criteria: EvaluationCriteria = Field(
        default_factory=EvaluationCriteria,
        description="Evaluation criteria for this task"
    )
    initial_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Initial environment/conversation state"
    )
    initial_messages: Optional[List[AnyMessage]] = Field(
        default=None,
        description="Pre-existing conversation history"
    )
    metadata: Optional[TaskMetadata] = Field(
        default=None,
        description="Task metadata"
    )

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario,
        task_id: Optional[str] = None,
        evaluation_criteria: Optional[EvaluationCriteria] = None
    ) -> "Task":
        """
        Create a Task from a Scenario with auto-generated evaluation criteria
        """
        import uuid
        
        # Generate task_id if not provided
        if task_id is None:
            task_id = f"{scenario.domain}_{uuid.uuid4().hex[:8]}"
        
        # Auto-generate evaluation criteria based on scenario
        if evaluation_criteria is None:
            evaluation_criteria = cls._generate_criteria_from_scenario(scenario)
        
        return cls(
            task_id=task_id,
            scenario=scenario,
            evaluation_criteria=evaluation_criteria
        )
    
    @staticmethod
    def _generate_criteria_from_scenario(scenario: Scenario) -> EvaluationCriteria:
        """
        Auto-generate evaluation criteria from scenario's state queue
        """
        state_queue = scenario.state_queue
        
        # Required info = all flow_info items (use helper for compatibility)
        flow_info_values = state_queue.get_flow_info_values() if hasattr(state_queue, 'get_flow_info_values') else list(state_queue.flow_info)
        required_info = flow_info_values if flow_info_values else None
        
        # Forbidden violations = all constraints (use helper for compatibility)
        constraint_values = state_queue.get_constraint_values() if hasattr(state_queue, 'get_constraint_values') else list(state_queue.constraints)
        forbidden_violations = constraint_values if constraint_values else None
        
        return EvaluationCriteria(
            reward_basis=[
                RewardType.INFO_GATHERING,
                RewardType.CONSTRAINT_RESPECT,
                RewardType.TASK_COMPLETION,
                RewardType.EFFICIENCY
            ],
            required_info_gathered=required_info,
            constraint_violations_forbidden=forbidden_violations,
            max_turns=10 + len(state_queue.flow_info) * 2  # Adaptive max turns
        )

    def get_difficulty(self) -> int:
        """Get task difficulty from scenario"""
        return self.scenario.difficulty
    
    def get_domain(self) -> str:
        """Get task domain from scenario"""
        return self.scenario.domain


class TaskSet(BaseModel):
    """Collection of tasks, typically for a domain or evaluation suite"""
    name: str = Field(..., description="Name of the task set")
    description: Optional[str] = Field(default=None, description="Description")
    tasks: List[Task] = Field(default_factory=list, description="List of tasks")
    domain: Optional[str] = Field(default=None, description="Domain if all tasks are same domain")
    version: str = Field(default="1.0", description="Version of the task set")
    
    def add_task(self, task: Task) -> None:
        """Add a task to the set"""
        self.tasks.append(task)
    
    def get_tasks_by_difficulty(self, difficulty: int) -> List[Task]:
        """Get tasks filtered by difficulty"""
        return [t for t in self.tasks if t.get_difficulty() == difficulty]
    
    def get_task_count(self) -> int:
        """Get total number of tasks"""
        return len(self.tasks)
