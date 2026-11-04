"""
Base Evaluator Interface

Abstract base class for all evaluators.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger

from benchmark.data_model.task import Task
from benchmark.data_model.simulation import SimulationResult


class ScoreType(str, Enum):
    """Types of scores"""
    INFO_EXTRACTION = "info_extraction"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    EFFICIENCY = "efficiency"
    COMMUNICATION = "communication"
    OVERALL = "overall"


class ScoreBreakdown(BaseModel):
    """Detailed breakdown of evaluation scores"""
    
    # Core scores (0.0 to 1.0)
    info_extraction: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Score for information extraction"
    )
    constraint_satisfaction: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Score for constraint satisfaction"
    )
    efficiency: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Score for conversation efficiency"
    )
    correctness: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Score for ground truth correctness (optional)"
    )
    communication: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Score for communication quality (optional)"
    )
    
    # Overall score
    overall: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Weighted overall score"
    )
    
    # Weights used
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "info_extraction": 0.35,
            "constraint_satisfaction": 0.40,
            "efficiency": 0.25,
        },
        description="Weights used for overall calculation"
    )
    
    def compute_overall(self) -> float:
        """Compute overall score from components"""
        total = 0.0
        weight_sum = 0.0
        
        for key, weight in self.weights.items():
            value = getattr(self, key, None)
            if value is not None:
                total += value * weight
                weight_sum += weight
        
        if weight_sum > 0:
            self.overall = total / weight_sum
        
        return self.overall


class EvalResult(BaseModel):
    """Complete evaluation result"""
    
    # Identifiers
    task_id: str = Field(..., description="Task identifier")
    scenario_id: str = Field(..., description="Scenario identifier")
    
    # Scores
    scores: ScoreBreakdown = Field(
        default_factory=ScoreBreakdown,
        description="Score breakdown"
    )
    
    # Success determination
    success: bool = Field(
        default=False,
        description="Whether the task was completed successfully"
    )
    
    # Detailed analysis
    info_extracted: List[str] = Field(
        default_factory=list,
        description="Information successfully extracted by agent"
    )
    info_missed: List[str] = Field(
        default_factory=list,
        description="Information the agent failed to extract"
    )
    constraints_satisfied: List[str] = Field(
        default_factory=list,
        description="Constraints that were satisfied"
    )
    constraints_violated: List[str] = Field(
        default_factory=list,
        description="Constraints that were violated"
    )
    
    # Turn analysis
    turn_count: int = Field(default=0, description="Number of turns")
    expected_turns_min: Optional[int] = Field(default=None)
    expected_turns_max: Optional[int] = Field(default=None)
    
    # Qualitative feedback
    feedback: Optional[str] = Field(
        default=None,
        description="Qualitative feedback about performance"
    )
    
    # Traceability
    evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Evidence for each scoring decision"
    )
    
    # Metadata
    evaluator_type: str = Field(default="unknown")
    evaluation_time_ms: Optional[float] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the evaluation"
    )
    
    # Composite per-phase scoring
    phase_scores: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-phase scoring: {'phase_1': PhaseScore, 'phase_2': PhaseScore}"
    )

    def is_verifiable(self) -> bool:
        """Check if this evaluation is fully verifiable"""
        # Verifiable if we have evidence for key decisions
        return len(self.evidence) > 0


class PhaseScore(BaseModel):
    """Per-phase scoring for composite scenarios."""
    task_type: str = Field(..., description="Task type for this phase")
    phase_name: str = Field(..., description="'phase_1' or 'phase_2'")
    scores: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    success: bool = Field(default=False)
    feedback: Optional[str] = Field(default=None)


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators.
    
    Evaluators assess agent performance based on:
    1. Information extraction accuracy
    2. Constraint satisfaction
    3. Conversation efficiency
    4. (Optional) Communication quality
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        success_threshold: float = 0.7,
    ):
        """
        Initialize the evaluator.
        
        Args:
            weights: Custom weights for score components
            success_threshold: Threshold for overall success
        """
        self.weights = weights or {
            "info_extraction": 0.35,
            "constraint_satisfaction": 0.40,
            "efficiency": 0.25,
        }
        self.success_threshold = success_threshold
    
    @abstractmethod
    def evaluate(
        self,
        task: Task,
        result: SimulationResult,
    ) -> EvalResult:
        """
        Evaluate a simulation result.
        
        Args:
            task: The task that was executed
            result: The simulation result
            
        Returns:
            EvalResult with scores and analysis
        """
        pass
    
    def _create_base_result(
        self,
        task: Task,
        result: SimulationResult,
    ) -> EvalResult:
        """Create a base evaluation result"""
        # Get expected turns range
        min_turns, max_turns = task.scenario.get_expected_turns_range()
        
        return EvalResult(
            task_id=task.task_id,
            scenario_id=task.scenario.scenario_id,
            scores=ScoreBreakdown(weights=self.weights),
            turn_count=result.turn_count,
            expected_turns_min=min_turns,
            expected_turns_max=max_turns,
        )
    
    def _determine_success(self, scores: ScoreBreakdown) -> bool:
        """Determine if the task was successful"""
        # Must satisfy most constraints
        if scores.constraint_satisfaction < 0.8:
            return False
        
        # Overall score must meet threshold
        return scores.overall >= self.success_threshold
