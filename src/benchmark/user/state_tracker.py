"""
State Tracker

Tracks the information revelation state during user simulation.
Tracks ALL information types: flow_info, constraints, and distractors.
Provides intelligent suggestions for what information to reveal next.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from loguru import logger

from benchmark.data_model.scenario import Scenario, StateQueue
from benchmark.data_model.message import AnyMessage, MessageRole


class InfoRevealEvent(BaseModel):
    """Record of an information revelation event"""
    turn: int
    info_index: int
    info_content: str
    info_type: str  # "flow_info", "constraint", "distractor"
    trigger: str  # "question", "proactive", "constraint_triggered", "side_note"


class StateTracker:
    """
    Tracks the state of information revelation during a conversation.
    
    This class tracks ALL types of information:
    1. flow_info - Core information pieces revealed progressively
    2. constraints - Hard requirements (mentioned once when relevant)
    3. distractors - Side notes/tangential thoughts
    
    All types are tracked for completeness evaluation.
    """
    
    def __init__(self, scenario: Scenario):
        """
        Initialize the state tracker.
        
        Args:
            scenario: The scenario being simulated
        """
        self.scenario = scenario
        self.state_queue = scenario.state_queue
        
        # Cache all info values for easier access
        self._flow_info_values = self._get_flow_info_values()
        self._constraint_values = self._get_constraint_values()
        self._distractor_values = self._get_distractor_values()  # Now a list
        
        # Tracking state for ALL info types
        self.revealed_flow_info_indices: List[int] = []
        self.revealed_constraint_indices: List[int] = []
        self.revealed_distractor_indices: List[int] = []
        
        # Legacy compatibility
        self.revealed_info_indices: List[int] = []  # Alias for revealed_flow_info_indices
        self.triggered_constraint_indices: List[int] = []  # Alias
        self.distractor_used: bool = False  # Legacy flag
        
        self.current_turn: int = 0
        
        # History - now tracks all types
        self.reveal_history: List[InfoRevealEvent] = []
    
    def _get_flow_info_values(self) -> List[str]:
        """Get flow_info as list of strings (compatible with both formats)"""
        sq = self.state_queue
        if hasattr(sq, 'get_flow_info_values'):
            return sq.get_flow_info_values()
        return [str(item) for item in sq.flow_info]
    
    def _get_constraint_values(self) -> List[str]:
        """Get constraints as list of strings (compatible with both formats)"""
        sq = self.state_queue
        if hasattr(sq, 'get_constraint_values'):
            return sq.get_constraint_values()
        return [str(item) for item in sq.constraints]
    
    def _get_distractor_values(self) -> List[str]:
        """Get distractors as list of strings (handles both single and list formats)"""
        sq = self.state_queue
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
    
    def _get_distractor_value(self) -> Optional[str]:
        """Legacy: Get distractor as single string"""
        values = self._distractor_values
        return " | ".join(values) if values else None
    
    def reset(self) -> None:
        """Reset the tracker for a new conversation"""
        self.revealed_flow_info_indices = []
        self.revealed_constraint_indices = []
        self.revealed_distractor_indices = []
        self.revealed_info_indices = []  # Legacy
        self.triggered_constraint_indices = []  # Legacy
        self.distractor_used = False
        self.current_turn = 0
        self.reveal_history = []
    
    # =========================================================================
    # Information Tracking
    # =========================================================================
    
    def mark_info_revealed(
        self,
        index: int,
        trigger: str = "proactive"
    ) -> None:
        """
        Mark a flow_info item as revealed.
        
        Args:
            index: Index of the flow_info item
            trigger: What triggered the revelation
        """
        if index not in self.revealed_flow_info_indices:
            self.revealed_flow_info_indices.append(index)
            self.revealed_info_indices.append(index)  # Legacy compatibility
            
            if 0 <= index < len(self._flow_info_values):
                event = InfoRevealEvent(
                    turn=self.current_turn,
                    info_index=index,
                    info_content=self._flow_info_values[index],
                    info_type="flow_info",
                    trigger=trigger
                )
                self.reveal_history.append(event)
                content_preview = self._flow_info_values[index][:50]
                logger.debug(f"Flow info revealed: [{index}] {content_preview}...")
    
    def mark_constraint_revealed(self, index: int, trigger: str = "mentioned") -> None:
        """
        Mark a constraint as revealed (mentioned once).
        
        Args:
            index: Index of the constraint
            trigger: What triggered the revelation
        """
        if index not in self.revealed_constraint_indices:
            self.revealed_constraint_indices.append(index)
            self.triggered_constraint_indices.append(index)  # Legacy compatibility
            
            if 0 <= index < len(self._constraint_values):
                event = InfoRevealEvent(
                    turn=self.current_turn,
                    info_index=index,
                    info_content=self._constraint_values[index],
                    info_type="constraint",
                    trigger=trigger
                )
                self.reveal_history.append(event)
                content_preview = self._constraint_values[index][:50]
                logger.debug(f"Constraint revealed: [{index}] {content_preview}...")
    
    def mark_constraint_triggered(self, index: int) -> None:
        """Legacy: Mark a constraint as having been triggered"""
        self.mark_constraint_revealed(index, trigger="triggered")
    
    def mark_distractor_revealed(self, index: int = 0, trigger: str = "side_note") -> None:
        """
        Mark a distractor as revealed.
        
        Args:
            index: Index of the distractor
            trigger: What triggered the revelation
        """
        if index not in self.revealed_distractor_indices:
            self.revealed_distractor_indices.append(index)
            self.distractor_used = True  # Legacy compatibility
            
            if 0 <= index < len(self._distractor_values):
                event = InfoRevealEvent(
                    turn=self.current_turn,
                    info_index=index,
                    info_content=self._distractor_values[index],
                    info_type="distractor",
                    trigger=trigger
                )
                self.reveal_history.append(event)
                content_preview = self._distractor_values[index][:50]
                logger.debug(f"Distractor revealed: [{index}] {content_preview}...")
    
    def mark_distractor_used(self) -> None:
        """Legacy: Mark the first distractor as used"""
        if self._distractor_values and 0 not in self.revealed_distractor_indices:
            self.mark_distractor_revealed(0, trigger="side_note")
        self.distractor_used = True
        logger.debug("Distractor used")
    
    def increment_turn(self) -> None:
        """Increment the turn counter"""
        self.current_turn += 1
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_unrevealed_info(self) -> List[Tuple[int, str]]:
        """
        Get list of unrevealed flow_info items.
        
        Returns:
            List of (index, content) tuples
        """
        unrevealed = []
        for i, info in enumerate(self._flow_info_values):
            if i not in self.revealed_info_indices:
                unrevealed.append((i, info))
        return unrevealed
    
    def get_revealed_info(self) -> List[Tuple[int, str]]:
        """
        Get list of revealed flow_info items.
        
        Returns:
            List of (index, content) tuples
        """
        revealed = []
        for i in self.revealed_flow_info_indices:
            if 0 <= i < len(self._flow_info_values):
                revealed.append((i, self._flow_info_values[i]))
        return revealed
    
    def get_unrevealed_constraints(self) -> List[Tuple[int, str]]:
        """
        Get list of constraints that haven't been revealed yet.
        
        Returns:
            List of (index, content) tuples
        """
        unrevealed = []
        for i, constraint in enumerate(self._constraint_values):
            if i not in self.revealed_constraint_indices:
                unrevealed.append((i, constraint))
        return unrevealed
    
    def get_unrevealed_distractors(self) -> List[Tuple[int, str]]:
        """
        Get list of distractors that haven't been revealed yet.
        
        Returns:
            List of (index, content) tuples
        """
        unrevealed = []
        for i, distractor in enumerate(self._distractor_values):
            if i not in self.revealed_distractor_indices:
                unrevealed.append((i, distractor))
        return unrevealed
    
    def get_untriggered_constraints(self) -> List[Tuple[int, str]]:
        """Legacy: Get list of constraints that haven't been triggered yet."""
        return self.get_unrevealed_constraints()
    
    def should_use_distractor(self) -> bool:
        """
        Check if it's a good time to use a distractor.
        
        Returns:
            True if there are unrevealed distractors and timing is right
        """
        unrevealed = self.get_unrevealed_distractors()
        if not unrevealed:
            return False
        
        # Use distractor between turns 2-5
        return 2 <= self.current_turn <= 5
    
    def get_next_info_suggestion(self) -> Optional[Tuple[int, str]]:
        """
        Suggest the next piece of information to reveal.
        
        Returns:
            Tuple of (index, content) or None if all revealed
        """
        unrevealed = self.get_unrevealed_info()
        if not unrevealed:
            return None
        
        # Return the first unrevealed item (maintain order)
        return unrevealed[0]
    
    def get_next_distractor_suggestion(self) -> Optional[Tuple[int, str]]:
        """
        Suggest the next distractor to mention.
        
        Returns:
            Tuple of (index, content) or None if all used
        """
        unrevealed = self.get_unrevealed_distractors()
        if not unrevealed:
            return None
        return unrevealed[0]
    
    # =========================================================================
    # Completion Checks
    # =========================================================================
    
    def can_complete(self) -> bool:
        """
        Check if the conversation can reasonably complete.
        
        Returns:
            True if minimum information has been exchanged
        """
        # At minimum, most info should be revealed (flow_info + constraints + distractors)
        return self.get_reveal_progress() >= 0.7
    
    # =========================================================================
    # State Summary
    # =========================================================================
    
    def get_state_summary(self) -> dict:
        """
        Get a summary of the current tracking state.
        
        Includes ALL info types: flow_info, constraints, distractors.
        
        Returns:
            Dictionary with state information
        """
        # Total counts
        total_flow_info = len(self._flow_info_values)
        total_constraints = len(self._constraint_values)
        total_distractors = len(self._distractor_values)
        total_all = total_flow_info + total_constraints + total_distractors
        
        # Revealed counts
        revealed_flow_info = len(self.revealed_flow_info_indices)
        revealed_constraints = len(self.revealed_constraint_indices)
        revealed_distractors = len(self.revealed_distractor_indices)
        revealed_all = revealed_flow_info + revealed_constraints + revealed_distractors
        
        return {
            "current_turn": self.current_turn,
            # Flow info tracking
            "total_flow_info": total_flow_info,
            "revealed_flow_info_count": revealed_flow_info,
            "revealed_flow_info_indices": list(self.revealed_flow_info_indices),
            "unrevealed_flow_info_indices": [i for i in range(total_flow_info) 
                                             if i not in self.revealed_flow_info_indices],
            # Constraint tracking
            "total_constraints": total_constraints,
            "revealed_constraint_count": revealed_constraints,
            "revealed_constraint_indices": list(self.revealed_constraint_indices),
            "unrevealed_constraint_indices": [i for i in range(total_constraints)
                                              if i not in self.revealed_constraint_indices],
            # Distractor tracking
            "total_distractors": total_distractors,
            "revealed_distractor_count": revealed_distractors,
            "revealed_distractor_indices": list(self.revealed_distractor_indices),
            "unrevealed_distractor_indices": [i for i in range(total_distractors)
                                              if i not in self.revealed_distractor_indices],
            # Aggregated metrics
            "total_info_count": total_all,
            "revealed_count": revealed_all,  # Total revealed across all types
            "reveal_progress": revealed_all / total_all if total_all > 0 else 1.0,
            "can_complete": self.can_complete(),
            # Legacy compatibility
            "revealed_indices": list(self.revealed_flow_info_indices),
            "distractor_used": self.distractor_used,
            "triggered_constraints": revealed_constraints,
        }
    
    def get_reveal_progress(self) -> float:
        """
        Get the overall progress of information revelation (0-1).
        Includes flow_info, constraints, and distractors.
        
        Returns:
            Float between 0 and 1
        """
        total = len(self._flow_info_values) + len(self._constraint_values) + len(self._distractor_values)
        if total == 0:
            return 1.0
        revealed = (len(self.revealed_flow_info_indices) + 
                   len(self.revealed_constraint_indices) + 
                   len(self.revealed_distractor_indices))
        return revealed / total
    
    def get_flow_info_progress(self) -> float:
        """Get progress of flow_info revelation only (0-1)."""
        total = len(self._flow_info_values)
        if total == 0:
            return 1.0
        return len(self.revealed_flow_info_indices) / total
    
    def is_all_info_revealed(self) -> bool:
        """Check if ALL information has been revealed (flow_info + constraints + distractors)"""
        flow_complete = len(self.revealed_flow_info_indices) >= len(self._flow_info_values)
        constraints_complete = len(self.revealed_constraint_indices) >= len(self._constraint_values)
        distractors_complete = len(self.revealed_distractor_indices) >= len(self._distractor_values)
        return flow_complete and constraints_complete and distractors_complete
    
    def is_flow_info_complete(self) -> bool:
        """Check if all flow_info has been revealed"""
        return len(self.revealed_flow_info_indices) >= len(self._flow_info_values)
    
    def get_tracking_prompt_context(self) -> str:
        """
        Generate context string for LLM about current state.
        
        This can be appended to the prompt to help the LLM
        make better decisions about what to reveal.
        
        Returns:
            Context string
        """
        unrevealed_flow = self.get_unrevealed_info()
        unrevealed_constraints = self.get_unrevealed_constraints()
        unrevealed_distractors = self.get_unrevealed_distractors()
        
        total_info = len(self._flow_info_values) + len(self._constraint_values) + len(self._distractor_values)
        total_revealed = (len(self.revealed_flow_info_indices) + 
                         len(self.revealed_constraint_indices) + 
                         len(self.revealed_distractor_indices))
        
        lines = [
            f"[Tracking Context - Turn {self.current_turn}]",
            f"Overall progress: {total_revealed}/{total_info} items revealed",
            f"  - Flow info: {len(self.revealed_flow_info_indices)}/{len(self._flow_info_values)}",
            f"  - Constraints: {len(self.revealed_constraint_indices)}/{len(self._constraint_values)}",
            f"  - Side notes: {len(self.revealed_distractor_indices)}/{len(self._distractor_values)}",
        ]
        
        if unrevealed_flow:
            lines.append(f"Remaining flow_info: {len(unrevealed_flow)} items")
        
        if unrevealed_constraints:
            lines.append(f"Remaining constraints to mention: {len(unrevealed_constraints)}")
        
        if unrevealed_distractors and self.should_use_distractor():
            lines.append("Consider mentioning a side note naturally.")
        
        if self.can_complete():
            lines.append("Enough information exchanged - can conclude if satisfied.")
        
        return "\n".join(lines)
