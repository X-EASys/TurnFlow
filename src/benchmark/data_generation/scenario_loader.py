"""
Scenario Loader

Utilities for loading scenario configurations from JSON files.
Supports both legacy format (simple anchor) and new format (AnchorConfig).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union
from loguru import logger

from benchmark.data_model.scenario import Scenario, StateQueue, AnchorConfig


def _convert_state_queue_data(sq_data: dict) -> dict:
    """
    Convert state_queue data to handle both legacy and new formats.
    
    Handles:
    - Legacy format: simple 'anchor' string
    - New format: 'anchor_config' object with disguised anchor support
    
    Args:
        sq_data: Raw state_queue dictionary
        
    Returns:
        Processed state_queue dictionary compatible with StateQueue model
    """
    # Make a copy to avoid modifying original
    sq_data = sq_data.copy()
    
    # Check if we have new format anchor_config
    if "anchor_config" in sq_data and isinstance(sq_data["anchor_config"], dict):
        # New format: convert anchor_config dict to AnchorConfig object
        anchor_config_data = sq_data["anchor_config"]
        sq_data["anchor_config"] = AnchorConfig(**anchor_config_data)
        
        # Ensure legacy anchor is also set for compatibility
        if not sq_data.get("anchor"):
            sq_data["anchor"] = anchor_config_data.get("initial_statement", "")
    
    elif "anchor" in sq_data and not sq_data.get("anchor_config"):
        # Legacy format: only simple anchor string
        # Create anchor_config from legacy anchor
        anchor_value = sq_data.get("anchor", "")
        sq_data["anchor_config"] = AnchorConfig(
            initial_statement=anchor_value,
            underlying_problem_type="general",
            vagueness_strategy="legacy_format",
            is_disguised=False
        )
    
    return sq_data


class ScenarioLoader:
    """
    Loader for scenario JSON files.
    
    Supports loading single files, directories, and nested structures.
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the scenario loader.
        
        Args:
            base_path: Base path for relative file lookups
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
    
    def load_from_file(self, file_path: Union[str, Path]) -> Scenario:
        """
        Load a single scenario from a JSON file.
        
        Supports both legacy format (simple anchor) and new format (AnchorConfig).
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Scenario object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        file_path = self._resolve_path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {file_path}")
        
        if not file_path.suffix.lower() == ".json":
            raise ValueError(f"Expected JSON file, got: {file_path.suffix}")
        
        logger.debug(f"Loading scenario from: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle new format with nested scenario, generation_metadata, design_plan
        if "scenario" in data and isinstance(data["scenario"], dict):
            # New format: extract scenario from nested structure
            scenario_data = data["scenario"]
            # Merge top-level metadata into scenario if scenario doesn't have its own
            if "metadata" in data and scenario_data.get("metadata") is None:
                scenario_data["metadata"] = data["metadata"]
        else:
            # Old format: data is the scenario itself
            scenario_data = data
        
        # Handle nested state_queue with anchor_config conversion
        if "state_queue" in scenario_data and isinstance(scenario_data["state_queue"], dict):
            sq_data = _convert_state_queue_data(scenario_data["state_queue"])
            scenario_data["state_queue"] = StateQueue(**sq_data)
        
        # Generate scenario_id from filename if not provided
        if "scenario_id" not in scenario_data or scenario_data["scenario_id"] is None:
            scenario_data["scenario_id"] = file_path.stem
        
        return Scenario(**scenario_data)
    
    def load_full_from_file(self, file_path: Union[str, Path]) -> dict:
        """
        Load full scenario data including metadata and design plan.
        
        Supports both legacy format (simple anchor) and new format (AnchorConfig).
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with keys: scenario, generation_metadata, design_plan (if available)
        """
        file_path = self._resolve_path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle new format
        if "scenario" in data and isinstance(data["scenario"], dict):
            scenario_data = data["scenario"]
            if "state_queue" in scenario_data and isinstance(scenario_data["state_queue"], dict):
                sq_data = _convert_state_queue_data(scenario_data["state_queue"])
                scenario_data["state_queue"] = StateQueue(**sq_data)
            
            if "scenario_id" not in scenario_data or scenario_data["scenario_id"] is None:
                scenario_data["scenario_id"] = file_path.stem
            
            return {
                "scenario": Scenario(**scenario_data),
                "generation_metadata": data.get("generation_metadata", {}),
                "design_plan": data.get("design_plan"),
            }
        else:
            # Old format: just scenario data
            if "state_queue" in data and isinstance(data["state_queue"], dict):
                sq_data = _convert_state_queue_data(data["state_queue"])
                data["state_queue"] = StateQueue(**sq_data)
            
            if "scenario_id" not in data or data["scenario_id"] is None:
                data["scenario_id"] = file_path.stem
            
            return {
                "scenario": Scenario(**data),
                "generation_metadata": {},
                "design_plan": None,
            }
    
    def load_from_dict(self, data: dict) -> Scenario:
        """
        Load a scenario from a dictionary.
        
        Args:
            data: Dictionary containing scenario data
            
        Returns:
            Scenario object
        """
        # Handle nested state_queue
        if "state_queue" in data and isinstance(data["state_queue"], dict):
            data = data.copy()
            sq_data = _convert_state_queue_data(data["state_queue"])
            data["state_queue"] = StateQueue(**sq_data)
        
        return Scenario(**data)
    
    def load_from_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        domain_filter: Optional[str] = None
    ) -> List[Scenario]:
        """
        Load all scenarios from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
            domain_filter: Optional domain to filter by
            
        Returns:
            List of Scenario objects
        """
        directory_path = self._resolve_path(directory_path)
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        scenarios = []
        pattern = "**/*.json" if recursive else "*.json"
        
        for json_file in directory_path.glob(pattern):
            try:
                scenario = self.load_from_file(json_file)
                
                # Apply domain filter if specified
                if domain_filter and scenario.domain != domain_filter:
                    continue
                
                scenarios.append(scenario)
                logger.debug(f"Loaded scenario: {scenario.scenario_id}")
                
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(scenarios)} scenarios from {directory_path}")
        return scenarios
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path relative to base_path if not absolute"""
        path = Path(path)
        if not path.is_absolute():
            path = self.base_path / path
        return path


# Convenience functions

def load_scenario_from_json(file_path: Union[str, Path]) -> Scenario:
    """
    Load a single scenario from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Scenario object
    """
    loader = ScenarioLoader()
    return loader.load_from_file(file_path)


def load_scenarios_from_directory(
    directory_path: Union[str, Path],
    recursive: bool = True,
    domain_filter: Optional[str] = None
) -> List[Scenario]:
    """
    Load all scenarios from a directory.
    
    Args:
        directory_path: Path to the directory
        recursive: Whether to search recursively
        domain_filter: Optional domain to filter by
        
    Returns:
        List of Scenario objects
    """
    loader = ScenarioLoader()
    return loader.load_from_directory(
        directory_path,
        recursive=recursive,
        domain_filter=domain_filter
    )


def create_scenario_from_dict(data: dict) -> Scenario:
    """
    Create a scenario from a dictionary.
    
    This is useful for programmatic scenario creation.
    
    Args:
        data: Dictionary with scenario data
        
    Returns:
        Scenario object
    """
    loader = ScenarioLoader()
    return loader.load_from_dict(data)
