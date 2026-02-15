"""
Project: Studojo Agents
File: src/tools/assignment/get_assignment_spec.py
Description: Tool for retrieving current assignment specification state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AssignmentSpecResult(BaseModel):
    """Result of getting assignment spec."""

    spec: Dict[str, Any] = Field(description="Assignment specification dictionary")
    success: bool = Field(description="Whether retrieval was successful")
    error: Optional[str] = Field(None, description="Error message if retrieval failed")


@tool
def get_assignment_spec(
    user_id: str = "",
) -> AssignmentSpecResult:
    """
    Get the current assignment specification state.

    This tool retrieves the current assignment spec so the agent can check
    what information has been collected and what sections have been generated.

    Note: In a real implementation, this would load from persistent storage.
    For now, this is a placeholder that returns an empty spec.

    Args:
        user_id: User ID to retrieve spec for (optional, for future use)

    Returns:
        AssignmentSpecResult with current assignment spec
    """
    try:
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        # Add paths for imports
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root / "src") not in sys.path:
            sys.path.insert(0, str(project_root / "src"))

        from models.assignment import AssignmentSpec

        # TODO: In a real implementation, this would load from storage/database
        # For now, return a fresh spec as placeholder
        # The agent should maintain state or use a session management system

        logger.info(f"Getting assignment spec for user_id: {user_id or 'default'}")

        # Create a fresh spec (in real implementation, load from storage)
        assignment_spec = AssignmentSpec()

        return AssignmentSpecResult(
            spec=assignment_spec.model_dump(),
            success=True,
            error=None,
        )

    except Exception as e:
        logger.error(f"Error getting assignment spec: {e}", exc_info=True)
        return AssignmentSpecResult(
            spec={},
            success=False,
            error=f"Error getting assignment spec: {str(e)}",
        )

