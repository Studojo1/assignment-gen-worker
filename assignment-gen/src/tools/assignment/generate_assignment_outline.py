"""
Project: Studojo Agents
File: src/tools/assignment/generate_assignment_outline.py
Description: Tool for generating assignment outline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global variable to store LLM instance - will be set by agent
_agent_llm = None


def set_agent_llm_for_outline(llm) -> None:
    """Set the LLM instance for outline generation."""
    global _agent_llm
    _agent_llm = llm


class OutlineResult(BaseModel):
    """Result of outline generation."""

    outline: Dict[str, Any] = Field(description="Generated outline structure with sections")
    success: bool = Field(description="Whether outline generation was successful")
    error: Optional[str] = Field(None, description="Error message if generation failed")


@tool
def generate_assignment_outline(
    spec: Dict[str, Any],
) -> OutlineResult:
    """
    Generate an outline for an assignment based on the specification.

    This tool creates a structured outline with sections and subsections,
    including target word counts for each. The agent can use this outline
    to then generate content section by section.

    Args:
        spec: Assignment specification dictionary (AssignmentSpec as dict)

    Returns:
        OutlineResult with generated outline structure
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
        from chains.assignment.assignment_outline import build_assignment_outline_chain
        from chains import build_shared_llm

        # Get LLM instance
        if _agent_llm is None:
            # Try to get from global agent LLM
            try:
                from tools.assignment.generate_assignment import _agent_llm as agent_llm

                if agent_llm is None:
                    return OutlineResult(
                        outline={},
                        success=False,
                        error="LLM instance not available. Please ensure the agent is properly initialized.",
                    )
                shared_llm = build_shared_llm(agent_llm)
            except ImportError:
                return OutlineResult(
                    outline={},
                    success=False,
                    error="LLM instance not available.",
                )
        else:
            shared_llm = build_shared_llm(_agent_llm)

        # Parse assignment spec
        try:
            assignment_spec = AssignmentSpec.model_validate(spec)
        except Exception as e:
            return OutlineResult(
                outline={},
                success=False,
                error=f"Invalid assignment spec: {str(e)}",
            )

        # Check if spec has minimum required fields
        if not assignment_spec.description:
            return OutlineResult(
                outline={},
                success=False,
                error="Assignment description is required to generate outline.",
            )

        # Build outline chain
        outline_chain = build_assignment_outline_chain(shared_llm)

        # Generate outline
        logger.info("Generating assignment outline...")
        outline = outline_chain.invoke(assignment_spec)

        if not outline or "sections" not in outline:
            return OutlineResult(
                outline={},
                success=False,
                error="Generated outline is invalid or missing sections.",
            )

        logger.info(f"Outline generated successfully with {len(outline.get('sections', []))} sections")

        return OutlineResult(
            outline=outline,
            success=True,
            error=None,
        )

    except Exception as e:
        logger.error(f"Error generating assignment outline: {e}", exc_info=True)
        return OutlineResult(
            outline={},
            success=False,
            error=f"Error generating outline: {str(e)}",
        )

