"""
Project: Studojo Agents
File: src/tools/assignment/update_section_content.py
Description: Tool for updating/replacing content for a specific section.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UpdateSectionResult(BaseModel):
    """Result of section content update."""

    section_id: str = Field(description="ID of the updated section")
    success: bool = Field(description="Whether update was successful")
    message: str = Field(description="Status message")
    error: Optional[str] = Field(None, description="Error message if update failed")


@tool
def update_section_content(
    section_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> UpdateSectionResult:
    """
    Update or replace content for a specific section.

    This tool allows the agent to iterate and refine section content.
    Useful for editing generated content or replacing it with improved versions.

    Args:
        section_id: ID of the section to update (e.g., "1", "2.1")
        content: New content text for the section
        metadata: Optional metadata dict with additional info:
            - title: Section title
            - graphs: List of graph dicts
            - references: List of reference dicts
            - humanized: Whether content is humanized
            - etc.

    Returns:
        UpdateSectionResult with update status
    """
    try:
        if not section_id or not section_id.strip():
            return UpdateSectionResult(
                section_id="",
                success=False,
                message="",
                error="Section ID is required",
            )

        if not content or not content.strip():
            return UpdateSectionResult(
                section_id=section_id,
                success=False,
                message="",
                error="Content cannot be empty",
            )

        # TODO: In a real implementation, this would update persistent storage
        # For now, this is a placeholder that just validates the input
        # The agent should maintain state or use a session management system

        logger.info(f"Updating section {section_id} (content length: {len(content)} chars)")

        metadata_info = ""
        if metadata:
            metadata_info = f" with metadata: {list(metadata.keys())}"

        return UpdateSectionResult(
            section_id=section_id,
            success=True,
            message=f"Section {section_id} content updated successfully{metadata_info}",
            error=None,
        )

    except Exception as e:
        logger.error(f"Error updating section content: {e}", exc_info=True)
        return UpdateSectionResult(
            section_id=section_id,
            success=False,
            message="",
            error=f"Error updating section content: {str(e)}",
        )

