"""
Project: Studojo Agents
File: src/tools/assignment/humanize_section_content.py
Description: Tool for humanizing a single section's content using Rephrasy.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HumanizeResult(BaseModel):
    """Result of content humanization."""

    humanized_content: str = Field(description="Humanized content (or original if humanization failed)")
    success: bool = Field(description="Whether humanization was successful")
    used_fallback: bool = Field(description="Whether original text was used as fallback")
    error: Optional[str] = Field(None, description="Error message if humanization failed")


@tool
def humanize_section_content(
    content: str,
    order_id: Optional[str] = None,
) -> HumanizeResult:
    """
    Humanize a single section's content using Rephrasy.

    This tool makes the content more natural and less AI-detectable.
    If humanization fails or returns foreign language, it falls back to the original text.

    Args:
        content: The section content text to humanize
        order_id: Optional order ID for tracking (e.g., assignment_id)

    Returns:
        HumanizeResult with humanized content or original if humanization failed
    """
    try:
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        # Add paths for imports
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root / "src") not in sys.path:
            sys.path.insert(0, str(project_root / "src"))

        from tools.common.rephrasy import RephrasyService
        from chains import build_shared_llm

        if not content or not content.strip():
            return HumanizeResult(
                humanized_content=content,
                success=False,
                used_fallback=True,
                error="Content is empty",
            )

        # Get LLM instance for language validation (if available)
        llm = None
        try:
            from tools.assignment.generate_assignment import _agent_llm

            if _agent_llm is not None:
                llm = build_shared_llm(_agent_llm)
        except ImportError:
            logger.warning("Could not access agent LLM for language validation")

        # Create Rephrasy service
        rephrasy_service = RephrasyService()

        # Humanize the content
        logger.info(f"Humanizing content (length: {len(content)} chars)...")
        try:
            humanized_content = rephrasy_service.humanize_text_or_fallback(
                content, order_id=order_id, llm=llm
            )

            # Check if fallback was used (content unchanged)
            used_fallback = humanized_content == content

            if used_fallback:
                logger.info("Humanization failed or was skipped, using original content")
            else:
                logger.info("Content successfully humanized")

            return HumanizeResult(
                humanized_content=humanized_content,
                success=not used_fallback,
                used_fallback=used_fallback,
                error=None,
            )

        except Exception as e:
            logger.warning(f"Humanization failed: {e}, using original content")
            return HumanizeResult(
                humanized_content=content,
                success=False,
                used_fallback=True,
                error=f"Humanization failed: {str(e)}",
            )

    except Exception as e:
        logger.error(f"Error in humanize_section_content: {e}", exc_info=True)
        return HumanizeResult(
            humanized_content=content,
            success=False,
            used_fallback=True,
            error=f"Error: {str(e)}",
        )

