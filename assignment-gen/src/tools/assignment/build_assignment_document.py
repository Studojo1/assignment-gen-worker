"""
Project: Studojo Agents
File: src/tools/assignment/build_assignment_document.py
Description: Tool for building final assignment document from assembled sections.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentBuildResult(BaseModel):
    """Result of document building."""

    doc_path: str = Field(description="Path to the generated document file")
    success: bool = Field(description="Whether document building was successful")
    num_sections: int = Field(description="Number of sections included in the document")
    error: Optional[str] = Field(None, description="Error message if building failed")


@tool
def build_assignment_document(
    sections: List[Dict[str, Any]],
    title: str,
    format_style: Optional[str] = None,
    output_filename: str = "assignment.docx",
    enable_title_page: bool = True,
    enable_borders: bool = True,
    enable_page_numbers: bool = True,
) -> DocumentBuildResult:
    """
    Build a final assignment document from assembled sections.

    This tool takes a list of sections (with content, graphs, references) and
    assembles them into a formatted Word document (.docx).

    Args:
        sections: List of section dictionaries, each with:
            - section: Section ID (e.g., "1")
            - subsection: Optional subsection ID (e.g., "1.1")
            - title: Section/subsection title
            - content: Body text content
            - graphs: Optional list of graph dicts with "path" and "description"
            - references: Optional list of reference dicts
        title: Overall document title
        format_style: Citation style (IEEE, APA, MLA, Chicago, etc.)
        output_filename: Desired filename for the document
        enable_title_page: Whether to include a title page
        enable_borders: Whether to draw borders around pages
        enable_page_numbers: Whether to add page numbers

    Returns:
        DocumentBuildResult with document path and metadata
    """
    try:
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        # Add paths for imports
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root / "src") not in sys.path:
            sys.path.insert(0, str(project_root / "src"))

        from tools.common.doc import generate_formatted_document

        if not sections:
            return DocumentBuildResult(
                doc_path="",
                success=False,
                num_sections=0,
                error="No sections provided. Cannot build document without sections.",
            )

        if not title or not title.strip():
            return DocumentBuildResult(
                doc_path="",
                success=False,
                num_sections=0,
                error="Document title is required.",
            )

        # Validate sections structure
        validated_sections = []
        for i, section in enumerate(sections):
            if not isinstance(section, dict):
                logger.warning(f"Section {i} is not a dictionary, skipping")
                continue

            # Ensure required fields exist
            if "content" not in section or not section.get("content"):
                logger.warning(f"Section {i} has no content, skipping")
                continue

            # Ensure section has at least section ID or title
            if "section" not in section and "title" not in section:
                logger.warning(f"Section {i} has no section ID or title, skipping")
                continue

            validated_sections.append(section)

        if not validated_sections:
            return DocumentBuildResult(
                doc_path="",
                success=False,
                num_sections=0,
                error="No valid sections found. All sections must have content and at least a section ID or title.",
            )

        logger.info(f"Building document '{title}' with {len(validated_sections)} sections...")

        # Build the document using the existing tool
        result = generate_formatted_document.func(
            title=title,
            sections=validated_sections,
            output_filename=output_filename,
            enable_title_page=enable_title_page,
            enable_borders=enable_borders,
            enable_page_numbers=enable_page_numbers,
            format_style=format_style,
        )

        logger.info(f"Document built successfully: {result.file_path}")

        return DocumentBuildResult(
            doc_path=result.file_path,
            success=True,
            num_sections=result.num_sections,
            error=None,
        )

    except Exception as e:
        logger.error(f"Error building assignment document: {e}", exc_info=True)
        return DocumentBuildResult(
            doc_path="",
            success=False,
            num_sections=0,
            error=f"Error building document: {str(e)}",
        )

