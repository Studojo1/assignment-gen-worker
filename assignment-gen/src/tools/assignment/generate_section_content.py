"""
Project: Studojo Agents
File: src/tools/assignment/generate_section_content.py
Description: Tool for generating content for a single section/subsection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global variable to store LLM instance - will be set by agent
_agent_llm: Optional[BaseChatModel] = None


def set_agent_llm_for_section_content(llm: BaseChatModel) -> None:
    """Set the LLM instance for section content generation."""
    global _agent_llm
    _agent_llm = llm


class SectionContentResult(BaseModel):
    """Result of section content generation."""

    section_id: str = Field(description="ID of the section")
    content: str = Field(description="Generated content for the main section")
    subsections: Dict[str, str] = Field(
        default_factory=dict, description="Generated content for subsections (key -> content)"
    )
    success: bool = Field(description="Whether generation was successful")
    error: Optional[str] = Field(None, description="Error message if generation failed")


@tool
def generate_section_content(
    spec: Dict[str, Any],
    section_id: str,
    section_title: str,
    target_words: int,
    outline: Optional[Dict[str, Any]] = None,
    web_sources: Optional[List[Dict[str, Any]]] = None,
    courseplan_analysis: str = "",
    user_responses: str = "",
    citation_instruction: str = "",
) -> SectionContentResult:
    """
    Generate content for a single section and its subsections.

    This tool generates academically styled content for one section of an assignment.
    It can handle both main sections and subsections based on the outline structure.

    Args:
        spec: Assignment specification dictionary (AssignmentSpec as dict)
        section_id: ID of the section to generate (e.g., "1", "2.1")
        section_title: Title of the section
        target_words: Target word count for this section
        outline: Optional outline dictionary with section structure
        web_sources: Optional list of web search results to use as sources
        courseplan_analysis: Course plan or assignment description context
        user_responses: User-provided responses or additional context
        citation_instruction: Instructions for citation format

    Returns:
        SectionContentResult with generated content for main section and subsections
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
        from chains.assignment.assignment_content import (
            SubsectionSpec,
            build_assignment_content_chain,
        )
        from chains import build_shared_llm

        # Get LLM instance
        if _agent_llm is None:
            return SectionContentResult(
                section_id=section_id,
                content="",
                subsections={},
                success=False,
                error="LLM instance not available. Please ensure the agent is properly initialized.",
            )

        shared_llm = build_shared_llm(_agent_llm)

        # Parse assignment spec
        try:
            assignment_spec = AssignmentSpec.model_validate(spec)
        except Exception as e:
            return SectionContentResult(
                section_id=section_id,
                content="",
                subsections={},
                success=False,
                error=f"Invalid assignment spec: {str(e)}",
            )

        # Build citation instruction if not provided
        if not citation_instruction and assignment_spec.format_style:
            style_lower = assignment_spec.format_style.lower()
            if style_lower == "ieee":
                citation_instruction = (
                    "Cite all sources using IEEE format. Use in-text citations like [1] "
                    "and include full references in the References section."
                )
            elif style_lower == "apa":
                citation_instruction = (
                    "Cite all sources using APA format. Use in-text citations like (Author, Year) "
                    "and include full references in the References section."
                )
            elif style_lower == "mla":
                citation_instruction = (
                    "Cite all sources using MLA format. Use in-text citations like (Author) "
                    "and include full references in the References section."
                )
            elif style_lower == "chicago":
                citation_instruction = (
                    "Cite all sources using Chicago format. Use in-text citations like (Author Year) "
                    "and include full references in the References section."
                )
            else:
                citation_instruction = (
                    f"Cite all sources using {assignment_spec.format_style} format. "
                    "Include full references in the References section."
                )
        elif not citation_instruction:
            citation_instruction = "Cite all sources appropriately and include full references in the References section."

        # Prepare web search block
        web_search_block = ""
        if web_sources:
            web_parts = []
            for i, source in enumerate(web_sources, 1):
                title = source.get("title", "Unknown Title")
                url = source.get("url", "")
                snippet = source.get("snippet", "")
                content = source.get("content", "")
                web_parts.append(
                    f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n"
                    f"Content: {content[:500]}..."
                    if len(content) > 500
                    else f"Content: {content}"
                )
            web_search_block = "\n\n".join(web_parts)

        # Extract subsections from outline if provided
        subsections: List[SubsectionSpec] = []
        if outline and "sections" in outline:
            for section_data in outline["sections"]:
                if str(section_data.get("id", "")) == section_id:
                    subsections_list = section_data.get("subsections", [])
                    for sub in subsections_list:
                        sub_target_words = int(sub.get("target_words", 200))
                        sub_pages = max(0.1, sub_target_words / 300.0)
                        sub_id = str(sub.get("id", ""))
                        subsections.append(
                            {
                                "key": sub_id.replace(".", "_"),
                                "title": str(sub.get("title", "")),
                                "pages": sub_pages,
                            }
                        )
                    break

        # If no subsections found, create a synthetic "main" subsection
        if not subsections:
            section_pages = max(0.1, target_words / 300.0)
            subsections = [
                {"key": "main", "title": section_title, "pages": section_pages}
            ]

        # Build content chain
        content_chain = build_assignment_content_chain(shared_llm)

        # Generate content
        logger.info(f"Generating content for section {section_id}: {section_title}")
        json_content: Dict[str, str] = asyncio.run(
            content_chain.ainvoke(
                {
                    "section_number": section_id,
                    "section_title": section_title,
                    "target_pages": max(0.1, target_words / 300.0),
                    "target_words_total": target_words,
                    "courseplan_analysis": courseplan_analysis or (assignment_spec.description or ""),
                    "user_responses": user_responses,
                    "graph_context_block": "",
                    "web_search_block": web_search_block,
                    "excel_context_snippet": "",
                    "citation_instruction": citation_instruction,
                    "dynamic_forbidden_block": "",
                    "subsections": subsections,
                }
            )
        )

        # Extract main content
        main_content = json_content.get("main", "")

        # Extract subsection content
        subsection_content = {}
        for sub in subsections:
            sub_key = sub["key"]
            sub_content = json_content.get(sub_key, "")
            if sub_content:
                subsection_content[sub_key] = sub_content

        return SectionContentResult(
            section_id=section_id,
            content=main_content,
            subsections=subsection_content,
            success=True,
            error=None,
        )

    except Exception as e:
        logger.error(f"Error generating section content: {e}", exc_info=True)
        return SectionContentResult(
            section_id=section_id,
            content="",
            subsections={},
            success=False,
            error=f"Error generating section content: {str(e)}",
        )

