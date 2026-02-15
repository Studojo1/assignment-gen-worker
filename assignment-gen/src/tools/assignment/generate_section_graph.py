"""
Project: Studojo Agents
File: src/tools/assignment/generate_section_graph.py
Description: Tool for generating graphs for a single section.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SectionGraphResult(BaseModel):
    """Result of section graph generation."""

    graph_path: str = Field(description="Path to the generated graph image file")
    description: str = Field(description="Description/title of the graph")
    success: bool = Field(description="Whether generation was successful")
    error: Optional[str] = Field(None, description="Error message if generation failed")


@tool
def generate_section_graph(
    content: str,
    section_title: str,
    web_sources: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
) -> SectionGraphResult:
    """
    Generate a graph for a specific section's content.

    This tool analyzes the section content and generates a relevant graph if appropriate.
    The agent can decide which sections need graphs based on the content.

    Args:
        content: The section content text to analyze
        section_title: Title of the section (for context and filename)
        web_sources: Optional list of web search results for additional data
        output_dir: Optional directory for output (defaults to DOC_OUTPUT_DIR env var)

    Returns:
        SectionGraphResult with graph path and description, or error if generation failed
    """
    try:
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        # Add paths for imports
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root / "src") not in sys.path:
            sys.path.insert(0, str(project_root / "src"))

        from tools.common.graph import (
            analyze_content_for_graphs,
            generate_content_aware_graph,
            set_agent_llm_for_graph,
        )
        from chains import build_shared_llm

        # Get LLM instance from global (set by agent)
        # We need to check if it's available
        try:
            from tools.assignment.generate_assignment import _agent_llm

            if _agent_llm is None:
                return SectionGraphResult(
                    graph_path="",
                    description="",
                    success=False,
                    error="LLM instance not available. Please ensure the agent is properly initialized.",
                )

            shared_llm = build_shared_llm(_agent_llm)
            set_agent_llm_for_graph(shared_llm)
        except ImportError:
            return SectionGraphResult(
                graph_path="",
                description="",
                success=False,
                error="Could not access agent LLM instance.",
            )

        if not content or len(content.strip()) < 100:
            return SectionGraphResult(
                graph_path="",
                description="",
                success=False,
                error="Content is too short (less than 100 characters). Need more content to generate a meaningful graph.",
            )

        # Determine output directory
        if not output_dir:
            output_dir = os.getenv("DOC_OUTPUT_DIR", "generated_docs")
        os.makedirs(output_dir, exist_ok=True)

        # Analyze content to find relevant graphs
        logger.info(f"Analyzing content for section '{section_title}' to find relevant graphs...")
        graph_specs = await analyze_content_for_graphs(
            content,
            max_graphs=1,
            web_sources=web_sources,
            llm=shared_llm,
        )

        if not graph_specs:
            return SectionGraphResult(
                graph_path="",
                description="",
                success=False,
                error="No relevant graphs found in content. The content may not contain sufficient numerical data or trends suitable for visualization.",
            )

        spec_obj = graph_specs[0]
        logger.info(
            f"Found relevant graph for section '{section_title}': {spec_obj.title}"
        )

        # Generate the graph
        graph_filename = f"graph_{section_title.replace(' ', '_')}_{int(time.time())}.png"
        graph_output_path = os.path.join(os.path.abspath(output_dir), graph_filename)

        # Run async function
        import asyncio
        result = asyncio.run(
            generate_content_aware_graph.coroutine(
                content=content,
                section_title=section_title,
                output_filename=graph_output_path,
                web_sources=web_sources,
            )
        )

        if result.success and result.image_path and os.path.exists(result.image_path):
            graph_path = os.path.abspath(result.image_path)
            logger.info(
                f"Successfully generated graph for section '{section_title}': {graph_path}"
            )
            return SectionGraphResult(
                graph_path=graph_path,
                description=spec_obj.title,
                success=True,
                error=None,
            )
        else:
            return SectionGraphResult(
                graph_path="",
                description="",
                success=False,
                error=result.error or "Failed to generate graph image",
            )

    except Exception as e:
        logger.error(f"Error generating section graph: {e}", exc_info=True)
        return SectionGraphResult(
            graph_path="",
            description="",
            success=False,
            error=f"Error generating section graph: {str(e)}",
        )

