"""
Project: Studojo Agents
File: src/tools/generate_assignment.py
Description: Tool wrapper around assignment pipeline for agent use.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AssignmentGenerationResult(BaseModel):
    """Result of assignment generation."""

    success: bool = Field(description="Whether generation was successful")
    doc_path: Optional[str] = Field(
        default=None, description="Path to generated document (local)"
    )
    message: str = Field(description="Human-readable message about the result")
    uniqueness_score: Optional[float] = Field(
        default=None, description="Uniqueness score if available"
    )
    s3_key: Optional[str] = Field(
        default=None, description="S3 key of the uploaded document"
    )
    download_url: Optional[str] = Field(
        default=None, description="Presigned download URL for the document"
    )


# Global variable to store LLM instance - will be set by agent
_agent_llm: Optional[BaseChatModel] = None


def set_agent_llm(llm: BaseChatModel) -> None:
    """Set the LLM instance for the agent tools."""
    global _agent_llm
    _agent_llm = llm
    # Also set for file summarization
    from tools.common.summarize_file import set_agent_llm_for_summarize

    set_agent_llm_for_summarize(llm)
    # Also set for graph tools
    from tools.common.graph import set_agent_llm_for_graph

    set_agent_llm_for_graph(llm)


@tool
def generate_assignment(
    user_id: str = "",
    output_filename: str = "assignment.docx",
    force_regenerate_outline: bool = False,
    user_message: str = "",
) -> AssignmentGenerationResult:
    """
    Generate a complete assignment document based on the user's assignment specification.

    This is a high-level orchestrator tool that handles the entire assignment generation
    pipeline in one call. For more control, agents can use the operation-level tools:
    - generate_assignment_outline: Generate outline structure
    - generate_section_content: Generate content for individual sections
    - generate_section_graph: Generate graphs for specific sections
    - humanize_section_content: Humanize section content
    - check_content_uniqueness: Check content uniqueness
    - build_assignment_document: Assemble sections into final document

    This tool will:
    1. Check if the assignment spec is ready (has all required information)
    2. If user_message is provided, it will try to update the spec first
    3. Generate an outline if one doesn't exist
    4. Generate the full assignment document
    5. Upload the document to S3
    6. Return the download URL

    Args:
        user_id: User ID to access their session and assignment spec.
        output_filename: Desired filename for the generated document (default: "assignment.docx").
        force_regenerate_outline: Whether to regenerate the outline even if one exists (default: False).
        user_message: Optional message from the user containing assignment details. If provided,
            the tool will try to update the assignment spec with this information before generating.

    Returns:
        AssignmentGenerationResult with success status, download URL, and message.
    """
    try:
        # Import here to avoid circular dependencies
        import sys
        import os
        from pathlib import Path

        # Add paths for imports
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root / "src") not in sys.path:
            sys.path.insert(0, str(project_root / "src"))

        from models.assignment import AssignmentSpec, AssignmentStatus
        from chains.assignment import (
            build_assignment_outline_chain,
            build_assignment_pipeline,
            build_assignment_input_collector_chain,
        )
        from chains import build_shared_llm

        # For barebones app, assignment_spec should be passed in or created fresh
        # This is a simplified version - in a real implementation, assignment_spec
        # would be passed as a parameter or loaded from storage
        if not user_id:
            user_id = os.getenv("CURRENT_USER_ID", "default_user")
        
        # Create a fresh assignment spec if needed
        # In a real implementation, this would come from storage or be passed in
        assignment_spec = AssignmentSpec()

        # Get LLM instance
        if _agent_llm is None:
            return AssignmentGenerationResult(
                success=False,
                message="LLM instance not available. Please ensure the agent is properly initialized.",
            )

        shared_llm = build_shared_llm(_agent_llm)

        # Process user message if provided to update spec
        if user_message:
            logger.info(f"Processing user message in generate_assignment: {user_message[:50]}...")
            try:
                # Build assignment input collector chain
                assignment_chain = build_assignment_input_collector_chain(shared_llm)

                # Collect assignment input
                collector_output = asyncio.run(
                    assignment_chain.ainvoke(
                        {
                            "user_message": user_message,
                            "assignment_spec": session.assignment_spec,
                        }
                    )
                )

                # Update assignment spec from collector output
                assignment_spec = collector_output.assignment_spec
                logger.info("Updated assignment spec from user message")
            except Exception as e:
                logger.warning(f"Failed to process user message in generate_assignment: {e}")

        # Check if spec is ready
        if assignment_spec.status != AssignmentStatus.READY:
            # Auto-finalization check: if we have the basics, force it to READY
            # This handles cases where the agent calls generate but the spec isn't explicitly marked READY yet
            has_basics = (
                assignment_spec.assignment_type
                and assignment_spec.description
                and (assignment_spec.length_words or assignment_spec.length_pages)
                and assignment_spec.format_style
            )
            
            if has_basics:
                logger.info("Spec has basic fields, forcing status to READY for generation")
                assignment_spec.status = AssignmentStatus.READY
            else:
                missing = []
                if not assignment_spec.assignment_type: missing.append("assignment_type")
                if not assignment_spec.description: missing.append("description")
                if not (assignment_spec.length_words or assignment_spec.length_pages): missing.append("length")
                if not assignment_spec.format_style: missing.append("format_style")
                
                return AssignmentGenerationResult(
                    success=False,
                    message=(
                        f"Assignment spec is not ready yet (status: {assignment_spec.status}). "
                        f"Missing fields: {', '.join(missing)}. "
                        "Please provide these details first."
                    ),
                )

        # Generate outline (no session storage in barebones app)
        assignment_outline = None
        if force_regenerate_outline or True:  # Always generate for barebones app
            logger.info("Generating assignment outline...")
            outline_chain = build_assignment_outline_chain(shared_llm)
            assignment_outline = outline_chain.invoke(assignment_spec)
            logger.info("Assignment outline generated successfully")

        # Build assignment pipeline
        assignment_pipeline = build_assignment_pipeline(shared_llm)

        # Combine description with uploaded file content for courseplan_analysis
        courseplan_analysis = assignment_spec.description or ""
        if assignment_spec.uploaded_files:
            file_contents = []
            for file_info in assignment_spec.uploaded_files:
                if file_info.get("extracted_text"):
                    file_contents.append(
                        f"\n\n--- Content from {file_info.get('filename', 'uploaded file')} ---\n"
                        f"{file_info.get('extracted_text', '')}"
                    )
            if file_contents:
                courseplan_analysis += "\n\n" + "\n".join(file_contents)

        # Use uploaded file content as corpus for uniqueness checking
        corpus_texts = []
        if assignment_spec.uploaded_files:
            corpus_texts = [
                file_info.get("extracted_text", "")
                for file_info in assignment_spec.uploaded_files
                if file_info.get("extracted_text")
            ]

        # Run the pipeline
        logger.info("Starting assignment generation pipeline...")
        try:
            result = asyncio.run(
                assignment_pipeline(
                    assignment_spec,
                    outline=assignment_outline,
                    courseplan_analysis=courseplan_analysis,
                    user_responses="",
                    corpus_texts=corpus_texts,
                    output_filename=output_filename,
                )
            )

            uniqueness_score = None
            if result.uniqueness:
                uniqueness_score = result.uniqueness.score

            # For barebones app, just return local file path
            # S3 upload would be handled by caller if needed
            msg = f"Assignment generated successfully at: {result.doc_path}"

            return AssignmentGenerationResult(
                success=True,
                doc_path=result.doc_path,
                message=msg,
                uniqueness_score=uniqueness_score,
                s3_key=None,
                download_url=None
            )
        except Exception as e:
            logger.error(f"Error in assignment pipeline: {e}", exc_info=True)
            return AssignmentGenerationResult(
                success=False,
                message=f"Assignment generation failed: {str(e)}",
            )

    except Exception as e:
        logger.error(f"Error generating assignment: {e}", exc_info=True)
        return AssignmentGenerationResult(success=False, message=f"Error: {str(e)}")


@tool
def collect_assignment_input(
    user_message: str,
    user_id: str = "",
) -> str:
    """
    Collect and update assignment specification based on user input.

    This tool helps gather information about the assignment the user wants to create.
    It will ask follow-up questions if needed, or confirm when the spec is complete.

    Args:
        user_message: The user's message about their assignment.
        user_id: User ID to access their session and assignment spec.

    Returns:
        Assistant's response message, which may include questions or confirmations.
    """
    try:
        # Import here to avoid circular dependencies
        import sys
        import os
        from pathlib import Path

        # Add paths for imports
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root / "src") not in sys.path:
            sys.path.insert(0, str(project_root / "src"))

        from models.assignment import AssignmentSpec
        from chains.assignment import build_assignment_input_collector_chain
        from chains import build_shared_llm

        # Get LLM instance
        if _agent_llm is None:
            return "Error: LLM instance not available"

        shared_llm = build_shared_llm(_agent_llm)

        # Create a fresh assignment spec (in real implementation, this would come from storage)
        assignment_spec = AssignmentSpec()

        # Build assignment input collector chain
        assignment_chain = build_assignment_input_collector_chain(shared_llm)

        # Collect assignment input
        collector_output = asyncio.run(
            assignment_chain.ainvoke(
                {
                    "user_message": user_message,
                    "assignment_spec": assignment_spec,
                }
            )
        )

        return collector_output.assistant_message

    except Exception as e:
        logger.error(f"Error collecting assignment input: {e}", exc_info=True)
        return f"Error collecting assignment information: {str(e)}"
