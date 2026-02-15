"""
Project: Studojo Agents
File: src/tools/summarize_file.py
Description: Tool to summarize and read uploaded files from session.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from chains import build_shared_llm
from chains.utils import build_file_summarization_chain

logger = logging.getLogger(__name__)

# Global variable to store LLM instance - will be set by agent
_agent_llm: Optional[BaseChatModel] = None


class FileSummary(BaseModel):
    """Summary of a file."""

    filename: str = Field(description="Name of the file")
    file_type: str = Field(description="Type of file (pdf, docx, txt, image, etc.)")
    content_preview: str = Field(
        description="Preview of file content (first 1000 chars)"
    )
    full_content_length: int = Field(
        description="Total length of file content in characters"
    )
    summary: Optional[str] = Field(
        default=None, description="AI-generated summary of the file content"
    )


def set_agent_llm_for_summarize(llm: BaseChatModel) -> None:
    """Set the LLM instance for file summarization."""
    global _agent_llm
    _agent_llm = llm


@tool
def summarize_file(
    filename: Optional[str] = None,
    user_id: str = "",
    max_preview_length: int = 1000,
) -> List[FileSummary]:
    """
    Summarize and read uploaded files from the user's session.

    If filename is provided, returns summary for that specific file.
    If filename is None, returns summaries for all uploaded files.

    Args:
        filename: Optional specific filename to summarize. If None, summarizes all files.
        user_id: User ID to access their session and uploaded files.
        max_preview_length: Maximum length of content preview to return (default: 1000 chars).

    Returns:
        List of FileSummary objects containing file information and content preview.
    """
    try:
        # For barebones app, uploaded_files should be passed in or empty
        # In a real implementation, this would come from storage
        uploaded_files = []

        if not uploaded_files:
            logger.info(f"No uploaded files found for user {user_id}")
            return []

        # Filter by filename if provided
        if filename:
            uploaded_files = [
                f for f in uploaded_files if f.get("filename") == filename
            ]
            if not uploaded_files:
                logger.info(f"File '{filename}' not found in uploaded files")
                return []

        # Build summaries
        summaries = []
        for file_info in uploaded_files:
            file_name = file_info.get("filename", "Unknown")
            file_type = file_info.get("file_type", "unknown")
            extracted_text = file_info.get("extracted_text", "")

            # Create preview
            content_preview = extracted_text[:max_preview_length]
            if len(extracted_text) > max_preview_length:
                content_preview += "..."

            # Generate AI summary if LLM is available
            ai_summary = None
            if _agent_llm and extracted_text:
                try:
                    shared_llm = build_shared_llm(_agent_llm)
                    summarization_chain = build_file_summarization_chain(shared_llm)
                    summary_input = {
                        "filename": file_name,
                        "file_type": file_type,
                        "content": extracted_text,
                        "max_length": 300,  # Keep summaries concise
                    }
                    import asyncio

                    ai_summary = asyncio.run(summarization_chain.ainvoke(summary_input))
                    logger.info(
                        f"Generated AI summary for {file_name} ({len(ai_summary)} chars)"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to generate AI summary for {file_name}: {e}"
                    )
                    # Continue without summary

            summaries.append(
                FileSummary(
                    filename=file_name,
                    file_type=file_type,
                    content_preview=content_preview,
                    full_content_length=len(extracted_text),
                    summary=ai_summary,
                )
            )

        logger.info(f"Summarized {len(summaries)} file(s) for user {user_id}")
        return summaries

    except Exception as e:
        logger.error(f"Error summarizing files: {e}", exc_info=True)
        return []


@tool
def read_file_content(
    filename: str,
    user_id: str = "",
) -> str:
    """
    Read the full content of a specific uploaded file from the user's session.

    Args:
        filename: Name of the file to read.
        user_id: User ID to access their session and uploaded files.

    Returns:
        Full text content of the file, or empty string if file not found.
    """
    try:
        # For barebones app, uploaded_files should be passed in or empty
        # In a real implementation, this would come from storage
        uploaded_files = []
        for file_info in uploaded_files:
            if file_info.get("filename") == filename:
                return file_info.get("extracted_text", "")

        logger.warning(f"File '{filename}' not found for user {user_id}")
        return ""

    except Exception as e:
        logger.error(f"Error reading file content: {e}", exc_info=True)
        return ""
