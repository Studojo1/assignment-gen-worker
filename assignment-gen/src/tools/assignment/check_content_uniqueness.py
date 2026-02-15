"""
Project: Studojo Agents
File: src/tools/assignment/check_content_uniqueness.py
Description: Tool for checking content uniqueness against a corpus.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UniquenessCheckResult(BaseModel):
    """Result of uniqueness check."""

    uniqueness_score: float = Field(
        description="Uniqueness score (0-1, higher is more unique)"
    )
    max_similarity: float = Field(
        description="Maximum similarity to any item in corpus (0-1)"
    )
    similar_indices: List[int] = Field(
        default_factory=list,
        description="Indices of corpus items that exceed similarity threshold",
    )
    success: bool = Field(description="Whether check was successful")
    error: Optional[str] = Field(None, description="Error message if check failed")


@tool
def check_content_uniqueness(
    content: str,
    corpus_texts: Optional[List[str]] = None,
    similarity_threshold: float = 0.85,
) -> UniquenessCheckResult:
    """
    Check the uniqueness of content against a corpus of texts.

    This tool uses embeddings-based similarity to detect if content is too similar
    to existing texts. Useful for checking before/after humanization or against
    uploaded files.

    Args:
        content: The content text to check for uniqueness
        corpus_texts: List of texts to compare against (e.g., uploaded files, previous assignments)
        similarity_threshold: Threshold above which content is considered "too similar" (default: 0.85)

    Returns:
        UniquenessCheckResult with uniqueness score and similarity details
    """
    try:
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        # Add paths for imports
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root / "src") not in sys.path:
            sys.path.insert(0, str(project_root / "src"))

        from tools.common.uniqueness import UniquenessChecker

        if not content or not content.strip():
            return UniquenessCheckResult(
                uniqueness_score=1.0,
                max_similarity=0.0,
                similar_indices=[],
                success=False,
                error="Content is empty",
            )

        # If no corpus provided, content is considered unique
        if not corpus_texts or len(corpus_texts) == 0:
            logger.info("No corpus provided, content is considered unique")
            return UniquenessCheckResult(
                uniqueness_score=1.0,
                max_similarity=0.0,
                similar_indices=[],
                success=True,
                error=None,
            )

        # Filter out empty corpus texts
        valid_corpus = [text for text in corpus_texts if text and text.strip()]
        if not valid_corpus:
            logger.info("No valid corpus texts, content is considered unique")
            return UniquenessCheckResult(
                uniqueness_score=1.0,
                max_similarity=0.0,
                similar_indices=[],
                success=True,
                error=None,
            )

        # Create uniqueness checker
        checker = UniquenessChecker()

        # Check uniqueness
        logger.info(
            f"Checking uniqueness of content ({len(content)} chars) against corpus ({len(valid_corpus)} texts)..."
        )
        result = checker.check_uniqueness(
            draft=content,
            corpus=valid_corpus,
            similarity_threshold=similarity_threshold,
        )

        logger.info(
            f"Uniqueness check complete: score={result.score:.3f}, max_similarity={result.max_similarity:.3f}"
        )

        return UniquenessCheckResult(
            uniqueness_score=result.score,
            max_similarity=result.max_similarity,
            similar_indices=result.similar_indices,
            success=True,
            error=None,
        )

    except Exception as e:
        logger.error(f"Error checking content uniqueness: {e}", exc_info=True)
        return UniquenessCheckResult(
            uniqueness_score=0.0,
            max_similarity=1.0,
            similar_indices=[],
            success=False,
            error=f"Error checking uniqueness: {str(e)}",
        )

