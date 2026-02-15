"""
Core LangChain building blocks (shared LLM + utility chains).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

# Export utility chains
from chains.utils import (
    build_numerical_data_extraction_chain,
    build_file_summarization_chain,
)

__all__ = [
    "build_shared_llm",
    "build_numerical_data_extraction_chain",
    "build_file_summarization_chain",
]


def build_shared_llm(llm: BaseChatModel | None = None) -> BaseChatModel:
    """
    Return the shared chat model instance.

    For now we accept an optional pre-configured `llm` so that:
    - CLI / tests can inject a fake or mocked model.
    - Callers can provide their own configured model instance.

    If `llm` is None, callers are expected to construct one themselves.
    """

    if llm is None:
        raise ValueError(
            "No LLM instance provided. "
            "Construct a BaseChatModel (e.g., ChatOpenAI) and pass it in."
        )
    return llm
