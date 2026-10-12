"""
Project: Studojo Assignment Generator
File: src/chains/azure_openai.py
Description: Azure OpenAI chat model and embeddings shared by the worker scripts and tools.

Every OpenAI model call goes to Azure OpenAI; the direct OpenAI key is retired.
Configuration comes from the environment:
- AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY: required
- AZURE_OPENAI_MINI_DEPLOYMENT: chat deployment (default "gpt-4o-mini")
- AZURE_OPENAI_EMBEDDING_DEPLOYMENT: embeddings deployment (default "text-embedding-3-small")
- OPENAI_API_VERSION: API version (default "2024-10-21")
"""

from __future__ import annotations

import os
from typing import Any, Dict

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

REQUIRED_ENV_VARS = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY")


def require_azure_openai_env() -> None:
    """Raise ValueError if the Azure OpenAI endpoint or key is not set."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        noun = "variable" if len(missing) == 1 else "variables"
        raise ValueError(f"{' and '.join(missing)} environment {noun} not set")


def _client_kwargs() -> Dict[str, Any]:
    # The key is passed explicitly: left unset, langchain_openai falls back to
    # OPENAI_API_KEY, the retired direct key.
    require_azure_openai_env()
    return {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "api_version": os.getenv("OPENAI_API_VERSION") or "2024-10-21",
    }


def build_azure_chat_llm(temperature: float) -> AzureChatOpenAI:
    """Chat model on the gpt-4o-mini deployment."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT") or "gpt-4o-mini",
        temperature=temperature,
        **_client_kwargs(),
    )


def build_azure_embeddings() -> AzureOpenAIEmbeddings:
    """Embeddings on the text-embedding-3-small deployment."""
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        or "text-embedding-3-small",
        **_client_kwargs(),
    )
