"""
Project: Studojo Agents
File: src/tools/uniqueness.py
Author: Rithul Kamesh
Created: 17 November 2025
Description: Embeddings-based uniqueness checker using OpenAI embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from langchain_openai import OpenAIEmbeddings
from numpy import ndarray
import numpy as np


@dataclass
class UniquenessResult:
    score: float
    """1 - max cosine similarity. Higher is more unique (0–1)."""

    max_similarity: float
    """Maximum cosine similarity to any item in corpus."""

    similar_indices: List[int]
    """Indices of corpus items that exceed the configured threshold."""


class UniquenessChecker:
    """
    Simple embeddings-based uniqueness checker.

    This is intentionally minimal and in-memory for now; later we can
    plug in a persistent vector store.
    """

    def __init__(self, embeddings: OpenAIEmbeddings | None = None) -> None:
        self.embeddings = embeddings or OpenAIEmbeddings()

    def _embed(self, texts: Sequence[str]) -> ndarray:
        vectors = self.embeddings.embed_documents(list(texts))
        return np.array(vectors, dtype=float)

    def _cosine_similarities(self, v: ndarray, matrix: ndarray) -> ndarray:
        if matrix.size == 0:
            return np.array([], dtype=float)

        v_norm = np.linalg.norm(v)
        m_norms = np.linalg.norm(matrix, axis=1)
        # Avoid division by zero
        v_norm = v_norm if v_norm != 0 else 1e-8
        m_norms = np.where(m_norms == 0, 1e-8, m_norms)

        sims = matrix @ v / (m_norms * v_norm)
        return sims

    def check_uniqueness(
        self,
        draft: str,
        corpus: Iterable[str],
        similarity_threshold: float = 0.85,
    ) -> UniquenessResult:
        """
        Compare a draft against a corpus and return a uniqueness score.

        - similarity_threshold: above this cosine similarity, we consider items
          "too similar" and list their indices.
        """

        corpus_list = list(corpus)
        if not corpus_list:
            return UniquenessResult(
                score=1.0,
                max_similarity=0.0,
                similar_indices=[],
            )

        all_texts = [draft, *corpus_list]
        matrix = self._embed(all_texts)
        draft_vec = matrix[0]
        corpus_vecs = matrix[1:]

        sims = self._cosine_similarities(draft_vec, corpus_vecs)
        max_sim = float(np.max(sims)) if sims.size > 0 else 0.0

        similar_indices = [i for i, s in enumerate(sims) if s >= similarity_threshold]

        uniqueness_score = float(1.0 - max_sim)

        return UniquenessResult(
            score=uniqueness_score,
            max_similarity=max_sim,
            similar_indices=similar_indices,
        )
