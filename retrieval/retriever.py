"""Evidence retrieval using TF-IDF or BM25."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class Retriever:
    """Build and query a TF-IDF or BM25 index over an evidence corpus.

    Args:
        config: Runtime configuration supplying retrieval parameters.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        # Internal index state — populated by build_index() or load_index()
        self._corpus: list[str] = []
        self._index: Any = None  # TfidfVectorizer or BM25Okapi
        self._matrix: Any = None  # sparse TF-IDF matrix (TF-IDF mode only)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, corpus: list[str]) -> None:
        """Index the corpus.  Persists to ``config.index_path`` if set.

        Args:
            corpus: List of evidence sentences to index.
        """
        self._corpus = list(corpus)

        if not corpus:
            logger.warning("build_index called with an empty corpus; index will be empty.")
            self._index = None
            self._matrix = None
            if self._config.index_path:
                self._persist_index()
            return

        if self._config.retrieval_method == "bm25":
            self._build_bm25(corpus)
        else:
            self._build_tfidf(corpus)

        if self._config.index_path:
            self._persist_index()

    def _build_tfidf(self, corpus: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(corpus)
        self._index = vectorizer
        self._matrix = matrix
        logger.debug("Built TF-IDF index over %d documents.", len(corpus))

    def _build_bm25(self, corpus: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [doc.split() for doc in corpus]
        self._index = BM25Okapi(tokenized)
        self._matrix = None
        logger.debug("Built BM25 index over %d documents.", len(corpus))

    def _persist_index(self) -> None:
        """Pickle the current index state to ``config.index_path``."""
        path = self._config.index_path
        assert path is not None  # guarded by callers
        payload = {
            "corpus": self._corpus,
            "index": self._index,
            "matrix": self._matrix,
            "retrieval_method": self._config.retrieval_method,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.debug("Persisted index to %s.", path)

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def load_index(self) -> None:
        """Restore a pre-built index from ``config.index_path``.

        Raises:
            FileNotFoundError: If the index file does not exist.
        """
        path = self._config.index_path
        if path is None:
            raise FileNotFoundError(
                "config.index_path is not set; cannot load a pre-built index."
            )
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Index file not found: {path!r}"
            )
        with open(path, "rb") as fh:
            payload: dict = pickle.load(fh)

        self._corpus = payload["corpus"]
        self._index = payload["index"]
        self._matrix = payload.get("matrix")
        logger.debug("Loaded index from %s (%d documents).", path, len(self._corpus))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, claim: str, top_k: int | None = None) -> list[str]:
        """Return the top-K evidence sentences most relevant to *claim*.

        Args:
            claim: The natural language claim to retrieve evidence for.
            top_k: Number of sentences to return.  Defaults to
                ``config.retrieval_top_k`` when *None*.

        Returns:
            A list of evidence sentences in descending relevance order.
            Returns ``[]`` and logs a warning when no relevant results exist.
        """
        k = top_k if top_k is not None else self._config.retrieval_top_k

        if not self._corpus or self._index is None:
            logger.warning(
                "retrieve() called but the index is empty (claim: %r).", claim
            )
            return []

        if self._config.retrieval_method == "bm25":
            return self._retrieve_bm25(claim, k)
        return self._retrieve_tfidf(claim, k)

    def _retrieve_tfidf(self, claim: str, top_k: int) -> list[str]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._index.transform([claim])
        scores: np.ndarray = cosine_similarity(query_vec, self._matrix).flatten()

        if scores.max() == 0.0:
            logger.warning(
                "All TF-IDF scores are zero for claim: %r — returning empty result.", claim
            )
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._corpus[i] for i in top_indices]

    def _retrieve_bm25(self, claim: str, top_k: int) -> list[str]:
        query_tokens = claim.split()
        scores: np.ndarray = np.array(self._index.get_scores(query_tokens))

        if scores.max() == 0.0:
            logger.warning(
                "All BM25 scores are zero for claim: %r — returning empty result.", claim
            )
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._corpus[i] for i in top_indices]
