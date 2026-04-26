"""Evidence retrieval using TF-IDF, BM25, or hybrid TF-IDF + semantic embeddings.

The hybrid method (default when sentence-transformers is available) combines:
- TF-IDF cosine similarity (lexical overlap)
- Sentence embedding cosine similarity (semantic similarity)

Scores are combined as: hybrid = alpha * tfidf_score + (1 - alpha) * semantic_score
where alpha defaults to 0.3 (favouring semantic similarity).

If sentence-transformers is not installed, the retriever falls back to TF-IDF only.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from config import Config

logger = logging.getLogger(__name__)

# Weight given to TF-IDF in the hybrid score (0 = pure semantic, 1 = pure TF-IDF)
_HYBRID_ALPHA = 0.3


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class Retriever:
    """Build and query a retrieval index over an evidence corpus.

    Supports three retrieval methods:
    - ``"tfidf"``   — TF-IDF cosine similarity (sklearn)
    - ``"bm25"``    — BM25 (rank-bm25)
    - ``"hybrid"``  — TF-IDF + sentence-embedding cosine similarity (default
                      when sentence-transformers is installed)

    Args:
        config: Runtime configuration supplying retrieval parameters.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._corpus: list[str] = []
        self._index: Any = None          # TfidfVectorizer or BM25Okapi
        self._matrix: Any = None         # sparse TF-IDF matrix
        self._embeddings: Any = None     # np.ndarray of sentence embeddings
        self._embed_model: Any = None    # SentenceTransformer instance

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, corpus: list[str]) -> None:
        """Index the corpus.  Persists to ``config.index_path`` if set."""
        self._corpus = list(corpus)

        if not corpus:
            logger.warning("build_index called with an empty corpus.")
            self._index = None
            self._matrix = None
            self._embeddings = None
            if self._config.index_path:
                self._persist_index()
            return

        method = self._config.retrieval_method

        if method == "bm25":
            self._build_bm25(corpus)
        elif method == "hybrid" or (
            method == "tfidf" and _sentence_transformers_available()
        ):
            # Use hybrid when explicitly requested OR when tfidf is requested
            # but sentence-transformers is available (automatic upgrade)
            self._build_tfidf(corpus)
            self._build_embeddings(corpus)
            logger.info(
                "Built hybrid (TF-IDF + semantic) index over %d documents.", len(corpus)
            )
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

    def _build_embeddings(self, corpus: list[str]) -> None:
        """Encode the corpus with sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            if self._embed_model is None:
                logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)…")
                self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embeddings = self._embed_model.encode(
                corpus, convert_to_numpy=True, show_progress_bar=False
            )
            logger.debug(
                "Built semantic embeddings for %d documents.", len(corpus)
            )
        except Exception as exc:
            logger.warning(
                "Semantic embedding failed (%s) — falling back to TF-IDF only.", exc
            )
            self._embeddings = None

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _persist_index(self) -> None:
        path = self._config.index_path
        assert path is not None
        payload = {
            "corpus": self._corpus,
            "index": self._index,
            "matrix": self._matrix,
            "embeddings": self._embeddings,
            "retrieval_method": self._config.retrieval_method,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.debug("Persisted index to %s.", path)

    def load_index(self) -> None:
        """Restore a pre-built index from ``config.index_path``."""
        path = self._config.index_path
        if path is None:
            raise FileNotFoundError(
                "config.index_path is not set; cannot load a pre-built index."
            )
        if not Path(path).exists():
            raise FileNotFoundError(f"Index file not found: {path!r}")
        with open(path, "rb") as fh:
            payload: dict = pickle.load(fh)
        self._corpus = payload["corpus"]
        self._index = payload["index"]
        self._matrix = payload.get("matrix")
        self._embeddings = payload.get("embeddings")
        logger.debug("Loaded index from %s (%d documents).", path, len(self._corpus))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, claim: str, top_k: int | None = None) -> list[str]:
        """Return the top-K evidence sentences most relevant to *claim*."""
        k = top_k if top_k is not None else self._config.retrieval_top_k

        if not self._corpus or self._index is None:
            logger.warning("retrieve() called but the index is empty (claim: %r).", claim)
            return []

        if self._config.retrieval_method == "bm25":
            return self._retrieve_bm25(claim, k)

        # Hybrid: combine TF-IDF and semantic scores when embeddings are available
        if self._embeddings is not None and self._embed_model is not None:
            return self._retrieve_hybrid(claim, k)

        return self._retrieve_tfidf(claim, k)

    def _retrieve_tfidf(self, claim: str, top_k: int) -> list[str]:
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self._index.transform([claim])
        scores: np.ndarray = cosine_similarity(query_vec, self._matrix).flatten()
        if scores.max() == 0.0:
            logger.warning("All TF-IDF scores are zero for claim: %r", claim)
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._corpus[i] for i in top_indices]

    def _retrieve_bm25(self, claim: str, top_k: int) -> list[str]:
        query_tokens = claim.split()
        scores: np.ndarray = np.array(self._index.get_scores(query_tokens))
        if scores.max() == 0.0:
            logger.warning("All BM25 scores are zero for claim: %r", claim)
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._corpus[i] for i in top_indices]

    def _retrieve_hybrid(self, claim: str, top_k: int) -> list[str]:
        """Combine TF-IDF and semantic similarity scores."""
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

        # TF-IDF scores (normalised to [0, 1])
        query_vec = self._index.transform([claim])
        tfidf_scores: np.ndarray = sk_cosine(query_vec, self._matrix).flatten()
        tfidf_max = tfidf_scores.max()
        if tfidf_max > 0:
            tfidf_scores = tfidf_scores / tfidf_max

        # Semantic scores (normalised to [0, 1])
        claim_emb = self._embed_model.encode([claim], convert_to_numpy=True)
        # cosine similarity between claim embedding and corpus embeddings
        norms_corpus = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms_corpus = np.where(norms_corpus == 0, 1e-9, norms_corpus)
        norm_claim = np.linalg.norm(claim_emb)
        if norm_claim == 0:
            norm_claim = 1e-9
        sem_scores: np.ndarray = (
            self._embeddings.dot(claim_emb.T).flatten()
            / (norms_corpus.flatten() * norm_claim)
        )
        # Shift from [-1, 1] to [0, 1]
        sem_scores = (sem_scores + 1.0) / 2.0

        # Hybrid combination
        hybrid_scores = _HYBRID_ALPHA * tfidf_scores + (1.0 - _HYBRID_ALPHA) * sem_scores

        if hybrid_scores.max() == 0.0:
            logger.warning("All hybrid scores are zero for claim: %r", claim)
            return []

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [self._corpus[i] for i in top_indices]
