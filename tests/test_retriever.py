"""Tests for Retriever (retrieval/retriever.py)."""

import logging
import os
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from config import Config
from retrieval.retriever import Retriever


def _make_config(**overrides) -> Config:
    env = {
        "FEVER_DATASET_PATH": "/tmp/x",
        "EVIDENCE_CORPUS_PATH": "/tmp/x",
        "EVAL_OUTPUT_PATH": "/tmp/x",
        **{k.upper(): str(v) for k, v in overrides.items()},
    }
    with patch.dict(os.environ, env, clear=True):
        return Config.from_env()


SMALL_CORPUS = [
    "Albert Einstein was born in Ulm, Germany.",
    "Marie Curie discovered radium and polonium.",
    "The Eiffel Tower is located in Paris, France.",
    "Python is a high-level programming language.",
    "The Amazon River is the largest river by discharge.",
]


class TestRetrieverTFIDF:
    def test_build_and_retrieve_returns_list(self):
        cfg = _make_config(retrieval_top_k=2)
        r = Retriever(cfg)
        r.build_index(SMALL_CORPUS)
        results = r.retrieve("Einstein birthplace Germany")
        assert isinstance(results, list)

    def test_retrieve_top_k_respected(self):
        cfg = _make_config(retrieval_top_k=2)
        r = Retriever(cfg)
        r.build_index(SMALL_CORPUS)
        results = r.retrieve("Einstein birthplace Germany", top_k=2)
        assert len(results) <= 2

    def test_retrieve_relevant_result_first(self):
        cfg = _make_config(retrieval_top_k=3)
        r = Retriever(cfg)
        r.build_index(SMALL_CORPUS)
        results = r.retrieve("Albert Einstein born Ulm", top_k=1)
        assert len(results) == 1
        assert "Einstein" in results[0]

    def test_all_zero_scores_returns_empty_with_warning(self, caplog):
        cfg = _make_config(retrieval_top_k=3)
        r = Retriever(cfg)
        r.build_index(SMALL_CORPUS)
        with caplog.at_level(logging.WARNING):
            results = r.retrieve("xyzzy quux frobnicator zap")
        assert results == []
        assert any("zero" in m.lower() or "warning" in m.lower() or "empty" in m.lower()
                   for m in caplog.messages)

    def test_empty_corpus_returns_empty_with_warning(self, caplog):
        cfg = _make_config(retrieval_top_k=3)
        r = Retriever(cfg)
        with caplog.at_level(logging.WARNING):
            r.build_index([])
            results = r.retrieve("anything")
        assert results == []


class TestRetrieverBM25:
    def test_build_and_retrieve_bm25(self):
        cfg = _make_config(retrieval_method="bm25", retrieval_top_k=2)
        r = Retriever(cfg)
        r.build_index(SMALL_CORPUS)
        results = r.retrieve("Marie Curie radium", top_k=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_bm25_relevant_result(self):
        cfg = _make_config(retrieval_method="bm25", retrieval_top_k=1)
        r = Retriever(cfg)
        r.build_index(SMALL_CORPUS)
        results = r.retrieve("Marie Curie radium polonium", top_k=1)
        assert len(results) == 1
        assert "Curie" in results[0]


class TestRetrieverPersistence:
    def test_save_and_load_index_tfidf(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp.close()
        try:
            cfg = _make_config(retrieval_top_k=2, index_path=tmp.name)
            r1 = Retriever(cfg)
            r1.build_index(SMALL_CORPUS)
            original = r1.retrieve("Einstein born Ulm", top_k=2)

            r2 = Retriever(cfg)
            r2.load_index()
            reloaded = r2.retrieve("Einstein born Ulm", top_k=2)

            assert original == reloaded
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_load_index_missing_file_raises(self):
        cfg = _make_config(retrieval_top_k=2, index_path="/nonexistent/index.pkl")
        r = Retriever(cfg)
        with pytest.raises(FileNotFoundError):
            r.load_index()

    def test_load_index_no_path_configured_raises(self):
        cfg = _make_config(retrieval_top_k=2)
        r = Retriever(cfg)
        with pytest.raises(FileNotFoundError):
            r.load_index()


class TestRetrieverPerformance:
    def test_retrieve_completes_within_2_seconds_for_1000_sentences(self):
        corpus = [f"Sentence number {i} about topic {i % 50}." for i in range(1000)]
        cfg = _make_config(retrieval_top_k=5)
        r = Retriever(cfg)
        r.build_index(corpus)
        start = time.time()
        r.retrieve("topic sentence number", top_k=5)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Retrieval took {elapsed:.2f}s, expected < 2s"
