"""Tests for DataLoader (data/loader.py)."""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from config import Config
from data.loader import DataLoader


def _make_config(path: str, max_records=None) -> Config:
    env = {
        "FEVER_DATASET_PATH": path,
        "EVIDENCE_CORPUS_PATH": path,
        "EVAL_OUTPUT_PATH": "/tmp/eval.txt",
    }
    if max_records is not None:
        env["MAX_RECORDS"] = str(max_records)
    with patch.dict(os.environ, env, clear=True):
        return Config.from_env()


def _write_jsonl(records: list[dict]) -> str:
    """Write records to a temp JSONL file and return the path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


SAMPLE_RECORDS = [
    {
        "claim": "Paris is in France.",
        "label": "SUPPORTS",
        "evidence": [["Paris is the capital of France."]],
    },
    {
        "claim": "London is in Germany.",
        "label": "REFUTES",
        "evidence": [["London is in England."]],
    },
    {
        "claim": "The sky is green.",
        "label": "NOT ENOUGH INFO",
        "evidence": [[]],
    },
]


class TestDataLoaderLoad:
    def test_loads_all_records(self):
        path = _write_jsonl(SAMPLE_RECORDS)
        try:
            loader = DataLoader(_make_config(path))
            records = loader.load()
            assert len(records) == 3
        finally:
            Path(path).unlink()

    def test_record_fields(self):
        path = _write_jsonl(SAMPLE_RECORDS[:1])
        try:
            loader = DataLoader(_make_config(path))
            records = loader.load()
            r = records[0]
            assert r.claim == "Paris is in France."
            assert r.label == "SUPPORTS"
            assert isinstance(r.evidence, list)
            assert r.record_index == 0
        finally:
            Path(path).unlink()

    def test_label_normalized_to_uppercase(self):
        records = [
            {"claim": "test", "label": "supports", "evidence": [[]]},
        ]
        path = _write_jsonl(records)
        try:
            loader = DataLoader(_make_config(path))
            result = loader.load()
            assert result[0].label == "SUPPORTS"
        finally:
            Path(path).unlink()

    def test_file_not_found_raises(self):
        cfg = _make_config("/nonexistent/path/fever.jsonl")
        loader = DataLoader(cfg)
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load()
        assert "/nonexistent/path/fever.jsonl" in str(exc_info.value)

    def test_malformed_json_skipped_with_warning(self, caplog):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        f.write("not valid json\n")
        f.write(json.dumps(SAMPLE_RECORDS[0]) + "\n")
        f.close()
        try:
            loader = DataLoader(_make_config(f.name))
            with caplog.at_level(logging.WARNING):
                records = loader.load()
            assert len(records) == 1
            assert any("Skipping" in m for m in caplog.messages)
        finally:
            Path(f.name).unlink()

    def test_missing_required_field_skipped_with_warning(self, caplog):
        bad = {"claim": "test"}  # missing label and evidence
        path = _write_jsonl([bad, SAMPLE_RECORDS[0]])
        try:
            loader = DataLoader(_make_config(path))
            with caplog.at_level(logging.WARNING):
                records = loader.load()
            assert len(records) == 1
            assert any("Skipping" in m for m in caplog.messages)
        finally:
            Path(path).unlink()

    def test_max_records_limits_output(self):
        path = _write_jsonl(SAMPLE_RECORDS)
        try:
            loader = DataLoader(_make_config(path, max_records=2))
            records = loader.load()
            assert len(records) == 2
        finally:
            Path(path).unlink()

    def test_empty_lines_skipped(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        f.write("\n")
        f.write(json.dumps(SAMPLE_RECORDS[0]) + "\n")
        f.write("\n")
        f.close()
        try:
            loader = DataLoader(_make_config(f.name))
            records = loader.load()
            assert len(records) == 1
        finally:
            Path(f.name).unlink()


class TestDataLoaderLoadCorpus:
    def test_returns_list_of_strings(self):
        path = _write_jsonl(SAMPLE_RECORDS)
        try:
            loader = DataLoader(_make_config(path))
            corpus = loader.load_corpus()
            assert isinstance(corpus, list)
            assert all(isinstance(s, str) for s in corpus)
        finally:
            Path(path).unlink()

    def test_deduplicates_evidence_sentences(self):
        # Two records share the same evidence sentence
        records = [
            {"claim": "A", "label": "SUPPORTS", "evidence": [["Shared sentence."]]},
            {"claim": "B", "label": "REFUTES", "evidence": [["Shared sentence."]]},
        ]
        path = _write_jsonl(records)
        try:
            loader = DataLoader(_make_config(path))
            corpus = loader.load_corpus()
            assert corpus.count("Shared sentence.") == 1
        finally:
            Path(path).unlink()

    def test_corpus_contains_evidence_sentences(self):
        path = _write_jsonl(SAMPLE_RECORDS[:2])
        try:
            loader = DataLoader(_make_config(path))
            corpus = loader.load_corpus()
            # At least some sentences should be present
            assert len(corpus) > 0
        finally:
            Path(path).unlink()
