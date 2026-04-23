"""Tests for Config (config.py)."""

import os
import logging
import pytest
from unittest.mock import patch

from config import Config, ConfigError


def _base_env():
    return {
        "FEVER_DATASET_PATH": "/data/fever.jsonl",
        "EVIDENCE_CORPUS_PATH": "/data/corpus.jsonl",
        "EVAL_OUTPUT_PATH": "/out/eval.txt",
    }


class TestConfigRequiredKeys:
    def test_all_required_keys_present(self):
        with patch.dict(os.environ, _base_env(), clear=True):
            cfg = Config.from_env()
        assert cfg.fever_dataset_path == "/data/fever.jsonl"
        assert cfg.evidence_corpus_path == "/data/corpus.jsonl"
        assert cfg.eval_output_path == "/out/eval.txt"

    def test_missing_fever_dataset_path_raises(self):
        env = _base_env()
        del env["FEVER_DATASET_PATH"]
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError) as exc_info:
                Config.from_env()
        assert "FEVER_DATASET_PATH" in str(exc_info.value)

    def test_missing_evidence_corpus_path_raises(self):
        env = _base_env()
        del env["EVIDENCE_CORPUS_PATH"]
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError) as exc_info:
                Config.from_env()
        assert "EVIDENCE_CORPUS_PATH" in str(exc_info.value)

    def test_missing_eval_output_path_raises(self):
        env = _base_env()
        del env["EVAL_OUTPUT_PATH"]
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError) as exc_info:
                Config.from_env()
        assert "EVAL_OUTPUT_PATH" in str(exc_info.value)


class TestConfigDefaults:
    def test_retrieval_top_k_default(self):
        with patch.dict(os.environ, _base_env(), clear=True):
            cfg = Config.from_env()
        assert cfg.retrieval_top_k == 5

    def test_nlp_engine_default(self):
        with patch.dict(os.environ, _base_env(), clear=True):
            cfg = Config.from_env()
        assert cfg.nlp_engine == "spacy"

    def test_llm_backend_default(self):
        with patch.dict(os.environ, _base_env(), clear=True):
            cfg = Config.from_env()
        assert cfg.llm_backend == "huggingface"

    def test_max_records_default_is_none(self):
        with patch.dict(os.environ, _base_env(), clear=True):
            cfg = Config.from_env()
        assert cfg.max_records is None

    def test_index_path_default_is_none(self):
        with patch.dict(os.environ, _base_env(), clear=True):
            cfg = Config.from_env()
        assert cfg.index_path is None

    def test_spacy_model_default(self):
        with patch.dict(os.environ, _base_env(), clear=True):
            cfg = Config.from_env()
        assert cfg.spacy_model == "en_core_web_sm"


class TestConfigOptionalOverrides:
    def test_retrieval_top_k_override(self):
        env = {**_base_env(), "RETRIEVAL_TOP_K": "10"}
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        assert cfg.retrieval_top_k == 10

    def test_max_records_override(self):
        env = {**_base_env(), "MAX_RECORDS": "500"}
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        assert cfg.max_records == 500

    def test_invalid_retrieval_top_k_raises(self):
        env = {**_base_env(), "RETRIEVAL_TOP_K": "not-a-number"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError):
                Config.from_env()

    def test_invalid_max_records_raises(self):
        env = {**_base_env(), "MAX_RECORDS": "abc"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError):
                Config.from_env()


class TestConfigApiKeySecurity:
    def test_llm_api_key_not_in_log_output(self, caplog):
        secret = "sk-supersecretkey12345"
        env = {**_base_env(), "LLM_API_KEY": secret}
        with patch.dict(os.environ, env, clear=True):
            with caplog.at_level(logging.DEBUG):
                cfg = Config.from_env()
        # The secret value must not appear anywhere in log output
        assert secret not in caplog.text
        # But the key should be loaded
        assert cfg.llm_api_key == secret

    def test_llm_api_key_absent_is_none(self):
        with patch.dict(os.environ, _base_env(), clear=True):
            cfg = Config.from_env()
        assert cfg.llm_api_key is None
