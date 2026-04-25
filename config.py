"""Configuration management for the KRR Fact Verification System.

Loads all settings from OS environment variables.  Call :func:`load_env_file`
(or use ``python-dotenv`` directly) before :meth:`Config.from_env` to populate
the environment from a ``.env`` file.  OS environment variables always take
precedence over ``.env`` file values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


class ConfigError(Exception):
    """Raised when a required configuration key is missing or invalid."""


def load_env_file(env_file: str = ".env", override: bool = False) -> None:
    """Populate ``os.environ`` from *env_file* using ``python-dotenv``.

    Args:
        env_file: Path to the ``.env`` file (default: ``".env"``).
        override: When ``True``, values in the file override existing env vars.
                  Defaults to ``False`` so OS env vars take precedence.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import]
    except ImportError:
        return  # python-dotenv not installed; silently skip
    load_dotenv(env_file, override=override)


@dataclass
class Config:
    """All runtime configuration for the KRR pipeline.

    Instantiate via :meth:`from_env` rather than directly.
    """

    # Required fields
    fever_dataset_path: str
    evidence_corpus_path: str
    eval_output_path: str

    # Optional fields with defaults
    retrieval_top_k: int = 5
    retrieval_method: Literal["tfidf", "bm25"] = "tfidf"
    nlp_engine: Literal["spacy", "openie"] = "spacy"
    llm_backend: Literal["huggingface", "openai"] = "huggingface"
    llm_api_key: str | None = None
    hf_model_name: str = "google/flan-t5-base"
    openai_model: str = "gpt-3.5-turbo"
    max_records: int | None = None
    index_path: str | None = None
    spacy_model: str = "en_core_web_sm"
    json_output_path: str | None = None

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "Config":
        """Load configuration from OS environment variables.

        Call :func:`load_env_file` (or ``python-dotenv``'s ``load_dotenv``)
        before this method to populate ``os.environ`` from a ``.env`` file.
        This method reads *only* from ``os.environ`` so that test patches using
        ``patch.dict(os.environ, ..., clear=True)`` are fully respected.

        The *env_file* parameter is accepted for API compatibility but is no
        longer used to load the file inside this method.

        Args:
            env_file: Ignored (kept for backward compatibility).

        Returns:
            A fully populated :class:`Config` instance.

        Raises:
            ConfigError: If any required configuration key is absent.
        """
        required_keys = [
            "FEVER_DATASET_PATH",
            "EVIDENCE_CORPUS_PATH",
            "EVAL_OUTPUT_PATH",
        ]

        # Validate required keys
        for key in required_keys:
            if not os.environ.get(key):
                raise ConfigError(f"Missing required configuration key: {key}")

        # Parse optional integer fields
        retrieval_top_k_raw = os.environ.get("RETRIEVAL_TOP_K", "5")
        try:
            retrieval_top_k = int(retrieval_top_k_raw)
        except ValueError:
            raise ConfigError(
                f"RETRIEVAL_TOP_K must be an integer, got: {retrieval_top_k_raw!r}"
            )

        max_records: int | None = None
        max_records_raw = os.environ.get("MAX_RECORDS")
        if max_records_raw is not None:
            try:
                max_records = int(max_records_raw)
            except ValueError:
                raise ConfigError(
                    f"MAX_RECORDS must be an integer, got: {max_records_raw!r}"
                )

        return cls(
            fever_dataset_path=os.environ["FEVER_DATASET_PATH"],
            evidence_corpus_path=os.environ["EVIDENCE_CORPUS_PATH"],
            eval_output_path=os.environ["EVAL_OUTPUT_PATH"],
            retrieval_top_k=retrieval_top_k,
            retrieval_method=os.environ.get("RETRIEVAL_METHOD", "tfidf"),  # type: ignore[arg-type]
            nlp_engine=os.environ.get("NLP_ENGINE", "spacy"),  # type: ignore[arg-type]
            llm_backend=os.environ.get("LLM_BACKEND", "huggingface"),  # type: ignore[arg-type]
            llm_api_key=os.environ.get("LLM_API_KEY") or None,
            hf_model_name=os.environ.get("HF_MODEL_NAME", "google/flan-t5-base"),
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            max_records=max_records,
            index_path=os.environ.get("INDEX_PATH") or None,
            spacy_model=os.environ.get("SPACY_MODEL", "en_core_web_sm"),
            json_output_path=os.environ.get("JSON_OUTPUT_PATH") or None,
        )
