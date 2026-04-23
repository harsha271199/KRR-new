"""Baseline RAG pipeline: retrieve evidence, build prompt, classify via LLM."""

from __future__ import annotations

import logging
import re

from config import Config
from models import BaselineResult
from retrieval.retriever import Retriever
from baseline.llm import LLMBackend

logger = logging.getLogger(__name__)

_VERDICT_PATTERNS = [
    # Check NOT ENOUGH INFO first to avoid partial match on SUPPORTS/REFUTES
    ("NOT ENOUGH INFO", re.compile(r"not enough info", re.IGNORECASE)),
    ("SUPPORTS", re.compile(r"supports", re.IGNORECASE)),
    ("REFUTES", re.compile(r"refutes", re.IGNORECASE)),
]

_PROMPT_TEMPLATE = (
    "You are a fact-checking assistant. Given the following claim and evidence, \n"
    "classify the claim as one of: SUPPORTS, REFUTES, NOT ENOUGH INFO.\n"
    "\n"
    "Claim: {claim}\n"
    "\n"
    "Evidence:\n"
    "{evidence_sentences_numbered}\n"
    "\n"
    "Respond with exactly one of: SUPPORTS, REFUTES, NOT ENOUGH INFO"
)


def _build_prompt(claim: str, evidence: list[str]) -> str:
    """Format the prompt template with the claim and numbered evidence sentences."""
    numbered = "\n".join(f"{i + 1}. {sentence}" for i, sentence in enumerate(evidence))
    return _PROMPT_TEMPLATE.format(claim=claim, evidence_sentences_numbered=numbered)


def _parse_verdict(response: str) -> str | None:
    """Search response for a valid verdict label (case-insensitive).

    Checks NOT ENOUGH INFO before SUPPORTS/REFUTES to avoid partial matches.

    Returns:
        The matched verdict string, or None if no valid verdict found.
    """
    for verdict, pattern in _VERDICT_PATTERNS:
        if pattern.search(response):
            return verdict
    return None


class BaselinePipeline:
    """Baseline RAG pipeline that classifies claims using an LLM.

    Args:
        retriever: A pre-built Retriever instance for evidence retrieval.
        llm_backend: An LLMBackend implementation (HuggingFace, OpenAI, or Mock).
        config: Runtime configuration supplying retrieval parameters.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_backend: LLMBackend,
        config: Config,
    ) -> None:
        self._retriever = retriever
        self._llm_backend = llm_backend
        self._config = config

    def run(self, claim: str) -> BaselineResult:
        """Retrieve evidence, build prompt, call LLM, and parse verdict.

        Args:
            claim: The natural language claim to verify.

        Returns:
            A BaselineResult with all fields populated. On LLM failure, verdict
            is "NOT ENOUGH INFO" and the error field is set.
        """
        # Step 1: Retrieve top-K evidence
        evidence = self._retriever.retrieve(claim)

        # Step 2: Build prompt
        prompt = _build_prompt(claim, evidence)

        # Step 3 & 4: Call LLM and parse verdict
        raw_response = ""
        try:
            raw_response = self._llm_backend.classify(prompt)
        except Exception as exc:
            logger.error(
                "LLM backend raised an exception for claim %r: %s", claim, exc
            )
            return BaselineResult(
                claim=claim,
                retrieved_evidence=evidence,
                prompt=prompt,
                verdict="NOT ENOUGH INFO",
                raw_llm_response=raw_response,
                error=str(exc),
            )

        # Step 4: Parse verdict from response
        verdict = _parse_verdict(raw_response)

        # Step 5: Fall back to NOT ENOUGH INFO if no valid verdict found
        if verdict is None:
            logger.warning(
                "LLM response did not contain a recognizable verdict for claim %r. "
                "Response: %r. Defaulting to NOT ENOUGH INFO.",
                claim,
                raw_response,
            )
            verdict = "NOT ENOUGH INFO"

        # Step 7: Return populated result
        return BaselineResult(
            claim=claim,
            retrieved_evidence=evidence,
            prompt=prompt,
            verdict=verdict,  # type: ignore[arg-type]
            raw_llm_response=raw_response,
            error=None,
        )
