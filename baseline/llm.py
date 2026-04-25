"""LLM backend implementations for the Baseline RAG pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from config import Config


@runtime_checkable
class LLMBackend(Protocol):
    """Abstract interface for LLM backends used by the Baseline pipeline."""

    def classify(self, prompt: str) -> str:
        """Submit a prompt and return the raw LLM response string.

        Args:
            prompt: The text prompt to send to the LLM.

        Returns:
            The raw response string from the LLM.

        Raises:
            TypeError: If prompt is not a str.
        """
        ...


class HuggingFaceLLM:
    """LLM backend using a HuggingFace text-generation pipeline.

    Args:
        config: Runtime configuration supplying the HuggingFace model name.
    """

    def __init__(self, config: Config) -> None:
        from transformers import pipeline as hf_pipeline

        self._pipeline = hf_pipeline(
            "text-generation",
            model=config.hf_model_name,
        )

    def classify(self, prompt: str) -> str:
        """Run the HuggingFace pipeline and return the generated text.

        Args:
            prompt: The text prompt to classify.

        Returns:
            The generated text string from the model.

        Raises:
            TypeError: If prompt is not a str.
        """
        if not isinstance(prompt, str):
            raise TypeError(
                f"classify() expected prompt to be a str, got {type(prompt).__name__!r}"
            )
        result = self._pipeline(prompt, max_new_tokens=20)
        # result is a list of dicts; the generated text is in result[0]["generated_text"]
        return result[0]["generated_text"]


class OpenAILLM:
    """LLM backend using the OpenAI ChatCompletion API.

    Args:
        config: Runtime configuration supplying the OpenAI model name.
    """

    def __init__(self, config: Config) -> None:
        import openai

        self._openai = openai
        self._model = config.openai_model

    def classify(self, prompt: str) -> str:
        """Call the OpenAI ChatCompletion API and return the message content.

        Args:
            prompt: The text prompt to classify.

        Returns:
            The message content string from the API response.

        Raises:
            TypeError: If prompt is not a str.
        """
        if not isinstance(prompt, str):
            raise TypeError(
                f"classify() expected prompt to be a str, got {type(prompt).__name__!r}"
            )
        response = self._openai.ChatCompletion.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        return response.choices[0].message.content


class MockLLM:
    """Configurable mock LLM backend for testing.

    Args:
        response: The fixed string to return from every classify() call.
    """

    def __init__(self, response: str) -> None:
        self.response = response

    def classify(self, prompt: str) -> str:
        """Return the fixed response string regardless of the prompt.

        Args:
            prompt: The text prompt (ignored).

        Returns:
            The fixed response string supplied at construction time.

        Raises:
            TypeError: If prompt is not a str.
        """
        if not isinstance(prompt, str):
            raise TypeError(
                f"classify() expected prompt to be a str, got {type(prompt).__name__!r}"
            )
        return self.response


class KeywordLLM:
    """Lightweight keyword-based classifier that requires no model download.

    Parses the structured prompt produced by BaselinePipeline to extract the
    claim and evidence sentences, then applies heuristic rules to return one
    of the three valid verdict labels.

    Rules (applied in order):
    1. If any evidence sentence contains a strong negation of the claim's key
       noun phrase → REFUTES.
    2. If any evidence sentence contains the claim's key noun phrase and no
       negation → SUPPORTS.
    3. Otherwise → NOT ENOUGH INFO.

    This backend is intended for offline evaluation and smoke-testing when no
    real LLM is available.
    """

    # Negation words that flip a matching sentence to REFUTES
    _NEGATIONS = frozenset([
        "not", "no", "never", "contrary", "incorrect", "false",
        "unlike", "cannot", "isn't", "aren't", "wasn't", "weren't",
    ])

    def classify(self, prompt: str) -> str:
        """Classify the claim in *prompt* using keyword heuristics.

        Args:
            prompt: The formatted prompt string from BaselinePipeline.

        Returns:
            One of "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO".

        Raises:
            TypeError: If prompt is not a str.
        """
        if not isinstance(prompt, str):
            raise TypeError(
                f"classify() expected prompt to be a str, got {type(prompt).__name__!r}"
            )

        claim, evidence_lines = self._parse_prompt(prompt)
        if not claim or not evidence_lines:
            return "NOT ENOUGH INFO"

        claim_keywords = self._extract_keywords(claim)
        if not claim_keywords:
            return "NOT ENOUGH INFO"

        for line in evidence_lines:
            line_lower = line.lower()
            line_words = set(line_lower.split())

            # Check how many claim keywords appear in this evidence line
            overlap = claim_keywords & line_words
            if not overlap:
                continue

            # Check for negation words in the same sentence
            has_negation = bool(self._NEGATIONS & line_words)
            if has_negation:
                return "REFUTES"
            return "SUPPORTS"

        return "NOT ENOUGH INFO"

    @staticmethod
    def _parse_prompt(prompt: str) -> tuple[str, list[str]]:
        """Extract the claim string and evidence lines from the prompt."""
        claim = ""
        evidence_lines: list[str] = []
        in_evidence = False

        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("Claim:"):
                claim = stripped[len("Claim:"):].strip()
            elif stripped.startswith("Evidence:"):
                in_evidence = True
            elif in_evidence and stripped and stripped[0].isdigit() and "." in stripped:
                # Numbered evidence line: "1. Some sentence."
                dot_idx = stripped.index(".")
                evidence_lines.append(stripped[dot_idx + 1:].strip())
            elif in_evidence and stripped.startswith("Respond with"):
                in_evidence = False

        return claim, evidence_lines

    @staticmethod
    def _extract_keywords(text: str) -> frozenset[str]:
        """Return content words (length ≥ 4) from *text*, lowercased."""
        stopwords = frozenset([
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "that", "this", "these", "those", "with", "from", "into",
            "than", "then", "when", "where", "which", "who", "whom",
            "what", "how", "also", "just", "more", "most", "some",
        ])
        words = frozenset(
            w.strip(".,!?;:\"'()").lower()
            for w in text.split()
            if len(w) >= 4
        )
        return words - stopwords
