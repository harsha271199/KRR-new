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
