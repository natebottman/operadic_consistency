"""
llm_answerer.py

An Answerer implementation backed by Together.ai's chat completions API.

The model is prompted to answer a question concisely — a single short
phrase or a few words — which is appropriate for the factoid QA questions
that come from HotpotQA/Break.

Usage:
    from operadic_consistency.model_interface.llm_answerer import TogetherAnswerer

    answerer = TogetherAnswerer(
        model="meta-llama/Llama-3.1-8B-Instruct-Turbo",
        api_key="...",   # or set TOGETHER_API_KEY env var
    )
    answer = answerer("Who was President when WW2 ended?")
    print(answer.text)   # "Harry S. Truman"
"""

import os
import re
from typing import Optional

from operadic_consistency.core.interfaces import Answer


_SYSTEM_PROMPT = (
    "You are a concise factual question-answering assistant. "
    "Answer the user's question with a short phrase or a few words. "
    "Do not explain your reasoning. Do not use full sentences unless necessary."
)


class TogetherAnswerer:
    """
    Answerer backed by Together.ai chat completions.

    Args:
        model: Together model ID. Default: meta-llama/Llama-3.1-8B-Instruct-Turbo.
        api_key: Together API key. Falls back to TOGETHER_API_KEY env var.
        max_tokens: Maximum tokens in the model's reply.
        temperature: Sampling temperature. Use 0 for deterministic outputs.
    """

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key: Optional[str] = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> None:
        import together  # type: ignore

        key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not key:
            raise ValueError(
                "Together API key required: pass api_key= or set TOGETHER_API_KEY."
            )

        self._client = together.Together(api_key=key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def __call__(self, question: str, *, context: Optional[str] = None) -> Answer:
        """
        Answer a question using the Together LLM.

        Args:
            question: The fully-instantiated question string (placeholders
                      already substituted by the evaluation loop).
            context: Optional context string passed through from the
                     consistency check. Not currently used in the prompt
                     but preserved for interface compatibility.

        Returns:
            Answer with the model's reply as .text.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        text = response.choices[0].message.content.strip()
        # Strip <think>...</think> blocks produced by reasoning models (e.g. DeepSeek-R1)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return Answer(
            text=text,
            meta={
                "model": self._model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            },
        )
