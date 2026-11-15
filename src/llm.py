
"""Call Groq and validate medication-review responses with StructuredOutputRAG.

This checks the response format. Source citations and treatment wording must
still be checked by the RAG pipeline before displaying a finding.
"""

import os

from groq import Groq

from src.config import (
    GROQ_MODEL,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    MAX_TOKENS,
    REASONING_EFFORT,
    TEMPERATURE,
    TOP_P,
)
from src.modelos import StructuredOutputRAG


def llm_medication_review(messages: list[dict[str, str]]) -> StructuredOutputRAG:
    """Send prepared messages to Groq and return a validated response model."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_groq_api_key_here":
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    client = Groq(
        api_key=api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )

    # Other models may not support GPT-OSS reasoning controls.
    reasoning_options = {}
    if GROQ_MODEL in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
        reasoning_options["reasoning_effort"] = REASONING_EFFORT

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        **reasoning_options,
    )

    if not response.choices:
        raise ValueError("Groq returned no response choices")

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("Groq reached the completion token limit; response rejected")

    content = choice.message.content

    if not content:
        raise ValueError("Groq returned an empty response")

    # JSON mode requests JSON; Pydantic checks the required fields and rules.
    # Invalid JSON or an invalid response structure raises ValidationError.
    return StructuredOutputRAG.model_validate_json(content)
