
"""Shared Groq client for JSON responses from the language model."""

import json
import os

from groq import Groq

from src.config import GROQ_MODEL, MAX_TOKENS, TEMPERATURE


def generar_json(mensajes: list[dict[str, str]]) -> dict:
    """Send messages to Groq and return a parsed JSON object."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_groq_api_key_here":
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=mensajes,
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Groq returned an empty response")

    result = json.loads(content)

    if not isinstance(result, dict):
        raise ValueError("Groq response must be a JSON object")

    return result
