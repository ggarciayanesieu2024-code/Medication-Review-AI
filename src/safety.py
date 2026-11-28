"""Shared screening of generated medication instructions."""

import re
import unicodedata

from src.config import (
    FORBIDDEN_PHRASES,
    SOURCE_QUOTE_FORBIDDEN_PHRASES,
)

def normalise_safety_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.split())


def find_forbidden_phrases(text: str, *, source_quote=False):
    normalised = normalise_safety_text(text)

    if source_quote:
        return [
            p for p in SOURCE_QUOTE_FORBIDDEN_PHRASES
            if normalise_safety_text(p) in normalised
        ]

    hits = [
        p for p in FORBIDDEN_PHRASES
        if re.search(
            r"\b" + re.escape(normalise_safety_text(p)) + r"\b",
            normalised,
        )
    ]

    patterns = {
        "direct treatment instruction": (
            r"(?:^|[.!?;]\s*)"
            r"(?:please\s+)?"
            r"(?:stop|replace|prescribe|start|continue|avoid)\s+"
        ),
        "modal treatment instruction": (
            r"\b(?:should|must|needs? to|recommend(?:ed|s)?)"
            r"\s+(?:stop|replace|prescribe|start|continue|avoid)\s+"
        ),
    }

    for label, pattern in patterns.items():
        if re.search(pattern, normalised):
            hits.append(label)

    return hits
