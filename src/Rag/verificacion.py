
"""Verify AI citations against official CIMA document fragments."""

from pydantic import BaseModel, Field
from src.config import MIN_QUOTE_WORDS


class Fragment(BaseModel):
    """One text fragment from an official CIMA technical-sheet section."""

    nregistro: str = Field(min_length=1)
    nombre: str | None = None
    seccion: str = Field(min_length=1)
    titulo: str | None = None
    texto: str = Field(min_length=1)


class Citation(BaseModel):
    """An exact quotation linked to a CIMA medicine and section."""

    nregistro: str = Field(min_length=1)
    seccion: str = Field(min_length=1)
    literal_quote: str = Field(min_length=1)


def normalize_text(text: str) -> str:
    """Convert text to lowercase and remove extra whitespace."""
    return " ".join(text.casefold().split())


def verificar_cita(cita: Citation, fragmento: Fragment) -> bool:
    """Check whether a citation matches its official source fragment."""

    # The medicine and document section must match.
    if cita.nregistro != fragmento.nregistro:
        return False

    if cita.seccion != fragmento.seccion:
        return False

    # Normalize the quotation and source text.
    quote = normalize_text(cita.literal_quote)
    source_text = normalize_text(fragmento.texto)

    # Reject quotations containing fewer than six words.
    if len(quote.split()) < MIN_QUOTE_WORDS:
        return False

    # Accept only if the complete quotation exists in the source.
    return quote in source_text

