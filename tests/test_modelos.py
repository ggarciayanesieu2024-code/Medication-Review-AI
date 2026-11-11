
"""Tests for the shared RAG response data model."""

import pytest
from pydantic import ValidationError

from src.modelos import RespuestaRAG


def test_valid_rag_response():
    """Accept a valid response with a supporting citation."""
    response = RespuestaRAG.model_validate({
        "status": "found",
        "description": "Test explanation",
        "citations": [{
            "nregistro": "123",
            "seccion": "4.5",
            "literal_quote": "This is a sample quotation for testing"
        }]
    })

    assert response.status == "found"
    assert len(response.citations) == 1


def test_not_found_response():
    """Accept a response when no evidence is found."""
    response = RespuestaRAG.model_validate({
        "status": "not_found",
        "description": "No relevant evidence was found.",
        "citations": []
    })

    assert response.status == "not_found"
    assert response.citations == []


def test_invalid_status():
    """Reject statuses not allowed by the RAG specification."""
    with pytest.raises(ValidationError):
        RespuestaRAG.model_validate({
            "status": "safe",
            "description": "Unsupported conclusion",
            "citations": []
        })


def test_missing_citation_field():
    """Reject a citation missing its source section."""
    with pytest.raises(ValidationError):
        RespuestaRAG.model_validate({
            "status": "found",
            "description": "Test explanation",
            "citations": [{
                "nregistro": "123",
                "literal_quote": "This is a sample quotation for testing"
            }]
        })

def test_found_without_citations():
    """Reject a finding without supporting evidence."""
    with pytest.raises(ValidationError):
        RespuestaRAG.model_validate({
            "status": "found",
            "description": "Potential interaction found.",
            "citations": []
        })


def test_not_found_with_citations():
    """Reject unsupported citations in a not_found response."""
    with pytest.raises(ValidationError):
        RespuestaRAG.model_validate({
            "status": "not_found",
            "description": "No relevant evidence found.",
            "citations": [{
                "nregistro": "123",
                "seccion": "4.5",
                "literal_quote": "This is a sample quotation for testing"
            }]
        })
