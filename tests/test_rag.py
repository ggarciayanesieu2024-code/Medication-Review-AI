"""Exercise retrieval-to-output checks using fake sources and Groq responses."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import llm
from src.Rag import rag


@pytest.fixture
def pipeline(monkeypatch):
    """Keep real prompt building, client parsing, models, and citation checks."""
    fragment = {
        "nregistro": "0123",
        "nombre": "Medicamento ficticio",
        "seccion": "4.5.1",
        "titulo": "Interacciones",
        "texto": "La warfarina puede aumentar el efecto de otros medicamentos anticoagulantes.",
    }
    search = Mock(return_value=[fragment])
    monkeypatch.setattr(rag, "search_fragments", search)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = Mock()
    monkeypatch.setattr(llm, "Groq", Mock(return_value=client))
    answer = {
        "status": "found",
        "interaction_detected": True,
        "description": "Información para revisión profesional.",
        "citations": [{
            "nregistro": "0123",
            "seccion": "4.5.1",
            "literal_quote": "puede aumentar el efecto de otros medicamentos anticoagulantes",
        }],
    }
    return search, client.chat.completions.create, answer


def api_response(answer, finish_reason="stop"):
    """Create the response envelope read by the actual shared client."""
    content = json.dumps(answer) if isinstance(answer, dict) else answer
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason,
    )])


def test_general_answer_passes_all_checks(pipeline):
    search, generate, answer = pipeline
    generate.return_value = api_response(answer)

    result = rag.answer_question("Consulta", nregistro="0123", sections=["4.5"])

    assert result["status"] == "found"
    assert result["response"] == answer
    assert result["reason"] is None
    search.assert_called_once_with(
        question="Consulta", nregistro="0123", sections=["4.5"], k=rag.K_RESULTS,
    )
    messages = generate.call_args.kwargs["messages"]
    assert messages[0]["content"] == rag.GENERAL_SYSTEM_PROMPT
    assert json.loads(messages[1]["content"])["fragments"][0]["nregistro"] == "0123"


def test_interaction_uses_filtered_sources_and_prompt(pipeline):
    search, generate, answer = pipeline
    generate.return_value = api_response(answer)

    result = rag.review_interaction("0123", "warfarina")

    assert result["status"] == "found"
    assert search.call_args.kwargs["sections"] == ["4.5"]
    assert generate.call_args.kwargs["messages"][0]["content"] == rag.INTERACTION_SYSTEM_PROMPT


@pytest.mark.parametrize("empty_search", [True, False])
def test_no_evidence_skips_generation(pipeline, empty_search):
    search, generate, _ = pipeline
    if empty_search:
        search.return_value = []

    result = rag.review_interaction("0123", "ingredient_not_mentioned")

    assert result["status"] == "not_found"
    assert result["reason"] == "no_evidence"
    assert result["response"]["citations"] == []
    generate.assert_not_called()


def test_invalid_schema_retries_once_then_succeeds(pipeline):
    _, generate, answer = pipeline
    invalid = {"status": "found", "description": "Unsupported", "citations": []}
    generate.side_effect = [api_response(invalid), api_response(answer)]

    result = rag.answer_question("Consulta")

    assert result["status"] == "found"
    assert generate.call_count == 2
    assert len(generate.call_args.kwargs["messages"]) == 3


def test_repeated_invalid_json_is_rejected(pipeline):
    _, generate, _ = pipeline
    generate.return_value = api_response("not JSON")

    result = rag.answer_question("Consulta")

    assert result["status"] == "rejected"
    assert result["reason"] == "invalid_response"
    assert result["response"] is None
    assert generate.call_count == 2


@pytest.mark.parametrize("field", ["description", "quote"])
def test_forbidden_phrase_rejects_entire_answer(pipeline, field):
    _, generate, answer = pipeline
    unsafe = "REDUZCA   LA\nDOSIS"
    if field == "description":
        answer["description"] = unsafe
    else:
        answer["citations"][0]["literal_quote"] = unsafe
    generate.return_value = api_response(answer)

    result = rag.answer_question("Consulta")

    assert result["reason"] == "forbidden_phrase"
    assert result["status"] == "rejected"
    assert result["response"] is None


@pytest.mark.parametrize("change", [
    {"nregistro": "999"},
    {"seccion": "4.5"},
    {"literal_quote": "Esta cita inventada no figura en el documento"},
])
def test_bad_citation_rejects_entire_answer(pipeline, change):
    _, generate, answer = pipeline
    answer["citations"][0].update(change)
    generate.return_value = api_response(answer)

    result = rag.answer_question("Consulta")

    assert result["status"] == "rejected"
    assert result["reason"] == "invalid_citation"
    assert result["response"] is None


def test_model_not_found_remains_distinct(pipeline):
    _, generate, _ = pipeline
    generate.return_value = api_response({
        "status": "not_found", "description": "Información no encontrada.", "citations": [],
    })

    result = rag.answer_question("Consulta")

    assert result["status"] == "not_found"
    assert result["reason"] == "insufficient_evidence"


def test_missing_key_is_an_error_not_missing_evidence(pipeline, monkeypatch):
    _, generate, _ = pipeline
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    result = rag.answer_question("Consulta")

    assert result["status"] == "error"
    assert result["reason"] == "generation_failed"
    assert result["response"] is None
    generate.assert_not_called()


def test_token_truncation_is_rejected_without_schema_retry(pipeline):
    _, generate, answer = pipeline
    generate.return_value = api_response(answer, finish_reason="length")

    result = rag.answer_question("Consulta")

    assert result["status"] == "rejected"
    assert result["reason"] == "invalid_response"
    assert generate.call_count == 1


@pytest.mark.parametrize("detected", [True, False])
def test_interaction_evidence_keeps_its_explicit_classification(pipeline, detected):
    _, generate, answer = pipeline
    answer["interaction_detected"] = detected
    generate.return_value = api_response(answer)
    result = rag.review_interaction("0123", "warfarina")
    assert result["status"] == "found"
    assert result["response"]["interaction_detected"] is detected
    assert result["response"]["citations"] == answer["citations"]


def test_interaction_evidence_without_classification_is_rejected(pipeline):
    _, generate, answer = pipeline
    del answer["interaction_detected"]
    generate.return_value = api_response(answer)
    result = rag.review_interaction("0123", "warfarina")
    assert result["status"] == "rejected"
    assert result["reason"] == "invalid_response"
    assert result["response"] is None


def test_multiple_verified_citations_are_preserved(pipeline):
    _, generate, answer = pipeline
    answer["citations"].append({
        "nregistro": "0123", "seccion": "4.5.1",
        "literal_quote": "La warfarina puede aumentar el efecto de otros medicamentos",
    })
    generate.return_value = api_response(answer)
    result = rag.review_interaction("0123", "warfarina")
    assert result["status"] == "found"
    assert result["response"]["citations"] == answer["citations"]
