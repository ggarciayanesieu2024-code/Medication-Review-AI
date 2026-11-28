"""Check evaluation scoring without an index, downloads or live API calls."""

import sys
from types import SimpleNamespace

import pytest

from src.Evaluation.evaluar import evaluate_cases, score_retrieval


def test_source_scoring_counts_unique_exact_sections():
    sources = [{"nregistro": "1", "seccion": "4.5"},
               {"nregistro": "1", "seccion": "4.5.1"}]
    assert score_retrieval([sources[0], sources[0]], sources) == {
        "source_hit": True, "source_recall": 0.5,
    }
    assert score_retrieval([], sources) == {"source_hit": False, "source_recall": 0.0}
    assert score_retrieval(sources, []) == {"source_hit": None, "source_recall": None}


def test_empty_benchmark_has_no_invented_scores():
    run = evaluate_cases([])
    assert run["results"] == []
    assert run["summary"]["source_hit"] is None
    assert run["summary"]["source_hit_cases"] == 0


def test_retrieval_only_never_generates(monkeypatch):
    source = {"nregistro": "1", "seccion": "4.5"}
    calls = []

    def prepare(question, **kwargs):
        calls.append((question, kwargs))
        return {"status": "ready", "fragments": [source]}

    def forbidden(*args, **kwargs):
        pytest.fail("Retrieval-only evaluation must not generate an answer.")

    monkeypatch.setitem(sys.modules, "src.Rag.rag", SimpleNamespace(
        prepare_general_search=prepare, prepare_interaction_search=forbidden,
        answer_question=forbidden, review_interaction=forbidden,
    ))
    case = {"id": "a", "question": "Question", "expected_sources": [source],
            "expected_status": "found"}
    run = evaluate_cases([case], k=5)
    assert calls[0][1]["k"] == 5
    assert run["summary"]["source_hit"] == 1.0
    assert run["summary"]["status_match"] is None


def test_generation_failures_never_become_correct_negatives(monkeypatch):
    def answer(question, **kwargs):
        raise RuntimeError("Unavailable retrieval resource")

    def interaction(*args, **kwargs):
        return {"status": "rejected", "reason": "invalid_citation",
                "fragments": [], "response": None}

    monkeypatch.setitem(sys.modules, "src.Rag.rag", SimpleNamespace(
        prepare_general_search=None, prepare_interaction_search=None,
        answer_question=answer, review_interaction=interaction,
    ))
    cases = [
        {"id": "a", "question": "Question", "expected_status": "not_found"},
        {"id": "b", "task": "interaction", "nregistro": "1", "ingredient": "x",
         "expected_interaction_detected": False},
    ]
    run = evaluate_cases(cases, generate=True)
    assert run["summary"]["errors"] == 1
    assert run["summary"]["rejected"] == 1
    assert run["results"][0]["status_match"] is False
    assert run["results"][1]["interaction_match"] is False
    assert run["summary"]["source_hit_cases"] == 0


def test_invalid_labels_fail_before_execution():
    with pytest.raises(ValueError, match="boolean"):
        evaluate_cases([{"id": "a", "question": "Q", "expected_interaction_detected": "false"}])
    with pytest.raises(ValueError, match="unique"):
        evaluate_cases([{"id": "a"}, {"id": "a"}])
