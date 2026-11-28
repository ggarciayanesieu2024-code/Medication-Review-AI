"""Run labelled RAG cases through the real pipeline and score supplied labels.

The notebook handles exploration and manual answer ratings. This script runs
repeatable batches; it does not invent reference answers or clinical labels.
"""

from datetime import datetime, timezone
from time import perf_counter

from src.config import EMBEDDING_MODEL, GROQ_MODEL, K_RESULTS, TEMPERATURE


def score_retrieval(fragments: list[dict], expected_sources: list[dict]) -> dict:
    """Score exact medicine/section labels, keeping subsection IDs distinct.

    Hit means at least one labelled source was retrieved. Recall measures the
    fraction of labelled sources retrieved. Neither measures clinical accuracy
    or whether a particular passage supports the answer.
    """
    expected = {(item["nregistro"], item["seccion"]) for item in expected_sources}
    if not expected:
        return {"source_hit": None, "source_recall": None}
    retrieved = {(item["nregistro"], item["seccion"]) for item in fragments}
    matched = expected & retrieved
    return {"source_hit": bool(matched), "source_recall": len(matched) / len(expected)}


def summarise_results(results: list[dict]) -> dict:
    """Average scored cases only and show each metric's denominator."""
    summary = {
        "cases": len(results),
        "errors": sum(row["status"] == "error" for row in results),
        "rejected": sum(row["status"] == "rejected" for row in results),
    }
    for metric in ("source_hit", "source_recall", "status_match", "interaction_match"):
        values = [row[metric] for row in results if row.get(metric) is not None]
        summary[metric] = sum(values) / len(values) if values else None
        summary[metric + "_cases"] = len(values)
    return summary


def evaluate_cases(
    cases: list[dict], *, k: int = K_RESULTS, generate: bool = False,
    generation_options: dict | None = None,
) -> dict:
    """Run general or interaction cases; generation is explicitly opt-in.

    General cases require id and question. Interaction cases require id,
    task='interaction', nregistro and ingredient. Optional labels are
    expected_sources, expected_status and expected_interaction_detected.
    Exceptions, rejected answers and missing evidence remain distinct.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation case IDs must be unique.")

    # Check definitions before making retrieval or API requests.
    for case in cases:
        task = case.get("task", "general")
        if task not in ("general", "interaction"):
            raise ValueError(f"Unknown task for {case['id']}: {task}")
        required = ("nregistro", "ingredient") if task == "interaction" else ("question",)
        if any(not isinstance(case.get(key), str) or not case[key].strip() for key in required):
            raise ValueError(f"Case {case['id']} requires nonempty {required}.")
        for source in case.get("expected_sources", []):
            if not all(isinstance(source.get(key), str) and source[key].strip()
                       for key in ("nregistro", "seccion")):
                raise ValueError("Expected sources need string nregistro and seccion values.")
        if "expected_status" in case and case["expected_status"] not in ("found", "not_found"):
            raise ValueError("Expected status must be found or not_found.")
        if "expected_interaction_detected" in case and type(case["expected_interaction_detected"]) is not bool:
            raise ValueError("Expected interaction classification must be a boolean.")

    started_at = datetime.now(timezone.utc).isoformat()
    generation_options = generation_options or {}
    options = {"generation_options": generation_options} if generate and generation_options else {}
    results = []
    if cases:
        from src.Rag.rag import (
            answer_question, prepare_general_search, prepare_interaction_search, review_interaction,
        )

    for case in cases:
        started = perf_counter()
        row = {
            "id": case["id"], "case": case, "k": k, "status": "error",
            "reason": None, "fragments": [], "response": None,
            "source_hit": None, "source_recall": None,
            "status_match": None, "interaction_match": None,
        }
        try:
            if case.get("task", "general") == "interaction":
                run = review_interaction if generate else prepare_interaction_search
                output = run(case["nregistro"], case["ingredient"], k=k, **options)
            else:
                run = answer_question if generate else prepare_general_search
                output = run(case["question"], nregistro=case.get("nregistro"),
                             sections=case.get("sections"), k=k, **options)
            row.update({key: output.get(key) for key in ("status", "reason", "fragments", "response")})
            row.update(score_retrieval(row["fragments"], case.get("expected_sources", [])))
            if generate and "expected_status" in case:
                row["status_match"] = row["status"] == case["expected_status"]
            if generate and "expected_interaction_detected" in case:
                actual = (row["response"] or {}).get("interaction_detected")
                row["interaction_match"] = (
                    row["status"] == "found" and actual is case["expected_interaction_detected"]
                )
        except Exception as exc:
            row.update(status="error", reason=type(exc).__name__)
            if generate and "expected_status" in case:
                row["status_match"] = False
            if generate and "expected_interaction_detected" in case:
                row["interaction_match"] = False
        row["seconds"] = round(perf_counter() - started, 3)
        results.append(row)

    return {
        "started_at": started_at,
        "settings": {"k": k, "generate": generate, "embedding_model": EMBEDDING_MODEL,
                     "groq_model": generation_options.get("model") or GROQ_MODEL,
                     "temperature": generation_options.get("temperature")
                     if generation_options.get("temperature") is not None else TEMPERATURE},
        "summary": summarise_results(results), "results": results,
    }
