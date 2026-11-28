"""Check selected notebook parameters propagate without changing project defaults."""

from unittest.mock import Mock

from src import llm, main
from src.config import PATIENTS_DIR
from src.Rag import rag


OPTIONS = {"model": "test-model", "temperature": 0.3}


def patient_and_catalogue():
    catalogue = main.load_catalogue()

    patient = main.Patient(
        id="TEST_NOTEBOOK",
        edad=78,
        grupo="desarrollo",
        medicamentos=[
            {
                "nregistro": "68339",
                "nombre": catalogue["68339"].nombre,
                "dosis": "1 al día",
                "prescriptor": "Doctor A",
            },
            {
                "nregistro": "59718",
                "nombre": catalogue["59718"].nombre,
                "dosis": "1 al día",
                "prescriptor": "Doctor B",
            },
        ],
    )

    return patient, catalogue


def test_standard_patient_review_uses_notebook_retrieval_and_generation_options(monkeypatch):
    patient, catalogue = patient_and_catalogue()
    review = Mock(return_value={"status": "not_found", "reason": "no_evidence", "response": None})
    monkeypatch.setattr(rag, "review_interaction", review)
    result = main.review_request(patient, catalogue, use_model=True, k=7, generation_options=OPTIONS)
    assert len(result.checks.log) == review.call_count == 2
    assert all(call.kwargs == {"k": 7, "generation_options": OPTIONS} for call in review.call_args_list)


def test_patient_question_passes_options_to_planning_checks_and_final_answer(monkeypatch):
    patient, catalogue = patient_and_catalogue()
    planner = Mock(return_value=llm.QuestionPlan(
        intent="medicine_question", search_question="Consulta", medicine_ids=["68339"],
    ))
    review = Mock(return_value={"status": "not_found", "reason": "no_evidence", "response": None})
    answer = Mock(return_value={"status": "not_found", "reason": "no_evidence", "response": {
        "answer": "No evidence supplied.", "citations": [], "finding_ids": [],
    }})
    monkeypatch.setattr(llm, "llm_prepare_question", planner)
    monkeypatch.setattr(rag, "review_interaction", review)
    monkeypatch.setattr(rag, "answer_patient_question", answer)
    main.review_request(patient, catalogue, "Question", use_model=True, k=7, generation_options=OPTIONS)
    assert planner.call_args.kwargs == OPTIONS
    assert review.call_args.kwargs == {"k": 7, "generation_options": OPTIONS}
    assert answer.call_args.kwargs == {"k": 7, "generation_options": OPTIONS}


def test_patient_answer_retrieves_with_the_selected_limit(monkeypatch):
    patient, _ = patient_and_catalogue()
    search = Mock(return_value=[])
    monkeypatch.setattr(rag, "search_fragments", search)
    rag.answer_patient_question("Question", "Consulta", patient.model_dump(), ["68339"], [], [], k=7)
    search.assert_called_once_with("Consulta", nregistro="68339", k=7)


def test_generation_options_reach_client_without_mutating_defaults(monkeypatch):
    from types import SimpleNamespace
    import json

    before = (llm.GROQ_MODEL, llm.TEMPERATURE)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=json.dumps({
            "status": "not_found", "description": "No evidence supplied.", "citations": [],
        })),
    )]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "Groq", Mock(return_value=client))
    llm.llm_medication_review([], **OPTIONS)
    assert create.call_args.kwargs["model"] == "test-model"
    assert create.call_args.kwargs["temperature"] == 0.3
    llm.llm_medication_review([])
    assert create.call_args.kwargs["model"] == before[0]
    assert create.call_args.kwargs["temperature"] == before[1]
    assert (llm.GROQ_MODEL, llm.TEMPERATURE) == before
