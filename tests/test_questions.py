"""Patient-question routing and evidence checks without API calls or an index."""

import json
from dataclasses import asdict
from unittest.mock import Mock

import pytest

from src import llm, main
from src.Rag import rag
from src.Reports.informe import Patient, generate_report


@pytest.fixture
def patient():
    return Patient(id="QUESTION_TEST", edad=78, grupo="desarrollo", medicamentos=[{
        "nregistro": "68339", "nombre": "Paracetamol Kern Pharma 1 g",
        "dosis": "Según ficha del paciente", "prescriptor": "Doctor",
    }])


@pytest.fixture
def catalogue():
    return main.load_catalogue()


@pytest.fixture
def no_evidence():
    return Mock(return_value={"status": "not_found", "reason": "no_evidence", "response": None})


@pytest.mark.parametrize("question", [None, "", "  \n "])
def test_blank_question_uses_the_standard_review(patient, catalogue, no_evidence, monkeypatch, question):
    planner = Mock()
    monkeypatch.setattr(llm, "llm_prepare_question", planner)
    expected = main.review_patient(patient, catalogue, use_model=True, review=no_evidence)
    result = main.review_request(patient, catalogue, question, use_model=True, review=no_evidence)
    assert asdict(result.checks) == asdict(expected)
    assert result.question == "" and result.answer is None
    planner.assert_not_called()


def test_new_medicine_question_reuses_checks_and_supplies_the_patient_sheet(
    patient, catalogue, no_evidence, monkeypatch,
):
    planner = Mock(return_value=llm.QuestionPlan(
        intent="new_medicine", search_question="Duplicidad con Antidol",
        proposed_nregistro="85780", medicine_ids=["85780"],
    ))
    answer = Mock(return_value={"status": "found", "reason": None, "response": {
        "status": "found", "answer": "Se identifica un principio activo compartido.",
        "citations": [], "finding_ids": ["F1"],
    }})
    monkeypatch.setattr(llm, "llm_prepare_question", planner)
    monkeypatch.setattr(rag, "answer_patient_question", answer)
    question = "What should be reviewed if I add Antidol 1 g?"
    result = main.review_request(patient, catalogue, question, use_model=True, review=no_evidence)
    assert result.proposed.nregistro == "85780"
    assert [finding.tipo for finding in result.checks.findings] == ["duplicado"]
    assert result.supporting_findings == result.checks.findings
    assert planner.call_args.args[1] == patient.model_dump(mode="json")
    payload = answer.call_args.args
    assert payload[0] == question
    assert payload[2]["edad"] == 78
    assert payload[2]["medicamentos"] == patient.model_dump(mode="json")["medicamentos"]
    assert payload[2]["proposed_medicine"]["nregistro"] == "85780"
    assert set(payload[3]) == {"68339", "85780"}
    report = generate_report(
        patient, result.checks.findings, proposed=result.proposed,
        question=result.question, answer=result.answer,
        answer_citations=result.citations, supporting_findings=result.supporting_findings,
    )
    assert "Doctor's question" in report.markdown
    assert result.answer in report.markdown
    assert "Supporting check" in report.markdown


@pytest.mark.parametrize("plan", [
    llm.QuestionPlan(intent="clarify", search_question="¿Qué ibuprofeno?", clarification="Which product?"),
    llm.QuestionPlan(intent="new_medicine", search_question="Unknown medicine", proposed_nregistro="UNKNOWN"),
    llm.QuestionPlan(intent="new_medicine", search_question="Missing product"),
])
def test_unclear_or_unknown_medicines_require_clarification(patient, catalogue, no_evidence, monkeypatch, plan):
    monkeypatch.setattr(llm, "llm_prepare_question", Mock(return_value=plan))
    answer = Mock()
    monkeypatch.setattr(rag, "answer_patient_question", answer)
    result = main.review_request(patient, catalogue, "Can I add this medicine?", use_model=True, review=no_evidence)
    assert result.clarification
    assert result.proposed is None and result.answer is None
    no_evidence.assert_not_called()
    answer.assert_not_called()


def test_question_without_model_explains_how_to_run_the_standard_review(patient, catalogue):
    result = main.review_request(patient, catalogue, "What should be reviewed?")
    assert "Leave the question empty" in result.clarification
    assert result.answer is None


def test_general_question_preserves_the_question_and_restricts_search(patient, catalogue, no_evidence, monkeypatch):
    monkeypatch.setattr(llm, "llm_prepare_question", Mock(return_value=llm.QuestionPlan(
        intent="medicine_question", search_question="Advertencias del paracetamol", medicine_ids=["68339"],
    )))
    answer = Mock(return_value={"status": "not_found", "reason": "no_evidence", "response": {
        "status": "not_found", "answer": "No supporting information was found.", "citations": [], "finding_ids": [],
    }})
    monkeypatch.setattr(rag, "answer_patient_question", answer)
    question = "What warnings matter for this patient?"
    result = main.review_request(patient, catalogue, question, use_model=True, review=no_evidence)
    assert result.question == question
    assert result.proposed is None
    assert answer.call_args.args[3] == ["68339"]


@pytest.fixture
def question_sources(monkeypatch):
    source = {"nregistro": "68339", "nombre": "Synthetic medicine", "seccion": "4.4",
              "titulo": "Warnings", "texto": "This synthetic passage describes a qualification relevant to the question."}
    search = Mock(return_value=[source])
    monkeypatch.setattr(rag, "search_fragments", search)
    response = llm.PatientQuestionAnswer(
        status="found", answer="The supplied source includes a qualification.",
        citations=[{"nregistro": "68339", "seccion": "4.4",
                    "literal_quote": "This synthetic passage describes a qualification relevant to the question"}],
    )
    generate = Mock(return_value=response)
    monkeypatch.setattr(llm, "llm_answer_patient_question", generate)
    return search, generate, response


def test_patient_question_passes_patient_facts_and_verified_evidence(patient, question_sources):
    search, generate, response = question_sources
    result = rag.answer_patient_question("Original question", "Consulta clara", patient.model_dump(), ["68339"], [], [])
    assert result["status"] == "found"
    assert result["response"]["citations"] == response.model_dump()["citations"]
    search.assert_called_once_with("Consulta clara", nregistro="68339", k=rag.K_RESULTS)
    context = json.loads(generate.call_args.args[0][1]["content"])
    assert context["question"] == "Original question"
    assert context["patient"]["edad"] == 78
    assert context["fragments"][0]["nregistro"] == "68339"


@pytest.mark.parametrize("failure", ["citation", "finding", "treatment"])
def test_unsupported_or_unsafe_question_answers_are_rejected(patient, question_sources, failure):
    _, _, response = question_sources
    if failure == "citation":
        response.citations[0].nregistro = "OTHER"
    elif failure == "finding":
        response.finding_ids = ["F999"]
    else:
        response.answer = "Start taking the proposed medicine."
    result = rag.answer_patient_question("Question", "Consulta", patient.model_dump(), ["68339"], [], [])
    assert result["status"] == "rejected" and result["response"] is None


def test_rule_only_question_can_cite_a_checked_finding(patient, question_sources):
    search, _, response = question_sources
    search.return_value = []
    response.citations = []
    response.finding_ids = ["F1"]
    result = rag.answer_patient_question("Question", "Consulta", patient.model_dump(), ["68339"],
                                         [{"tipo": "cascada", "descripcion": "Synthetic cascade hypothesis"}], [])
    assert result["status"] == "found"
    assert result["response"]["finding_ids"] == ["F1"]


def test_no_question_evidence_skips_generation(patient, question_sources):
    search, generate, _ = question_sources
    search.return_value = []
    result = rag.answer_patient_question("Question", "Consulta", patient.model_dump(), ["68339"], [], [])
    assert result["status"] == "not_found"
    generate.assert_not_called()



def test_patient_clinical_context_is_preserved_in_the_answer_request(patient, catalogue, no_evidence, monkeypatch):
    patient.contexto_clinico = "Context explicitly supplied by the doctor."
    monkeypatch.setattr(llm, "llm_prepare_question", Mock(return_value=llm.QuestionPlan(
        intent="medicine_question", search_question="Consulta", medicine_ids=["68339"],
    )))
    answer = Mock(return_value={"status": "not_found", "reason": "no_evidence", "response": {
        "answer": "No supporting information.", "citations": [], "finding_ids": [],
    }})
    monkeypatch.setattr(rag, "answer_patient_question", answer)
    main.review_request(patient, catalogue, "Question", use_model=True, review=no_evidence)
    assert answer.call_args.args[2]["contexto_clinico"] == patient.contexto_clinico


def test_unknown_patient_registrations_are_not_silently_skipped(patient, catalogue):
    patient.medicamentos[0].nregistro = "UNKNOWN"
    result = main.review_request(patient, catalogue)
    assert "UNKNOWN" in result.clarification
    assert result.checks.findings == []


def test_planner_failure_does_not_produce_a_medical_answer(patient, catalogue, monkeypatch):
    monkeypatch.setattr(llm, "llm_prepare_question", Mock(side_effect=RuntimeError("unavailable")))
    result = main.review_request(patient, catalogue, "Question", use_model=True)
    assert result.clarification and result.answer is None
    assert result.checks.could_not_check
