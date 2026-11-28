"""Tests for src/main.py: the checks behind both tabs of the screen.

A stand-in replaces src.Rag.rag.review_interaction, so no search index or Groq key
is needed (Person 3's own tests cover the real pipeline). The rules tests use
Person 2's real src/Cascades/reglas.py with nomenclator codes written in the test, so
they do not need data/processed/prescriptions.json.
"""
import json
import builtins
from dataclasses import replace

import pytest

from src import main
from src.config import FRAGMENTS_FILE, PATIENTS_DIR, PROJECT_ROOT
from src.Reports.informe import FOOTER, generate_report
from src.Reports.informe import Medication, Patient
from src.main import (
    check_new_medicine,
    load_catalogue,
    model_status,
    planned_questions,
    review_patient,
    rules_status,
)

CATALOGUE = load_catalogue()


def _patient(meds, pid="T"):
    return Patient(id=pid, edad=80, grupo="desarrollo", medicamentos=[
        {"nregistro": n, "nombre": CATALOGUE[n].nombre, "dosis": "1 al día", "prescriptor": who}
        for n, who in meds
    ])


def _proposed(nregistro, prescriber="Dra. Pérez (Atención Primaria)"):
    return Medication(nregistro=nregistro, nombre=CATALOGUE[nregistro].nombre,
                      dosis="1 al día", prescriptor=prescriber)


def _found(nregistro):
    return {"status": "found", "reason": None, "response": {
        "status": "found",
        "interaction_detected": True,
        "description": "La ficha describe un posible aumento del riesgo de sangrado.",
        "citations": [{"nregistro": nregistro, "seccion": "4.5",
                       "literal_quote": "puede aumentar el riesgo de hemorragia en pacientes tratados"}],
    }}


class FakeModel:
    """Records every question and answers from a table; default is not_found."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []

    def __call__(self, nregistro, ingredient):
        self.calls.append((nregistro, ingredient))
        answer = self.answers.get((nregistro, ingredient), "not_found")
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, dict):
            return answer
        if answer == "found":
            return _found(nregistro)
        if answer in ("rejected", "error"):
            reason = "forbidden_phrase" if answer == "rejected" else "generation_failed"
            return {"status": answer, "reason": reason, "response": None}
        return {"status": "not_found", "reason": "no_evidence", "response": None}


# ---------------------------------------------------------------- medicine list

def _all_patient_medicines():
    paths = sorted(path for path in PATIENTS_DIR.glob("*.json") if path.name != "patient_scenarios.json") + [PROJECT_ROOT / "data" / "demo" / "pacientes_demo.json"]
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for patient in data if isinstance(data, list) else [data]:
            yield from patient["medicamentos"]


def test_every_patient_medicine_is_in_the_list():
    missing = {m["nregistro"] for m in _all_patient_medicines()} - set(CATALOGUE)
    assert not missing, f"add these registration numbers to data/catalogo_medicamentos.json: {missing}"


def test_every_entry_has_an_ingredient():
    for entry in CATALOGUE.values():
        assert entry.principios_activos and all(i.strip() for i in entry.principios_activos)


def test_list_matches_the_shared_cima_file():
    """Runs only on a computer that has the shared CIMA file."""
    if not FRAGMENTS_FILE.exists():
        pytest.skip("shared CIMA file not in data/processed/")
    fragments = json.loads(FRAGMENTS_FILE.read_text(encoding="utf-8"))
    with_sheet = {f["nregistro"] for f in fragments}
    marked = {n for n, e in CATALOGUE.items() if e.ficha}
    assert marked == with_sheet, (
        f"missing from the list: {with_sheet - marked}; marked but no sheet: {marked - with_sheet}"
    )


# ---------------------------------------------------------------- new medicine, model part

def test_both_directions_are_asked_and_found_becomes_a_finding():
    patient = _patient([("63064", "Dr. Gómez (Cardiología)")])  # warfarin
    model = FakeModel({("88372", "warfarina"): "found"})          # Ibudol's sheet mentions warfarin
    result = check_new_medicine(patient, _proposed("88372"), CATALOGUE, review=model)

    assert model.calls == [("88372", "warfarina"), ("63064", "ibuprofeno")]
    [finding] = result.findings
    assert finding.tipo == "interaccion" and finding.cita.nregistro == "88372"
    assert "Dr. Gómez (Cardiología)" in finding.pregunta_medico
    assert result.checked_without_evidence == ["Aldocumar 5 mg comprimidos sheet, searched for ibuprofeno"]
    assert result.could_not_check == []


def test_documented_absence_is_information_not_an_interaction_alert():
    patient = _patient([("63064", "Doctor")])
    answer = _found("88372")
    answer["response"]["interaction_detected"] = False
    answer["response"]["description"] = "La fuente no describe interacci?n en las condiciones indicadas."
    model = FakeModel({("88372", "warfarina"): answer})
    result = check_new_medicine(patient, _proposed("88372"), CATALOGUE, review=model)
    [finding] = result.findings
    assert finding.tipo == "informacion"
    assert finding.all_citations
    report = generate_report(patient, result.findings)
    assert report.counts["interaccion"] == 0
    assert "Documented information (not interaction alerts)" in report.markdown
    assert answer["response"]["citations"][0]["literal_quote"] in report.markdown


def test_found_without_interaction_classification_is_not_an_alert():
    patient = _patient([("63064", "Doctor")])
    answer = _found("88372")
    del answer["response"]["interaction_detected"]
    result = check_new_medicine(
        patient, _proposed("88372"), CATALOGUE,
        review=FakeModel({("88372", "warfarina"): answer}),
    )
    assert result.findings == []
    assert any("expected format" in item for item in result.could_not_check)


def test_all_rag_citations_reach_the_report():
    patient = _patient([("63064", "Doctor")])
    answer = _found("88372")
    answer["response"]["citations"].append({
        "nregistro": "88372", "seccion": "4.5.1",
        "literal_quote": "Segunda cita que conserva las condiciones descritas en la fuente",
    })
    result = check_new_medicine(
        patient, _proposed("88372"), CATALOGUE,
        review=FakeModel({("88372", "warfarina"): answer}),
    )
    [finding] = result.findings
    assert len(finding.all_citations) == 2
    report = generate_report(patient, result.findings, proposed=_proposed("88372"))
    assert [block.text for block in report.blocks if block.kind == "quote"] == [
        citation["literal_quote"] for citation in answer["response"]["citations"]
    ]


def test_default_review_delegates_to_shared_rag(monkeypatch):
    from src.Rag import rag
    patient = _patient([("63064", "Doctor")])
    model = FakeModel()
    monkeypatch.setattr(rag, "review_interaction", model)
    check_new_medicine(patient, _proposed("88372"), CATALOGUE)
    assert model.calls == [("88372", "warfarina"), ("63064", "ibuprofeno")]


def test_combination_medicine_is_searched_for_each_ingredient():
    patient = _patient([("65222", "Dr. Ibáñez (Urgencias)")])  # amoxicillin + clavulanic acid
    model = FakeModel()
    check_new_medicine(patient, _proposed("63064"), CATALOGUE, review=model)
    assert ("63064", "amoxicilina") in model.calls
    assert ("63064", "ácido clavulánico") in model.calls
    assert ("65222", "warfarina") in model.calls


def test_shared_ingredient_does_not_hide_other_combination_ingredients():
    catalogue = {**CATALOGUE, "combo": main.CatalogueEntry(
        nregistro="combo", nombre="Synthetic combination",
        principios_activos=["paracetamol", "ibuprofeno"], ficha=True,
    )}
    patient = _patient([("68339", "Doctor")])
    proposed = Medication(nregistro="combo", nombre="Synthetic combination",
                          dosis="Not specified", prescriptor="Doctor")
    model = FakeModel()
    result = check_new_medicine(patient, proposed, catalogue, review=model)
    assert model.calls == [
        ("combo", "paracetamol"), ("68339", "paracetamol"), ("68339", "ibuprofeno"),
    ]
    assert any(finding.tipo == "duplicado" for finding in result.findings)
    assert len(result.log) == planned_questions(patient, proposed, catalogue) == 3


def test_current_medicine_without_sheet_is_searched_in_one_direction():
    patient = _patient([("07383002", "Dra. Medina (Atención Primaria)")])
    # Simulate missing coverage independently of the real shared CIMA file.
    catalogue = {**CATALOGUE, "07383002": replace(CATALOGUE["07383002"], ficha=False)}
    model = FakeModel()
    result = check_new_medicine(patient, _proposed("55211"), catalogue, review=model)
    assert model.calls == [("55211", "sitagliptina")]
    assert any("only the Dianben" in item for item in result.could_not_check)


def test_proposed_medicine_without_sheet_is_looked_for_in_the_other_sheets():
    patient = _patient([("63064", "Dr. Gómez (Cardiología)")])
    catalogue = {**CATALOGUE, "07383002": replace(CATALOGUE["07383002"], ficha=False)}
    model = FakeModel()
    result = check_new_medicine(patient, _proposed("07383002"), catalogue, review=model)
    assert model.calls == [("63064", "sitagliptina")]
    assert any("only the Aldocumar" in item for item in result.could_not_check)


def test_blocked_answers_and_failures_are_never_nothing_found():
    patient = _patient([("63064", "Dr. Gómez (Cardiología)"), ("65461", "Dr. López (Cardiología)")])
    model = FakeModel({
        ("88372", "warfarina"): "rejected",
        ("63064", "ibuprofeno"): "error",
        ("88372", "amlodipino"): RuntimeError("index missing"),
    })
    result = check_new_medicine(patient, _proposed("88372"), CATALOGUE, review=model)
    assert len(result.could_not_check) == 3
    assert not any("warfarina" in item for item in result.checked_without_evidence)
    assert any("safety check" in item for item in result.could_not_check)
    assert any("could not be reached" in item for item in result.could_not_check)
    assert {e["status"] for e in result.log} == {"rejected", "error", "exception", "not_found"}


def test_progress_reaches_the_end():
    seen = []
    patient = _patient([("63064", "Dr. Gómez"), ("65461", "Dr. López")])
    proposed = _proposed("88372")
    check_new_medicine(patient, proposed, CATALOGUE, review=FakeModel(),
                       progress=lambda done, total, label: seen.append((done, total)))
    assert seen[-1] == (4, 4) == (planned_questions(patient, proposed, CATALOGUE),) * 2


# ---------------------------------------------------------------- without rules or model

def test_without_rules_duplicates_use_names_and_the_rest_is_not_checked():
    patient = _patient([("68339", "Dra. Martín (Traumatología)")])  # paracetamol
    model = FakeModel()
    result = check_new_medicine(patient, _proposed("85780"), CATALOGUE, review=model)  # Antidol
    assert model.calls == []
    [finding] = result.findings
    assert finding.tipo == "duplicado" and "paracetamol" in finding.descripcion
    assert result.unchecked == {"advertencia_atc", "cascada"}


def test_without_model_no_question_is_asked_and_interactions_are_not_checked():
    patient = _patient([("63064", "Dr. Gómez")])
    model = FakeModel()
    result = check_new_medicine(patient, _proposed("88372"), CATALOGUE, use_model=False, review=model)
    assert model.calls == []
    assert "interaccion" in result.unchecked


def test_review_asks_about_every_pair():
    patient = _patient([("63064", "Dr. Gómez"), ("65461", "Dr. López"), ("88372", "Dra. Navarro")])
    model = FakeModel({("88372", "warfarina"): "found"})
    result = review_patient(patient, CATALOGUE, use_model=True, review=model)
    assert len(model.calls) == planned_questions(patient, None, CATALOGUE) == 6
    [finding] = [f for f in result.findings if f.tipo == "interaccion"]
    assert "Aldocumar" in finding.pregunta_medico and "Ibudol" in finding.pregunta_medico


# ---------------------------------------------------------------- rules (Person 2)

def _real_rules():
    try:
        from src.Cascades import reglas
    except SyntaxError as exc:
        pytest.skip(f"src/Cascades/reglas.py does not load yet ({exc.msg}); fix it to run these tests")
    return reglas


# Codes from Person 2's tests (checked in the nomenclator); ibuprofen and
# furosemide use invented ingredient codes with real ATC codes, as they do.
DRUG_INFO = {
    "68339": {"atc": ["N02BE01"], "active_ingredients": ["350"]},
    "85780": {"atc": ["N02BE01"], "active_ingredients": ["350"]},
    "63064": {"atc": ["B01AA03"], "active_ingredients": ["4411"]},
    "65222": {"atc": ["J01CR02"], "active_ingredients": ["160", "5418"]},
    "65461": {"atc": ["C08CA01"], "active_ingredients": ["7177"]},
    "55211": {"atc": ["A10BA02"], "active_ingredients": ["4379"]},
    "88372": {"atc": ["M01AE01"], "active_ingredients": ["T-IBU"]},
    "64140": {"atc": ["C03CA01"], "active_ingredients": ["T-FUR"]},
}


@pytest.fixture
def rule_data():
    reglas = _real_rules()
    return DRUG_INFO, reglas.load_cascades()


def test_review_with_rules_finds_the_duplicate(rule_data):
    patient = _patient([("68339", "Dra. Martín"), ("85780", "Dr. Ruiz")])
    result = review_patient(patient, CATALOGUE, rule_data=rule_data)
    assert [f.tipo for f in result.findings] == ["duplicado"]
    assert result.unchecked == {"interaccion"}


def test_rules_use_shared_builder_and_cascade_detector(monkeypatch, rule_data):
    from unittest.mock import Mock
    from src.Cascades import reglas
    build = Mock(wraps=reglas.build_medications)
    cascades = Mock(wraps=reglas.find_cascades)
    monkeypatch.setattr(reglas, "build_medications", build)
    monkeypatch.setattr(reglas, "find_cascades", cascades)
    patient = _patient([("65461", "Doctor")])
    result = check_new_medicine(patient, _proposed("64140"), CATALOGUE,
                                rule_data=rule_data, use_model=False)
    build.assert_called_once()
    cascades.assert_called_once()
    assert [finding.tipo for finding in result.findings] == ["cascada"]


def test_new_medicine_can_complete_a_cascade(rule_data):
    patient = _patient([("65461", "Dr. López (Cardiología)")])  # amlodipine
    result = check_new_medicine(patient, _proposed("64140"), CATALOGUE,
                                rule_data=rule_data, use_model=False)  # furosemide
    [cascade] = [f for f in result.findings if f.tipo == "cascada"]
    assert cascade.descripcion.startswith("C01:")
    assert "The proposed medicine, 'Furosemida Cinfa 40 mg'" in cascade.descripcion
    assert "also takes 'Furosemida" not in cascade.descripcion
    assert "diuretic" in cascade.pregunta_medico.lower()
    assert result.unchecked == {"interaccion"}


def test_new_medicine_check_ignores_findings_between_current_medicines(rule_data):
    patient = _patient([("68339", "Dra. Martín"), ("85780", "Dr. Ruiz")])  # already a duplicate
    result = check_new_medicine(patient, _proposed("63064"), CATALOGUE,
                                rule_data=rule_data, use_model=False)
    assert result.findings == []


def test_medicine_missing_from_the_nomenclator_is_reported(rule_data):
    patient = _patient([("63064", "Dr. Gómez"), ("07383002", "Dra. Medina")])  # Januvia: no codes
    result = review_patient(patient, CATALOGUE, rule_data=rule_data)
    assert any("Januvia" in item and "nomenclator" in item for item in result.could_not_check)


# ---------------------------------------------------------------- report

def test_check_report_lists_the_proposed_medicine_and_unfinished_checks():
    patient = _patient([("63064", "Dr. Gómez (Cardiología)")])
    proposed = _proposed("88372")
    model = FakeModel({("88372", "warfarina"): "found", ("63064", "ibuprofeno"): "rejected"})
    result = check_new_medicine(patient, proposed, CATALOGUE, review=model)
    report = generate_report(
        patient, result.findings, checked_without_evidence=result.checked_without_evidence,
        could_not_check=result.could_not_check, proposed=proposed, unchecked_types=result.unchecked,
    )
    md = report.markdown
    assert "Medicine check before prescribing" in md
    assert "Proposed medicine, not yet prescribed: Dra. Pérez (Atención Primaria)" in md
    assert "## Checks that could not be completed" in md
    assert "Not checked: this part of the system was not available" in md
    assert "Ibudol 400 mg cápsulas blandas (reg. 88372), section 4.5" in md
    assert report.unchecked == {"advertencia_atc", "cascada"}
    assert report.file_stem.startswith("comprobacion_T_")
    assert report.markdown.rstrip().endswith(f"*{FOOTER}*")


# ---------------------------------------------------------------- status checks

def test_model_status_without_libraries(monkeypatch):
    original_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "src.Rag":
            raise ModuleNotFoundError("No module named 'groq'", name="groq")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    ready, message = model_status()
    assert not ready and "groq" in message


def test_model_status_without_index(monkeypatch, tmp_path):
    ready, message = model_status(tmp_path / "missing")
    assert not ready and "index" in message


def test_model_status_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    ready, message = model_status(tmp_path)
    assert not ready and "GROQ_API_KEY" in message


def test_model_status_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert model_status(tmp_path) == (True, "connected")


def test_rules_status_when_the_rules_file_is_broken(monkeypatch):
    original_import = builtins.__import__
    def broken(name, *args, **kwargs):
        if name == "src.Cascades":
            raise SyntaxError("from __future__ imports must occur at the beginning of the file")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", broken)
    ready, message = rules_status()
    assert not ready and "could not be loaded" in message


def test_rules_status_without_nomenclator_file(monkeypatch, tmp_path):
    ready, message = rules_status(tmp_path / "prescriptions.json")
    assert not ready and "prescriptions.json" in message


def test_rules_status_ready(monkeypatch, tmp_path):
    path = tmp_path / "prescriptions.json"
    path.write_text("{}", encoding="utf-8")
    assert rules_status(path) == (True, "connected")

def test_duplicate_ingredient_does_not_trigger_self_interactions():
    patient = _patient([
        ("68339", "Doctor A"),
        ("85780", "Doctor B"),
    ])

    model = FakeModel({
        ("68339", "paracetamol"): "found",
        ("85780", "paracetamol"): "found",
    })

    result = review_patient(
        patient,
        CATALOGUE,
        use_model=True,
        review=model,
    )

    assert model.calls == []
    assert planned_questions(patient, None, CATALOGUE) == 0
    assert any(f.tipo == "duplicado" for f in result.findings)
    assert not any(f.tipo == "interaccion" for f in result.findings)



def test_repeated_search_reuses_result_within_review():
    patient = _patient([
        ("68339", "Doctor A"),
        ("85780", "Doctor B"),
    ])

    model = FakeModel()

    result = check_new_medicine(
        patient,
        _proposed("88372"),
        CATALOGUE,
        review=model,
    )

    assert model.calls.count(
        ("88372", "paracetamol")
    ) == 1

    assert len(model.calls) == 3
    assert len(result.log) == 4

    # A separate review must not reuse the previous cache.
    check_new_medicine(
        patient,
        _proposed("88372"),
        CATALOGUE,
        review=model,
    )

    assert len(model.calls) == 6
