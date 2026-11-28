"""Tests for the report (src/Reports/informe.py) and its download formats (src/Reports/exportar.py).

The full set of unsafe and safe example sentences (tests/fixtures/safety_cases.json)
is Person 5's next deliverable; these tests cover the footer, the report structure,
the shared forbidden-phrase check, and the PDF and HTML files.
"""
import json
from pathlib import Path

import pytest

from src.config import PATIENTS_DIR, PROJECT_ROOT
from src.Reports.exportar import render_html, render_pdf
from src.Reports.informe import FOOTER, find_forbidden_phrases, generate_report, save_report
from src.Reports.informe import Finding, Patient

DEMO_DIR = PROJECT_ROOT / "data" / "demo"
SAMPLE = json.loads((DEMO_DIR / "hallazgos_ejemplo.json").read_text(encoding="utf-8"))


def _load(path: Path) -> list[Patient]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Patient.model_validate(p) for p in (data if isinstance(data, list) else [data])]


DEV_PATIENTS = [
    p
    for path in sorted(path for path in PATIENTS_DIR.glob("*.json") if path.name != "patient_scenarios.json") + [DEMO_DIR / "pacientes_demo.json"]
    for p in _load(path)
    if p.grupo == "desarrollo"
]
BY_ID = {p.id: p for p in DEV_PATIENTS}


def _report(patient, **kwargs):
    entry = SAMPLE.get(patient.id, {})
    return generate_report(
        patient,
        entry.get("hallazgos", []),
        checked_without_evidence=entry.get("pares_sin_informacion", []),
        sample=True,
        **kwargs,
    )


# ---------------------------------------------------------------- sample data

def test_sample_findings_use_the_shared_finding_model():
    for key, entry in SAMPLE.items():
        if key.startswith("_"):
            continue
        assert key in BY_ID, f"sample findings for unknown development patient {key}"
        for finding in entry["hallazgos"]:
            Finding.model_validate(finding)


def test_sample_citations_point_to_a_medicine_of_that_patient():
    for key, entry in SAMPLE.items():
        if key.startswith("_"):
            continue
        regs = {m.nregistro for m in BY_ID[key].medicamentos}
        for finding in entry["hallazgos"]:
            if finding["cita"]:
                assert finding["cita"]["nregistro"] in regs


# ---------------------------------------------------------------- dev vs evaluation

EVAL_RAW = [
    item
    for path in sorted(path for path in PATIENTS_DIR.glob("*.json") if path.name != "patient_scenarios.json")
    for item in (lambda d: d if isinstance(d, list) else [d])(json.loads(path.read_text(encoding="utf-8")))
    if item.get("grupo") == "evaluacion"
]


def _checked_pairs(raw: dict) -> list[frozenset]:
    """Medicine pairs whose result an evaluation patient checks (duplicates, interactions)."""
    expected = raw.get("expected_result") or raw.get("resultado_esperado") or {}
    pairs = []
    for dup in expected.get("duplicidades", []):
        regs = dup.get("nregistros", [])
        pairs += [frozenset((a, b)) for i, a in enumerate(regs) for b in regs[i + 1:]]
    for inter in expected.get("interacciones", []):
        if len(inter.get("par", [])) == 2:
            pairs.append(frozenset(inter["par"]))
    return pairs


def test_development_patients_do_not_copy_evaluation_cases():
    """Keep the evaluation honest: no development patient contains an evaluation
    patient's whole medicine list, or a pair whose result the evaluation checks."""
    assert EVAL_RAW, "no evaluation patients found"
    for raw in EVAL_RAW:
        eval_regs = {m["nregistro"] for m in raw["medicamentos"]}
        for patient in DEV_PATIENTS:
            dev_regs = {m.nregistro for m in patient.medicamentos}
            assert not eval_regs <= dev_regs, f"{patient.id} contains every medicine of {raw['id']}"
            for pair in _checked_pairs(raw):
                assert not pair <= dev_regs, f"{patient.id} repeats pair {sorted(pair)} from {raw['id']}"


# ---------------------------------------------------------------- footer

@pytest.mark.parametrize("patient", DEV_PATIENTS, ids=lambda p: p.id)
def test_footer_is_always_last(patient):
    report = _report(patient)
    assert report.blocks[-1].kind == "footer"
    assert report.markdown.rstrip().endswith(f"*{FOOTER}*")


def test_footer_appears_with_no_findings():
    report = generate_report(BY_ID["C"], [])
    assert report.markdown.rstrip().endswith(f"*{FOOTER}*")


def test_footer_appears_when_all_generated_text_is_withheld():
    finding = {"tipo": "interaccion", "descripcion": "Stop taking warfarin.",
               "cita": None, "pregunta_medico": "Reduce the dose?"}
    report = generate_report(BY_ID["B"], [finding])
    assert report.markdown.rstrip().endswith(f"*{FOOTER}*")


# ---------------------------------------------------------------- structure

def test_medicines_grouped_by_prescriber_in_order_of_appearance():
    patient = BY_ID["DEMO_G"]
    headings = [b.text for b in _report(patient).blocks if b.kind == "h3"]
    expected = list(dict.fromkeys(m.prescriptor for m in patient.medicamentos))
    found = [h.rsplit(" (", 1)[0] for h in headings if h.rsplit(" (", 1)[0] in expected]
    assert found == expected


def test_interaction_shows_medicine_name_and_section():
    md = _report(BY_ID["B"]).markdown
    assert "Amoxicilina/Ácido clavulánico Tevagen" in md
    assert "section 4.5 (Interaction with other medicinal products)" in md


def test_pairs_without_information_are_not_called_safe():
    md = _report(BY_ID["C"]).markdown
    assert "Directional evidence checks with no information found" in md
    assert "does not mean the combination is safe" in md


def test_sample_banner_only_when_sample():
    assert "Sample data" in _report(BY_ID["A"]).markdown
    assert "Sample data" not in generate_report(BY_ID["A"], []).markdown


# ---------------------------------------------------------------- safety

def test_forbidden_description_is_withheld_and_flagged():
    finding = {"tipo": "interaccion", "descripcion": "The patient should stop taking Aldocumar.",
               "cita": None, "pregunta_medico": None}
    report = generate_report(BY_ID["B"], [finding])
    assert "stop taking" not in report.markdown
    assert len(report.safety_flags) == 1


def test_forbidden_question_is_dropped_and_flagged():
    finding = {"tipo": "duplicado", "descripcion": "Same ingredient twice.",
               "cita": None, "pregunta_medico": "¿Reduzca la dosis de paracetamol?"}
    report = generate_report(BY_ID["A"], [finding])
    assert "Reduzca" not in report.markdown
    assert [f.where for f in report.safety_flags] == ["duplicado question"]


def test_phrase_check_ignores_case_and_accents():
    assert find_forbidden_phrases("DÉJE DE TOMAR el medicamento")


def test_sample_data_passes_the_safety_check():
    for patient in DEV_PATIENTS:
        assert _report(patient).safety_flags == []


# ---------------------------------------------------------------- downloads

def test_html_is_complete_and_escaped():
    patient = BY_ID["A"].model_copy(update={"id": "<script>x</script>"})
    page = render_html(generate_report(patient, []))
    assert page.startswith("<!DOCTYPE html>")
    assert FOOTER in page
    assert "<script>x" not in page


def test_pdf_is_valid_and_has_the_footer():
    pdf = render_pdf(_report(BY_ID["DEMO_G"]))
    assert pdf.startswith(b"%PDF")
    pypdf = pytest.importorskip("pypdf")
    import io

    text = " ".join(
        " ".join(page.extract_text().split()) for page in pypdf.PdfReader(io.BytesIO(pdf)).pages
    )
    assert FOOTER in text


def test_pdf_handles_symbols_outside_the_basic_font():
    finding = {"tipo": "cascada", "descripcion": "Amlodipino → edema → furosemida (dosis ≥ 25 mg) ⚠",
               "cita": None, "pregunta_medico": None}
    assert render_pdf(generate_report(BY_ID["C"], [finding])).startswith(b"%PDF")


def test_save_report_writes_pdf_and_html(tmp_path):
    paths = save_report(_report(BY_ID["A"]), tmp_path)
    assert set(paths) == {"pdf", "html"}
    for path in paths.values():
        assert path.parent == tmp_path and path.name.startswith("informe_A_")
        assert path.stat().st_size > 0


# ---------------------------------------------------------------- safety cases (deliverable 3)
#
# tests/fixtures/safety_cases.json holds unsafe sentences that must be blocked and
# safe ones that must pass. Unsafe sentences marked "known_gap" are not caught by
# the shared FORBIDDEN_PHRASES yet: they run as strict expected failures (XFAIL),
# so a fix shows up as XPASS and the fixture must then be updated.

from src.Reports.informe import _md  # noqa: E402  (Markdown escaping used by the report)

SAFETY = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "safety_cases.json").read_text(encoding="utf-8"))


def _unsafe_params():
    for case in SAFETY["unsafe"]:
        marks = [pytest.mark.xfail(strict=True, reason="known gap: not in FORBIDDEN_PHRASES yet")] \
            if case.get("known_gap") else []
        yield pytest.param(case["text"], marks=marks, id=case["text"][:40])


CAUGHT = [c["text"] for c in SAFETY["unsafe"] if not c.get("known_gap")]
SAFE = [c["text"] for c in SAFETY["safe"]]


def test_safety_cases_cover_both_languages():
    for group in ("unsafe", "safe"):
        assert {c["lang"] for c in SAFETY[group]} == {"en", "es"}
        assert all(c["why"].strip() for c in SAFETY[group])
    assert len(SAFETY["unsafe"]) >= 10 and len(SAFETY["safe"]) >= 10


@pytest.mark.parametrize("text", list(_unsafe_params()))
def test_unsafe_wording_is_caught(text):
    assert find_forbidden_phrases(text), f"not blocked: {text!r}"


@pytest.mark.parametrize("text", SAFE, ids=[t[:40] for t in SAFE])
def test_safe_wording_passes(text):
    assert find_forbidden_phrases(text) == []


@pytest.mark.parametrize("text", CAUGHT, ids=[t[:40] for t in CAUGHT])
def test_unsafe_generated_text_is_withheld_and_the_disclaimer_stays(text):
    finding = {"tipo": "interaccion", "descripcion": text, "cita": None, "pregunta_medico": text}
    report = generate_report(BY_ID["B"], [finding])
    assert _md(text) not in report.markdown
    assert len(report.safety_flags) == 2  # the description and the question
    assert report.markdown.rstrip().endswith(f"*{FOOTER}*")
    assert FOOTER in render_html(report)


@pytest.mark.parametrize("text", SAFE, ids=[t[:40] for t in SAFE])
def test_safe_generated_text_is_shown(text):
    finding = {"tipo": "interaccion", "descripcion": text, "cita": None, "pregunta_medico": None}
    report = generate_report(BY_ID["B"], [finding])
    assert report.safety_flags == []
    assert _md(text) in report.markdown


def test_report_check_matches_the_rag_pipeline():
    """The report and src/rag.py must normalise text the same way."""
    rag = pytest.importorskip("src.Rag.rag", reason="model libraries not installed")
    from src.Reports.informe import normalise_text

    for case in SAFETY["unsafe"] + SAFETY["safe"]:
        assert normalise_text(case["text"]) == rag.normalise_text(case["text"])
