"""Streamlit interface for the medication review (Person 5).

Run from the project root:

    streamlit run app.py

For now the screen uses SAMPLE findings (data/demo/hallazgos_ejemplo.json).
When the rules (Person 2) and the RAG pipeline (Person 3) are ready, connect
them in run_review() and set USE_SAMPLE_FINDINGS = False.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from src.config import PATIENTS_DIR, PROJECT_ROOT
from src.informe import generate_report, save_report
from src.modelos import Patient

DEMO_DIR = PROJECT_ROOT / "data" / "demo"
DEMO_PATIENTS_FILE = DEMO_DIR / "pacientes_demo.json"
SAMPLE_FINDINGS_FILE = DEMO_DIR / "hallazgos_ejemplo.json"

# Switch to False once run_review() calls the real rules and RAG pipeline.
USE_SAMPLE_FINDINGS = True


def load_patients() -> tuple[list[tuple[Patient, str]], list[str]]:
    """Read Lucía's patient files and the demo patients, validated with Patient."""
    files = [(path, "patients") for path in sorted(PATIENTS_DIR.glob("*.json"))]
    if DEMO_PATIENTS_FILE.exists():
        files.append((DEMO_PATIENTS_FILE, "demo"))

    patients, problems = [], []
    for path, source in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{path.name}: could not be read ({exc})")
            continue
        for item in data if isinstance(data, list) else [data]:
            try:
                patients.append((Patient.model_validate(item), source))
            except ValidationError as exc:
                problems.append(f"{path.name}: {exc.error_count()} validation error(s)")
    return patients, problems


def load_sample_findings() -> dict:
    return json.loads(SAMPLE_FINDINGS_FILE.read_text(encoding="utf-8"))


def run_review(patient: Patient) -> tuple[list[dict], list[str]]:
    """Return (findings, pairs checked without information) for one patient.

    TODO (integration), keeping this return shape:
      - Person 2: duplicates, ATC warnings and cascades from src/reglas.py,
        as Finding objects.
      - Person 3: src.rag.review_interaction() for each medicine pair.
        status "found"     -> Finding(tipo="interaccion", ...) with its citation
        status "not_found" -> a short text for the pair, e.g. "A + B"
        status "rejected"/"error" -> not shown; counted for the evaluation.
    """
    if USE_SAMPLE_FINDINGS:
        entry = load_sample_findings().get(patient.id, {})
        return entry.get("hallazgos", []), entry.get("pares_sin_informacion", [])
    raise NotImplementedError("Connect the rules and RAG pipeline here.")


st.set_page_config(page_title="Medication review", page_icon="💊", layout="wide")

st.title("Medication review")
st.caption(
    "One view of every medicine a polymedicated patient takes, with each finding traced "
    "to its official source. The system informs; the clinician decides."
)

patients, problems = load_patients()

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "1. Choose a patient.\n"
        "2. Click **Review**.\n"
        "3. Read the report, or download it as a PDF or a web page.\n\n"
        "Each report is also saved in the `output/` folder."
    )
    show_evaluation = st.checkbox(
        "Show evaluation patients",
        value=False,
        help="Evaluation patients are kept aside for the final evaluation. "
        "Build and test with development patients.",
    )
    if USE_SAMPLE_FINDINGS:
        st.warning("Using **sample findings**. They are placeholders, not clinical information.")
    st.caption("Synthetic patients only.")
    for problem in problems:
        st.error(problem)

visible = [(p, s) for p, s in patients if show_evaluation or p.grupo != "evaluacion"]
visible.sort(key=lambda ps: (ps[0].grupo == "evaluacion", ps[1] == "demo", ps[0].id))
if not visible:
    st.error("No patient files found in data/pacientes.")
    st.stop()

by_id = {p.id: (p, s) for p, s in visible}


def patient_label(pid: str) -> str:
    p, source = by_id[pid]
    tags = (["demo"] if source == "demo" else []) + (["evaluation"] if p.grupo == "evaluacion" else [])
    extra = f"  ({', '.join(tags)})" if tags else ""
    return f"Patient {p.id} · {p.edad} years · {len(p.medicamentos)} medicines{extra}"


patient_id = st.selectbox("Patient", list(by_id), format_func=patient_label)
patient, _ = by_id[patient_id]

if USE_SAMPLE_FINDINGS:
    note = load_sample_findings().get(patient.id, {}).get("nota")
    if note:
        st.caption(note)

prescribers = {m.prescriptor for m in patient.medicamentos}
st.write(f"**{len(patient.medicamentos)} medicines** from **{len(prescribers)} prescribers**")
with st.expander("Medicines on file"):
    st.dataframe(
        [
            {"Medicine": m.nombre, "Dose": m.dosis, "Prescriber": m.prescriptor, "Reg. no.": m.nregistro}
            for m in patient.medicamentos
        ],
        hide_index=True,
    )

if st.button("Review", type="primary"):
    try:
        with st.spinner("Reviewing medicines…"):
            findings, pairs = run_review(patient)
            report = generate_report(
                patient, findings, checked_without_evidence=pairs, sample=USE_SAMPLE_FINDINGS
            )
            paths = save_report(report)
        st.session_state["result"] = {
            "patient_id": patient.id,
            "report": report,
            "files": {kind: (path.name, path.read_bytes()) for kind, path in paths.items()},
        }
    except Exception as exc:  # show the problem on screen, not a stack trace
        st.session_state.pop("result", None)
        st.error(f"The review could not be completed: {exc}")

result = st.session_state.get("result")
if result and result["patient_id"] == patient.id:
    report = result["report"]
    c = report.counts

    cols = st.columns(4)
    cols[0].metric("Duplicates", c["duplicado"])
    cols[1].metric("Same group (ATC)", c["advertencia_atc"])
    cols[2].metric("Interactions", c["interaccion"])
    cols[3].metric("Possible cascades", c["cascada"])

    if report.safety_flags:
        st.warning(
            f"{len(report.safety_flags)} generated text(s) were withheld because they "
            "contained treatment-change wording."
        )

    pdf_name, pdf_bytes = result["files"]["pdf"]
    html_name, html_bytes = result["files"]["html"]
    st.markdown("**Download the report**")
    d1, d2, d3 = st.columns([1, 1, 2])
    d1.download_button("PDF", data=pdf_bytes, file_name=pdf_name, mime="application/pdf")
    d2.download_button("Web page", data=html_bytes, file_name=html_name, mime="text/html")
    d3.caption(
        "PDF opens in Preview or any PDF reader. The web page opens in your browser "
        "and can be printed. Both are also saved in `output/`."
    )

    st.divider()
    st.markdown(report.markdown)
else:
    st.info("Choose a patient and click **Review** to see the report.")
