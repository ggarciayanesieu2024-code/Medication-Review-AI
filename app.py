"""Streamlit interface for the medication review (Person 5).

Run from the project root:

    streamlit run app.py

Two tabs:

* Review current medicines - checks every medicine the patient already takes.
* Check a new medicine - checks a medicine the prescriber is considering,
  against the patient's current medicines, before it is prescribed.

Both use the real system when it is available (see src/revision.py):

* Rules (Person 2): nomenclator codes from data/processed/prescriptions.json,
  or the ATC codes in data/catalogo_medicamentos.json when that file is missing.
* Interactions (Person 3's model): asked live when the model is set up (full
  requirements, search index, GROQ_API_KEY); otherwise read from
  data/interacciones_guardadas.json, made once by `python -m src.guardar_interacciones`.

The report always says which checks did not run. When neither part is
available, the review tab shows SAMPLE findings under a "Sample data" banner.
"""
from __future__ import annotations

import json

import streamlit as st
from pydantic import ValidationError

from src.config import PATIENTS_DIR, PROJECT_ROOT
from src.informe import generate_report, save_report
from src.modelos import Medication, Patient
from src.revision import (
    check_new_medicine,
    load_catalogue,
    load_rule_data,
    load_stored_answers,
    model_status,
    planned_questions,
    review_patient,
    rules_status,
    stored_review,
)

DEMO_DIR = PROJECT_ROOT / "data" / "demo"
DEMO_PATIENTS_FILE = DEMO_DIR / "pacientes_demo.json"
SAMPLE_FINDINGS_FILE = DEMO_DIR / "hallazgos_ejemplo.json"

TYPE_LABELS = {
    "duplicado": "Duplicates",
    "advertencia_atc": "Same drug family",
    "interaccion": "Interactions",
    "cascada": "Possible cascades",
}


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


@st.cache_resource(show_spinner="Loading the medicine codes…")
def cached_rule_data():
    return load_rule_data(catalogue=load_catalogue())


def saved_files(report) -> dict:
    paths = save_report(report)
    return {kind: (path.name, path.read_bytes()) for kind, path in paths.items()}


def show_counts(report, labels: list[tuple[str, str]]) -> None:
    cols = st.columns(len(labels))
    for col, (key, label) in zip(cols, labels):
        col.metric(label, "Not checked" if key in report.unchecked else report.counts[key])


def show_report(report, files: dict, key: str) -> None:
    """Downloads plus the report itself, for either tab."""
    if report.safety_flags:
        st.warning(
            f"{len(report.safety_flags)} generated text(s) were withheld because they "
            "contained treatment-change wording."
        )
    pdf_name, pdf_bytes = files["pdf"]
    html_name, html_bytes = files["html"]
    st.markdown("**Download the report**")
    d1, d2, d3 = st.columns([1, 1, 2])
    d1.download_button("PDF", data=pdf_bytes, file_name=pdf_name,
                       mime="application/pdf", key=f"{key}_pdf")
    d2.download_button("Web page", data=html_bytes, file_name=html_name,
                       mime="text/html", key=f"{key}_html")
    d3.caption(
        "PDF opens in Preview or any PDF reader. The web page opens in your browser "
        "and can be printed. Both are also saved in `output/`."
    )
    st.divider()
    st.markdown(report.markdown)


def what_runs(rules_ready: bool, interactions: str) -> str:
    parts = ["duplicates, drug families and cascades" if rules_ready
             else "duplicates by ingredient name only"]
    parts.append({
        "live": "interactions, asked to the model now",
        "stored": "interactions, from the model's stored answers",
        "off": "no interactions",
    }[interactions])
    return "This check covers " + "; ".join(parts) + "."


def progress_bar():
    bar = st.progress(0.0, text="Starting…")

    def update(done: int, total: int, label: str) -> None:
        bar.progress(done / total if total else 1.0, text=label)

    return bar, update


st.set_page_config(page_title="Medication review", page_icon="💊", layout="wide")

st.title("Medication review")
st.caption(
    "One view of every medicine a polymedicated patient takes, with each finding traced "
    "to its official source. The system informs; the clinician decides."
)

patients, problems = load_patients()
catalogue = load_catalogue()
model_ready, model_message = model_status()
rules_ready, rules_message = rules_status()
rule_data = None
if rules_ready:
    try:
        rule_data = cached_rule_data()
    except Exception as exc:  # unreadable codes file
        rules_ready, rules_message = False, f"the medicine codes could not be read ({exc})."

# Interactions: the live model, else the stored answers, else not checked.
stored = None if model_ready else load_stored_answers()
if model_ready:
    interactions, review_fn, interaction_note = "live", None, None
elif stored is not None:
    interactions, review_fn = "stored", stored_review(stored)
    interaction_note = (f"These answers were produced by the model on {stored.generated_at} and stored; "
                        "each quotation was verified against its section when it was produced.")
else:
    interactions, review_fn, interaction_note = "off", None, None
use_model = interactions != "off"
live = rules_ready or use_model

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "1. Choose a patient.\n"
        "2. **Review current medicines**, or **check a new medicine** before prescribing it.\n"
        "3. Read the report, or download it as a PDF or a web page.\n\n"
        "Each report is also saved in the `output/` folder."
    )
    show_evaluation = st.checkbox(
        "Show evaluation patients",
        value=False,
        help="Evaluation patients are kept aside for the final evaluation. "
        "Build and test with development patients.",
    )
    st.markdown("**What this version checks**")
    st.markdown(
        ("✅" if rules_ready else "⚪") + " Duplicates, drug families and cascades  \n"
        + {"live": "✅ Interactions (live model)",
           "stored": f"✅ Interactions (stored answers, {stored.generated_at if stored else ''})",
           "off": "⚪ Interactions"}[interactions]
    )
    if not live:
        st.caption("**Review current medicines** shows example results for now.")
    with st.expander("Technical details"):
        st.caption(f"Rules: {rules_message}." if rules_ready else f"Rules: not available. Reason: {rules_message}")
        st.caption("Model: connected." if model_ready else f"Model: not available here. Reason: {model_message}")
        if not model_ready:
            st.caption("Stored answers: " + (f"from {stored.generated_at}, {len(stored.answers)} searches."
                                             if stored else "none yet (data/interacciones_guardadas.json)."))
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

tab_review, tab_check = st.tabs(["Review current medicines", "Check a new medicine"])

# --------------------------------------------------------------------------
# Tab 1: review the current medicines
# --------------------------------------------------------------------------
with tab_review:
    if live:
        st.caption(what_runs(rules_ready, interactions))
        if interactions == "live":
            n = planned_questions(patient, None, catalogue)
            st.caption(f"The model will be asked {n} question(s) for this patient, "
                       "so it can take a minute.")
    else:
        note = load_sample_findings().get(patient.id, {}).get("nota")
        if note:
            st.caption(f"Sample: {note}")

    if st.button("Review", type="primary", key="review"):
        bar = None
        try:
            if live:
                bar, update = progress_bar()
                checked = review_patient(
                    patient, catalogue, rule_data=rule_data, use_model=use_model,
                    review=review_fn, progress=update,
                )
                report = generate_report(
                    patient,
                    checked.findings,
                    checked_without_evidence=checked.checked_without_evidence,
                    could_not_check=checked.could_not_check,
                    unchecked_types=checked.unchecked,
                    interaction_note=interaction_note,
                )
            else:
                entry = load_sample_findings().get(patient.id, {})
                report = generate_report(
                    patient, entry.get("hallazgos", []),
                    checked_without_evidence=entry.get("pares_sin_informacion", []), sample=True,
                )
            st.session_state["result"] = {
                "patient_id": patient.id, "report": report, "files": saved_files(report),
            }
        except Exception as exc:  # show the problem on screen, not a stack trace
            st.session_state.pop("result", None)
            st.error(f"The review could not be completed: {exc}")
        finally:
            if bar is not None:
                bar.empty()

    result = st.session_state.get("result")
    if result and result["patient_id"] == patient.id:
        show_counts(result["report"], list(TYPE_LABELS.items()))
        show_report(result["report"], result["files"], key="review")
    else:
        st.info("Click **Review** to see the report for this patient.")

# --------------------------------------------------------------------------
# Tab 2: check a new medicine before prescribing it
# --------------------------------------------------------------------------
with tab_check:
    st.write(
        "Choose the medicine you're thinking of prescribing. The system checks it against this "
        "patient's current medicines and shows what the official technical sheets say, with "
        "the quoted text. It does not decide whether to prescribe."
    )
    choices = sorted(catalogue.values(), key=lambda e: e.nombre.casefold())
    nregistro = st.selectbox(
        "Medicine to prescribe",
        options=[e.nregistro for e in choices],
        format_func=lambda n: catalogue[n].nombre + ("" if catalogue[n].ficha
                                                     else "  (technical sheet not in the system)"),
        index=None,
        placeholder="Choose a medicine",
        help="Medicines without a technical sheet can still be checked, but only the other "
        "medicines' sheets are searched for them.",
    )
    c1, c2 = st.columns(2)
    dose = c1.text_input("Dose (optional)", placeholder="1 g cada 8 horas")
    prescriber = c2.text_input("Prescriber (optional)", placeholder="Dra. Pérez (Atención Primaria)")
    st.caption(what_runs(rules_ready, interactions))

    if st.button("Check", type="primary", key="check"):
        if nregistro is None:
            st.error("Choose a medicine first.")
        else:
            proposed = Medication(
                nregistro=nregistro,
                nombre=catalogue[nregistro].nombre,
                dosis=dose.strip() or "Not specified",
                prescriptor=prescriber.strip() or "Not specified",
            )
            bar, update = progress_bar()
            try:
                checked = check_new_medicine(
                    patient, proposed, catalogue,
                    rule_data=rule_data, use_model=use_model, review=review_fn, progress=update,
                )
                report = generate_report(
                    patient,
                    checked.findings,
                    checked_without_evidence=checked.checked_without_evidence,
                    could_not_check=checked.could_not_check,
                    proposed=proposed,
                    unchecked_types=checked.unchecked,
                    interaction_note=interaction_note,
                )
                st.session_state["check_result"] = {
                    "patient_id": patient.id, "report": report,
                    "files": saved_files(report), "log": checked.log,
                }
            except Exception as exc:  # show the problem on screen, not a stack trace
                st.session_state.pop("check_result", None)
                st.error(f"The check could not be completed: {exc}")
            finally:
                bar.empty()

    checked_result = st.session_state.get("check_result")
    if checked_result and checked_result["patient_id"] == patient.id:
        show_counts(checked_result["report"], [
            ("duplicado", "Same ingredient"),
            ("interaccion", "Interactions found"),
            ("checked_without_evidence", "Nothing found"),
            ("could_not_check", "Could not check"),
        ])
        show_report(checked_result["report"], checked_result["files"], key="check")
