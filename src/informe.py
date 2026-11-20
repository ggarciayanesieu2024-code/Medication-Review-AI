"""Medication review report (Person 5).

Builds the report for one patient from:

* the patient file (``Patient`` in ``src/modelos.py``, Lucía's format), and
* the findings produced by the rest of the system (``Finding``): duplicates and
  ATC warnings and cascades from the rules (Person 2), interactions from the RAG
  pipeline (Person 3), already checked by the shared citation verifier and
  forbidden-phrase list (Person 4).

The report
1. groups the patient's medicines by prescriber,
2. lists duplicates, ATC warnings, interactions with their CIMA citation, and
   possible prescribing cascades (as hypotheses to review),
3. lists the questions for the next appointment,
4. ALWAYS ends with the mandatory footer.

The report is built once as a list of blocks and then rendered to Markdown (for
the screen) or, in ``src/exportar.py``, to HTML and PDF. All formats come from
the same blocks, so they always say the same thing.

Defence in depth: generated text (finding descriptions and questions) is checked
again here against the shared ``FORBIDDEN_PHRASES``. Text that fails is withheld
and recorded in ``Report.safety_flags``. Literal CIMA quotations are shown as
they are: they are the official source text.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.config import FORBIDDEN_PHRASES, PROJECT_ROOT
from src.modelos import Finding, Medication, Patient

FOOTER = (
    "Automatically generated support information. "
    "Does not replace the judgement of a healthcare professional."
)

OUTPUT_DIR = PROJECT_ROOT / "output"

SECTION_TITLES = {
    "4.2": "Posology and method of administration",
    "4.3": "Contraindications",
    "4.4": "Special warnings and precautions for use",
    "4.5": "Interaction with other medicinal products",
    "4.8": "Undesirable effects",
}

# Report sections for each Finding.tipo, in report order.
FINDING_SECTIONS = [
    ("duplicado", "Duplicate active ingredients", None),
    ("advertencia_atc", "Same therapeutic group (ATC)", None),
    (
        "interaccion",
        "Possible interactions",
        "Information retrieved from the official technical sheets (CIMA). Every quotation "
        "below was checked against the section it cites.",
    ),
    (
        "cascada",
        "Possible prescribing cascades (hypotheses to review)",
        "Matched against the group's curated cascade table. These are hypotheses for the "
        "clinician to review, not diagnoses.",
    ),
]

SAMPLE_BANNER = (
    "The findings below are placeholders used to build the interface. "
    "They are not clinical information."
)
WITHHELD_TEXT = (
    "The generated text for this finding was withheld because it contained "
    "treatment-change wording."
)
GENERAL_QUESTION = (
    "Is the patient taking any other medicines, supplements or herbal products "
    "that are not on this list?"
)
GROUP_LABELS = {"desarrollo": "development patient", "evaluacion": "evaluation patient"}


# --------------------------------------------------------------------------
# Safety check (same normalisation as src/rag.py)
# --------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """Ignore case, accents and repeated whitespace."""
    text = unicodedata.normalize("NFKD", (text or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def find_forbidden_phrases(text: str) -> list[str]:
    """Return the shared FORBIDDEN_PHRASES found in ``text`` (empty if none)."""
    normalised = normalise_text(text)
    return [p for p in FORBIDDEN_PHRASES if normalise_text(p) in normalised]


# --------------------------------------------------------------------------
# Report structure
# --------------------------------------------------------------------------

@dataclass
class Block:
    """One piece of the report. ``kind`` decides how each renderer draws it.

    Kinds: title, meta, banner, h2, h3, para, note, table, bullets, numbered,
    source, quote, footer.
    """

    kind: str
    text: str = ""
    label: str = ""
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    items: list[Any] = field(default_factory=list)


@dataclass
class SafetyFlag:
    """Generated text withheld because it contained a forbidden phrase."""

    where: str
    text: str
    phrases: list[str]


@dataclass
class Report:
    patient_id: str
    generated_at: datetime
    blocks: list[Block]
    counts: dict[str, int] = field(default_factory=dict)
    safety_flags: list[SafetyFlag] = field(default_factory=list)
    sample: bool = False
    proposed_name: str | None = None
    # Finding types that were not looked for (their sections say "Not checked").
    unchecked: set[str] = field(default_factory=set)

    @property
    def title(self) -> str:
        if self.proposed_name:
            return f"Medicine check before prescribing - patient {self.patient_id}"
        return f"Medication review report - patient {self.patient_id}"

    @property
    def file_stem(self) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", self.patient_id)
        kind = "comprobacion" if self.proposed_name else "informe"
        return f"{kind}_{safe_id}_{self.generated_at:%Y%m%d_%H%M%S}"

    @property
    def markdown(self) -> str:
        return render_markdown(self.blocks)


# --------------------------------------------------------------------------
# Building the report
# --------------------------------------------------------------------------

def _section_label(section: str) -> str:
    """'4.5' or a subsection like '4.5.1' -> 'section 4.5.1 (Interaction ...)'."""
    main = ".".join(section.split(".")[:2])
    title = SECTION_TITLES.get(main)
    return f"section {section} ({title})" if title else f"section {section}"


def generate_report(
    patient: Patient | dict,
    findings: Iterable[Finding | dict] | None = None,
    *,
    checked_without_evidence: Iterable[str] | None = None,
    could_not_check: Iterable[str] | None = None,
    proposed: Medication | dict | None = None,
    unchecked_types: Iterable[str] = (),
    sample: bool = False,
    generated_at: datetime | None = None,
) -> Report:
    """Build the report for one patient. The footer is always the last block.

    ``checked_without_evidence`` lists the medicine pairs the interaction search
    looked at without finding information. They are shown separately, with a note
    that this does not mean the combination is safe.

    ``could_not_check`` lists searches that did not finish (blocked answer,
    unverifiable quotation, model unreachable). They are never shown as
    "nothing found".

    ``proposed`` turns the report into a check of a new medicine the prescriber
    is considering: it is listed apart from the current medicines.

    ``unchecked_types`` are Finding types that were not looked for at all; their
    sections say "Not checked" instead of "None found".
    """
    patient = patient if isinstance(patient, Patient) else Patient.model_validate(patient)
    findings = [f if isinstance(f, Finding) else Finding.model_validate(f) for f in (findings or [])]
    if proposed is not None and not isinstance(proposed, Medication):
        proposed = Medication.model_validate(proposed)
    not_found = list(checked_without_evidence or [])
    failed = list(could_not_check or [])
    unchecked = set(unchecked_types)
    generated_at = generated_at or datetime.now()
    flags: list[SafetyFlag] = []

    def screened(text: str | None, where: str) -> str | None:
        if not text:
            return None
        hits = find_forbidden_phrases(text)
        if hits:
            flags.append(SafetyFlag(where=where, text=text, phrases=hits))
            return None
        return text

    # Group medicines by prescriber, in order of first appearance.
    by_prescriber: dict[str, list] = {}
    for med in patient.medicamentos:
        by_prescriber.setdefault(med.prescriptor, []).append(med)
    names = {med.nregistro: med.nombre for med in patient.medicamentos}
    if proposed is not None:
        names[proposed.nregistro] = proposed.nombre

    by_type = {tipo: [f for f in findings if f.tipo == tipo] for tipo, _, _ in FINDING_SECTIONS}
    counts = {
        "medicines": len(patient.medicamentos),
        "prescribers": len(by_prescriber),
        **{tipo: len(items) for tipo, items in by_type.items()},
        "checked_without_evidence": len(not_found),
        "could_not_check": len(failed),
    }

    blocks: list[Block] = []
    add = blocks.append

    # Header
    meta = [
        ("Patient", f"{patient.id} · {patient.edad} years · "
                    f"{GROUP_LABELS.get(patient.grupo, patient.grupo)}"),
        ("Generated", f"{generated_at:%Y-%m-%d %H:%M}"),
    ]
    if proposed is not None:
        add(Block("title", "Medicine check before prescribing"))
        meta += [
            ("Proposed medicine", f"{proposed.nombre} · {proposed.dosis} · {proposed.prescriptor}"),
            ("Current medicines", f"{counts['medicines']} from {counts['prescribers']} prescriber(s)"),
        ]
        add(Block("meta", items=meta))
        add(Block("note", "This report checks a proposed medicine against the patient's current "
                          "medicines, using the official technical sheets. It informs the "
                          "prescriber; it is not a decision."))
    else:
        add(Block("title", "Medication review report"))
        meta.append(("Medicines reviewed",
                     f"{counts['medicines']} from {counts['prescribers']} prescriber(s)"))
        add(Block("meta", items=meta))
    if sample:
        add(Block("banner", SAMPLE_BANNER, label="Sample data."))

    # Summary
    add(Block("h2", "Summary"))
    summary_rows = [[title, "Not checked" if tipo in unchecked else str(counts[tipo])]
                    for tipo, title, _ in FINDING_SECTIONS]
    if not_found:
        summary_rows.append(["Pairs checked with no information found", str(len(not_found))])
    if failed:
        summary_rows.append(["Checks that could not be completed", str(len(failed))])
    add(Block("table", header=["Check", "Result"], rows=summary_rows))

    # 1. Medicines by prescriber
    add(Block("h2", "1. Current medicines by prescriber" if proposed else "1. Medicines by prescriber"))
    for prescriber, meds in by_prescriber.items():
        add(Block("h3", f"{prescriber} ({len(meds)})"))
        add(Block("table", header=["Medicine", "Dose", "Reg. no."],
                  rows=[[m.nombre, m.dosis, m.nregistro] for m in meds]))
    if proposed is not None:
        add(Block("h3", f"Proposed medicine, not yet prescribed: {proposed.prescriptor}"))
        add(Block("table", header=["Medicine", "Dose", "Reg. no."],
                  rows=[[proposed.nombre, proposed.dosis, proposed.nregistro]]))

    # 2-5. Findings
    for number, (tipo, title, intro) in enumerate(FINDING_SECTIONS, start=2):
        add(Block("h2", f"{number}. {title}"))
        if intro:
            add(Block("note", intro))
        items = by_type[tipo]
        if not items and tipo in unchecked:
            add(Block("note", "Not checked: this part of the system was not available for this report."))
        elif not items:
            add(Block("para", "None found."))
        for i, finding in enumerate(items, start=1):
            if len(items) > 1:
                add(Block("h3", f"Finding {number}.{i}"))
            text = screened(finding.descripcion, f"{tipo} description")
            add(Block("para", text) if text else Block("note", WITHHELD_TEXT))
            if finding.cita:
                c = finding.cita
                medicine = names.get(c.nregistro, "Medicine")
                add(Block("source", f"{medicine} (reg. {c.nregistro}), {_section_label(c.seccion)}",
                          label="Source"))
                add(Block("quote", c.literal_quote))
        if tipo == "interaccion" and not_found:
            add(Block("h3", "Pairs checked with no information found"))
            add(Block("bullets", items=not_found))
            add(Block("note", "No information in the available documents does not mean "
                              "the combination is safe."))

    # Checks that did not finish (never shown as "nothing found")
    if failed:
        add(Block("h2", "Checks that could not be completed"))
        add(Block("bullets", items=failed))
        add(Block("note", "These checks did not give a usable result, so these medicines "
                          "or combinations need to be reviewed by hand."))

    # 6. Questions
    questions: list[str] = []
    for finding in findings:
        q = screened(finding.pregunta_medico, f"{finding.tipo} question")
        if q and q not in questions:
            questions.append(q)
    questions.append(GENERAL_QUESTION)
    add(Block("h2", "6. Questions to consider before prescribing" if proposed
              else "6. Questions for the next appointment"))
    add(Block("numbered", items=questions))

    # Footer: always last
    add(Block("footer", FOOTER))

    return Report(
        patient_id=patient.id,
        generated_at=generated_at,
        blocks=blocks,
        counts=counts,
        safety_flags=flags,
        sample=sample,
        proposed_name=proposed.nombre if proposed is not None else None,
        unchecked=unchecked,
    )


# --------------------------------------------------------------------------
# Markdown (screen)
# --------------------------------------------------------------------------

_MD_SPECIAL = re.compile(r"([\\`*_\[\]#|$<>])")


def _md(text: Any) -> str:
    """Escape Markdown (and Streamlit's $ maths) in data text; keep it on one line."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(text).split()))


def render_markdown(blocks: list[Block]) -> str:
    out: list[str] = []
    for b in blocks:
        if b.kind == "title":
            out.append(f"# {_md(b.text)}")
        elif b.kind == "meta":
            out.append("  \n".join(f"**{_md(k)}:** {_md(v)}" for k, v in b.items))
        elif b.kind == "banner":
            out.append(f"> **{_md(b.label)}** {_md(b.text)}")
        elif b.kind == "h2":
            out.append(f"## {_md(b.text)}")
        elif b.kind == "h3":
            out.append(f"### {_md(b.text)}")
        elif b.kind == "para":
            out.append(_md(b.text))
        elif b.kind == "note":
            out.append(f"_{_md(b.text)}_")
        elif b.kind == "table":
            lines = ["| " + " | ".join(_md(h) for h in b.header) + " |",
                     "|" + "---|" * len(b.header)]
            lines += ["| " + " | ".join(_md(c) for c in row) + " |" for row in b.rows]
            out.append("\n".join(lines))
        elif b.kind == "bullets":
            out.append("\n".join(f"- {_md(i)}" for i in b.items))
        elif b.kind == "numbered":
            out.append("\n".join(f"{n}. {_md(i)}" for n, i in enumerate(b.items, start=1)))
        elif b.kind == "source":
            out.append(f"**{_md(b.label)}:** {_md(b.text)}")
        elif b.kind == "quote":
            out.append(f"> “{_md(b.text)}”")
        elif b.kind == "footer":
            out.append(f"---\n\n*{b.text}*")
    return "\n\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------

def save_report(report: Report, output_dir: str | Path = OUTPUT_DIR) -> dict[str, Path]:
    """Save the report to output/ as PDF and as a web page. Returns the paths."""
    from src.exportar import render_html, render_pdf  # local import: optional PDF library

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": output_dir / f"{report.file_stem}.pdf",
        "html": output_dir / f"{report.file_stem}.html",
    }
    paths["pdf"].write_bytes(render_pdf(report))
    paths["html"].write_text(render_html(report), encoding="utf-8")
    return paths
