"""Run the real checks for the screen (Person 5).

Two entry points, one per tab:

* ``review_patient``: checks every medicine a patient already takes.
* ``check_new_medicine``: checks a proposed medicine against the patient's
  current medicines, before it is prescribed.

Both use whatever parts of the system are available, and say which parts were
not checked, so the report never shows "None found" for something nobody looked
at:

1. **Rules (Person 2, ``src/reglas.py``):** duplicates (shared active-ingredient
   code), same drug family (ATC) and prescribing cascades. They need
   ``data/processed/prescriptions.json``. Without it, duplicates fall back to a
   simple name comparison from ``data/catalogo_medicamentos.json``, and drug
   families and cascades are marked "not checked".
2. **Model (Person 3, ``src.rag.review_interaction``):** for each pair of
   medicines, each technical sheet is searched for the other medicine's active
   ingredient (both directions). It needs the search index and GROQ_API_KEY.
   Without them, interactions are marked "not checked".

Each model answer becomes one of three things, kept apart on purpose:

* ``found``: a Finding with the verified quotation from the technical sheet.
* ``not_found``: "searched, nothing found", listed separately. This does not
  mean the combination is safe.
* ``rejected`` / ``error`` / an exception: "could not be checked". Never shown
  as "nothing found"; the pair needs manual review.

The system informs; the prescriber decides. Nothing here says whether a
medicine can or cannot be prescribed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable

from src.config import DATA_DIR, DATABASE_DIR, PRESCRIPTIONS_FILE
from src.informe import normalise_text
from src.modelos import Citation, Finding, Medication, Patient

CATALOGUE_FILE = DATA_DIR / "catalogo_medicamentos.json"

REASON_TEXT = {
    "forbidden_phrase": "the model's answer was blocked by the safety check",
    "invalid_citation": "the model's answer quoted text that could not be verified",
    "invalid_response": "the model's answer was not in the expected format",
    "generation_failed": "the model could not be reached",
}

Progress = Callable[[int, int, str], None]


@dataclass
class CatalogueEntry:
    nregistro: str
    nombre: str
    principios_activos: list[str]
    ficha: bool


@dataclass
class CheckResult:
    findings: list[Finding] = field(default_factory=list)
    checked_without_evidence: list[str] = field(default_factory=list)
    could_not_check: list[str] = field(default_factory=list)
    # Finding types that were not looked for at all (shown as "Not checked").
    unchecked: set[str] = field(default_factory=set)
    # One entry per model question, for the evaluation (deliverable 5).
    log: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# Data and availability
# --------------------------------------------------------------------------

def load_catalogue(path: str | Path = CATALOGUE_FILE) -> dict[str, CatalogueEntry]:
    """Read the medicine list, keyed by registration number."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        item["nregistro"]: CatalogueEntry(
            nregistro=item["nregistro"],
            nombre=item["nombre"],
            principios_activos=list(item["principios_activos"]),
            ficha=bool(item["ficha"]),
        )
        for item in data["medicamentos"]
    }


def _import_rag():
    from src import rag  # needs groq, chromadb and sentence-transformers

    return rag


def _import_rules():
    from src import reglas

    return reglas


def model_status(database_dir: Path = DATABASE_DIR) -> tuple[bool, str]:
    """Return (ready, explanation) for the model, without calling it."""
    try:
        _import_rag()
    except ImportError as exc:
        return False, (
            f"the model's libraries are not installed here ({exc.name}). "
            "Install the full requirements.txt."
        )
    if not Path(database_dir).exists():
        return False, (
            "the search index is missing. Put the shared CIMA file in data/processed/ "
            "and run src/indice.py."
        )
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "your_groq_api_key_here":
        return False, "GROQ_API_KEY is missing from the .env file."
    return True, "connected"


def rules_status(prescriptions_file: Path = PRESCRIPTIONS_FILE) -> tuple[bool, str]:
    """Return (ready, explanation) for Person 2's rules."""
    try:
        _import_rules()
    except Exception as exc:  # a SyntaxError or ImportError inside src/reglas.py
        return False, f"src/reglas.py could not be loaded ({type(exc).__name__}: {exc})."
    if not Path(prescriptions_file).exists():
        return False, (
            "data/processed/prescriptions.json is missing. It is made by "
            "src/download_nomenclator.py and src/parse_nomenclator.py."
        )
    return True, "connected"


def load_rule_data(prescriptions_file: Path = PRESCRIPTIONS_FILE) -> tuple[dict, dict]:
    """(drug codes per registration number, cascade table) for the rules."""
    reglas = _import_rules()
    return reglas.load_drug_info(Path(prescriptions_file)), reglas.load_cascades()


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

def _join(items: list[str]) -> str:
    items = list(dict.fromkeys(i for i in items if i))
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _rule_findings(
    medications: list[Medication], rule_data: tuple[dict, dict], proposed: int | None, result: CheckResult
) -> None:
    """Run Person 2's rules. With ``proposed`` (an index into ``medications``),
    keep only the findings that involve that medicine."""
    reglas = _import_rules()
    drug_info, cascade_table = rule_data
    rule_meds, keep = [], None
    for i, med in enumerate(medications):
        info = drug_info.get(med.nregistro)
        if info is None:
            result.could_not_check.append(
                f"{med.nombre}: not found in the nomenclator data, so it was left out of the "
                "duplicate, drug-family and cascade checks."
            )
            continue
        rule_med = reglas.Medication(
            nregistro=med.nregistro, name=med.nombre, prescriber=med.prescriptor,
            atc=list(info["atc"]), active_ingredients=list(info["active_ingredients"]),
        )
        rule_meds.append(rule_med)
        if i == proposed:
            keep = rule_med
    if proposed is not None and keep is None:
        return  # the proposed medicine has no codes; already noted above

    def involved(*meds) -> bool:
        return keep is None or any(m is keep for m in meds)

    def other(a, b):
        return b if a is keep else a

    for d in reglas.find_duplicates(rule_meds):
        a, b = d.medication_a, d.medication_b
        if not involved(a, b):
            continue
        question = (
            f"The patient already takes {other(a, b).name}. What total daily amount of the "
            "shared active ingredient would the patient take with both?"
            if keep else
            f"Is the patient taking both {a.name} and {b.name}, and what total daily amount "
            "of the shared active ingredient does that add up to?"
        )
        result.findings.append(Finding(tipo="duplicado", descripcion=d.message, pregunta_medico=question))

    for w in reglas.find_family_warnings(rule_meds):
        a, b = w.medication_a, w.medication_b
        if not involved(a, b):
            continue
        result.findings.append(Finding(
            tipo="advertencia_atc",
            descripcion=w.message,
            pregunta_medico=(
                f"{a.name} and {b.name} belong to the same drug family (ATC {w.atc_family}). "
                "Is taking both intended, and is each prescriber aware of the other?"
            ),
        ))

    for c in reglas.find_cascades(rule_meds, cascade_table):
        if not involved(c.index_medication, c.cascade_medication):
            continue
        effect = c.adverse_effect[:1].lower() + c.adverse_effect[1:]
        if keep is c.cascade_medication:  # the proposed medicine would complete the cascade
            text = (f"{c.cascade_id}: '{c.index_medication.name}', which the patient already takes, can "
                    f"cause {effect}. The proposed medicine, '{keep.name}', is the kind of medicine "
                    "often added to treat it. Possible prescribing cascade to review.")
        elif keep is c.index_medication:
            text = (f"{c.cascade_id}: the proposed medicine, '{keep.name}', can cause {effect}, and the "
                    f"patient already takes '{c.cascade_medication.name}', which is often used to treat "
                    "it. Possible prescribing cascade to review.")
        else:
            text = f"{c.cascade_id}: {c.message}"
        if c.note:
            text += f" {c.note}"
        if c.sources:
            text += f" Sources: {'; '.join(c.sources)}"
        result.findings.append(Finding(tipo="cascada", descripcion=text, pregunta_medico=c.question))


def _name_duplicates(
    pairs: list[tuple[Medication, Medication]], catalogue: dict[str, CatalogueEntry],
    proposed: Medication | None, result: CheckResult,
) -> None:
    """Fallback when the rules are not available: compare ingredient names."""
    for x, y in pairs:
        ex, ey = catalogue.get(x.nregistro), catalogue.get(y.nregistro)
        if ex is None or ey is None:
            continue
        names_y = {normalise_text(i) for i in ey.principios_activos}
        shared = [i for i in ex.principios_activos if normalise_text(i) in names_y]
        if not shared:
            continue
        ingredient = _join(shared)
        if proposed is not None:
            result.findings.append(Finding(
                tipo="duplicado",
                descripcion=(
                    f"The proposed medicine ({x.nombre}) and {y.nombre}, prescribed by "
                    f"{y.prescriptor}, contain the same active ingredient: {ingredient}."
                ),
                pregunta_medico=(
                    f"The patient already takes {y.nombre}. What total daily amount of "
                    f"{ingredient} would the patient take with both?"
                ),
            ))
        else:
            result.findings.append(Finding(
                tipo="duplicado",
                descripcion=(
                    f"{x.nombre} ({x.prescriptor}) and {y.nombre} ({y.prescriptor}) contain "
                    f"the same active ingredient: {ingredient}."
                ),
                pregunta_medico=(
                    f"Is the patient taking both {x.nombre} and {y.nombre}, and what total "
                    f"daily amount of {ingredient} does that add up to?"
                ),
            ))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def _interaction_plan(
    pairs: list[tuple[Medication, Medication]], catalogue: dict[str, CatalogueEntry], result: CheckResult,
) -> list[tuple[CatalogueEntry, str, Medication, Medication]]:
    """The model questions to ask: (sheet, ingredient to look for, medicine x, medicine y)."""
    plan = []
    for x, y in pairs:
        ex, ey = catalogue.get(x.nregistro), catalogue.get(y.nregistro)
        missing = [m.nombre for m, e in ((x, ex), (y, ey)) if e is None]
        if missing:
            for name in missing:
                note = f"{name}: not in the medicine list, so its active ingredient is unknown."
                if note not in result.could_not_check:
                    result.could_not_check.append(note)
            continue
        if {normalise_text(i) for i in ex.principios_activos} & {normalise_text(i) for i in ey.principios_activos}:
            continue  # same ingredient: a duplicate, not an interaction question
        if ex.ficha:
            plan += [(ex, ing, x, y) for ing in ey.principios_activos]
        if ey.ficha:
            plan += [(ey, ing, x, y) for ing in ex.principios_activos]
        if ex.ficha != ey.ficha:
            searched, absent = (ex, ey) if ex.ficha else (ey, ex)
            result.could_not_check.append(
                f"{absent.nombre} sheet: not in the system, so only the {searched.nombre} sheet was searched."
            )
        elif not ex.ficha:
            result.could_not_check.append(
                f"{x.nombre} + {y.nombre}: neither technical sheet is in the system, so this pair was not searched."
            )
    return plan


def _ask_model(
    plan: list[tuple[CatalogueEntry, str, Medication, Medication]],
    review: Callable[[str, str], dict],
    result: CheckResult,
    progress: Progress | None,
) -> None:
    total = len(plan)
    for done, (sheet, ingredient, x, y) in enumerate(plan):
        label = f"{sheet.nombre} sheet, searched for {ingredient}"
        if progress:
            progress(done, total, f"Searching the {sheet.nombre} sheet for {ingredient}")
        try:
            answer = review(sheet.nregistro, ingredient)
        except Exception as exc:  # index or search failure: never "nothing found"
            result.log.append({"sheet": sheet.nregistro, "ingredient": ingredient,
                               "status": "exception", "reason": type(exc).__name__})
            result.could_not_check.append(f"{label}: the search failed ({type(exc).__name__}).")
            continue

        status, reason = answer.get("status"), answer.get("reason")
        result.log.append({"sheet": sheet.nregistro, "ingredient": ingredient,
                           "status": status, "reason": reason})
        if status == "found":
            response = answer.get("response") or {}
            citations = response.get("citations") or []
            if not citations:  # the pipeline should not allow this; be safe anyway
                result.could_not_check.append(f"{label}: the answer had no quotation.")
                continue
            who = _join([y.prescriptor, x.prescriptor if x.prescriptor != "Not specified" else ""])
            result.findings.append(Finding(
                tipo="interaccion",
                descripcion=response.get("description") or "Information found in the technical sheet.",
                cita=Citation.model_validate(citations[0]),
                pregunta_medico=(
                    f"Have the prescribers involved ({who}) reviewed the combined use of "
                    f"{x.nombre} and {y.nombre}?"
                ),
            ))
        elif status == "not_found":
            result.checked_without_evidence.append(label)
        else:
            why = REASON_TEXT.get(reason, f"the check did not finish ({status}, {reason})")
            result.could_not_check.append(f"{label}: {why}.")
    if progress:
        progress(total, total, "Done")


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def _run(
    medications: list[Medication],
    pairs: list[tuple[Medication, Medication]],
    catalogue: dict[str, CatalogueEntry],
    *,
    proposed_index: int | None,
    rule_data: tuple[dict, dict] | None,
    use_model: bool,
    review: Callable[[str, str], dict] | None,
    progress: Progress | None,
) -> CheckResult:
    result = CheckResult()
    if rule_data is not None:
        _rule_findings(medications, rule_data, proposed_index, result)
    else:
        proposed = medications[proposed_index] if proposed_index is not None else None
        _name_duplicates(pairs, catalogue, proposed, result)
        result.unchecked |= {"advertencia_atc", "cascada"}

    if use_model:
        if review is None:
            review = _import_rag().review_interaction
        _ask_model(_interaction_plan(pairs, catalogue, result), review, result, progress)
    else:
        result.unchecked.add("interaccion")
    return result


def check_new_medicine(
    patient: Patient,
    proposed: Medication,
    catalogue: dict[str, CatalogueEntry],
    *,
    rule_data: tuple[dict, dict] | None = None,
    use_model: bool = True,
    review: Callable[[str, str], dict] | None = None,
    progress: Progress | None = None,
) -> CheckResult:
    """Check ``proposed`` against every medicine in ``patient``.

    ``rule_data`` comes from ``load_rule_data()``; without it the rules part
    falls back to a name comparison. ``review`` defaults to
    ``src.rag.review_interaction``; tests pass a stand-in.
    """
    if proposed.nregistro not in catalogue:
        raise ValueError(f"{proposed.nombre} is not in the medicine list.")
    medications = list(patient.medicamentos) + [proposed]
    pairs = [(proposed, current) for current in patient.medicamentos]
    return _run(medications, pairs, catalogue, proposed_index=len(medications) - 1,
                rule_data=rule_data, use_model=use_model, review=review, progress=progress)


def review_patient(
    patient: Patient,
    catalogue: dict[str, CatalogueEntry],
    *,
    rule_data: tuple[dict, dict] | None = None,
    use_model: bool = False,
    review: Callable[[str, str], dict] | None = None,
    progress: Progress | None = None,
) -> CheckResult:
    """Check every pair of the patient's current medicines."""
    medications = list(patient.medicamentos)
    pairs = list(combinations(medications, 2))
    return _run(medications, pairs, catalogue, proposed_index=None,
                rule_data=rule_data, use_model=use_model, review=review, progress=progress)


def planned_questions(patient: Patient, proposed: Medication | None, catalogue: dict[str, CatalogueEntry]) -> int:
    """How many model questions a check or review will ask (to warn about time)."""
    meds = list(patient.medicamentos)
    pairs = [(proposed, m) for m in meds] if proposed is not None else list(combinations(meds, 2))
    return len(_interaction_plan(pairs, catalogue, CheckResult()))
