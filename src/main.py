"""Coordinate the checks behind a medication review.

Entry points:

* ``review_patient`` checks every pair of medicines a patient currently takes.
* ``check_new_medicine`` checks a proposed medicine against the patient's
  current medicines, retaining only findings involving the proposed medicine.
* ``review_request`` accepts the patient sheet and an optional question. Blank
  questions use the standard review; other questions receive a focused answer
  grounded in CIMA evidence and the shared rule results.

Both combine the available parts of the system:

1. **Deterministic rules (``src/Cascades/reglas.py``):** identify duplicate
   active ingredients, medicines in the same ATC family and possible prescribing
   cascades. These checks need ``data/processed/prescriptions.json`` and the
   cascade table. Without rule data, duplicates fall back to ingredient-name
   comparison using ``data/catalogo_medicamentos.json``; ATC-family and cascade
   checks are marked "not checked".
2. **RAG (``src/Rag/rag.py``):** search each medicine's technical sheet for the
   other medicine's active ingredients, in both directions where sheets are
   available. This needs the search index, model libraries and ``GROQ_API_KEY``.
   When disabled, interaction checks are marked "not checked".

This module plans the checks and adapts their results for reporting. The shared
rules module performs rule matching; the RAG pipeline retrieves evidence,
generates responses and verifies their citations.

RAG outcomes remain distinct:

* ``found`` means relevant evidence was found. ``interaction_detected`` decides
  whether it becomes a potential-interaction alert or documented information
  without an alert. All verified citations are retained.
* ``not_found`` means the search produced no relevant evidence. It does not
  establish that the combination is safe.
* ``rejected``, ``error`` or an exception means the check could not be completed.
  These outcomes are never reported as "nothing found".

Imports for optional components stay at their points of use, allowing
rules-only reviews without the RAG libraries, index or API key.
The system informs; the clinician decides.

``MAIN.ipynb`` at the repository root is the interactive runner for these
production functions. Scenario selection, patient overrides and execution
switches are caller inputs; the notebook passes the resolved Patient and
question to ``review_request``. It does not implement a separate review engine.
That entry point already forwards optional ``k`` and ``generation_options``
through patient-question planning, pair checks and the final grounded answer.

For a standalone question without a patient, callers use
``src.Rag.rag.answer_question(question, nregistro=None, sections=None, ...)``.
The notebook splits that same operation into preparation and generation so
the retrieved evidence and exact prompt can be inspected before a model call.
The website exposes standalone questions separately; ``review_request`` requires a Patient.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from functools import partial
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Callable

from src.config import DATA_DIR, DATABASE_DIR, PRESCRIPTIONS_FILE
from src.Rag.verificacion import Citation
from src.Reports.informe import Finding, Medication, Patient, normalise_text

CATALOGUE_FILE = DATA_DIR / "catalogo_medicamentos.json"
STORED_ANSWERS_FILE = DATA_DIR / "interacciones_guardadas.json"

REASON_TEXT = {
    "forbidden_phrase": "the model's answer was blocked by the safety check",
    "invalid_citation": "the model's answer quoted text that could not be verified",
    "invalid_response": "the model's answer was not in the expected format",
    "generation_failed": "the model could not be reached",
    "context_too_large": "the evidence exceeds the model request-size limit",
    "legacy_unclassified": "the historical answer requires updated interaction classification",
    "not_stored": "this search is not in the stored model answers",
}

Progress = Callable[[int, int, str], None]


@dataclass
class CatalogueEntry:
    """Medicine metadata used to plan interaction searches."""

    nregistro: str
    nombre: str
    principios_activos: list[str]
    ficha: bool
    atc: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
    """Findings, coverage limitations and the audit log for one review."""

    findings: list[Finding] = field(default_factory=list)
    checked_without_evidence: list[str] = field(default_factory=list)
    could_not_check: list[str] = field(default_factory=list)
    # Finding categories that could not be checked.
    unchecked: set[str] = field(default_factory=set)
    # One audit entry per RAG request, including failures.
    log: list[dict] = field(default_factory=list)


@dataclass
class QuestionResult:
    """A standard review or a focused answer with its supporting checks."""

    checks: CheckResult
    question: str = ""
    proposed: Medication | None = None
    answer: str | None = None
    citations: list[Citation] = field(default_factory=list)
    supporting_findings: list[Finding] = field(default_factory=list)
    clarification: str | None = None


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
            atc=list(item.get("atc", [])),
        )
        for item in data["medicamentos"]
    }


def model_status(database_dir: Path = DATABASE_DIR) -> tuple[bool, str]:
    """Return (ready, explanation) for the model, without calling it."""
    try:
        from src.Rag import rag  # Import only; do not load the index or call the API.
    except ImportError as exc:
        return False, (
            f"the model's libraries are not installed here ({exc.name}). "
            "Install the full requirements.txt."
        )
    from importlib.util import find_spec

    for package in ("chromadb", "sentence_transformers", "torch"):
        if find_spec(package) is None:
            return False, f"Required package missing: {package}."

    if not Path(database_dir).exists():
        return False, (
            "the search index is missing. Put the shared CIMA file in data/processed/ "
            "and run python -m src.Rag.indice."
        )
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "your_groq_api_key_here":
        return False, "GROQ_API_KEY is missing from the .env file."
    return True, "connected"


def rules_status(prescriptions_file: Path = PRESCRIPTIONS_FILE) -> tuple[bool, str]:
    """Check rule-module availability and the required nomenclator file."""
    try:
        from src.Cascades import reglas
    except Exception as exc:  # a SyntaxError or ImportError inside src/Cascades/reglas.py
        return False, f"src/Cascades/reglas.py could not be loaded ({type(exc).__name__}: {exc})."
    if not Path(prescriptions_file).exists():
        return True, (
            "connected, using the ATC codes in the medicine list "
            "(data/processed/prescriptions.json was not found)"
        )
    return True, "connected, using the nomenclator codes"


def load_rule_data(
    prescriptions_file: Path = PRESCRIPTIONS_FILE,
    catalogue: dict[str, CatalogueEntry] | None = None,
) -> tuple[dict, dict]:
    """(drug codes per registration number, cascade table) for the rules.

    Uses the nomenclator file when it exists. Otherwise builds the codes from the
    medicine list: its ATC codes, and its ingredient names in place of the
    nomenclator ingredient codes (the rules only compare them for equality).
    """
    from src.Cascades import reglas
    if Path(prescriptions_file).exists():
        drug_info = reglas.load_drug_info(Path(prescriptions_file))
    else:
        catalogue = catalogue if catalogue is not None else load_catalogue()
        drug_info = {
            n: {"atc": list(e.atc),
                "active_ingredients": [f"name:{normalise_text(i)}" for i in e.principios_activos]}
            for n, e in catalogue.items() if e.atc
        }
    return drug_info, reglas.load_cascades()


# --------------------------------------------------------------------------
# Stored model answers (for computers without the model, such as the shared link)
# --------------------------------------------------------------------------

@dataclass
class StoredAnswers:
    generated_at: str
    answers: dict[tuple[str, str], dict]


def load_stored_answers(path: str | Path = STORED_ANSWERS_FILE) -> StoredAnswers | None:
    """Read data/interacciones_guardadas.json, or None if it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return StoredAnswers(
        generated_at=data.get("generated_at", "an unknown date"),
        answers={(a["sheet"], normalise_text(a["ingredient"])): a for a in data.get("answers", [])},
    )


def stored_review(stored: StoredAnswers) -> Callable[[str, str], dict]:
    """A stand-in for review_interaction that answers from the stored file."""
    def review(nregistro: str, ingredient: str) -> dict:
        answer = stored.answers.get((nregistro, normalise_text(ingredient)))
        if answer is None:
            return {"status": "error", "reason": "not_stored", "response": None}
        response = answer.get("response")
        if answer["status"] == "found" and (
            not isinstance(response, dict)
            or type(response.get("interaction_detected")) is not bool
        ):
            return {
                "status": "error",
                "reason": "legacy_unclassified",
                "response": None,
            }
        return {"status": answer["status"], "reason": answer.get("reason"),
                "response": response}
    return review


def all_questions(catalogue: dict[str, CatalogueEntry]) -> list[tuple[str, str]]:
    """Every (sheet, ingredient) search any patient or new medicine can need."""
    ingredients: dict[str, str] = {}
    for entry in catalogue.values():
        for ingredient in entry.principios_activos:
            ingredients.setdefault(normalise_text(ingredient), ingredient)
    questions = []
    for entry in sorted(catalogue.values(), key=lambda e: e.nregistro):
        if not entry.ficha:
            continue
        own = {normalise_text(i) for i in entry.principios_activos}
        questions += [(entry.nregistro, name) for key, name in sorted(ingredients.items()) if key not in own]
    return questions


def save_answers(
    path: str | Path,
    review: Callable[[str, str], dict],
    questions: list[tuple[str, str]],
    *,
    pause: float = 0.0,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Ask the questions not stored yet and save the answers after each one.

    Found, not found and rejected answers are kept. Errors (for example the API
    limit) are left out, so running the script again retries only those.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"answers": []}
    stored = {(a["sheet"], normalise_text(a["ingredient"])) for a in data["answers"]}
    counts = {"asked": 0, "skipped": 0, "kept": 0, "retry_later": 0}
    total = len(questions)
    for number, (sheet, ingredient) in enumerate(questions, start=1):
        if (sheet, normalise_text(ingredient)) in stored:
            counts["skipped"] += 1
            continue
        counts["asked"] += 1
        try:
            answer = review(sheet, ingredient)
        except Exception as exc:
            counts["retry_later"] += 1
            log(f"[{number}/{total}] {sheet} / {ingredient}: failed ({type(exc).__name__}); run again later")
            continue
        status = answer.get("status")
        if status not in ("found", "not_found", "rejected"):
            counts["retry_later"] += 1
            log(f"[{number}/{total}] {sheet} / {ingredient}: {status} ({answer.get('reason')}); run again later")
            continue
        data["answers"].append({
            "sheet": sheet,
            "ingredient": ingredient,
            "status": status,
            "reason": answer.get("reason"),
            "response": answer.get("response"),
            "sources": [{"nregistro": f.get("nregistro"), "seccion": f.get("seccion")}
                        for f in answer.get("fragments") or [] if isinstance(f, dict)],
        })
        counts["kept"] += 1
        data["generated_at"] = date.today().isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        log(f"[{number}/{total}] {sheet} / {ingredient}: {status}")
        if pause and answer.get("fragments"):  # only searches that reached the model count
            time.sleep(pause)
    return counts


# --------------------------------------------------------------------------
# Public review API
# --------------------------------------------------------------------------

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
    ``src.Rag.rag.review_interaction``; tests pass a stand-in.
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


def review_request(
    patient: Patient,
    catalogue: dict[str, CatalogueEntry],
    question: str | None = None,
    *,
    rule_data: tuple[dict, dict] | None = None,
    use_model: bool = False,
    review: Callable[[str, str], dict] | None = None,
    progress: Progress | None = None,
    k: int | None = None,
    generation_options: dict | None = None,
) -> QuestionResult:
    """Review the patient sheet, optionally answering a doctor's question.

    A blank question follows the existing review exactly, without a planning
    or synthesis call. A nonempty question is resolved against the catalogue;
    ambiguous medicines require clarification rather than a guessed product.
    Optional k and generation_options apply to the searches and answers in
    this request; omitted values retain the project's defaults.
    """
    if k is not None and k < 1:
        raise ValueError("k must be at least 1.")
    rag_options = {}
    if k is not None:
        rag_options["k"] = k
    if generation_options:
        rag_options["generation_options"] = generation_options
    if use_model and review is None and rag_options:
        from src.Rag.rag import review_interaction

        review = partial(review_interaction, **rag_options)
    question = (question or "").strip()
    unknown = {medicine.nregistro for medicine in patient.medicamentos} - catalogue.keys()
    if unknown:
        return QuestionResult(
            checks=CheckResult(), question=question,
            clarification="These patient medicines are not in the supported catalogue: "
                          + ", ".join(sorted(unknown)) + ". Their identity must be resolved before review.",
        )
    if not question:
        return QuestionResult(checks=review_patient(
            patient, catalogue, rule_data=rule_data, use_model=use_model,
            review=review, progress=progress,
        ))
    result = QuestionResult(checks=CheckResult(), question=question)
    if not use_model:
        result.clarification = (
            "Question answering is unavailable because the model is not connected. "
            "Leave the question empty to run the available standard checks."
        )
        return result

    from src.llm import llm_prepare_question
    from src.Rag.rag import answer_patient_question

    if progress:
        progress(0, 1, "Understanding the question")
    try:
        plan = llm_prepare_question(
            question, patient.model_dump(mode="json"),
            [{"nregistro": entry.nregistro, "nombre": entry.nombre,
              "principios_activos": entry.principios_activos} for entry in catalogue.values()],
            **(generation_options or {}),
        )
    except Exception as exc:
        result.clarification = "The question could not be interpreted. Try again or leave it empty for a standard review."
        result.checks.could_not_check.append(f"Question interpretation failed ({type(exc).__name__}).")
        return result
    if plan.intent == "clarify":
        result.clarification = plan.clarification or "Which medicine or patient detail do you mean?"
        return result
    requested = set(plan.medicine_ids)
    if plan.intent == "new_medicine":
        if not plan.proposed_nregistro:
            result.clarification = "Which specific medicine are you considering adding?"
            return result
        requested.add(plan.proposed_nregistro)
    if requested - catalogue.keys():
        result.clarification = "The medicine could not be matched to the available catalogue. Please specify the product."
        return result

    if plan.intent == "new_medicine":
        entry = catalogue[plan.proposed_nregistro]
        result.proposed = Medication(nregistro=entry.nregistro, nombre=entry.nombre,
                                     dosis="Not specified", prescriptor="Not specified")
        result.checks = check_new_medicine(
            patient, result.proposed, catalogue, rule_data=rule_data,
            use_model=True, review=review, progress=progress,
        )
    else:
        result.checks = review_patient(
            patient, catalogue, rule_data=rule_data,
            use_model=True, review=review, progress=progress,
        )

    scope = list(plan.medicine_ids)
    if result.proposed or not scope:
        scope += [medicine.nregistro for medicine in patient.medicamentos]
    if result.proposed:
        scope.append(result.proposed.nregistro)
    limitations = result.checks.could_not_check + [
        f"Not checked: {kind}" for kind in sorted(result.checks.unchecked)
    ] + [f"No evidence found: {item}" for item in result.checks.checked_without_evidence]
    if progress:
        progress(0, 1, "Answering the question from the available evidence")
    patient_context = patient.model_dump(mode="json")
    if result.proposed:
        patient_context["proposed_medicine"] = result.proposed.model_dump(mode="json")
    try:
        answer = answer_patient_question(
            question, plan.search_question, patient_context, scope,
            [finding.model_dump(mode="json") for finding in result.checks.findings], limitations,
            **rag_options,
        )
    except Exception as exc:
        result.checks.could_not_check.append(f"Question search failed ({type(exc).__name__}).")
        answer = {"status": "error", "reason": "generation_failed", "response": None}
    result.checks.log.append({"question": question, "status": answer["status"], "reason": answer["reason"]})
    response = answer.get("response")
    if response is not None:
        result.answer = response["answer"]
        result.citations = [Citation.model_validate(item) for item in response["citations"]]
        result.supporting_findings = [
            finding for i, finding in enumerate(result.checks.findings, 1)
            if f"F{i}" in response["finding_ids"]
        ]
    else:
        reason = REASON_TEXT.get(answer["reason"], "the answer could not be verified",)
        result.answer = f"The question could not be answered: {reason}. See the available checks below."
        result.checks.could_not_check.append(f"Question answer: {reason}.")
    if progress:
        progress(1, 1, "Done")
    return result


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
    """Adapt shared rule results to report findings.

    When checking a proposed medicine, retain only findings involving it.
    Missing registrations are reported and excluded before rule execution.
    """
    from src.Cascades import reglas
    drug_info, cascade_table = rule_data
    available = []
    proposed_position = None
    for i, med in enumerate(medications):
        if med.nregistro not in drug_info:
            result.could_not_check.append(
                f"{med.nombre}: not found in the nomenclator data, so it was left out of the "
                "duplicate, drug-family and cascade checks."
            )
            continue
        if i == proposed:
            proposed_position = len(available)
        available.append(med.model_dump())

    rule_meds = reglas.build_medications({"medicamentos": available}, drug_info)
    keep = rule_meds[proposed_position] if proposed_position is not None else None
    if proposed is not None and keep is None:
        return  # the proposed medicine has no codes; already noted above

    def involved(*meds) -> bool:
        return keep is None or any(m is keep for m in meds)

    for d in reglas.find_duplicates(rule_meds):
        a, b = d.medication_a, d.medication_b
        if not involved(a, b):
            continue
        other = b if a is keep else a
        question = (
            f"The patient already takes {other.name}. What total daily amount of the "
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
    """Provide a name-based duplicate check when nomenclator codes are unavailable.

    This fallback does not attempt ATC-family or cascade matching.
    """
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
# RAG request planning and result handling
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
        # Identical single-ingredient medicines are handled
        # by duplicate detection, not drug-pair interaction checks.
        ingredients_x = {
            ing.strip().casefold() for ing in ex.principios_activos
        }
        ingredients_y = {
            ing.strip().casefold() for ing in ey.principios_activos
        }

        if len(ingredients_x) == 1 and ingredients_x == ingredients_y:
            continue
        # Duplicate detection and evidence retrieval are separate checks.
        # Shared ingredients must not hide interactions in combination products.
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
    """Convert RAG outcomes to report results without reinterpreting evidence."""
    request_cache = {}
    total = len(plan)
    for done, (sheet, ingredient, x, y) in enumerate(plan):
        label = f"{sheet.nombre} sheet, searched for {ingredient}"
        if progress:
            progress(done, total, f"Searching the {sheet.nombre} sheet for {ingredient}")
        try:
            cache_key = (sheet.nregistro, ingredient)

            if cache_key in request_cache:
                answer = request_cache[cache_key]
            else:
                answer = review(sheet.nregistro, ingredient)

                if answer.get("status") in {"found", "not_found"}:
                    request_cache[cache_key] = answer
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
            if not citations:  # A found response must include supporting evidence.
                result.could_not_check.append(f"{label}: the answer had no quotation.")
                continue
            interaction = response.get("interaction_detected")
            if interaction is not True and interaction is not False:
                result.could_not_check.append(f"{label}: {REASON_TEXT['invalid_response']}.")
                continue
            who = _join([y.prescriptor, x.prescriptor if x.prescriptor != "Not specified" else ""])
            result.findings.append(Finding(
                tipo="interaccion" if interaction else "informacion",
                descripcion=response.get("description") or "Information found in the technical sheet.",
                cita=Citation.model_validate(citations[0]),
                citations=[Citation.model_validate(citation) for citation in citations],
                pregunta_medico=(
                    f"Have the prescribers involved ({who}) reviewed the combined use of "
                    f"{x.nombre} and {y.nombre}?"
                ) if interaction else None,
            ))
        elif status == "not_found":
            result.checked_without_evidence.append(label)
        else:
            why = REASON_TEXT.get(reason, f"the check did not finish ({status}, {reason})")
            result.could_not_check.append(f"{label}: {why}.")
    if progress:
        progress(total, total, "Done")


# --------------------------------------------------------------------------
# Shared execution flow
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
    """Run rule checks, then the planned RAG requests, without changing their logic."""
    result = CheckResult()
    if rule_data is not None:
        _rule_findings(medications, rule_data, proposed_index, result)
    else:
        proposed = medications[proposed_index] if proposed_index is not None else None
        _name_duplicates(pairs, catalogue, proposed, result)
        result.unchecked |= {"advertencia_atc", "cascada"}

    if use_model:
        if review is None:
            from src.Rag.rag import review_interaction

            review = review_interaction
        _ask_model(_interaction_plan(pairs, catalogue, result), review, result, progress)
    else:
        result.unchecked.add("interaccion")
    return result
