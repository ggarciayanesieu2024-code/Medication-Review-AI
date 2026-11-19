"""Placeholder for deterministic medication-review rules."""
"""Deterministic rules: duplicates, same-family warnings and prescribing cascades.

No AI is used here. These checks work on structured codes (ATC and active
ingredient codes from the AEMPS nomenclator), so the result is always the same
for the same input and is easy to test and explain.

The rules never recommend stopping, changing or starting a medicine. They only
flag findings for a professional to review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

# Number of ATC characters that define a chemical subgroup (e.g. "N02BE").
ATC_FAMILY_LENGTH = 5

DEFAULT_CASCADES_PATH = Path(__file__).resolve().parent.parent / "data" / "cascadas.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Medication:
    """One medicine of a patient, with the codes the rules need."""

    nregistro: str
    name: str
    prescriber: str
    atc: list[str] = field(default_factory=list)
    active_ingredients: list[str] = field(default_factory=list)


@dataclass
class DuplicateFinding:
    """Two medicines that share at least one active ingredient."""

    medication_a: Medication
    medication_b: Medication
    shared_ingredients: list[str]
    message: str


@dataclass
class FamilyWarning:
    """Two different medicines from the same ATC chemical subgroup."""

    medication_a: Medication
    medication_b: Medication
    atc_family: str
    message: str


@dataclass
class CascadeHypothesis:
    """A possible prescribing cascade. Always a hypothesis, never a conclusion."""

    cascade_id: str
    index_medication: Medication
    cascade_medication: Medication
    adverse_effect: str
    question: str
    sources: list[str]
    note: str | None
    message: str


# ---------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------

def load_cascades(path: Path = DEFAULT_CASCADES_PATH) -> dict:
    """Load the hand-curated cascade table ({"sources": ..., "cascades": [...]})."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_medications(patient: dict, drug_info: dict[str, dict]) -> list[Medication]:
    """Combine a patient's medicines with their codes from the nomenclator.

    `drug_info` maps each nregistro to {"atc": [...], "active_ingredients": [...]}.
    It should be built from the parsed nomenclator. If the parser uses other
    key names, adapt them here only.
    """
    medications = []
    for med in patient["medicamentos"]:
        info = drug_info.get(med["nregistro"])
        if info is None:
            raise KeyError(f"nregistro {med['nregistro']} not found in the nomenclator data")
        medications.append(
            Medication(
                nregistro=med["nregistro"],
                name=med["nombre"],
                prescriber=med.get("prescriptor", ""),
                atc=list(info["atc"]),
                active_ingredients=list(info["active_ingredients"]),
            )
        )
    return medications


# ---------------------------------------------------------------------------
# Rule 1: duplicates
# ---------------------------------------------------------------------------

def find_duplicates(medications: list[Medication]) -> list[DuplicateFinding]:
    """Flag pairs of medicines that share an active ingredient.

    A shared ingredient also catches combination products, e.g. a cold remedy
    that contains paracetamol taken together with plain paracetamol.
    """
    findings = []
    for a, b in combinations(medications, 2):
        shared = sorted(set(a.active_ingredients) & set(b.active_ingredients))
        if shared:
            findings.append(
                DuplicateFinding(
                    medication_a=a,
                    medication_b=b,
                    shared_ingredients=shared,
                    message=(
                        f"'{a.name}' ({a.prescriber}) and '{b.name}' ({b.prescriber}) "
                        f"contain the same active ingredient. Possible duplicate to review."
                    ),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Rule 2: same ATC family
# ---------------------------------------------------------------------------

def _families(med: Medication) -> set[str]:
    return {code[:ATC_FAMILY_LENGTH] for code in med.atc if len(code) >= ATC_FAMILY_LENGTH}


def find_family_warnings(medications: list[Medication]) -> list[FamilyWarning]:
    """Flag pairs from the same ATC chemical subgroup that are not duplicates.

    Example: two different NSAIDs (both M01AE). Pairs already reported as
    duplicates are skipped, so each pair is reported only once.
    """
    warnings = []
    for a, b in combinations(medications, 2):
        if set(a.active_ingredients) & set(b.active_ingredients):
            continue  # already a duplicate
        shared = sorted(_families(a) & _families(b))
        if shared:
            warnings.append(
                FamilyWarning(
                    medication_a=a,
                    medication_b=b,
                    atc_family=shared[0],
                    message=(
                        f"'{a.name}' and '{b.name}' belong to the same drug family "
                        f"(ATC {shared[0]}). Warning to review."
                    ),
                )
            )
    return warnings


# ---------------------------------------------------------------------------
# Rule 3: prescribing cascades
# ---------------------------------------------------------------------------

def _matches(med: Medication, prefixes: list[str], excluded: list[str]) -> bool:
    """True if any ATC code of the medicine starts with one of the prefixes."""
    return any(
        code.startswith(tuple(prefixes)) and not code.startswith(tuple(excluded))
        for code in med.atc
    )


def find_cascades(medications: list[Medication], cascade_table: dict) -> list[CascadeHypothesis]:
    """Flag possible prescribing cascades from the hand-curated table.

    A hypothesis is created when the patient takes a medicine matching the
    index drug AND another medicine matching the cascade drug. The order of
    prescription is not checked because the patient data has no dates.
    """
    hypotheses = []
    for cascade in cascade_table["cascades"]:
        excluded = cascade.get("exclude_atc", [])
        index_meds = [m for m in medications if _matches(m, cascade["index_atc"], excluded)]
        cascade_meds = [m for m in medications if _matches(m, cascade["cascade_atc"], excluded)]
        for index_med in index_meds:
            for cascade_med in cascade_meds:
                if index_med is cascade_med:
                    continue
                hypotheses.append(
                    CascadeHypothesis(
                        cascade_id=cascade["id"],
                        index_medication=index_med,
                        cascade_medication=cascade_med,
                        adverse_effect=cascade["adverse_effect"],
                        question=cascade["question"],
                        sources=[cascade_table["sources"][s] for s in cascade["sources"]],
                        note=cascade.get("note"),
                        message=(
                            f"Possible prescribing cascade to review: '{index_med.name}' "
                            f"can cause {cascade['adverse_effect'].lower()}, and the patient "
                            f"also takes '{cascade_med.name}'."
                        ),
                    )
                )
    return hypotheses


# ---------------------------------------------------------------------------
# All rules together
# ---------------------------------------------------------------------------

def run_rules(patient: dict, drug_info: dict[str, dict], cascade_table: dict | None = None) -> dict:
    """Run the three rules on one patient and return all findings."""
    if cascade_table is None:
        cascade_table = load_cascades()
    medications = build_medications(patient, drug_info)
    return {
        "duplicates": find_duplicates(medications),
        "family_warnings": find_family_warnings(medications),
        "cascades": find_cascades(medications, cascade_table),
    }
