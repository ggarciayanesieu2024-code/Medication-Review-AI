"""Tests for the deterministic rules in src/reglas.py."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reglas import (  # noqa: E402
    build_medications,
    find_cascades,
    find_duplicates,
    find_family_warnings,
    load_cascades,
    load_drug_info,
    run_rules,
)

PATIENTS_DIR = ROOT / "data" / "pacientes"

# Codes checked in the AEMPS nomenclator (Prescripcion.xml, 2026-11-10).
DRUG_INFO = {
    "77758": {"atc": ["N02BE01"], "active_ingredients": ["350"]},   # Paracetamol Abamed 1 g
    "85780": {"atc": ["N02BE01"], "active_ingredients": ["350"]},   # Antidol 1 g
    "63064": {"atc": ["B01AA03"], "active_ingredients": ["4411"]},  # Aldocumar 5 mg
    "66337": {"atc": ["J01CR02"], "active_ingredients": ["160", "5418"]},  # Amoxicillin/clavulanic acid
    "65461": {"atc": ["C08CA01"], "active_ingredients": ["7177"]},  # Amlodipine 10 mg
    "55211": {"atc": ["A10BA02"], "active_ingredients": ["4379"]},  # Dianben 850 mg
    # Invented entries for rule tests only (ATC codes are real, IDs are not).
    "TEST-FUROSEMIDE": {"atc": ["C03CA01"], "active_ingredients": ["T1"]},
    "TEST-IBUPROFEN": {"atc": ["M01AE01"], "active_ingredients": ["T2"]},
    "TEST-NAPROXEN": {"atc": ["M01AE02"], "active_ingredients": ["T3"]},
    "TEST-LITHIUM": {"atc": ["N05AN01"], "active_ingredients": ["T4"]},
    "TEST-LEVODOPA": {"atc": ["N04BA01"], "active_ingredients": ["T5"]},
    "TEST-PARACETAMOL-CODEINE": {"atc": ["N02AJ06"], "active_ingredients": ["350", "T6"]},
}


def make_patient(*nregistros):
    """Build a minimal synthetic patient from a list of nregistro values."""
    return {
        "id": "TEST",
        "medicamentos": [
            {"nregistro": n, "nombre": n, "prescriptor": f"Doctor {i}"}
            for i, n in enumerate(nregistros)
        ],
    }


def load_patient(name):
    path = PATIENTS_DIR / f"paciente_{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cascades():
    return load_cascades()


# --- Cascade table -----------------------------------------------------------

def test_cascade_table_is_valid(cascades):
    ids = [c["id"] for c in cascades["cascades"]]
    assert len(ids) == len(set(ids)), "cascade ids must be unique"
    for c in cascades["cascades"]:
        assert c["index_atc"] and c["cascade_atc"]
        for source in c["sources"]:
            assert source in cascades["sources"], f"{c['id']}: unknown source {source}"


# --- Development patients ------------------------------------------------------

def test_patient_a_has_paracetamol_duplicate(cascades):
    findings = run_rules(load_patient("A"), DRUG_INFO, cascades)
    assert len(findings["duplicates"]) == 1
    assert findings["duplicates"][0].shared_ingredients == ["350"]


@pytest.mark.parametrize("name", ["B", "C"])
def test_patients_b_and_c_have_no_rule_findings(name, cascades):
    findings = run_rules(load_patient(name), DRUG_INFO, cascades)
    assert findings["duplicates"] == []
    assert findings["family_warnings"] == []
    assert findings["cascades"] == []


# --- Rule 1: duplicates --------------------------------------------------------

def test_combination_product_sharing_ingredient_is_duplicate():
    meds = build_medications(make_patient("77758", "TEST-PARACETAMOL-CODEINE"), DRUG_INFO)
    assert len(find_duplicates(meds)) == 1


def test_different_ingredients_are_not_duplicates():
    meds = build_medications(make_patient("65461", "55211"), DRUG_INFO)
    assert find_duplicates(meds) == []


# --- Rule 2: same ATC family -----------------------------------------------------

def test_two_nsaids_give_family_warning_not_duplicate():
    meds = build_medications(make_patient("TEST-IBUPROFEN", "TEST-NAPROXEN"), DRUG_INFO)
    assert find_duplicates(meds) == []
    warnings = find_family_warnings(meds)
    assert len(warnings) == 1
    assert warnings[0].atc_family == "M01AE"


def test_duplicate_pair_is_not_also_a_family_warning():
    meds = build_medications(make_patient("77758", "85780"), DRUG_INFO)
    assert find_family_warnings(meds) == []


# --- Rule 3: cascades -------------------------------------------------------------

def test_amlodipine_plus_diuretic_triggers_c01(cascades):
    meds = build_medications(make_patient("65461", "TEST-FUROSEMIDE"), DRUG_INFO)
    ids = [h.cascade_id for h in find_cascades(meds, cascades)]
    assert ids == ["C01"]


def test_amlodipine_alone_triggers_nothing(cascades):
    meds = build_medications(make_patient("65461"), DRUG_INFO)
    assert find_cascades(meds, cascades) == []


def test_lithium_is_excluded_from_antipsychotic_cascade(cascades):
    meds = build_medications(make_patient("TEST-LITHIUM", "TEST-LEVODOPA"), DRUG_INFO)
    assert find_cascades(meds, cascades) == []


def test_cascade_message_is_a_hypothesis(cascades):
    meds = build_medications(make_patient("65461", "TEST-FUROSEMIDE"), DRUG_INFO)
    hypothesis = find_cascades(meds, cascades)[0]
    assert hypothesis.message.startswith("Possible prescribing cascade to review")
    assert hypothesis.sources, "every hypothesis must carry its sources"


# --- Loading the parsed nomenclator --------------------------------------------------

def test_load_drug_info_merges_packages_of_same_medicine(tmp_path):
    # Same shape as the output of src/parse_nomenclator.py.
    parsed = {
        "listprescriptiondate": "2026-11-10",
        "prescripciones": [
            {"nro_definitivo": "77758", "cod_atc": "N02BE01",
             "composicion_pa": [{"cod_principio_activo": "350"}]},
            {"nro_definitivo": "77758", "cod_atc": "N02BE01",  # another package size
             "composicion_pa": [{"cod_principio_activo": "350"}]},
            {"nro_definitivo": "66337", "cod_atc": "J01CR02",
             "composicion_pa": [{"cod_principio_activo": "160"},
                                {"cod_principio_activo": "5418"}]},
        ],
    }
    path = tmp_path / "prescriptions.json"
    path.write_text(json.dumps(parsed), encoding="utf-8")

    info = load_drug_info(path)
    assert info["77758"] == {"atc": ["N02BE01"], "active_ingredients": ["350"]}
    assert info["66337"]["active_ingredients"] == ["160", "5418"]


# --- Errors ------------------------------------------------------------------------

def test_unknown_nregistro_raises_error():
    with pytest.raises(KeyError):
        build_medications(make_patient("00000"), DRUG_INFO)
