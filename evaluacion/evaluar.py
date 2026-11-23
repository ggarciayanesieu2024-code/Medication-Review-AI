"""Evaluation of the Medication Review AI against the labelled patients.

WHY THIS FILE EXISTS
--------------------
We claim the system finds duplicates, interactions and cascades correctly. To
show that, we need numbers: we give it patients whose correct answer we wrote
down BEFORE running anything, and we count how often the system agrees.

WHAT IT DOES, for each evaluation patient
(data/pacientes/paciente_E*.json and data/pacientes/evaluacion_pacients.json):
1. Runs the same review the app runs: Person 2's rules (duplicates, same ATC
   family, cascades) plus the stored model answers
   (data/interacciones_guardadas.json). The stored answers mean no Groq API key
   and no internet are needed to evaluate.
2. Compares what the system found with the patient's expected result, one check
   at a time: duplicates, interactions (including which section is cited),
   cascades, grouping by prescriber, and safety (no forbidden phrase appears in
   any text the system generated).
3. Prints a table and writes evaluacion/resultados.json.

HOW RESULTS ARE REPORTED
------------------------
As "passed of total" AND as a percentage, always together, for example
"5 of 6 (83%)". The count is shown next to the percentage on purpose: there are
only a handful of patients, so a single failure moves the percentage a lot and
the reader must be able to see how many cases it is based on.

Coverage is reported separately. The number of searches that could not be
completed (for example because a technical sheet is missing) is printed next to
each patient, so a pass is never mistaken for "everything was checked".

IMPORTANT RULE
--------------
Expected results are written before running and are never adjusted to fit the
output. Two formats exist today: patients E1 to E5 use the key
``expected_result`` and EVAL_P4 uses ``resultado_esperado``. This script reads
both so nobody has to rewrite the patient files.

Run from the project root:
    python evaluacion/evaluar.py            # every evaluation patient
    python evaluacion/evaluar.py --only E5  # a single patient
"""

# Makes type hints such as "tuple[bool, str] | None" work on older Python 3
# versions, because with this line the hints are not evaluated while running.
from __future__ import annotations

import argparse  # reads the command line options (--only)
import json  # reads the patient files and writes resultados.json
import re  # regular expressions, used to read text like "59718, sección 4.5"
import sys  # used below to tell Python where the project's own modules are
from datetime import date  # stamps the results file with the day it was run
from pathlib import Path  # builds file paths that work on Windows and Linux

# This file lives in evaluacion/, one folder below the project root. Path(__file__)
# is this file, .resolve() makes it an absolute path, and .parent.parent climbs
# two levels (evaluacion/ -> project root).
ROOT = Path(__file__).resolve().parent.parent

# When you run "python evaluacion/evaluar.py", Python only looks for imports in
# the evaluacion/ folder. We add the project root so that "from src.xxx import"
# finds our own code in src/. The "not in" check avoids adding it twice.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # position 0 = look here first

# These imports must come AFTER the sys.path line above, otherwise Python could
# not find the "src" package. "noqa: E402" tells the code checker that the late
# position is on purpose.
from src.config import FORBIDDEN_PHRASES  # noqa: E402  (the 10 banned phrases)
from src.modelos import Patient  # noqa: E402  (the validated patient model)
from src.revision import (  # noqa: E402  (the review code the app itself uses)
    load_catalogue,  # reads data/catalogo_medicamentos.json
    load_rule_data,  # builds the ATC and ingredient codes the rules need
    load_stored_answers,  # reads data/interacciones_guardadas.json
    review_patient,  # reviews every pair of a patient's medicines
    stored_review,  # answers model questions from the stored file
)

# Folder that holds the labelled patients.
PATIENTS_DIR = ROOT / "data" / "pacientes"
# File where the results of each run are written.
RESULTS_FILE = ROOT / "evaluacion" / "resultados.json"

# The five kinds of check, in the order they are shown in the table. A patient
# only gets a check if its expected result contains something to compare with.
CHECK_TYPES = ["duplicates", "interactions", "cascades", "grouping", "safety"]


# --------------------------------------------------------------------------
# Loading the patients
# --------------------------------------------------------------------------

def load_patients() -> list[dict]:
    """Return the raw patient dictionaries, including their expected result.

    We keep them as plain dictionaries (not Patient objects) because the
    expected result is stored inside the same JSON file, and the Patient model
    rejects unknown fields.
    """
    patients = []  # will hold one dictionary per patient
    # glob finds paciente_E1.json ... paciente_E5.json; sorted() keeps E1..E5
    # in a stable order so the table always looks the same.
    for path in sorted(PATIENTS_DIR.glob("paciente_E*.json")):
        # Read the file as text (UTF-8 keeps accents such as "sección") and
        # convert the JSON text into a Python dictionary.
        patients.append(json.loads(path.read_text(encoding="utf-8")))

    # EVAL_P4 lives in its own file. It may hold one patient (a dictionary) or a
    # list of patients, so we accept both shapes.
    extra = PATIENTS_DIR / "evaluacion_pacients.json"
    if extra.exists():  # skip silently if the file is missing
        data = json.loads(extra.read_text(encoding="utf-8"))
        # "+=" on a list appends every element of the right-hand list.
        patients += data if isinstance(data, list) else [data]
    return patients  # the list of raw patient dictionaries


def expected_of(raw: dict) -> dict:
    """Return the expected result of a patient, whichever key it uses.

    E1 to E5 call it "expected_result" and EVAL_P4 calls it "resultado_esperado".
    "or" returns the first one that exists, and {} (nothing expected) if neither.
    """
    return raw.get("expected_result") or raw.get("resultado_esperado") or {}


def as_patient(raw: dict) -> Patient:
    """Turn the raw dictionary into the Patient object the review code expects.

    The Patient model forbids extra fields (a safety measure of the project), so
    we copy only the four fields it knows and leave the expected result out.
    """
    # The "**" unpacks the dictionary into keyword arguments: Patient(id=...,
    # edad=..., grupo=..., medicamentos=...). Pydantic validates every field.
    return Patient(**{k: raw[k] for k in ("id", "edad", "grupo", "medicamentos")})


# --------------------------------------------------------------------------
# What the system found
# --------------------------------------------------------------------------

def observed(raw: dict, patient: Patient, catalogue, rule_data, review) -> dict:
    """Run the review for one patient and return its findings in a form that
    can be compared with the expected result."""
    # Imported here, not at the top, so a problem inside reglas.py only breaks
    # the evaluation when it is actually run.
    import src.reglas as reglas

    # rule_data is a pair: the ATC/ingredient codes per medicine, and the table
    # of the 13 hand-curated cascades. We split it into two names.
    drug_info, cascade_table = rule_data

    # Person 2's rules, called directly. This gives us STRUCTURED findings that
    # say exactly which two medicines (by registration number) are involved,
    # which the text-only report findings do not.
    rules = reglas.run_rules(raw, drug_info, cascade_table)

    # The same review the app runs. use_model=True asks the "model" questions,
    # but "review" is the stored-answers stand-in, so no API call happens.
    result = review_patient(patient, catalogue, rule_data=rule_data, use_model=True, review=review)

    # A duplicate is a pair of medicines. frozenset((a, b)) is an unordered pair,
    # so (A, B) and (B, A) count as the same duplicate.
    duplicates = {frozenset((d.medication_a.nregistro, d.medication_b.nregistro))
                  for d in rules["duplicates"]}

    # Same-ATC-family warnings, collected the same way. They are reported but
    # NOT scored (see the E4 note in docs/person5.md: the group still has to
    # decide whether that warning is "expected" for E4).
    family = {frozenset((w.medication_a.nregistro, w.medication_b.nregistro))
              for w in rules["family_warnings"]}

    # Cascades: a list of (trigger medicine name, treatment medicine name).
    cascades = [(c.index_medication.name, c.cascade_medication.name) for c in rules["cascades"]]

    # INTERACTIONS. The review log has one entry per model question, saying which
    # technical sheet was searched and for which ingredient, and whether the
    # answer was "found". The finding itself only keeps the sheet that holds the
    # quotation, not the second medicine. So we rebuild the pair: the second
    # medicine is the patient's medicine whose ingredient was searched for.
    interactions: dict[frozenset, set] = {}  # pair -> {(sheet, section), ...}
    # Log entries whose model answer was "found" (the others found nothing).
    found_logs = [e for e in result.log if e["status"] == "found"]
    # The interaction findings the review produced, in the same order.
    found_findings = [f for f in result.findings if f.tipo == "interaccion"]
    # Both lists are filled in the same order (one finding per "found" answer),
    # so zip() walks them side by side.
    for entry, finding in zip(found_logs, found_findings):
        for med in patient.medicamentos:  # try every medicine of the patient
            cat = catalogue.get(med.nregistro)  # its catalogue entry, if any
            # Skip the sheet's own medicine, then check whether this medicine
            # contains the ingredient that was searched for (case-insensitive).
            if med.nregistro != entry["sheet"] and cat and any(
                i.casefold() == entry["ingredient"].casefold() for i in cat.principios_activos
            ):
                # The pair = the sheet's medicine + the medicine that has the
                # searched ingredient. frozenset again, so order does not matter.
                pair = frozenset((entry["sheet"], med.nregistro))
                # setdefault creates the empty set the first time we see a pair.
                cited = interactions.setdefault(pair, set())
                if finding.cita:  # only if the finding has a quotation
                    # Remember (sheet, section) of the quotation, so we can check
                    # later that the right section was cited (for example 4.5).
                    cited.add((finding.cita.nregistro, finding.cita.seccion))

    # Grouping by prescriber: {prescriber: [registration numbers]}. This is the
    # same grouping the report prints in its first block.
    groups: dict[str, list[str]] = {}
    for med in patient.medicamentos:
        # setdefault gives an empty list the first time a prescriber appears.
        groups.setdefault(med.prescriptor, []).append(med.nregistro)

    # Every sentence the system wrote (descriptions and questions for the
    # doctor). The safety check scans these for forbidden phrases. "or ''" turns
    # a missing question (None) into an empty string so the scan never breaks.
    texts = [f.descripcion for f in result.findings] + [
        f.pregunta_medico or "" for f in result.findings
    ]
    return {
        "duplicates": duplicates, "family": family, "cascades": cascades,
        "interactions": interactions, "groups": groups, "texts": texts,
        # Coverage numbers: searches that failed or were impossible, and pairs
        # that were searched and gave nothing.
        "not_completed": len(result.could_not_check),
        "no_evidence": len(result.checked_without_evidence),
    }


# --------------------------------------------------------------------------
# Comparison with the expected result
# Each check returns (passed: bool, detail: str). The detail is a short text
# such as "expected 1, found 0" that explains a failure at a glance.
# --------------------------------------------------------------------------

def check_duplicates(exp: dict, obs: dict) -> tuple[bool, str]:
    """The duplicates found must be exactly the expected ones, no more, no less."""
    # Expected duplicates as unordered pairs of registration numbers.
    want = {frozenset(d["nregistros"]) for d in exp.get("duplicidades", [])}
    # Sets are equal only if they hold exactly the same pairs.
    return obs["duplicates"] == want, f"expected {len(want)}, found {len(obs['duplicates'])}"


def check_interactions(exp: dict, obs: dict) -> tuple[bool, str]:
    """The interaction pairs found must match the expected ones exactly, and the
    cited section must be the expected one."""
    wanted = exp.get("interacciones", [])  # expected interactions (may be empty)
    want_pairs = {frozenset(i["par"]) for i in wanted}  # as unordered pairs
    got_pairs = set(obs["interactions"])  # the pairs the system found
    # Same pairs in both directions: nothing missing and nothing extra.
    ok = got_pairs == want_pairs
    for item in wanted:
        # Some patients also say WHICH section must be cited, for example E3:
        # "59718, sección 4.5, apartado Warfarina". The regular expression pulls
        # out the sheet number (group 1) and the section number (group 2).
        # secci[oó]n accepts the word with or without the accent.
        m = re.match(r"\s*(\d+),\s*secci[oó]n\s*([\d.]+)", item.get("cita_principal", ""))
        pair = frozenset(item["par"])
        if m and pair in obs["interactions"]:  # only if there is a section to check
            sheet, section = m.groups()  # e.g. "59718" and "4.5"
            # startswith lets "4.5" also accept a subsection such as "4.5.1".
            # If the system cited another section (for example 4.4), it fails,
            # even though the interaction itself was detected.
            ok = ok and any(s == sheet and sec.startswith(section)
                            for s, sec in obs["interactions"][pair])
    return ok, f"expected {len(want_pairs)}, found {len(got_pairs)}"


def check_cascades(exp: dict, obs: dict) -> tuple[bool, str]:
    """The cascade hypotheses must match the expected ones.

    If a patient lists no cascades, finding any is a failure: a false alarm
    counts as an error just like a missed cascade.
    """
    wanted = exp.get("cascadas", [])  # expected cascades (may be empty)
    found = list(obs["cascades"])  # the cascades the system proposed
    # Same count first, so extra or missing cascades are caught.
    ok = len(found) == len(wanted)
    for item in wanted:
        # Lower-case both names so capitalisation never causes a false failure.
        trig, treat = item["desencadenante"].casefold(), item["tratamiento"].casefold()
        # The expected file gives short names ("amlodipino", "furosemida") and the
        # system gives full medicine names ("Amlodipino Cinfa 10 mg"), so we test
        # that the short name is CONTAINED in the full one.
        ok = ok and any(trig in a.casefold() and treat in b.casefold() for a, b in found)
    return ok, f"expected {len(wanted)}, found {len(found)}"


def check_grouping(exp: dict, obs: dict) -> tuple[bool, str] | None:
    """Compare the grouping by prescriber. Returns None when the patient has no
    grouping label, so this check is simply not scored for it."""
    # EVAL_P4 gives the exact groups: {prescriber: [registration numbers]}.
    if "agrupacion_por_prescriptor" in exp:
        # sorted() so the order of the numbers inside a group does not matter.
        want = {k: sorted(v) for k, v in exp["agrupacion_por_prescriptor"].items()}
        got = {k: sorted(v) for k, v in obs["groups"].items()}
        return got == want, f"expected {len(want)} groups, found {len(got)}"
    # E5 only states how many groups: "un grupo por prescriptor (4 grupos)".
    if "agrupacion" in exp:
        # Find the number inside the parentheses, e.g. "(4 grupos)" -> "4".
        m = re.search(r"\((\d+)\s+grupos?\)", exp["agrupacion"])
        if m:
            return len(obs["groups"]) == int(m.group(1)), (
                f"expected {m.group(1)} groups, found {len(obs['groups'])}")
    return None  # no label for grouping: do not score this check


def check_safety(obs: dict) -> tuple[bool, str]:
    """No generated text may contain a forbidden phrase (project hard rule: the
    system never recommends stopping, changing or starting a medicine)."""
    # casefold() ignores upper/lower case. The list comprehension collects every
    # forbidden phrase found in any generated text.
    bad = [p for p in FORBIDDEN_PHRASES for t in obs["texts"] if p.casefold() in t.casefold()]
    return not bad, f"{len(bad)} forbidden phrases"  # empty list -> passed


def evaluate(raw: dict, catalogue, rule_data, review) -> dict:
    """Evaluate one patient: run the review, then run every applicable check."""
    exp = expected_of(raw)  # what the patient SHOULD produce
    obs = observed(raw, as_patient(raw), catalogue, rule_data, review)  # what it DID
    checks = {}  # check name -> (passed, detail)
    # Duplicates, interactions and cascades are only scored when the patient has
    # a label for them. EVAL_P4 only labels the grouping, so it skips these three
    # instead of being scored against an expectation nobody wrote.
    if "duplicidades" in exp or "interacciones" in exp:
        checks["duplicates"] = check_duplicates(exp, obs)
        checks["interactions"] = check_interactions(exp, obs)
        checks["cascades"] = check_cascades(exp, obs)
    grouping = check_grouping(exp, obs)  # None if the patient has no label
    if grouping:
        checks["grouping"] = grouping
    # Safety is checked for every patient.
    checks["safety"] = check_safety(obs)
    return {
        "patient": raw["id"],
        # A patient passes only if EVERY one of its checks passes.
        "passed": all(ok for ok, _ in checks.values()),
        # Turn each (ok, detail) tuple into a small dictionary for the JSON file.
        "checks": {k: {"ok": ok, "detail": d} for k, (ok, d) in checks.items()},
        "family_warnings_not_scored": len(obs["family"]),
        "searches_not_completed": obs["not_completed"],
        "pairs_no_evidence": obs["no_evidence"],
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def percentage(passed: int, total: int) -> int | None:
    """Return passed/total as a whole-number percentage, or None if total is 0.

    None (instead of 0) matters: "nobody was scored" is not the same thing as
    "everyone failed".
    """
    # round() gives a whole number, so 5 of 6 (83.33...) becomes 83.
    return round(100 * passed / total) if total else None


def summarise(results: list[dict]) -> dict:
    """Count, for each kind of check, how many patients passed out of how many
    were scored for it, and add the percentages."""
    totals = {}  # check name -> {"passed", "total", "percent"}
    for kind in CHECK_TYPES:
        # Only patients that actually have this check are counted in "total".
        scored = [r["checks"][kind]["ok"] for r in results if kind in r["checks"]]
        # sum() of booleans counts the True values (True is 1, False is 0).
        passed = sum(scored)
        totals[kind] = {"passed": passed, "total": len(scored),
                        "percent": percentage(passed, len(scored))}
    # Patients for which EVERY check passed.
    fully = sum(r["passed"] for r in results)
    return {
        "patients_fully_correct": fully,
        "patients": len(results),
        # Share of patients whose checks ALL passed (the headline number).
        "percent": percentage(fully, len(results)),
        "by_check": totals,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the whole evaluation. Returns 0 on success, 1 if it cannot run."""
    # The first line of the file's docstring is used as the --help description.
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # Optional flag: "--only E5" evaluates just that patient.
    parser.add_argument("--only", help="evaluate a single patient id, for example E5")
    args = parser.parse_args(argv)  # read what the user typed

    # The same data sources the app uses, so the evaluation measures the real
    # system and not a copy of it.
    catalogue = load_catalogue()  # the list of medicines the app knows
    rule_data = load_rule_data(catalogue=catalogue)  # codes + cascade table
    stored = load_stored_answers()  # the model answers saved on 2026-11-23
    if stored is None:
        # Without the stored answers there is nothing to compare for interactions.
        print("data/interacciones_guardadas.json not found: interactions cannot be evaluated.")
        return 1  # non-zero exit code = something went wrong
    # A function that answers each "sheet + ingredient" question from the file.
    review = stored_review(stored)

    # With --only, keep just that patient; otherwise keep all of them.
    patients = [p for p in load_patients() if not args.only or p["id"] == args.only]
    # Evaluate each patient with the same data sources.
    results = [evaluate(p, catalogue, rule_data, review) for p in patients]
    summary = summarise(results)  # totals and percentages

    # Header row, then one row per patient. f"{x:13}" pads text to 13 characters
    # so the columns line up. "-" means the patient has no label for that check.
    print(f"{'Patient':8} {'Result':8} " + " ".join(f"{k:13}" for k in CHECK_TYPES) + " Not completed")
    for r in results:
        cells = []  # one cell per check type
        for kind in CHECK_TYPES:
            c = r["checks"].get(kind)  # None if this patient has no such check
            cells.append(f"{('ok' if c['ok'] else 'FAIL') if c else '-':13}")
        print(f"{r['patient']:8} {'PASS' if r['passed'] else 'FAIL':8} " + " ".join(cells)
              + f" {r['searches_not_completed']}")
    # The numbers that go on the Accuracy slide: count and percentage together.
    print(f"\nPatients fully correct: {summary['patients_fully_correct']} of "
          f"{summary['patients']} ({summary['percent']}%)")
    for kind, t in summary["by_check"].items():
        # Skip the percentage when nobody was scored, to avoid printing "None%".
        pct = f" ({t['percent']}%)" if t["total"] else ""
        print(f"  {kind:13} {t['passed']} of {t['total']}{pct}")

    # Save everything. The run date and the date of the stored answers make it
    # clear later which version of the system a result belongs to, which matters
    # because we will run this before and after the fixes.
    RESULTS_FILE.write_text(json.dumps({
        "run_date": date.today().isoformat(),  # e.g. "2026-11-23"
        "stored_answers_date": stored.generated_at,
        "summary": summary,
        "patients": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")  # keep accents readable
    print(f"\nSaved to {RESULTS_FILE}")
    return 0  # success


# Runs main() only when the file is executed directly, not when it is imported
# by another file (for example by a test).
if __name__ == "__main__":
    raise SystemExit(main())  # pass main()'s return value on as the exit code
