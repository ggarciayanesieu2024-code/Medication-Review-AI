"""
Parse Prescripcion.xml from the AEMPS nomenclator into a clean JSON file.

What this script does:
1. Reads every <prescription> record in data/raw/nomenclator/Prescripcion.xml.
2. Keeps only the fields the project needs, and gives them English names.
3. Leaves out the official recommendation texts, on purpose (see below).
4. Writes the result to data/processed/prescriptions.json.

Design note: the nomenclator contains official recommendation texts such as
"Suspender el principio(s) activo(s)...". Our project rule is that the system
never recommends stopping a medicine, so this script never reads those fields.

Namespace note: every XML tag is prefixed with a namespace, for example
{http://schemas.aemps.es/prescripcion/aemps_prescripcion}des_nomco.
Searching for the plain name "des_nomco" would find nothing. The helper
functions below build the full name for us.

Run from the project root:
    python src/parse_nomenclator.py
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path


# Namespace declared at the top of Prescripcion.xml.
NAMESPACE = "http://schemas.aemps.es/prescripcion/aemps_prescripcion"

INPUT_FILE = Path("data") / "raw" / "nomenclator" / "Prescripcion.xml"
OUTPUT_FILE = Path("data") / "processed" / "prescriptions.json"


def tag(name: str) -> str:
    """Return the full, namespaced name of a tag.

    For example, tag("des_nomco") gives
    "{http://schemas.aemps.es/prescripcion/aemps_prescripcion}des_nomco".
    """
    return f"{{{NAMESPACE}}}{name}"


def find_all(element, *names):
    """Return every element that matches a path of tag names.

    Example: find_all(record, "atc", "duplicidades") first finds every <atc>
    child of the record, then every <duplicidades> child of each of those.

    The path is walked one level at a time. At each level, we collect all the
    matching children, so a path can return several elements.
    """
    current_level = [element]
    for name in names:
        next_level = []
        for node in current_level:
            next_level.extend(node.findall(tag(name)))
        current_level = next_level
    return current_level


def find_text(element, *names):
    """Return the text of the first element matching the path, or None.

    We strip spaces from the text. If the tag exists but is empty, we return
    None instead of an empty string, so that later code can check for missing
    values in one place.
    """
    matches = find_all(element, *names)
    if not matches or matches[0].text is None:
        return None
    text = matches[0].text.strip()
    return text or None


def parse_prescription(record) -> dict:
    """Convert one <prescription> element into a plain dictionary.

    The keys use the official AEMPS field names, so every value can be checked
    against the AEMPS technical description of the nomenclator. The code that
    builds the dictionary, and its comments, are in English.
    """
    composiciones_pa = []
    for composition in find_all(record, "formasfarmaceuticas", "composicion_pa"):
        composiciones_pa.append({
            "cod_principio_activo": find_text(composition, "cod_principio_activo"),
            "dosis_pa": find_text(composition, "dosis_pa"),
            "unidad_dosis_pa": find_text(composition, "unidad_dosis_pa"),
        })

    viasadministracion = [
        find_text(route, "cod_via_admin")
        for route in find_all(record, "formasfarmaceuticas", "viasadministracion")
    ]

    # Each duplicidades group compares this medicine against others in the same
    # ATC group. We keep the group and the description, but not the official
    # recomendacion_duplicidad text (see the design note at the top of the file).
    duplicidades = []
    for group in find_all(record, "atc", "duplicidades"):
        duplicidades.append({
            "atc_duplicidad": find_text(group, "atc_duplicidad"),
            "descripcion_atc_duplicidad": find_text(group, "descripcion_atc_duplicidad"),
            "efecto_duplicidad": find_text(group, "efecto_duplicidad"),
        })

    return {
        "cod_nacion": find_text(record, "cod_nacion"),
        # nro_definitivo is the CIMA registration number. It links this medicine
        # to its official technical sheet.
        "nro_definitivo": find_text(record, "nro_definitivo"),
        "des_nomco": find_text(record, "des_nomco"),
        "des_prese": find_text(record, "des_prese"),
        "cod_atc": find_text(record, "atc", "cod_atc"),
        "composicion_pa": composiciones_pa,
        "viasadministracion": viasadministracion,
        "duplicidades": duplicidades,
        "url_fictec": find_text(record, "url_fictec"),
    }


def main() -> None:
    tree = ET.parse(INPUT_FILE)
    root = tree.getroot()

    # The date in the header tells us how fresh the data is. We store it so
    # the report can say which day of the nomenclator it comes from.
    source_date = find_text(root, "header", "listprescriptiondate")

    records = find_all(root, "prescription")
    prescriptions = [parse_prescription(record) for record in records]

    output = {"listprescriptiondate": source_date, "prescripciones": prescriptions}

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, ensure_ascii=False, indent=2)

    print(f"Parsed {len(prescriptions)} prescriptions (source date: {source_date})")
    print(f"Saved to {OUTPUT_FILE}")

    # A quick check that the data makes sense: find medicines named paracetamol.
    matches = [p for p in prescriptions if p["des_nomco"] and "PARACETAMOL" in p["des_nomco"].upper()]
    print(f"Medicines whose name contains PARACETAMOL: {len(matches)}")


if __name__ == "__main__":
    main()
