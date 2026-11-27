"""
Download selected technical-sheet sections from the CIMA REST API and turn
them into clean text chunks.

What this script does, for each registration number (nregistro):
1. Asks CIMA for the medicine name (medicamento?nregistro=).
2. Asks CIMA for the list of sections in its technical sheet
   (docSegmentado/secciones/1?nregistro=).
3. For every section in WANTED_SECTIONS that the sheet has, asks for its text
   (docSegmentado/contenido/1?nregistro=&seccion=).
4. Keeps one chunk per returned element whose text is not empty, labelled with
   its exact section number (for example 4.2 or 4.2.2).
5. Writes all chunks to data/processed/fragmentos_cima.json.

Why one chunk per element: verified against real responses (see
docs/decisions.md, section 7). Asking for a section can return the section
and its subsections as separate elements. Sometimes the main section is empty
and the text lives only in a subsection, so each citation must point to the
element where the text really is.

Error handling: every call is retried once. If it fails again, the error is
logged and the script carries on with the rest.

Raw responses are only saved, unmodified, to data/raw/cima/ when --guardar-raw
is given. By default nothing is written there.

Run from the project root:
    python -m src.Data.parse_cima 66337 77758
    python -m src.Data.parse_cima 66337 77758 --guardar-raw
"""

import argparse
import html
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


from src.config import CIMA_RAW_DIR, FRAGMENTS_FILE


# Settings used only when downloading and parsing CIMA sections.
CIMA_BASE_URL = "https://cima.aemps.es/cima/rest"
WANTED_SECTIONS = {"4.2", "4.3", "4.4", "4.5", "4.8"}
TECHNICAL_SHEET = 1  # CIMA document type for the technical sheet.
CIMA_TIMEOUT_SECONDS = 30
CIMA_RETRY_DELAY_SECONDS = 2

logger = logging.getLogger("parse_cima")


class FetchError(Exception):
    """Raised when a CIMA call fails twice in a row."""


class NotFoundError(FetchError):
    """Raised when CIMA says the requested item does not exist.

    CIMA does not use HTTP 404 for this. Verified against real responses:
    - medicamento?nregistro= answers HTTP 204 with an empty body;
    - docSegmentado/secciones and docSegmentado/contenido answer HTTP 200 with
      {"error": "No existen secciones para el medicamento indicado"}.
    Asking again would give the same answer, so this error is never retried.
    """


def fetch_json(path: str, params: dict, raw_name: str | None, save_raw: bool):
    """Call one CIMA endpoint and return the decoded JSON.

    Network errors and invalid JSON are retried once. A "not found" answer
    raises NotFoundError straight away, without retrying. The raw bytes are
    written to disk exactly as received, before decoding, so the saved file is
    never modified.
    """
    url = f"{CIMA_BASE_URL}/{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})

    last_error = None
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(request, timeout=CIMA_TIMEOUT_SECONDS) as response:
                status = response.status
                body = response.read()
            # NotFoundError is not caught by the except below, so it is not
            # retried.
            if status == 204 or not body.strip():
                raise NotFoundError(f"{url}: not found (HTTP {status}, empty body)")
            data = json.loads(body.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            # URLError covers HTTP errors and network errors; ValueError covers
            # a body that is not valid JSON.
            last_error = error
            logger.warning("Attempt %d failed for %s: %s", attempt, url, error)
            if attempt == 1:
                time.sleep(CIMA_RETRY_DELAY_SECONDS)
            continue

        if save_raw and raw_name:
            CIMA_RAW_DIR.mkdir(parents=True, exist_ok=True)
            (CIMA_RAW_DIR / raw_name).write_bytes(body)

        if isinstance(data, dict) and "error" in data:
            raise NotFoundError(f"{url}: not found ({data['error']})")
        return data

    raise FetchError(f"{url}: {last_error}")


def clean_html(content: str) -> str:
    """Turn CIMA section HTML into plain text, keeping paragraph breaks.

    Steps, in this order:
    1. Turn </p> and <br> into line breaks, so paragraphs stay separate.
    2. Remove every remaining tag.
    3. Decode HTML entities (for example &#xa0; or &amp;).
    4. Tidy spaces: collapse runs of spaces inside each line and drop
       empty lines.

    Without step 1, a heading and the paragraph after it would be glued
    together ("Anticoagulantes orales Los anticoagulantes..."), and a quoted
    citation might no longer match the text.
    """
    text = re.sub(r"</p\s*>", "\n", content, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    # Non-breaking spaces become normal spaces, so text matching is simpler.
    text = text.replace("\xa0", " ")

    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def section_file_label(section: str) -> str:
    """Return a file-name-safe label for a section number, e.g. 4.5 -> 4_5."""
    return section.replace(".", "_")


def chunks_for_medicine(nregistro: str, save_raw: bool, errors: list) -> list:
    """Download and clean the wanted sections of one technical sheet.

    Errors are appended to `errors` instead of being raised, so that one
    failing call does not stop the rest.
    """
    try:
        medicine = fetch_json(
            "medicamento", {"nregistro": nregistro},
            f"medicamento_{nregistro}.json", save_raw,
        )
        nombre = medicine.get("nombre")
    except FetchError as error:
        # Without the name we can still build chunks; nombre stays empty.
        logger.error("Could not get the name of %s: %s", nregistro, error)
        errors.append({"nregistro": nregistro, "paso": "medicamento", "error": str(error)})
        nombre = None

    try:
        section_list = fetch_json(
            f"docSegmentado/secciones/{TECHNICAL_SHEET}", {"nregistro": nregistro},
            f"secciones_{nregistro}.json", save_raw,
        )
    except FetchError as error:
        logger.error("Could not get the section list of %s: %s", nregistro, error)
        errors.append({"nregistro": nregistro, "paso": "secciones", "error": str(error)})
        return []

    if not isinstance(section_list, list):
        message = f"section list is a {type(section_list).__name__}, expected a list"
        logger.error("Skipping %s: %s", nregistro, message)
        errors.append({"nregistro": nregistro, "paso": "secciones", "error": message})
        return []

    sections_to_fetch = [
        item["seccion"] for item in section_list if item.get("seccion") in WANTED_SECTIONS
    ]

    chunks = []
    seen_sections = set()
    for section in sections_to_fetch:
        try:
            elements = fetch_json(
                f"docSegmentado/contenido/{TECHNICAL_SHEET}",
                {"nregistro": nregistro, "seccion": section},
                f"seccion_{section_file_label(section)}_{nregistro}.json", save_raw,
            )
        except FetchError as error:
            logger.error("Could not get section %s of %s: %s", section, nregistro, error)
            errors.append({"nregistro": nregistro, "paso": f"seccion {section}", "error": str(error)})
            continue

        if not isinstance(elements, list):
            # Stop with this medicine, but keep the chunks already built.
            message = f"section {section} content is a {type(elements).__name__}, expected a list"
            logger.error("Skipping the rest of %s: %s", nregistro, message)
            errors.append({"nregistro": nregistro, "paso": f"seccion {section}", "error": message})
            return chunks

        # One request can return the section plus its subsections. Each one
        # with text becomes its own chunk; empty elements are discarded.
        for element in elements:
            numero = element.get("seccion")
            if numero in seen_sections:
                continue
            texto = clean_html(element.get("contenido") or "")
            if not texto:
                continue
            seen_sections.add(numero)
            chunks.append({
                "nregistro": nregistro,
                "nombre": nombre,
                "seccion": numero,
                "titulo": element.get("titulo"),
                "texto": texto,
            })

    return chunks


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("nregistros", nargs="+", help="CIMA registration numbers")
    parser.add_argument(
        "--guardar-raw", action="store_true",
        help="also save the unmodified CIMA responses to data/raw/cima/",
    )
    parser.add_argument(
        "--salida", type=Path, default=FRAGMENTS_FILE,
        help=f"where to write the chunks JSON (default: {FRAGMENTS_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_arguments()

    all_chunks = []
    errors = []
    for nregistro in args.nregistros:
        try:
            chunks = chunks_for_medicine(nregistro, args.guardar_raw, errors)
        except Exception as error:
            # Safety net: an unexpected error in one medicine must not stop
            # the others. It is logged with its traceback and kept in the
            # final list of failures, never silenced.
            logger.exception("Unexpected error while processing %s", nregistro)
            errors.append({
                "nregistro": nregistro,
                "paso": "inesperado",
                "error": f"{type(error).__name__}: {error}",
            })
            continue
        sections = ", ".join(chunk["seccion"] for chunk in chunks)
        logger.info("%s: %d chunks (%s)", nregistro, len(chunks), sections)
        all_chunks.extend(chunks)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as output_file:
        json.dump(all_chunks, output_file, ensure_ascii=False, indent=2)

    logger.info("Saved %d chunks to %s", len(all_chunks), args.salida)
    if errors:
        logger.warning("%d calls failed:", len(errors))
        for error in errors:
            logger.warning("  %s", error)
        # A non-zero exit code lets other scripts or tools know that
        # something failed, even though the output file was written.
        sys.exit(1)


if __name__ == "__main__":
    main()
