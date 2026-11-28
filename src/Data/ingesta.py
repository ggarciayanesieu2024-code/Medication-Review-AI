"""Prepare missing notebook data using the existing download and index scripts.

Valid local data is reused. CIMA downloads only fill missing medicines; a
failed or partial download never replaces an existing medicine's fragments.
Optional remote update checks report changes without refreshing the dataset.
"""

import json
from datetime import datetime

from src.config import (
    CASCADES_FILE, COLLECTION_NAME, DATABASE_DIR, FRAGMENTS_FILE,
    NOMENCLATOR_DIR, NOMENCLATOR_XML_FILE, NOMENCLATOR_ZIP_FILE, PRESCRIPTIONS_FILE,
)
from src.Rag.verificacion import Fragment


def load_fragments() -> list[dict]:
    """Read and validate local fragments; malformed files need explicit repair."""
    if not FRAGMENTS_FILE.exists():
        return []
    fragments = json.loads(FRAGMENTS_FILE.read_text(encoding="utf-8"))
    if not isinstance(fragments, list):
        raise ValueError("The CIMA fragments file must contain a list.")
    for fragment in fragments:
        Fragment.model_validate(fragment)
    ids = [(item["nregistro"], item["seccion"]) for item in fragments]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate medicine/section IDs in the CIMA file.")
    return fragments


def ensure_nomenclator() -> str:
    """Reuse parsed prescriptions, or prepare them from local XML/ZIP or AEMPS."""
    from src.Cascades.reglas import load_drug_info
    from src.Data import download_nomenclator, parse_nomenclator

    if PRESCRIPTIONS_FILE.exists():
        if not load_drug_info(PRESCRIPTIONS_FILE):
            raise ValueError("The local nomenclator has no usable medicine records.")
        return "Reused validated nomenclator."
    if not NOMENCLATOR_XML_FILE.exists():
        if not NOMENCLATOR_ZIP_FILE.exists():
            download_nomenclator.download_zip(
                download_nomenclator.NOMENCLATOR_URL, NOMENCLATOR_ZIP_FILE,
            )
        download_nomenclator.unpack_zip(NOMENCLATOR_ZIP_FILE, NOMENCLATOR_DIR)
    parse_nomenclator.main()
    if not load_drug_info(PRESCRIPTIONS_FILE):
        raise ValueError("The prepared nomenclator has no usable medicine records.")
    return "Prepared missing nomenclator."


def ensure_fragments(registrations: list[str]) -> tuple[list[dict], list[str]]:
    """Fill missing medicine coverage, retaining all existing source records."""
    from src.Data.parse_cima import chunks_for_medicine

    fragments = load_fragments()
    covered = {item["nregistro"] for item in fragments}
    messages = []
    added = []
    for registration in sorted(set(registrations) - covered):
        errors = []
        chunks = chunks_for_medicine(registration, save_raw=False, errors=errors)
        if errors or not chunks:
            messages.append(f"{registration}: download incomplete/unavailable; no fragments added.")
            continue
        for chunk in chunks:
            Fragment.model_validate(chunk)
            if chunk["nregistro"] != registration:
                raise ValueError(f"Downloaded fragments do not belong to {registration}.")
        if len({item["seccion"] for item in chunks}) != len(chunks):
            raise ValueError(f"Duplicate downloaded sections for {registration}.")
        added.extend(chunks)
        messages.append(f"{registration}: downloaded {len(chunks)} fragments.")
    if added:
        # Replace the JSON only after the complete new content is written.
        FRAGMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = FRAGMENTS_FILE.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(fragments + added, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(FRAGMENTS_FILE)
        finally:
            temporary.unlink(missing_ok=True)
        fragments += added
    return fragments, messages or ["Reused validated CIMA fragments; nothing to download."]


def ensure_index(fragments: list[dict], *, rebuild: bool = False) -> str:
    """Compare stored IDs, text and metadata before deciding to rebuild."""
    if not fragments:
        return "Index not prepared: no CIMA fragments available."
    import chromadb
    from src.Rag import busqueda, indice

    expected = {f"{item['nregistro']}_{item['seccion']}": item for item in fragments}
    matches = False
    if (DATABASE_DIR / "chroma.sqlite3").exists():
        client = chromadb.PersistentClient(path=str(DATABASE_DIR))
        if COLLECTION_NAME in [collection.name for collection in client.list_collections()]:
            collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
            stored = collection.get(include=["documents", "metadatas"])
            matches = set(stored["ids"]) == set(expected) and all(
                document == expected[key]["texto"] and all(
                    metadata.get(field) == expected[key].get(field)
                    for field in ("nregistro", "nombre", "seccion", "titulo")
                )
                for key, document, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"])
            )
    if matches and not rebuild:
        return f"Reused index: {len(expected)} matching fragments."
    indice.main()
    busqueda.load_search_resources.cache_clear()
    reason = "explicit rebuild" if rebuild else "missing or different local contents"
    return f"Built index from {len(expected)} fragments ({reason})."


def check_cima_updates(registrations: list[str], since: str) -> list[dict]:
    """Report CIMA changes since an explicit dd/mm/yyyy date; do not download.

    An old file's modification time is not a reliable source-version date.
    The caller must choose the comparison date. Failed checks remain unknown.
    """
    from src.Data.parse_cima import FetchError, fetch_json

    datetime.strptime(since, "%d/%m/%Y")
    results = []
    for registration in sorted(set(registrations)):
        try:
            changes = []
            page = 1
            while True:
                payload = fetch_json("registroCambios", {
                    "fecha": since, "nregistro": registration, "pagina": page,
                }, None, False)
                if isinstance(payload, list):  # Older responses were unpaginated.
                    changes.extend(payload)
                    break
                if (not isinstance(payload, dict) or not isinstance(payload.get("resultados"), list)
                        or not isinstance(payload.get("totalFilas"), int)
                        or payload.get("pagina") != page):
                    raise ValueError("Unexpected update response; completeness unknown.")
                changes.extend(payload["resultados"])
                if len(changes) >= payload["totalFilas"]:
                    break
                if not payload["resultados"]:
                    raise ValueError("Incomplete update response; freshness unknown.")
                page += 1
            results.append({"nregistro": registration, "changes": changes, "check": "completed"})
        except (FetchError, ValueError) as exc:
            results.append({"nregistro": registration, "check": "failed", "reason": str(exc)})
    return results


def prepare_notebook_data(
    registrations: list[str], *, check_updates_since: str | None = None,
    rebuild_index: bool = False,
) -> dict:
    """Ensure local rules, source coverage and index; optionally report changes."""
    from src.Cascades.reglas import load_cascades

    if check_updates_since is not None:
        datetime.strptime(check_updates_since, "%d/%m/%Y")
    # Cascades are a team-curated input, not an official download.
    if not CASCADES_FILE.exists():
        raise FileNotFoundError("Restore data/cascadas.json from the repository; it is not downloadable from CIMA.")
    load_cascades(CASCADES_FILE)
    nomenclator = ensure_nomenclator()
    fragments, messages = ensure_fragments(registrations)
    report = {
        "nomenclator": nomenclator, "CIMA": messages,
        "index": ensure_index(fragments, rebuild=rebuild_index),
        "still_missing": sorted(set(registrations) - {item["nregistro"] for item in fragments}),
    }
    if check_updates_since is not None:
        report["changes_since"] = check_updates_since
        report["remote_updates"] = check_cima_updates(registrations, check_updates_since)
    return report
