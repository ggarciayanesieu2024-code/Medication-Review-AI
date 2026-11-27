"""Check conditional preparation without downloads or rebuilding a real index."""

import json
import sys
from types import SimpleNamespace

import pytest

from src.Data import ingesta, parse_cima


def fragment(registration="1", section="4.5", text="Original source text"):
    return {"nregistro": registration, "seccion": section, "texto": text,
            "nombre": "Medicine", "titulo": "Interactions"}


def test_present_fragments_skip_downloads(tmp_path, monkeypatch):
    path = tmp_path / "fragments.json"
    original = json.dumps([fragment()])
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(ingesta, "FRAGMENTS_FILE", path)
    monkeypatch.setattr(parse_cima, "chunks_for_medicine", lambda *a, **kw: pytest.fail("Unexpected download"))
    rows, messages = ingesta.ensure_fragments(["1"])
    assert rows == [fragment()]
    assert path.read_text(encoding="utf-8") == original
    assert "nothing to download" in messages[0]


def test_missing_medicine_is_merged_and_failed_download_preserves_file(tmp_path, monkeypatch):
    path = tmp_path / "fragments.json"
    path.write_text(json.dumps([fragment()]), encoding="utf-8")
    monkeypatch.setattr(ingesta, "FRAGMENTS_FILE", path)
    calls = []

    def download(registration, save_raw, errors):
        calls.append(registration)
        if registration == "3":
            errors.append({"error": "missing section"})
        return [fragment(registration)]

    monkeypatch.setattr(parse_cima, "chunks_for_medicine", download)
    rows, messages = ingesta.ensure_fragments(["1", "2", "3"])
    assert calls == ["2", "3"]
    assert {row["nregistro"] for row in rows} == {"1", "2"}
    assert json.loads(path.read_text(encoding="utf-8")) == rows
    assert "incomplete" in messages[1]
    before = path.read_bytes()
    ingesta.ensure_fragments(["3"])
    assert path.read_bytes() == before


def test_invalid_file_is_not_silently_replaced(tmp_path, monkeypatch):
    path = tmp_path / "fragments.json"
    path.write_text("not JSON", encoding="utf-8")
    monkeypatch.setattr(ingesta, "FRAGMENTS_FILE", path)
    with pytest.raises(ValueError):
        ingesta.ensure_fragments(["1"])
    assert path.read_text(encoding="utf-8") == "not JSON"


def test_index_text_change_triggers_rebuild_but_matching_data_skips(tmp_path, monkeypatch):
    import src.Rag

    (tmp_path / "chroma.sqlite3").touch()
    monkeypatch.setattr(ingesta, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(ingesta, "COLLECTION_NAME", "test")
    item = fragment()
    stored = {"ids": ["1_4.5"], "documents": [item["texto"]], "metadatas": [item]}
    collection = SimpleNamespace(get=lambda **kwargs: stored)
    client = SimpleNamespace(list_collections=lambda: [SimpleNamespace(name="test")],
                             get_collection=lambda *args, **kwargs: collection)
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **kwargs: client))
    rebuilt = []
    monkeypatch.setattr(src.Rag, "indice", SimpleNamespace(main=lambda: rebuilt.append(True)), raising=False)
    monkeypatch.setattr(src.Rag, "busqueda", SimpleNamespace(
        load_search_resources=SimpleNamespace(cache_clear=lambda: None)), raising=False)
    assert "Reused" in ingesta.ensure_index([item])
    assert rebuilt == []
    assert "Built" in ingesta.ensure_index([item], rebuild=True)
    assert rebuilt == [True]
    rebuilt.clear()
    stored["documents"] = ["Different text"]
    assert "Built" in ingesta.ensure_index([item])
    assert rebuilt == [True]


def test_update_check_uses_official_endpoint_and_failures_are_unknown(monkeypatch):
    calls = []

    def fetch(path, params, *args):
        calls.append((path, params))
        if params["nregistro"] == "2":
            raise parse_cima.FetchError("Unavailable")
        return []

    monkeypatch.setattr(parse_cima, "fetch_json", fetch)
    results = ingesta.check_cima_updates(["1", "2"], "10/11/2026")
    assert calls[0] == ("registroCambios", {"fecha": "10/11/2026", "nregistro": "1", "pagina": 1})
    assert results[0]["check"] == "completed"
    assert results[1]["check"] == "failed"


def test_update_check_collects_all_pages(monkeypatch):
    def fetch(path, params, *args):
        page = params["pagina"]
        return {"totalFilas": 2, "pagina": page, "resultados": [{"change": page}]}

    monkeypatch.setattr(parse_cima, "fetch_json", fetch)
    result = ingesta.check_cima_updates(["1"], "10/11/2026")[0]
    assert result["check"] == "completed"
    assert result["changes"] == [{"change": 1}, {"change": 2}]
