"""Test search filtering and returned results without downloads or API calls.

These tests use a small fake collection and embedding model. They check the
search code, not the relevance of the real E5 model's rankings.

Run from the project root:
    python -m pytest tests/test_busqueda.py -v
"""

from unittest.mock import Mock

import pytest

from src import busqueda


@pytest.fixture
def search_resources(monkeypatch):
    """Provide sources that distinguish medicines, sections, and subsections."""
    fragments = [
        {"id": "0123_4.5", "nregistro": "0123", "seccion": "4.5", "texto": "Interaction text"},
        {"id": "0123_4.5.1", "nregistro": "0123", "seccion": "4.5.1", "texto": "Subsection text"},
        {"id": "0123_4.50", "nregistro": "0123", "seccion": "4.50", "texto": "Different section"},
        {"id": "0123_4.3", "nregistro": "0123", "seccion": "4.3", "texto": "Contraindication text"},
        {"id": "999_4.5", "nregistro": "999", "seccion": "4.5", "texto": "Another medicine"},
    ]
    collection = Mock()
    model = Mock()
    model.encode.return_value.tolist.return_value = [[0.1, 0.2]]

    def get_candidates(include, where=None):
        # Simulate Chroma's medicine filter, keeping IDs and metadata aligned.
        selected = [
            fragment for fragment in fragments
            if where is None or fragment["nregistro"] == where["nregistro"]
        ]
        return {
            "ids": [fragment["id"] for fragment in selected],
            "metadatas": [
                {"nregistro": fragment["nregistro"], "seccion": fragment["seccion"]}
                for fragment in selected
            ],
        }

    def query_candidates(query_embeddings, ids, n_results, include):
        # Reverse candidate order to simulate a ranking supplied by Chroma.
        ranked = [fragment for fragment in reversed(fragments) if fragment["id"] in ids]
        ranked = ranked[:n_results]
        return {
            "ids": [[fragment["id"] for fragment in ranked]],
            "documents": [[fragment["texto"] for fragment in ranked]],
            "metadatas": [[
                {"nregistro": fragment["nregistro"], "seccion": fragment["seccion"]}
                for fragment in ranked
            ]],
            "distances": [[0.1 * (index + 1) for index in range(len(ranked))]],
        }

    collection.get.side_effect = get_candidates
    collection.query.side_effect = query_candidates
    monkeypatch.setattr(busqueda, "load_search_resources", Mock(return_value=(collection, model)))
    return collection, model


def test_medicine_and_section_filters(search_resources):
    """Include subsections, exclude other medicines and section 4.50."""
    collection, _ = search_resources

    matches = busqueda.search_fragments("Interactions", nregistro="0123", sections=["4.5"])

    assert [match["id"] for match in matches] == ["0123_4.5.1", "0123_4.5"]
    collection.get.assert_called_once_with(include=["metadatas"], where={"nregistro": "0123"})
    assert set(collection.query.call_args.kwargs["ids"]) == {"0123_4.5", "0123_4.5.1"}


def test_query_prefix_and_result_contents(search_resources):
    """E5 receives a query prefix; returned text, metadata and distance agree."""
    _, model = search_resources

    matches = busqueda.search_fragments("  Interactions  ", nregistro="0123", sections=["4.5"], k=1)

    assert model.encode.call_args.args[0] == ["query: Interactions"]
    assert matches == [{
        "id": "0123_4.5.1", "nregistro": "0123", "seccion": "4.5.1",
        "texto": "Subsection text", "distance": 0.1,
    }]


def test_k_is_limited_to_available_sources(search_resources):
    collection, _ = search_resources

    matches = busqueda.search_fragments("Contraindications", nregistro="0123", sections=["4.3"], k=10)

    assert len(matches) == 1
    assert collection.query.call_args.kwargs["n_results"] == 1


def test_unfiltered_search_can_return_other_medicines(search_resources):
    collection, _ = search_resources

    matches = busqueda.search_fragments("General question", k=10)

    assert len(matches) == 5
    assert {match["nregistro"] for match in matches} == {"0123", "999"}
    assert "where" not in collection.get.call_args.kwargs


@pytest.mark.parametrize("filters", [
    {"nregistro": "missing"},
    {"nregistro": "0123", "sections": ["9.9"]},
])
def test_no_candidates_skips_embedding_and_ranking(search_resources, filters):
    collection, model = search_resources

    assert busqueda.search_fragments("Question", **filters) == []
    model.encode.assert_not_called()
    collection.query.assert_not_called()


@pytest.mark.parametrize("arguments", [
    {"question": "   "},
    {"question": "Question", "k": 0},
    {"question": "Question", "k": -1},
    {"question": "Question", "sections": []},
])
def test_invalid_inputs_fail_before_loading_resources(monkeypatch, arguments):
    load = Mock()
    monkeypatch.setattr(busqueda, "load_search_resources", load)

    with pytest.raises(ValueError):
        busqueda.search_fragments(**arguments)

    load.assert_not_called()


def test_pair_search_uses_interaction_section(search_resources):
    collection, model = search_resources

    matches = busqueda.search_for_pair("0123", "  warfarina  ")

    assert model.encode.call_args.args[0] == ["query: interacciones de warfarina"]
    assert {match["seccion"] for match in matches} == {"4.5", "4.5.1"}
    assert collection.get.call_args.kwargs["where"] == {"nregistro": "0123"}


def test_pair_search_rejects_empty_ingredient(monkeypatch):
    load = Mock()
    monkeypatch.setattr(busqueda, "load_search_resources", load)

    with pytest.raises(ValueError):
        busqueda.search_for_pair("0123", "   ")

    load.assert_not_called()
