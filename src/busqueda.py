"""
Search the existing CIMA vector index for relevant source fragments.

What this script does:
1. Opens the ChromaDB database created by indice.py.
2. Loads the same E5 model used for indexing.
3. Optionally filters fragments by medicine and document sections.
4. Embeds the question and retrieves the closest matching fragments.
5. Returns their original text, source metadata, and cosine distance.

Examples of supported searches:
- Interactions: section 4.5.
- Dosage: section 4.2.
- Contraindications and warnings: sections 4.3 and 4.4.
- General questions: all indexed sections.

Selecting a section also includes its subsections.

This script retrieves evidence. It does not generate answers or establish
whether a combination is safe.

Run from the project root:
    python -m src.busqueda
"""

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

# Reuse the same paths, model, and collection as the indexing script.
from src.indice import (
    COLLECTION_NAME,
    DATABASE_DIR,
    MODEL_CACHE_DIR,
    MODEL_NAME,
)


@lru_cache(maxsize=1)
def load_search_resources():
    """Load the database collection and E5 once per Python process."""
    if not DATABASE_DIR.exists():
        raise FileNotFoundError("Database folder missing. Run src/indice.py first.")

    client = chromadb.PersistentClient(path=str(DATABASE_DIR))
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(MODEL_CACHE_DIR),
    )

    return collection, model


def search_fragments(
    question: str,
    nregistro: str | None = None,
    sections: list[str] | None = None,
    k: int = 3,
) -> list:
    """Retrieve up to k fragments, with optional medicine and section filters."""
    question = question.strip()

    if not question:
        raise ValueError("The question cannot be empty.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if sections == []:
        raise ValueError("Provide section numbers or use None for all sections.")

    collection, model = load_search_resources()

    # 1. Read candidate IDs and metadata, without loading their full text.
    # If no medicine is specified, consider all medicines in the collection.
    filters = {}
    if nregistro is not None:
        filters["where"] = {"nregistro": nregistro}

    candidates = collection.get(
        include=["metadatas"],
        **filters,
    )

    # 2. Keep the requested sections and their subsections.
    # For example, "4.5" includes both "4.5" and "4.5.1".
    eligible_ids = []
    for fragment_id, metadata in zip(
        candidates["ids"],
        candidates["metadatas"],
    ):
        section = metadata["seccion"]

        if sections is None or any(
            section == requested or section.startswith(requested + ".")
            for requested in sections
        ):
            eligible_ids.append(fragment_id)

    if not eligible_ids:
        return []

    # 3. Turn the question into a vector using E5's query prefix.
    query_embedding = model.encode(
        [f"query: {question}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # 4. Rank only the eligible fragments by vector similarity.
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        ids=eligible_ids,
        n_results=min(k, len(eligible_ids)),
        include=["documents", "metadatas", "distances"],
    )

# FOR FUTURE: I believe we talked about using a more complicated ranking function.

    # 5. Return one dictionary per result.
    # Chroma groups results by query; [0] selects our single query.
    matches = []
    for fragment_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "id": fragment_id,
            **metadata,
            "texto": text,
            "distance": distance,
        })

    return matches


def search_for_pair(nregistro_x: str, ingredient_y: str, k: int = 3) -> list:
    """Search medicine x's interaction sections for ingredient y."""
    ingredient_y = ingredient_y.strip()
    if not ingredient_y:
        raise ValueError("The other medicine's ingredient cannot be empty.")

    return search_fragments(
        question=f"interacciones de {ingredient_y}",
        nregistro=nregistro_x,
        sections=["4.5"],
        k=k,
    )