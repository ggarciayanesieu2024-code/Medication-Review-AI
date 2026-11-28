# FOR FUTURE: CHATGPT mentioned something about how some sections larger than the
# models input limit; definitely check out if true


"""
Build the local search index from the cleaned CIMA fragments.

What this script does:
1. Reads data/processed/fragmentos_cima.json.
2. Prepares each fragment with the "passage: " prefix required by E5.
3. Uses multilingual-e5-small to convert each passage into an embedding.
4. Rebuilds the cima_fragments collection in the local ChromaDB database.
5. Stores each embedding alongside its original text and source metadata.

The original section text is preserved for retrieval and citation checking.
Long passages are truncated by the embedding model to its input limit;
the complete text is still stored in ChromaDB.

This script does not detect interactions or call the language model.

Run from the project root:
    python -m src.Rag.indice
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

# Run from the project root with python -m src.Rag.indice.
from src.config import (
    COLLECTION_NAME,
    DATABASE_DIR,
    DISTANCE_METRIC,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    FRAGMENTS_FILE,
    MODEL_CACHE_DIR,
    NORMALIZE_EMBEDDINGS,
)


def main() -> None:
    # 1. Read the JSON file created by Lucia's CIMA parser.
    with open(FRAGMENTS_FILE, encoding="utf-8") as input_file:
        fragments = json.load(input_file)

    if not fragments:
        raise ValueError("The CIMA fragments file is empty.")

    print(f"Loaded {len(fragments)} fragments.")

    ids = []
    passages = []
    documents = []
    metadatas = []

    # 2. Prepare the four matching lists that ChromaDB needs.
    # The same position in every list refers to the same fragment.
    for fragment in fragments:
        ids.append(f"{fragment['nregistro']}_{fragment['seccion']}")

        # E5 embedding model uses this prefix to recognise text being indexed.
        passages.append(
            f"passage: {fragment['titulo']}. {fragment['texto']}"
        )

        # Preserve the original text without adding the E5 prefix.
        documents.append(fragment["texto"])

        # Metadata lets us filter searches and identify citation sources.
        metadatas.append({
            "nregistro": fragment["nregistro"],
            "nombre": fragment["nombre"],
            "seccion": fragment["seccion"],
            "titulo": fragment["titulo"],
        })

    # Duplicate IDs would mean two fragments claim to be the same section.
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate medicine-section IDs found.")

    # 3. Load the embedding model.
    # The first run downloads it; later runs reuse the local cached files.
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=EMBEDDING_DEVICE,
        cache_folder=str(MODEL_CACHE_DIR),
    )

    # An embedding is a numerical representation of a passage's meaning.
    # Normalisation puts the vectors on a common scale for comparison.
    embeddings = model.encode(
        passages,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        show_progress_bar=True,
    )

    # 4. Open a database that saves its contents to disk.
    client = chromadb.PersistentClient(path=str(DATABASE_DIR))

    # Rebuild only our collection, as required by the project plan.
    # Do this after embedding succeeds so a download failure keeps the old index.
    existing_names = [collection.name for collection in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,  # We provide our own E5 embeddings.
        configuration={"hnsw": {"space": DISTANCE_METRIC}},
    )

    # 5. Store the vectors, original texts, and identifying metadata together.
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} fragments in '{COLLECTION_NAME}'.")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Database saved to: {DATABASE_DIR}")


# Run main() when executing this file, but not when another file imports it.
if __name__ == "__main__":
    main()
