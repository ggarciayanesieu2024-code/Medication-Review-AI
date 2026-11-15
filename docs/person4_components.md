
# Person 4 - Shared Components

This document describes the shared configuration, data models, citation verification, and Groq client implemented for the Medication Review AI project.

## 1. Configuration and Data Models

`src/config.py` stores the shared settings, including file paths, the Groq model, token limits, and RAG search parameters.

All implemented Python components now read their shared paths and settings from
this file. Paths are resolved from the project root, including the existing
database at `data/embeddings/chroma_db` and model cache at `data/embeddings/E5_model`.
The data preparation scripts support both direct execution and `python -m src.<name>`.
Their download URLs, timeouts, and other script-specific settings stay in those
scripts; shared file paths remain in `config.py`.
Raw nomenclator files are only needed to regenerate the processed catalogue;
they are not required to search an existing index.

Current LLM defaults are temperature 0, `top_p=1`, and a completion limit of
4,096 tokens. GPT-OSS requests use medium reasoning effort. The client timeout
is 60 seconds per request, with up to two SDK retries for eligible API/network
errors. These retries do not validate or repair JSON responses. Settings are
starting values and have not been tuned through evaluation.

Embedding settings are shared by indexing and search: multilingual-e5-small,
CPU, normalised embeddings, indexing batch size 4, and cosine distance.
Retrieval defaults to three results. Changing the embedding model or vector
settings requires rebuilding the index before searching with the new settings.

`src/modelos.py` uses Pydantic to validate patients, medications, CIMA fragments, citations, findings, and prescribing cascades.

The `StructuredOutputRAG` model also validates the AI response structure. Responses marked as `found` must contain citations, while `not_found` responses must have an empty citation list.

## 2. Citation Verification

`src/verificacion.py` checks whether an AI-generated quotation exists in the referenced CIMA fragment.

A citation is accepted only when:
- The medication registration number matches.
- The document section matches.
- The quotation contains at least six words and appears in the source text.

The comparison ignores capitalization and repeated whitespace. Invalid quotations are rejected.

This prevents accepting fabricated or incorrectly attributed quotations, but does not independently confirm the medical interpretation.

## 3. Groq API Client

`src/llm.py` connects to the language model through the Groq API.

The API key is loaded from a local `.env` file and is never committed to GitHub.
`llm_medication_review(messages)` uses the shared configuration, requests JSON
output, and validates the returned text with `StructuredOutputRAG.model_validate_json()`.
It returns a `StructuredOutputRAG` object, with attributes such as `status`,
`description`, and `citations`. Use `.model_dump()` when a dictionary is needed.
Invalid JSON or schema violations raise a Pydantic `ValidationError`; empty or
token-truncated responses are rejected with `ValueError`.
This is local validation after generation: JSON mode does not enforce the schema
on the server. Citation-content checks and forbidden-phrase checks remain part
of the pending RAG integration.

The current model is `openai/gpt-oss-120b`.

## 4. Testing

The following checks were completed:

- 4 citation-verification tests passed.
- 6 RAG response-validation tests passed.
- All 10 automated tests passed together.
- A real Groq API request successfully returned valid JSON.
- The Groq client accepts the message format prepared by `rag.py`.

Full RAG integration and end-to-end testing are still pending.

## 5. Integration

The shared components are submitted in PR #11.

Person 3 can use `llm_medication_review()` to call the model and receive a
validated `StructuredOutputRAG`, then `verificar_cita()` to check the supporting
quotations against the retrieved fragments.

The final integration will be reviewed once the RAG pipeline is completed.
