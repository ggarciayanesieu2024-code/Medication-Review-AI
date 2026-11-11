
# Person 4 - Shared Components

This document describes the shared configuration, data models, citation verification, and Groq client implemented for the Medication Review AI project.

## 1. Configuration and Data Models

`src/config.py` stores the shared settings, including file paths, the Groq model, token limits, and RAG search parameters.

`src/modelos.py` uses Pydantic to validate patients, medications, CIMA fragments, citations, findings, and prescribing cascades.

The `RespuestaRAG` model also validates the AI response structure. Responses marked as `found` must contain citations, while `not_found` responses must have an empty citation list.

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

The API key is loaded from a local `.env` file and is never committed to GitHub. The client uses the shared configuration, requests JSON output, and returns a parsed Python dictionary.

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

Person 3 can use `generar_json()` to call the model, `RespuestaRAG` to validate the response structure, and `verificar_cita()` to check the supporting quotations.

The final integration will be reviewed once the RAG pipeline is completed.