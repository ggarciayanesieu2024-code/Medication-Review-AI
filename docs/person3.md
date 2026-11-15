# Person 3 - Working Notes

Changes made by Gui (Person 3), kept here for review and later documentation cleanup.

Last updated: 15 November 2026.

## Current status and completed work

Indexing, retrieval, prompt preparation, and generation/validation are connected.
All 41 automated tests passed, and a fresh index build passed in an isolated
temporary database. Gui subsequently ran the real Groq interaction checks for
development patients A, B, and C. His live results exposed relevance and
treatment-wording failures that are not yet fixed. Implementation is in place,
but correct patient findings have not been fully validated.

Person 3 owns `src/indice.py`, `src/busqueda.py`, `src/rag.py`, their tests,
and documentation of indexing, retrieval, and RAG. Recent work also aligned
shared configuration, models, the LLM client, and cascade data for integration.
This does not transfer ownership of rules, interface, reports, or final
evaluation to Person 3.

### Shared configuration and model changes

- Implemented consumers obtain shared paths from `src/config.py`. Settings
  used only by one data preparation script remain in that script.
- `download_nomenclator.py` was restored to its original Git version at Gui's
  request, including local relative paths. It is an exception to path
  centralisation and should be run from the project root.
- Database: `data/embeddings/chroma_db`. E5 cache: `data/embeddings/E5_model`.
- Shared classes now use English names: `Fragment`, `Citation`, `Finding`,
  `Medication`, `Patient`, `CascadeCase`, and `StructuredOutputRAG`. Spanish
  JSON field names are retained. Cascade helper models are described below.
- `llm_medication_review(messages)` replaces `generar_json()` and returns a
  validated `StructuredOutputRAG`. Empty responses, missing choices, and
  responses cut off by the token limit are rejected.
- Groq defaults: `openai/gpt-oss-120b` (overridable by `GROQ_MODEL`), temperature
  0, `top_p=1`, completion limit 4,096, medium reasoning effort for GPT-OSS,
  request timeout 60 seconds, and two SDK retries for eligible API failures.
  These are starting settings, not values selected through evaluation.
- `tests/test_llm.py` was removed at Gui's request. RAG tests exercise the real
  client parsing and validation through mocked API responses.

## Indexing and retrieval

`src/indice.py` reads `data/processed/fragmentos_cima.json`. Each source section
or subsection remains one indexed fragment with ID `nregistro_seccion`.
It embeds `passage: ` plus title and text with
`intfloat/multilingual-e5-small`, on CPU, batch size 4, using normalised
384-dimensional vectors. Chroma uses cosine distance and collection
`cima_fragments`. Running the script rebuilds this collection. Complete
original text and metadata are stored for quotation checks.

`src/busqueda.py` caches the loaded model and collection per Python process.
`search_fragments()` embeds questions with the E5 `query: ` prefix, filters by
medicine registration and requested sections, and returns up to `k=3` results.
Section 4.5 includes 4.5.1 but not 4.50. `search_for_pair()` searches one
medicine's 4.5 sections for another ingredient. RAG preparation additionally
requires an ingredient mention before calling the LLM. Missing mentions are
not evidence that a combination is safe.

The current source file contains 99 fragments for 18 registration numbers.
SHA256: `0A3FEF99CDBB8C2882F6065CF5213E196C778731540B24F2DC6C9077A1355768`.
The catalogue contains 30,894 medicine-presentation records. Identifiers
remain strings to preserve leading zeroes.

## Connected RAG pipeline

`src/rag.py` now provides two complete entry functions:

```python
from src.rag import answer_question, review_interaction

result = answer_question(
    "Consulta sobre contraindicaciones",
    nregistro="65222",
    sections=["4.3"],
)
result = review_interaction("65222", "warfarina")
```

Each uses the existing preparation function to retrieve evidence and build the
appropriate prompt. `run_prepared_request()` then calls `llm_medication_review()`,
which validates JSON with `StructuredOutputRAG`. Invalid JSON or schema output
gets one fresh attempt with the same sources and a format reminder. This is
separate from SDK network retries. Empty or token-truncated responses are
rejected without this schema retry.

Before accepting a response, the pipeline checks its description and every
quotation against `FORBIDDEN_PHRASES`, ignoring case, accents, and repeated
whitespace. It then checks every citation with `verificar_cita()` against the
fragments actually sent to the LLM. Registration numbers and exact sections,
including subsections, must match. If any check fails, the entire answer is
rejected and its generated text is not returned.

Both entry functions return a dictionary with `status`, `question`, `fragments`,
`response`, and `reason`. `response` is a serialisable dictionary from the
validated model, or `None` for rejected output and generation errors.

| Status | Reason | Meaning |
| --- | --- | --- |
| `found` | `None` | Response passed schema, phrase, and citation checks |
| `not_found` | `no_evidence` | No fragments survived retrieval/filtering; no API call |
| `not_found` | `insufficient_evidence` | The model reported insufficient supplied evidence |
| `rejected` | `invalid_response` | Invalid JSON/schema after retry, or empty/truncated output |
| `rejected` | `forbidden_phrase` | Description or quotation contains a configured phrase |
| `rejected` | `invalid_citation` | At least one citation failed source verification |
| `error` | `generation_failed` | Groq/API failure or missing key |

Invalid search inputs, retrieval failures, and malformed source records still
raise exceptions rather than being labelled as absent evidence. Raw sources
are returned for traceability; they are not generated recommendations and may
contain treatment language from official technical sheets.

`tests/test_rag.py` checks the connected flow with fake retrieved sources and
mocked Groq responses, while using the real client parsing, Pydantic models,
and citation verifier. It makes no live API calls.

The initial integration check used the local index for `65222` and `warfarina`,
a mocked Groq response, and the real validation pipeline; the section 4.5
citation passed. This was a wiring check. Later automated, local-data, and
live checks are recorded below with their limitations.

Remaining limits: the phrase list is a simple filter, not a complete safety
classifier. Literal citation verification does not establish that the answer's
interpretation is correct; its existing substring check can also accept a quote
ending partway through a source word. Interaction review checks one medicine's
sheet per call and retains the ingredient-mention filter, which can miss class
names or synonyms. Patient-wide pair iteration, rules, and report integration
are not implemented by these entry functions.

## Unified cascade data

`data/cascadas.json` is now a single JSON array validated by `CascadeCase`.
The required fields are `id`, `medicamento_causante`, `efecto_adverso`,
`medicamento_cascada`, and `pregunta_medico`. `fuente` can be `null`, an empty
list, or a list of references with `cita` and optional `pmid`, `doi`, and `url`.
Missing sources are not replaced with invented references.

Both medicine fields use `CascadeMedication`:

```json
{
  "nombre": "metformina",
  "cod_principio_activo": ["4379"],
  "cod_atc": ["A10BA02"],
  "atc_excluidos": []
}
```

Codes remain strings. At least one ingredient code or ATC prefix is required.
Names are for display, not matching. The intended rules-engine behavior is:

- Within each code list, any listed code can match.
- An ingredient code must equal a code in the medicine's `composicion_pa`.
- ATC codes use prefix matching, so `C03` includes its subgroups.
- If both code lists are nonempty, both must match the same medicine.
- An excluded ATC prefix overrides a match on that side of the cascade.
- The causing and cascade medicine must be distinct patient medicines.

These matching rules are documented here but are not yet implemented in
`src/reglas.py`. Schema validation does not establish clinical validity.

Ale's C01-C13 retain their broad ATC selectors and existing reference details.
Empty ingredient lists for those class rules are intentional; adding one
example ingredient would narrow the rule. Exclusions now belong to the
relevant medicine dictionary: C05 and C11 on the causing side, C13 on the
cascade side.

Lucía's entries were migrated as follows:

| Original ID | Current location |
| --- | --- |
| C1: amlodipine → oedema → furosemide | Example under C01; keeps evaluation-case tag E2 |
| C3: ibuprofen → hypertension → amlodipine | Example under C07 |
| C2: ibuprofen → digestive bleeding → pantoprazole | C14 |
| C4: simvastatin → muscle symptoms → ibuprofen | C15 |
| C5: metformin → diarrhoea → loperamide | C16 |

The merged examples are descriptive metadata, not additional rules, avoiding
duplicate conceptual alerts. The three added entries have `fuente: null` and
retain their original hypothesis text. Their ingredient codes and exact ATC
codes were checked against local `prescriptions.json`, using registrations
88372, 74483, 65879, 55211, and 87970. Merged examples were also checked using
65461 and 64140. This verifies catalogue mappings, not the proposed cascades.

All entries are hypotheses for professional review. Original `verificado`
claims are preserved as contributor-attributed notes rather than an automated
verification flag. Existing dose limitations and other notes are retained.
C14 also explicitly notes that gastroprotection may be intentional and
appropriate. No new literature review was performed during this migration.

## Verification results - 15 November 2026

### Automated and local checks

| Check | Result and scope |
| --- | --- |
| Full pytest suite | 41 passed: 12 retrieval, 14 RAG, 11 model/data, and 4 citation tests |
| Syntax and imports | All current Python components compiled and imported; placeholders do not implement behavior |
| Shared data | 99 fragments and 16 cascades passed their models; source IDs were unique |
| Development patients | A/B/C passed `Patient`; their medicine registrations exist in the catalogue and source data |
| Downloader, XML inspector, catalogue parser | Ran together using local HTTP/ZIP/XML fixtures; not a live AEMPS download or full catalogue regeneration |
| CIMA parser | Local API-response fixture produced four valid cleaned section/subsection fragments; not a live CIMA download |
| Fresh index build | All 99 real fragments embedded in a temporary database; 384 dimensions and original stored text/identifiers checked; temporary files removed |
| Working index retrieval | 99 records; filtered searches for 4.2, 4.3, 4.4, 4.5, and 4.8 returned corresponding sections/subsections for 65222 |
| Embedding length audit | 58 of 99 title-plus-text passages exceed the actual 512-token input limit; largest passage measured 4,095 tokens |
| Citation boundary check | Confirmed that the substring verifier accepts a quotation ending partway through a word; unresolved |

`tests/test_busqueda.py` uses fake resources to check medicine/section filters,
subsections, query prefixes, output alignment, result limits, empty candidate
sets, invalid inputs, and the pair-search wrapper. It does not measure real E5
relevance or Recall@3.

Run all automated tests from the project root:

```powershell
.\.venv\Scripts\python -m pytest -v
```

Run only retrieval tests:

```powershell
.\.venv\Scripts\python -m pytest tests/test_busqueda.py -v
```

### Live development-patient output supplied by Gui

Gui ran the real pipeline locally and pasted its output into the chat. These
were directional interaction queries using development patients' medicines,
not complete patient reports or the final labelled evaluation. Expected
results have not been changed to match observed output.

| Patient | Medicine sheet / requested ingredient | Returned outcome | Assessment |
| --- | --- | --- | --- |
| A | 68339 / paracetamol | `found`, citation 68339_4.5 | Off-target answer about flucloxacillin, absent from this patient's pair |
| A | 85780 / paracetamol | `found`, citation 85780_4.5 | Off-target answer about anticoagulants, absent from this patient's pair |
| B | 63064 / amoxicilina | `not_found`, `no_evidence` | Ingredient filter left no sources; no generation |
| B | 63064 / acido clavulanico | `not_found`, `no_evidence` | Ingredient filter left no sources; no generation |
| B | 65222 / warfarina | `found`, citation 65222_4.5 | Expected interaction evidence, but description adds dose-adjustment advice and statements not supported by its selected quotation |
| C | 65461 / metformina | `not_found`, `no_evidence` | No interaction finding generated for this directional query |
| C | 55211 / amlodipino | `not_found`, `no_evidence` | No interaction finding generated for this directional query |

Patient A's intended issue is ingredient duplication. These queries do not
test duplicate detection, and no patient-level routing currently sends
shared-ingredient pairs to deterministic rules. Both A answers passed literal
citation checks but did not answer the actual pair question. Source matching
alone therefore does not establish relevance.

Patient B's accepted description included "ajustar la dosis del anticoagulante
si es necesario", which the configured phrase list did not catch. Its quote
describes reported INR increases, but does not support all the description's
monitoring/dose-adjustment language. This fails the intended output restriction
despite literal citation acceptance. The other sheet's `no_evidence` outcomes
do not cancel the evidence found in 65222.

Patient C's outcomes match the intended absence of generated interaction
findings, but do not establish overall safety or complete coverage. No accuracy
percentage is claimed from these seven diagnostic queries.

## Remaining work

1. Strengthen interaction input/prompts to focus on the exact pair and exclude
   unrelated co-medications. Keep answer relevance distinct from literal
   citation verification.
2. Expand forbidden-wording handling, including Spanish dose-adjustment
   language, and add regression cases for the observed failures. These fixes
   have not been implemented.
3. Agree on patient-level integration: identify ingredient duplicates before
   interaction review, form distinct medicine pairs, check both directions as
   appropriate, and combine repeated findings. That orchestration is absent.
4. Review the verifier's incomplete-word acceptance with Person 4.
5. Evaluate smaller chunks because 58 sections exceed E5's input limit. Full
   stored text is preserved, but only the beginning contributes to an
   overlength section's embedding.
6. Repeat live checks after fixes, support Person 5's separate labelled
   evaluation, and transfer the final Person 3 material into the group
   technical and end-to-end documents.

The renamed evaluation-patient file is intentionally left as agreed. The
rules engine, interface, and evaluation runner remain placeholders. No
changes to these components are implied by this documentation update.
