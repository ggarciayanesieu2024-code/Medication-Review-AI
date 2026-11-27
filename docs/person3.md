# Person 3 - Working Notes

Changes made by Gui (Person 3). Last updated: 27 November 2026.

## Most Recent Changes

The work progressed from checking that the team's components could run together to simplifying their structure and building a local inspection/evaluation workflow. Patient reviews now connect medicine identification, deterministic rules, RAG, validated answers and report generation. `MAIN.ipynb` runs the same production functions in stages so we can inspect failures and compare settings locally before adapting the website.

First, integration checks exposed mismatched data structures and component interfaces. The cascade JSON and its Pydantic models were aligned, loaders were connected to the actual processed files, and the coordinator's findings were made compatible with the reports. Evidence availability and interaction classification are now separate: a supported negative finding is information, missing evidence is a limitation, and an error or rejected answer is an incomplete check.

Next, the repository was reorganised by responsibility. Shared logic stays in its production module instead of being copied into tests, notebook cells or the interface. The notebook supplies inputs and displays outputs; `src/main.py` coordinates reviews, rules perform code matching, and RAG owns search/generation/verification. This makes experiments inspect the behavior the website uses while reducing duplicated orchestration.

In parallel, evaluation cases and the root notebook were assembled. The benchmark definitions are ready for review and experiments, but live quality evaluation and tuning are still outstanding. Earlier tests and diagnostic runs do not establish clinical accuracy.

**Person 3's core ownership remains indexing, retrieval and RAG. Cross-component alignment supported integration; it does not transfer ownership of the interface, report or clinical rule content.**

### Shared configuration and model changes

| Location                    | Responsibility                                                                                        |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| `src/main.py`             | `review_request()`, `review_patient()` and `check_new_medicine()` coordinate patient-level work |
| `src/Rag/`                | Index, search, prompts, grounded answers and literal citation checks                                  |
| `src/Data/`               | Source download/parsing and resource preparation through `ingesta.py`                               |
| `src/Cascades/reglas.py`  | Rule matching plus `CascadeCase` / `CascadeTable` validation                                      |
| `src/Reports/informe.py`  | `Patient`, `Medication`, `Finding` and report composition                                       |
| `src/Rag/verificacion.py` | `Fragment`, `Citation` and quotation verification                                                 |
| `src/llm.py`              | Shared Groq client and `StructuredOutputRAG`, `QuestionPlan`, `PatientQuestionAnswer` contracts |
| `src/Evaluation/`         | Reusable evaluator and labelled inspection cases                                                      |

The final cascade format is `{sources: {...}, cascades: [...]}` with C01-C13. Validation checks required fields, unique IDs and bibliography references. `load_cascades()` returns the dictionary expected by the existing rules. `load_drug_info()` merges package-level nomenclator rows into medicine-level ingredient/ATC codes. Model definitions live alongside their owning components.

`src/config.py` provides shared paths and defaults: Chroma at `data/embeddings/chroma_db`, E5 cache at `data/embeddings/E5_model`, `k=3`, and the Groq model `openai/gpt-oss-120b` unless overridden. Temperature defaults to 0. Notebook `k` and optional `generation_options` are passed as arguments, so experiments do not mutate shared settings. `review_request()` forwards them to applicable interaction searches, patient-question planning and final answers. These are starting settings, not tuned values.

## Indexing and retrieval

`src/Rag/indice.py` reads `data/processed/fragmentos_cima.json`. A section or subsection is one fragment, identified by `nregistro_seccion`. It embeds `passage: ` plus title/text with `intfloat/multilingual-e5-small` on CPU, using normalised 384-dimensional vectors and cosine distance. The full original source and metadata remain stored. Running the indexing module rebuilds the collection.

`search_fragments()` in `busqueda.py` caches the model/collection, embeds queries with the E5 `query: ` prefix and returns up to `k` results within optional medicine/section filters. Section 4.5 includes 4.5.1 but not 4.50. `nregistro=None` and `sections=None` search across the available corpus. Registration numbers stay strings to preserve leading zeroes.

The labelled source snapshot contains 99 sections for 18 registrations. The parsed nomenclator and the smaller UI medicine catalogue have different roles; they should not be treated as the same dataset. Snapshot hashes are recorded in `src/Evaluation/benchmark_manifest.json`. Notebook preparation may extend this snapshot.

## Connected RAG pipeline

The production paths are:

- **Standalone question:** `answer_question()` in `src/Rag/rag.py`; optional medicine/section filters, no patient required.
- **Directional interaction:** `review_interaction()` searches one medicine's 4.5 sections for another ingredient and requires an explicit interaction classification for an accepted `found` answer.
- **Patient review:** `review_request()` in `src/main.py` routes a blank question to `review_patient()`, combining available rules and optional interaction checks across eligible pairs/directions.
- **Patient question:** the coordinator calls `llm_prepare_question()` to resolve intent/medicines, performs review or proposed-medicine checks, then `answer_patient_question()` produces an answer from patient facts, evidence, findings and limitations. Ambiguous identities can require clarification.

`prepare_general_search()` / `prepare_interaction_search()` retrieve evidence and build exact model messages without generation. `run_prepared_request()` generates and validates a structured answer; the notebook separates these steps for inspection. Invalid JSON/schema output gets one fresh attempt. Empty/truncated responses are rejected; API failures remain distinct from absent evidence.

The pipeline checks configured forbidden phrases and verifies each citation against the actual supplied fragments. Medicine registration and exact subsection must match, and the quotation must occur in the source. Rejected generated text is withheld. `Finding.all_citations` preserves all accepted supporting citations for reporting.

| Outcome                      | Interpretation                                                                               |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| `found`, interaction true  | Evidence-supported potential interaction alert                                               |
| `found`, interaction false | Explicit documented negative finding within its stated conditions                            |
| `not_found`                | No evidence survived filtering, or supplied evidence was insufficient; never proof of safety |
| `rejected` / `error`     | Validation or generation failure; the check could not be completed                           |

Duplicate pairs receive deterministic warnings and may still undergo RAG checks. Same-family warnings omit pairs already classified as ingredient duplicates. Proposed-medicine checks retain findings involving the proposed product. Cascade rules match ATC co-occurrence and produce hypotheses, not causal diagnoses. The current website uses patient-level coordination; exposing a question without a patient remains a UI task, with the standalone RAG function already available.

## Local notebook and evaluation

Run `.venv\Scripts\python -m jupyterlab MAIN.ipynb` from the repository root and select the project kernel. Cells share a persistent Python session, so rerun dependent stages after input changes and restart after editing imported code.

Section 1A selects a prepared scenario or custom patient/question and displays the active patient test. `custom_patient_overrides` changes context or medicines in memory. Independent standalone inputs (`search_question`, `rag_task`, `nregistro`, `sections`, `ingredient`) drive sections 4-5. Section 1B contains `k`, `generation_options`, run switches, preparation controls and batch selection. The current switches enable model calls; choose the stages before using Run All.

Section 2 uses `prepare_notebook_data()` to reuse validated nomenclator/sources, fetch missing medicine sections and rebuild the index if its contents differ. Existing covered sections are not automatically refreshed. `check_updates_since` reports changes since an explicit date; failure does not establish freshness. Sections 3-6 expose rule findings, retrieved/filtered fragments, exact prompts, checked answers, audit logs, incomplete checks and the complete report. The notebook imports production functions instead of test doubles or duplicate implementations.

Evaluation files and patient scenarios:

| File                          | Use                                                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `evaluation_questions.json` | 20 standalone RAG cases: 15 development and 5 reserved; questions, source labels, reference answers and manual criteria    |
| `evaluar.py`                | `evaluate_cases()` runs retrieval-only or generated-answer batches and retains settings, fragments, outcomes and timings |
| `data/pacientes/patient_scenarios.json`    | 12 scenarios using A/B/C and E1-E6, including proposed medicine, clarification and added context; loaded in section 1A     |
| `benchmark_manifest.json`   | Source/patient file fingerprints identifying the labelled snapshot; not automatically enforced by the notebook             |

Section 7 evaluates **RAG**, using exact source Hit@k/Recall@k and optional generated-status/interaction matches. Missing labels remain unscored; errors and denominators must be reported. Reference answers, claim support and citation relevance require manual judgment. Cascades, duplicates, family warnings and grouping are inspected through patient scenarios, not scored by this batch evaluator. Automated regression tests are a separate layer.

The labels are source-checked drafts pending review. E6 is the distinct former EVAL_P4 grouping case, normalised to the E1-E5 format. E4 needs a same-family warning despite no ingredient duplicate. Class-based evidence in D07/H02 deliberately exposes the literal-filter limitation. Existing evaluation patients and reserved questions have been inspected and share concepts with development cases; they are not an untouched clinical test set. Do not change labels merely to agree with model output.
