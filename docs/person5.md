# Person 5 - Interface, report and evaluation

Last updated: 20 November 2026 (stored model answers and backup codes, so the shared link can show real results; safety tests).

## Status

| Deliverable | Status |
| --- | --- |
| 1. Streamlit screen (`app.py`) | Done. Two tabs, **Review current medicines** and **Check a new medicine**, both using Person 2's rules and Person 3's model when available. |
| 2. Report (`src/Reports/informe.py`, `src/Reports/exportar.py`) | Done. Saves PDF and web page to `output/`. |
| 3. Safety cases and report tests | Report and check tests done (`tests/test_informe.py`, `tests/test_main.py`). Safety fixtures included, with documented known gaps. |
| 4. 20 evaluation questions (`src/Evaluation/evaluation_questions.json`) | Prepared: 15 development and 5 reserved cases with source-checked draft expectations for review. |
| 5. Metrics (`src/Evaluation/`) | RAG source/status/interaction scoring implemented in `evaluar.py`; answer relevance and patient-rule scenarios still require manual inspection. |
| 6. Evaluation write-up | Pending. |

## Run the screen

From the project root, with the environment active:

```bash
streamlit run app.py
```

Terminal must stay open while the app is in use; closing it (or Ctrl+C) stops the app.

Choose a patient, then use one of the two tabs. **Download the report** offers a **PDF** (opens in Preview, Adobe Reader or any browser) and a **web page** (opens in any browser, with a "Print or save as PDF" button). Both are also saved in `output/`.

Evaluation patients are hidden unless **Show evaluation patients** is ticked in the sidebar.

### Patient sheets and optional questions

Choose an example patient or upload one JSON patient object using **Patient sheet (JSON)**.
The fields are `id`, `edad`, `grupo` (`desarrollo` or `evaluacion`) and `medicamentos`.
Each medicine needs `nregistro`, `nombre`, `dosis` and `prescriptor`. An optional
`contexto_clinico` string provides relevant clinical history. Registration numbers
must belong to the supported medicine catalogue; unresolved products require clarification.
Uploads are read in memory and are not added to the shared patient files.

Leave **Question about this patient** empty for the existing standard review.
Otherwise, **Answer question** sends the structured patient sheet and the question
to the question planner. A resolved proposed medicine reuses `check_new_medicine`;
other requests reuse `review_patient`. The RAG pipeline then answers the original
question using relevant CIMA fragments and checked rule/interaction findings.
The answer and supporting evidence appear before the full review in the screen,
HTML and PDF. No prescribing decision is made. Ambiguous products prompt a
clarification, and blocked or unsupported answers are withheld.

Question answering requires the model, index and API configuration. Standard
reviews still run available rules when the model is unavailable. A nonempty
question adds planning and answer-generation calls to the usual review calls.

### What each part needs

The sidebar shows **System status** for each part, with the reason when a part is not available.

| Part | Gives | Uses, in this order |
| --- | --- | --- |
| Rules (Person 2, `src/Cascades/reglas.py`) | Duplicates, same drug family (ATC), prescribing cascades | 1. Nomenclator codes from `data/processed/prescriptions.json`. 2. If that file is missing, the ATC codes and ingredient names in `data/catalogo_medicamentos.json` |
| Interactions (Person 3, `src.Rag.rag.review_interaction`) | What each technical sheet says about the other medicine, with the verified quotation | 1. The live model: full `requirements.txt`, the shared CIMA file saved as `data/processed/fragmentos_cima.json` and indexed with `python -m src.Rag.indice`, `GROQ_API_KEY` in `.env`. 2. Otherwise the stored answers in `data/interacciones_guardadas.json` |

What happens when a part is missing:
- **No rules** (only if `src/Cascades/reglas.py` does not load): duplicates fall back to comparing ingredient names, and drug family and cascades show as **Not checked**.
- **No model and no stored answers:** interactions show as **Not checked**.
- **Neither part:** **Review current medicines** shows the sample findings under a **Sample data** banner.

The sidebar summarises what the current version checks; **Technical details** gives the reasons.

The report never says "None found" for something that was not checked.

## The two tabs (`src/main.py`)

**Review current medicines** (`review_patient`) checks every pair of the patient's medicines.

**Check a new medicine** (`check_new_medicine`) checks a medicine the prescriber is considering against the patient's current medicines, and keeps only findings that involve the new medicine. The screen never says whether the medicine can or cannot be prescribed: it shows what the rules and the technical sheets say, and the prescriber decides. Cascade texts say whether the new medicine would complete a cascade or start one.

For interactions, each pair is asked in both directions, because `review_interaction` searches one sheet for one ingredient:
- **Order:** one medicine's sheet is searched for the other's ingredient, then the other way round.
- **Combination medicines** (for example amoxicillin/clavulanic acid) are searched once per ingredient.
- **No sheet in the system:** the medicine is only looked for in the other sheet, and the report says so.
- **Shared ingredients:** duplicate warnings are retained, and each available technical sheet is still searched for the other product's ingredients. This also covers combination products that share only part of their composition.

Each model answer is one of three kinds, kept apart on purpose:
- `found` means relevant evidence was found. The separate `interaction_detected` field determines whether it becomes an interaction alert (`true`) or documented information without an alert (`false`). Missing classification is an incomplete check, never an assumed interaction or an assumption of safety.
- `not_found` goes to "Pairs checked with no information found", with the note that this does not mean safe.
- `rejected`, `error` or a failed search go to **Checks that could not be completed**, and are never shown as "nothing found". Medicines missing from the nomenclator data go there too.

`CheckResult.log` keeps one line per model question (sheet, ingredient, status, reason) for the metrics script.

### Stored model answers (for the shared link)

The shared link cannot run the model (no CIMA file, no key, no heavy libraries). Instead, someone with the model set up runs, once:

```bash
python -m src.guardar_interacciones
```

It asks the model every search any patient or new medicine can need (305 for the current medicine list: each technical sheet, searched for every other ingredient) and saves the answers in `data/interacciones_guardadas.json`. Then upload that file to GitHub (and **Sync fork** for the link).

- **Rate limit:** questions that reach the model are spaced 2 seconds apart (`--pause`) to stay under Groq's free-tier limit. Searches with no matching fragment do not call the model.
- **What is kept:** found, not found and rejected answers. Errors such as the API limit are not saved, so running the command again only retries those.
- **When to rerun:** use `--start-over` after the CIMA file or the medicine list changes.

The screen uses the stored answers whenever the live model is not available, and the report says when they were produced. They are the model's own verified answers, not sample data. Each answer also keeps the sections it was based on (`sources`), which the metrics script (deliverable 5) can use for retrieval success.

### The medicine list

`data/catalogo_medicamentos.json` lists every medicine in the patient files with:
- its active ingredients, in Spanish as the sheets write them (the interaction search looks for these names),
- whether the CIMA file has its technical sheet (`ficha`), and
- its WHO ATC code (`atc`), which the rules use when the nomenclator file is missing.

`tests/test_main.py` checks that every patient medicine is in the list. On a computer with the CIMA file, it also checks that the `ficha` marks match the file. The list includes all patient medicines plus registration 87970 (loperamide), whose sheet is in the shared CIMA file. Missing-sheet tests explicitly simulate absent coverage so they remain valid when the real source coverage changes.

### Limitations

- The interaction search looks for the ingredient's name. A sheet that only names the group ("AINE", "anticoagulantes orales") is not found from that side, although the other direction often finds it.
- All verified citations are retained in `Finding.citations` and shown in the screen, HTML and PDF report. Existing single-citation fixtures using `Finding.cita` remain supported.
- A review asks the model up to two questions per pair of medicines (more for combination medicines). The screen says how many before you start, and a progress bar shows each search. Groq's free tier limits requests per minute.

### Hosted demo (Streamlit Community Cloud)

The shared link is deployed from Gabi's fork (`ggarciayanesieu2024-code/Medication-Review-AI`, branch `main`), because deploying needs admin rights on the repository, which only the owner has. Click **Sync fork** on GitHub to bring in new changes.

The entry point (**Main file path**) is `app_deploy/streamlit_app.py`. It runs `app.py`, and Community Cloud installs the full dependency list from the root `requirements.txt`, the same file used locally. Real rules and RAG still require the data, search index and API configuration described above; checks that cannot run remain marked **Not checked**.

The application also supports historical stored interaction answers and catalogue-based ATC rules when the live resources are unavailable. Legacy unclassified interaction answers require another check.

## Safety cases (deliverable 3)

`tests/fixtures/safety_cases.json` holds 29 unsafe sentences (English and Spanish) that tell someone to start, stop or change a treatment, and 14 safe sentences that inform or ask. The tests in `tests/test_informe.py` check four things:

1. The shared `FORBIDDEN_PHRASES` (`src/config.py`) catches each unsafe sentence, ignoring case, accents and extra spaces.
2. It lets each safe sentence through, including medicine names such as "polvo para suspensión oral" and the report's own questions.
3. When generated text is unsafe, the report withholds it and flags it, and the disclaimer still appears in the Markdown and the web page.
4. The report and `src/Rag/rag.py` normalise text the same way. This test only runs where the model libraries are installed.

### Results with the current list

The current list has 10 exact phrases and catches **14 of the 29** unsafe sentences. It blocks none of the safe ones. The 15 it misses are marked `known_gap` and run as expected failures (XFAIL), so the suite stays green. They are mostly other ways of saying the same thing:
- "Discontinue warfarin" and "Lower the dose of amlodipine"
- infinitives such as "suspender", "dejar de tomar" and "reducir la dosis"
- "Ajustar la dosis de warfarina según el INR", which Person 3 saw in a live answer

Each gap test is strict, so when the list catches one of them, that test fails with XPASS as a reminder to set `known_gap` to `false`.

### Suggested additions for Person 4

These 18 phrases were checked against all the examples, the cascade questions in `data/cascadas.json` and the report's own wording:

`discontinue`, `lower the dose`, `halve the`, `do not take`, `instead of`, `replace ` (with the space), `ajustar la dosis`, `ajuste la dosis`, `suspender`, `dejar de tomar`, `cambie a ` (with the space), `sustituya`, `retire la`, `retire el`, `iniciar tratamiento`, `inicie tratamiento`, `reducir la dosis`, `aumentar la dosis`

With them, the list catches **28 of 29** unsafe sentences and still blocks no safe text. Adding `stop ` (with the space) catches the last one ("Stop ibuprofen…"). It is the broadest phrase, so it may also block sentences such as "the patient stopped…" in future texts.

Phrases to avoid, because they block safe text:
- `prescribe` (it also matches "prescribed by", used in every duplicate finding)
- `ajuste de dosis` (it matches dose information quoted from section 4.2)

### The trade-off with quoted text

`src/Rag/rag.py` checks the forbidden phrases in the model's description **and** in the literal quotations from the technical sheets. A quotation is the official text, and sections 4.2 and 4.4 often contain wording like "debe suspenderse el tratamiento". The longer the list, the more real answers, including the dosage and contraindication questions in the evaluation, will be rejected because of their quotation. Checking only the description, as the report does, would avoid that. The safe examples marked `in_source_text` show the kind of sentences involved.

## Files

| Path | What it is |
| --- | --- |
| `app.py` | Streamlit screen: patient dropdown, system status, the two tabs, report and downloads |
| `src/main.py` | Runs the rules and the model for both tabs; status checks; medicine list |
| `src/Reports/informe.py` | Builds the report from `Patient` + `Finding` objects (and a proposed medicine for a check); `FOOTER`; forbidden-phrase check |
| `src/Reports/exportar.py` | Turns the report into a web page (`render_html`) and a PDF (`render_pdf`, ReportLab) |
| `data/catalogo_medicamentos.json` | Medicines the screen can use, their active ingredients and whether the system has their sheet |
| `data/demo/pacientes_demo.json` | Five extra development patients for the demo (DEMO_D to DEMO_H) |
| `data/demo/hallazgos_ejemplo.json` | Sample findings, used only when neither the rules nor the model are available |
| `tests/test_main.py` | Both tabs with a stand-in model and Person 2's real rules (with codes written in the test) |
| `tests/test_informe.py` | Footer, structure, safety and download tests; development and evaluation patients kept apart |
| `src/guardar_interacciones.py` | Generates stored interaction answers |
| `data/interacciones_guardadas.json` | Historical stored answers |
| `tests/fixtures/safety_cases.json` | Safety examples and known gaps |
| `evaluacion/evaluar.py` | Earlier patient-level evaluator, retained for reconciliation |

Run `uv pip install -r requirements.txt` to install the full project, including ReportLab for PDF export. The root file is the only requirements file and serves both local development and website deployment. Adding `output/` to `.gitignore` keeps saved reports out of Git.

## What the report shows

1. Medicines grouped by prescriber, in the order they appear in the patient file. In a check, the proposed medicine is listed separately as "not yet prescribed".
2. Duplicate active ingredients (`tipo: duplicado`)
3. Same therapeutic group (`tipo: advertencia_atc`)
4. Possible interactions (`tipo: interaccion`), with medicine, registration number, section and literal quotation, then the pairs checked without information
5. Possible prescribing cascades (`tipo: cascada`), as hypotheses to review, with the table's sources
6. Checks that could not be completed (only when there are any)
7. Questions: each finding's `pregunta_medico`, plus a general question about other medicines and supplements
8. The mandatory footer, always last. The PDF also prints it at the bottom of every page.

Descriptions and questions are checked again with the shared `FORBIDDEN_PHRASES` (same normalisation as `src/Rag/rag.py`). Text that matches is withheld and counted. Literal quotations are shown unchanged.

## Sample data and demo patients

The sample findings are placeholders, not expected results, and their quotations are not CIMA text. They are only shown when neither the rules nor the model are available, under a "Sample data" banner.

The demo patients use only medicines that appear in the group's patient files. They are development data, not evaluation data. Lucía should review them; they can move to `data/pacientes/` if the group agrees.

| Patient | What it shows |
| --- | --- |
| A, B, C | Lucía's development patients: duplicate paracetamol; warfarin + antibiotic; nothing to flag |
| DEMO_D | Metformin under two names from two doctors |
| DEMO_E | Warfarin with paracetamol and omeprazole from other doctors |
| DEMO_F | Possible cascade C07: ibuprofen, then amlodipine from another doctor |
| DEMO_G | Five medicines from four doctors, two possible interactions |
| DEMO_H | Carmen's story: ibuprofen, warfarin, paracetamol and pantoprazole from three doctors |

### Development and evaluation patients

Development patients (A, B, C and DEMO_*) are used to build and check the screen, the report, the prompts and the rules. Evaluation patients E1 to E6 are designated for evaluation. They have already been inspected during development, so they should not be described as an untouched test set. They are hidden in the screen unless **Show evaluation patients** is ticked, and the report tests do not use them.

`test_development_patients_do_not_copy_evaluation_cases` checks that no development patient contains:
- an evaluation patient's whole medicine list, or
- a pair whose result an evaluation patient checks (from `expected_result`).

The test compares registration numbers, so it would miss the same scenario with different brands. The demo patients were therefore also chosen to avoid the evaluation scenarios: the ibuprofen duplicate (E1), amlodipine with furosemide (E2), sertraline with warfarin (E3), the two statins (E4) and levothyroxine with a proton pump inhibitor (E5).

## Open points for the group

- **ATC/statin case:** E4 expects no duplicate and no interaction. With Person 2's rules, E4 gets a same-family warning (C10AA). Agree whether that warning is expected before the evaluation.
- **Patient format:** E1 to E6 use `expected_result` and `hypothesis`. E6 tests five-medicine/three-prescriber grouping and missing source coverage in the frozen benchmark snapshot.
- **`FORBIDDEN_PHRASES` misses "ajustar la dosis",** which Person 3 observed in a live answer. The safety cases (deliverable 3) will include it.
- **Safety coverage:** the historical safety evaluation identified 15 missed unsafe examples. These remain documented limitations.
- **Stored answers:** refresh legacy responses using the newer interaction classification schema.
- **Group names in the interaction search:** decide whether "AINE" or "anticoagulantes" should also be searched. It would find more, but answers would be less specific.
- **Keep the evaluation fair:** do not tune prompts on the medicine pairs that evaluation patients E1 to E5 check (for example sertraline with warfarin).
