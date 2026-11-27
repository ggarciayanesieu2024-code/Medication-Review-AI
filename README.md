# Medication Review AI

IE University AI and Machine Learning Foundations group project: a **prototype for healthcare professionals** to review synthetic medication lists using deterministic rules and official AEMPS/CIMA technical sheets.

The system flags shared active ingredients, medicines in the same ATC family, possible prescribing cascades, and evidence-supported interactions. It can review a patient's medicines, answer a question using the patient sheet, or check a proposed medicine. Findings include sources and coverage limitations. The clinician decides; the system does not diagnose or prescribe treatment changes.

The Streamlit interface uses the production modules in `src/`. `MAIN.ipynb` runs those same functions locally, with separate stages for inspecting inputs, retrieval, prompts, answers and evaluation. Standalone questions without a patient are available through the notebook/RAG API; the website currently requires a patient.

## 1. Set up Python

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/). From the repository root:

```powershell
uv venv .venv
uv pip install -r requirements.txt
```

Alternatively, create the environment with `py -m venv .venv`, then install with `.venv\Scripts\python -m pip install -r requirements.txt`. The root `requirements.txt` is shared by local use and deployment.

To activate the environment:

| Terminal | Command |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

You can also call `.venv\Scripts\python` directly on Windows without activation. The commands below assume the repository root as the working directory.

## 2. Prepare the API key

Copy `.env.example` to `.env` and set `GROQ_API_KEY` for model calls:

```powershell
Copy-Item .env.example .env
```

Keep `.env` private. Paths and default model/retrieval settings are in `src/config.py`; `GROQ_MODEL` can override the default model. Rules-only checks need no API key. Retrieval needs the CIMA index and E5 embedding model, which may download on first use.

## 3. Repository layout

| Path | Use |
| --- | --- |
| `MAIN.ipynb` | Interactive local runner for the production pipeline and evaluation |
| `app.py` / `app_deploy/streamlit_app.py` | Local Streamlit interface / hosted entry point for the same app |
| `src/main.py` | Coordinates patient reviews, patient questions and proposed-medicine checks |
| `src/config.py` / `src/llm.py` | Shared paths/settings and Groq calls with Pydantic response validation |
| `src/Data/` | Downloading, parsing and conditional preparation of local resources |
| `src/Rag/` | Indexing, filtered retrieval, prompts, answers and citation verification |
| `src/Cascades/` | Ingredient duplicates, ATC-family warnings and prescribing-cascade hypotheses |
| `src/Reports/` | Patient/finding models, report composition and HTML/PDF export |
| `src/Evaluation/` | Batch RAG evaluator, labelled questions, patient scenarios and source manifest |
| `data/raw/` / `data/processed/` | Downloaded responses / parsed nomenclator and CIMA sections |
| `data/embeddings/` | Local Chroma index and cached E5 model |
| `data/pacientes/` / `data/catalogo_medicamentos.json` | Synthetic patients / medicine selection catalogue |
| `data/cascadas.json` | Curated cascade patterns and bibliography |
| `tests/` / `docs/` | Automated regression tests / component and project documentation |

Launch the website with the environment active:

```powershell
streamlit run app.py
```

Choose a patient or upload a patient JSON file. Leave the question blank for a standard review, enter a patient question, or use the proposed-medicine tab. See `docs/person5.md` for patient format and interface details; `docs/person3.md` explains the integrated RAG and notebook workflow.

## Manual checks and RAG evaluation

Open `MAIN.ipynb` with the project's `.venv` kernel, or launch it with:

```powershell
.venv\Scripts\python -m jupyterlab MAIN.ipynb
```

Run imports, choose a prepared scenario or custom patient/question in section 1A, then select controls in 1B. Set `k`, optional `generation_options` and the relevant run switches before continuing; the current switches enable model calls and batch evaluation. Sections 3 and 6 inspect rules and the complete patient workflow; sections 4-5 inspect a separate standalone RAG request. Rerun dependent cells after input changes and restart the kernel after editing imported code.

Section 2 reuses valid local data, downloads missing medicine sections and rebuilds the index when its stored contents differ. Optional update checks report changes without refreshing existing sections. This preparation can change source coverage. For manual data preparation, the existing modules can also be run from the root: `python -m src.Data.download_nomenclator`, `python -m src.Data.parse_nomenclator`, `python -m src.Data.parse_cima --help`, and `python -m src.Rag.indice`.

Section 7 calls `src/Evaluation/evaluar.py` with `evaluation_questions.json`: 15 development and 5 reserved RAG cases. It scores expected source sections and optional answer-status/interaction labels; answer correctness and citation relevance still require inspection. `patient_scenarios.json` supplies 12 scenarios across 9 patients for manual workflow checks, including cascades and duplicates. `benchmark_manifest.json` records the source snapshot; the notebook does not automatically verify its hashes. Labels are source-checked drafts, and the reserved cases have already been inspected rather than kept completely blind.

Run regression tests with `.venv\Scripts\python -m pytest -q`. Passing tests checks exercised behavior; it does not establish live retrieval quality or clinical correctness. Clear notebook outputs before sharing synthetic patient experiments.

## Libraries

The shared requirements include Streamlit, Pydantic, Groq/OpenAI clients, requests/Beautiful Soup, sentence-transformers, ChromaDB, ReportLab, pytest and Jupyter tooling. Installing sentence-transformers also installs PyTorch, so the first installation may take time.

## Scope and safeguards

Use synthetic patients only. Keep absent evidence, failed checks and documented negative findings distinct: `not_found` never establishes safety. Cascade matches are hypotheses based on co-occurrence; the rules do not establish prescribing sequence, causality or dose-dependent risk. Automatic citation checks verify quotation presence and identifiers, not whether a quotation supports the answer. The prototype still needs manual review, live evaluation and tuning.

## Known limitations of CIMA data

Some mathematical symbols appear as `?` in the supplied CIMA sections. Long sections can exceed E5's input limit: the index keeps the complete source text, but its embedding represents only the supported input length. The interaction mention filter can miss class names and synonyms. Selected benchmark passages have been checked against official sheets; the entire local corpus has not been verified sentence by sentence.

## Shared data

Large raw/generated data and local indexes are not in Git. The team's shared CIMA snapshot is `fragmentos_cima_2026-11-10_v3.json` (18 registrations, 99 sections). Place it at `data/processed/fragmentos_cima.json` before preparation. Obtain the parsed nomenclator through the shared data or the preparation scripts. Check supplied SHA256 values with `Get-FileHash <file> -Algorithm SHA256`. If notebook preparation updates the sources, recheck affected benchmark labels and snapshot hashes.
