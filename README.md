# Medication Review AI

Group project for IE University's AI and Machine Learning Foundations course. The planned product is a **prototype for healthcare professionals** to review a synthetic patient's medication list, highlight possible duplicate ingredients, retrieve relevant information from official CIMA drug documents, and show the source of each finding.

**Current status: repository setup only.** The app, data pipeline, rules, evaluation, and example patients have not been implemented. The commands below prepare the Python environment; they do not launch a working demo yet.

## 1. Set up Python

Install Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/). Open a terminal in this repository, then run:

```powershell
uv venv .venv
uv pip install -r requirements.txt
```

This creates a local Python virtual environment named `.venv` and installs the project's planned libraries. Each teammate runs these two commands on their own computer. The `.venv` folder is intentionally excluded from Git.

If you do not use uv, the equivalent commands are:

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

### Activate it when you return to the project

Run the appropriate command **from the repository folder** in each new terminal session:

| Terminal | Command |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| macOS or Linux | `source .venv/bin/activate` |

The terminal should show `(.venv)`. Run `python --version` to check that Python works. Run `deactivate` when finished. If PowerShell blocks the activation script, you can still use `.venv\Scripts\python` directly; activation is optional for `uv pip install -r requirements.txt` because uv finds `.venv` automatically.

## 2. Prepare the API key

When the LLM part is implemented, copy `.env.example` to `.env` and put your own Groq API key in `.env`. On PowerShell:

```powershell
Copy-Item .env.example .env
```

Never commit `.env` or paste an API key into GitHub. No key is needed to create the environment. The model and free-tier limits should be checked against Groq's current documentation before coding that part.

## 3. Planned repository layout

| Path | Intended use |
| --- | --- |
| `data/raw/` | Downloaded CIMA responses for a repeatable demo |
| `data/fichas/` | Cleaned drug-information sections |
| `data/pacientes/` | Invented patient examples only |
| `data/cascadas.json` | Curated, sourced prescribing-cascade hypotheses |
| `src/ingesta.py` | CIMA download and cleaning |
| `src/indice.py` | Section indexing and embeddings |
| `src/rag.py` | Retrieval and LLM call |
| `src/reglas.py` | Deterministic checks |
| `src/verificacion.py` | Citation verification |
| `src/modelos.py` | Pydantic data models |
| `app.py` | Streamlit interface |
| `evaluacion/` | Labelled questions and evaluation script |
| `tests/` | Tests for the rules |
| `docs/` | Technical, end-to-end, and ethics documentation |

The files and folders above are placeholders for the group to fill in. `data/cascadas.json` and `evaluacion/preguntas.json` currently contain empty lists, not evaluated data. The planned Python files contain no implementation.

## Libraries

`requirements.txt` contains only the direct libraries suggested by the project outline: Streamlit for the interface; requests and Beautiful Soup for CIMA retrieval and HTML cleaning; sentence-transformers and ChromaDB for local retrieval; OpenAI's compatible client for the proposed Groq API; Pydantic for validation; python-dotenv for local configuration; and pytest for future tests. Installing sentence-transformers also installs PyTorch, so the first installation may take a while.

## Scope and safeguards

The course brief requires source code, clear run instructions, technical and end-to-end documentation, a presentation, and a live demo. This repository currently provides the starting structure for that work.

Use **synthetic patient data only**. The eventual tool should support professional review and cite the exact source section for a finding. It must not diagnose or recommend stopping or changing treatment. Proposed clinical rules, including duplicate detection from ATC codes and prescribing-cascade examples, need source review and testing before implementation; the outline is a plan, not validated clinical logic.

## Known limitations of CIMA data

- Some mathematical symbols (≥, ≤) appear as `?` in certain fragments of `fragmentos_cima.json`. This comes from the leaflet published on CIMA, not from the parser: the same leaflet shows `?` on some lines and `≥` on others.
- The affected fragments are in sections 4.2, 4.4 and 4.8 (Aldocumar, Ibudol and Januvia leaflets). None belongs to section 4.5, which the system uses to detect interactions.
- Sentence-by-sentence verification against the official leaflet has been done for three section 4.5 fragments: 64014 (PPI), 59718 (warfarin) and 74483. The remaining fragments have not been compared.

## Shared data

Large generated files are not in Git. They are shared through the team Drive folder, with a `.txt` containing the SHA256 hash. Check the hash before use (`Get-FileHash <file> -Algorithm SHA256`).

Current CIMA file: `fragmentos_cima_2026-11-10_v3.json` (18 registration numbers, 99 fragments). Earlier versions are obsolete.
