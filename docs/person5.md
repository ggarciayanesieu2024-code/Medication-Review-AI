# Person 5 - Interface, report and evaluation

Last updated: 16 November 2026 (after the E1 to E5 evaluation patients were added).

## Status

| Deliverable | Status |
| --- | --- |
| 1. Streamlit screen (`app.py`) | Done with sample findings. Not yet connected to the rules or RAG pipeline. |
| 2. Report (`src/informe.py`, `src/exportar.py`) | Done. Saves PDF and web page to `output/`. |
| 3. Safety cases and report tests | Report tests done (`tests/test_informe.py`). `tests/fixtures/safety_cases.json` pending. |
| 4. 20 evaluation questions (`evaluacion/preguntas.json`) | Pending. Needs the shared CIMA file to check expected answers. |
| 5. Metrics (`evaluacion/`) | Pending. Needs the connected system. |
| 6. Evaluation write-up | Pending. |

## Run the screen

From the project root, with the environment active:

```bash
streamlit run app.py
```

To try only the screen without the full installation, these packages are enough: `streamlit pydantic python-dotenv reportlab`.

Choose a patient, click **Review**, and read the report. **Download the report** offers:

- **PDF**: opens in Preview, Adobe Reader or any browser.
- **Web page**: opens in any browser and has a "Print or save as PDF" button.

Both files are also saved in `output/`, which Git ignores.

Evaluation patients are hidden unless **Show evaluation patients** is ticked in the sidebar.

### Hosted demo (Streamlit Community Cloud)

The shared link is deployed from `main` with **Main file path** `deploy/streamlit_app.py`. That entry point runs `app.py` but makes Community Cloud install only `deploy/requirements.txt` (Streamlit, Pydantic, python-dotenv, ReportLab) instead of the full project list with PyTorch. When the app is connected to the real pipeline, add the packages it needs to `deploy/requirements.txt`, and the Groq key under **Advanced settings → Secrets**. Deploying needs admin rights on the GitHub repository. An app deployed from a private repository is private until it is shared.

## Files

| Path | What it is |
| --- | --- |
| `app.py` | Streamlit screen: patient dropdown, Review button, report, downloads |
| `src/informe.py` | Builds the report from `Patient` + `Finding` objects; `FOOTER`; forbidden-phrase check |
| `src/exportar.py` | Turns the report into a web page (`render_html`) and a PDF (`render_pdf`, ReportLab) |
| `data/demo/pacientes_demo.json` | Four extra development patients for the demo (DEMO_D to DEMO_G) |
| `data/demo/hallazgos_ejemplo.json` | Sample findings per patient (placeholders) and a short note on what each case shows |
| `tests/test_informe.py` | Footer, structure, safety and download tests |

`requirements.txt` gains `reportlab`. `.gitignore` gains `output/*`.

## What the report shows

1. Medicines grouped by prescriber, in the order they appear in the patient file
2. Duplicate active ingredients (`tipo: duplicado`)
3. Same therapeutic group (`tipo: advertencia_atc`)
4. Possible interactions (`tipo: interaccion`) with medicine, registration number, section and literal quotation. Pairs checked without information are listed separately, with a note that this does not mean the combination is safe.
5. Possible prescribing cascades (`tipo: cascada`), as hypotheses to review
6. Questions for the next appointment: each finding's `pregunta_medico`, plus a general question about other medicines and supplements
7. The mandatory footer, always last. The PDF also prints it at the bottom of every page.

Descriptions and questions are checked again with the shared `FORBIDDEN_PHRASES` (same normalisation as `src/rag.py`). Text that matches is withheld and counted. Literal quotations are shown unchanged.

## What the screen needs from Persons 2-4

`run_review(patient)` in `app.py` must return:

1. a list of `Finding` objects or dictionaries (`src/modelos.py`), and
2. a list of short texts for pairs checked without information, for example `"Aldocumar 5 mg + Amoxicilina/Ácido clavulánico Tevagen"`.

Suggested mapping from `src.rag.review_interaction()`:

- `found` → `Finding(tipo="interaccion", descripcion=response["description"], cita=<first citation>, pregunta_medico=...)`
- `not_found` → one text for the pair
- `rejected` or `error` → not shown, but counted for the evaluation

To be agreed: who writes the `pregunta_medico` for interactions (rules, RAG, or a fixed template in the report), and whether a finding can carry more than one citation (`Finding.cita` holds one).

## Sample data and demo patients

The sample findings are placeholders for building the screen. They are not expected results, and their quotations are not CIMA text. The report shows a "Sample data" banner while they are in use.

The demo patients use only medicines that appear in the group's patient files, so the real system can analyse them once connected. They are development data, not evaluation data. Lucía should review them; they can move to `data/pacientes/` if the group agrees.

| Patient | What it shows |
| --- | --- |
| A, B, C | Lucía's development patients: duplicate paracetamol; warfarin + antibiotic; nothing to flag |
| DEMO_D | Metformin under two names from two doctors |
| DEMO_E | Warfarin with paracetamol and omeprazole from other doctors |
| DEMO_F | Possible cascade C07: ibuprofen, then amlodipine from another doctor |
| DEMO_G | Five medicines from four doctors, two possible interactions |
| DEMO_H | Carmen's story: ibuprofen, warfarin, paracetamol and pantoprazole from three doctors; three interactions and cascade C14 |

No demo patient shows an ATC warning: the only example available is the statin case, which is evaluation patient E4.

### Development and evaluation patients

Development patients (A, B, C and DEMO_*) are used to build and check the screen, the report and, later, the prompts and rules. Evaluation patients (EVAL_P4 and E1 to E5) are kept aside so the final results measure the system on cases it was not tuned on. They are hidden in the screen unless **Show evaluation patients** is ticked, and the report tests do not use them.

`test_development_patients_do_not_copy_evaluation_cases` checks that no development patient contains an evaluation patient's whole medicine list, or a pair whose result an evaluation patient checks (from `expected_result`). It compares registration numbers, so it does not catch the same scenario with different brands. For that reason, the demo patients were also chosen to avoid the evaluation scenarios: the ibuprofen duplicate (E1), the amlodipine and furosemide cascade (E2), sertraline with warfarin (E3), the two statins (E4) and levothyroxine with a proton pump inhibitor (E5).

## Open points for the group

- ATC/statin case: E4 now says two different statins are not a duplicate. It does not say whether an ATC warning ("same therapeutic group") should appear. Agree this before the evaluation.
- Evaluation patients use two formats: `EVAL_P4` has `resultado_esperado`, and E1 to E5 have `expected_result` and `hypothesis`. The metrics script (deliverable 5) will read both, but one format would be simpler.
- Evaluation patients are not used in Person 5's report tests or sample data.
- `FORBIDDEN_PHRASES` does not catch "ajustar la dosis", which Person 3 observed in a live answer. The safety cases (deliverable 3) will include it.
- Cascade questions in `data/cascadas.json` are in Spanish and the report is in English. Decide one language for the report.
