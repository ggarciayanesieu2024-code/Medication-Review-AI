# Design Decisions

This document records the goal of the project, the rules it must follow, the
technical decisions taken so far and the reasons behind them. Each decision
explains **what** was chosen, **how** it is used and **why** it was chosen.

---

## 1. Project goal

A decision-support tool for polymedicated older adults.

Given a patient's medication list coming from several prescribers, the tool:

1. Shows the medications **grouped by prescriber**.
2. Detects **duplicate medications**.
3. Detects **possible drug interactions**, using retrieval-augmented generation
   (RAG) over official drug information.
4. Flags **possible prescribing cascades** as *hypotheses to review*, never as
   conclusions.

The tool supports a professional's review. It does not replace it.

---

## 2. Hard rules

These rules are not negotiable and apply to every part of the system.

| Rule | Why |
|------|-----|
| The system **never recommends stopping, changing or starting** a medicine. | Treatment decisions belong to the prescriber. The tool only surfaces information for review. |
| The system **never diagnoses**. | Diagnosis is a clinical act outside the scope of a decision-support tool. |
| Patient data is **synthetic only**. | No real health data may enter the project, which avoids privacy and legal risks. |
| **No secrets** in the repository. | API keys and credentials live in a local `.env` file (see `.env.example`), never in version control. |
| Code, comments and docstrings are **in English**. | Keeps the codebase consistent and readable for any contributor. |
| `src/ingesta.py` belongs to a teammate and **must not be modified without agreement**. | Avoids conflicts and respects ownership of that module. |

---

## 3. Technical decisions

### 3.1 LLM: Groq API

- **What:** The language model is accessed through the Groq API (free tier).
- **How:** Using the official `groq` Python package. The API key is read from
  the environment, never hard-coded.
- **Why:** It is free, fast, and needs no local hardware to run the model.

### 3.2 Embeddings: `intfloat/multilingual-e5-small`

- **What:** Text embeddings are produced by `intfloat/multilingual-e5-small`.
- **How:** Run locally with `sentence-transformers`. Following the model's
  usage convention:
  - Passages (drug-information chunks) are prefixed with `"passage: "`.
  - Queries are prefixed with `"query: "`.
- **Why:** It is multilingual (the source documents are in Spanish), small, and
  runs comfortably on a laptop CPU.

### 3.3 Vector database: ChromaDB

- **What:** Embeddings are stored and searched in ChromaDB.
- **How:** Used locally, persisted on disk.
- **Why:** It installs with `pip` and needs no separate server.

### 3.4 Interface: Streamlit

- **What:** The user interface is a Streamlit app.
- **How:** The entry point is `app.py`.
- **Why:** It gives a web interface written in pure Python, with no frontend
  stack to maintain.

### 3.5 Data validation: Pydantic

- **What:** Data structures (patients, medications, findings, citations, etc.)
  are validated with Pydantic.
- **How:** Models are defined in `src/modelos.py`.
- **Why:** Explicit, validated data models catch malformed input early and
  document the shape of the data in code.

### 3.6 Chunking: one chunk per section

- **What:** Drug information is split into one chunk per section of the
  official document. The sections used are:
  - 4.2 Posology and method of administration
  - 4.3 Contraindications
  - 4.4 Special warnings and precautions for use
  - 4.5 Interaction with other medicinal products
  - 4.8 Undesirable effects
- **How:** Each chunk keeps metadata identifying the medicine and the section
  number.
- **Why:** Every citation then points to one specific, identifiable section,
  which makes citations precise and easy to check.

### 3.7 Citation verification in code

- **What:** Every quotation produced by the LLM must exist literally in the
  section it cites.
- **How:** Checked programmatically after generation, by comparing the quoted
  text against the stored section text. Citations that fail the check are
  rejected.
- **Why:** This is the main defence against hallucinated citations. The model's
  output is not trusted unless the source confirms it.

### 3.8 Rules before AI

- **What:** Parts of the analysis that can be solved deterministically do not
  use AI:
  - **Duplicates** are detected using ATC codes.
  - **Prescribing cascades** are detected using a hand-curated table.
- **How:** Plain Python logic over structured data.
- **Why:** Deterministic results are reproducible, easier to test and easier to
  defend. AI is reserved for the task that needs it: reading free-text drug
  information to find possible interactions.

### 3.9 No fine-tuning

- **What:** The models are not fine-tuned.
- **How:** Quality is improved by tuning other parts of the pipeline instead:
  - chunk size,
  - number of retrieved chunks,
  - prompts,
  - deterministic rules.
- **Why:** Fine-tuning needs GPUs and thousands of labelled examples, which the
  project does not have.

---

## 4. Data sources

### 4.1 AEMPS prescription nomenclator

- **What:** The official list of medicines available for prescription in Spain.
- **Where:** <http://listadomedicamentos.aemps.gob.es/prescripcion.zip>
- **How:** Downloaded by `src/download_nomenclator.py`.
- **Why:** It is the authoritative source of medicines, codes (including ATC)
  and identifiers needed to build synthetic patients and detect duplicates. to be verified after download

### 4.2 CIMA REST API

- **What:** The AEMPS medicines information centre, which provides the official
  drug information (summary of product characteristics) by section.
- **Where:** <https://cima.aemps.es/cima/rest>
- **How:** Queried to download sections 4.2, 4.3, 4.4, 4.5 and 4.8 for the
  medicines used by the synthetic patients.
- **Why:** It is the official source for the text indexed in the RAG pipeline.

> **Note:** The CIMA field names used in the code are **to be verified against
> real responses**. They must not be assumed correct until checked.

---

## 5. Current status

- Repository cloned.
- Virtual environment created.
- Dependencies installed.
- `groq` added to `requirements.txt`.
- `src/download_nomenclator.py` written, but **not yet run successfully**.

---

## 6. Next steps (in order)

1. Run the nomenclator download and inspect the downloaded files.
2. Write the nomenclator parser.
3. Create three synthetic patients, using only medicines present in the
   nomenclator.
4. Download the drug sections for those medicines from CIMA.
5. Build the vector index.

---

## 7. Integración con la API de CIMA (verificado)

This section records what has been checked against real CIMA responses. Raw
responses are saved, unmodified, in `data/raw/cima/`.

### 7.1 Verified endpoints

All endpoints are relative to <https://cima.aemps.es/cima/rest>.

| Endpoint | Returns |
|----------|---------|
| `medicamentos?nombre=` | A paginated search result (`totalFilas`, `pagina`, `tamanioPagina`, `resultados`). |
| `medicamento?nregistro=` | The full record of one medicine, including `docs` with links to its technical sheet. |
| `docSegmentado/secciones/1?nregistro=` | The list of sections in the technical sheet (`seccion`, `titulo`, `orden`), without their text. |
| `docSegmentado/contenido/1?nregistro=&seccion=` | The text of the requested section and its subsections. |

- **Where the section endpoints come from:** the official manual
  `CIMA_REST_API.pdf` v1.23
  (<https://cima.aemps.es/cima/resources/docs/CIMA_REST_API.pdf>).
- **Why not Swagger:** the Swagger page
  (<https://cima.aemps.es/cima/publico/swagger.html>) returns HTTP 404.
- **How to call them:** with the header `Accept: application/json`. The `1` in
  the path means technical sheet (`2` would be the patient leaflet).

### 7.2 Response structure of `contenido`

- **What:** The response is a **list**, not an object. Each element has:
  - `seccion`: the section number, for example `4.5` or `4.6.1`;
  - `titulo`: the section title;
  - `orden`: the display order (it restarts for each main section, so it is not
    a unique identifier);
  - `contenido`: the section text, in HTML.
- **Why it matters:** asking for one section can return several elements, one
  for the section and one for each of its subsections.

### 7.3 Chunking rule

- **What:** One chunk per element **that has text**, labelled with its exact
  number (`4.2`, `4.2.2`, `4.6.1`, ...). Elements with empty content are
  discarded. Citations point to the exact subsection.
- **How we found it:** the content is not placed consistently:
  - in 66337, section `4.6` is empty and the text is in `4.6.1` and `4.6.2`;
  - in 66337, section `4.2` has its own text, and `4.2.2` has separate text that
    is not repeated in `4.2`.
- **Why:** this rule covers both cases. Each citation points to the element
  where the text really is, which keeps citation verification (3.7) exact.
- This refines decision 3.6: the unit is the element returned by CIMA, not only
  the main section.

### 7.4 Checked with two technical sheets

| nregistro | Medicine | Structure |
|-----------|----------|-----------|
| 66337 | Amoxicillin with clavulanic acid | 36 sections, with subsections (4.2.2, 4.6.1, 4.6.2, 5.2.x) |
| 77758 | Paracetamol | 28 sections, no subsections |

Section 4.5 has the same shape in both: a list with one element containing
`seccion`, `titulo`, `orden` and `contenido`.

### 7.5 Pending, not confirmed

- **Gaps in numbering:** the `orden` values in the section lists skip numbers
  (for example 3→6 or 9→11). We do not yet know what these gaps correspond to.
- **HTML cleaning:** stripping the HTML must keep the breaks between paragraphs.
  A naive clean-up joins headings and paragraphs (for example "Anticoagulantes
  orales Los anticoagulantes..."), which could make citation checks fail.
