# Full project review and integration

## Improvements

- Corrected false interaction alerts between identical single-ingredient medicines.
- Added per-review caching for repeated interaction searches.
- Increased retrieval depth from 3 to 5.
- Added ATC-supported ingredient-class retrieval.
- Improved quotation verification and corrective retries.
- Introduced request-size controls for long CIMA sections.
- Selected GPT-OSS-20B with medium reasoning.
- Added standalone medicine questions and improved patient-question planning.
- Redesigned the Streamlit interface and reduced startup import overhead.
- Added CIMA source links to reports and standalone answers.
- Improved report scope, missing-evidence disclosure and stale-result protection.
- Integrated the team's offline rules, stored answers and safety fixtures.

## Evaluation

Development set: 15 questions.
- Source retrieval: 13/13.
- Evidence status agreement: 15/15.
- Interaction classification agreement: 7/7.

Held-out set: 5 questions.
- Source retrieval: 3/4 (75%).
- Evidence status agreement: 4/5 (80%).
- Interaction classification agreement: 2/3 (66.67%).
- Generation errors: 0.
- Rejected answers: 0.
- Median runtime: 2.577 seconds.
- Mean runtime: 3.9466 seconds.

H02 failed because the expected evidence was not retrieved.
No further tuning was performed on the held-out cases.

## Integration verification

- All six merge conflicts resolved.
- 235 automated tests passed.
- 15 expected safety-test failures remain.
- 23 medicines with ATC codes.
- 128 indexed CIMA fragments covering all 23 catalogue registrations.
- Offline and live website workflows tested locally.
- Valid and invalid JSON uploads tested before integration.
- PDF and HTML exports tested before integration.

## Limitations

This is an academic clinical decision-support prototype for synthetic patients,
not a validated prescribing system.

Reference labels have not received independent clinical validation.
The evaluation samples are small. Exact quotation verification does not
guarantee that every generated claim is fully supported by its citations.

The 15 expected safety-test failures represent known gaps in detecting
treatment-change instructions.

Historical stored interaction answers lacking interaction classification
require another check. They must not be interpreted as confirmed alerts.

The earlier patient-level evaluator and baseline results are retained for
team review. Its scoring logic has not been validated against the new
interaction response schema.

The RAG benchmark used 99 fragments and 18 registrations, before integration
expanded the corpus to 128 fragments and 23 registrations. The original
local CIMA file hash differs from the historical benchmark manifest.
The actual local hashes are recorded in project_audit_results.json.

Hosted deployment and external source-link navigation require final
verification. Local success does not establish hosted deployment readiness.
