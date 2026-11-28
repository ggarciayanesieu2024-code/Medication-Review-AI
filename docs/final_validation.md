# Final project validation

## Implementation

- Shared bilingual safety screening across RAG, reports and evaluation.
- Verified ATC class-aware retrieval and generation.
- Correct interaction classification in patient evaluation.
- Expanded CIMA coverage: 23 medicines and 128 fragments.
- Notebook defaults aligned with the optimized retrieval configuration.
- Validated stored interaction answers for hosted demonstrations.

## Final results

| Measure | Result |
| --- | --- |
| Automated tests | 266 passed, zero failures |
| Existing unsafe examples detected | 29/29 |
| Existing safe examples preserved | 14/14 |
| Stored searches completed | 414/435 (95.2%) |
| Stored quotations verified | 27/27 |
| Patients passing all labelled checks | 5/6 (83.3%) |
| Duplicate detection | 5/5 |
| Interaction evaluation | 4/5 |
| Cascade evaluation | 1/1 |
| Prescriber grouping | 2/2 |
| Patient safety screening | 6/6 |
| Offline notebook execution | 11/11 code cells |
| Development source retrieval | 13/13 labelled cases |
| Hosted-mode simulation | Passed |

## Limitations

- Groq rate limits prevented completion of 21 stored searches.
- Missing searches are reported as unchecked, not as evidence of safety.
- Completed searches include one rejected response and 391 not-found outcomes.
- Unaffected stored answers were reused across retrieval revisions.
- E5 remains unsuccessful against the original evaluation labels.
- E6 passed its labelled checks despite two incomplete searches.
- Grouping evaluation checks input-derived groups rather than independently scoring rendered reports.
- Safety examples were used during development and are not an independent clinical safety benchmark.
- Exact quotation verification does not guarantee complete clinical interpretation.
- Earlier benchmark results refer to their original source snapshots and configurations.
- Live notebook generation after the final changes remains unverified.
- Actual Streamlit Cloud deployment remains to be verified.
- This is an academic prototype using synthetic patients, not a clinically validated medical device.

The original evaluation labels and earlier benchmark results have been preserved.

## Hosted deployment

Repository entry point: app_deploy/streamlit_app.py

Stored-answer mode supports rule-based reviews, cached interaction findings and report exports without a Groq API key or local vector index.

Live medicine questions require an available model, configured API access and prepared retrieval resources.

See final_validation_results.json for the detailed results.
