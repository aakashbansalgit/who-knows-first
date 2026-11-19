# Archive

Superseded and exploratory code, kept for the record. Nothing here is run by the pipeline or needed to reproduce the paper.

| File | What it was | Replaced by |
| --- | --- | --- |
| `02_registry_api_versions.py` | First version of stage 02, reading trial version histories from the ClinicalTrials.gov website | `pipeline/02_registry_snapshots.py` (AACT monthly snapshots) |
| `03_draft_figures.py` | Draft figures from stage 03 outputs | Not used in the paper |
| `05_option_order_check.py` | Re-asks the language model with the answer options reversed, to measure position bias | Not run for the paper (Section 10.1 notes the possible bias) |
| `10_step14_runup_early_releases.py` | Run-up without early press releases, 5 placebo dates per readout | Step 19 of `pipeline/10_robustness.py` (20 placebo dates) |
