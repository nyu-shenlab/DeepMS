# Performance reporting

DeepMS automatically builds a dataset-aware performance report after successful
inference. The implementation is derived from the patient-level workflow in
analysis_final_1016_new.ipynb and is also available as a standalone command for
historical predictions.

## Primary aggregation contract

The notebook-compatible result is intentionally separate from the generic
patient aggregation files.

For each patient:

1. Keep the original scan-level prediction rows.
2. Map structural sequences to FLAIR, T1-CE, or T1-NCE.
3. Average logits directly over all available rows inside each group.
4. Average the available group logits with equal group weight.
5. Apply sigmoid once to the final mean logit.
6. Use a probability threshold of 0.5 for accuracy, sensitivity, and
   specificity.

This direct row-level calculation matters when a patient has repeated rows for
one modality. It is not necessarily equal to first averaging each exact
modality and then averaging modalities.

Both raw logits and the notebook temperature-scaled logits are reported.
Temperature scaling uses the following fixed divisors and zero bias:

| Modality type | Temperature |
| --- | ---: |
| 3D FLAIR | 1.73 |
| 2D FLAIR | 1.82 |
| 3D T1-CE | 1.81 |
| 3D T1-NCE | 1.29 |
| 2D T1-CE | 1.87 |
| 2D T1-NCE | 1.10 |

The report contains FLAIR-only and sMRI results. The primary sMRI contract used
by Internal and Public External includes all notebook FLAIR sequences,
3DT1_NCE, 3DT1_CE, and 2DT1_CE. The Krakow contract additionally permits
2DT1_NCE.

## Report profiles

Every released inference job supplies one explicit report profile.

| Profile | Primary result | Additional cohort behavior |
| --- | --- | --- |
| internal | notebook_primary / sMRI / notebook_temperature | Optional private sidecar can reproduce study-specific exclusions and label corrections. |
| krakow | notebook_primary / sMRI / notebook_temperature | Only patients with 3DFLAIR_NCE are included, matching the notebook. |
| public_external_unmasked | notebook_primary / sMRI / notebook_temperature | Uses the 15-dataset Public External notebook cohort. |
| public_external_masked | masking_comparable / FLAIR / raw | Uses the exact seven-dataset lesion-masking comparison cohort. |
| generic | manifest_all / sMRI / raw | No dataset-specific exclusions. |

All profiles additionally report raw and temperature-scaled FLAIR and sMRI
results for every named aggregate cohort. Per-dataset performance rows
intersect notebook_primary, so excluded CIS cases and out-of-contract datasets
cannot leak into dataset AUCs. dataset_inventory retains the raw manifest counts
alongside notebook-primary and masking-comparable counts, including datasets
whose reportable patient count is zero. The JSON has an explicit `primary`
selector and four ablation selectors: raw and notebook-temperature variants for
both FLAIR and sMRI. Its path-free `prediction_inventory` records the actual
modalities, datasets, scan rows, and source/report patient counts in each
checkpoint run, so a diffusion-map ablation output is self-describing without
exposing clinical image or checkpoint locations. Consumers do not have to infer
the intended row
from table order.

## Public External cohort definitions

The Public External notebook primary cohort contains:

- MSSEG2
- WMH
- ISLES-2022
- open_ms_cross_sectional
- MPI-Leipzig
- MSSEG-2016
- open_ms_longitudinal
- UCSF-PDGM
- MrBrainS18
- OpenNeuro-epilepsy
- PediMS
- QSM
- BraTS_Met
- MS-ISBI
- PediDemi

MSLesSeg remains visible in manifest_all and its own dataset row, but it is not
part of notebook_primary because it was absent from the reference notebook
cohort. The eight legacy CIS identifiers listed by the public profile are
excluded. PediMS uses patient_id as the reporting identifier, matching the
notebook's longitudinal normalization.

The broad notebook WML subgroup contains 11 datasets:

- MSSEG2
- WMH
- ISLES-2022
- open_ms_cross_sectional
- MSSEG-2016
- open_ms_longitudinal
- MrBrainS18
- PediMS
- QSM
- MS-ISBI
- PediDemi

The before/after lesion-masking comparison must use the narrower, identical
seven-dataset masking_comparable cohort:

- WMH
- open_ms_cross_sectional
- MSSEG-2016
- MrBrainS18
- PediMS
- PediDemi
- ISLES-2022

Do not compare masked and unmasked manifest_all results as a lesion-masking
ablation. Use cohort=masking_comparable, ensemble=FLAIR, calibration=raw in both
runs. The patient-level report file allows an additional exact common-patient
join when two runs are compared.

The masked launcher uses masked_image_path when it exists and preprocessing as a
fallback outside the exact comparison cohort. Every contributing FLAIR row in
the seven-dataset masking_comparable primary cohort must use an explicit masked
image; a new inference run fails report generation if any such row falls back.
The report records masked and fallback row counts globally, by dataset, by
cohort, and in the patient-level table. Historical prediction files that lack
row-level source provenance remain reportable, but
`mask_provenance.available` is false. The unmasked launcher uses preprocessing
and never selects masked_image_path. A profile/image-policy mismatch is rejected.

## Metrics and uncertainty

Each report includes:

- patient, positive, negative, and contributing-row counts;
- accuracy, sensitivity, specificity, and confusion-matrix counts at 0.5;
- ROC-AUC;
- trapezoidal PR-AUC;
- average precision;
- deterministic 95% bootstrap confidence intervals;
- standardized partial ROC-AUC at the configured target FPR;
- the best sensitivity and corresponding threshold at or below that FPR.

Aggregate cohorts use 2,000 bootstrap samples and seed 42 by default. Per-dataset
rows contain point estimates but skip bootstrap intervals to keep reporting
fast. One-class datasets retain counts and operating metrics while ROC/PR
quantities that require both classes are recorded as null.

Set DEEPMS_REPORT_BOOTSTRAPS to change the aggregate bootstrap count for Slurm
runs. A value of zero is useful for a fast smoke test.

## Output artifacts

Every immediate, non-deferred inference report contains:

| File | Purpose |
| --- | --- |
| performance_report.json | Full machine-readable contract, cohort definitions, inventory, primary/ablation/masking selectors, provenance checks, and metrics |
| performance_summary.csv | Tidy cohort-by-ensemble-by-calibration metrics, including every dataset |
| performance_report.md | Compact human-readable report |
| prediction_patient_report.csv | Patient predictions and cohort-membership flags for downstream ablation comparisons |

These files are in addition to prediction_all_modalities.csv, the existing
patient aggregation tables, metrics.json, and coverage.json.

With `--defer_performance_report`, inference intentionally skips the four
report files. It still writes `prediction_all_modalities.csv` and a
`coverage.json` completion record containing the profile, image policy,
prediction-row count, and report configuration.

## Rebuild a report without rerunning the model

~~~bash
uv run --locked --no-sync python report_predictions.py \
  --predictions /path/to/prediction_all_modalities.csv \
  --report_profile public_external_unmasked \
  --image_policy preprocessing
~~~

For a private internal cohort sidecar:

~~~bash
uv run --locked --no-sync python report_predictions.py \
  --predictions /path/to/prediction_all_modalities.csv \
  --report_profile internal \
  --cohort_overrides /private/path/internal_report_overrides.csv \
  --image_policy preprocessing
~~~

The optional sidecar has columns m_id, include, and label_override. Only counts
and the sidecar basename are written to the report; its subject IDs are not
copied into public configuration.

## Ablation runs

Every trained ablation checkpoint produces the same scan-level prediction
schema. The dMRI map is a training ablation; the reference comparison performs
structural-only inference, so a diffusion map is not averaged directly into the
final patient prediction.

Each per-run JSON retains all four audit rows:

- `ablation_results.flair_raw`
- `ablation_results.flair_notebook_temperature`
- `ablation_results.smri_raw`
- `ablation_results.smri_notebook_temperature`

The final collector intentionally does not accept an arbitrary selector. It
supports two mutually exclusive scientific contracts:

| Evaluation mode | Selected result | Cohort / ensemble / calibration | Required profiles |
| --- | --- | --- | --- |
| `dataset_calibrated` | `ablation_results.smri_notebook_temperature` | `notebook_primary` / sMRI / `notebook_temperature` | Internal, Krakow, Public External unmasked, or generic |
| `masking_raw` | `masking_comparison` | `masking_comparable` / FLAIR / raw | Exactly paired Public External unmasked and lesion-masked runs |

`dataset_calibrated` applies the fixed modality temperatures recorded in the
reference notebook. They are never estimated on the test set. The same fixed
mapping is applied deterministically to every checkpoint. `masking_raw`
performs no calibration on either side, so the only intended experimental
difference is the lesion-masked versus unmasked image policy.

For one-command operation, `scripts/slurm/ablation/run_diffusion_ablation_pipeline.sbatch` performs
preflight and submits the training, inference, and summary jobs with explicit
`afterok` dependencies. `DEEPMS_PIPELINE_MODE=both` trains once and reuses
the Public External unmasked inference for the calibrated and raw-pair reports.
The ablation array forces `--defer_performance_report`. Submit
`scripts/slurm/ablation/summarize_inference_runs.sbatch` with `afterok` dependencies on all required
inference arrays. The CPU-only job recursively discovers complete runs,
verifies every prediction row count, and recomputes all metrics once. Its
mode-aware default is 12 runs for `dataset_calibrated` and 24 runs for
`masking_raw`.

For dataset evaluation, exact patient/label fingerprints must agree across
checkpoints within each profile and image policy. For masking, fingerprints
must agree across every checkpoint and both image policies; each
`<family>/<map>` must contain exactly one unmasked and one masked run. A
partial collection, duplicate pair, cohort drift, label drift, or mixed
aggregation/calibration contract fails closed.

The final directory contains:

- `ablation_performance_summary.csv`
- `ablation_performance_metrics.csv`
- `ablation_performance_report.json`
- `ablation_performance_report.md`
- `_SUCCESS`, written last before atomic publication

`masking_raw` additionally writes `masking_pairwise_deltas.csv`, with raw
unmasked metrics, raw masked metrics, and `masked - unmasked` deltas for every
comparison unit. The JSON and Markdown reports include the same paired rows.
