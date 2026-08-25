# Data and metadata contract

DeepMS uses one row per image acquisition. Repeated acquisitions for the same
patient and modality are allowed. The generic inference hierarchy first averages
within exact patient-modality pairs; the notebook-compatible performance report
instead averages original scan rows directly within the FLAIR, T1-CE, and T1-NCE
groups to reproduce the published analysis. Image files are not discovered
recursively: every image used by preprocessing, training, or inference must be
listed explicitly in a CSV. Absolute image paths are recommended on clusters.

Do not commit clinical metadata, patient identifiers, images, or derived
patient-level outputs to this repository.

## Shared conventions

- `m_id`: de-identified case identifier, parsed as a string so leading zeros
  are preserved.
- `modality`: one of the modality names recognized by the code.
- `ms` or `label`: binary target, where `1` is MS and `0` is non-MS. When both
  columns are supplied for inference, they must agree.
- `Age`: numeric age.
- `Sex`: `M` or `F`.
- `non-preprocessing`: source NIfTI path.
- `bet`: skull-stripped NIfTI path.
- `preprocessing`: registered/preprocessed NIfTI path.
- `masked_image_path`: optional lesion-masked NIfTI path.

Paths may contain `.nii` or `.nii.gz` files as supported by MONAI and
NiBabel. The CSV itself is not copied into an output directory.

## Training inputs

The active training Slurm job requires two image-level CSV manifests.

### Image-level training CSV

Required columns:

| Column | Meaning |
| --- | --- |
| `m_id` | De-identified case identifier |
| `modality` | Image modality |
| `ms` | Binary diagnosis target |
| `Age` | Numeric age |
| `Sex` | `M` or `F` |
| `source` | Dataset or cohort provenance label |
| `bet` | Skull-stripped image path used by the released training recipe |

The optional `label` column may be supplied as an alias for `ms`. Supply
`label` in both training and validation CSVs, or omit it from both.

### Image-level validation CSV

Required columns are `m_id`, `modality`, `ms`, `Age`, `Sex`, and
`preprocessing`. The released training recipe validates on the
`preprocessing` path even though training uses `bet`.

### Legacy auxiliary CSV options

Older development code accepted a patient-level diagnosis CSV through
`--train_diagnosis_df` and a patient-level lesion CSV through
`--white_matter_list`. They produced the `important_diagnosis` and
`white_matter_lesion` metadata fields, respectively. Neither field is consumed
by the active model, loss, sampler, or validation code.

The two command-line options remain accepted for compatibility but are ignored;
the released Slurm jobs do not require either CSV. In particular, do not copy or
publish the original clinical tables merely to run training.

## Inference input

Each released inference profile accepts one image-level CSV. It must contain:

- `m_id`
- `modality`
- either `label` or `ms`
- the path column selected by the command:
  - `preprocessing` for `--use_preprocess`
  - `bet` for `--use_bet_only`
  - `masked_image_path` for `--use_mask_img`
  - `non-preprocessing` when none of those flags is used

When `--use_mask_img` is selected, the image policy is explicit:
`masked_image_path` is preferred row by row and `preprocessing` is used
only when that row has no masked image. The code never labels a fallback row as
an explicitly masked row. `coverage.json` records
`explicit_masked_image_rows`, `preprocessing_fallback_rows`,
`masked_image_column_present`, and the resolved `image_policy`.

If a masked path is supplied, the visualization workflow looks for
`lesion_mask_new.nii.gz` and then `lesion_mask.nii.gz` in the same
directory.

Inference preserves the source CSV position as `source_row` and assigns a
contiguous `row_id` after modality, label, and image-path filtering. By default,
rows outside labels `0` and `1` are excluded. `--use_cis` explicitly maps label
`2` to positive; the released Slurm jobs expose this as `DEEPMS_USE_CIS=1`.
Missing requested modalities are reported rather than silently fabricated.

The single inference process writes outputs after exact row-coverage validation:

- `prediction_<modality>.csv`: scan-level output for one modality
- `prediction_all_modalities.csv`: all scan-level predictions
- `prediction_patient_modality.csv`: repeated scans averaged within modality
- `prediction_patient_flat_logit.csv`: flat patient-modality logit average
- `prediction_patient_smri.csv`: grouped structural-MRI output
- `prediction_patient_multimodal.csv`: training-aligned two-stage sMRI/dMRI
  probability ensemble
- `coverage.json`: filtering, modality availability, and prediction coverage
- `metrics.json`: scan- and patient-level generic metric summaries
- `performance_report.json`: notebook-compatible primary, ablation, masking,
  cohort, provenance, and metric contract
- `performance_summary.csv`: tidy aggregate- and dataset-level metrics
- `performance_report.md`: compact human-readable summary
- `prediction_patient_report.csv`: patient predictions with cohort flags


When `--defer_performance_report` is active, the final four report files are
omitted from each GPU run. `coverage.json` then records
`inference_complete`, `performance_report_status`, the report profile,
image policy, prediction-row count, and shared final-report configuration.
`summarize_inference_runs.py` consumes these completed scan-level outputs.
`dataset_calibrated` selects fixed notebook-temperature sMRI results;
`masking_raw` requires paired Public External profiles and selects raw FLAIR
results on the identical seven-dataset masking cohort.
The performance report uses only an allowlist of non-path manifest metadata;
image paths are not copied into prediction artifacts. See
[PERFORMANCE_REPORTING.md](PERFORMANCE_REPORTING.md) for the exact aggregation,
calibration, Public External dataset, and masked/unmasked contracts.

## Structural MRI preprocessing input

`preprocessing/Structural_MRI_Preprocessing.py` requires:

| Column | Requirement |
| --- | --- |
| `m_id` or `patient_id` | Case identifier selected by `--id_col` |
| `modality` | Structural modality |
| `non-preprocessing` | Source NIfTI path |
| `dataset` | Optional; required only when using `--dataset` |
| `mask_path` | Optional lesion-mask path used with `--process_mask` |

The generated manifest uses the canonical downstream column names `bet` and
`preprocessing`, so it can be merged into the training or inference metadata
contract.

HD-BET 2.0.1 expects a reoriented 3D NIfTI file and writes the brain-extracted
image to the requested output filename. With `--save_bet_mask`, its mask uses
the `*_bet.nii.gz` suffix. The DeepMS wrapper handles those v2 names.

## Diffusion MRI preprocessing input

`preprocessing/dmri_preprocessing.py` requires `m_id`, `modality`, and
`preprocessing`. It writes normalized quantitative maps under
`<output_base>/<m_id>/processed_params/` and produces an updated metadata CSV.

## Modalities in the released jobs

The training and internal-inference jobs use:

- structural/b0: `2DT1_NCE`, `2DT1_CE`, `2DFLAIR_NCE`,
  `2DFLAIR_CE`, `3DT1_NCE`, `3DT1_CE`, `3DFLAIR_NCE`,
  `3DFLAIR_CE`, `b0`
- DTI: `ad_dti`, `fa_dti`, `md_dti`, `rd_dti`
- SMI: `Da_smi`, `DePar_smi`, `DePerp_smi`, `f_smi`, `p2_smi`
- WDKI: `ak_wdki`, `mk_wdki`, `rk_wdki`

The released inference profiles are intentionally disjoint:

| Profile | Manifest variable | Modalities | Image policy |
| --- | --- | --- | --- |
| Internal | `DEEPMS_INTERNAL_TEST_CSV` | Full structural, DTI, SMI, and WDKI set above | `preprocessing` |
| Krakow/UJ | `DEEPMS_KRAKOW_TEST_CSV` | `3DT1_NCE`, `3DT1_CE`, `3DFLAIR_NCE` | `preprocessing` |
| Public External, unmasked | `DEEPMS_PUBLIC_EXTERNAL_TEST_CSV` | `2DFLAIR_NCE`, `2DT1_NCE`, `3DT1_NCE`, `3DT1_CE`, `3DFLAIR_NCE` | `preprocessing` |
| Public External, lesion-masked | `DEEPMS_PUBLIC_EXTERNAL_TEST_CSV` | Same Public External modalities | `masked_image_path`, then per-row `preprocessing` fallback |

The two Public External launchers deliberately use the same manifest,
checkpoint, modalities, and aggregation code. The unmasked profile passes
`--use_preprocess`; the lesion-masked profile passes `--use_mask_img`.
Their output directories are separate.
