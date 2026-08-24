<div align="center">

# Diagnosis of Multiple Sclerosis Using Multimodal Deep Learning Integrating Lesion and Normal-Appearing White Matter

[![RSNA Award](https://img.shields.io/badge/RSNA_2025-Winner-gold?style=for-the-badge&logo=medal)](https://www.rsna.org/research/research-awards/kuo-york-chynn-neuro-research-award)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

<h3>🏆 Winner of the Kuo York Chynn Neuroradiology Research Award at RSNA 2025</h3>

[**Manuscript**](https://www.medrxiv.org/content/10.64898/2026.03.04.26347460v1) • [**Citation**](#citation)

</div>

---

## Overview

Current diagnostic criteria for multiple sclerosis (MS) rely heavily on visible white matter lesions (WMLs), which are often non-specific and can also be observed in many MS mimics. DeepMS is a multimodal deep learning framework designed to improve diagnostic specificity by integrating lesion-related signals with abnormalities in normal-appearing white matter (NAWM).

The model is trained using paired diffusion MRI (dMRI) and structural MRI (sMRI), but is designed to operate on routine structural MRI alone at deployment. This enables DeepMS to extract diagnostically useful NAWM-related information without requiring quantitative MRI in real-world clinical workflows.

Our study includes large-scale model development, multireader comparison with established 2024 McDonald biomarkers, lesion-masking analyses, and international multicentre external validation across diverse public datasets and an independent clinical cohort.

---

## Key Highlights

- **Routine MRI deployment**: trained with multimodal MRI, deployed using routine structural MRI alone
- **Lesion + NAWM integration**: captures both focal lesion burden and latent NAWM-related diagnostic signals
- **Strong generalizability**: validated on an independent Krakow cohort and 15 public external datasets
- **Lesion-masking robustness**: retains diagnostic performance after removal of visible white matter lesions
- **Clinical relevance**: compared against established biomarkers including DIS, DIT, CVS, and PRL

---

## Project Status

We are actively preparing the codebase and release materials for public dissemination.

- [x] Manuscript available on medRxiv
- [x] External public datasets documented
- [x] Preprocessing pipeline
- [x] Model architecture and training framework
- [x] Validation and inference workflow
- [x] Reproducible uv environment and Slurm training/inference jobs

---

## Study Design

Our study aims to improve MS diagnosis from routine MRI by integrating both visible white matter lesions and latent abnormalities in normal-appearing white matter.

![Study Design Diagram](assets/study_design.jpg)

*Figure 1. Overview of the study design, development pipeline, validation strategy, and multimodal learning framework.*

---

## Model Architecture

DeepMS uses a multimodal deep learning architecture designed to learn NAWM-related diagnostic signatures from paired dMRI and sMRI during training, while remaining deployable on routine sMRI alone.

![Model Architecture](assets/model_architecture.png)

*Figure 2. Schematic overview of the DeepMS model architecture.*

---

## Results Snapshot

DeepMS demonstrated robust performance across internal and external evaluations:

- **Internal test cohort**: AUC 0.968
- **Independent Krakow cohort**: AUC 0.940
- **Public multi-site external cohort**: AUC 0.974
- **Lesion-masking analysis**: substantial diagnostic signal retained after all WMLs removal (AUC 0.974)
- **Multireader study**: outperformed established lesion-based biomarkers


![Reader Study](assets/reader_study.png)
*Figure 3. Comparison with established lesion-based biomarkers in reader study.*

These findings support the presence of diagnostically meaningful NAWM-related information in routine structural MRI.

---

## External Datasets

To improve robustness and assess generalizability, we incorporated additional ADNI data during development and evaluated DeepMS across 15 public external datasets spanning both MS and non-MS conditions.

### Multiple Sclerosis Datasets

| Dataset | Subjects (N) | Diagnosis | Source / Access |
| :--- | :---: | :--- | :--- |
| **QSM** | 150 | MS (100) / HC (50) | [Univ. of Bologna](https://zenodo.org/records/10931121) |
| **MSSEG-2016** | 53 | MS | [Inria / MSSEG](https://portal.fli-iam.irisa.fr/msseg-challenge/overview) |
| **Open MS Data** | 50 | MS | [Univ. of Ljubljana](https://github.com/muschellij2/open_ms_data) |
| **MSSEG-2** | 40 | MS | [Inria / MSSEG-2](https://portal.fli-iam.irisa.fr/msseg-2/data/) |
| **MS-ISBI** | 19 | MS | [JHU / IACL](https://smart-stats-tools.org/lesion-challenge) |
| **PediMS** | 9 | Pediatric MS | [Babeș-Bolyai Univ.](https://github.com/DanieleStefano/PediMS-dataset) |

### Non-MS Datasets

To evaluate specificity and out-of-distribution robustness, we included datasets covering other neurological conditions such as Alzheimer's disease, stroke, tumor, epilepsy, white matter lesion burden, and non-MS demyelination.

| Dataset | Subjects (N) | Diagnosis | Source / Access |
| :--- | :---: | :--- | :--- |
| **ADNI (Train only)** | 1,822 | MCI / AD / NC | [ADNI](http://adni.loni.usc.edu/data-samples/data-types/) |
| **UCSF-PDGM** | 501 | Primary Tumor | [TCIA](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) |
| **ISLES-2022** | 250 | Stroke (CVD) | [TU Munich](https://isles22.grand-challenge.org/) |
| **MetsToBrain** | 200 | Metastasis Tumor | [TCIA](https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/) |
| **WMH** | 170 | White Matter Lesions | [WMH Challenge](https://wmh.isi.uu.nl/) |
| **OpenNeuro-epilepsy** | 170 | Epilepsy / HC | [OpenNeuro (ds004199)](https://openneuro.org/datasets/ds004199) |
| **MPI-Leipzig** | 117 | Aged Healthy Control | [OpenNeuro (ds000221)](https://openneuro.org/datasets/ds000221) |
| **MrBrainS18** | 30 | White Matter Lesions | [UMC Utrecht](https://mrbrains18.isi.uu.nl/) |
| **PediDemi** | 13 | Non-MS demyelination | [figshare](https://doi.org/10.6084/m9.figshare.28694435) |

> **Note:** Access to some datasets may require registration, approval, or data use agreements from the original hosting institutions.

---

## Code and Data Availability

The source code for structural MRI preprocessing, model development, training, validation, and inference is publicly available in this repository.

Links and identifiers for all public external datasets used in this study are provided above and in the manuscript.

Internal clinical datasets and trained model weights are not publicly available because of patient privacy protections, institutional regulations, and hospital data governance policies. De-identified access to internal data may be considered upon reasonable request, subject to institutional review, regulatory approval, and execution of any required data use agreements.


---

## Repository Layout

```text
DeepMS/
├── configs/accelerate/       # Reproducible multi-GPU training configuration
├── docs/                     # Data and metadata contracts
├── model/                    # Model architectures
├── preprocessing/            # Structural and diffusion MRI preprocessing
├── scripts/slurm/            # Training and internal/external inference jobs
├── tests/                    # Unit tests and a two-process validation smoke test
├── utils/                    # Dataset, schedule, and aggregation utilities
├── infer.py                  # Single-GPU inference entry point
├── train.py                  # Training entry point
├── pyproject.toml            # Direct and optional dependencies
└── uv.lock                   # Fully resolved Linux x86_64 environment
```

## Getting Started

### Requirements

- Linux x86_64
- an NVIDIA driver compatible with the locked PyTorch 2.6.0 / CUDA 12.4 build
- [uv](https://docs.astral.sh/uv/)
- Slurm for the provided cluster jobs

Clone the repository and create the core training/inference environment:

```bash
git clone https://github.com/nyu-shenlab/DeepMS.git
cd DeepMS
uv sync --locked --no-dev
```

The project pins Python 3.11 and the direct package versions used in the
original `preprocessing` Conda environment. The lockfile also fixes all
transitive dependencies.

Install the optional preprocessing tools, including ANTsPy and HD-BET 2.0.1:

```bash
uv sync --locked --no-dev --extra preprocessing
```

HD-BET downloads its model parameters on first use. To enable Weights & Biases
logging, add the tracking extra:

```bash
uv sync --locked --no-dev --extra tracking
```

Verify the environment without starting a job:

```bash
uv run --locked --no-dev python -c \
  "import torch, monai; print(torch.__version__, torch.version.cuda, monai.__version__)"
uv run --locked --no-dev python train.py --help
uv run --locked --no-dev python infer.py --help
```

### Data preparation

The image-level and patient-level CSV schemas are documented in
[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). Use de-identified identifiers and
keep all clinical metadata and images outside the Git repository.

Structural MRI preprocessing:

```bash
uv run --locked --no-dev --extra preprocessing python \
  preprocessing/Structural_MRI_Preprocessing.py \
  --csv /path/to/structural_metadata.csv \
  --template /path/to/mni_template.nii \
  --tpl_mask /path/to/mni_template_mask.nii \
  --out_dir /path/to/structural_outputs \
  --skip_exist
```

Diffusion-map normalization:

```bash
uv run --locked --no-dev python preprocessing/dmri_preprocessing.py \
  --dataset_path /path/to/dmri_metadata.csv \
  --output_base_path /path/to/dmri_outputs \
  --output_csv_path /path/to/dmri_metadata_processed.csv
```

Run either command with `--help` before processing a new dataset. The
structural workflow is GPU-gated because its default pipeline includes HD-BET.

### Training with Slurm

The released job mirrors the full multimodal, two-GPU training configuration
used by the development repository. Set the five required inputs and submit
from the repository root:

```bash
export DEEPMS_TRAIN_CSV=/path/to/train_images.csv
export DEEPMS_VAL_CSV=/path/to/validation_images.csv
export DEEPMS_DIAGNOSIS_CSV=/path/to/train_diagnoses.csv
export DEEPMS_WM_LESION_CSV=/path/to/white_matter_lesions.csv
export DEEPMS_PRETRAINED_PATH=/path/to/VoComni_B.pt

sbatch scripts/slurm/train.sbatch
```

Optional controls include `DEEPMS_OUTPUT_ROOT`,
`DEEPMS_RESUME_CHECKPOINT`, `DEEPMS_NUM_EPOCHS`,
`DEEPMS_BATCH_SIZE`, `DEEPMS_VAL_BATCH_SIZE`, `DEEPMS_LEARNING_RATE`,
`DEEPMS_MIN_LR`, `DEEPMS_GRADIENT_ACCUMULATION_STEPS`,
`DEEPMS_AUC_METRIC`, `DEEPMS_SEED`, and `DEEPMS_FOLD`. Warmup is disabled
by default. Set an exact `DEEPMS_WARMUP_STEPS`, or set
`DEEPMS_USE_WARMUP=1` with `DEEPMS_WARMUP_EPOCHS`.

The training batch size is global across GPUs and accumulation steps. The
cosine horizon is computed from the prepared per-rank training loader:

```text
updates per epoch = ceil(prepared loader batches / gradient accumulation)
total updates = updates per epoch * configured epochs
```

The scheduler advances after each successful optimizer update—not once per
epoch or once per raw batch—and resumes from the saved update count. Validation
uses both training ranks: predictions are gathered globally, checked for
one-to-one row coverage, and early stopping is synchronized across ranks. See
[docs/DISTRIBUTED_EXECUTION.md](docs/DISTRIBUTED_EXECUTION.md) for the exact
batch, schedule, validation, and aggregation contracts.

The script uses `uv run --locked --no-sync`: create the environment on the
login node before submitting so compute jobs never mutate the environment or
contact package indexes.

### Internal inference with Slurm

```bash
export DEEPMS_MODEL_PATH=/path/to/best_model.pth
export DEEPMS_INTERNAL_TEST_CSV=/path/to/internal_test_images.csv

sbatch scripts/slurm/infer_internal.sbatch
```

The internal job evaluates the complete structural, DTI, SMI, and WDKI
modality set. It requests one GPU and runs inference in one Python process. Set
`DEEPMS_USE_CIS=1` only when label `2` should explicitly be mapped to the
positive class; otherwise non-binary rows are excluded and counted.

### External inference with Slurm

```bash
export DEEPMS_MODEL_PATH=/path/to/best_model.pth
export DEEPMS_EXTERNAL_TEST_CSV=/path/to/external_test_images.csv

sbatch scripts/slurm/infer_external.sbatch
```

The external job uses the structural-only deployment modalities from the
released experiment. Lesion-masked inputs and NIfTI visualizations are enabled
by default to match the reference job. Set
`DEEPMS_USE_MASKED_IMAGES=0` and/or `DEEPMS_SAVE_VISUALIZATIONS=0` to
disable them.

Both inference jobs request one GPU. `DEEPMS_BATCH_SIZE` is the ordinary
inference-loader batch size and `DEEPMS_NUM_WORKERS` is the loader worker
count.

All three jobs accept `DEEPMS_PROJECT_ROOT` when submitted outside the
repository root and `DEEPMS_MIXED_PRECISION` (`no`, `fp16`, or `bf16`). Only
the multi-GPU training job uses `DEEPMS_MASTER_PORT`.

### Tests and preflight checks

Install the development group, then run the full deterministic suite:

```bash
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked ruff check .
bash -n scripts/slurm/*.sbatch
sbatch --test-only scripts/slurm/train.sbatch
sbatch --test-only scripts/slurm/infer_internal.sbatch
sbatch --test-only scripts/slurm/infer_external.sbatch
```

The tests include a real two-process CPU launch that calls the production
`train.validate_model` function and verifies exact global row coverage. Slurm
`--test-only` validates scheduler acceptance without submitting a job; it is
not a substitute for a representative GPU smoke run.

### Outputs

Training creates a timestamped experiment directory under `outputs/train/`
by default, including TensorBoard logs, update-aware resume checkpoints,
`best_model.pth`, and `final_model.pth`.

Inference writes per-modality and combined scan-level predictions plus four
explicit patient-level tables: patient-modality, flat-logit, structural-MRI,
and the training-aligned two-stage multimodal ensemble. `metrics.json` records
metrics at each level, while `coverage.json` records requested, excluded,
missing-modality, predicted-row, and predicted-patient counts. Optional NIfTI
visualizations are written under `visualizations/` using row-specific paths.

Model weights and clinical data are not distributed in this repository. The
VoCo initialization is available from the
[Large-Scale Medical repository](https://github.com/Luffy03/Large-Scale-Medical);
users are responsible for confirming the upstream checkpoint and its terms.

---

## Acknowledgements

We thank the authors of the following repositories for their open-source contributions, which were instrumental to this research:

- **dMRI Preprocessing & Quantitative Maps:** [NYU-DiffusionMRI/DESIGNER-v2](https://github.com/NYU-DiffusionMRI/DESIGNER-v2)
- **Pre-trained Models:** [Luffy03/Large-Scale-Medical](https://github.com/Luffy03/Large-Scale-Medical)
- **Brain Extraction:** [MIC-DKFZ/HD-BET](https://github.com/MIC-DKFZ/HD-BET)

---

## Citation

If you find this work helpful for your research, please cite our manuscript:

```bibtex
@article{Ma2026.03.04.26347460,
  author = {Ma, Jiajian and Stepanov, Valentin and Rui, Wushuang and Chen, Hsuan-Chih and Lis, Maciej and Stanek, Aleksandra and Puto, Tomasz and Lan, Michael and Chen, Jenny and Liu, Timothy and Patel, Roshni and Breen, Matthew and Lee, Matthew and Eikermann-Haerter, Katharina and Shepherd, Timothy M. and Novikov, Dmitry S. and O'Neill, Kimberly A. and Fieremans, Els and Shen, Yiqiu},
  title = {Diagnosis of Multiple Sclerosis Using Multimodal Deep Learning Integrating Lesion and Normal-Appearing White Matter: A Retrospective Study with International Multicentre External Validation},
  elocation-id = {2026.03.04.26347460},
  year = {2026},
  doi = {10.64898/2026.03.04.26347460},
  publisher = {Cold Spring Harbor Laboratory Press},
  journal = {medRxiv},
  url = {https://www.medrxiv.org/content/10.64898/2026.03.04.26347460v1}
}
```

---

## License

DeepMS is released under the [MIT License](LICENSE).
