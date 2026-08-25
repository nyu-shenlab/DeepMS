# Diffusion-map ablation Slurm workflow

This directory contains the jobs that automate the 12-arm single-map dMRI
ablation. Run every command from the DeepMS repository root. The shared
single-run training and inference profiles remain one level above this
directory and are called by these wrappers.

## Recommended: clone and submit the complete workflow

```bash
git clone https://github.com/nyu-shenlab/DeepMS.git
cd DeepMS
uv sync --locked --no-dev

cp configs/slurm.shenlab.env .env
# Other clusters: copy configs/slurm.env.example and replace its placeholders.

set -a
source .env
set +a

sbatch --test-only scripts/slurm/ablation/run_diffusion_ablation_pipeline.sbatch
sbatch scripts/slurm/ablation/run_diffusion_ablation_pipeline.sbatch
```

The Shenlab profile uses the shared GPFS inputs, derives the project root from
the current clone, and generates a timestamped evaluation ID. It does not
contain credentials or clinical records.

For `both`, the required site-local inputs are:

- `DEEPMS_PROJECT_ROOT`
- `DEEPMS_UV_BIN`
- `DEEPMS_TRAIN_CSV`
- `DEEPMS_VAL_CSV`
- `DEEPMS_PRETRAINED_PATH`
- `DEEPMS_PUBLIC_EXTERNAL_TEST_CSV`

The Shenlab and example configurations set `b0` on by default through the training
wrapper. Set `DEEPMS_INCLUDE_B0=0` only for a deliberately strict
T1/FLAIR-only baseline. Keep `.env`, clinical manifests, checkpoints, and
outputs outside Git.

## Jobs

| Script | Role | Resources |
| --- | --- | --- |
| `run_diffusion_ablation_pipeline.sbatch` | Preflight and dependency-graph submission | CPU only |
| `train_diffusion_ablation.sbatch` | 12 map-specific training tasks | 2 GPUs per task |
| `infer_diffusion_ablation.sbatch` | 12 map-specific, structural-only inference tasks per profile | 1 GPU per task |
| `summarize_inference_runs.sbatch` | Complete-run validation and final report | CPU only |

The default `both` graph is:

```text
12 training tasks
  -> 12 Public External unmasked inference tasks
       -> calibrated dataset report
  -> 12 Public External lesion-masked inference tasks
       -> raw paired masking report, after both inference arrays
```

The orchestration job exits after scheduling its children; it does not hold a
GPU while waiting. All downstream jobs use `afterok`. The default array cap is
four concurrent tasks and can be changed with
`DEEPMS_ABLATION_CONCURRENCY=1..12`.

## Monitoring and completion

The submission records every child job ID in:

```text
outputs/slurm/pipelines/<evaluation-id>/pipeline_jobs.env
```

Monitor jobs with `squeue -u "$USER"`. In `both` mode, completion means both
of these markers exist beneath the configured inference root:

```text
<evaluation-id>/public_external_unmasked/summary-dataset_calibrated/_SUCCESS
<evaluation-id>/summary-masking_raw/_SUCCESS
```

The principal tables are `ablation_performance_summary.csv` for calibrated
dataset evaluation and `masking_pairwise_deltas.csv` for the raw masked minus
unmasked comparison.

The pipeline intentionally refuses an existing evaluation output root, an
