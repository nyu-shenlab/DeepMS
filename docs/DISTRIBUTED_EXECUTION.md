# Multi-GPU training and single-GPU inference

DeepMS uses Hugging Face Accelerate for single-node data-parallel training. The
released training job requests two GPUs and starts one launcher task;
Accelerate then creates one Python process per GPU. The released internal and
external inference jobs intentionally request one GPU and run `infer.py`
directly in one Python process.

## Training batch and learning-rate contract

`--batch_size` is the effective global training batch size:

```text
global batch = per-rank microbatch * number of processes * gradient accumulation steps
```

The value must be divisible by `number of processes * gradient accumulation
steps`. `--val_batch_size` is a per-rank validation-loader batch size. The
single-process inference `--batch_size` is its ordinary loader batch size.

The warmup/cosine schedule is indexed by successful optimizer updates, not by
epochs or raw dataloader batches. An update is successful only when gradient
accumulation reaches a synchronization boundary and AMP does not skip the
optimizer step. The scheduler is called after that optimizer step, so the
current learning rate is the one used by the current update.

The planned horizon is calculated after Accelerate prepares and shards the
training loader:

```text
updates per epoch = ceil(per-rank prepared loader batches / gradient accumulation)
total updates = updates per epoch * configured epochs
```

Early stopping can end training before this planned horizon, but it does not
silently redefine the cosine curve.

Without warmup, update zero uses `--lr`. With warmup, updates start at
`--warmup_start_lr`, reach `--lr`, and then enter cosine decay.
`--warmup_steps` is exact and overrides the epoch-derived value selected by
`--use_warmup --warmup_epochs`. The cosine phase reaches `--min_lr` when it
contains at least two planned updates; a degenerate one-update phase uses the
base LR instead of jumping directly from warmup to the minimum.

Resume checkpoints contain the completed update count and the complete
scheduler state. A resume fails clearly if schedule-defining values such as
total updates, warmup updates, or base learning rates differ from the saved
state; this prevents a silent change in the LR trajectory.

## Distributed validation

Accelerate is the only component that shards validation loaders. Every selected
validation row receives a stable `row_id`. Predictions and identifiers are
gathered across all ranks, final-batch padding is removed with
`gather_for_metrics`, and rank zero verifies that every expected row appears
exactly once before calculating metrics.

Early stopping is decided on rank zero and reduced to every rank before any
process exits the epoch loop. This prevents one rank from stopping while other
ranks continue into a collective operation.

Validation reports four checkpoint-selection choices:

- `micro`: pooled scan-level AUC
- `macro`: unweighted mean of available modality AUCs
- `hierarchical`: modality-family hierarchy AUC
- `ensemble`: patient-level two-stage sMRI/dMRI probability ensemble

## Inference execution

The released inference jobs intentionally use one GPU; multi-GPU inference is
not required for the documented workflow. `infer.py` still assigns stable row
IDs, verifies exact prediction coverage, keeps patient/modality aggregation
explicit, and has one writer for CSV and JSON outputs. Visualization paths
include the stable row ID to prevent multiple scans from overwriting one
another.

## Aggregation contract

Inference preserves several explicitly named levels instead of reporting one
ambiguous "ensemble":

1. scan-level predictions;
2. mean prediction within each patient-modality pair;
3. flat mean of patient-modality logits;
4. grouped structural-MRI logits; and
5. the training-aligned two-stage probability ensemble, which first balances
   modality families within dMRI and then equally averages available sMRI and
   dMRI branches.

Repeated scans therefore do not receive extra weight merely because a patient
has more acquisitions of one modality.

## Verification

Run the deterministic test suite on a login or CPU compute node:

```bash
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked ruff check .
bash -n scripts/slurm/*.sbatch
```

The pytest suite includes a two-process CPU launch that directly exercises the
production `train.validate_model` function, including padded validation
gathering and exact global row coverage. Single-process inference tests cover
its aggregation and FP32 accumulation behavior. `sbatch --test-only` checks
Slurm directive acceptance, but it does not load MRI data, initialize a GPU
model, or prove that a full training job completes. A real end-to-end GPU smoke
job with representative metadata and a checkpoint remains the final
cluster-specific acceptance test.
