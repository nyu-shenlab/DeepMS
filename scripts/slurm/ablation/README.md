# Diffusion-map ablation Slurm workflow

This directory contains the jobs that automate the 12-arm single-map dMRI
ablation. Run every command from the DeepMS repository root. The shared
single-run training and inference profiles remain one level above this
directory and are called by these wrappers.

## Recommended: checked one-command Shenlab submission

```bash
git clone https://github.com/nyu-shenlab/DeepMS.git
cd DeepMS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Start a new shell if `uv` is not immediately available on PATH.
./scripts/bootstrap_env.sh

# Refreshes origin and checks the exact scheduler requests without creating jobs.
./scripts/slurm/ablation/guarded_shenlab_ablation.sh

# Submit the complete graph and wait until all child jobs are recorded in Slurm.
./scripts/slurm/ablation/guarded_shenlab_ablation.sh --submit
```

The default command checks the clean Git/upstream state, locked `uv`
environment, input files, safe worker/GPU/concurrency policy, required bad-node
exclusions, script syntax, and fresh output roots. It then runs Slurm's
`sbatch --test-only` against the CPU launcher, training array, inference array,
and summary request. These scheduler checks create no jobs.

`--submit` creates one lightweight CPU launcher and waits for it to finish
constructing the existing full graph:

```text
12-task training array
  -> task-correlated unmasked and masked inference arrays
       -> calibrated and masking summaries
```

Every child job is first submitted in a held state. Only after every job ID and
dependency is valid are downstream jobs released, followed by training last.
If graph construction fails, only the held jobs created by that launcher are
cancelled. `aftercorr` keeps each inference task paired with its training task;
`afterok` and `--kill-on-invalid-dep=yes` prevent failed upstream work from
leaving unusable downstream jobs pending.

The launcher generates a timestamp/PID/random run ID. Outputs, state, log job
names, and every child job are run-specific. The exact clean Git commit is
rechecked inside every allocation, inherited fixed distributed ports are
removed, known bad nodes are excluded, and automatic requeue is disabled. The
state file records the run and every child job ID at:

```text
outputs/slurm/pipelines/<evaluation-id>/pipeline_jobs.env
```

## Manual immediate launcher (advanced)

The older direct launcher remains available for deliberate manual operation,
but it bypasses the Git freshness, scheduler request, and graph receipt checks:

```bash
sbatch --test-only scripts/slurm/ablation/submit_shenlab_ablation.sbatch
sbatch scripts/slurm/ablation/submit_shenlab_ablation.sbatch
```

Both launchers load the committed Shenlab profile, use the shared GPFS inputs,
resolve the `uv` installed by the current user, and derive the project root from
the current clone. They do not contain credentials or clinical records. Slurm
jobs validate the locked environment without installing packages.

## Targeted training-only rerun of the remaining nine maps

The Shenlab recovery job below omits the three completed maps (`DePerp_smi`,
`f_smi`, and `md_dti`) and trains only array indices
`0,1,4,5,6,8,9,10,11`. It is pinned to the shared canonical checkout, so a
collaborator can use the absolute command from any working directory:

```bash
sbatch --test-only /gpfs/data/shenlab/Jiajian/MS_Project/code/DeepMS/scripts/slurm/ablation/train_remaining_diffusion_ablation.sbatch
sbatch /gpfs/data/shenlab/Jiajian/MS_Project/code/DeepMS/scripts/slurm/ablation/train_remaining_diffusion_ablation.sbatch
```

The array runs at most four tasks concurrently, with two A100 GPUs per task.
Every submission writes to its own
`outputs/train/diffusion_single_map/remaining-9maps-<array-job-id>/` root, so
simultaneous submissions do not share model directories. The runtime guard
also rejects indices 2, 3, and 7 if someone overrides the array on the command
line. This job performs training only; it creates no inference or summary jobs.

On another cluster, load a site-local copy of the portable template instead:

```bash
cp configs/slurm.env.example .env
# Replace every placeholder with an absolute path available on that cluster.
set -a
source .env
set +a

sbatch --test-only scripts/slurm/ablation/run_diffusion_ablation_pipeline.sbatch
sbatch scripts/slurm/ablation/run_diffusion_ablation_pipeline.sbatch
```

For `both`, the required site-local inputs are:

- `DEEPMS_PROJECT_ROOT`
- `DEEPMS_UV_BIN`
- `DEEPMS_TRAIN_CSV`
- `DEEPMS_VAL_CSV`
- `DEEPMS_PRETRAINED_PATH`
- `DEEPMS_PUBLIC_EXTERNAL_TEST_CSV`

The Shenlab and example configurations set `b0` on by default through the
training wrapper. Set `DEEPMS_INCLUDE_B0=0` only for a deliberately strict
T1/FLAIR-only baseline. Keep `.env`, clinical manifests, checkpoints, and
outputs outside Git.

Training and validation worker pools are configured separately. Training keeps
its workers persistent for throughput, while validation defaults to
`DEEPMS_VAL_NUM_WORKERS=0`; one validation loader exists per modality, so
keeping all of those worker pools resident can exhaust host memory.

## Jobs

| Script | Role | Resources |
| --- | --- | --- |
| `guarded_shenlab_ablation.sh` | Git/environment/scheduler checks; explicit direct graph submission | Login node, then one CPU launcher |
| `submit_shenlab_ablation.sbatch` | Direct Shenlab profile loading and complete workflow submission | CPU only |
| `run_diffusion_ablation_pipeline.sbatch` | Preflight and dependency-graph submission | CPU only |
| `train_diffusion_ablation.sbatch` | 12 map-specific training tasks | 2 GPUs per task |
| `train_remaining_diffusion_ablation.sbatch` | Shenlab-only rerun of the 9 incomplete maps | 2 GPUs per task, at most 4 tasks |
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
GPU while waiting. Each inference array uses `aftercorr` so an element starts
only after the matching training element succeeds. Summary jobs still use
`afterok` and require every expected result. Invalid dependencies are cancelled
instead of remaining pending indefinitely. The default array cap is four
concurrent tasks and can be changed with `DEEPMS_ABLATION_CONCURRENCY=1..12`.

`DEEPMS_TRAIN_EXCLUDE_NODES` accepts a comma-separated, site-local list of
training nodes with known GPU health problems. The Shenlab profile currently
excludes `a100-4011,a100-4012,a100-4024,a100-4033`; every training allocation also probes both
visible CUDA devices before loading data. The `a100_dev` inference partition
does not contain those excluded training nodes, and every inference allocation
runs the same context, compute, synchronization, and PCI-inventory probe on its
single visible GPU. Do not use `--exclusive` as a substitute for these health
checks: a two-GPU task would otherwise reserve an entire four-GPU node.
Application failures are not automatically requeued. After a failed scientific
run, correct the cause and use a new evaluation ID rather than writing into a
partially populated output root. Inference also requires the atomic
`training_complete.json` written at a successful training exit; a checkpoint
left behind by an interrupted or OOM-killed run is not accepted.

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
