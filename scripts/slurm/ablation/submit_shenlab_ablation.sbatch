#!/usr/bin/env bash
#SBATCH --job-name=deepms-shenlab-submit
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=cpu_short

set -euo pipefail
umask 002

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -n "${SLURM_JOB_ID:-}" ]] || \
    fail "Submit this launcher with sbatch from the DeepMS repository root."

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_ROOT}"
[[ -f pyproject.toml && -f uv.lock ]] || \
    fail "Submit from the DeepMS repository root."

SITE_PROFILE="${DEEPMS_SITE_PROFILE:-${PROJECT_ROOT}/configs/slurm.shenlab.env}"
[[ -f "${SITE_PROFILE}" ]] || fail "Shenlab site profile not found: ${SITE_PROFILE}"

# The committed profile contains the verified shared-data paths and derives
# DEEPMS_PROJECT_ROOT from this clone's current working directory.
# shellcheck disable=SC1090
source "${SITE_PROFILE}"
[[ "${DEEPMS_PROJECT_ROOT}" == "$(pwd -P)" ]] || \
    fail "The site profile did not resolve to this DeepMS clone."

PIPELINE_JOB="${PROJECT_ROOT}/scripts/slurm/ablation/run_diffusion_ablation_pipeline.sbatch"
[[ -f "${PIPELINE_JOB}" ]] || fail "Pipeline job not found: ${PIPELINE_JOB}"

printf 'Loaded Shenlab profile: %s\n' "${SITE_PROFILE}"
printf 'Evaluation ID: %s\n' "${DEEPMS_ABLATION_EVAL_ID}"
printf 'Launching the complete train -> infer -> summary dependency graph.\n'

exec bash "${PIPELINE_JOB}"
