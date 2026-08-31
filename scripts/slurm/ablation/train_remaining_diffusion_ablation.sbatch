#!/usr/bin/env bash
#SBATCH --job-name=deepms-dmri-remaining
#SBATCH --output=slurm-%x-%A_%a-r%r.out
#SBATCH --error=slurm-%x-%A_%a-r%r.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:2
#SBATCH --array=0-1,4-6,8-11%4
#SBATCH --exclude=a100-4011,a100-4012,a100-4015,a100-4024,a100-4028,a100-4033

set -euo pipefail
umask 002

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

# Keep code, logs, and outputs under the clone from which this job was
# submitted. Submit from the intended DeepMS clone root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd -- "${PROJECT_ROOT}" || fail "DeepMS repository is unavailable: ${PROJECT_ROOT}"
PROJECT_ROOT="$(pwd -P)"
[[ -f pyproject.toml && -f uv.lock ]] || \
    fail "Submit from the intended DeepMS repository root: ${PROJECT_ROOT}"

SITE_PROFILE="${DEEPMS_SITE_PROFILE:-${PROJECT_ROOT}/configs/slurm.shenlab.env}"
[[ -f "${SITE_PROFILE}" ]] || fail "Shenlab site profile not found: ${SITE_PROFILE}"
# shellcheck disable=SC1090
source "${SITE_PROFILE}"
[[ "${DEEPMS_PROJECT_ROOT}" == "${PROJECT_ROOT}" ]] || \
    fail "The site profile did not resolve to this DeepMS checkout."

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-}"
case "${TASK_INDEX}" in
    0) DIFFUSION_FAMILY=SMI; DIFFUSION_MAP=Da_smi ;;
    1) DIFFUSION_FAMILY=SMI; DIFFUSION_MAP=DePar_smi ;;
    4) DIFFUSION_FAMILY=SMI; DIFFUSION_MAP=p2_smi ;;
    5) DIFFUSION_FAMILY=DTI; DIFFUSION_MAP=ad_dti ;;
    6) DIFFUSION_FAMILY=DTI; DIFFUSION_MAP=fa_dti ;;
    8) DIFFUSION_FAMILY=DTI; DIFFUSION_MAP=rd_dti ;;
    9) DIFFUSION_FAMILY=DKI; DIFFUSION_MAP=ak_wdki ;;
    10) DIFFUSION_FAMILY=DKI; DIFFUSION_MAP=mk_wdki ;;
    11) DIFFUSION_FAMILY=DKI; DIFFUSION_MAP=rk_wdki ;;
    2) fail "Task 2 (DePerp_smi) is already complete and is excluded." ;;
    3) fail "Task 3 (f_smi) is already complete and is excluded." ;;
    7) fail "Task 7 (md_dti) is already complete and is excluded." ;;
    *)
        fail "Expected one remaining array index in 0,1,4,5,6,8,9,10,11; got ${TASK_INDEX:-unset}."
        ;;
esac

[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]] || \
    fail "SLURM_ARRAY_JOB_ID is missing; submit this file with sbatch."
[[ -z "${DEEPMS_RESUME_CHECKPOINT:-}" ]] || \
    fail "The nine-map rerun cannot use one shared resume checkpoint."

RUN_ID="${DEEPMS_REMAINING_EVAL_ID:-remaining-9maps-${SLURM_ARRAY_JOB_ID}}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    fail "DEEPMS_REMAINING_EVAL_ID must be one safe path component."
RUN_ROOT="${DEEPMS_REMAINING_TRAIN_ROOT:-${PROJECT_ROOT}/outputs/train/diffusion_single_map/${RUN_ID}}"
[[ "${RUN_ROOT}" == /* && "${RUN_ROOT}" != "/" && "${RUN_ROOT}" != "${PROJECT_ROOT}" ]] || \
    fail "DEEPMS_REMAINING_TRAIN_ROOT must be a safe absolute directory."
[[ "${RUN_ROOT}" != *','* && "${RUN_ROOT}" != *$'\n'* ]] || \
    fail "The training output root cannot contain commas or newlines."

TARGET_ROOT="${RUN_ROOT}/${DIFFUSION_FAMILY}/${DIFFUSION_MAP}"
if [[ -e "${TARGET_ROOT}" ]]; then
    [[ -d "${TARGET_ROOT}" ]] || \
        fail "Map output path exists but is not a directory: ${TARGET_ROOT}"
    TARGET_CONTENT="$(find "${TARGET_ROOT}" -mindepth 1 -print -quit)"
    [[ -z "${TARGET_CONTENT}" ]] || \
        fail "This map already has a non-empty output directory: ${TARGET_ROOT}"
fi

MAX_CUDA_REQUEUES="${DEEPMS_CUDA_MAX_REQUEUES:-3}"
[[ "${MAX_CUDA_REQUEUES}" =~ ^[0-9]+$ ]] || \
    fail "DEEPMS_CUDA_MAX_REQUEUES must be a non-negative integer."
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
[[ "${RESTART_COUNT}" =~ ^[0-9]+$ ]] || \
    fail "SLURM_RESTART_COUNT must be a non-negative integer."

export DEEPMS_PROJECT_ROOT="${PROJECT_ROOT}"
export DEEPMS_ABLATION_OUTPUT_ROOT="${RUN_ROOT}"

TRAIN_JOB="${PROJECT_ROOT}/scripts/slurm/ablation/train_diffusion_ablation.sbatch"
[[ -f "${TRAIN_JOB}" ]] || fail "Shared ablation training job not found: ${TRAIN_JOB}"

printf 'Targeted DeepMS diffusion-map rerun\n'
printf '  array task:    %s\n' "${TASK_INDEX}"
printf '  selected map:  %s/%s\n' "${DIFFUSION_FAMILY}" "${DIFFUSION_MAP}"
printf '  excluded maps: DePerp_smi f_smi md_dti\n'
printf '  concurrency:   at most 4 array tasks (2 A100 GPUs each)\n'
printf '  output root:   %s\n' "${RUN_ROOT}"
printf '  bad-node list: a100-4011 a100-4012 a100-4015 a100-4024 a100-4028 a100-4033\n'
printf '  CUDA retries:  restart %s; at most %s requeues\n' \
    "${RESTART_COUNT}" "${MAX_CUDA_REQUEUES}"
printf '  downstream:    training only; no inference or summary jobs\n'

status=0
bash "${TRAIN_JOB}" || status=$?

if (( status == 0 )); then
    exit 0
fi

# The CUDA allocation helper reserves exit 75 for a GPU that Slurm exposed but
# could not initialize. Do not retry training failures, OOMs, or user errors.
if (( status != 75 )); then
    exit "${status}"
fi

if (( RESTART_COUNT >= MAX_CUDA_REQUEUES )); then
    printf 'ERROR: CUDA preflight still fails after %s requeues.\n' \
        "${RESTART_COUNT}" >&2
    exit 75
fi

command -v scontrol >/dev/null 2>&1 || \
    fail "scontrol is unavailable; cannot requeue CUDA preflight failure."
JOB_ELEMENT="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
printf 'CUDA allocation unavailable on %s; requeueing %s (%s/%s).\n' \
    "${SLURMD_NODENAME:-unknown}" \
    "${JOB_ELEMENT}" \
    "$((RESTART_COUNT + 1))" \
    "${MAX_CUDA_REQUEUES}"

if ! scontrol requeue "${JOB_ELEMENT}"; then
    printf 'ERROR: failed to requeue %s.\n' "${JOB_ELEMENT}" >&2
    exit 75
fi
exit 0
