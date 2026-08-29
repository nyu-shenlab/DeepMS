#!/usr/bin/env bash
#SBATCH --job-name=deepms-ablation-infer
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%4

set -euo pipefail
umask 002

PROJECT_ROOT="${DEEPMS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

DIFFUSION_MAPS=(
    Da_smi DePar_smi DePerp_smi f_smi p2_smi
    ad_dti fa_dti md_dti rd_dti
    ak_wdki mk_wdki rk_wdki
)
DIFFUSION_FAMILIES=(
    SMI SMI SMI SMI SMI
    DTI DTI DTI DTI
    DKI DKI DKI
)

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-${DEEPMS_ABLATION_INDEX:-}}"
if ! [[ "${TASK_INDEX}" =~ ^[0-9]+$ ]] || (( TASK_INDEX >= ${#DIFFUSION_MAPS[@]} )); then
    printf 'ERROR: ablation index must be between 0 and %d; got %q.\n' \
        "$(( ${#DIFFUSION_MAPS[@]} - 1 ))" "${TASK_INDEX}" >&2
    exit 1
fi
DIFFUSION_MAP="${DIFFUSION_MAPS[TASK_INDEX]}"
DIFFUSION_FAMILY="${DIFFUSION_FAMILIES[TASK_INDEX]}"

CHECKPOINT_ROOT="${DEEPMS_ABLATION_CHECKPOINT_ROOT:-${PROJECT_ROOT}/outputs/train/diffusion_single_map}"
MAP_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}/${DIFFUSION_FAMILY}/${DIFFUSION_MAP}"
[[ -d "${MAP_CHECKPOINT_ROOT}" ]] || \
    fail "Checkpoint directory not found: ${MAP_CHECKPOINT_ROOT}"

mapfile -d '' -t COMPLETION_MANIFESTS < <(
    find "${MAP_CHECKPOINT_ROOT}" -type f -name training_complete.json -print0
)
if (( ${#COMPLETION_MANIFESTS[@]} != 1 )); then
    fail "Expected exactly one completed training attempt under ${MAP_CHECKPOINT_ROOT}; found ${#COMPLETION_MANIFESTS[@]} completion manifests."
fi
COMPLETION_MANIFEST="${COMPLETION_MANIFESTS[0]}"
COMPLETED_ATTEMPT_ROOT="$(dirname "${COMPLETION_MANIFEST}")"
CHECKPOINT="${COMPLETED_ATTEMPT_ROOT}/best_model.pth"
[[ -f "${CHECKPOINT}" ]] || \
    fail "Completed training attempt has no best_model.pth: ${COMPLETED_ATTEMPT_ROOT}"

PROFILE="${DEEPMS_ABLATION_INFER_PROFILE:-public_external_unmasked}"
case "${PROFILE}" in
    internal)
        PROFILE_SCRIPT="infer_internal.sbatch"
        ;;
    krakow)
        PROFILE_SCRIPT="infer_krakow.sbatch"
        ;;
    public_external_unmasked)
        PROFILE_SCRIPT="infer_public_external_unmasked.sbatch"
        ;;
    public_external_masked)
        PROFILE_SCRIPT="infer_public_external_lesion_masked.sbatch"
        ;;
    *)
        fail "Unsupported DEEPMS_ABLATION_INFER_PROFILE: ${PROFILE}"
        ;;
esac

EVALUATION_ID="${DEEPMS_ABLATION_EVAL_ID:-job-${SLURM_ARRAY_JOB_ID:-manual}}"
INFERENCE_ROOT="${DEEPMS_ABLATION_INFERENCE_ROOT:-${PROJECT_ROOT}/outputs/inference/diffusion_single_map}"
RUNS_ROOT="${INFERENCE_ROOT}/${EVALUATION_ID}/${PROFILE}"

export DEEPMS_MODEL_PATH="${CHECKPOINT}"
export DEEPMS_INFERENCE_OUTPUT_DIR="${RUNS_ROOT}/${DIFFUSION_FAMILY}/${DIFFUSION_MAP}"
export DEEPMS_DEFER_PERFORMANCE_REPORT=1

PROFILE_JOB="${PROJECT_ROOT}/scripts/slurm/${PROFILE_SCRIPT}"
[[ -f "${PROFILE_JOB}" ]] || fail "Inference profile job not found: ${PROFILE_JOB}"

printf 'Single-map diffusion ablation inference\n'
printf '  task:       %s/%s\n' "${TASK_INDEX}" "$(( ${#DIFFUSION_MAPS[@]} - 1 ))"
printf '  family:     %s\n' "${DIFFUSION_FAMILY}"
printf '  map:        %s\n' "${DIFFUSION_MAP}"
printf '  profile:    %s\n' "${PROFILE}"
printf '  checkpoint: %s\n' "${DEEPMS_MODEL_PATH}"
printf '  completion: %s\n' "${COMPLETION_MANIFEST}"
printf '  output:     %s\n' "${DEEPMS_INFERENCE_OUTPUT_DIR}"
printf '  final root: %s\n' "${RUNS_ROOT}"

exec bash "${PROFILE_JOB}"
