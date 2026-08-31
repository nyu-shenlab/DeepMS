#!/usr/bin/env bash
#SBATCH --job-name=deepms-public-ext-mask
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:1

set -euo pipefail
umask 002
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${DEEPMS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -f infer.py ]] || \
    fail "Submit from the DeepMS repository root or set DEEPMS_PROJECT_ROOT."
RUNTIME_ENVIRONMENT="${PROJECT_ROOT}/scripts/slurm/runtime_environment.sh"
[[ -f "${RUNTIME_ENVIRONMENT}" ]] || fail "Runtime environment helper not found: ${RUNTIME_ENVIRONMENT}"
# shellcheck disable=SC1090
source "${RUNTIME_ENVIRONMENT}"
deepms_verify_guarded_source_tree "${PROJECT_ROOT}"
deepms_activate_locked_environment "${PROJECT_ROOT}"

: "${DEEPMS_MODEL_PATH:?Set DEEPMS_MODEL_PATH to best_model.pth}"
: "${DEEPMS_PUBLIC_EXTERNAL_TEST_CSV:?Set DEEPMS_PUBLIC_EXTERNAL_TEST_CSV to the public External lesion-masked metadata CSV}"
[[ -f "${DEEPMS_MODEL_PATH}" ]] || fail "Checkpoint not found: ${DEEPMS_MODEL_PATH}"
[[ -f "${DEEPMS_PUBLIC_EXTERNAL_TEST_CSV}" ]] || \
    fail "Test CSV not found: ${DEEPMS_PUBLIC_EXTERNAL_TEST_CSV}"

OUTPUT_DIR="${DEEPMS_INFERENCE_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/inference/public-external-lesion-masked/job-${SLURM_JOB_ID:-manual}}"
VISUALIZATION_DIR="${DEEPMS_VISUALIZATION_DIR:-${PROJECT_ROOT}/visualizations/public-external-lesion-masked/job-${SLURM_JOB_ID:-manual}}"
NUM_WORKERS="${DEEPMS_NUM_WORKERS:-4}"
BATCH_SIZE="${DEEPMS_INFERENCE_BATCH_SIZE:-8}"
MIXED_PRECISION="${DEEPMS_MIXED_PRECISION:-fp16}"

[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
    fail "DEEPMS_INFERENCE_BATCH_SIZE must be a positive integer."

case "${MIXED_PRECISION}" in
    no|fp16|bf16) ;;
    *) fail "DEEPMS_MIXED_PRECISION must be no, fp16, or bf16." ;;
esac

MODALITIES=(2DFLAIR_NCE 2DT1_NCE 3DT1_NCE 3DT1_CE 3DFLAIR_NCE)

CIS_ARGS=()
case "${DEEPMS_USE_CIS:-0}" in
    1) CIS_ARGS=(--use_cis) ;;
    0) ;;
    *) fail "DEEPMS_USE_CIS must be 0 or 1." ;;
esac

VISUALIZATION_ARGS=()
case "${DEEPMS_SAVE_VISUALIZATIONS:-0}" in
    1)
        VISUALIZATION_ARGS=(--visualization --visualization_dir "${VISUALIZATION_DIR}")
        ;;
    0) ;;
    *) fail "DEEPMS_SAVE_VISUALIZATIONS must be 0 or 1." ;;
esac

REPORT_ARGS=()
case "${DEEPMS_DEFER_PERFORMANCE_REPORT:-0}" in
    1) REPORT_ARGS=(--defer_performance_report) ;;
    0) ;;
    *) fail "DEEPMS_DEFER_PERFORMANCE_REPORT must be 0 or 1." ;;
esac

INFER_COMMAND=(
    "${UV_RUN[@]}" python "${PROJECT_ROOT}/infer.py"
    --modalities "${MODALITIES[@]}"
    --model_paths "${DEEPMS_MODEL_PATH}"
    --test_patient_ids "${DEEPMS_PUBLIC_EXTERNAL_TEST_CSV}"
    --output_dir "${OUTPUT_DIR}"
    --use_mask_img
    --direct_resize
    --use_global_transform
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --num_channels 1
    --resize_size 128
    --backbone VoCo_Salient_2
    --num_experts 1
    --gating_kernel_size 1
    --num_heads 2
    --loss_type bce_with_logits
    --mixed_precision "${MIXED_PRECISION}"
    --report_profile public_external_masked
    --report_bootstrap_samples "${DEEPMS_REPORT_BOOTSTRAPS:-2000}"
    "${REPORT_ARGS[@]}"
    "${VISUALIZATION_ARGS[@]}"
    "${CIS_ARGS[@]}"
)

printf 'Inference profile: Public External / lesion-masked\n'
printf 'Image policy: masked_image_path, then preprocessing fallback when absent\n'
printf 'Modalities: %s\n' "${MODALITIES[*]}"
printf 'Test CSV: %s\n' "${DEEPMS_PUBLIC_EXTERNAL_TEST_CSV}"
printf 'Batch size: %s\n' "${BATCH_SIZE}"
printf 'Output: %s\n' "${OUTPUT_DIR}"

case "${DEEPMS_PREFLIGHT_ONLY:-0}" in
    1)
        "${INFER_COMMAND[@]}" --preflight_only
        exit 0
        ;;
    0) ;;
    *) fail "DEEPMS_PREFLIGHT_ONLY must be 0 or 1." ;;
esac

mkdir -p "${OUTPUT_DIR}"
if [[ "${DEEPMS_SAVE_VISUALIZATIONS:-0}" == "1" ]]; then
    mkdir -p "${VISUALIZATION_DIR}"
fi

printf 'Slurm job: %s | inference: single GPU\n' "${SLURM_JOB_ID:-local}"
CUDA_PREFLIGHT="${PROJECT_ROOT}/scripts/check_cuda_devices.py"
[[ -f "${CUDA_PREFLIGHT}" ]] || fail "CUDA preflight helper not found: ${CUDA_PREFLIGHT}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi
"${UV_RUN[@]}" python "${CUDA_PREFLIGHT}" --expected 1

srun "${INFER_COMMAND[@]}"
