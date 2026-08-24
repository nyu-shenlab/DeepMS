#!/usr/bin/env bash
#SBATCH --job-name=deepms-external
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

command -v uv >/dev/null 2>&1 || fail "uv is not available on PATH."
[[ -f pyproject.toml && -f uv.lock && -f infer.py ]] || \
    fail "Submit from the DeepMS repository root or set DEEPMS_PROJECT_ROOT."
[[ -x .venv/bin/python ]] || \
    fail "Missing .venv. Run: uv sync --locked --no-dev"

: "${DEEPMS_MODEL_PATH:?Set DEEPMS_MODEL_PATH to best_model.pth}"
: "${DEEPMS_EXTERNAL_TEST_CSV:?Set DEEPMS_EXTERNAL_TEST_CSV to the external metadata CSV}"
[[ -f "${DEEPMS_MODEL_PATH}" ]] || fail "Checkpoint not found: ${DEEPMS_MODEL_PATH}"
[[ -f "${DEEPMS_EXTERNAL_TEST_CSV}" ]] || fail "Test CSV not found: ${DEEPMS_EXTERNAL_TEST_CSV}"

OUTPUT_DIR="${DEEPMS_INFERENCE_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/inference/external/job-${SLURM_JOB_ID:-manual}}"
VISUALIZATION_DIR="${DEEPMS_VISUALIZATION_DIR:-${PROJECT_ROOT}/visualizations/external/job-${SLURM_JOB_ID:-manual}}"
NUM_WORKERS="${DEEPMS_NUM_WORKERS:-4}"
MIXED_PRECISION="${DEEPMS_MIXED_PRECISION:-fp16}"

case "${MIXED_PRECISION}" in
    no|fp16|bf16) ;;
    *) fail "DEEPMS_MIXED_PRECISION must be no, fp16, or bf16." ;;
esac
mkdir -p "${OUTPUT_DIR}"

MODALITIES=(2DFLAIR_NCE 2DT1_NCE 3DT1_NCE 3DT1_CE 3DFLAIR_NCE)
IMAGE_ARGS=(--use_mask_img)
if [[ "${DEEPMS_USE_MASKED_IMAGES:-1}" == "0" ]]; then
    IMAGE_ARGS=(--use_preprocess)
fi
VISUALIZATION_ARGS=()
if [[ "${DEEPMS_SAVE_VISUALIZATIONS:-1}" == "1" ]]; then
    mkdir -p "${VISUALIZATION_DIR}"
    VISUALIZATION_ARGS=(--visualization --visualization_dir "${VISUALIZATION_DIR}")
fi
CIS_ARGS=()
if [[ "${DEEPMS_USE_CIS:-0}" == "1" ]]; then
    CIS_ARGS=(--use_cis)
fi

UV_RUN=(uv run --locked --no-sync)

printf 'DeepMS root: %s\n' "${PROJECT_ROOT}"
printf 'Slurm job: %s | inference: single GPU\n' "${SLURM_JOB_ID:-local}"
"${UV_RUN[@]}" python -c \
    'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

srun "${UV_RUN[@]}" python "${PROJECT_ROOT}/infer.py" \
    --modalities "${MODALITIES[@]}" \
    --model_paths "${DEEPMS_MODEL_PATH}" \
    --test_patient_ids "${DEEPMS_EXTERNAL_TEST_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    "${IMAGE_ARGS[@]}" \
    "${VISUALIZATION_ARGS[@]}" \
    --direct_resize \
    --use_global_transform \
    --batch_size "${DEEPMS_BATCH_SIZE:-8}" \
    --num_workers "${NUM_WORKERS}" \
    --num_channels 1 \
    --resize_size 128 \
    --backbone VoCo_Salient_2 \
    --num_experts 1 \
    --gating_kernel_size 1 \
    --num_heads 2 \
    --loss_type bce_with_logits \
    --mixed_precision "${MIXED_PRECISION}" \
    "${CIS_ARGS[@]}"
