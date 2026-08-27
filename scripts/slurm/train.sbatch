#!/usr/bin/env bash
#SBATCH --job-name=deepms-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:2

set -euo pipefail
umask 002

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_ROOT="${DEEPMS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -f train.py ]] || \
    fail "Submit from the DeepMS repository root or set DEEPMS_PROJECT_ROOT."
RUNTIME_ENVIRONMENT="${PROJECT_ROOT}/scripts/slurm/runtime_environment.sh"
[[ -f "${RUNTIME_ENVIRONMENT}" ]] || fail "Runtime environment helper not found: ${RUNTIME_ENVIRONMENT}"
# shellcheck disable=SC1090
source "${RUNTIME_ENVIRONMENT}"
deepms_activate_locked_environment "${PROJECT_ROOT}"

REQUIRED_VARS=(
    DEEPMS_TRAIN_CSV
    DEEPMS_VAL_CSV
    DEEPMS_PRETRAINED_PATH
)
for name in "${REQUIRED_VARS[@]}"; do
    [[ -n "${!name:-}" ]] || fail "Required environment variable ${name} is not set."
    [[ -f "${!name}" ]] || fail "${name} does not point to a file: ${!name}"
done

OUTPUT_ROOT="${DEEPMS_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/train}"
NUM_EPOCHS="${DEEPMS_NUM_EPOCHS:-40}"
EARLY_STOPPING_EPOCHS="${DEEPMS_EARLY_STOPPING_EPOCHS:-5}"
VAL_INTERVAL="${DEEPMS_VAL_INTERVAL:-1}"
SAVE_INTERVAL="${DEEPMS_SAVE_INTERVAL:-5}"
BATCH_SIZE="${DEEPMS_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${DEEPMS_VAL_BATCH_SIZE:-16}"
NUM_WORKERS="${DEEPMS_NUM_WORKERS:-4}"
LEARNING_RATE="${DEEPMS_LEARNING_RATE:-1e-4}"
MIN_LR="${DEEPMS_MIN_LR:-1e-6}"
WARMUP_START_LR="${DEEPMS_WARMUP_START_LR:-1e-6}"
MIXED_PRECISION="${DEEPMS_MIXED_PRECISION:-fp16}"
GRADIENT_ACCUMULATION_STEPS="${DEEPMS_GRADIENT_ACCUMULATION_STEPS:-1}"
AUC_METRIC="${DEEPMS_AUC_METRIC:-micro}"
SEED="${DEEPMS_SEED:-88}"
FOLD="${DEEPMS_FOLD:-1}"
JOB_ID="${SLURM_JOB_ID:-0}"
MASTER_PORT="${DEEPMS_MASTER_PORT:-$((20000 + JOB_ID % 20000))}"

case "${MIXED_PRECISION}" in
    no|fp16|bf16) ;;
    *) fail "DEEPMS_MIXED_PRECISION must be no, fp16, or bf16." ;;
esac
case "${AUC_METRIC}" in
    micro|macro|hierarchical|ensemble) ;;
    *) fail "DEEPMS_AUC_METRIC must be micro, macro, hierarchical, or ensemble." ;;
esac
[[ "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]] || \
    fail "DEEPMS_GRADIENT_ACCUMULATION_STEPS must be a positive integer."
for name in NUM_EPOCHS EARLY_STOPPING_EPOCHS VAL_INTERVAL SAVE_INTERVAL; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || \
        fail "${name} must be a positive integer."
done

mkdir -p "${OUTPUT_ROOT}"

SMRI_MODALITIES=(
    2DT1_NCE 2DT1_CE
    2DFLAIR_NCE 2DFLAIR_CE
    3DT1_NCE 3DT1_CE
    3DFLAIR_NCE 3DFLAIR_CE
    b0
)
SMRI_VAL_MODALITIES=(
    2DT1_CE
    2DFLAIR_CE
    3DT1_NCE 3DT1_CE
    3DFLAIR_NCE 3DFLAIR_CE
    b0
)
DTI_MODALITIES=(ad_dti fa_dti md_dti rd_dti)
SMI_MODALITIES=(Da_smi DePar_smi DePerp_smi f_smi p2_smi)
DKI_MODALITIES=(ak_wdki mk_wdki rk_wdki)

if [[ -n "${DEEPMS_MODALITIES:-}" ]]; then
    read -r -a TRAIN_MODALITIES <<< "${DEEPMS_MODALITIES}"
else
    TRAIN_MODALITIES=(
        "${SMRI_MODALITIES[@]}"
        "${DTI_MODALITIES[@]}"
        "${SMI_MODALITIES[@]}"
        "${DKI_MODALITIES[@]}"
    )
fi
if [[ -n "${DEEPMS_VAL_MODALITIES:-}" ]]; then
    read -r -a VALIDATION_MODALITIES <<< "${DEEPMS_VAL_MODALITIES}"
else
    VALIDATION_MODALITIES=(
        "${SMRI_VAL_MODALITIES[@]}"
        "${DTI_MODALITIES[@]}"
        "${SMI_MODALITIES[@]}"
        "${DKI_MODALITIES[@]}"
    )
fi
(( ${#TRAIN_MODALITIES[@]} > 0 )) || fail "No training modalities were selected."
(( ${#VALIDATION_MODALITIES[@]} > 0 )) || fail "No validation modalities were selected."

SCHEDULE_ARGS=(
    --min_lr "${MIN_LR}"
    --warmup_start_lr "${WARMUP_START_LR}"
)
if [[ -n "${DEEPMS_WARMUP_STEPS:-}" ]]; then
    [[ "${DEEPMS_WARMUP_STEPS}" =~ ^[0-9]+$ ]] ||         fail "DEEPMS_WARMUP_STEPS must be a non-negative integer."
    SCHEDULE_ARGS+=(--warmup_steps "${DEEPMS_WARMUP_STEPS}")
elif [[ "${DEEPMS_USE_WARMUP:-0}" == "1" ]]; then
    SCHEDULE_ARGS+=(--use_warmup --warmup_epochs "${DEEPMS_WARMUP_EPOCHS:-1}")
fi

RESUME_ARGS=()
if [[ -n "${DEEPMS_RESUME_CHECKPOINT:-}" ]]; then
    [[ -f "${DEEPMS_RESUME_CHECKPOINT}" ]] || \
        fail "Resume checkpoint not found: ${DEEPMS_RESUME_CHECKPOINT}"
    RESUME_ARGS=(--continue_training "${DEEPMS_RESUME_CHECKPOINT}")
fi

printf 'DeepMS root: %s\n' "${PROJECT_ROOT}"
printf 'Slurm job: %s | master port: %s\n' "${SLURM_JOB_ID:-local}" "${MASTER_PORT}"
printf 'Training modalities: %s\n' "${TRAIN_MODALITIES[*]}"
printf 'Training control: epochs=%s | validation interval=%s | early-stop patience=%s | checkpoint interval=%s\n' \
    "${NUM_EPOCHS}" "${VAL_INTERVAL}" "${EARLY_STOPPING_EPOCHS}" "${SAVE_INTERVAL}"
printf 'Validation modalities: %s\n' "${VALIDATION_MODALITIES[*]}"
case "${DEEPMS_PREFLIGHT_ONLY:-0}" in
    1)
        "${UV_RUN[@]}" python "${PROJECT_ROOT}/train.py" --help >/dev/null
        printf 'Preflight passed; no training process was started.\n'
        exit 0
        ;;
    0) ;;
    *) fail "DEEPMS_PREFLIGHT_ONLY must be 0 or 1." ;;
esac
"${UV_RUN[@]}" python -c \
    'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

srun "${UV_RUN[@]}" accelerate launch \
    --config_file "${PROJECT_ROOT}/configs/accelerate/multi_gpu.yaml" \
    --main_process_port "${MASTER_PORT}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${PROJECT_ROOT}/train.py" \
    --modalities "${TRAIN_MODALITIES[@]}" \
    --val_modalities "${VALIDATION_MODALITIES[@]}" \
    --train_patient_ids "${DEEPMS_TRAIN_CSV}" \
    --val_patient_ids "${DEEPMS_VAL_CSV}" \
    --pretrained_path "${DEEPMS_PRETRAINED_PATH}" \
    --output_path "${OUTPUT_ROOT}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --early_stopping_epochs "${EARLY_STOPPING_EPOCHS}" \
    --val_interval "${VAL_INTERVAL}" \
    --save_interval "${SAVE_INTERVAL}" \
    --num_workers "${NUM_WORKERS}" \
    --lr "${LEARNING_RATE}" \
    --roi_x 128 --roi_y 128 --roi_z 128 \
    --use_global_transform \
    --backbone VoCo_Salient_2 \
    --use_bet_only \
    --auc_metric "${AUC_METRIC}" \
    --loss_type bce_with_logits \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --oversampling \
    --merge_flair \
    --outside_reg_loss 1e-6 \
    --L1_loss 1e-7 \
    --num_heads 2 \
    --seed "${SEED}" \
    --fold "${FOLD}" \
    "${SCHEDULE_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
