#!/usr/bin/env bash
#SBATCH --job-name=deepms-dmri-ablation
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:2
#SBATCH --array=0-11%4

set -euo pipefail
umask 002

PROJECT_ROOT="${DEEPMS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Requested order: SMI, then DTI, then DKI. Each array task adds one map only.
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

STRUCTURAL_MODALITIES=(
    2DT1_NCE 2DT1_CE
    2DFLAIR_NCE 2DFLAIR_CE
    3DT1_NCE 3DT1_CE
    3DFLAIR_NCE 3DFLAIR_CE
)
STRUCTURAL_VAL_MODALITIES=(
    2DT1_CE
    2DFLAIR_CE
    3DT1_NCE 3DT1_CE
    3DFLAIR_NCE 3DFLAIR_CE
)

# Historical DeepMS evaluation groups b0 with structural MRI.
case "${DEEPMS_INCLUDE_B0:-1}" in
    1)
        STRUCTURAL_MODALITIES+=(b0)
        STRUCTURAL_VAL_MODALITIES+=(b0)
        ;;
    0) ;;
    *)
        printf 'ERROR: DEEPMS_INCLUDE_B0 must be 0 or 1.\n' >&2
        exit 1
        ;;
esac

export DEEPMS_PROJECT_ROOT="${PROJECT_ROOT}"
export DEEPMS_MODALITIES="${STRUCTURAL_MODALITIES[*]} ${DIFFUSION_MAP}"
export DEEPMS_VAL_MODALITIES="${STRUCTURAL_VAL_MODALITIES[*]} ${DIFFUSION_MAP}"
ABLATION_OUTPUT_ROOT="${DEEPMS_ABLATION_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/train/diffusion_single_map}"
export DEEPMS_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT}/${DIFFUSION_FAMILY}/${DIFFUSION_MAP}"

TRAIN_JOB="${PROJECT_ROOT}/scripts/slurm/train.sbatch"
[[ -f "${TRAIN_JOB}" ]] || {
    printf 'ERROR: shared training job not found: %s\n' "${TRAIN_JOB}" >&2
    exit 1
}

printf 'Single-map diffusion ablation\n'
printf '  task:       %s/%s\n' "${TASK_INDEX}" "$(( ${#DIFFUSION_MAPS[@]} - 1 ))"
printf '  family:     %s\n' "${DIFFUSION_FAMILY}"
printf '  map:        %s\n' "${DIFFUSION_MAP}"
printf '  train mods: %s\n' "${DEEPMS_MODALITIES}"
printf '  val mods:   %s\n' "${DEEPMS_VAL_MODALITIES}"
printf '  output:     %s\n' "${DEEPMS_OUTPUT_ROOT}"

exec bash "${TRAIN_JOB}"
