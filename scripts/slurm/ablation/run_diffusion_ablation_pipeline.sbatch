#!/usr/bin/env bash
#SBATCH --job-name=deepms-ablation-pipeline
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=cpu_short

set -euo pipefail
umask 002

PROJECT_ROOT="${DEEPMS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -n "${SLURM_JOB_ID:-}" ]] || \
    fail "Submit this orchestration job with sbatch."
for command_name in sbatch scontrol scancel; do
    command -v "${command_name}" >/dev/null 2>&1 || fail "${command_name} is not available."
done
UV_BIN="${DEEPMS_UV_BIN:-uv}"
command -v "${UV_BIN}" >/dev/null 2>&1 || fail "uv is not available: ${UV_BIN}"
[[ -f pyproject.toml && -f uv.lock ]] || \
    fail "Submit from the DeepMS repository root or set DEEPMS_PROJECT_ROOT."
RUNTIME_ENVIRONMENT="${PROJECT_ROOT}/scripts/slurm/runtime_environment.sh"
[[ -f "${RUNTIME_ENVIRONMENT}" ]] || fail "Runtime environment helper not found: ${RUNTIME_ENVIRONMENT}"
# shellcheck disable=SC1090
source "${RUNTIME_ENVIRONMENT}"
deepms_verify_guarded_source_tree "${PROJECT_ROOT}"
deepms_activate_locked_environment "${PROJECT_ROOT}"

ABLATION_SLURM_DIR="${PROJECT_ROOT}/scripts/slurm/ablation"
TRAIN_JOB="${ABLATION_SLURM_DIR}/train_diffusion_ablation.sbatch"
INFER_JOB="${ABLATION_SLURM_DIR}/infer_diffusion_ablation.sbatch"
SUMMARY_JOB="${ABLATION_SLURM_DIR}/summarize_inference_runs.sbatch"
for path in "${TRAIN_JOB}" "${INFER_JOB}" "${SUMMARY_JOB}"; do
    [[ -f "${path}" ]] || fail "Pipeline component not found: ${path}"
done

require_file_variable() {
    local name="$1"
    [[ -n "${!name:-}" ]] || fail "Required environment variable ${name} is not set."
    [[ -f "${!name}" ]] || fail "${name} does not point to a file: ${!name}"
}

for name in DEEPMS_TRAIN_CSV DEEPMS_VAL_CSV DEEPMS_PRETRAINED_PATH; do
    require_file_variable "${name}"
done
if [[ -n "${DEEPMS_REPORT_COHORT_OVERRIDES:-}" ]]; then
    [[ -f "${DEEPMS_REPORT_COHORT_OVERRIDES}" ]] || \
        fail "Report cohort override CSV not found: ${DEEPMS_REPORT_COHORT_OVERRIDES}"
fi
[[ -z "${DEEPMS_RESUME_CHECKPOINT:-}" ]] || \
    fail "The full pipeline does not accept one shared resume checkpoint across all map tasks."

PIPELINE_MODE="${DEEPMS_PIPELINE_MODE:-dataset_calibrated}"
DATASET_PROFILE="${DEEPMS_ABLATION_INFER_PROFILE:-public_external_unmasked}"
case "${PIPELINE_MODE}" in
    dataset_calibrated|masking_raw|both) ;;
    *)
        fail "DEEPMS_PIPELINE_MODE must be dataset_calibrated, masking_raw, or both."
        ;;
esac
if [[ "${PIPELINE_MODE}" == "dataset_calibrated" || "${PIPELINE_MODE}" == "both" ]]; then
    case "${DATASET_PROFILE}" in
        internal|krakow|public_external_unmasked) ;;
        *)
            fail "The calibrated profile must be internal, krakow, or public_external_unmasked."
            ;;
    esac
fi
if [[ "${PIPELINE_MODE}" == "both" && "${DATASET_PROFILE}" != "public_external_unmasked" ]]; then
    fail "DEEPMS_PIPELINE_MODE=both requires the public_external_unmasked profile."
fi

if [[ "${PIPELINE_MODE}" == "dataset_calibrated" || "${PIPELINE_MODE}" == "both" ]]; then
    case "${DATASET_PROFILE}" in
        internal) require_file_variable DEEPMS_INTERNAL_TEST_CSV ;;
        krakow) require_file_variable DEEPMS_KRAKOW_TEST_CSV ;;
        public_external_unmasked) require_file_variable DEEPMS_PUBLIC_EXTERNAL_TEST_CSV ;;
    esac
fi
if [[ "${PIPELINE_MODE}" == "masking_raw" || "${PIPELINE_MODE}" == "both" ]]; then
    require_file_variable DEEPMS_PUBLIC_EXTERNAL_TEST_CSV
fi

EVALUATION_ID="${DEEPMS_ABLATION_EVAL_ID:-pipeline-${SLURM_JOB_ID}}"
[[ "${EVALUATION_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    fail "DEEPMS_ABLATION_EVAL_ID must be one safe path component."
RUN_TAG="${EVALUATION_ID:0:32}"

CONCURRENCY="${DEEPMS_ABLATION_CONCURRENCY:-4}"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
    fail "DEEPMS_ABLATION_CONCURRENCY must be a positive integer."
(( CONCURRENCY <= 12 )) || \
    fail "DEEPMS_ABLATION_CONCURRENCY cannot exceed 12."
ARRAY_SPEC="0-11%${CONCURRENCY}"
TRAIN_EXCLUDE_NODES="${DEEPMS_TRAIN_EXCLUDE_NODES:-}"
if [[ -n "${TRAIN_EXCLUDE_NODES}" ]]; then
    [[ "${TRAIN_EXCLUDE_NODES}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*(,[A-Za-z0-9][A-Za-z0-9._-]*)*$ ]] || \
        fail "DEEPMS_TRAIN_EXCLUDE_NODES must be a comma-separated node list."
fi

TRAIN_ROOT="${DEEPMS_PIPELINE_TRAIN_ROOT:-${PROJECT_ROOT}/outputs/train/diffusion_single_map/${EVALUATION_ID}}"
INFERENCE_ROOT="${DEEPMS_ABLATION_INFERENCE_ROOT:-${PROJECT_ROOT}/outputs/inference/diffusion_single_map}"
EVALUATION_ROOT="${INFERENCE_ROOT}/${EVALUATION_ID}"
STATE_ROOT="${DEEPMS_PIPELINE_STATE_ROOT:-${PROJECT_ROOT}/outputs/slurm/pipelines}"
STATE_DIR="${STATE_ROOT}/${EVALUATION_ID}"

for value in "${PROJECT_ROOT}" "${TRAIN_ROOT}" "${INFERENCE_ROOT}" "${EVALUATION_ID}" "${STATE_ROOT}"; do
    [[ "${value}" != *','* && "${value}" != *$'\n'* ]] || \
        fail "Pipeline paths and IDs cannot contain commas or newlines."
done
[[ "${TRAIN_ROOT}" != "/" && "${TRAIN_ROOT}" != "${PROJECT_ROOT}" ]] || \
    fail "Unsafe pipeline training root: ${TRAIN_ROOT}"
[[ "${EVALUATION_ROOT}" != "/" && "${EVALUATION_ROOT}" != "${PROJECT_ROOT}" ]] || \
    fail "Unsafe pipeline inference root: ${EVALUATION_ROOT}"
[[ ! -e "${TRAIN_ROOT}" ]] || fail "Pipeline training root already exists: ${TRAIN_ROOT}"
[[ ! -e "${EVALUATION_ROOT}" ]] || fail "Pipeline inference root already exists: ${EVALUATION_ROOT}"

mkdir -p "${STATE_ROOT}"
mkdir "${STATE_DIR}" || fail "Evaluation ID is already reserved: ${EVALUATION_ID}"
STATE_FILE="${STATE_DIR}/pipeline_jobs.env"
{
    printf 'PIPELINE_STATUS=submitting_held\n'
    printf 'PIPELINE_MODE=%q\n' "${PIPELINE_MODE}"
    printf 'EVALUATION_ID=%q\n' "${EVALUATION_ID}"
    printf 'TRAIN_ROOT=%q\n' "${TRAIN_ROOT}"
    printf 'EVALUATION_ROOT=%q\n' "${EVALUATION_ROOT}"
    printf 'ARRAY_SPEC=%q\n' "${ARRAY_SPEC}"
    printf 'TRAIN_EXCLUDE_NODES=%q\n' "${TRAIN_EXCLUDE_NODES}"
} > "${STATE_FILE}"

GRAPH_RELEASED=0
SUBMITTED_JOB_IDS=()
rollback_partial_graph() {
    local exit_status=$?
    local cancel_status=0
    trap - EXIT
    if (( GRAPH_RELEASED == 0 )); then
        set +e
        if (( ${#SUBMITTED_JOB_IDS[@]} > 0 )); then
            scancel "${SUBMITTED_JOB_IDS[@]}"
            cancel_status=$?
        fi
        if (( cancel_status == 0 )); then
            printf 'PIPELINE_STATUS=rolled_back\n' >> "${STATE_FILE}"
        else
            printf 'PIPELINE_STATUS=rollback_failed\n' >> "${STATE_FILE}"
            printf 'ERROR: Could not cancel every held partial-graph job: %s\n' \
                "${SUBMITTED_JOB_IDS[*]}" >&2
        fi
    fi
    exit "${exit_status}"
}
trap rollback_partial_graph EXIT

parse_job_id() {
    local response="$1"
    local job_id="${response%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || \
        fail "Could not parse Slurm job ID from: ${response}"
    printf '%s\n' "${job_id}"
}

submit_training() {
    local export_spec
    local response
    local -a submit_args
    export_spec="ALL,DEEPMS_PROJECT_ROOT=${PROJECT_ROOT}"
    export_spec+=",DEEPMS_ABLATION_OUTPUT_ROOT=${TRAIN_ROOT}"
    export_spec+=",DEEPMS_PREFLIGHT_ONLY=0"
    submit_args=(
        --parsable
        --hold
        --no-requeue
        --job-name="deepms-train-${RUN_TAG}"
        --array="${ARRAY_SPEC}"
        --export="${export_spec}"
    )
    if [[ -n "${TRAIN_EXCLUDE_NODES}" ]]; then
        submit_args+=(--exclude="${TRAIN_EXCLUDE_NODES}")
    fi
    response="$(
        sbatch "${submit_args[@]}" "${TRAIN_JOB}"
    )"
    parse_job_id "${response}"
}

submit_inference() {
    local profile="$1"
    local training_job_id="$2"
    local export_spec
    local response
    export_spec="ALL,DEEPMS_PROJECT_ROOT=${PROJECT_ROOT}"
    export_spec+=",DEEPMS_ABLATION_CHECKPOINT_ROOT=${TRAIN_ROOT}"
    export_spec+=",DEEPMS_ABLATION_INFERENCE_ROOT=${INFERENCE_ROOT}"
    export_spec+=",DEEPMS_ABLATION_EVAL_ID=${EVALUATION_ID}"
    export_spec+=",DEEPMS_ABLATION_INFER_PROFILE=${profile}"
    export_spec+=",DEEPMS_PREFLIGHT_ONLY=0"
    response="$(
        sbatch \
            --parsable \
            --hold \
            --no-requeue \
            --job-name="deepms-infer-${profile:0:12}-${RUN_TAG}" \
            --array="${ARRAY_SPEC}" \
            --dependency="aftercorr:${training_job_id}" \
            --kill-on-invalid-dep=yes \
            --export="${export_spec}" \
            "${INFER_JOB}"
    )"
    parse_job_id "${response}"
}

submit_summary() {
    local evaluation_mode="$1"
    local runs_root="$2"
    local expected_runs="$3"
    local dependency="$4"
    local output_dir="${runs_root}/summary-${evaluation_mode}"
    local export_spec
    local response
    export_spec="ALL,DEEPMS_PROJECT_ROOT=${PROJECT_ROOT}"
    export_spec+=",DEEPMS_INFERENCE_RUNS_ROOT=${runs_root}"
    export_spec+=",DEEPMS_INFERENCE_SUMMARY_OUTPUT=${output_dir}"
    export_spec+=",DEEPMS_EVALUATION_MODE=${evaluation_mode}"
    export_spec+=",DEEPMS_EXPECTED_INFERENCE_RUNS=${expected_runs}"
    export_spec+=",DEEPMS_PREFLIGHT_ONLY=0"
    response="$(
        sbatch \
            --parsable \
            --hold \
            --no-requeue \
            --job-name="deepms-summary-${evaluation_mode:0:8}-${RUN_TAG}" \
            --dependency="${dependency}" \
            --kill-on-invalid-dep=yes \
            --export="${export_spec}" \
            "${SUMMARY_JOB}"
    )"
    parse_job_id "${response}"
}

TRAIN_JOB_ID="$(submit_training)"
SUBMITTED_JOB_IDS+=("${TRAIN_JOB_ID}")
printf 'TRAIN_JOB_ID=%q\n' "${TRAIN_JOB_ID}" >> "${STATE_FILE}"
printf 'Submitted held training array: %s\n' "${TRAIN_JOB_ID}"

DATASET_INFER_JOB_ID=""
UNMASKED_INFER_JOB_ID=""
MASKED_INFER_JOB_ID=""
DATASET_SUMMARY_JOB_ID=""
MASKING_SUMMARY_JOB_ID=""

case "${PIPELINE_MODE}" in
    dataset_calibrated)
        DATASET_INFER_JOB_ID="$(submit_inference "${DATASET_PROFILE}" "${TRAIN_JOB_ID}")"
        SUBMITTED_JOB_IDS+=("${DATASET_INFER_JOB_ID}")
        printf 'DATASET_INFER_JOB_ID=%q\n' "${DATASET_INFER_JOB_ID}" >> "${STATE_FILE}"
        DATASET_SUMMARY_JOB_ID="$(
            submit_summary \
                dataset_calibrated \
                "${EVALUATION_ROOT}/${DATASET_PROFILE}" \
                12 \
                "afterok:${DATASET_INFER_JOB_ID}"
        )"
        SUBMITTED_JOB_IDS+=("${DATASET_SUMMARY_JOB_ID}")
        printf 'DATASET_SUMMARY_JOB_ID=%q\n' "${DATASET_SUMMARY_JOB_ID}" >> "${STATE_FILE}"
        ;;
    masking_raw)
        UNMASKED_INFER_JOB_ID="$(submit_inference public_external_unmasked "${TRAIN_JOB_ID}")"
        SUBMITTED_JOB_IDS+=("${UNMASKED_INFER_JOB_ID}")
        MASKED_INFER_JOB_ID="$(submit_inference public_external_masked "${TRAIN_JOB_ID}")"
        SUBMITTED_JOB_IDS+=("${MASKED_INFER_JOB_ID}")
        printf 'UNMASKED_INFER_JOB_ID=%q\n' "${UNMASKED_INFER_JOB_ID}" >> "${STATE_FILE}"
        printf 'MASKED_INFER_JOB_ID=%q\n' "${MASKED_INFER_JOB_ID}" >> "${STATE_FILE}"
        MASKING_SUMMARY_JOB_ID="$(
            submit_summary \
                masking_raw \
                "${EVALUATION_ROOT}" \
                24 \
                "afterok:${UNMASKED_INFER_JOB_ID}:${MASKED_INFER_JOB_ID}"
        )"
        SUBMITTED_JOB_IDS+=("${MASKING_SUMMARY_JOB_ID}")
        printf 'MASKING_SUMMARY_JOB_ID=%q\n' "${MASKING_SUMMARY_JOB_ID}" >> "${STATE_FILE}"
        ;;
    both)
        UNMASKED_INFER_JOB_ID="$(submit_inference public_external_unmasked "${TRAIN_JOB_ID}")"
        SUBMITTED_JOB_IDS+=("${UNMASKED_INFER_JOB_ID}")
        MASKED_INFER_JOB_ID="$(submit_inference public_external_masked "${TRAIN_JOB_ID}")"
        SUBMITTED_JOB_IDS+=("${MASKED_INFER_JOB_ID}")
        printf 'UNMASKED_INFER_JOB_ID=%q\n' "${UNMASKED_INFER_JOB_ID}" >> "${STATE_FILE}"
        printf 'MASKED_INFER_JOB_ID=%q\n' "${MASKED_INFER_JOB_ID}" >> "${STATE_FILE}"
        DATASET_SUMMARY_JOB_ID="$(
            submit_summary \
                dataset_calibrated \
                "${EVALUATION_ROOT}/public_external_unmasked" \
                12 \
                "afterok:${UNMASKED_INFER_JOB_ID}"
        )"
        SUBMITTED_JOB_IDS+=("${DATASET_SUMMARY_JOB_ID}")
        MASKING_SUMMARY_JOB_ID="$(
            submit_summary \
                masking_raw \
                "${EVALUATION_ROOT}" \
                24 \
                "afterok:${UNMASKED_INFER_JOB_ID}:${MASKED_INFER_JOB_ID}"
        )"
        SUBMITTED_JOB_IDS+=("${MASKING_SUMMARY_JOB_ID}")
        printf 'DATASET_SUMMARY_JOB_ID=%q\n' "${DATASET_SUMMARY_JOB_ID}" >> "${STATE_FILE}"
        printf 'MASKING_SUMMARY_JOB_ID=%q\n' "${MASKING_SUMMARY_JOB_ID}" >> "${STATE_FILE}"
        ;;
esac

release_if_set() {
    local job_id="$1"
    [[ -z "${job_id}" ]] || scontrol release "${job_id}"
}

# Release downstream jobs first. They remain blocked by valid dependencies;
# training is released last, so no compute can begin while the graph is partial.
release_if_set "${DATASET_SUMMARY_JOB_ID}"
release_if_set "${MASKING_SUMMARY_JOB_ID}"
release_if_set "${DATASET_INFER_JOB_ID}"
release_if_set "${UNMASKED_INFER_JOB_ID}"
release_if_set "${MASKED_INFER_JOB_ID}"
scontrol release "${TRAIN_JOB_ID}"
GRAPH_RELEASED=1
printf 'PIPELINE_STATUS=scheduled\n' >> "${STATE_FILE}"

printf '\nDeepMS ablation dependency graph scheduled.\n'
printf '  mode:            %s\n' "${PIPELINE_MODE}"
printf '  evaluation ID:   %s\n' "${EVALUATION_ID}"
printf '  concurrency:     %s\n' "${CONCURRENCY}"
printf '  training array:  %s\n' "${TRAIN_JOB_ID}"
[[ -z "${DATASET_INFER_JOB_ID}" ]] || printf '  dataset infer:   %s\n' "${DATASET_INFER_JOB_ID}"
[[ -z "${UNMASKED_INFER_JOB_ID}" ]] || printf '  unmasked infer:  %s\n' "${UNMASKED_INFER_JOB_ID}"
[[ -z "${MASKED_INFER_JOB_ID}" ]] || printf '  masked infer:    %s\n' "${MASKED_INFER_JOB_ID}"
[[ -z "${DATASET_SUMMARY_JOB_ID}" ]] || printf '  dataset summary: %s\n' "${DATASET_SUMMARY_JOB_ID}"
[[ -z "${MASKING_SUMMARY_JOB_ID}" ]] || printf '  masking summary: %s\n' "${MASKING_SUMMARY_JOB_ID}"
printf '  state file:      %s\n' "${STATE_FILE}"
printf 'This orchestration job submits dependencies and then exits; compute jobs remain managed by Slurm.\n'
