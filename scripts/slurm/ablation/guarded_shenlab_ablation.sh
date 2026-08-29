#!/usr/bin/env bash

set -euo pipefail
umask 002

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: ./scripts/slurm/ablation/guarded_shenlab_ablation.sh [--check | --submit]

With no arguments (or --check), verify the current checkout, locked environment,
input paths, shell syntax, and all Slurm request shapes without creating jobs.

Pass --submit explicitly to submit the complete ablation graph. The command waits
for the lightweight CPU launcher to finish and returns only after every child job
has been accepted by Slurm and recorded in the run state file.
EOF
}

SUBMIT=0
MODE=""
while (( $# > 0 )); do
    case "$1" in
        --check)
            [[ -z "${MODE}" ]] || fail "Choose only one of --check or --submit."
            MODE=check
            shift
            ;;
        --submit)
            [[ -z "${MODE}" ]] || fail "Choose only one of --check or --submit."
            MODE=submit
            SUBMIT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
cd "${PROJECT_ROOT}"
[[ -f pyproject.toml && -f uv.lock ]] || fail "DeepMS metadata is missing under ${PROJECT_ROOT}."

SITE_PROFILE="${DEEPMS_SITE_PROFILE:-${PROJECT_ROOT}/configs/slurm.shenlab.env}"
[[ -f "${SITE_PROFILE}" ]] || fail "Shenlab site profile not found: ${SITE_PROFILE}"
# shellcheck disable=SC1090
source "${SITE_PROFILE}"
[[ "${DEEPMS_PROJECT_ROOT:-}" == "${PROJECT_ROOT}" ]] || \
    fail "The site profile did not resolve to this DeepMS clone."

GIT_BIN="${DEEPMS_GIT_BIN:-git}"
command -v "${GIT_BIN}" >/dev/null 2>&1 || fail "git is unavailable: ${GIT_BIN}"

GIT_COMMIT="$("${GIT_BIN}" -C "${PROJECT_ROOT}" rev-parse --verify HEAD)" || \
    fail "Could not resolve the current Git commit."
[[ "${GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "Unexpected Git commit: ${GIT_COMMIT}"
GIT_STATUS="$("${GIT_BIN}" -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)" || \
    fail "Could not inspect the Git worktree."
[[ -z "${GIT_STATUS}" ]] || fail "The Git worktree is not clean. Commit or discard local changes first."
GIT_UPSTREAM="$("${GIT_BIN}" -C "${PROJECT_ROOT}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')" || \
    fail "The current branch has no upstream."
UPSTREAM_REMOTE="${GIT_UPSTREAM%%/*}"
[[ -n "${UPSTREAM_REMOTE}" && "${UPSTREAM_REMOTE}" != "${GIT_UPSTREAM}" ]] || \
    fail "Could not identify the upstream remote from ${GIT_UPSTREAM}."
"${GIT_BIN}" -C "${PROJECT_ROOT}" fetch --quiet "${UPSTREAM_REMOTE}" || \
    fail "Could not refresh ${UPSTREAM_REMOTE}. Check network access, then retry."
UPSTREAM_COMMIT="$("${GIT_BIN}" -C "${PROJECT_ROOT}" rev-parse --verify "${GIT_UPSTREAM}")" || \
    fail "Could not resolve ${GIT_UPSTREAM}."
[[ "${GIT_COMMIT}" == "${UPSTREAM_COMMIT}" ]] || \
    fail "This checkout is not current (${GIT_COMMIT} != ${GIT_UPSTREAM} ${UPSTREAM_COMMIT}). Run git pull --ff-only."

require_file_variable() {
    local name="$1"
    [[ -n "${!name:-}" ]] || fail "Required environment variable ${name} is not set."
    [[ -f "${!name}" ]] || fail "${name} does not point to a file: ${!name}"
}

for name in DEEPMS_TRAIN_CSV DEEPMS_VAL_CSV DEEPMS_PRETRAINED_PATH DEEPMS_PUBLIC_EXTERNAL_TEST_CSV; do
    require_file_variable "${name}"
done
[[ "${DEEPMS_PIPELINE_MODE:-}" == "both" ]] || \
    fail "The Shenlab workflow requires DEEPMS_PIPELINE_MODE=both."
[[ "${DEEPMS_VAL_NUM_WORKERS:-}" == "0" ]] || \
    fail "DEEPMS_VAL_NUM_WORKERS must be 0 to avoid validation-worker host-memory spikes."
[[ "${DEEPMS_EXPECTED_GPUS:-}" == "2" ]] || \
    fail "DEEPMS_EXPECTED_GPUS must be 2."
CONCURRENCY="${DEEPMS_ABLATION_CONCURRENCY:-}"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
    fail "DEEPMS_ABLATION_CONCURRENCY must be a positive integer."
(( CONCURRENCY <= 4 )) || fail "The Shenlab workflow caps concurrency at 4."
[[ -z "${DEEPMS_RESUME_CHECKPOINT:-}" ]] || \
    fail "The complete ablation graph does not accept a shared resume checkpoint."

TRAIN_EXCLUDE_NODES="${DEEPMS_TRAIN_EXCLUDE_NODES:-}"
[[ "${TRAIN_EXCLUDE_NODES}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*(,[A-Za-z0-9][A-Za-z0-9._-]*)*$ ]] || \
    fail "DEEPMS_TRAIN_EXCLUDE_NODES must be a comma-separated node list."
for required_node in a100-4011 a100-4024; do
    case ",${TRAIN_EXCLUDE_NODES}," in
        *",${required_node},"*) ;;
        *) fail "The Shenlab exclusion list must include ${required_node}." ;;
    esac
done

# Never propagate a user-fixed distributed port into independently scheduled
# array elements. train.sbatch derives a distinct port from each Slurm task.
unset DEEPMS_MASTER_PORT

LAUNCHER="${PROJECT_ROOT}/scripts/slurm/ablation/submit_shenlab_ablation.sbatch"
PIPELINE_JOB="${PROJECT_ROOT}/scripts/slurm/ablation/run_diffusion_ablation_pipeline.sbatch"
TRAIN_JOB="${PROJECT_ROOT}/scripts/slurm/ablation/train_diffusion_ablation.sbatch"
INFER_JOB="${PROJECT_ROOT}/scripts/slurm/ablation/infer_diffusion_ablation.sbatch"
SUMMARY_JOB="${PROJECT_ROOT}/scripts/slurm/ablation/summarize_inference_runs.sbatch"
for path in "${LAUNCHER}" "${PIPELINE_JOB}" "${TRAIN_JOB}" "${INFER_JOB}" "${SUMMARY_JOB}"; do
    [[ -f "${path}" ]] || fail "Required workflow component not found: ${path}"
    bash -n "${path}" || fail "Shell syntax check failed: ${path}"
done
"${PROJECT_ROOT}/scripts/bootstrap_env.sh" --check
for command_name in sbatch squeue scontrol scancel; do
    command -v "${command_name}" >/dev/null 2>&1 || fail "${command_name} is unavailable on this host."
done

if [[ -n "${DEEPMS_SUBMISSION_ID:-}" ]]; then
    EVALUATION_ID="${DEEPMS_SUBMISSION_ID}"
else
    EVALUATION_ID="deepms-$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID:-$$}-${RANDOM}"
fi
[[ "${EVALUATION_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    fail "DEEPMS_SUBMISSION_ID must be one safe path component."
RUN_TAG="${EVALUATION_ID:0:32}"

OUTPUT_ROOT="${DEEPMS_SUBMISSION_OUTPUT_ROOT:-${DEEPMS_PROJECT_ROOT}/outputs}"
[[ "${OUTPUT_ROOT}" == /* && "${OUTPUT_ROOT}" != "/" && "${OUTPUT_ROOT}" != "${PROJECT_ROOT}" ]] || \
    fail "DEEPMS_SUBMISSION_OUTPUT_ROOT must be a safe absolute directory."
[[ "${OUTPUT_ROOT}" != *','* && "${OUTPUT_ROOT}" != *$'\n'* ]] || \
    fail "The submission output root cannot contain commas or newlines."
WRITABLE_PARENT="${OUTPUT_ROOT}"
while [[ ! -e "${WRITABLE_PARENT}" ]]; do
    NEXT_PARENT="$(dirname -- "${WRITABLE_PARENT}")"
    [[ "${NEXT_PARENT}" != "${WRITABLE_PARENT}" ]] || break
    WRITABLE_PARENT="${NEXT_PARENT}"
done
[[ -d "${WRITABLE_PARENT}" && -w "${WRITABLE_PARENT}" ]] || \
    fail "No writable existing parent is available for ${OUTPUT_ROOT}."
TRAIN_ROOT="${OUTPUT_ROOT}/train/diffusion_single_map/${EVALUATION_ID}"
INFERENCE_BASE="${OUTPUT_ROOT}/inference/diffusion_single_map"
INFERENCE_ROOT="${INFERENCE_BASE}/${EVALUATION_ID}"
STATE_ROOT="${OUTPUT_ROOT}/slurm/pipelines"
STATE_FILE="${STATE_ROOT}/${EVALUATION_ID}/pipeline_jobs.env"
for path in "${TRAIN_ROOT}" "${INFERENCE_ROOT}" "$(dirname -- "${STATE_FILE}")"; do
    [[ ! -e "${path}" ]] || fail "Run ID is already in use: ${path}"
done

export DEEPMS_ABLATION_EVAL_ID="${EVALUATION_ID}"
export DEEPMS_PIPELINE_TRAIN_ROOT="${TRAIN_ROOT}"
export DEEPMS_ABLATION_INFERENCE_ROOT="${INFERENCE_BASE}"
export DEEPMS_PIPELINE_STATE_ROOT="${STATE_ROOT}"

COMMON_EXPORT="ALL,DEEPMS_SITE_PROFILE=${SITE_PROFILE}"
COMMON_EXPORT+=",DEEPMS_ABLATION_EVAL_ID=${EVALUATION_ID}"
COMMON_EXPORT+=",DEEPMS_GUARDED_EXPECTED_COMMIT=${GIT_COMMIT}"
COMMON_EXPORT+=",DEEPMS_PIPELINE_TRAIN_ROOT=${TRAIN_ROOT}"
COMMON_EXPORT+=",DEEPMS_ABLATION_INFERENCE_ROOT=${INFERENCE_BASE}"
COMMON_EXPORT+=",DEEPMS_PIPELINE_STATE_ROOT=${STATE_ROOT}"

slurm_test() {
    local label="$1"
    shift
    local response
    if ! response="$(sbatch --test-only "$@" 2>&1)"; then
        fail "Slurm rejected ${label}: ${response}"
    fi
    printf 'Slurm accepted %-18s %s\n' "${label}:" "${response}"
}

slurm_test launcher \
    --chdir="${PROJECT_ROOT}" \
    --job-name="deepms-submit-${RUN_TAG}" \
    --export="${COMMON_EXPORT}" \
    "${LAUNCHER}"
slurm_test training-array \
    --array="0-11%${CONCURRENCY}" \
    --exclude="${TRAIN_EXCLUDE_NODES}" \
    --job-name="deepms-train-${RUN_TAG}" \
    "${TRAIN_JOB}"
slurm_test inference-array \
    --array="0-11%${CONCURRENCY}" \
    --job-name="deepms-infer-${RUN_TAG}" \
    "${INFER_JOB}"
slurm_test summaries \
    --job-name="deepms-summary-${RUN_TAG}" \
    "${SUMMARY_JOB}"

printf '\nSubmission checks passed.\n'
printf '  commit:        %s\n' "${GIT_COMMIT}"
printf '  evaluation ID: %s\n' "${EVALUATION_ID}"
printf '  train root:    %s\n' "${TRAIN_ROOT}"
printf '  result root:   %s\n' "${INFERENCE_ROOT}"

if (( SUBMIT == 0 )); then
    printf '\nCHECK-ONLY: Slurm validated every request shape; no job was created.\n'
    printf 'Rerun with --submit to create the complete graph.\n'
    exit 0
fi

printf '\nSubmitting the CPU launcher and waiting for it to create the child graph...\n'
if ! LAUNCHER_RESPONSE="$(
    sbatch \
        --parsable \
        --wait \
        --chdir="${PROJECT_ROOT}" \
        --job-name="deepms-submit-${RUN_TAG}" \
        --export="${COMMON_EXPORT}" \
        "${LAUNCHER}"
)"; then
    fail "The CPU launcher failed. Inspect slurm-deepms-submit-${RUN_TAG}-*.err."
fi
LAUNCHER_JOB_ID="${LAUNCHER_RESPONSE%%;*}"
[[ "${LAUNCHER_JOB_ID}" =~ ^[0-9]+$ ]] || \
    fail "Could not parse the launcher job ID from: ${LAUNCHER_RESPONSE}"
[[ -f "${STATE_FILE}" ]] || fail "The launcher completed without creating ${STATE_FILE}."
grep -qx 'PIPELINE_STATUS=scheduled' "${STATE_FILE}" || \
    fail "The child graph is not marked scheduled in ${STATE_FILE}."

JOB_IDS=()
for name in TRAIN_JOB_ID UNMASKED_INFER_JOB_ID MASKED_INFER_JOB_ID DATASET_SUMMARY_JOB_ID MASKING_SUMMARY_JOB_ID; do
    value="$(sed -n "s/^${name}=//p" "${STATE_FILE}")"
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} is missing or invalid in ${STATE_FILE}."
    JOB_IDS+=("${value}")
done

QUEUE_IDS="$(IFS=,; printf '%s' "${JOB_IDS[*]}")"
for job_id in "${JOB_IDS[@]}"; do
    JOB_STATES="$(squeue --noheader --job="${job_id}" --format='%T')" || \
        fail "Could not query child job ${job_id}."
    [[ -n "${JOB_STATES}" ]] || fail "Child job ${job_id} is no longer active in Slurm."
    while IFS= read -r job_state; do
        case "${job_state}" in
            PENDING|RUNNING|CONFIGURING|COMPLETING) ;;
            *) fail "Child job ${job_id} has unexpected state ${job_state}." ;;
        esac
    done <<< "${JOB_STATES}"
done
QUEUE_STATE="$(squeue --noheader --job="${QUEUE_IDS}" --format='%i|%T|%R')" || \
    fail "Could not verify the submitted child jobs with squeue."
[[ -n "${QUEUE_STATE}" ]] || fail "Slurm accepted the graph, but its child jobs are no longer active."

printf '\nDeepMS graph is live in Slurm.\n'
printf '  launcher job: %s\n' "${LAUNCHER_JOB_ID}"
printf '  child jobs:   %s\n' "${QUEUE_IDS}"
printf '  state file:   %s\n' "${STATE_FILE}"
printf '%s\n' "${QUEUE_STATE}"
