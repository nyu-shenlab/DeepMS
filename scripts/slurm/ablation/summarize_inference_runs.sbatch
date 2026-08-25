#!/usr/bin/env bash
#SBATCH --job-name=deepms-infer-summary
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short

set -euo pipefail
umask 002
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${DEEPMS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

UV_BIN="${DEEPMS_UV_BIN:-uv}"
command -v "${UV_BIN}" >/dev/null 2>&1 || fail "uv is not available: ${UV_BIN}"
[[ -f pyproject.toml && -f uv.lock && -f summarize_inference_runs.py ]] || \
    fail "Submit from the DeepMS repository root or set DEEPMS_PROJECT_ROOT."
[[ -x .venv/bin/python ]] || \
    fail "Missing .venv. Run: uv sync --locked --no-dev"

: "${DEEPMS_INFERENCE_RUNS_ROOT:?Set DEEPMS_INFERENCE_RUNS_ROOT to the completed inference collection}"
[[ -d "${DEEPMS_INFERENCE_RUNS_ROOT}" ]] || \
    fail "Inference runs root not found: ${DEEPMS_INFERENCE_RUNS_ROOT}"

EVALUATION_MODE="${DEEPMS_EVALUATION_MODE:-dataset_calibrated}"
case "${EVALUATION_MODE}" in
    dataset_calibrated)
        DEFAULT_EXPECTED_RUNS=12
        ;;
    masking_raw)
        DEFAULT_EXPECTED_RUNS=24
        ;;
    *)
        fail "Unsupported DEEPMS_EVALUATION_MODE: ${EVALUATION_MODE}"
        ;;
esac
EXPECTED_RUNS="${DEEPMS_EXPECTED_INFERENCE_RUNS:-${DEFAULT_EXPECTED_RUNS}}"
OUTPUT_DIR="${DEEPMS_INFERENCE_SUMMARY_OUTPUT:-${DEEPMS_INFERENCE_RUNS_ROOT}/summary-${EVALUATION_MODE}}"

[[ "${EXPECTED_RUNS}" =~ ^[1-9][0-9]*$ ]] || \
    fail "DEEPMS_EXPECTED_INFERENCE_RUNS must be a positive integer."

COHORT_ARGS=()
if [[ -n "${DEEPMS_REPORT_COHORT_OVERRIDES:-}" ]]; then
    [[ -f "${DEEPMS_REPORT_COHORT_OVERRIDES}" ]] || \
        fail "Report cohort override CSV not found: ${DEEPMS_REPORT_COHORT_OVERRIDES}"
    COHORT_ARGS=(
        --cohort_overrides "${DEEPMS_REPORT_COHORT_OVERRIDES}"
    )
fi

printf 'Final inference summary\n'
printf '  runs root:       %s\n' "${DEEPMS_INFERENCE_RUNS_ROOT}"
printf '  expected runs:   %s\n' "${EXPECTED_RUNS}"
printf '  evaluation mode: %s\n' "${EVALUATION_MODE}"
printf '  output:          %s\n' "${OUTPUT_DIR}"

"${UV_BIN}" run --locked --no-sync python \
    "${PROJECT_ROOT}/summarize_inference_runs.py" \
    --runs_root "${DEEPMS_INFERENCE_RUNS_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --evaluation_mode "${EVALUATION_MODE}" \
    --expected_runs "${EXPECTED_RUNS}" \
    --bootstrap_samples "${DEEPMS_REPORT_BOOTSTRAPS:-2000}" \
    --seed "${DEEPMS_REPORT_SEED:-42}" \
    --threshold "${DEEPMS_REPORT_THRESHOLD:-0.5}" \
    --target_fpr "${DEEPMS_REPORT_TARGET_FPR:-0.01}" \
    "${COHORT_ARGS[@]}"
