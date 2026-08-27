#!/usr/bin/env bash

set -euo pipefail
umask 002

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: ./scripts/bootstrap_env.sh [--check] [--dev] [--extra NAME] [--all-extras]

Synchronize the DeepMS environment from uv.lock and verify all core pins.

  --check        Validate the existing environment without changing it.
  --dev          Include the development dependency group.
  --extra NAME   Include one optional dependency extra; may be repeated.
  --all-extras   Include every optional dependency extra.

Set DEEPMS_UV_BIN when uv is not on PATH. Set UV_PROJECT_ENVIRONMENT and
UV_CACHE_DIR to place the environment and cache on site-local scratch.
EOF
}

CHECK_ONLY=0
INCLUDE_DEV=0
ALL_EXTRAS=0
EXTRAS=()
while (( $# > 0 )); do
    case "$1" in
        --check)
            CHECK_ONLY=1
            shift
            ;;
        --dev)
            INCLUDE_DEV=1
            shift
            ;;
        --extra)
            (( $# >= 2 )) || fail "--extra requires a name."
            EXTRAS+=("$2")
            shift 2
            ;;
        --all-extras)
            ALL_EXTRAS=1
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

UV_BIN="${DEEPMS_UV_BIN:-uv}"
command -v "${UV_BIN}" >/dev/null 2>&1 || fail \
    "uv is unavailable (${UV_BIN}). Install uv or set DEEPMS_UV_BIN to its executable."
[[ -f pyproject.toml && -f uv.lock && -f .python-version ]] || \
    fail "DeepMS project metadata is incomplete under ${PROJECT_ROOT}."

ENVIRONMENT_PATH="${UV_PROJECT_ENVIRONMENT:-${PROJECT_ROOT}/.venv}"
if (( CHECK_ONLY == 0 )); then
    SYNC_ARGS=(sync --locked)
    (( INCLUDE_DEV == 1 )) || SYNC_ARGS+=(--no-dev)
    if (( ALL_EXTRAS == 1 )); then
        SYNC_ARGS+=(--all-extras)
    fi
    for extra in "${EXTRAS[@]}"; do
        SYNC_ARGS+=(--extra "${extra}")
    done

    printf 'Synchronizing DeepMS environment from uv.lock\n'
    printf '  project:     %s\n' "${PROJECT_ROOT}"
    printf '  environment: %s\n' "${ENVIRONMENT_PATH}"
    printf '  uv:          %s\n' "$(command -v "${UV_BIN}")"
    "${UV_BIN}" "${SYNC_ARGS[@]}"
else
    printf 'Checking existing DeepMS environment: %s\n' "${ENVIRONMENT_PATH}"
fi

"${UV_BIN}" run --locked --no-sync python \
    "${PROJECT_ROOT}/scripts/check_environment.py"
