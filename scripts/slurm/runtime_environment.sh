#!/usr/bin/env bash

# Source this file after PROJECT_ROOT and fail() have been defined.
deepms_activate_locked_environment() {
    local project_root="$1"
    local environment_check="${project_root}/scripts/check_environment.py"

    UV_BIN="${DEEPMS_UV_BIN:-uv}"
    command -v "${UV_BIN}" >/dev/null 2>&1 || \
        fail "uv is unavailable (${UV_BIN}). Install uv or set DEEPMS_UV_BIN."
    [[ -f "${project_root}/pyproject.toml" && -f "${project_root}/uv.lock" ]] || \
        fail "DeepMS pyproject.toml or uv.lock is missing under ${project_root}."
    [[ -f "${environment_check}" ]] || \
        fail "Environment checker not found: ${environment_check}"

    # Compute jobs must never install packages or contact a package index.
    UV_RUN=("${UV_BIN}" run --locked --no-sync)
    "${UV_RUN[@]}" python "${environment_check}" --quiet || \
        fail "The locked environment is incomplete. Run ./scripts/bootstrap_env.sh before submitting."
}
