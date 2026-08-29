#!/usr/bin/env bash

# Source this file after PROJECT_ROOT and fail() have been defined.
deepms_verify_guarded_source_tree() {
    local project_root="$1"
    local expected_commit="${DEEPMS_GUARDED_EXPECTED_COMMIT:-}"
    local git_bin
    local git_library_path="${LD_LIBRARY_PATH:-}"
    local current_commit
    local git_status

    [[ -n "${expected_commit}" ]] || return 0
    [[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] || \
        fail "Invalid guarded expected Git commit: ${expected_commit}"

    git_bin="${DEEPMS_GIT_BIN:-git}"
    command -v "${git_bin}" >/dev/null 2>&1 || \
        fail "git is unavailable for the guarded source-tree check: ${git_bin}"
    [[ -x "${git_bin}" ]] || fail "Git executable is not runnable: ${git_bin}"
    if [[ -n "${DEEPMS_GIT_LIB_DIR:-}" ]]; then
        git_library_path="${DEEPMS_GIT_LIB_DIR}${git_library_path:+:${git_library_path}}"
    fi

    current_commit="$(
        env LD_LIBRARY_PATH="${git_library_path}" \
            "${git_bin}" -C "${project_root}" rev-parse --verify HEAD
    )" || \
        fail "Could not resolve the current Git commit under ${project_root}."
    [[ "${current_commit}" == "${expected_commit}" ]] || \
        fail "Repository HEAD changed after guarded submission: expected ${expected_commit}, found ${current_commit}."

    git_status="$(
        env LD_LIBRARY_PATH="${git_library_path}" \
            "${git_bin}" -C "${project_root}" status --porcelain --untracked-files=normal
    )" || fail "Could not inspect the guarded Git worktree under ${project_root}."
    [[ -z "${git_status}" ]] || \
        fail "Repository worktree changed after guarded submission; refusing to mix source versions."
}

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
