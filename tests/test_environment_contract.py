import importlib.metadata
import os
import shlex
import subprocess
import tomllib
from pathlib import Path

from scripts.check_environment import REPOSITORY_ROOT, audit_environment

BOOTSTRAP = REPOSITORY_ROOT / "scripts" / "bootstrap_env.sh"
RUNTIME_ENVIRONMENT = REPOSITORY_ROOT / "scripts" / "slurm" / "runtime_environment.sh"
SHENLAB_PROFILE = REPOSITORY_ROOT / "configs" / "slurm.shenlab.env"
EXPECTED_COMMIT = "a" * 40


def _fake_uv(tmp_path: Path) -> tuple[Path, Path]:
    command_log = tmp_path / "uv-commands.log"
    executable = tmp_path / "uv"
    executable.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "${FAKE_UV_LOG}"
printf '\n' >> "${FAKE_UV_LOG}"
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, command_log


def _run_guarded_source_check(
    tmp_path: Path,
    *,
    current_commit: str = EXPECTED_COMMIT,
    git_status: str = "",
) -> subprocess.CompletedProcess[str]:
    fake_git = tmp_path / "git"
    fake_git.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$*" in
    *" rev-parse --verify HEAD") printf '%s\n' "${FAKE_CURRENT_COMMIT}" ;;
    *" status --porcelain --untracked-files=normal") printf '%s' "${FAKE_GIT_STATUS}" ;;
    *) printf 'unexpected fake git call: %s\n' "$*" >&2; exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    project_root = tmp_path / "project"
    project_root.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "DEEPMS_GIT_BIN": str(fake_git),
            "DEEPMS_GUARDED_EXPECTED_COMMIT": EXPECTED_COMMIT,
            "FAKE_CURRENT_COMMIT": current_commit,
            "FAKE_GIT_STATUS": git_status,
            "FAKE_PROJECT_ROOT": str(project_root),
            "RUNTIME_ENVIRONMENT_PATH": str(RUNTIME_ENVIRONMENT),
        }
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
source "${RUNTIME_ENVIRONMENT_PATH}"
deepms_verify_guarded_source_tree "${FAKE_PROJECT_ROOT}"
""",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_bootstrap(tmp_path: Path, *arguments: str) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    fake_uv, command_log = _fake_uv(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "DEEPMS_UV_BIN": str(fake_uv),
            "FAKE_UV_LOG": str(command_log),
            "UV_PROJECT_ENVIRONMENT": str(tmp_path / "environment"),
        }
    )
    completed = subprocess.run(
        ["bash", str(BOOTSTRAP), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    commands = [shlex.split(line) for line in command_log.read_text(encoding="utf-8").splitlines()]
    return completed, commands


def test_core_environment_is_declared_with_exact_pins() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]

    assert dependencies
    assert all("==" in requirement for requirement in dependencies)
    assert (REPOSITORY_ROOT / "uv.lock").is_file()


def test_bootstrap_syncs_from_lock_then_runs_read_only_audit(tmp_path: Path) -> None:
    completed, commands = _run_bootstrap(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert commands[0] == ["sync", "--locked", "--no-dev"]
    assert commands[1][:4] == ["run", "--locked", "--no-sync", "python"]
    assert commands[1][-1].endswith("scripts/check_environment.py")


def test_bootstrap_check_mode_never_syncs(tmp_path: Path) -> None:
    completed, commands = _run_bootstrap(tmp_path, "--check")

    assert completed.returncode == 0, completed.stderr
    assert len(commands) == 1
    assert commands[0][:4] == ["run", "--locked", "--no-sync", "python"]


def test_environment_audit_reports_a_missing_pin(monkeypatch) -> None:
    real_version = importlib.metadata.version

    def missing_torch(name: str) -> str:
        if name == "torch":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", missing_torch)
    _, problems = audit_environment()

    assert "Missing distribution: torch==2.6.0" in problems


def test_slurm_runtime_is_validation_only_and_supports_custom_uv_environments() -> None:
    helper = RUNTIME_ENVIRONMENT.read_text(encoding="utf-8")

    assert "run --locked --no-sync" in helper
    assert "uv sync" not in helper
    assert ".venv/bin/python" not in helper
    assert "check_environment.py" in helper

    jobs = [
        REPOSITORY_ROOT / "scripts/slurm/train.sbatch",
        REPOSITORY_ROOT / "scripts/slurm/infer_internal.sbatch",
        REPOSITORY_ROOT / "scripts/slurm/infer_krakow.sbatch",
        REPOSITORY_ROOT / "scripts/slurm/infer_public_external_unmasked.sbatch",
        REPOSITORY_ROOT / "scripts/slurm/infer_public_external_lesion_masked.sbatch",
        REPOSITORY_ROOT / "scripts/slurm/ablation/run_diffusion_ablation_pipeline.sbatch",
        REPOSITORY_ROOT / "scripts/slurm/ablation/submit_shenlab_ablation.sbatch",
        REPOSITORY_ROOT / "scripts/slurm/ablation/summarize_inference_runs.sbatch",
    ]
    for job in jobs:
        text = job.read_text(encoding="utf-8")
        assert "runtime_environment.sh" in text
        assert 'deepms_verify_guarded_source_tree "${PROJECT_ROOT}"' in text
        assert ".venv/bin/python" not in text


def test_guarded_source_check_accepts_exact_commit_with_clean_worktree(tmp_path: Path) -> None:
    completed = _run_guarded_source_check(tmp_path)

    assert completed.returncode == 0, completed.stderr


def test_guarded_source_check_is_a_noop_without_expected_commit(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "DEEPMS_GIT_BIN": str(tmp_path / "missing-git"),
            "FAKE_PROJECT_ROOT": str(tmp_path),
            "RUNTIME_ENVIRONMENT_PATH": str(RUNTIME_ENVIRONMENT),
        }
    )
    environment.pop("DEEPMS_GUARDED_EXPECTED_COMMIT", None)

    completed = subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
source "${RUNTIME_ENVIRONMENT_PATH}"
deepms_verify_guarded_source_tree "${FAKE_PROJECT_ROOT}"
""",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_guarded_source_check_rejects_commit_drift(tmp_path: Path) -> None:
    completed = _run_guarded_source_check(tmp_path, current_commit="b" * 40)

    assert completed.returncode != 0
    assert "Repository HEAD changed after guarded submission" in completed.stderr


def test_guarded_source_check_rejects_dirty_worktree(tmp_path: Path) -> None:
    completed = _run_guarded_source_check(tmp_path, git_status=" M train.py")

    assert completed.returncode != 0
    assert "Repository worktree changed after guarded submission" in completed.stderr


def test_shenlab_setup_uses_each_collaborators_own_uv_installation() -> None:
    profile = SHENLAB_PROFILE.read_text(encoding="utf-8")
    assert 'DEEPMS_UV_BIN="${DEEPMS_UV_BIN:-uv}"' in profile

    user_setup_files = [
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "docs" / "ENVIRONMENT.md",
        REPOSITORY_ROOT / "scripts" / "slurm" / "ablation" / "README.md",
        SHENLAB_PROFILE,
    ]
    for path in user_setup_files:
        text = path.read_text(encoding="utf-8")
        assert "/gpfs/data/shenlab/Jiajian/software" not in text
