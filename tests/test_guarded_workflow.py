import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GUARDED_WORKFLOW = REPOSITORY_ROOT / "scripts/slurm/ablation/guarded_shenlab_ablation.sh"
COMMIT = "7861ecd32d56ffdf947578cb95d81418d1589877"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _fake_environment(
    tmp_path: Path,
    *,
    git_status: str = "",
    upstream_commit: str = COMMIT,
    reject_test_number: int = 0,
) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "sbatch.log"
    call_counter = tmp_path / "sbatch.counter"
    call_counter.write_text("0", encoding="utf-8")
    output_root = tmp_path / "outputs"
    state_file = output_root / "slurm/pipelines/guarded-test/pipeline_jobs.env"

    fake_git = fake_bin / "git"
    _write_executable(
        fake_git,
        f"""#!/usr/bin/env bash
set -euo pipefail
case "$*" in
    *" status --porcelain --untracked-files=normal") printf '%s' {shlex.quote(git_status)} ;;
    *" rev-parse --abbrev-ref --symbolic-full-name @{{upstream}}") printf 'origin/main\\n' ;;
    *" rev-parse --verify HEAD") printf '{COMMIT}\\n' ;;
    *" rev-parse --verify origin/main") printf '{upstream_commit}\\n' ;;
    *" fetch --quiet origin") ;;
    *) printf 'unexpected fake git call: %s\\n' "$*" >&2; exit 2 ;;
esac
""",
    )
    _write_executable(fake_bin / "uv", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "scontrol", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "scancel", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "squeue",
        """#!/usr/bin/env bash
if [[ " $* " == *" --format=%T "* ]]; then
    printf 'PENDING\\n'
    exit 0
fi
printf '1001_[0-11%%4]|PENDING|Resources\\n'
printf '1002_[0-11%%4]|PENDING|Dependency\\n'
printf '1003_[0-11%%4]|PENDING|Dependency\\n'
printf '1004|PENDING|Dependency\\n'
printf '1005|PENDING|Dependency\\n'
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/usr/bin/env bash
set -euo pipefail
count="$(( $(<"${FAKE_SBATCH_COUNTER}") + 1 ))"
printf '%s' "${count}" > "${FAKE_SBATCH_COUNTER}"
printf '%q ' "$@" >> "${FAKE_SBATCH_LOG}"
printf '\\n' >> "${FAKE_SBATCH_LOG}"
if [[ " $* " == *" --test-only "* ]]; then
    if [[ "${FAKE_REJECT_TEST_NUMBER:-0}" == "${count}" ]]; then
        printf 'requested node configuration is not available\\n' >&2
        exit 1
    fi
    printf 'Job 900%s to start at 2030-01-01T00:00:00 using 1 processor on nodes a100-4001\\n' "${count}"
    exit 0
fi
mkdir -p "$(dirname -- "${FAKE_STATE_FILE}")"
printf '%s\\n' \
    'PIPELINE_STATUS=submitting_held' \
    'TRAIN_JOB_ID=1001' \
    'UNMASKED_INFER_JOB_ID=1002' \
    'MASKED_INFER_JOB_ID=1003' \
    'DATASET_SUMMARY_JOB_ID=1004' \
    'MASKING_SUMMARY_JOB_ID=1005' \
    'PIPELINE_STATUS=scheduled' > "${FAKE_STATE_FILE}"
printf '2000;test-cluster\\n'
""",
    )

    inputs: dict[str, str] = {}
    for name in (
        "DEEPMS_TRAIN_CSV",
        "DEEPMS_VAL_CSV",
        "DEEPMS_PRETRAINED_PATH",
        "DEEPMS_PUBLIC_EXTERNAL_TEST_CSV",
    ):
        path = tmp_path / name.lower()
        path.write_text("fixture\n", encoding="utf-8")
        inputs[name] = str(path)

    profile = tmp_path / "shenlab.env"
    profile_values = {
        "DEEPMS_PROJECT_ROOT": str(REPOSITORY_ROOT),
        "DEEPMS_GIT_BIN": str(fake_git),
        "DEEPMS_UV_BIN": str(fake_bin / "uv"),
        **inputs,
        "DEEPMS_PIPELINE_MODE": "both",
        "DEEPMS_ABLATION_INFER_PROFILE": "public_external_unmasked",
        "DEEPMS_VAL_NUM_WORKERS": "0",
        "DEEPMS_EXPECTED_GPUS": "2",
        "DEEPMS_ABLATION_CONCURRENCY": "4",
        "DEEPMS_TRAIN_EXCLUDE_NODES": "a100-4011,a100-4012,a100-4024,a100-4033",
    }
    profile.write_text(
        "\n".join(f"export {name}={shlex.quote(value)}" for name, value in profile_values.items()) + "\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment.get('PATH', '')}",
            "DEEPMS_SITE_PROFILE": str(profile),
            "DEEPMS_SUBMISSION_ID": "guarded-test",
            "DEEPMS_SUBMISSION_OUTPUT_ROOT": str(output_root),
            "FAKE_SBATCH_COUNTER": str(call_counter),
            "FAKE_SBATCH_LOG": str(command_log),
            "FAKE_STATE_FILE": str(state_file),
            "FAKE_REJECT_TEST_NUMBER": str(reject_test_number),
        }
    )
    return environment, command_log, state_file


def _run(
    tmp_path: Path,
    *arguments: str,
    **environment_options: object,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    environment, command_log, state_file = _fake_environment(tmp_path, **environment_options)
    completed = subprocess.run(
        ["bash", str(GUARDED_WORKFLOW), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    commands = []
    if command_log.exists():
        commands = [shlex.split(line) for line in command_log.read_text(encoding="utf-8").splitlines()]
    return completed, commands, state_file


@pytest.mark.parametrize("arguments", [(), ("--check",)])
def test_default_checks_every_request_without_creating_jobs(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    completed, commands, state_file = _run(tmp_path, *arguments)

    assert completed.returncode == 0, completed.stderr
    assert len(commands) == 4
    assert all("--test-only" in command for command in commands)
    assert not state_file.exists()
    assert "CHECK-ONLY" in completed.stdout


def test_submit_waits_for_one_launcher_and_verifies_the_child_graph(tmp_path: Path) -> None:
    completed, commands, state_file = _run(tmp_path, "--submit")

    assert completed.returncode == 0, completed.stderr
    assert len(commands) == 5
    assert all("--test-only" in command for command in commands[:4])
    actual = commands[-1]
    assert "--wait" in actual
    assert "--parsable" in actual
    assert any(item.startswith("--job-name=deepms-submit-guarded-test") for item in actual)
    export = next(item for item in actual if item.startswith("--export="))
    assert f"DEEPMS_GUARDED_EXPECTED_COMMIT={COMMIT}" in export
    assert "DEEPMS_ABLATION_EVAL_ID=guarded-test" in export
    assert state_file.is_file()
    assert "DeepMS graph is live in Slurm" in completed.stdout
    assert "1001,1002,1003,1004,1005" in completed.stdout


def test_scheduler_rejection_blocks_real_submission(tmp_path: Path) -> None:
    completed, commands, state_file = _run(tmp_path, "--submit", reject_test_number=3)

    assert completed.returncode != 0
    assert len(commands) == 3
    assert all("--test-only" in command for command in commands)
    assert not state_file.exists()
    assert "Slurm rejected inference-array" in completed.stderr


@pytest.mark.parametrize(
    ("environment_options", "expected"),
    [
        ({"git_status": " M train.py"}, "worktree is not clean"),
        ({"upstream_commit": "0" * 40}, "Run git pull --ff-only"),
    ],
)
def test_git_preflight_failure_creates_no_slurm_requests(
    tmp_path: Path,
    environment_options: dict[str, object],
    expected: str,
) -> None:
    completed, commands, state_file = _run(tmp_path, "--submit", **environment_options)

    assert completed.returncode != 0
    assert commands == []
    assert not state_file.exists()
    assert expected in completed.stderr
