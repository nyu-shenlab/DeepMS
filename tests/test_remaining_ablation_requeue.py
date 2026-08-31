from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_JOB = REPOSITORY_ROOT / "scripts" / "slurm" / "ablation" / "train_remaining_diffusion_ablation.sbatch"


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fake'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "slurm.shenlab.env").write_text(
        'export DEEPMS_PROJECT_ROOT="$(pwd -P)"\n',
        encoding="utf-8",
    )

    ablation_dir = tmp_path / "scripts" / "slurm" / "ablation"
    ablation_dir.mkdir(parents=True)
    (ablation_dir / "train_diffusion_ablation.sbatch").write_text(
        '#!/usr/bin/env bash\nexit "${FAKE_TRAIN_EXIT:-0}"\n',
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_scontrol = bin_dir / "scontrol"
    fake_scontrol.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "${FAKE_SCONTROL_LOG:?}"\nexit "${FAKE_SCONTROL_EXIT:-0}"\n',
        encoding="utf-8",
    )
    fake_scontrol.chmod(0o755)
    return tmp_path


def run_recovery_job(
    project: Path,
    *,
    train_exit: int,
    restart_count: int = 0,
    max_requeues: int = 3,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    scontrol_log = project / "scontrol.log"
    env = os.environ.copy()
    env.update(
        {
            "DEEPMS_CUDA_MAX_REQUEUES": str(max_requeues),
            "FAKE_SCONTROL_LOG": str(scontrol_log),
            "FAKE_TRAIN_EXIT": str(train_exit),
            "PATH": f"{project / 'bin'}:{env['PATH']}",
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_RESTART_COUNT": str(restart_count),
            "SLURM_SUBMIT_DIR": str(project),
            "SLURMD_NODENAME": "a100-test",
        }
    )
    completed = subprocess.run(
        ["bash", str(RECOVERY_JOB)],
        cwd=project,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, scontrol_log


def test_cuda_preflight_failure_requeues_only_the_array_element(
    fake_project: Path,
) -> None:
    target = fake_project / "outputs" / "train" / "diffusion_single_map" / "remaining-9maps-12345" / "SMI" / "Da_smi"
    target.mkdir(parents=True)

    completed, scontrol_log = run_recovery_job(fake_project, train_exit=75)

    assert completed.returncode == 0, completed.stderr
    assert scontrol_log.read_text(encoding="utf-8") == "requeue 12345_0\n"
    assert "requeueing 12345_0 (1/3)" in completed.stdout


def test_non_cuda_failure_is_not_requeued(fake_project: Path) -> None:
    completed, scontrol_log = run_recovery_job(fake_project, train_exit=1)

    assert completed.returncode == 1
    assert not scontrol_log.exists()


def test_cuda_preflight_retry_limit_is_fail_closed(fake_project: Path) -> None:
    completed, scontrol_log = run_recovery_job(
        fake_project,
        train_exit=75,
        restart_count=3,
    )

    assert completed.returncode == 75
    assert not scontrol_log.exists()
    assert "CUDA preflight still fails after 3 requeues" in completed.stderr


def test_nonempty_map_directory_is_never_reused(fake_project: Path) -> None:
    target = fake_project / "outputs" / "train" / "diffusion_single_map" / "remaining-9maps-12345" / "SMI" / "Da_smi"
    target.mkdir(parents=True)
    (target / "partial-checkpoint.txt").write_text("do not overwrite\n", encoding="utf-8")

    completed, scontrol_log = run_recovery_job(fake_project, train_exit=75)

    assert completed.returncode == 1
    assert not scontrol_log.exists()
    assert "non-empty output directory" in completed.stderr
