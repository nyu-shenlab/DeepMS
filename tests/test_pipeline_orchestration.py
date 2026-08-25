import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = REPOSITORY_ROOT / "scripts" / "slurm" / "ablation" / "run_diffusion_ablation_pipeline.sbatch"
SHENLAB_LAUNCHER = REPOSITORY_ROOT / "scripts/slurm/ablation/submit_shenlab_ablation.sbatch"


def _fake_pipeline_environment(
    tmp_path: Path,
    *,
    mode: str,
    profile: str = "public_external_unmasked",
) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "sbatch-counter"
    counter.write_text("1000", encoding="utf-8")
    command_log = tmp_path / "sbatch-commands.log"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
job_id="$(( $(<"${FAKE_SBATCH_COUNTER}") + 1 ))"
printf '%s' "${job_id}" > "${FAKE_SBATCH_COUNTER}"
printf '%q ' "$@" >> "${FAKE_SBATCH_LOG}"
printf '\n' >> "${FAKE_SBATCH_LOG}"
printf '%s;test-cluster\n' "${job_id}"
""",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)

    input_files: dict[str, str] = {}
    for variable in (
        "DEEPMS_TRAIN_CSV",
        "DEEPMS_VAL_CSV",
        "DEEPMS_PRETRAINED_PATH",
        "DEEPMS_INTERNAL_TEST_CSV",
        "DEEPMS_KRAKOW_TEST_CSV",
        "DEEPMS_PUBLIC_EXTERNAL_TEST_CSV",
    ):
        path = tmp_path / f"{variable.lower()}.csv"
        path.write_text("fixture\n", encoding="utf-8")
        input_files[variable] = str(path)

    evaluation_id = f"test-{mode}"
    state_root = tmp_path / "state"
    environment = os.environ.copy()
    environment.update(
        {
            **input_files,
            "PATH": f"{fake_bin}:{environment.get('PATH', '')}",
            "FAKE_SBATCH_COUNTER": str(counter),
            "FAKE_SBATCH_LOG": str(command_log),
            "SLURM_JOB_ID": "9000",
            "SLURM_SUBMIT_DIR": str(REPOSITORY_ROOT),
            "DEEPMS_PROJECT_ROOT": str(REPOSITORY_ROOT),
            "DEEPMS_UV_BIN": "true",
            "DEEPMS_PIPELINE_MODE": mode,
            "DEEPMS_ABLATION_INFER_PROFILE": profile,
            "DEEPMS_ABLATION_EVAL_ID": evaluation_id,
            "DEEPMS_ABLATION_CONCURRENCY": "3",
            "DEEPMS_PIPELINE_TRAIN_ROOT": str(tmp_path / "train" / evaluation_id),
            "DEEPMS_ABLATION_INFERENCE_ROOT": str(tmp_path / "inference"),
            "DEEPMS_PIPELINE_STATE_ROOT": str(state_root),
        }
    )
    state_file = state_root / evaluation_id / "pipeline_jobs.env"
    return environment, command_log, state_file


def _run_fake_pipeline(
    tmp_path: Path,
    *,
    mode: str,
    profile: str = "public_external_unmasked",
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    environment, command_log, state_file = _fake_pipeline_environment(
        tmp_path,
        mode=mode,
        profile=profile,
    )
    completed = subprocess.run(
        ["bash", str(PIPELINE_SCRIPT)],
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


def _export_argument(command: list[str]) -> str:
    return next(argument for argument in command if argument.startswith("--export="))


def test_dataset_pipeline_submits_train_infer_summary_dependency_chain(
    tmp_path: Path,
) -> None:
    completed, commands, state_file = _run_fake_pipeline(
        tmp_path,
        mode="dataset_calibrated",
        profile="krakow",
    )

    assert completed.returncode == 0, completed.stderr
    assert len(commands) == 3
    training, inference, summary = commands
    assert "--array=0-11%3" in training
    assert training[-1].endswith("train_diffusion_ablation.sbatch")
    assert "--dependency=afterok:1001" in inference
    assert "DEEPMS_ABLATION_INFER_PROFILE=krakow" in _export_argument(inference)
    assert inference[-1].endswith("infer_diffusion_ablation.sbatch")
    assert "--dependency=afterok:1002" in summary
    summary_export = _export_argument(summary)
    assert "DEEPMS_EVALUATION_MODE=dataset_calibrated" in summary_export
    assert "DEEPMS_EXPECTED_INFERENCE_RUNS=12" in summary_export
    assert "/krakow" in summary_export
    assert summary[-1].endswith("summarize_inference_runs.sbatch")

    state = state_file.read_text(encoding="utf-8")
    assert "PIPELINE_STATUS=scheduled" in state
    assert "TRAIN_JOB_ID=1001" in state
    assert "DATASET_INFER_JOB_ID=1002" in state
    assert "DATASET_SUMMARY_JOB_ID=1003" in state


def test_both_pipeline_reuses_unmasked_inference_and_schedules_two_summaries(
    tmp_path: Path,
) -> None:
    completed, commands, state_file = _run_fake_pipeline(tmp_path, mode="both")

    assert completed.returncode == 0, completed.stderr
    assert len(commands) == 5
    training, unmasked, masked, dataset_summary, masking_summary = commands
    assert training[-1].endswith("train_diffusion_ablation.sbatch")
    assert "--dependency=afterok:1001" in unmasked
    assert "--dependency=afterok:1001" in masked
    assert "DEEPMS_ABLATION_INFER_PROFILE=public_external_unmasked" in _export_argument(unmasked)
    assert "DEEPMS_ABLATION_INFER_PROFILE=public_external_masked" in _export_argument(masked)

    assert "--dependency=afterok:1002" in dataset_summary
    assert "DEEPMS_EVALUATION_MODE=dataset_calibrated" in _export_argument(dataset_summary)
    assert "--dependency=afterok:1002:1003" in masking_summary
    masking_export = _export_argument(masking_summary)
    assert "DEEPMS_EVALUATION_MODE=masking_raw" in masking_export
    assert "DEEPMS_EXPECTED_INFERENCE_RUNS=24" in masking_export

    state = state_file.read_text(encoding="utf-8")
    assert "PIPELINE_STATUS=scheduled" in state
    assert "UNMASKED_INFER_JOB_ID=1002" in state
    assert "MASKED_INFER_JOB_ID=1003" in state
    assert "DATASET_SUMMARY_JOB_ID=1004" in state
    assert "MASKING_SUMMARY_JOB_ID=1005" in state


def test_shenlab_launcher_loads_a_site_profile_and_submits_the_full_graph(
    tmp_path: Path,
) -> None:
    environment, command_log, state_file = _fake_pipeline_environment(
        tmp_path,
        mode="both",
    )
    profile_keys = [
        "DEEPMS_PROJECT_ROOT",
        "DEEPMS_UV_BIN",
        "DEEPMS_TRAIN_CSV",
        "DEEPMS_VAL_CSV",
        "DEEPMS_PRETRAINED_PATH",
        "DEEPMS_INTERNAL_TEST_CSV",
        "DEEPMS_KRAKOW_TEST_CSV",
        "DEEPMS_PUBLIC_EXTERNAL_TEST_CSV",
        "DEEPMS_PIPELINE_MODE",
        "DEEPMS_ABLATION_INFER_PROFILE",
        "DEEPMS_ABLATION_EVAL_ID",
        "DEEPMS_ABLATION_CONCURRENCY",
        "DEEPMS_PIPELINE_TRAIN_ROOT",
        "DEEPMS_ABLATION_INFERENCE_ROOT",
        "DEEPMS_PIPELINE_STATE_ROOT",
    ]
    site_profile = tmp_path / "site-profile.env"
    site_profile.write_text(
        "\n".join(f"export {key}={shlex.quote(environment.pop(key))}" for key in profile_keys) + "\n",
        encoding="utf-8",
    )
    environment["DEEPMS_SITE_PROFILE"] = str(site_profile)

    completed = subprocess.run(
        ["bash", str(SHENLAB_LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    commands = [shlex.split(line) for line in command_log.read_text(encoding="utf-8").splitlines()]
    assert len(commands) == 5
    assert "Loaded Shenlab profile" in completed.stdout
    assert "PIPELINE_STATUS=scheduled" in state_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", ["unknown", "dataset_calibrated"])
def test_pipeline_preflight_failure_submits_no_child_jobs(
    tmp_path: Path,
    mode: str,
) -> None:
    profile = "public_external_masked" if mode == "dataset_calibrated" else "public_external_unmasked"
    completed, commands, state_file = _run_fake_pipeline(
        tmp_path,
        mode=mode,
        profile=profile,
    )

    assert completed.returncode != 0
    assert commands == []
    assert not state_file.exists()
