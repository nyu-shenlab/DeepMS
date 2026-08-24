import os
import subprocess
import sys
from pathlib import Path


def test_training_validation_runs_and_gathers_across_two_processes() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    worker = repository_root / "tests" / "_distributed_gather_worker.py"
    environment = os.environ.copy()
    python_paths = [str(repository_root)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(worker),
        ],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
