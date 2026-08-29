import json

import pytest

from train import write_training_completion


def test_training_completion_manifest_is_written_atomically(tmp_path) -> None:
    (tmp_path / "best_model.pth").write_bytes(b"checkpoint")
    (tmp_path / "final_model.pth").write_bytes(b"checkpoint")

    completion_path = write_training_completion(
        str(tmp_path),
        last_epoch=8,
        best_metric=0.91,
        best_metric_epoch=6,
        completed_updates=120,
        selection_metric="micro",
    )

    payload = json.loads((tmp_path / "training_complete.json").read_text(encoding="utf-8"))
    assert completion_path == str(tmp_path / "training_complete.json")
    assert payload == {
        "schema_version": 1,
        "status": "complete",
        "last_epoch": 8,
        "best_metric": 0.91,
        "best_metric_epoch": 6,
        "completed_updates": 120,
        "selection_metric": "micro",
        "best_model": "best_model.pth",
        "final_model": "final_model.pth",
    }
    assert not list(tmp_path.glob("*.tmp-*"))


def test_training_completion_requires_a_best_model(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="best_model.pth"):
        write_training_completion(
            str(tmp_path),
            last_epoch=1,
            best_metric=-1.0,
            best_metric_epoch=0,
            completed_updates=10,
            selection_metric="micro",
        )
