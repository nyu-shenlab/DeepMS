import json
from pathlib import Path

import pandas as pd
import pytest

from summarize_inference_runs import (
    discover_inference_runs,
    summarize_inference_runs,
)
from utils.analysis import sigmoid


def prediction_row(
    row_id: int,
    m_id: str,
    modality: str,
    label: int,
    score: float,
    *,
    dataset: str = "Synthetic",
    used_masked_image: bool = False,
) -> dict:
    return {
        "row_id": row_id,
        "m_id": m_id,
        "patient_id": m_id,
        "dataset": dataset,
        "modality": modality,
        "ms": label,
        "label": label,
        "ms_logits": score,
        "ms_prob": float(sigmoid([score])[0]),
        "used_masked_image": used_masked_image,
        "used_preprocessing_fallback": False,
    }


def predictions(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            prediction_row(0, "negative", "3DFLAIR_NCE", 0, -2.0 + offset),
            prediction_row(1, "negative", "3DT1_CE", 0, -1.0 + offset),
            prediction_row(2, "positive", "3DFLAIR_NCE", 1, 2.0 + offset),
            prediction_row(3, "positive", "3DT1_CE", 1, 1.0 + offset),
        ]
    )


def masking_predictions(*, masked: bool) -> pd.DataFrame:
    scores = (2.0, -2.0) if masked else (-2.0, 2.0)
    return pd.DataFrame(
        [
            prediction_row(
                0,
                "negative",
                "3DFLAIR_NCE",
                0,
                scores[0],
                dataset="WMH",
                used_masked_image=masked,
            ),
            prediction_row(
                1,
                "positive",
                "3DFLAIR_NCE",
                1,
                scores[1],
                dataset="WMH",
                used_masked_image=masked,
            ),
        ]
    )


def write_completed_run(
    root: Path,
    run_id: str,
    frame: pd.DataFrame,
    *,
    complete: bool = True,
    profile: str = "generic",
    cohort_overrides_configured: bool = False,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    frame.to_csv(run_dir / "prediction_all_modalities.csv", index=False)
    coverage = {
        "inference_complete": complete,
        "predicted_rows": len(frame),
        "performance_report_profile": profile,
        "performance_report_status": "deferred",
        "image_policy": {
            "internal": "preprocessing",
            "krakow": "preprocessing",
            "public_external_unmasked": "preprocessing",
            "public_external_masked": "masked_image_path_then_preprocessing",
        }.get(profile, "non-preprocessing"),
        "report_configuration": {
            "cohort_overrides_configured": cohort_overrides_configured,
        },
    }
    (run_dir / "coverage.json").write_text(
        json.dumps(coverage),
        encoding="utf-8",
    )
    return run_dir


def test_final_summary_waits_for_all_runs_and_writes_gated_artifacts(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    write_completed_run(runs_root, "SMI/Da_smi", predictions())
    write_completed_run(runs_root, "DTI/ad_dti", predictions(offset=0.1))
    output_dir = tmp_path / "summary"

    report = summarize_inference_runs(
        runs_root=runs_root,
        output_dir=output_dir,
        expected_runs=2,
        bootstrap_samples=0,
    )

    assert report["n_runs"] == 2
    assert report["evaluation_mode"] == "dataset_calibrated"
    assert report["result_key"] == "ablation_results.smri_notebook_temperature"
    assert report["calibration_policy"] == "fixed temperatures from the reference notebook"
    assert report["completeness_gate"] == "passed"
    assert report["patient_label_contract_gate"] == "passed"
    assert [row["run_id"] for row in report["runs"]] == [
        "SMI/Da_smi",
        "DTI/ad_dti",
    ]
    assert len({row["patient_label_fingerprint"] for row in report["runs"]}) == 1
    assert all(row["cohort"] == "notebook_primary" for row in report["runs"])
    assert all(row["ensemble"] == "sMRI" for row in report["runs"])
    assert all(row["calibration"] == "notebook_temperature" for row in report["runs"])
    assert report["pairwise_deltas"] == []
    assert {path.name for path in output_dir.iterdir()} == {
        "ablation_performance_summary.csv",
        "ablation_performance_metrics.csv",
        "ablation_performance_report.json",
        "ablation_performance_report.md",
        "_SUCCESS",
    }
    parsed = json.loads((output_dir / "ablation_performance_report.json").read_text())
    assert parsed["n_runs"] == 2


def test_final_summary_rejects_partial_collection(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_completed_run(runs_root, "SMI/Da_smi", predictions())

    with pytest.raises(ValueError, match="Expected exactly 12"):
        discover_inference_runs(runs_root, expected_runs=12)


def test_final_summary_rejects_incomplete_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_completed_run(
        runs_root,
        "SMI/Da_smi",
        predictions(),
        complete=False,
    )

    with pytest.raises(ValueError, match="inference_complete=true"):
        discover_inference_runs(runs_root)


def test_final_summary_rejects_patient_or_label_drift(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_completed_run(runs_root, "SMI/Da_smi", predictions())
    changed = predictions(offset=0.1)
    changed.loc[changed["m_id"].eq("positive"), ["ms", "label"]] = 0
    write_completed_run(runs_root, "DTI/ad_dti", changed)

    with pytest.raises(ValueError, match="exact same patients and labels"):
        summarize_inference_runs(
            runs_root=runs_root,
            output_dir=tmp_path / "summary",
            expected_runs=2,
            bootstrap_samples=0,
        )


def test_final_summary_requires_matching_internal_sidecar_configuration(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    write_completed_run(
        runs_root,
        "SMI/Da_smi",
        predictions(),
        profile="internal",
        cohort_overrides_configured=True,
    )

    with pytest.raises(ValueError, match="cohort-override configuration"):
        summarize_inference_runs(
            runs_root=runs_root,
            output_dir=tmp_path / "summary",
            bootstrap_samples=0,
        )


def test_final_summary_does_not_overwrite_an_existing_report(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    write_completed_run(runs_root, "SMI/Da_smi", predictions())
    output_dir = tmp_path / "summary"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        summarize_inference_runs(
            runs_root=runs_root,
            output_dir=output_dir,
            bootstrap_samples=0,
        )
    assert sentinel.read_text(encoding="utf-8") == "existing"


def test_masking_summary_is_raw_paired_and_writes_deltas(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_completed_run(
        runs_root,
        "public_external_unmasked/SMI/Da_smi",
        masking_predictions(masked=False),
        profile="public_external_unmasked",
    )
    write_completed_run(
        runs_root,
        "public_external_masked/SMI/Da_smi",
        masking_predictions(masked=True),
        profile="public_external_masked",
    )
    output_dir = tmp_path / "masking-summary"

    report = summarize_inference_runs(
        runs_root=runs_root,
        output_dir=output_dir,
        evaluation_mode="masking_raw",
        expected_runs=2,
        bootstrap_samples=0,
    )

    assert report["evaluation_mode"] == "masking_raw"
    assert report["result_key"] == "masking_comparison"
    assert report["calibration_policy"] == "none; raw logits"
    assert report["pairwise_delta_definition"] == "masked minus unmasked"
    assert {row["profile"] for row in report["runs"]} == {
        "public_external_unmasked",
        "public_external_masked",
    }
    assert all(row["cohort"] == "masking_comparable" for row in report["runs"])
    assert all(row["ensemble"] == "FLAIR" for row in report["runs"])
    assert all(row["calibration"] == "raw" for row in report["runs"])
    assert len({row["patient_label_fingerprint"] for row in report["runs"]}) == 1

    assert len(report["pairwise_deltas"]) == 1
    paired = report["pairwise_deltas"][0]
    assert paired["comparison_unit"] == "SMI/Da_smi"
    assert paired["roc_auc_unmasked"] == pytest.approx(1.0)
    assert paired["roc_auc_masked"] == pytest.approx(0.0)
    assert paired["roc_auc_masked_minus_unmasked"] == pytest.approx(-1.0)
    assert (output_dir / "masking_pairwise_deltas.csv").is_file()
    assert {path.name for path in output_dir.iterdir()} == {
        "ablation_performance_summary.csv",
        "masking_pairwise_deltas.csv",
        "ablation_performance_metrics.csv",
        "ablation_performance_report.json",
        "ablation_performance_report.md",
        "_SUCCESS",
    }


def test_masking_summary_requires_both_profiles(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_completed_run(
        runs_root,
        "public_external_unmasked/SMI/Da_smi",
        masking_predictions(masked=False),
        profile="public_external_unmasked",
    )

    with pytest.raises(ValueError, match="requires both"):
        summarize_inference_runs(
            runs_root=runs_root,
            output_dir=tmp_path / "summary",
            evaluation_mode="masking_raw",
            expected_runs=1,
            bootstrap_samples=0,
        )


def test_calibrated_dataset_mode_rejects_masked_profile(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_completed_run(
        runs_root,
        "public_external_masked/SMI/Da_smi",
        masking_predictions(masked=True),
        profile="public_external_masked",
    )

    with pytest.raises(ValueError, match="not allowed"):
        summarize_inference_runs(
            runs_root=runs_root,
            output_dir=tmp_path / "summary",
            evaluation_mode="dataset_calibrated",
            expected_runs=1,
            bootstrap_samples=0,
        )


def test_masking_summary_rejects_cross_policy_label_drift(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_completed_run(
        runs_root,
        "public_external_unmasked/SMI/Da_smi",
        masking_predictions(masked=False),
        profile="public_external_unmasked",
    )
    changed = masking_predictions(masked=True)
    changed.loc[changed["m_id"].eq("positive"), ["ms", "label"]] = 0
    write_completed_run(
        runs_root,
        "public_external_masked/SMI/Da_smi",
        changed,
        profile="public_external_masked",
    )

    with pytest.raises(ValueError, match="exact same patients and labels"):
        summarize_inference_runs(
            runs_root=runs_root,
            output_dir=tmp_path / "summary",
            evaluation_mode="masking_raw",
            expected_runs=2,
            bootstrap_samples=0,
        )
