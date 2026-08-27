import re
import shlex
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPOSITORY_ROOT / "scripts" / "slurm"
ABLATION_SLURM_DIR = SLURM_DIR / "ablation"


def extract_array(text: str, name: str) -> list[str]:
    match = re.search(rf"{name}=\((?P<body>.*?)\)", text, flags=re.DOTALL)
    assert match is not None, f"Missing shell array: {name}"
    return shlex.split(match.group("body"), comments=True)


def read_slurm(filename: str) -> str:
    return (SLURM_DIR / filename).read_text(encoding="utf-8")


def test_ablation_jobs_are_grouped_in_a_dedicated_subfolder() -> None:
    public_jobs = {path.name for path in ABLATION_SLURM_DIR.glob("*.sbatch") if not path.name.endswith("_local.sbatch")}
    assert public_jobs == {
        "infer_diffusion_ablation.sbatch",
        "run_diffusion_ablation_pipeline.sbatch",
        "submit_shenlab_ablation.sbatch",
        "summarize_inference_runs.sbatch",
        "train_diffusion_ablation.sbatch",
    }
    assert not list(SLURM_DIR.glob("*ablation*.sbatch"))
    assert "sbatch scripts/slurm/ablation/submit_shenlab_ablation.sbatch" in (
        ABLATION_SLURM_DIR / "README.md"
    ).read_text(encoding="utf-8")


def test_generic_training_job_does_not_require_legacy_clinical_csvs() -> None:
    text = read_slurm("train.sbatch")
    assert "DEEPMS_DIAGNOSIS_CSV" not in text
    assert "DEEPMS_WM_LESION_CSV" not in text
    assert "DEEPMS_MODALITIES" in text
    assert "DEEPMS_VAL_MODALITIES" in text
    assert "DEEPMS_EARLY_STOPPING_EPOCHS" in text
    assert "DEEPMS_VAL_INTERVAL" in text
    assert "DEEPMS_SAVE_INTERVAL" in text
    assert '--early_stopping_epochs "${EARLY_STOPPING_EPOCHS}"' in text
    assert '--val_interval "${VAL_INTERVAL}"' in text
    assert '--save_interval "${SAVE_INTERVAL}"' in text


def test_shenlab_site_profile_is_copy_ready() -> None:
    text = (REPOSITORY_ROOT / "configs" / "slurm.shenlab.env").read_text(encoding="utf-8")

    assert 'export DEEPMS_PROJECT_ROOT="$(pwd -P)"' in text
    assert (
        'export DEEPMS_TRAIN_CSV="/gpfs/data/shenlab/Jiajian/MS_Project/code/'
        "ms-diagnosis/meta_data/updated_label_dataset/"
        'train_dataset_all_latest_1230_ADNI_updated.csv"'
    ) in text
    assert (
        'export DEEPMS_VAL_CSV="/gpfs/data/shenlab/Jiajian/MS_Project/code/'
        "ms-diagnosis/meta_data/updated_label_dataset/reg/"
        'validation_dataset_all_latest_updated.csv"'
    ) in text
    assert (
        'export DEEPMS_PUBLIC_EXTERNAL_TEST_CSV="/gpfs/data/shenlab/Jiajian/'
        'MS_Project/ms_data/external_dataset/lesion_filling/0_filled_all_dil_3.csv"'
    ) in text
    assert "export DEEPMS_EARLY_STOPPING_EPOCHS=5" in text
    assert "export DEEPMS_VAL_INTERVAL=1" in text
    assert "export DEEPMS_SAVE_INTERVAL=5" in text
    assert "export DEEPMS_PIPELINE_MODE=both" in text
    assert "export DEEPMS_INCLUDE_B0=1" in text
    assert "export DEEPMS_SAVE_VISUALIZATIONS=0" in text
    assert "\nexport DEEPMS_REPORT_COHORT_OVERRIDES=" not in text

    readme = (ABLATION_SLURM_DIR / "README.md").read_text(encoding="utf-8")
    assert "sbatch --test-only scripts/slurm/ablation/submit_shenlab_ablation.sbatch" in readme
    assert "sbatch scripts/slurm/ablation/submit_shenlab_ablation.sbatch" in readme
    assert "sbatch --test-only scripts/slurm/ablation/run_diffusion_ablation_pipeline.sbatch" in readme
    assert "sbatch scripts/slurm/ablation/run_diffusion_ablation_pipeline.sbatch" in readme


def test_shenlab_direct_submit_job_loads_the_committed_profile() -> None:
    text = read_slurm("ablation/submit_shenlab_ablation.sbatch")

    assert "#SBATCH --partition=cpu_short" in text
    assert "#SBATCH --gres" not in text
    assert "#SBATCH --array" not in text
    assert 'PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"' in text
    assert "configs/slurm.shenlab.env" in text
    assert 'source "${SITE_PROFILE}"' in text
    assert "run_diffusion_ablation_pipeline.sbatch" in text
    assert 'exec bash "${PIPELINE_JOB}"' in text


def test_diffusion_ablation_order_and_single_map_contract() -> None:
    text = read_slurm("ablation/train_diffusion_ablation.sbatch")
    assert "#SBATCH --array=0-11%4" in text
    assert extract_array(text, "DIFFUSION_MAPS") == [
        "Da_smi",
        "DePar_smi",
        "DePerp_smi",
        "f_smi",
        "p2_smi",
        "ad_dti",
        "fa_dti",
        "md_dti",
        "rd_dti",
        "ak_wdki",
        "mk_wdki",
        "rk_wdki",
    ]
    assert extract_array(text, "DIFFUSION_FAMILIES") == [
        "SMI",
        "SMI",
        "SMI",
        "SMI",
        "SMI",
        "DTI",
        "DTI",
        "DTI",
        "DTI",
        "DKI",
        "DKI",
        "DKI",
    ]
    assert "${STRUCTURAL_MODALITIES[*]} ${DIFFUSION_MAP}" in text
    assert "${STRUCTURAL_VAL_MODALITIES[*]} ${DIFFUSION_MAP}" in text
    assert "${ABLATION_OUTPUT_ROOT}/${DIFFUSION_FAMILY}/${DIFFUSION_MAP}" in text


def test_diffusion_ablation_inference_and_final_summary_contract() -> None:
    training = read_slurm("ablation/train_diffusion_ablation.sbatch")
    inference = read_slurm("ablation/infer_diffusion_ablation.sbatch")
    summary = read_slurm("ablation/summarize_inference_runs.sbatch")

    assert "#SBATCH --array=0-11%4" in inference
    assert extract_array(inference, "DIFFUSION_MAPS") == extract_array(training, "DIFFUSION_MAPS")
    assert extract_array(inference, "DIFFUSION_FAMILIES") == extract_array(training, "DIFFUSION_FAMILIES")
    assert "find " in inference
    assert "-name best_model.pth" in inference
    assert "DEEPMS_DEFER_PERFORMANCE_REPORT=1" in inference
    assert "DEEPMS_ABLATION_INFER_PROFILE" in inference

    assert "#SBATCH --partition=cpu_short" in summary
    assert "#SBATCH --gres" not in summary
    assert "summarize_inference_runs.py" in summary
    assert "DEEPMS_EVALUATION_MODE:-dataset_calibrated" in summary
    assert "DEFAULT_EXPECTED_RUNS=12" in summary
    assert "DEFAULT_EXPECTED_RUNS=24" in summary
    assert "DEEPMS_EXPECTED_INFERENCE_RUNS:-${DEFAULT_EXPECTED_RUNS}" in summary
    assert '--evaluation_mode "${EVALUATION_MODE}"' in summary
    assert "summary-${EVALUATION_MODE}" in summary


def test_one_command_pipeline_is_cpu_only_and_dependency_driven() -> None:
    pipeline = read_slurm("ablation/run_diffusion_ablation_pipeline.sbatch")

    assert "#SBATCH --partition=cpu_short" in pipeline
    assert "#SBATCH --gres" not in pipeline
    assert "#SBATCH --array" not in pipeline
    assert 'ABLATION_SLURM_DIR="${PROJECT_ROOT}/scripts/slurm/ablation"' in pipeline
    assert "train_diffusion_ablation.sbatch" in pipeline
    assert "infer_diffusion_ablation.sbatch" in pipeline
    assert "summarize_inference_runs.sbatch" in pipeline
    assert "DEEPMS_PIPELINE_MODE:-dataset_calibrated" in pipeline
    assert "dataset_calibrated|masking_raw|both" in pipeline
    assert "DEEPMS_ABLATION_CONCURRENCY:-4" in pipeline
    assert 'ARRAY_SPEC="0-11%${CONCURRENCY}"' in pipeline
    assert '--dependency="afterok:${training_job_id}"' in pipeline
    assert '"afterok:${UNMASKED_INFER_JOB_ID}:${MASKED_INFER_JOB_ID}"' in pipeline
    assert "DEEPMS_ABLATION_OUTPUT_ROOT=${TRAIN_ROOT}" in pipeline
    assert "DEEPMS_ABLATION_CHECKPOINT_ROOT=${TRAIN_ROOT}" in pipeline
    assert "PIPELINE_STATUS=scheduled" in pipeline
    assert "runtime_environment.sh" in pipeline


def test_inference_profiles_are_explicit_and_disjoint() -> None:
    krakow = read_slurm("infer_krakow.sbatch")
    public_external_unmasked = read_slurm("infer_public_external_unmasked.sbatch")
    public_external_masked = read_slurm("infer_public_external_lesion_masked.sbatch")

    assert not (SLURM_DIR / "infer_external.sbatch").exists()

    assert "DEEPMS_KRAKOW_TEST_CSV" in krakow
    assert "DEEPMS_PUBLIC_EXTERNAL_TEST_CSV" not in krakow
    assert extract_array(krakow, "MODALITIES") == [
        "3DT1_NCE",
        "3DT1_CE",
        "3DFLAIR_NCE",
    ]
    assert "--use_preprocess" in krakow
    assert "--use_mask_img" not in krakow
    assert "outputs/inference/krakow" in krakow

    expected_external_modalities = [
        "2DFLAIR_NCE",
        "2DT1_NCE",
        "3DT1_NCE",
        "3DT1_CE",
        "3DFLAIR_NCE",
    ]
    for profile in (public_external_unmasked, public_external_masked):
        assert "DEEPMS_PUBLIC_EXTERNAL_TEST_CSV" in profile
        assert "DEEPMS_KRAKOW_TEST_CSV" not in profile
        assert extract_array(profile, "MODALITIES") == expected_external_modalities
        assert "DEEPMS_USE_MASKED_IMAGES" not in profile

    assert "--use_preprocess" in public_external_unmasked
    assert "--use_mask_img" not in public_external_unmasked
    assert "public-external-unmasked" in public_external_unmasked

    assert "--use_mask_img" in public_external_masked
    assert "--use_preprocess" not in public_external_masked
    assert "public-external-lesion-masked" in public_external_masked


def test_all_inference_profiles_are_single_gpu_and_have_manifest_preflight() -> None:
    profiles = {
        "infer_internal.sbatch": "internal",
        "infer_krakow.sbatch": "krakow",
        "infer_public_external_unmasked.sbatch": "public_external_unmasked",
        "infer_public_external_lesion_masked.sbatch": "public_external_masked",
    }
    for filename, report_profile in profiles.items():
        text = read_slurm(filename)
        assert "#SBATCH --gres=gpu:1" in text
        assert "DEEPMS_PREFLIGHT_ONLY" in text
        assert "--preflight_only" in text
        assert "accelerate launch" not in text
        assert f"--report_profile {report_profile}" in text
        assert "DEEPMS_DEFER_PERFORMANCE_REPORT" in text
        assert '"${REPORT_ARGS[@]}"' in text
        assert '--report_bootstrap_samples "${DEEPMS_REPORT_BOOTSTRAPS:-2000}"' in text


def test_visualization_is_opt_in_for_every_inference_profile() -> None:
    for filename in (
        "infer_internal.sbatch",
        "infer_krakow.sbatch",
        "infer_public_external_unmasked.sbatch",
        "infer_public_external_lesion_masked.sbatch",
    ):
        text = read_slurm(filename)
        assert "${DEEPMS_SAVE_VISUALIZATIONS:-0}" in text
        assert "${DEEPMS_SAVE_VISUALIZATIONS:-1}" not in text
        assert "VISUALIZATION_ARGS=()" in text
        assert '"${VISUALIZATION_ARGS[@]}"' in text


def test_public_slurm_assets_do_not_embed_site_specific_paths() -> None:
    public_assets = [
        SLURM_DIR / "train.sbatch",
        ABLATION_SLURM_DIR / "train_diffusion_ablation.sbatch",
        SLURM_DIR / "infer_internal.sbatch",
        SLURM_DIR / "infer_krakow.sbatch",
        SLURM_DIR / "infer_public_external_unmasked.sbatch",
        ABLATION_SLURM_DIR / "infer_diffusion_ablation.sbatch",
        ABLATION_SLURM_DIR / "summarize_inference_runs.sbatch",
        ABLATION_SLURM_DIR / "submit_shenlab_ablation.sbatch",
        SLURM_DIR / "infer_public_external_lesion_masked.sbatch",
        REPOSITORY_ROOT / "configs" / "slurm.env.example",
        ABLATION_SLURM_DIR / "run_diffusion_ablation_pipeline.sbatch",
        SLURM_DIR / "runtime_environment.sh",
        REPOSITORY_ROOT / "scripts" / "bootstrap_env.sh",
        REPOSITORY_ROOT / "scripts" / "check_environment.py",
    ]
    for path in public_assets:
        assert "/gpfs/" not in path.read_text(encoding="utf-8"), path
