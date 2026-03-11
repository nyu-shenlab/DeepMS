from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd, Transform


LOGGER = logging.getLogger(__name__)

# turn dmri into target range

# =========================================================
# Modality groups
# =========================================================
TARGET_MODALITIES: List[str] = [
    "fa_dti",
    "md_dti",
    "rd_dti",
    "ad_dti",
    "Da_smi",
    "DePar_smi",
    "DePerp_smi",
    "f_smi",
    "p2_smi",
    "ak_wdki",
    "rk_wdki",
    "mk_wdki",
    "ak_dki",
    "rk_dki",
    "mk_dki",
    "fa_dki",
    "md_dki",
    "rd_dki",
    "ad_dki",
    "md_wdki",
    "rd_wdki",
    "ad_wdki",
]

UNIT_RANGE_MODALITIES = {
    "fa_dti",
    "f_smi",
    "p2_smi",
    "fa_dki",
    "fa_wdki",
}

THREE_RANGE_MODALITIES = {
    "md_dti",
    "rd_dti",
    "ad_dti",
    "Da_smi",
    "DePar_smi",
    "DePerp_smi",
    "md_dki",
    "rd_dki",
    "ad_dki",
    "md_wdki",
    "rd_wdki",
    "ad_wdki",
    "ak_wdki",
    "mk_wdki",
    "ak_dki",
    "mk_dki",
}

FIVE_RANGE_MODALITIES = {
    "rk_wdki",
    "rk_dki",
}

B0_MODALITIES = {"b0"}

RESCALE_BY_3_MODALITIES = {
    "md_dti",
    "rd_dti",
    "ad_dti",
    "Da_smi",
    "DePar_smi",
    "DePerp_smi",
    "md_dki",
    "rd_dki",
    "ad_dki",
    "md_wdki",
    "rd_wdki",
    "ad_wdki",
    "ak_wdki",
    "mk_wdki",
    "ak_dki",
    "mk_dki",
}

RESCALE_BY_5_MODALITIES = {
    "rk_wdki",
    "rk_dki",
}


# =========================================================
# Custom transforms
# =========================================================
class NanToZero(Transform):
    """Replace NaN values in image tensor with 0."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data["image"]
        data["image"] = torch.nan_to_num(image, nan=0.0)
        return data


class ClipToRange(Transform):
    """Clip image intensity range according to modality."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        mod = data["modality"]
        image = data["image"]

        if mod in UNIT_RANGE_MODALITIES:
            image = image.clamp(0, 1)
        elif mod in THREE_RANGE_MODALITIES:
            image = image.clamp(0, 3)
        elif mod in FIVE_RANGE_MODALITIES:
            image = image.clamp(0, 5)
        elif mod in B0_MODALITIES:
            image = image.clamp(0, 800)
        else:
            # Fallback: avoid negative values, preserve dynamic upper bound
            max_val = torch.max(image)
            if torch.isfinite(max_val):
                image = image.clamp(0, max_val)
            else:
                image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        data["image"] = image
        return data


class RescaleByModality(Transform):
    """Rescale image values according to modality."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        mod = data["modality"]
        image = data["image"]

        if mod in RESCALE_BY_3_MODALITIES:
            image = image / 3.0
        elif mod in RESCALE_BY_5_MODALITIES:
            image = image / 5.0

        data["image"] = image
        return data


TRANSFORMS = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NanToZero(),
        Orientationd(keys=["image"], axcodes="RAS"),
        ClipToRange(),
        RescaleByModality(),
    ]
)


# =========================================================
# Utility functions
# =========================================================
def setup_logging(verbose: bool = False) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_output_path(output_base_path: str | Path, patient_id: Any, modality: str) -> Path:
    """
    Build output path:
        {output_base_path}/{patient_id}/processed_params/{modality}.nii
    """
    output_dir = Path(output_base_path) / str(patient_id) / "processed_params"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{modality}.nii"


def infer_missing_modality_path(patient_df: pd.DataFrame, modality: str) -> Optional[str]:
    """
    Infer missing modality path by replacing 'fa_dti' from an existing fa_dti path.
    """
    fa_row = patient_df[patient_df["modality"] == "fa_dti"]
    if fa_row.empty:
        return None

    fa_path = str(fa_row["preprocessing"].iloc[0])
    return fa_path.replace("fa_dti", modality)


def infer_b0_path(patient_df: pd.DataFrame) -> Optional[str]:
    """
    Infer B0 path from fa_dti path:
        .../params/...fa_dti... -> .../b0/...b0bc...
    """
    fa_row = patient_df[patient_df["modality"] == "fa_dti"]
    if fa_row.empty:
        return None

    sample_path = str(fa_row["preprocessing"].iloc[0])
    sample_mod = str(fa_row["modality"].iloc[0])
    return sample_path.replace(sample_mod, "b0bc").replace("params", "b0")


def load_and_transform_image(input_path: str | Path, modality: str) -> torch.Tensor:
    """
    Load image with MONAI pipeline and return transformed tensor / MetaTensor.
    """
    result = TRANSFORMS({"image": str(input_path), "modality": modality})
    return result["image"]


def extract_affine(image: Any) -> np.ndarray:
    """
    Safely extract affine from MONAI MetaTensor if present; otherwise use identity.
    """
    affine = getattr(image, "affine", None)
    if affine is None:
        return np.eye(4, dtype=np.float32)

    if isinstance(affine, torch.Tensor):
        return affine.detach().cpu().numpy()

    return np.asarray(affine)


def tensor_to_numpy_3d(image: torch.Tensor) -> np.ndarray:
    """
    Convert [C, H, W, D] or [1, H, W, D] tensor to 3D numpy array.
    """
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]

    return array.astype(np.float32)


def save_nifti(image: torch.Tensor, output_path: str | Path) -> None:
    """
    Save MONAI tensor / MetaTensor as NIfTI.
    """
    affine = extract_affine(image)
    array = tensor_to_numpy_3d(image)
    nib.save(nib.Nifti1Image(array, affine), str(output_path))


def make_new_row(template_row: pd.Series, modality: str, old_preprocessing: str, new_preprocessing: str) -> pd.Series:
    """
    Create a new dataset row for generated modality.
    """
    row = template_row.copy()
    row["modality"] = modality
    row["preprocessing"] = old_preprocessing
    row["latest_preprocessing"] = new_preprocessing
    return row


def apply_mask_if_available(image: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Apply mask if provided.
    """
    if mask is None:
        return image
    return image * mask


# =========================================================
# Core preprocessing
# =========================================================
def preprocess_single_patient(
    patient_id: Any,
    patient_df: pd.DataFrame,
    output_base_path: str | Path,
) -> Tuple[Dict[str, str], List[pd.Series]]:
    """
    Process one patient.
    
    Returns
    -------
    updated_existing : dict
        modality -> saved output path for existing rows
    new_rows : list[pd.Series]
        newly created rows for inferred modalities / B0
    """
    updated_existing: Dict[str, str] = {}
    new_rows: List[pd.Series] = []

    image_dict: Dict[str, torch.Tensor] = {}
    brain_mask: Optional[torch.Tensor] = None

    LOGGER.info("Processing patient ID: %s", patient_id)

    # -----------------------------------------------------
    # Process target diffusion modalities
    # -----------------------------------------------------
    for modality in TARGET_MODALITIES:
        mod_row = patient_df[patient_df["modality"] == modality]

        if not mod_row.empty:
            input_path = str(mod_row["preprocessing"].iloc[0])
            row_exists = True
        else:
            inferred_path = infer_missing_modality_path(patient_df, modality)
            if inferred_path is None:
                LOGGER.warning("Skipping %s for %s: fa_dti template not found.", modality, patient_id)
                continue
            input_path = inferred_path
            row_exists = False

        if not Path(input_path).exists():
            LOGGER.warning("Input file not found for %s (%s): %s", patient_id, modality, input_path)
            continue

        try:
            image = load_and_transform_image(input_path, modality)
            image_dict[modality] = image

            if modality == "md_dti":
                brain_mask = ((image > 0) & (image < 0.8)).to(image.dtype)

            if modality == "fa_dti":
                wm_mask = (image > 0.15).to(torch.int32)
                image_dict["wm_mask"] = wm_mask

        except Exception as exc:
            LOGGER.exception("Failed processing %s for %s: %s", modality, patient_id, exc)
            continue

        # Save immediately after successful processing
        try:
            image_to_save = apply_mask_if_available(image_dict[modality], brain_mask)
            output_path = get_output_path(output_base_path, patient_id, modality)
            save_nifti(image_to_save, output_path)
            LOGGER.info("Saved %s to %s", modality, output_path)

            if row_exists:
                updated_existing[modality] = str(output_path)
            else:
                template_row = patient_df.iloc[0]
                old_preproc = str(output_path).replace("processed_params", "params")
                new_rows.append(
                    make_new_row(
                        template_row=template_row,
                        modality=modality,
                        old_preprocessing=old_preproc,
                        new_preprocessing=str(output_path),
                    )
                )
        except Exception as exc:
            LOGGER.exception("Failed saving %s for %s: %s", modality, patient_id, exc)

    # -----------------------------------------------------
    # Save wm_mask if fa_dti existed
    # -----------------------------------------------------
    if "wm_mask" in image_dict:
        try:
            wm_mask = apply_mask_if_available(image_dict["wm_mask"], brain_mask)
            output_path = get_output_path(output_base_path, patient_id, "wm_mask")
            save_nifti(wm_mask, output_path)
            LOGGER.info("Saved wm_mask to %s", output_path)

            if "wm_mask" in patient_df["modality"].values:
                updated_existing["wm_mask"] = str(output_path)
            else:
                template_row = patient_df.iloc[0]
                old_preproc = str(output_path).replace("processed_params", "params")
                new_rows.append(
                    make_new_row(
                        template_row=template_row,
                        modality="wm_mask",
                        old_preprocessing=old_preproc,
                        new_preprocessing=str(output_path),
                    )
                )
        except Exception as exc:
            LOGGER.exception("Failed saving wm_mask for %s: %s", patient_id, exc)

    # -----------------------------------------------------
    # Process B0
    # -----------------------------------------------------
    b0_input_path = infer_b0_path(patient_df)
    if b0_input_path is not None:
        if Path(b0_input_path).exists():
            try:
                b0_image = load_and_transform_image(b0_input_path, "b0")
                b0_image = apply_mask_if_available(b0_image, brain_mask)

                b0_output_path = get_output_path(output_base_path, patient_id, "b0")
                save_nifti(b0_image, b0_output_path)
                LOGGER.info("Saved b0 to %s", b0_output_path)

                if "b0" in patient_df["modality"].values:
                    updated_existing["b0"] = str(b0_output_path)
                else:
                    template_row = patient_df.iloc[0]
                    new_rows.append(
                        make_new_row(
                            template_row=template_row,
                            modality="b0",
                            old_preprocessing=b0_input_path,
                            new_preprocessing=str(b0_output_path),
                        )
                    )
            except Exception as exc:
                LOGGER.exception("Failed processing B0 for %s: %s", patient_id, exc)
        else:
            LOGGER.warning("B0 image not found for %s: %s", patient_id, b0_input_path)

    return updated_existing, new_rows


def preprocess_images(args: argparse.Namespace) -> pd.DataFrame:
    """
    Main preprocessing entry point.
    """
    dataset = pd.read_csv(args.dataset_path).copy()

    required_columns = {"m_id", "modality", "preprocessing"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset CSV: {sorted(missing_columns)}")

    dataset["latest_preprocessing"] = dataset["preprocessing"].copy()

    all_new_rows: List[pd.Series] = []
    patient_ids = sorted(dataset["m_id"].dropna().unique())

    LOGGER.info("Found %d unique patients.", len(patient_ids))

    for patient_id in patient_ids:
        patient_df = dataset[dataset["m_id"] == patient_id]

        updated_existing, new_rows = preprocess_single_patient(
            patient_id=patient_id,
            patient_df=patient_df,
            output_base_path=args.output_base_path,
        )

        # Update existing rows
        for modality, new_path in updated_existing.items():
            mask = (dataset["m_id"] == patient_id) & (dataset["modality"] == modality)
            dataset.loc[mask, "latest_preprocessing"] = new_path

        all_new_rows.extend(new_rows)

    # Append newly generated rows
    if all_new_rows:
        new_df = pd.DataFrame(all_new_rows)
        dataset = pd.concat([dataset, new_df], ignore_index=True)

    # Finalize output columns
    dataset = dataset.sort_values(by=["m_id", "modality"]).reset_index(drop=True)
    dataset = dataset.rename(columns={"preprocessing": "old_preprocessing"})
    dataset = dataset.rename(columns={"latest_preprocessing": "preprocessing"})

    output_csv_path = Path(args.output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_csv_path, index=False)

    LOGGER.info("Updated dataset saved to %s", output_csv_path)
    return dataset


# =========================================================
# CLI
# =========================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess diffusion MRI images for GitHub-ready pipeline.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/gpfs/data/shenlab/Jiajian/MS_Project/code/ms-diagnosis/meta_data/updated_label_dataset/train_dataset_all.csv",
        help="Path to the dataset CSV file.",
    )
    parser.add_argument(
        "--output_base_path",
        type=str,
        default="/gpfs/scratch/jm10850/ms_data/MESO_V2.0/PROCESSED_OUTPUT",
        help="Base path for saving processed images.",
    )
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default="/gpfs/data/shenlab/Jiajian/MS_Project/code/ms-diagnosis/meta_data/updated_label_dataset/train_dataset_all_latest.csv",
        help="Path to save the updated dataset CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    preprocess_images(args)


if __name__ == "__main__":
    main()