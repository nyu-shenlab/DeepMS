from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class SingleModalityDataset(Dataset):
    """
    Dataset for loading single-modality neuroimaging samples from a pandas DataFrame.

    Expected columns
    ----------------
    Required:
        - image
        - modality
        - label
        - m_id

    Optional:
        - preprocessing
        - bet
        - SMI
        - mask_path
        - masked_image_path
        - modality_label

    Notes
    -----
    - If `train=True` and `use_both=True`, the dataset randomly chooses between
      `preprocessing` and `bet` as the input image for each sample.
    - If `SMI != 0`, a white matter mask path is inferred by replacing the
      modality substring in the image path with "wm_mask".
    - If `mask_path == 1`, the dataset searches for lesion masks in the same
      directory as `masked_image_path`, preferring:
          1) lesion_mask_new.nii.gz
          2) lesion_mask.nii.gz
    """

    def __init__(
        self,
        data: pd.DataFrame,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        train: bool = True,
        use_both: bool = False,
    ) -> None:
        self.data = data.copy().reset_index(drop=True)
        self.transform = transform
        self.train = train
        self.use_both = use_both

        if "mask_path" not in self.data.columns:
            self.data["mask_path"] = 0

        self._validate_columns()

    def _validate_columns(self) -> None:
        """Validate that the required columns exist."""
        required_columns = {"image", "modality", "label", "m_id"}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in dataset: {sorted(missing_columns)}"
            )

        if self.use_both:
            needed_for_dual_input = {"preprocessing", "bet"}
            missing_dual_columns = needed_for_dual_input - set(self.data.columns)
            if missing_dual_columns:
                raise ValueError(
                    "When use_both=True, the following columns are required: "
                    f"{sorted(missing_dual_columns)}"
                )

    def __len__(self) -> int:
        return len(self.data)

    def _select_image_path(self, sample: Dict[str, Any]) -> str:
        """
        Select the image path for the current sample.

        During training with `use_both=True`, randomly choose between
        `preprocessing` and `bet`. Otherwise, use the existing `image` field.
        """
        if self.train and self.use_both:
            return sample["preprocessing"] if random.random() < 0.5 else sample["bet"]
        return sample["image"]

    def _build_wm_mask_path(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Build the white matter mask path.

        Returns None if SMI == 0.
        """
        if sample.get("SMI", 0) == 0:
            return None

        image_path = sample["image"]
        modality = sample["modality"]

        if not isinstance(image_path, str) or not isinstance(modality, str):
            raise ValueError(
                f"Invalid image/modality for m_id={sample.get('m_id')}: "
                f"image={image_path}, modality={modality}"
            )

        return image_path.replace(modality, "wm_mask")

    def _find_lesion_mask(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Find the lesion mask path if mask_path == 1.

        Search order:
            1) lesion_mask_new.nii.gz
            2) lesion_mask.nii.gz
        """
        if sample.get("mask_path", 0) != 1:
            return None

        masked_image_path = sample.get("masked_image_path")
        if not masked_image_path:
            raise ValueError(
                f"'masked_image_path' is required when mask_path == 1 "
                f"(m_id={sample.get('m_id')})."
            )

        sample_dir = os.path.dirname(masked_image_path)
        candidate_paths = [
            os.path.join(sample_dir, "lesion_mask_new.nii.gz"),
            os.path.join(sample_dir, "lesion_mask.nii.gz"),
        ]

        for path in candidate_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Lesion mask not found for m_id={sample.get('m_id')} "
            f"in directory: {sample_dir}"
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load one sample and apply optional transforms.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the transformed sample and standardized keys.
        """
        sample = self.data.iloc[idx].to_dict()

        # Select input image path
        sample["image"] = self._select_image_path(sample)

        # Build auxiliary paths
        sample["wm_mask"] = self._build_wm_mask_path(sample)
        sample["lesion_mask"] = self._find_lesion_mask(sample)

        # Apply transform if provided
        if self.transform is not None:
            sample = self.transform(sample)

        # Standardize label type
        label = sample["label"]
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = label.to(torch.float32)

        # Put **sample first, then overwrite key fields with standardized outputs
        # so that important fields (especially `label`) are not accidentally replaced.
        return {
            **sample,
            "image": sample["image"],
            "lesion_mask": sample.get("lesion_mask"),
            "modality": sample["modality"],
            "modality_label": sample.get("modality_label"),
            "m_id": sample["m_id"],
            "label": label,
        }


def collate_skip_none(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Collate function that skips samples that are None.

    Parameters
    ----------
    batch : list
        A list of samples returned by the dataset.

    Returns
    -------
    dict or None
        - Collated batch if there is at least one valid sample
        - None if all samples are None or if collation fails
    """
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    try:
        return default_collate(batch)
    except Exception as exc:
        print(f"[collate_skip_none] Error during collation: {exc}", flush=True)
        print(f"[collate_skip_none] Batch content: {batch}", flush=True)
        return None