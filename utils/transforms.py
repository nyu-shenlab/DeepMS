"""
MONAI transforms used for MRI preprocessing and augmentation.

This module keeps the original behavior of your pipeline while making the code
cleaner and easier to maintain for a public GitHub repository:
- clearer class/function names
- shared helper utilities
- less duplicated logic for mask loading
- more explicit interpolation modes for images vs. masks
- metadata-preserving tensor updates
- logging-based error reporting
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandAxisFlipd,
    RandBiasFieldd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Resized,
    ScaleIntensityd,
    ScaleIntensityRanged,
)

logger = logging.getLogger(__name__)


def _as_tensor(x: torch.Tensor | MetaTensor) -> torch.Tensor:
    """Return the plain tensor view of a tensor or MetaTensor."""
    return x.as_tensor() if isinstance(x, MetaTensor) else x


def _with_same_meta(
    reference: torch.Tensor | MetaTensor,
    tensor: torch.Tensor,
) -> torch.Tensor | MetaTensor:
    """Wrap `tensor` with the metadata stored in `reference` when possible."""
    if isinstance(reference, MetaTensor):
        return MetaTensor(tensor, meta=reference.meta.copy())
    return tensor


def percentile_clip(image: torch.Tensor, q: float = 0.999) -> torch.Tensor:
    """
    Clip image intensities to the q-th quantile of positive voxels.

    This is kept as a utility because it may be useful later, even though it is
    not currently used in the active augmentation pipeline.
    """
    positive = image[image > 0]
    if positive.numel() == 0:
        return image

    clip_value = torch.quantile(positive.float(), q)
    return torch.clamp(image, max=clip_value.to(image.dtype))


PercentileClipd = Lambdad(keys=["image"], func=percentile_clip)


class NanToZeroD(MapTransform):
    """Replace NaN values with zeros while preserving MONAI metadata."""

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            reference = d[key]
            cleaned = torch.nan_to_num(_as_tensor(reference), nan=0.0)
            d[key] = _with_same_meta(reference, cleaned)
        return d


class GetNonBrainMaskD(MapTransform):
    """
    Build:
    - non_brain_mask: voxels equal to zero in the image
    - L1_mask: voxels not equal to zero in the image
    """

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            reference = d[key]
            tensor = _as_tensor(reference)

            non_brain_mask = (tensor == 0).to(dtype=tensor.dtype)
            l1_mask = (tensor != 0).to(dtype=tensor.dtype)

            d["non_brain_mask"] = _with_same_meta(reference, non_brain_mask)
            d["L1_mask"] = _with_same_meta(reference, l1_mask)

        return d


class _LoadOptionalMaskD(MapTransform):
    """
    Base class for loading optional mask files.

    If the provided mask path is None, a fallback mask is created from the input
    image shape:
    - all ones for wm_mask
    - all zeros for lesion_mask
    """

    def __init__(
        self,
        keys,
        output_key: str,
        default_fill_value: float,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        if len(self.keys) != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects exactly one input key, got {self.keys}."
            )

        self.input_key = self.keys[0]
        self.output_key = output_key
        self.default_fill_value = default_fill_value
        self.loader = LoadImaged(keys=[self.input_key], ensure_channel_first=True)

    def _fallback_mask(
        self,
        reference: torch.Tensor | MetaTensor,
    ) -> torch.Tensor | MetaTensor:
        fallback = torch.full_like(
            _as_tensor(reference),
            fill_value=self.default_fill_value,
            dtype=torch.float32,
        )
        return _with_same_meta(reference, fallback)

    @staticmethod
    def _validate_mask_shape(mask: torch.Tensor | MetaTensor, path: str) -> None:
        """
        Expect mask shape to be [C, H, W, D] after `ensure_channel_first=True`.

        This raises a clear error instead of silently guessing what to do with
        extra dimensions.
        """
        if mask.ndim != 4:
            raise ValueError(
                f"Expected mask '{path}' to have shape [C, H, W, D] after loading, "
                f"but got shape {tuple(mask.shape)}."
            )

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = dict(data)
        path = d.get(self.input_key)

        if path is None:
            if "image" not in d:
                raise KeyError(
                    f"Missing reference key 'image' while building fallback for '{self.output_key}'."
                )
            d[self.output_key] = self._fallback_mask(d["image"])
            return d

        loaded = self.loader({self.input_key: path})
        mask = loaded[self.input_key]
        self._validate_mask_shape(mask, path)

        d[self.output_key] = mask
        return d


class GetWhiteMatterMaskD(_LoadOptionalMaskD):
    """Load WM mask or create an all-ones fallback mask."""

    def __init__(self, keys, allow_missing_keys: bool = False) -> None:
        super().__init__(
            keys=keys,
            output_key="wm_mask",
            default_fill_value=1.0,
            allow_missing_keys=allow_missing_keys,
        )


class GetLesionMaskD(_LoadOptionalMaskD):
    """Load lesion mask or create an all-zeros fallback mask."""

    def __init__(self, keys, allow_missing_keys: bool = False) -> None:
        super().__init__(
            keys=keys,
            output_key="lesion_mask",
            default_fill_value=0.0,
            allow_missing_keys=allow_missing_keys,
        )


class InvertBinaryMaskD(MapTransform):
    """Invert a binary mask and preserve metadata."""

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            reference = d[key]
            tensor = _as_tensor(reference)
            inverted = (1.0 - tensor > 0.5).to(dtype=tensor.dtype)
            d[key] = _with_same_meta(reference, inverted)
        return d


class FilterImages:
    """
    Main transform wrapper used by the dataset.

    Parameters
    ----------
    dat_type:
        'trn' for training transforms; any other value selects validation transforms.
    args:
        Namespace-like object that provides:
        - roi_x, roi_y, roi_z
        - resize_size
    """

    def __init__(self, dat_type: str, args: Any) -> None:
        self.img_size = self._resolve_image_size(args)

        self.transforms_smri = self._build_pipeline(
            is_train=(dat_type == "trn"),
            is_structural=True,
        )
        self.transforms_dmri = self._build_pipeline(
            is_train=(dat_type == "trn"),
            is_structural=False,
        )

    @staticmethod
    def _resolve_image_size(args: Any) -> tuple[int, int, int]:
        """Resolve final ROI size from explicit ROI args or fallback resize size."""
        if args.roi_x is None or args.roi_y is None or args.roi_z is None:
            return (args.resize_size, args.resize_size, args.resize_size)
        return (args.roi_x, args.roi_y, args.roi_z)

    def _build_common_transforms(self) -> list:
        """Transforms shared by training and validation."""
        return [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NanToZeroD(keys=["image"]),
            GetWhiteMatterMaskD(keys=["wm_mask"]),
            GetLesionMaskD(keys=["lesion_mask"]),
            Orientationd(keys=["image", "wm_mask", "lesion_mask"], axcodes="RAS"),
            CropForegroundd(
                keys=["image", "wm_mask", "lesion_mask"],
                source_key="image",
            ),
            Resized(
                keys=["image", "wm_mask", "lesion_mask"],
                spatial_size=self.img_size,
                mode=("trilinear", "nearest", "nearest"),
            ),
            InvertBinaryMaskD(keys=["wm_mask"]),
            GetNonBrainMaskD(keys=["image"]),
        ]

    @staticmethod
    def _build_train_augmentation() -> list:
        """Data augmentation applied only during training."""
        mask_keys = ["non_brain_mask", "wm_mask", "L1_mask", "lesion_mask"]
        all_keys = ["image", *mask_keys]

        return [
            RandAxisFlipd(keys=all_keys, prob=0.5),
            RandAffined(
                keys=all_keys,
                rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
                translate_range=(5, 5, 5),
                mode=("bilinear", "nearest", "nearest", "nearest", "nearest"),
                prob=0.5,
            ),
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.3), prob=0.3),
            RandBiasFieldd(keys=["image"], prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.3),
            RandGaussianSmoothd(keys=["image"], prob=0.3),
        ]

    @staticmethod
    def _build_normalization(is_structural: bool):
        """Choose the normalization strategy based on MRI modality."""
        if is_structural:
            return ScaleIntensityd(keys=["image"])

        return ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=1,
            b_min=0,
            b_max=1,
            clip=True,
        )

    def _build_pipeline(self, is_train: bool, is_structural: bool) -> Compose:
        """Assemble a full MONAI Compose pipeline."""
        transforms = self._build_common_transforms()

        if is_train:
            transforms.extend(self._build_train_augmentation())

        transforms.append(self._build_normalization(is_structural=is_structural))
        return Compose(transforms)

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any] | None:
        """
        Process one sample.

        Expected input keys include:
        - image
        - wm_mask
        - lesion_mask
        - label
        - structural_mri (optional, defaults to 1)
        """
        image_path = data.get("image")
        wm_mask_path = data.get("wm_mask")
        lesion_mask_path = data.get("lesion_mask")

        transform = (
            self.transforms_smri
            if data.get("structural_mri", 1) == 1
            else self.transforms_dmri
        )

        payload = {
            "image": image_path,
            "wm_mask": wm_mask_path,
            "lesion_mask": lesion_mask_path,
            "label": data.get("label"),
        }

        try:
            processed = transform(payload)
        except Exception:
            logger.exception(
                "Failed to process sample. image=%s, wm_mask=%s, lesion_mask=%s",
                image_path,
                wm_mask_path,
                lesion_mask_path,
            )
            return None

        output = dict(data)
        output.update(
            {
                "image": processed["image"],
                "non_brain_mask": processed["non_brain_mask"],
                "wm_mask": processed["wm_mask"],
                "L1_mask": processed["L1_mask"],
                "lesion_mask": processed["lesion_mask"],
            }
        )
        return output


# ---------------------------------------------------------------------
# Backward-compatible aliases
# Keep these aliases if other parts of your project still import the old names.
# ---------------------------------------------------------------------
NanToZerod = NanToZeroD
get_non_brain_mask = GetNonBrainMaskD
get_wm_mask = GetWhiteMatterMaskD
get_lesion_mask = GetLesionMaskD
invert_wm_mask = InvertBinaryMaskD
