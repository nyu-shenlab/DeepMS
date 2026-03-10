

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.append(os.getcwd())

from model.Models import VisualEncoder
from utils.analysis import avg_logits_ensemble, grouped_avg_prob_ensemble_smri
from utils.dataset import SingleModalityDataset, collate_skip_none
from utils.transforms import FilterImages


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STRUCTURAL_MRI_LIST = [
    "3DFLAIR_NCE", "3DFLAIR_CE", "3DT1_NCE", "3DT1_CE", "3DT2_NCE", "3DT2_CE",
    "2DFLAIR_NCE", "2DFLAIR_CE", "2DT1_NCE", "2DT1_CE", "2DT2_NCE", "2DT2_CE",
    "b0",
]

SMI_LIST = [
    "Da_smi", "DePar_smi", "DePerp_smi", "f_smi", "p2_smi", "p4_smi",
]

DTI_LIST = ["fa_dti", "ad_dti", "md_dti", "rd_dti"]
DKI_LIST = ["fa_dki", "ad_dki", "md_dki", "rd_dki"]
WDKI_LIST = ["ak_wdki", "rk_wdki", "mk_wdki"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def setup_logging() -> None:
    """Configure root logger."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference for saliency-based multi-modality MRI classification models."
    )

    # Data / metadata
    parser.add_argument(
        "--base_root",
        type=str,
        default="/gpfs/data/shenlab/Jiajian/MS_Project/ms_data/MESO_V2.0/ALLSUBJS_2.0",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--test_patient_ids",
        type=str,
        required=True,
        help="CSV file containing the test metadata.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        required=True,
        help="Modalities to run inference on.",
    )

    # Preprocessing / image selection
    parser.add_argument("--use_preprocess", action="store_true", help="Use preprocessed images.")
    parser.add_argument("--use_mask_img", action="store_true", help="Use lesion-masked images.")
    parser.add_argument("--use_bet_only", action="store_true", help="Use BET-only images.")
    parser.add_argument("--use_both", action="store_true", help="Reserved for compatibility.")
    parser.add_argument("--use_global_transform", action="store_true", help="Passed to transform config.")
    parser.add_argument("--direct_resize", action="store_true", help="Passed to transform config.")

    # Loader / shape
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--resize_size", type=int, default=96, help="Default isotropic resize size.")
    parser.add_argument("--roi_x", type=int, default=None, help="ROI x size.")
    parser.add_argument("--roi_y", type=int, default=None, help="ROI y size.")
    parser.add_argument("--roi_z", type=int, default=None, help="ROI z size.")

    # Model
    parser.add_argument(
        "--model_paths",
        nargs="+",
        required=True,
        help="Model checkpoint path(s). Either one path for all modalities or one per modality.",
    )
    parser.add_argument("--backbone", type=str, default="VoCo_Salient_2", help="Backbone model name.")
    parser.add_argument("--num_channels", type=int, default=1, help="Number of image input channels.")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of saliency heads.")
    parser.add_argument("--num_experts", type=int, default=1, help="Reserved for compatibility.")
    parser.add_argument("--gating_kernel_size", type=int, default=1, help="Reserved for compatibility.")

    # Inference / loss convention
    parser.add_argument(
        "--loss_type",
        type=str,
        default="bce_with_logits",
        choices=["bce", "bce_with_logits"],
        help="How to interpret model score output.",
    )
    parser.add_argument("--use_cis", action="store_true", help="Map CIS label 2 -> positive class 1.")

    # Outputs
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSV outputs.")
    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Save NIfTI visualizations (image / prob_map / attention_map / lesion_mask).",
    )
    parser.add_argument(
        "--visualization_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualization NIfTI files.",
    )

    return parser.parse_args()


def get_image_size(args: argparse.Namespace) -> Tuple[int, int, int]:
    """Return image size based on ROI arguments or default resize."""
    if args.roi_x is None or args.roi_y is None or args.roi_z is None:
        return (args.resize_size, args.resize_size, args.resize_size)
    return (args.roi_x, args.roi_y, args.roi_z)


def normalize_model_paths(model_paths: Sequence[str], modalities: Sequence[str]) -> List[str]:
    """
    Normalize model paths to one-per-modality.

    Rules
    -----
    - If one path is given, reuse it for all modalities.
    - Otherwise, the number of model paths must equal the number of modalities.
    """
    if len(model_paths) == 1:
        logging.info("Using a single checkpoint for all modalities.")
        return list(model_paths) * len(modalities)

    if len(model_paths) != len(modalities):
        raise ValueError(
            f"Expected either 1 model path or {len(modalities)} model paths, "
            f"but got {len(model_paths)}."
        )

    return list(model_paths)


def infer_probability(score: torch.Tensor, loss_type: str) -> torch.Tensor:
    """
    Convert model score to positive-class probability.

    Supported cases
    ---------------
    - BCE: score is already probability
    - BCEWithLogits: apply sigmoid
    """
    if score.dim() == 2 and score.shape[1] == 2:
        return torch.softmax(score, dim=1)[:, 1]

    if score.dim() == 2 and score.shape[1] == 1:
        score = score.squeeze(1)

    if loss_type == "bce":
        return score
    if loss_type == "bce_with_logits":
        return torch.sigmoid(score)

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def safe_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Compute ROC AUC safely; return NaN if invalid."""
    try:
        if len(set(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_prob)
    except Exception as exc:
        logging.warning(f"Failed to compute AUC: {exc}")
        return float("nan")


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def prepare_test_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """
    Prepare test dataframe and align fields with dataset / training expectations.
    """
    if not os.path.exists(args.test_patient_ids):
        raise FileNotFoundError(f"Test CSV not found: {args.test_patient_ids}")

    df = pd.read_csv(args.test_patient_ids)
    df = df[df["modality"].isin(args.modalities)].copy()

    if df.empty:
        raise ValueError("No rows left after filtering requested modalities.")

    # Labels
    if "ms" not in df.columns and "label" in df.columns:
        df["ms"] = df["label"]
    if "label" not in df.columns and "ms" in df.columns:
        df["label"] = df["ms"]

    if "ms" not in df.columns or "label" not in df.columns:
        raise ValueError("Test CSV must contain either 'ms' or 'label'.")

    if args.use_cis:
        df.loc[df["ms"] == 2, "ms"] = 1
        df.loc[df["label"] == 2, "label"] = 1

    df = df[df["label"].isin([0, 1])].copy()

    # Metadata
    df["structural_mri"] = df["modality"].isin(STRUCTURAL_MRI_LIST).astype(int)
    df["SMI"] = df["modality"].isin(SMI_LIST).astype(int)

    modality_to_idx = {m: i for i, m in enumerate(args.modalities)}
    df["modality_label"] = df["modality"].map(modality_to_idx).astype(int)

    # Optional columns
    if "Sex" not in df.columns:
        df["Sex"] = 0
    if "Age" not in df.columns:
        df["Age"] = 0

    df["Sex"] = df["Sex"].fillna(0)
    df["Age"] = df["Age"].fillna(0)
    df["Sex"] = df["Sex"].replace({"F": 1, "M": 2}).astype(int)

    # Lesion-mask availability
    df["mask_path"] = 0
    if "masked_image_path" in df.columns:
        df.loc[df["masked_image_path"].notna(), "mask_path"] = 1
        if "preprocessing" in df.columns:
            df.loc[df["masked_image_path"].isna(), "masked_image_path"] = df["preprocessing"]
    else:
        if "preprocessing" in df.columns:
            df["masked_image_path"] = df["preprocessing"]
        else:
            df["masked_image_path"] = None

    # Image path selection
    if args.use_preprocess:
        required_col = "preprocessing"
        df["image"] = df["preprocessing"]
    elif args.use_bet_only:
        required_col = "bet"
        df["image"] = df["bet"]
    elif args.use_mask_img:
        required_col = "masked_image_path"
        df["image"] = df["masked_image_path"]
    else:
        required_col = "non-preprocessing"
        df["image"] = df["non-preprocessing"]

    if required_col not in df.columns:
        raise ValueError(f"Required column '{required_col}' not found in test CSV.")

    df = df[df["image"].notna()].copy()

    if df.empty:
        raise ValueError("No valid test samples remain after image-path filtering.")

    return df


def build_test_loader(
    modality_df: pd.DataFrame,
    args: argparse.Namespace,
) -> DataLoader:
    """Create test dataset and dataloader for one modality."""
    test_filter_transform = FilterImages(dat_type="vld", args=args)
    dataset = SingleModalityDataset(
        data=modality_df,
        transform=test_filter_transform,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_skip_none,
    )
    return loader


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def build_model(
    args: argparse.Namespace,
    model_path: str,
    image_size: Tuple[int, int, int],
    device: torch.device,
) -> torch.nn.Module:
    """
    Build and load inference model.

    This uses the unified VisualEncoder interface from Models.py.
    """
    logging.info(f"Loading model checkpoint: {model_path}")

    model = VisualEncoder(
        encoder_name=args.backbone,
        in_channels=args.num_channels,
        number_of_classes=2,
        finetuned_backbone=model_path,
        image_size=image_size,
        num_heads=args.num_heads,
    )
    model = model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def _extract_affine_from_image_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract affine from MONAI MetaTensor if available; otherwise return identity.
    """
    try:
        if hasattr(image_tensor, "meta") and isinstance(image_tensor.meta, dict):
            affine = image_tensor.meta.get("affine", None)
            if isinstance(affine, torch.Tensor):
                return affine.detach().cpu().numpy()
            if isinstance(affine, np.ndarray):
                return affine
    except Exception as exc:
        logging.warning(f"Failed to extract affine from MetaTensor: {exc}")

    return np.eye(4, dtype=np.float32)


def save_visualization_batch(
    val_images: torch.Tensor,
    prob_map: torch.Tensor,
    attention_map: torch.Tensor,
    lesion_mask: torch.Tensor,
    mask_path: torch.Tensor,
    m_ids: Sequence,
    modality: str,
    visualization_dir: str,
) -> None:
    """
    Save image / probability map / attention map / lesion mask as NIfTI files.

    Saved files per case
    --------------------
    - {modality}.nii.gz
    - {modality}_prob_map_{i}.nii.gz
    - {modality}_attention_map_{i}.nii.gz
    - lesion_mask.nii.gz (if available)
    """
    batch_size = val_images.shape[0]

    if isinstance(m_ids, torch.Tensor):
        m_ids = [item.item() if hasattr(item, "item") else item for item in m_ids]
    m_ids = [str(x) for x in m_ids]

    if len(m_ids) != batch_size:
        logging.error(
            f"Visualization skipped due to mismatched batch size: "
            f"{batch_size} images vs {len(m_ids)} IDs."
        )
        return

    prob_map_np = prob_map.detach().cpu().numpy()
    attention_map_np = attention_map.detach().cpu().numpy()
    lesion_mask_np = lesion_mask.detach().cpu().numpy()

    for i in range(batch_size):
        case_id = m_ids[i]
        case_dir = os.path.join(visualization_dir, case_id)
        ensure_dir(case_dir)

        image_tensor = val_images[i]
        affine = _extract_affine_from_image_tensor(image_tensor)

        image_np = image_tensor.detach().cpu().numpy()
        if image_np.ndim > 3 and image_np.shape[0] == 1:
            image_np = np.squeeze(image_np, axis=0)

        current_prob = prob_map_np[i]
        current_attention = attention_map_np[i]
        current_lesion = lesion_mask_np[i]

        try:
            nib.save(
                nib.Nifti1Image(image_np, affine),
                os.path.join(case_dir, f"{modality}.nii.gz"),
            )

            for head_idx in range(current_prob.shape[0]):
                nib.save(
                    nib.Nifti1Image(current_prob[head_idx], affine),
                    os.path.join(case_dir, f"{modality}_prob_map_{head_idx}.nii.gz"),
                )

            for head_idx in range(current_attention.shape[0]):
                nib.save(
                    nib.Nifti1Image(current_attention[head_idx], affine),
                    os.path.join(case_dir, f"{modality}_attention_map_{head_idx}.nii.gz"),
                )

            if int(mask_path[i]) == 1:
                nib.save(
                    nib.Nifti1Image(current_lesion[0], affine),
                    os.path.join(case_dir, "lesion_mask.nii.gz"),
                )

        except Exception as exc:
            logging.error(
                f"Failed to save visualization for case={case_id}, modality={modality}: {exc}"
            )


# -----------------------------------------------------------------------------
# Inference core
# -----------------------------------------------------------------------------
def run_inference_for_modality(
    model: torch.nn.Module,
    dataloader: DataLoader,
    modality: str,
    args: argparse.Namespace,
    device: torch.device,
    visualization_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run inference for one modality and return a result dataframe.

    Output columns
    --------------
    - ms_prob
    - accurate
    - weighted_prob_sum
    - sa_map_sum
    - ms_logits
    - pos_map_sum
    - neg_map_sum
    - max_ratio
    """
    all_rows: List[Dict] = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, train=False)

            score = outputs["score"]
            prob_map = outputs["prob"]
            sa_map = outputs["SA_map"]

            if score.dim() == 2 and score.shape[1] == 1:
                score = score.squeeze(1)

            ms_prob = infer_probability(score, args.loss_type)
            pred_labels = (ms_prob > 0.5).int()

            weighted_prob_sum = prob_map.sum(dim=(1, 2, 3, 4)).cpu()
            pos_map_sum = (prob_map * (prob_map > 0)).sum(dim=(1, 2, 3, 4)).cpu()
            neg_map_sum = -(prob_map * (prob_map <= 0)).sum(dim=(1, 2, 3, 4)).cpu()
            max_ratio = torch.max(pos_map_sum, neg_map_sum) / (pos_map_sum + neg_map_sum + 1e-8)
            sa_map_sum = sa_map.sum(dim=(1, 2, 3, 4)).cpu()

            lesion_mask = batch.get("lesion_mask", None)
            mask_path = batch.get("mask_path", None)

            if (
                args.visualization
                and visualization_dir is not None
                and lesion_mask is not None
                and mask_path is not None
            ):
                save_visualization_batch(
                    val_images=batch["image"],
                    prob_map=prob_map,
                    attention_map=sa_map,
                    lesion_mask=lesion_mask,
                    mask_path=mask_path,
                    m_ids=batch["m_id"],
                    modality=modality,
                    visualization_dir=visualization_dir,
                )

            batch_size = labels.shape[0]
            for i in range(batch_size):
                row = {
                    "m_id": batch["m_id"][i],
                    "modality": modality,
                    "label": int(labels[i].item()),
                    "ms": int(labels[i].item()),
                    "ms_prob": float(ms_prob[i].cpu().item()),
                    "accurate": bool(pred_labels[i].cpu().item() == labels[i].cpu().item()),
                    "weighted_prob_sum": float(weighted_prob_sum[i].item()),
                    "sa_map_sum": float(sa_map_sum[i].item()),
                    "ms_logits": float(score[i].detach().cpu().item()) if score.dim() == 1 else score[i].detach().cpu().tolist(),
                    "pos_map_sum": float(pos_map_sum[i].item()),
                    "neg_map_sum": float(neg_map_sum[i].item()),
                    "max_ratio": float(max_ratio[i].item()),
                }

                if "structural_mri" in batch:
                    row["structural_mri"] = int(batch["structural_mri"][i])
                if "SMI" in batch:
                    row["SMI"] = int(batch["SMI"][i])

                all_rows.append(row)

    result_df = pd.DataFrame(all_rows)

    if result_df.empty:
        logging.warning(f"No inference results for modality: {modality}")
        return result_df

    acc = result_df["accurate"].mean()
    auc = safe_auc(result_df["label"].tolist(), result_df["ms_prob"].tolist())

    logging.info(f"[{modality}] N={len(result_df)} | Accuracy={acc:.4f} | AUC={auc:.4f}")
    return result_df


def summarize_activation_statistics(result_df: pd.DataFrame, modality: str) -> None:
    """Print simple activation statistics for positive cases."""
    pos_df = result_df[result_df["label"] == 1]
    neg_df = result_df[result_df["label"] == 0]

    if pos_df.empty:
        logging.info(f"[{modality}] No positive cases.")
        return

    logging.info(
        f"[{modality}] Avg of sum of prediction heatmaps in positive cases = {pos_df['weighted_prob_sum'].mean():.4f}"
    )
    logging.info(
        f"[{modality}] Avg of sum of prediction heatmaps in negative cases = {neg_df['weighted_prob_sum'].mean():.4f}"
    )


# -----------------------------------------------------------------------------
# Higher-level orchestration
# -----------------------------------------------------------------------------
def run_test_inference(
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    image_size: Tuple[int, int, int],
) -> pd.DataFrame:
    """
    Run inference across all requested modalities.

    Returns
    -------
    Combined dataframe of all modality-level predictions.
    """
    ensure_dir(args.output_dir)

    model_paths = normalize_model_paths(args.model_paths, args.modalities)

    visualization_dir = None
    if args.visualization:
        visualization_dir = os.path.join(
            args.visualization_dir,
            datetime.now().strftime("%Y%m%d"),
        )
        ensure_dir(visualization_dir)

    loaded_model = None
    current_model_path = None
    all_results: List[pd.DataFrame] = []

    for modality, model_path in zip(args.modalities, model_paths):
        logging.info("=" * 80)
        logging.info(f"Processing modality: {modality}")

        modality_df = test_df[test_df["modality"] == modality].copy()

        if modality_df.empty:
            logging.warning(f"No test rows found for modality: {modality}")
            continue

        logging.info(f"Test size: {len(modality_df)}")
        logging.info(f"MS positive: {(modality_df['label'] == 1).sum()}")
        logging.info(f"MS negative: {(modality_df['label'] == 0).sum()}")

        dataloader = build_test_loader(modality_df, args)

        if loaded_model is None or current_model_path != model_path:
            loaded_model = build_model(args, model_path, image_size, device)
            current_model_path = model_path
        else:
            logging.info(f"Reusing loaded checkpoint for modality: {modality}")

        result_df = run_inference_for_modality(
            model=loaded_model,
            dataloader=dataloader,
            modality=modality,
            args=args,
            device=device,
            visualization_dir=visualization_dir,
        )

        if result_df.empty:
            continue

        summarize_activation_statistics(result_df, modality)

        modality_csv = os.path.join(args.output_dir, f"prediction_{modality}.csv")
        result_df.to_csv(modality_csv, index=False)
        logging.info(f"Saved: {modality_csv}")

        all_results.append(result_df)

    if not all_results:
        raise RuntimeError("No results were generated for any modality.")

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_csv = os.path.join(args.output_dir, "prediction_all_modalities.csv")
    all_results_df.to_csv(all_csv, index=False)
    logging.info(f"Saved combined results: {all_csv}")

    return all_results_df


# -----------------------------------------------------------------------------
# Optional ensemble reporting
# -----------------------------------------------------------------------------
def run_group_ensembles(all_results_df: pd.DataFrame, modalities: Sequence[str]) -> None:
    """
    Run simple ensemble summaries following the original project logic.
    """
    print("\n" + "-" * 40)
    print("Ensemble summary")

    if any(m in modalities for m in STRUCTURAL_MRI_LIST):
        print("\nAll structural MRI")
        df_structural = all_results_df[all_results_df["modality"].isin(STRUCTURAL_MRI_LIST)]
        if not df_structural.empty:
            avg_logits_ensemble(df_structural)

    if any(m in modalities for m in DTI_LIST):
        print("\nAll DTI")
        df_dti = all_results_df[all_results_df["modality"].isin(DTI_LIST)]
        if not df_dti.empty:
            avg_logits_ensemble(df_dti)

    if any(m in modalities for m in SMI_LIST):
        print("\nAll SMI")
        df_smi = all_results_df[all_results_df["modality"].isin(SMI_LIST)]
        if not df_smi.empty:
            avg_logits_ensemble(df_smi)

    if any(m in modalities for m in WDKI_LIST):
        print("\nAll WDKI")
        df_wdki = all_results_df[all_results_df["modality"].isin(WDKI_LIST)]
        if not df_wdki.empty:
            avg_logits_ensemble(df_wdki)

    if any(m in modalities for m in DKI_LIST):
        print("\nAll DKI")
        df_dki = all_results_df[all_results_df["modality"].isin(DKI_LIST)]
        if not df_dki.empty:
            avg_logits_ensemble(df_dki)

    print("\nAll modalities")
    avg_logits_ensemble(all_results_df)
    grouped_avg_prob_ensemble_smri(all_results_df)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    setup_logging()

    logging.info(f"Backbone: {args.backbone}")
    logging.info(f"Modalities: {args.modalities}")
    logging.info(f"Test CSV: {args.test_patient_ids}")
    logging.info(f"Output dir: {args.output_dir}")

    image_size = get_image_size(args)
    logging.info(f"Image size: {image_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    test_df = prepare_test_dataframe(args)
    all_results_df = run_test_inference(
        test_df=test_df,
        args=args,
        device=device,
        image_size=image_size,
    )

    run_group_ensembles(all_results_df, args.modalities)

    logging.info("Inference completed successfully.")


if __name__ == "__main__":
    print("Start inferencing...")
    parsed_args = parse_args()
    main(parsed_args)