import argparse
import datetime
import logging
import os
import random
import sys
from typing import Any, Dict, Optional

import monai
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, broadcast_object_list
from monai.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())

from model.Models import VisualEncoder
from utils.dataset import SingleModalityDataset, collate_skip_none
from utils.evaluation import summarize_validation_predictions, validate_prediction_coverage
from utils.scheduling import (
    UpdateWarmupCosineScheduler,
    compute_optimizer_update_counts,
)
from utils.transforms import FilterImages

# ---------------------------------------------------------------------
# Optional external logging
# ---------------------------------------------------------------------
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def is_saliency_backbone(args) -> bool:
    """
    Return True if the current backbone exposes saliency-related outputs.
    """
    return args.backbone == "VoCo_Salient_2"


def log_peak_memory_usage(accelerator: Accelerator) -> None:
    """
    Log peak GPU memory usage across all processes.
    """
    if not torch.cuda.is_available():
        if accelerator.is_main_process:
            print("CUDA is not available. Skipping memory usage logging.", flush=True)
        return

    local_mem_usages = [
        torch.cuda.max_memory_allocated(device=f"cuda:{i}") / (1024 ** 2)
        for i in range(torch.cuda.device_count())
    ]

    if dist.is_available() and dist.is_initialized():
        device = accelerator.device
        local_tensor = torch.tensor(local_mem_usages, dtype=torch.float32, device=device)
        gather_list = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_list, local_tensor)

        if accelerator.is_main_process:
            print("GPU Peak Memory Usage (MB) across ranks:", flush=True)
            for rank, usage_tensor in enumerate(gather_list):
                print(f"  Rank {rank}: {usage_tensor.cpu().tolist()}", flush=True)
    else:
        if accelerator.is_main_process:
            print(f"GPU Peak Memory Usage (MB): {local_mem_usages}", flush=True)


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: UpdateWarmupCosineScheduler,
    best_metric: float,
    best_metric_epoch: int,
    non_improve_epochs: int,
    accelerator: Accelerator,
) -> None:
    """Save model, optimizer, update scheduler, and early-stopping state."""
    torch.save(
        {
            "epoch": epoch,
            "completed_updates": scheduler.completed_steps,
            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "best_metric_epoch": best_metric_epoch,
            "non_improve_epochs": non_improve_epochs,
        },
        path,
    )


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def get_args():
    """
    Parse command line arguments.

    This GitHub version removes several legacy / unused arguments to keep the
    public training interface cleaner and easier to maintain.
    """
    parser = argparse.ArgumentParser(description="Train MRI classification models")

    # -----------------------------------------------------------------
    # Data paths
    # -----------------------------------------------------------------
    parser.add_argument("--modalities", nargs="+", type=str, default=["3DFLAIR_NCE"])
    parser.add_argument("--val_modalities", nargs="+", type=str, default=None)

    parser.add_argument(
        "--base_root", type=str, default=None,
        help="Deprecated compatibility option; image paths are read from the metadata CSV.",
    )
    parser.add_argument("--train_patient_ids", type=str, required=True, help="Training image metadata CSV.")
    parser.add_argument("--val_patient_ids", type=str, required=True, help="Validation image metadata CSV.")
    parser.add_argument(
        "--train_diagnosis_df",
        type=str,
        default=None,
        help="Deprecated compatibility option; not used by the active training pipeline.",
    )
    parser.add_argument(
        "--white_matter_list",
        type=str,
        default=None,
        help="Deprecated compatibility option; not used by the active training pipeline.",
    )

    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--pretrained_path", type=str, default="pretrain_weights/VoCo/VoComni_B.pt")
    parser.add_argument("--continue_training", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None)

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--early_stopping_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=24)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=5)

    # -----------------------------------------------------------------
    # Loss / regularization
    # -----------------------------------------------------------------
    parser.add_argument(
        "--loss_type",
        type=str,
        default="bce_with_logits",
        choices=["bce", "bce_with_logits", "ce", "weighted_ce"],
        help="Loss function type.",
    )
    parser.add_argument("--outside_reg_loss", type=float, default=1e-6)
    parser.add_argument("--L1_loss", type=float, default=0.0)
    parser.add_argument("--pos_penalty", action="store_true")

    # -----------------------------------------------------------------
    # LR scheduler
    # -----------------------------------------------------------------
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Exact optimizer-update warmup length; overrides --warmup_epochs.",
    )
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--use_warmup", action="store_true")

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    parser.add_argument(
        "--backbone",
        type=str,
        default="VoCo_Salient_2",
        choices=["VoCo", "ViT_Classifier", "VoCo_Salient_2", "BrainMVP"],
    )
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument(
        "--auc_metric",
        type=str,
        default="hierarchical",
        choices=["micro", "macro", "hierarchical", "ensemble"],
    )
    parser.add_argument("--freeze_backbone", action="store_true")

    # -----------------------------------------------------------------
    # Data processing
    # -----------------------------------------------------------------
    parser.add_argument("--use_preprocess", action="store_true")
    parser.add_argument("--use_bet_only", action="store_true")
    parser.add_argument("--use_both", action="store_true")

    parser.add_argument("--resize_size", type=int, default=96)
    parser.add_argument("--roi_x", type=int, default=None)
    parser.add_argument("--roi_y", type=int, default=None)
    parser.add_argument("--roi_z", type=int, default=None)

    parser.add_argument("--use_global_transform", action="store_true")
    parser.add_argument("--rotate_prob", type=float, default=0.8)
    parser.add_argument("--pseudo_2D_prob", type=float, default=0.0)

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------
    parser.add_argument("--oversampling", action="store_true")
    parser.add_argument("--weight_power", type=float, default=1.0)
    parser.add_argument("--smooth_factor", type=float, default=0.0)
    parser.add_argument("--use_max_weight", action="store_true")
    parser.add_argument("--merge_flair", action="store_true")

    # -----------------------------------------------------------------
    # Precision / distributed
    # -----------------------------------------------------------------
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--find_unused_parameters", action="store_true")

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------
    parser.add_argument("--use_wandb", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------
def prepare_datasets(args, logger, accelerator):
    """
    Build training dataset and validation datasets.

    Notes:
        - Validation is split by modality to preserve per-modality evaluation.
        - This function assumes the CSV files contain the columns used below.
        - Some dataset-specific logic is intentionally preserved to match your
          current project behavior.
    """
    if args.roi_x is None or args.roi_y is None or args.roi_z is None:
        image_size = (args.resize_size, args.resize_size, args.resize_size)
    else:
        image_size = (args.roi_x, args.roi_y, args.roi_z)

    if args.val_modalities is None:
        args.val_modalities = args.modalities

    try:
        train_df = pd.read_csv(args.train_patient_ids, dtype={"m_id": "string"})
        val_df = pd.read_csv(args.val_patient_ids, dtype={"m_id": "string"})
    except Exception as e:
        raise FileNotFoundError(f"Error loading dataset metadata: {e}")

    if accelerator.is_main_process and (
        args.train_diagnosis_df is not None or args.white_matter_list is not None
    ):
        logger.warning(
            "--train_diagnosis_df and --white_matter_list are deprecated compatibility "
            "options and are not used by the active training pipeline."
        )

    train_df = train_df[train_df["modality"].isin(args.modalities)].copy()
    val_df = val_df[val_df["modality"].isin(args.val_modalities)].copy()

    # -----------------------------------------------------------------
    # Modality indices / modality families
    # -----------------------------------------------------------------
    modality_to_idx = {m: i for i, m in enumerate(args.modalities)}
    train_df["modality_label"] = train_df["modality"].map(modality_to_idx).astype(int)
    val_df["modality_label"] = val_df["modality"].map(modality_to_idx).astype(int)

    structural_mri_list = [
        "3DFLAIR_NCE", "3DFLAIR_CE", "3DT1_NCE", "3DT1_CE", "3DT2_NCE", "3DT2_CE",
        "2DFLAIR_NCE", "2DFLAIR_CE", "2DT1_NCE", "2DT1_CE", "2DT2_NCE", "2DT2_CE", "b0",
    ]
    train_df["structural_mri"] = train_df["modality"].apply(lambda x: 1 if x in structural_mri_list else 0)
    val_df["structural_mri"] = val_df["modality"].apply(lambda x: 1 if x in structural_mri_list else 0)

    smi_list = ["Da_smi", "DePar_smi", "DePerp_smi", "f_smi", "p2_smi"]
    train_df["SMI"] = train_df["modality"].apply(lambda x: 1 if x in smi_list else 0)
    val_df["SMI"] = val_df["modality"].apply(lambda x: 1 if x in smi_list else 0)

    # -----------------------------------------------------------------
    # Image path selection
    # -----------------------------------------------------------------
    if args.use_preprocess or args.use_both:
        train_df["image"] = train_df["preprocessing"]
        val_df["image"] = val_df["preprocessing"]
    elif args.use_bet_only:
        train_df["image"] = train_df["bet"]
        val_df["image"] = val_df["preprocessing"]
    else:
        train_df["image"] = train_df["non-preprocessing"]
        val_df["image"] = val_df["preprocessing"]

    train_df = train_df[train_df["image"].notna()].copy()
    val_df = val_df[val_df["image"].notna()].copy()

    # -----------------------------------------------------------------
    # Labels
    # -----------------------------------------------------------------
    if "label" not in train_df.columns:
        train_df["label"] = train_df["ms"]
        val_df["label"] = val_df["ms"]

    train_df = train_df[train_df["label"].isin([0, 1])].copy()
    val_df = val_df[val_df["label"].isin([0, 1])].copy()
    val_df = val_df.reset_index(drop=True)
    val_df["row_id"] = np.arange(len(val_df), dtype=np.int64)

    # -----------------------------------------------------------------
    # Simple metadata filling
    # -----------------------------------------------------------------
    train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
    val_df["Age"] = val_df["Age"].fillna(val_df["Age"].median())

    train_df["Sex"] = train_df["Sex"].fillna(train_df["Sex"].mode()[0])
    val_df["Sex"] = val_df["Sex"].fillna(val_df["Sex"].mode()[0])
    train_df["Sex"] = train_df["Sex"].map({"M": 0, "F": 1})
    val_df["Sex"] = val_df["Sex"].map({"M": 0, "F": 1})

    train_df["2D_images"] = train_df["modality"].apply(
        lambda x: 1 if x in ["2DFLAIR_CE", "2DFLAIR_NCE", "2DT1_NCE", "b0"] else 0
    )

    used_cols_train = [
        "m_id",
        "modality",
        "label",
        "ms",
        "Age",
        "Sex",
        "structural_mri",
        "SMI",
        "2D_images",
        "image",
        "modality_label",
        "source",
    ]
    used_cols_val = [
        "row_id",
        "m_id",
        "modality",
        "label",
        "ms",
        "Age",
        "Sex",
        "structural_mri",
        "SMI",
        "image",
        "modality_label",
    ]

    train_df = train_df[used_cols_train].copy()
    val_df = val_df[used_cols_val].copy()

    # -----------------------------------------------------------------
    # Optional oversampling
    # -----------------------------------------------------------------
    sampling_weights = None
    if args.oversampling:
        if args.merge_flair:
            modality_mapping = {
                "2DFLAIR_CE": "2DFLAIR",
                "2DFLAIR_NCE": "2DFLAIR",
                "3DFLAIR_CE": "3DFLAIR",
                "3DFLAIR_NCE": "3DFLAIR",
            }
            train_df["merged_modality"] = train_df["modality"].apply(lambda x: modality_mapping.get(x, x))
            groupby_column = "merged_modality"
        else:
            groupby_column = "modality"

        train_counts = train_df.groupby([groupby_column, "label"]).size().reset_index(name="counts")
        train_counts["adj_counts"] = (train_counts["counts"] + args.smooth_factor) ** args.weight_power

        if args.use_max_weight:
            min_count = train_counts["adj_counts"].median() / 5
            train_counts["adj_counts"] = train_counts["adj_counts"].apply(lambda x: max(x, min_count))

        counts_dict = {}
        for _, row in train_counts.iterrows():
            mod, label, count = row[groupby_column], row["label"], row["adj_counts"]
            counts_dict.setdefault(mod, {})[label] = count

        sampling_weights = []
        weight_details = []

        for i, sample in enumerate(train_df.to_dict(orient="records")):
            mod_key = sample["merged_modality"] if groupby_column == "merged_modality" else sample["modality"]
            label = sample["label"]

            if mod_key not in counts_dict or label not in counts_dict[mod_key]:
                logger.error(f"Missing count for modality={mod_key}, label={label}")
                weight = 0.0
            else:
                weight = 1.0 / counts_dict[mod_key][label]

            sampling_weights.append(weight)
            weight_details.append(
                {
                    "idx": i,
                    "modality": sample["modality"],
                    "mod_key": mod_key,
                    "label": label,
                    "weight": weight,
                }
            )

        total_weight = sum(sampling_weights)
        if total_weight > 0:
            sampling_weights = [w / total_weight for w in sampling_weights]
            for i in range(len(weight_details)):
                weight_details[i]["norm_weight"] = sampling_weights[i]

        if accelerator.is_main_process:
            weight_df = pd.DataFrame(weight_details)
            mod_label_weights = (
                weight_df.groupby(["mod_key", "label"])["norm_weight"]
                .agg(["sum", "mean", "count"])
                .reset_index()
            )
            logger.info(f"Oversampling weights by {groupby_column} / label:")
            logger.info(mod_label_weights)

    if accelerator.is_main_process:
        logger.info(f"Training dataset size: {len(train_df)}")
        logger.info(f"Validation dataset size: {len(val_df)}")
        logger.info("Training modality-label counts:")
        logger.info(train_df.groupby(["modality", "label"]).size().reset_index(name="counts"))
        logger.info("Validation modality-label counts:")
        logger.info(val_df.groupby(["modality", "label"]).size().reset_index(name="counts"))

    # -----------------------------------------------------------------
    # Dataset objects
    # -----------------------------------------------------------------
    trn_filter_transform = FilterImages(dat_type="trn", args=args)
    vld_filter_transform = FilterImages(dat_type="vld", args=args)

    if "source" in train_df.columns:
        train_df = train_df.drop(columns=["source"])
    if "source" in val_df.columns:
        val_df = val_df.drop(columns=["source"])

    train_ds = SingleModalityDataset(
        data=train_df,
        transform=trn_filter_transform,
        train=True,
        use_both=args.use_both,
    )

    val_datasets = {
        modality: SingleModalityDataset(
            data=val_df[val_df["modality"] == modality],
            transform=vld_filter_transform,
            train=False,
        )
        for modality in args.val_modalities
    }

    return train_ds, val_datasets, sampling_weights, image_size


# ---------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------
def create_dataloaders(train_ds, val_datasets, sampling_weights, args, accelerator):
    """Build loaders once; Accelerate performs the only rank-level sharding."""
    world_size = accelerator.num_processes
    batch_divisor = world_size * args.gradient_accumulation_steps
    if args.batch_size % batch_divisor != 0:
        raise ValueError(
            "Global batch size must be divisible by "
            f"num_processes * gradient_accumulation_steps ({batch_divisor})."
        )
    local_batch_size = args.batch_size // batch_divisor

    sampler = None
    if args.oversampling and sampling_weights is not None:
        generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sampling_weights, dtype=torch.double),
            num_samples=len(sampling_weights),
            replacement=True,
            generator=generator,
        )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_skip_none,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=local_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        **loader_kwargs,
    )

    val_dataloaders = {
        modality: DataLoader(
            val_ds,
            batch_size=args.val_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        for modality, val_ds in val_datasets.items()
    }
    return train_loader, val_dataloaders


# ---------------------------------------------------------------------
# Loss builder
# ---------------------------------------------------------------------
def build_loss_function(args, train_ds, logger):
    """
    Build the main classification loss.

    Supported:
        - BCE / BCEWithLogits for saliency-style binary outputs
        - CE / weighted CE for standard logits over 2 classes
    """
    if args.loss_type == "bce":
        logger.info("Using BCELoss")
        return nn.BCELoss()

    if args.loss_type == "bce_with_logits":
        logger.info("Using BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()

    if args.loss_type == "ce":
        logger.info("Using CrossEntropyLoss")
        return nn.CrossEntropyLoss()

    if args.loss_type == "weighted_ce":
        label_counts = train_ds.data["label"].value_counts().sort_index()
        total_labels = label_counts.sum()
        class_weights = total_labels / (len(label_counts) * label_counts.values)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        logger.info(f"Class counts: {label_counts.to_dict()}")
        logger.info(f"Class weights: {class_weights}")
        logger.info("Using weighted CrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=class_weights)

    raise ValueError(f"Unknown loss_type: {args.loss_type}")


# ---------------------------------------------------------------------
# Saliency regularization
# ---------------------------------------------------------------------
def compute_saliency_regularization_losses(
    output_dict: Dict[str, Optional[torch.Tensor]],
    batch_data: Dict[str, Any],
    labels: torch.Tensor,
    args,
    epoch: int,
    accelerator: Accelerator,
) -> Dict[str, torch.Tensor]:
    """
    Compute saliency-specific regularization terms.

    Current GitHub version keeps only:
        - outside_reg_loss
        - L1_loss

    L1 schedule follows the original experiment logic:
        - epoch <= 5:  weight = 0
        - 6 <= epoch <= 15: linearly increase to target
        - epoch > 15: weight = 0
    """
    device = accelerator.device
    zero = torch.zeros(1, device=device)

    losses = {
        "non_brain_reg_loss": zero,
        "L1_loss": zero,
    }

    aux_outputs = output_dict.get("prob", None)
    if aux_outputs is None:
        return losses

    aux_outputs = aux_outputs.squeeze(1)
    non_brain_mask = batch_data["non_brain_mask"].squeeze(1)

    # -----------------------------------------------------------------
    # Outside-brain regularization
    # -----------------------------------------------------------------
    if args.outside_reg_loss != 0:
        losses["non_brain_reg_loss"] = (
            torch.norm(aux_outputs * non_brain_mask, p=1, dim=(1, 2, 3)).mean()
            * args.outside_reg_loss
        )

    # -----------------------------------------------------------------
    # L1 schedule: match the original code behavior exactly
    # -----------------------------------------------------------------
    if epoch <= 5:
        effective_L1_weight = 0.0
    elif epoch <= 15:
        warmup_factor = min((epoch - 5) / 10.0, 1.0)
        effective_L1_weight = args.L1_loss * warmup_factor
    else:
        effective_L1_weight = 0.0

    if effective_L1_weight != 0:
        l1_mask = batch_data["L1_mask"].squeeze(1)
        labels_broadcast = labels.view(labels.size(0), 1, 1, 1)

        if args.pos_penalty: # penalty for positive labels
            penalty_map = torch.where(
                labels_broadcast == 1,
                F.relu(-aux_outputs),
                F.relu(aux_outputs),
            )

        else:
            penalty_map = torch.where(
                labels_broadcast == 1,
                torch.zeros_like(aux_outputs),
                F.relu(aux_outputs), # only penalty for negative cases
            )

        losses["L1_loss"] = (
            torch.norm(penalty_map * l1_mask, p=1, dim=(1, 2, 3)).mean()
            * effective_L1_weight
        )

    return losses


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
def train_one_epoch(
    model, train_loader, optimizer, lr_scheduler, loss_function, accelerator, args, epoch, writer=None
):
    """
    Train the model for one epoch.

    Important:
        The wrapper model is called as:
            output_dict = model(inputs, train=True)

        This `train=True` flag is NOT a replacement for `model.train()`.
        It is only used by saliency backbones to control which spatial map
        variant is returned.
    """
    model.train()

    use_saliency = is_saliency_backbone(args)
    epoch_len = len(train_loader)

    total_loss = 0.0
    total_main_loss = 0.0
    total_non_brain_reg = 0.0
    total_l1_reg = 0.0

    step = 0
    updates_this_epoch = 0

    if accelerator.is_main_process and use_saliency:
        print(
            f"Epoch {epoch} | "
            f"outside_reg={args.outside_reg_loss:.6f}, "
            f"L1={args.L1_loss:.6f}"
        )

    for batch_data in train_loader:
        if batch_data is None:
            continue

        step += 1
        inputs = batch_data["image"]

        if args.loss_type in ["bce", "bce_with_logits"]:
            labels = batch_data["label"].float().unsqueeze(1)
        else:
            labels = batch_data["label"].long()

        with accelerator.accumulate(model):
            with accelerator.autocast():
                output_dict = model(inputs, train=True)
                outputs = output_dict["score"]

                reg_losses = {
                    "non_brain_reg_loss": torch.zeros(1, device=accelerator.device),
                    "L1_loss": torch.zeros(1, device=accelerator.device),
                }

                if use_saliency:
                    reg_losses = compute_saliency_regularization_losses(
                        output_dict=output_dict,
                        batch_data=batch_data,
                        labels=labels,
                        args=args,
                        epoch=epoch,
                        accelerator=accelerator,
                    )

                main_loss = loss_function(outputs, labels)
                loss = (
                    main_loss
                    + reg_losses["non_brain_reg_loss"]
                    + reg_losses["L1_loss"]
                )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                    updates_this_epoch += 1
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_non_brain_reg += reg_losses["non_brain_reg_loss"].item()
        total_l1_reg += reg_losses["L1_loss"].item()

        if writer is not None and step % 10 == 0:
            global_step = epoch_len * (epoch - 1) + step
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/main_loss", main_loss.item(), global_step)
            writer.add_scalar("train/non_brain_reg_loss", reg_losses["non_brain_reg_loss"].item(), global_step)
            writer.add_scalar("train/L1_loss", reg_losses["L1_loss"].item(), global_step)

            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f"train/lr_group_{i}", param_group["lr"], global_step)


    totals = torch.tensor(
        [total_loss, total_main_loss, total_non_brain_reg, total_l1_reg, step],
        dtype=torch.float64,
        device=accelerator.device,
    )
    totals = accelerator.reduce(totals, reduction="sum")
    denom = max(float(totals[4].item()), 1.0)

    avg_loss = float(totals[0].item() / denom)
    avg_main_loss = float(totals[1].item() / denom)
    avg_non_brain_reg = float(totals[2].item() / denom)
    avg_l1_reg = float(totals[3].item() / denom)

    if accelerator.is_main_process:
        print(
            f"Epoch {epoch} | "
            f"loss={avg_loss:.4f}, "
            f"main={avg_main_loss:.4f}, "
            f"non_brain={avg_non_brain_reg:.4f}, "
            f"L1={avg_l1_reg:.4f}"
        )

    if writer is not None:
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("train/epoch_main_loss", avg_main_loss, epoch)
        writer.add_scalar("train/epoch_non_brain_reg_loss", avg_non_brain_reg, epoch)
        writer.add_scalar("train/epoch_L1_loss", avg_l1_reg, epoch)

    return avg_loss, updates_this_epoch


# ---------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------
def validate_model(model, val_dataloaders, accelerator, args, logger):
    """Run sharded validation and compute metrics from one globally gathered table."""
    accelerator.wait_for_everyone()
    model.eval()
    use_saliency = is_saliency_backbone(args)
    prediction_frames = []
    expected_row_ids = []

    raw_model = accelerator.unwrap_model(model)
    if (
        accelerator.is_main_process
        and use_saliency
        and hasattr(raw_model.encoder, "predictor")
    ):
        try:
            logger.info(
                "Classifier bias: %.8f",
                raw_model.encoder.predictor.classifier_bias.item(),
            )
        except Exception:
            pass
    del raw_model

    with torch.inference_mode():
        for modality, val_loader in val_dataloaders.items():
            modality_expected_ids = [
                int(value) for value in val_loader.dataset.data["row_id"].tolist()
            ]
            expected_row_ids.extend(modality_expected_ids)
            modality_frames = []

            for val_data in val_loader:
                if val_data is None:
                    raise RuntimeError(
                        f"Validation produced an empty batch for modality {modality}."
                    )

                val_images = val_data["image"].to(accelerator.device, non_blocking=True)
                val_labels = val_data["label"].to(accelerator.device).reshape(-1)
                row_ids = val_data["row_id"].to(accelerator.device).long().reshape(-1)

                with accelerator.autocast():
                    output_dict = model(val_images, train=True)
                    score = output_dict["score"]
                    if use_saliency:
                        if args.loss_type == "bce":
                            probabilities = score.reshape(-1)
                        elif args.loss_type == "bce_with_logits":
                            probabilities = torch.sigmoid(score).reshape(-1)
                        else:
                            raise ValueError(
                                "VoCo_Salient_2 validation requires bce or bce_with_logits."
                            )
                    else:
                        probabilities = torch.softmax(score, dim=1)[:, 1]

                l1_values = torch.full_like(probabilities, torch.nan)
                weighted_values = torch.full_like(probabilities, torch.nan)
                if use_saliency:
                    probability_map = output_dict.get("prob")
                    attention_map = output_dict.get("SA_map")
                    if probability_map is not None:
                        probability_map = probability_map.float().squeeze(1)
                        l1_values = F.relu(probability_map).sum(dim=(1, 2, 3))
                    if probability_map is not None and attention_map is not None:
                        attention_map = attention_map.float().mean(dim=1)
                        weighted_values = (probability_map * attention_map).sum(
                            dim=(1, 2, 3)
                        )

                gathered = accelerator.gather_for_metrics(
                    {
                        "row_id": row_ids,
                        "ms": val_labels.long(),
                        "ms_prob": probabilities.float(),
                        "l1_sum": l1_values.float(),
                        "weighted_prob_sum": weighted_values.float(),
                    }
                )
                gathered_m_ids = accelerator.gather_for_metrics(
                    [str(value) for value in val_data["m_id"]],
                    use_gather_object=True,
                )

                if accelerator.is_main_process:
                    count = int(gathered["row_id"].numel())
                    if len(gathered_m_ids) != count:
                        raise RuntimeError(
                            f"Gathered {count} tensors but {len(gathered_m_ids)} IDs."
                        )
                    modality_frames.append(
                        pd.DataFrame(
                            {
                                "row_id": gathered["row_id"].detach().cpu().numpy(),
                                "m_id": [str(value) for value in gathered_m_ids],
                                "modality": [modality] * count,
                                "ms": gathered["ms"].detach().cpu().numpy(),
                                "ms_prob": gathered["ms_prob"].detach().cpu().numpy(),
                                "l1_sum": gathered["l1_sum"].detach().cpu().numpy(),
                                "weighted_prob_sum": gathered["weighted_prob_sum"].detach().cpu().numpy(),
                            }
                        )
                    )

            if accelerator.is_main_process and modality_expected_ids:
                if not modality_frames:
                    raise RuntimeError(f"No validation predictions for modality {modality}.")
                modality_frame = validate_prediction_coverage(
                    pd.concat(modality_frames, ignore_index=True),
                    expected_row_ids=modality_expected_ids,
                )
                prediction_frames.append(modality_frame)

                positives = modality_frame[modality_frame["ms"] == 1]
                negatives = modality_frame[modality_frame["ms"] == 0]
                if use_saliency:
                    if not positives.empty:
                        logger.info(
                            "[%s] positive activation sum=%.4f, weighted sum=%.4f",
                            modality,
                            positives["l1_sum"].mean(),
                            positives["weighted_prob_sum"].mean(),
                        )
                    if not negatives.empty:
                        logger.info(
                            "[%s] negative activation sum=%.4f, weighted sum=%.4f",
                            modality,
                            negatives["l1_sum"].mean(),
                            negatives["weighted_prob_sum"].mean(),
                        )

            accelerator.wait_for_everyone()

    results = {}
    if accelerator.is_main_process:
        if not prediction_frames:
            raise RuntimeError("Validation generated no predictions.")
        predictions = validate_prediction_coverage(
            pd.concat(prediction_frames, ignore_index=True),
            expected_row_ids=expected_row_ids,
        )
        results = summarize_validation_predictions(
            predictions,
            requested_modalities=args.val_modalities,
            auc_metric=args.auc_metric,
            expected_row_ids=expected_row_ids,
        )
        logger.info(
            "Validation micro AUC=%.4f | macro AUC=%.4f | hierarchical AUC=%.4f | ensemble AUC=%.4f",
            results["micro_avg"]["auc"],
            results["macro_avg"]["auc"],
            results["hierarchical_avg_auc"],
            results["ensemble"]["auc"],
        )

    accelerator.wait_for_everyone()
    return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args):
    """Train with global validation gathering and update-based LR scheduling."""
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=args.find_unused_parameters
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        device_placement=True,
        kwargs_handlers=[ddp_kwargs],
    )

    logger = logging.getLogger("training_logger")
    logging_level = logging.INFO if accelerator.is_main_process else logging.ERROR
    logging.basicConfig(stream=sys.stdout, level=logging_level)

    if accelerator.is_main_process:
        monai.config.print_config()
    set_random_seed(args.seed)

    if accelerator.is_main_process:
        logger.info("Random seed: %s", args.seed)
        logger.info("Backbone: %s", args.backbone)
        logger.info("Modalities: %s", args.modalities)
        logger.info("Global batch size: %s", args.batch_size)
        logger.info("Base LR: %s", args.lr)
        logger.info("Mixed precision: %s", args.mixed_precision)
        logger.info("Gradient accumulation steps: %s", args.gradient_accumulation_steps)
        logger.info("Number of processes: %s", accelerator.num_processes)
        logger.info("Number of epochs: %s", args.num_epochs)
        logger.info("Checkpoint-selection metric: %s", args.auc_metric)

    train_ds, val_datasets, sampling_weights, image_size = prepare_datasets(
        args, logger, accelerator
    )
    train_loader, val_dataloaders = create_dataloaders(
        train_ds, val_datasets, sampling_weights, args, accelerator
    )

    timestamp_holder = [
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if accelerator.is_main_process
        else None
    ]
    broadcast_object_list(timestamp_holder)
    timestamp = timestamp_holder[0]
    if args.fold is not None:
        timestamp += f"_{args.fold}"

    processing_tag = (
        "preprocess"
        if args.use_preprocess
        else "bet_only"
        if args.use_bet_only
        else "non_preprocess"
    )
    experiment_name = (
        f"{args.backbone}_{processing_tag}_"
        f"{'_'.join(args.modalities)}_"
        f"{'oversampling_' if args.oversampling else ''}"
        f"lr{args.lr}_bs{args.batch_size}_ep{args.num_epochs}"
    )
    output_path = os.path.join(args.output_path, experiment_name, timestamp)
    if accelerator.is_main_process:
        os.makedirs(output_path, exist_ok=True)
        logger.info("Output path: %s", output_path)
    accelerator.wait_for_everyone()

    writer = SummaryWriter(log_dir=output_path) if accelerator.is_main_process else None
    if WANDB_AVAILABLE and accelerator.is_main_process and args.use_wandb:
        wandb.init(
            project="medical-image-classification",
            name=experiment_name,
            config=vars(args),
        )

    model = VisualEncoder(
        encoder_name=args.backbone,
        in_channels=args.num_channels,
        number_of_classes=2,
        image_size=image_size,
        pretrained_path=args.pretrained_path,
        num_heads=args.num_heads,
    )
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            if "classifier" not in name:
                parameter.requires_grad = False
        if accelerator.is_main_process:
            logger.info("Backbone frozen; only classifier parameters will be trained.")

    loss_function = build_loss_function(args, train_ds, logger)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
    )

    start_epoch = 1
    best_metric = -1.0
    best_metric_epoch = 0
    non_improve_epochs = 0
    checkpoint = None
    if args.continue_training is not None:
        if accelerator.is_main_process:
            logger.info("Resuming from checkpoint: %s", args.continue_training)
        checkpoint = torch.load(
            args.continue_training, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", -1.0))
        best_metric_epoch = int(
            checkpoint.get("best_metric_epoch", checkpoint.get("epoch", 0))
        )
        non_improve_epochs = int(checkpoint.get("non_improve_epochs", 0))

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    for modality, val_dataloader in val_dataloaders.items():
        val_dataloaders[modality] = accelerator.prepare(val_dataloader)

    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader contains zero batches.")
    updates_per_epoch, total_updates = compute_optimizer_update_counts(
        num_batches_per_process=len(train_loader),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
    )
    if args.warmup_steps is not None:
        warmup_updates = args.warmup_steps
    elif args.use_warmup:
        warmup_updates = args.warmup_epochs * updates_per_epoch
    else:
        warmup_updates = 0

    # Checkpoints store the current optimizer LR. Reset the schedule's base LR
    # before constructing it; loading scheduler state below restores the exact
    # next-update LR for new-format checkpoints.
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = args.lr
    lr_scheduler = UpdateWarmupCosineScheduler(
        optimizer,
        total_steps=total_updates,
        warmup_steps=warmup_updates,
        min_lr=args.min_lr,
        warmup_start_lr=args.warmup_start_lr,
    )

    if checkpoint is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if (
            isinstance(scheduler_state, dict)
            and scheduler_state.get("state_version")
            == UpdateWarmupCosineScheduler.state_version
        ):
            lr_scheduler.load_state_dict(scheduler_state)
        else:
            completed_updates = int(
                checkpoint.get(
                    "completed_updates", (start_epoch - 1) * updates_per_epoch
                )
            )
            lr_scheduler.set_completed_steps(completed_updates)
            if accelerator.is_main_process:
                logger.warning(
                    "Converted a legacy epoch scheduler at completed update %d.",
                    completed_updates,
                )

    if accelerator.is_main_process:
        logger.info(
            "Update schedule: updates_per_epoch=%d total_updates=%d warmup_updates=%d current_update=%d current_lr=%.8f",
            updates_per_epoch,
            total_updates,
            warmup_updates,
            lr_scheduler.completed_steps,
            lr_scheduler.get_last_lr()[0],
        )

    for epoch in range(start_epoch, args.num_epochs + 1):
        if accelerator.is_main_process:
            logger.info("-" * 60)
            logger.info("Epoch %d/%d", epoch, args.num_epochs)
            logger.info(
                "Starting LR %.8f at update %d",
                lr_scheduler.get_last_lr()[0],
                lr_scheduler.completed_steps,
            )

        epoch_loss, updates_this_epoch = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_function=loss_function,
            accelerator=accelerator,
            args=args,
            epoch=epoch,
            writer=writer,
        )
        if updates_this_epoch <= 0:
            raise RuntimeError(f"Epoch {epoch} completed without an optimizer update.")

        if accelerator.is_main_process:
            logger.info(
                "Epoch %d average loss %.4f | successful updates %d | next LR %.8f",
                epoch,
                epoch_loss,
                updates_this_epoch,
                lr_scheduler.get_last_lr()[0],
            )
            if writer is not None:
                writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
                writer.add_scalar(
                    "train/learning_rate", lr_scheduler.get_last_lr()[0], epoch
                )
                writer.add_scalar(
                    "train/completed_updates", lr_scheduler.completed_steps, epoch
                )
            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": epoch_loss,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/completed_updates": lr_scheduler.completed_steps,
                    }
                )

        if torch.cuda.is_available():
            log_peak_memory_usage(accelerator)

        should_stop = False
        if epoch % args.val_interval == 0:
            val_results = validate_model(
                model=model,
                val_dataloaders=val_dataloaders,
                accelerator=accelerator,
                args=args,
                logger=logger,
            )
            if accelerator.is_main_process:
                for modality in args.val_modalities:
                    if modality not in val_results:
                        continue
                    metrics = val_results[modality]
                    logger.info(
                        "Modality %s: accuracy=%.4f, auc=%.4f, count=%d",
                        modality,
                        metrics["accuracy"],
                        metrics["auc"],
                        metrics["count"],
                    )
                    if writer is not None:
                        writer.add_scalar(
                            f"val/{modality}/accuracy", metrics["accuracy"], epoch
                        )
                        writer.add_scalar(
                            f"val/{modality}/auc", metrics["auc"], epoch
                        )

                for metric_name in ("micro_avg", "macro_avg", "ensemble"):
                    metrics = val_results[metric_name]
                    logger.info(
                        "%s: accuracy=%.4f, auc=%.4f",
                        metric_name,
                        metrics["accuracy"],
                        metrics["auc"],
                    )
                    if writer is not None:
                        writer.add_scalar(
                            f"val/{metric_name}/accuracy", metrics["accuracy"], epoch
                        )
                        writer.add_scalar(
                            f"val/{metric_name}/auc", metrics["auc"], epoch
                        )
                logger.info(
                    "hierarchical_avg: auc=%.4f",
                    val_results["hierarchical_avg_auc"],
                )

                current_metric = float(val_results["best_metric"])
                if current_metric > best_metric:
                    non_improve_epochs = 0
                    best_metric = current_metric
                    best_metric_epoch = epoch
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(),
                        os.path.join(output_path, "best_model.pth"),
                    )
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(),
                        os.path.join(output_path, f"best_model_epoch_{epoch}.pth"),
                    )
                    logger.info("Saved new best model.")
                else:
                    non_improve_epochs += 1

                logger.info(
                    "Epoch %d | current %s=%.4f, best=%.4f at epoch %d",
                    epoch,
                    args.auc_metric,
                    current_metric,
                    best_metric,
                    best_metric_epoch,
                )
                if WANDB_AVAILABLE and args.use_wandb:
                    wandb.log(
                        {
                            "val/best_metric": best_metric,
                            "val/current_metric": current_metric,
                            "epoch": epoch,
                        }
                    )
                should_stop = non_improve_epochs >= args.early_stopping_epochs

        if accelerator.is_main_process and epoch % args.save_interval == 0:
            save_checkpoint(
                path=os.path.join(output_path, f"checkpoint_epoch_{epoch}.pth"),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                best_metric=best_metric,
                best_metric_epoch=best_metric_epoch,
                non_improve_epochs=non_improve_epochs,
                accelerator=accelerator,
            )

        stop_tensor = torch.tensor(
            int(should_stop), device=accelerator.device, dtype=torch.int32
        )
        stop_tensor = accelerator.reduce(stop_tensor, reduction="sum")
        if int(stop_tensor.item()) > 0:
            if accelerator.is_main_process:
                logger.info(
                    "Early stopping synchronized across all ranks after %d non-improving validations.",
                    non_improve_epochs,
                )
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(
            "Training completed. Best metric=%.4f at epoch=%d; completed updates=%d",
            best_metric,
            best_metric_epoch,
            lr_scheduler.completed_steps,
        )
        torch.save(
            accelerator.unwrap_model(model).state_dict(),
            os.path.join(output_path, "final_model.pth"),
        )
        if writer is not None:
            writer.close()
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.finish()

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    return {
        "best_metric": best_metric,
        "best_epoch": best_metric_epoch,
        "completed_updates": lr_scheduler.completed_steps,
        "output_path": output_path,
    }


if __name__ == "__main__":
    print("Start training...")
    args = get_args()
    print(main(args))
