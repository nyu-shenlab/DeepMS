import os
import sys
import math
import random
import logging
import argparse
import datetime
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import monai
from monai.data import DataLoader
from monai.utils import first

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from sklearn.metrics import roc_auc_score

sys.path.append(os.getcwd())

from utils.dataset import SingleModalityDataset, collate_skip_none
from utils.transforms import FilterImages
from utils.analysis import grouped_avg_prob_ensemble
from model.Models import VisualEncoder


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
# Scheduler
# ---------------------------------------------------------------------
class WarmupCosineScheduler:
    """
    Epoch-based learning rate scheduler:
        1) linear warmup
        2) cosine decay

    Call step() once per epoch.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-7,
        verbose: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.verbose = verbose

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = 0
        self._step_count = 0

    def step(self) -> None:
        """Advance scheduler by one epoch."""
        self._step_count += 1
        new_lrs = self.get_lr()

        for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, new_lrs)):
            param_group["lr"] = lr
            if self.verbose:
                print(f"[Scheduler] Epoch {self._step_count}: group {i} lr -> {lr:.8f}")

        self.last_epoch = self._step_count

    def get_lr(self):
        """Compute learning rates for the current scheduler step."""
        lrs = []

        for base_lr in self.base_lrs:
            if self._step_count <= self.warmup_epochs:
                lr = (
                    (base_lr - self.warmup_start_lr)
                    * self._step_count
                    / max(self.warmup_epochs, 1)
                    + self.warmup_start_lr
                )
            else:
                denom = max(self.total_epochs - self.warmup_epochs, 1)
                progress = (self._step_count - self.warmup_epochs) / denom
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine_decay

            lrs.append(lr)

        return lrs

    def get_last_lr(self):
        """Return the current learning rates stored in optimizer param groups."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """Serialize scheduler state."""
        return {
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.base_lrs = state_dict["base_lrs"]
        self.last_epoch = state_dict["last_epoch"]
        self._step_count = state_dict["_step_count"]


# ---------------------------------------------------------------------
# Distributed weighted sampler
# ---------------------------------------------------------------------
class WeightedDistributedSampler(Sampler):
    """
    Weighted sampler for distributed training.

    We sample a global weighted list of indices and split it across ranks,
    so each process gets a disjoint slice.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        replacement: bool = True,
        generator=None,
        rank: Optional[int] = None,
        num_replicas: Optional[int] = None,
    ) -> None:
        if rank is None:
            if not dist.is_initialized():
                raise RuntimeError("DDP not initialized. Please provide `rank` manually.")
            rank = dist.get_rank()

        if num_replicas is None:
            if not dist.is_initialized():
                raise RuntimeError("DDP not initialized. Please provide `num_replicas` manually.")
            num_replicas = dist.get_world_size()

        self.weights = weights.clone().detach()
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.rank = rank
        self.num_replicas = num_replicas

        self.num_samples_per_rank = int(math.ceil(self.num_samples / self.num_replicas))
        self.total_size = self.num_samples_per_rank * self.num_replicas

    def __iter__(self):
        generator = self.generator if self.generator is not None else torch.default_generator

        indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=generator,
        ).tolist()

        indices_rank = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices_rank)

    def __len__(self):
        return self.num_samples_per_rank


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
    scheduler,
    best_metric: float,
    accelerator: Accelerator,
) -> None:
    """
    Save a full training checkpoint.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
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
        "--base_root",
        type=str,
        default="/gpfs/data/shenlab/Jiajian/MS_Project/ms_data/MESO_V2.0/ALLSUBJS_2.0",
    )
    parser.add_argument(
        "--train_patient_ids",
        type=str,
        default="/gpfs/data/shenlab/Jiajian/MS_Project/code/meta_data/updated_label_dataset/train_dataset_all.csv",
    )
    parser.add_argument(
        "--train_diagnosis_df",
        type=str,
        default="/gpfs/data/shenlab/Jiajian/MS_Project/code/ms-diagnosis/meta_data/updated_label_dataset/train_set.csv",
    )
    parser.add_argument(
        "--val_patient_ids",
        type=str,
        default="/gpfs/data/shenlab/Jiajian/MS_Project/code/meta_data/updated_label_dataset/validation_dataset_all.csv",
    )
    parser.add_argument(
        "--white_matter_list",
        type=str,
        default="/gpfs/data/shenlab/Jiajian/MS_Project/code/ms-diagnosis/meta_data/clinical_validation/ouput/0802/MESO_v2_WML_LLM_output_Qwen3_updated.csv",
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
        train_df = pd.read_csv(args.train_patient_ids)
        val_df = pd.read_csv(args.val_patient_ids)
        white_matter_df = pd.read_csv(args.white_matter_list)
    except Exception as e:
        raise FileNotFoundError(f"Error loading dataset metadata: {e}")

    # -----------------------------------------------------------------
    # Merge WM lesion metadata
    # -----------------------------------------------------------------
    white_matter_df = white_matter_df[["m_id", "wm_lesion"]].copy()
    white_matter_df.columns = ["m_id", "white_matter_lesion"]

    train_df = train_df[train_df["modality"].isin(args.modalities)].copy()
    val_df = val_df[val_df["modality"].isin(args.val_modalities)].copy()

    train_df = train_df.merge(white_matter_df, on="m_id", how="left")
    train_df["white_matter_lesion"] = train_df["white_matter_lesion"].fillna(0)
    train_df.loc[
        (train_df["ms"] == 1) & (train_df["white_matter_lesion"] != 1),
        "white_matter_lesion"
    ] = 1

    # -----------------------------------------------------------------
    # Add diagnosis metadata
    # -----------------------------------------------------------------
    used_cols = [
        "m_id",
        "migraine",
        "cerebral_vessel",
        "NMOSD",
        "mog",
        "other_demylin",
        "unspecified_demyelinating",
    ]
    train_diagnosis_df = pd.read_csv(args.train_diagnosis_df, usecols=used_cols)
    train_df = train_df.merge(train_diagnosis_df, on="m_id", how="left", validate="m:1")

    flag_cols = used_cols[1:]
    train_df.loc[:, flag_cols] = train_df[flag_cols].fillna(0).astype(int)

    mask_ms = train_df["ms"].eq(1)
    other_flags = ["migraine", "cerebral_vessel", "NMOSD", "mog", "other_demylin"]
    mask_others = train_df[other_flags].eq(1).any(axis=1) & train_df["unspecified_demyelinating"].eq(0)
    train_df = train_df.assign(important_diagnosis=(mask_ms | mask_others).astype("int8"))

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
        "white_matter_lesion",
        "important_diagnosis",
        "2D_images",
        "image",
        "modality_label",
        "source",
    ]
    used_cols_val = [
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
    """
    Build training and validation dataloaders.
    """
    world_size = accelerator.num_processes

    if args.oversampling and sampling_weights is not None:
        sampler = WeightedDistributedSampler(
            weights=torch.tensor(sampling_weights, dtype=torch.float),
            num_samples=len(sampling_weights),
            replacement=True,
            rank=accelerator.process_index,
            num_replicas=world_size,
        )
    else:
        sampler = None

    effective_batch_size = args.batch_size // world_size // args.gradient_accumulation_steps
    effective_batch_size = max(effective_batch_size, 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_skip_none,
        drop_last=True,
    )

    val_dataloaders = {}
    for modality, val_ds in val_datasets.items():
        val_dataloaders[modality] = DataLoader(
            val_ds,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_skip_none,
            shuffle=False,
            drop_last=False,
        )

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
def train_one_epoch(model, train_loader, optimizer, loss_function, accelerator, args, epoch, writer=None):
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
                optimizer.zero_grad()

        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_non_brain_reg += reg_losses["non_brain_reg_loss"].item()
        total_l1_reg += reg_losses["L1_loss"].item()

        if writer is not None and step % 10 == 0:
            global_step = epoch_len * epoch + step
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/main_loss", main_loss.item(), global_step)
            writer.add_scalar("train/non_brain_reg_loss", reg_losses["non_brain_reg_loss"].item(), global_step)
            writer.add_scalar("train/L1_loss", reg_losses["L1_loss"].item(), global_step)

            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f"train/lr_group_{i}", param_group["lr"], global_step)


    denom = max(step, 1)

    avg_loss = total_loss / denom
    avg_main_loss = total_main_loss / denom
    avg_non_brain_reg = total_non_brain_reg / denom
    avg_l1_reg = total_l1_reg / denom

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

    return avg_loss


# ---------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------
def validate_model(model, val_dataloaders, accelerator, args, logger):
    """
    Validate model across modality-specific dataloaders.

    Returns:
        results: dict with per-modality metrics and aggregated metrics.
    """
    accelerator.wait_for_everyone()
    model.eval()

    results = {}

    y_total_pred_list = []
    y_total_list = []
    m_id_total_list = []
    modality_type_total_list = []

    l1_pos_total_list = []
    l1_neg_total_list = []
    weighted_prob_sum_pos_total_list = []
    weighted_prob_sum_neg_total_list = []

    use_saliency = is_saliency_backbone(args)

    with torch.no_grad():
        raw_model = accelerator.unwrap_model(model)

        # Best-effort debug print for saliency backbones only
        if use_saliency and hasattr(raw_model.encoder, "predictor"):
            try:
                print(f"Classifier bias: {raw_model.encoder.predictor.classifier_bias.item()}")
            except Exception:
                pass

        del raw_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for modality, val_loader in val_dataloaders.items():
            y_pred_all = []
            y_all = []
            m_id_all = []

            l1_loss_all = []
            weighted_prob_sum_all = []

            is_smi_modality = False
            try:
                first_batch = next(iter(val_loader), None)
                if first_batch is not None and "SMI" in first_batch and first_batch["SMI"][0] == 1:
                    is_smi_modality = True
            except Exception:
                is_smi_modality = False
            _ = is_smi_modality  # reserved in case you want modality-specific analysis later

            for val_data in val_loader:
                if val_data is None:
                    continue

                val_images = val_data["image"].to(accelerator.device)
                val_labels = val_data["label"].to(accelerator.device)
                m_id = val_data["m_id"]

                with accelerator.autocast():
                    output_dict = model(val_images, train=True) # still use train=True in the validation setting
                    score = output_dict["score"]

                    if use_saliency:
                        if args.loss_type == "bce":
                            outputs = score.squeeze(1)
                        elif args.loss_type == "bce_with_logits":
                            outputs = torch.sigmoid(score).squeeze(1)
                        else:
                            raise ValueError("VoCo_Salient_2 should use bce or bce_with_logits.")

                        aux_outputs = output_dict["prob"]
                        aux_sa_outputs = output_dict["SA_map"]

                        if aux_outputs is not None:
                            aux_outputs = aux_outputs.squeeze(1)

                        if aux_sa_outputs is not None:
                            aux_sa_outputs = aux_sa_outputs.mean(dim=1)

                        if aux_outputs is not None:
                            l1_loss = F.relu(aux_outputs).sum(dim=(1, 2, 3))
                            l1_loss_all.append(accelerator.gather_for_metrics(l1_loss))

                        if aux_outputs is not None and aux_sa_outputs is not None:
                            weighted_prob_sum = (aux_outputs * aux_sa_outputs).sum(dim=(1, 2, 3))
                            weighted_prob_sum_all.append(accelerator.gather_for_metrics(weighted_prob_sum))

                    else:
                        outputs = score

                y_pred_all.append(accelerator.gather_for_metrics(outputs))
                y_all.append(accelerator.gather_for_metrics(val_labels))
                m_id_all.extend(accelerator.gather_for_metrics(m_id))

            y_pred_gathered = (
                torch.cat(y_pred_all, dim=0)
                if y_pred_all else torch.empty(0, device=accelerator.device)
            )
            y_all_gathered = (
                torch.cat(y_all, dim=0)
                if y_all else torch.empty(0, device=accelerator.device)
            )
            modality_type = [modality] * len(m_id_all)

            if use_saliency and len(y_all_gathered) > 0 and len(l1_loss_all) > 0:
                pos_indices = y_all_gathered == 1
                neg_indices = y_all_gathered == 0

                l1_loss_gathered = torch.cat(l1_loss_all, dim=0)[pos_indices]
                l1_loss_negative_gathered = torch.cat(l1_loss_all, dim=0)[neg_indices]

                if len(weighted_prob_sum_all) > 0:
                    weighted_prob_sum_gathered = torch.cat(weighted_prob_sum_all, dim=0)[pos_indices]
                    weighted_prob_sum_negative_gathered = torch.cat(weighted_prob_sum_all, dim=0)[neg_indices]

                    weighted_prob_sum_pos_total_list.append(weighted_prob_sum_gathered)
                    weighted_prob_sum_neg_total_list.append(weighted_prob_sum_negative_gathered)

                l1_pos_total_list.append(l1_loss_gathered)
                l1_neg_total_list.append(l1_loss_negative_gathered)

                if accelerator.is_main_process:
                    print("=" * 100)
                    print(f"{modality}, sum of activation map in positive cases: {l1_loss_gathered.mean().cpu().numpy():.4f}")
                    print(f"{modality}, sum of activation map in negative cases: {l1_loss_negative_gathered.mean().cpu().numpy():.4f}")
                    if len(weighted_prob_sum_all) > 0:
                        print(f"{modality}, weighted sum of activation map in positive cases: {weighted_prob_sum_gathered.mean().cpu().numpy():.4f}")
                        print(f"{modality}, weighted sum of activation map in negative cases: {weighted_prob_sum_negative_gathered.mean().cpu().numpy():.4f}")
                    print("=" * 100)

            if accelerator.is_main_process and len(y_all_gathered) > 0:
                if use_saliency:
                    acc_value = torch.eq(y_pred_gathered > 0.5, y_all_gathered)
                else:
                    acc_value = torch.eq(y_pred_gathered.argmax(dim=1), y_all_gathered)

                acc_metric = acc_value.sum().item() / len(acc_value)

                try:
                    if use_saliency:
                        y_pred_prob = y_pred_gathered.cpu().numpy()
                    else:
                        y_pred_prob = y_pred_gathered.softmax(dim=1)[:, 1].cpu().numpy()

                    y_true = y_all_gathered.cpu().numpy()

                    if len(set(y_true.tolist())) < 2:
                        auc_result = 0.5
                        logger.warning(f"Modality {modality} has only one class. AUC set to 0.5.")
                    else:
                        auc_result = roc_auc_score(y_true, y_pred_prob)
                except Exception as e:
                    logger.error(f"Error computing AUC for {modality}: {str(e)}")
                    auc_result = 0.5

                results[modality] = {
                    "accuracy": acc_metric,
                    "auc": auc_result,
                    "count": len(y_all_gathered),
                }

                y_total_pred_list.append(y_pred_gathered)
                y_total_list.append(y_all_gathered)
                m_id_total_list.extend(m_id_all)
                modality_type_total_list.extend(modality_type)

            accelerator.wait_for_everyone()

    if accelerator.is_main_process and len(y_total_list) > 0:
        y_total_pred = torch.cat(y_total_pred_list, dim=0)
        y_total = torch.cat(y_total_list, dim=0)

        if use_saliency:
            acc_value = torch.eq(y_total_pred > 0.5, y_total)
            y_pred_prob = y_total_pred.cpu().numpy()
        else:
            acc_value = torch.eq(y_total_pred.argmax(dim=1), y_total)
            y_pred_prob = y_total_pred.softmax(dim=1)[:, 1].cpu().numpy()

        total_acc = acc_value.sum().item() / len(acc_value)
        y_true = y_total.cpu().numpy()

        try:
            total_auc = 0.5 if len(set(y_true.tolist())) < 2 else roc_auc_score(y_true, y_pred_prob)
        except Exception as e:
            logger.error(f"Error computing combined AUC: {str(e)}")
            total_auc = 0.5

        results["total"] = {"accuracy": total_acc, "auc": total_auc, "count": len(y_total)}

        results_df = pd.DataFrame(
            {
                "m_id": m_id_total_list,
                "modality": modality_type_total_list,
                "ms_prob": y_pred_prob,
                "ms": y_true,
            }
        )

        _, ensemble_acc, ensemble_auc, _, _ = grouped_avg_prob_ensemble(
            results_df,
            print_result=True,
            return_metrics=True,
        )

        total_samples = sum(
            r["count"] for r in results.values()
            if isinstance(r, dict) and "count" in r
        )

        micro_avg_acc = (
            sum(r["accuracy"] * r["count"] for r in results.values() if isinstance(r, dict) and "count" in r)
            / total_samples
            if total_samples > 0 else 0
        )
        micro_avg_auc = (
            sum(r["auc"] * r["count"] for r in results.values() if isinstance(r, dict) and "count" in r)
            / total_samples
            if total_samples > 0 else 0
        )

        results["micro_avg"] = {"accuracy": micro_avg_acc, "auc": micro_avg_auc}
        results["ensemble"] = {"accuracy": ensemble_acc, "auc": ensemble_auc}

        # -------------------------------------------------------------
        # Hierarchical averaging
        # -------------------------------------------------------------
        hierarchical_avg_auc = 0.0

        hierarchy = {
            "FLAIR": ["3DFLAIR_NCE", "3DFLAIR_CE", "2DFLAIR_NCE", "2DFLAIR_CE"],
            "T1": ["3DT1_NCE", "2DT1_NCE"],
            "T1CE": ["3DT1_CE", "2DT1_CE"],
            "b0": ["b0"],
            "DTI": ["fa_dti", "md_dti", "ad_dti", "rd_dti"],
            "SMI": ["f_smi", "p2_smi", "DePerp_smi", "DePar_smi", "Da_smi"],
            "DKI": ["ak_wdki", "mk_wdki", "rk_wdki"],
        }

        second_level_hierarchy = {
            "sMRI": ["FLAIR", "T1", "T1CE", "b0"],
            "dMRI": ["DTI", "DKI", "SMI"],
        }

        hierarchical_aucs = {}

        for group, modalities in hierarchy.items():
            aucs_in_group = [results[m]["auc"] for m in modalities if m in results and results[m]["count"] > 0]
            if len(aucs_in_group) > 0:
                hierarchical_aucs[group] = sum(aucs_in_group) / len(aucs_in_group)

        for group, sub_groups in second_level_hierarchy.items():
            aucs_in_group = [hierarchical_aucs[sg] for sg in sub_groups if sg in hierarchical_aucs]
            if len(aucs_in_group) > 0:
                hierarchical_aucs[group] = sum(aucs_in_group) / len(aucs_in_group)

        final_level_aucs = [hierarchical_aucs[g] for g in ["sMRI", "dMRI"] if g in hierarchical_aucs]
        if len(final_level_aucs) > 0:
            hierarchical_avg_auc = sum(final_level_aucs) / len(final_level_aucs)

        results["hierarchical_aucs"] = hierarchical_aucs
        results["hierarchical_avg_auc"] = hierarchical_avg_auc

        logger.info(
            f"Micro Avg AUC: {micro_avg_auc:.4f} | "
            f"Macro Avg AUC: {total_auc:.4f} | "
            f"Hierarchical Avg AUC: {hierarchical_avg_auc:.4f} | "
            f"Ensemble AUC: {ensemble_auc:.4f}"
        )
        logger.info(f"Hierarchical breakdown: {hierarchical_aucs}")

        if args.auc_metric == "micro":
            results["best_metric"] = micro_avg_auc
        elif args.auc_metric == "macro":
            results["best_metric"] = total_auc
        elif args.auc_metric == "hierarchical":
            results["best_metric"] = hierarchical_avg_auc
        elif args.auc_metric == "ensemble":
            results["best_metric"] = ensemble_auc
        else:
            logger.warning(f"Unknown auc_metric '{args.auc_metric}'. Defaulting to hierarchical.")
            results["best_metric"] = hierarchical_avg_auc

    return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args):
    """
    Main training entry point.

    High-level workflow:
        1) Initialize accelerator and logging
        2) Prepare datasets and dataloaders
        3) Build model / optimizer / scheduler
        4) Train + validate
        5) Save checkpoints and best model
    """
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

    monai.config.print_config()
    set_random_seed(args.seed)

    if accelerator.is_main_process:
        logger.info(f"Random seed: {args.seed}")
        logger.info(f"Backbone: {args.backbone}")
        logger.info(f"Modalities: {args.modalities}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"LR: {args.lr}")
        logger.info(f"Mixed precision: {args.mixed_precision}")
        logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        logger.info(f"Number of GPUs: {accelerator.num_processes}")
        logger.info(f"Number of epochs: {args.num_epochs}")
        logger.info(f"AUC metric: {args.auc_metric}")
        if is_saliency_backbone(args):
            logger.info(f"Outside reg loss: {args.outside_reg_loss}")
            logger.info(f"L1 loss: {args.L1_loss}")
            logger.info(f"Num heads: {args.num_heads}")

    # -------------------------------------------------------------
    # Dataset / dataloader
    # -------------------------------------------------------------
    train_ds, val_datasets, sampling_weights, image_size = prepare_datasets(args, logger, accelerator)
    train_loader, val_dataloaders = create_dataloaders(train_ds, val_datasets, sampling_weights, args, accelerator)

    check_data = first(train_loader)
    if check_data is not None and accelerator.is_main_process:
        logger.info("Training data check:")
        logger.info(f"Image shape: {check_data['image'].shape}")
        logger.info(f"Label: {check_data['label']}")
        logger.info(f"Modality: {check_data['modality']}")
    elif accelerator.is_main_process:
        logger.warning("No training batch available for inspection.")

    # -------------------------------------------------------------
    # Output directories
    # -------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.fold is not None:
        timestamp += f"_{args.fold}"

    processing_tag = (
        "preprocess" if args.use_preprocess else
        "bet_only" if args.use_bet_only else
        "non_preprocess"
    )

    experiment_name = (
        f"{args.backbone}_{processing_tag}_"
        f"{'_'.join(args.modalities)}_"
        f"{'oversampling_' if args.oversampling else ''}"
        f"lr{args.lr}_bs{args.batch_size}_ep{args.num_epochs}"
    )
    output_path = os.path.join(args.output_path, experiment_name, timestamp)
    os.makedirs(output_path, exist_ok=True)

    if accelerator.is_main_process:
        logger.info(f"Output path: {output_path}")

    writer = SummaryWriter(log_dir=output_path) if accelerator.is_main_process else None

    if WANDB_AVAILABLE and accelerator.is_main_process and args.use_wandb:
        wandb.init(
            project="medical-image-classification",
            name=experiment_name,
            config=vars(args),
        )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    model = VisualEncoder(
        encoder_name=args.backbone,
        in_channels=args.num_channels,
        number_of_classes=2,
        image_size=image_size,
        pretrained_path=args.pretrained_path,
        num_heads=args.num_heads,
    )

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        if accelerator.is_main_process:
            logger.info("Backbone frozen. Only classifier-related parameters will be trained.")

    loss_function = build_loss_function(args, train_ds, logger)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    if args.use_warmup:
        lr_scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.num_epochs,
            min_lr=args.min_lr,
            warmup_start_lr=args.warmup_start_lr,
            verbose=(accelerator.is_main_process and args.num_epochs <= 20),
        )
    else:
        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.num_epochs,
            eta_min=args.min_lr,
        )

    start_epoch = 1
    best_metric = -1.0
    best_metric_epoch = 0
    non_improve_epochs = 0

    if args.continue_training is not None:
        if accelerator.is_main_process:
            logger.info(f"Resuming from checkpoint: {args.continue_training}")

        checkpoint = torch.load(args.continue_training, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_metric = checkpoint.get("best_metric", -1.0)

        if "scheduler_state_dict" in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                if accelerator.is_main_process:
                    logger.warning(f"Could not load scheduler state: {e}")

        start_epoch = checkpoint["epoch"] + 1
        best_metric_epoch = checkpoint["epoch"]

    # -------------------------------------------------------------
    # Accelerator prepare
    # -------------------------------------------------------------
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for modality, val_dataloader in val_dataloaders.items():
        val_dataloaders[modality] = accelerator.prepare(val_dataloader)

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------
    for epoch in range(start_epoch, args.num_epochs + 1):
        lr_scheduler.step()

        if accelerator.is_main_process:
            logger.info("-" * 60)
            logger.info(f"Epoch {epoch}/{args.num_epochs}")
            logger.info(f"Current learning rate: {lr_scheduler.get_last_lr()[0]:.8f}")

        cur_lr = torch.tensor([optimizer.param_groups[0]["lr"]], device=accelerator.device, dtype=torch.float64)
        all_lrs = accelerator.gather_for_metrics(cur_lr)
        if accelerator.is_main_process:
            lr_str = [f"{x:.8f}" for x in all_lrs.cpu().tolist()]
            logger.info(f"[All ranks LR] {lr_str}")

        epoch_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            accelerator=accelerator,
            args=args,
            epoch=epoch,
            writer=writer,
        )


        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} average loss: {epoch_loss:.4f}")

            if writer is not None:
                writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
                writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], epoch)

            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": epoch_loss,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                    }
                )

        if torch.cuda.is_available():
            log_peak_memory_usage(accelerator)

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------
        if epoch % args.val_interval == 0:
            val_results = validate_model(
                model=model,
                val_dataloaders=val_dataloaders,
                accelerator=accelerator,
                args=args,
                logger=logger,
            )

            if accelerator.is_main_process:
                for modality, metrics in val_results.items():
                    if modality in ["total", "micro_avg", "hierarchical_aucs", "hierarchical_avg_auc", "best_metric", "ensemble"]:
                        continue

                    logger.info(
                        f"Modality {modality}: "
                        f"accuracy={metrics.get('accuracy', 0):.4f}, "
                        f"auc={metrics.get('auc', 0):.4f}, "
                        f"count={metrics.get('count', 0)}"
                    )

                    if writer is not None:
                        writer.add_scalar(f"val/{modality}/accuracy", metrics.get("accuracy", 0), epoch)
                        writer.add_scalar(f"val/{modality}/auc", metrics.get("auc", 0), epoch)

                    if WANDB_AVAILABLE and args.use_wandb:
                        wandb.log(
                            {
                                f"val/{modality}/accuracy": metrics.get("accuracy", 0),
                                f"val/{modality}/auc": metrics.get("auc", 0),
                                "epoch": epoch,
                            }
                        )

                if "total" in val_results:
                    logger.info(
                        f"Total: accuracy={val_results['total'].get('accuracy', 0):.4f}, "
                        f"auc={val_results['total'].get('auc', 0):.4f}"
                    )
                    if writer is not None:
                        writer.add_scalar("val/total/accuracy", val_results["total"].get("accuracy", 0), epoch)
                        writer.add_scalar("val/total/auc", val_results["total"].get("auc", 0), epoch)

                if "micro_avg" in val_results:
                    logger.info(
                        f"Micro-average: accuracy={val_results['micro_avg'].get('accuracy', 0):.4f}, "
                        f"auc={val_results['micro_avg'].get('auc', 0):.4f}"
                    )
                    if writer is not None:
                        writer.add_scalar("val/micro_avg/accuracy", val_results["micro_avg"].get("accuracy", 0), epoch)
                        writer.add_scalar("val/micro_avg/auc", val_results["micro_avg"].get("auc", 0), epoch)

                if "hierarchical_avg_auc" in val_results:
                    logger.info(f"Hierarchical-average AUC: {val_results['hierarchical_avg_auc']:.4f}")

                current_metric = val_results.get("best_metric", 0.0)

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
                    f"Epoch {epoch} | current {args.auc_metric} metric={current_metric:.4f}, "
                    f"best={best_metric:.4f} at epoch {best_metric_epoch}"
                )

                if WANDB_AVAILABLE and args.use_wandb:
                    wandb.log(
                        {
                            "val/best_metric": best_metric,
                            "val/current_metric": current_metric,
                            "epoch": epoch,
                        }
                    )

                if non_improve_epochs >= args.early_stopping_epochs:
                    logger.info(
                        f"Early stopping triggered after {non_improve_epochs} epochs without improvement."
                    )
                    break

        # ---------------------------------------------------------
        # Periodic checkpoint
        # ---------------------------------------------------------
        if accelerator.is_main_process and epoch % args.save_interval == 0:
            save_checkpoint(
                path=os.path.join(output_path, f"checkpoint_epoch_{epoch}.pth"),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                best_metric=best_metric,
                accelerator=accelerator,
            )

    # -------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------
    if accelerator.is_main_process:
        logger.info(f"Training completed. Best metric={best_metric:.4f} at epoch={best_metric_epoch}")

        torch.save(
            accelerator.unwrap_model(model).state_dict(),
            os.path.join(output_path, "final_model.pth"),
        )

        if writer is not None:
            writer.close()

        if WANDB_AVAILABLE and args.use_wandb:
            wandb.finish()

    accelerator.free_memory()

    return {
        "best_metric": best_metric,
        "best_epoch": best_metric_epoch,
        "output_path": output_path,
    }


if __name__ == "__main__":
    print("Start training...")
    args = get_args()
    print(main(args))