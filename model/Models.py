import os
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import SwinUNETR

sys.path.append(os.getcwd())

from model.Uniformer import make_model
from model.vit import ViT
from utils.pos_embed import interpolate_pos_embed


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def compare_model_weights(model: nn.Module, pretrained_state_dict: Dict[str, torch.Tensor]) -> None:
    """Print mismatched keys between model weights and a pretrained state dict."""
    model_dict = model.state_dict()
    missing_in_model = sorted([k for k in pretrained_state_dict if k not in model_dict])
    missing_in_pretrained = sorted([k for k in model_dict if k not in pretrained_state_dict])

    if missing_in_model:
        print("Weights in pretrained but NOT in the model:")
        for k in missing_in_model:
            print("  ", k)
    else:
        print("No keys missing in the model.")

    if missing_in_pretrained:
        print("\nWeights in model but NOT in pretrained:")
        for k in missing_in_pretrained:
            print("  ", k)
    else:
        print("\nNo extra keys in the model.")


def load_backbone_weights(
    model: nn.Module,
    checkpoint_path: str,
    in_channels: int,
    first_conv_keys: Optional[Sequence[str]] = None,
    key_prefixes_to_remove: Tuple[str, ...] = ("module.",),
    key_prefixes_to_replace: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> None:
    """
    Load a checkpoint into model with optional key correction and first-conv replication.

    Args:
        model: Target model.
        checkpoint_path: Path to checkpoint file.
        in_channels: Current model input channels.
        first_conv_keys: Candidate keys for first conv weights to adapt.
        key_prefixes_to_remove: Prefixes to strip from checkpoint keys.
        key_prefixes_to_replace: Prefix replacements, e.g. {"resnet.": "features."}.
        strict: Passed to load_state_dict.
    """
    if not os.path.isfile(checkpoint_path):
        print(f"[Warning] Checkpoint not found: {checkpoint_path}. Using random init.")
        return

    print(f"[Info] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    corrected_sd = {}
    for k, v in checkpoint.items():
        new_k = k

        for prefix in key_prefixes_to_remove:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]

        if key_prefixes_to_replace is not None:
            for old_p, new_p in key_prefixes_to_replace.items():
                if new_k.startswith(old_p):
                    new_k = new_p + new_k[len(old_p):]

        corrected_sd[new_k] = v

    if first_conv_keys is not None and in_channels != 1:
        for first_conv_key in first_conv_keys:
            if first_conv_key not in corrected_sd:
                continue

            old_weight = corrected_sd[first_conv_key]
            old_in_channels = old_weight.shape[1]

            if old_in_channels == in_channels:
                continue

            if in_channels % old_in_channels != 0:
                raise ValueError(
                    f"Cannot replicate first conv from {old_in_channels} to {in_channels}. "
                    "Target channels must be a multiple of source channels."
                )

            factor = in_channels // old_in_channels
            print(
                f"[Info] Repeating first-conv '{first_conv_key}' "
                f"from {old_in_channels} -> {in_channels} by factor {factor}."
            )
            corrected_sd[first_conv_key] = old_weight.repeat(1, factor, 1, 1, 1) / factor

    missing_keys, unexpected_keys = model.load_state_dict(corrected_sd, strict=strict)
    print("[Result] Missing keys:", missing_keys)
    print("[Result] Unexpected keys:", unexpected_keys)
    print("[Info] Done loading.")


# ----------------------------------------------------------------------
# Multi-head ABMIL predictor
# ----------------------------------------------------------------------
class MultiABMILPredictor(nn.Module):
    """
    3D voxel-wise attention-based multiple instance learning.

    Input:
        x: (B, C, D, H, W)

    Output:
        prediction: (B, n_classes)
        prob_map:   (B, n_classes, D, H, W) or activation-like map
        attention_map: (B, num_heads, D, H, W)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 128,
        head_dim: int = 128,
        num_heads: int = 1,
        n_classes: int = 1,
        dropout: bool = True,
    ) -> None:
        super().__init__()

        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels must be divisible by num_heads, "
                f"but got {hidden_channels} and {num_heads}."
            )
        if hidden_channels != head_dim * num_heads:
            raise ValueError(
                f"hidden_channels must equal head_dim * num_heads, "
                f"but got hidden_channels={hidden_channels}, head_dim={head_dim}, num_heads={num_heads}."
            )

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.n_classes = n_classes

        self.pre_proj = nn.Conv3d(input_channels, hidden_channels, kernel_size=1)

        attention_v = [nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1), nn.Tanh()]
        attention_u = [nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1), nn.Sigmoid()]
        if dropout:
            attention_v.append(nn.Dropout(0.25))
            attention_u.append(nn.Dropout(0.25))

        self.attention_V = nn.Sequential(*attention_v)
        self.attention_U = nn.Sequential(*attention_u)

        self.attention_weights = nn.Conv3d(
            hidden_channels,
            num_heads,
            kernel_size=1,
            groups=num_heads,
        )

        self.classifier = nn.Conv3d(hidden_channels, n_classes, kernel_size=1, bias=False)
        self.classifier_bias = nn.Parameter(-0.3 * torch.ones(n_classes))

    def forward(self, x: torch.Tensor, train: bool = False):
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got {x.dim()}D")

        b, _, d, h, w = x.shape

        x = self.pre_proj(x)  # (B, hidden_channels, D, H, W)

        a_v = self.attention_V(x)
        a_u = self.attention_U(x)
        a = a_v * a_u
        # use sigmoid rather than softmax to avoid competition between voxels
        attention_map = torch.sigmoid(self.attention_weights(a))  # (B, num_heads, D, H, W) 

        x_reshaped = x.view(b, self.num_heads, self.head_dim, d, h, w)
        attention_map_expanded = attention_map.unsqueeze(2)  # (B, num_heads, 1, D, H, W)

        norm_factor = attention_map_expanded.sum(dim=(3, 4, 5), keepdim=True) + 1e-9
        attention_map_expanded = attention_map_expanded / norm_factor

        weighted_features = attention_map_expanded * x_reshaped
        weighted_features_reshaped = weighted_features.view(b, self.hidden_channels, d, h, w)

        aggregated_vectors_per_head = weighted_features.sum(dim=(3, 4, 5))  # (B, num_heads, head_dim)
        aggregated_vector = aggregated_vectors_per_head.view(b, self.hidden_channels, 1, 1, 1)

        prediction = self.classifier(aggregated_vector)[..., 0, 0, 0]
        prediction = prediction + self.classifier_bias.view(1, -1)

        if train:
            prob_map = self.classifier(x) # return activation map
        else:
            prob_map = self.classifier(weighted_features_reshaped) # return prediction heatmap

        return prediction, prob_map, attention_map


# ----------------------------------------------------------------------
# Unified visual encoder wrapper
# ----------------------------------------------------------------------
class VisualEncoder(nn.Module):
    """
    Unified wrapper for different visual encoders.

    All submodels are called with a unified signature:
        forward(x, modalities=None, train=False)

    Output format is normalized to:
        {
            "score": Tensor,
            "prob": Optional[Tensor],
            "SA_map": Optional[Tensor],
        }
    """

    def __init__(
        self,
        encoder_name: str = "VoCo",
        in_channels: int = 1,
        number_of_classes: int = 2,
        finetuned_backbone: Optional[str] = None,
        image_size: Tuple[int, int, int] = (96, 96, 96),
        pretrained_path: str = "pretrain_weights/VoCo/VoComni_B.pt",
        num_heads: int = 1,
    ) -> None:
        super().__init__()

        self.encoder_name = encoder_name

        encoder_registry = {
            "VoCo": lambda: VoCo(
                in_channels=in_channels,
                num_classes=number_of_classes,
                image_size=image_size,
                pretrained_path=pretrained_path,
            ),
            "ViT_Classifier": lambda: ViTClassifier(
                in_channels=in_channels,
                num_classes=number_of_classes,
                image_size=image_size,
            ),
            "VoCo_Salient_2": lambda: VoCoSalient2(
                in_channels=in_channels,
                image_size=image_size,
                pretrained_path=pretrained_path,
                num_heads=num_heads,
            ),
            "BrainMVP": lambda: make_model(
                in_channels=in_channels,
                out_channels=number_of_classes,
                img_size=image_size[0],
            ),
        }

        if encoder_name not in encoder_registry:
            raise ValueError(f"Unsupported encoder name: {encoder_name}")

        self.encoder = encoder_registry[encoder_name]()

        if finetuned_backbone is not None:
            self._load_finetuned_backbone(finetuned_backbone)
        else:
            print("[Info] Using model's own pretrained (or random) weights logic.")

    def _load_finetuned_backbone(self, finetuned_backbone: str) -> None:
        print(f"[Info] Loading finetuned backbone from: {finetuned_backbone}")
        state_dict = torch.load(finetuned_backbone, map_location="cpu", weights_only=False)

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        corrected_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                corrected_state_dict[k[len("encoder."):]] = v
            elif k.startswith("resnet."):
                corrected_state_dict["features" + k[len("resnet"):]] = v
            else:
                corrected_state_dict[k] = v

        missing_keys, unexpected_keys = self.encoder.load_state_dict(corrected_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print("[Info] Loaded finetuned backbone weights with corrected keys.")

    def forward(
        self,
        x: torch.Tensor,
        modalities: Optional[torch.Tensor] = None,
        train: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        out = self.encoder(x, modalities=modalities, train=train)

        if isinstance(out, dict):
            return out

        return {
            "score": out,
            "prob": None,
            "SA_map": None,
        }


# ----------------------------------------------------------------------
# ViT classifier
# ----------------------------------------------------------------------
class ViTClassifier(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 1,
        num_classes: int = 2,
        hidden_size: int = 768,
        pretrained_path: str = "pretrain_weights/vit/mae_BrainMRI_patch_mni152_ukb.pt",
    ) -> None:
        super().__init__()

        self.features = ViT(
            img_size=image_size,
            patch_size=(16, 16, 16),
            hidden_size=hidden_size,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            in_chans=in_channels,
            dropout_rate=0.0,
            spatial_dims=3,
            patch_embed="conv",
            pos_embed="sincos",
            classification=False,
            num_classes=num_classes,
            post_activation="Tanh",
            qkv_bias=True,
            use_flash_attn=False,
            pooling="patch_level",
        )
        self.class_head = nn.Linear(hidden_size, num_classes)

        loaded_state_dict = torch.load(pretrained_path, map_location="cpu")["state_dict"]
        new_state_dict = {k.replace("module.", ""): v for k, v in loaded_state_dict.items()}
        interpolate_pos_embed(self.features, new_state_dict)

        missing_keys, unexpected_keys = self.features.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print(f"Loaded pretrained: {pretrained_path} (3D ViT).")

    def forward(
        self,
        x: torch.Tensor,
        modalities: Optional[torch.Tensor] = None,
        train: bool = False,
    ) -> torch.Tensor:
        del modalities, train
        x = self.features(x)
        x = torch.mean(x, dim=1)
        return self.class_head(x)


# ----------------------------------------------------------------------
# VoCo
# ----------------------------------------------------------------------
class VoCo(nn.Module):
    def __init__(
        self,
        feature_dimension: int = 256,
        image_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        num_classes: int = 2,
        pretrained_path: str = "pretrain_weights/VoCo/VoComni_B.pt",
    ) -> None:
        super().__init__()
        del feature_dimension

        self.features = SwinUNETR(
            img_size=image_size,
            in_channels=in_channels,
            out_channels=21,
            feature_size=48,
            use_checkpoint=True,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            drop_rate=0.0,
            use_v2=True,
        )
        self.features.out = nn.Identity()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.class_head = nn.Linear(48, num_classes)

        if os.path.isfile(pretrained_path):
            load_backbone_weights(
                model=self.features,
                checkpoint_path=pretrained_path,
                in_channels=in_channels,
                first_conv_keys=[
                    "swinViT.patch_embed.proj.weight",
                    "encoder1.layer.conv1.conv.weight",
                    "encoder1.layer.conv3.conv.weight",
                ],
                key_prefixes_to_remove=("module.",),
                key_prefixes_to_replace=None,
                strict=False,
            )
        else:
            print(f"[Warning] No pretrained file found at: {pretrained_path} => random init.")

    def forward(
        self,
        x: torch.Tensor,
        modalities: Optional[torch.Tensor] = None,
        train: bool = False,
    ) -> torch.Tensor:
        del modalities, train
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.class_head(x)


# ----------------------------------------------------------------------
# VoCo Salient
# ----------------------------------------------------------------------
class VoCoSalient2(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 1,
        num_classes: int = 2,
        num_heads: int = 1,
        pretrained_path: str = "pretrain_weights/VoCo/VoComni_B.pt",
    ) -> None:
        super().__init__()
        del num_classes  # current predictor uses n_classes=1 by design

        self.features = SwinUNETR(
            img_size=image_size,
            in_channels=in_channels,
            out_channels=21,
            feature_size=48,
            use_checkpoint=True,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            drop_rate=0.0,
            use_v2=True,
        )
        self.features.out = nn.Identity()

        self.predictor = MultiABMILPredictor(
            input_channels=48,
            hidden_channels=128,
            head_dim=128 // num_heads,
            num_heads=num_heads,
            n_classes=1,
        )

        if os.path.isfile(pretrained_path):
            load_backbone_weights(
                model=self.features,
                checkpoint_path=pretrained_path,
                in_channels=in_channels,
                first_conv_keys=[
                    "swinViT.patch_embed.proj.weight",
                    "encoder1.layer.conv1.conv.weight",
                    "encoder1.layer.conv3.conv.weight",
                ],
                key_prefixes_to_remove=("module.",),
                key_prefixes_to_replace=None,
                strict=False,
            )
        else:
            print(f"[Warning] No pretrained file found at: {pretrained_path} => random init.")

    def forward(
        self,
        x: torch.Tensor,
        modalities: Optional[torch.Tensor] = None,
        train: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        del modalities
        features = self.features(x)
        score, prob, sa_map = self.predictor(features, train=train)

        return {
            "score": score,
            "prob": prob,
            "SA_map": sa_map,
        }