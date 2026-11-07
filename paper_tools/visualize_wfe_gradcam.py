#!/usr/bin/env python3
"""Visualize 3D Grad-CAM immediately before and after the first WFE block."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import numpy as np
import torch
from pytorch_grad_cam import GradCAM

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.DIY_boundary_msag import (
    DIYBoundaryCMSAG,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = (
    PROJECT_ROOT
    / "nnUNet/nnUNet_preprocessed/Dataset141_FullPICAI/nnUNetPlans_3d_fullres"
)
SPLITS_FILE = (
    PROJECT_ROOT / "nnUNet/nnUNet_preprocessed/Dataset141_FullPICAI/splits_final.json"
)
CHECKPOINT = (
    PROJECT_ROOT
    / "nnUNet/nnUNet_results/Dataset141_FullPICAI/"
    "DIYBoundaryCMSAGAuxTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "paper_tools/Dataset141_WFE_GradCAM"


class GlandSegmentationTarget:
    """Sum foreground logits inside the ground-truth prostate mask."""

    def __init__(self, category: int, mask: np.ndarray):
        self.category = category
        self.mask = torch.as_tensor(mask, dtype=torch.float32)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        mask = self.mask.to(device=model_output.device, dtype=model_output.dtype)
        return (model_output[self.category] * mask).sum()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=2)
    parser.add_argument("--seed", type=int, default=141)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> DIYBoundaryCMSAG:
    model = DIYBoundaryCMSAG(
        in_channels=1,
        out_channels=2,
        spatial_dims=3,
        feature_size=16,
        norm_name="instance",
        dropout_rate=0.0,
        depth=4,
        boundary_channels=1,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["network_weights"], strict=True)
    model.eval()
    return model


def compute_gradcams(
    model: DIYBoundaryCMSAG,
    image: np.ndarray,
    gt: np.ndarray,
    device: torch.device,
    stage: int,
) -> tuple[np.ndarray, np.ndarray]:
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    target = GlandSegmentationTarget(category=1, mask=gt)

    def gradcam(target_layer: torch.nn.Module) -> np.ndarray:
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=image_tensor, targets=[target])[0, :]
        return grayscale_cam.astype(np.float32, copy=False)

    # Stage 1 uses stem -> enhance[0]. Later stages use enc{stage-1} -> enhance[stage-1].
    before_layer = model.backbone.stem if stage == 1 else getattr(model.backbone, f"enc{stage - 1}")
    after_layer = model.backbone.enhance[stage - 1]
    before_wfe = gradcam(before_layer)
    after_wfe = gradcam(after_layer)
    return before_wfe, after_wfe


def display_image(image_slice: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image_slice, (1, 99))
    if hi <= lo:
        return np.zeros_like(image_slice, dtype=np.float32)
    return np.clip((image_slice - lo) / (hi - lo), 0, 1)


def save_case_figures(
    case_id: str,
    image: np.ndarray,
    gt: np.ndarray,
    before_wfe: np.ndarray,
    after_wfe: np.ndarray,
    output_dir: Path,
    stage: int,
) -> None:
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    depth = image.shape[1]
    if before_wfe.shape != image.shape[1:] or after_wfe.shape != image.shape[1:]:
        raise ValueError(
            f"CAM/image shape mismatch for {case_id}: image={image.shape}, "
            f"before={before_wfe.shape}, after={after_wfe.shape}"
        )

    for slice_index in range(depth):
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), constrained_layout=True)
        axes[0].imshow(display_image(image[0, slice_index]), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("T2W", fontsize=11)
        if np.any(gt[slice_index] > 0):
            axes[0].contour(
                gt[slice_index],
                levels=[0.5],
                colors=["magenta"],
                linewidths=1.5,
                alpha=0.5,
            )
        before_im = axes[1].imshow(before_wfe[slice_index], cmap="jet", norm=NoNorm())
        axes[1].set_title(f"Before WFE-S{stage}", fontsize=11)
        axes[2].imshow(after_wfe[slice_index], cmap="jet", norm=NoNorm())
        axes[2].set_title(f"After WFE-S{stage}", fontsize=11)
        for ax in axes:
            ax.axis("off")
        fig.colorbar(
            before_im,
            ax=axes[1:],
            fraction=0.035,
            pad=0.02,
            label="Grad-CAM",
        )
        fig.savefig(case_dir / f"slice_{slice_index:02d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = OUTPUT_DIR if args.stage == 1 else OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_Encoder{args.stage}"

    with SPLITS_FILE.open() as f:
        val_cases = json.load(f)[0]["val"]
    selected_cases = random.Random(args.seed).sample(val_cases, args.num_cases)

    output_dir.mkdir(parents=True, exist_ok=True)
    before_layer_name = "backbone.stem" if args.stage == 1 else f"backbone.enc{args.stage - 1}"
    after_layer_name = f"backbone.enhance.{args.stage - 1}"
    metadata = {
        "seed": args.seed,
        "cases": selected_cases,
        "checkpoint": str(args.checkpoint.resolve()),
        "target": "sum of foreground logits inside the ground-truth gland mask",
        "stage": args.stage,
        "before_wfe_layer": before_layer_name,
        "after_wfe_layer": after_layer_name,
        "normalization": "standard pytorch-grad-cam output via cam(...)[0, :]",
        "matplotlib_norm": "NoNorm",
        "note": "CAM values follow the library's default Grad-CAM pipeline; imshow does not apply min-max normalization.",
    }
    with (output_dir / "selection.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    dataset = nnUNetDatasetBlosc2(str(PREPROCESSED_DIR), identifiers=selected_cases)
    model = load_model(args.checkpoint, device)
    for case_id in selected_cases:
        data, seg, _, _ = dataset.load_case(case_id)
        image = np.asarray(data[:], dtype=np.float32)
        gt = (np.asarray(seg[:])[0] == 1).astype(np.float32)
        before_wfe, after_wfe = compute_gradcams(model, image, gt, device, args.stage)
        save_case_figures(
            case_id,
            image,
            gt,
            before_wfe,
            after_wfe,
            output_dir,
            args.stage,
        )
        print(f"Saved {image.shape[1]} slices for {case_id}")

    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
