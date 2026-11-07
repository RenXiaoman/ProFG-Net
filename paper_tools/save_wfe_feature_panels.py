#!/usr/bin/env python3
"""Save selected WFE feature maps as individual PNG files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_erosion, generate_binary_structure

from visualize_wfe_feature_maps import (
    CHECKPOINT,
    PREPROCESSED_DIR,
    extract_feature_magnitudes,
    load_model,
    normalize_for_plot,
)
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paper_tools/Dataset141_WFE_FeatureMaps_Single"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save five selected panels (T2W, T2W+GT, Before WFE, Stem Conv1, After WFE) "
        "for one case and slice, without titles or colorbars."
    )
    parser.add_argument("--case", required=True, help="Validation case identifier, e.g. 10012_1000012")
    parser.add_argument("--slice", type=int, required=True, dest="slice_index", help="Zero-based axial slice index")
    parser.add_argument("--norm", action="store_true", help="Normalize each displayed map to [0, 1] using the 1st/99th percentiles")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def save_panel(path: Path, values: np.ndarray, title: str, normalize: bool) -> None:
    shown = normalize_for_plot(values) if normalize else values
    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    image = ax.imshow(shown, cmap="jet", interpolation="nearest")
    # ax.set_title(title, fontsize=11)
    ax.axis("off")
    # fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Normalized activation" if normalize else "RMS activation")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_mri_panel(path: Path, image_slice: np.ndarray, gt_slice: np.ndarray | None, normalize: bool) -> None:
    shown = normalize_for_plot(image_slice) if normalize else image_slice
    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    ax.imshow(shown, cmap="gray", interpolation="nearest")
    if gt_slice is not None and np.any(gt_slice):
        boundary = gt_slice.astype(bool) & ~binary_erosion(
            gt_slice.astype(bool), generate_binary_structure(2, 1)
        )
        yy, xx = np.where(boundary)
        ax.plot(xx, yy, ".", color="red", markersize=1.0, linewidth=0)
    ax.axis("off")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dataset = nnUNetDatasetBlosc2(str(PREPROCESSED_DIR), identifiers=[args.case])
    data, seg, _, _ = dataset.load_case(args.case)
    image = np.asarray(data[:], dtype=np.float32)
    gt = (np.asarray(seg[:])[0] == 1).astype(np.uint8)
    if args.slice_index < 0 or args.slice_index >= image.shape[1]:
        raise ValueError(f"--slice must be in [0, {image.shape[1] - 1}], got {args.slice_index}")

    model = load_model(args.checkpoint, device)
    before, _raw_low, _low, _raw_high, _modulated_high, stem_conv1, after, _, _ = extract_feature_magnitudes(
        model, image, device
    )

    output_dir = args.output_dir / args.case
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"slice_{args.slice_index:02d}"
    save_mri_panel(output_dir / f"{suffix}_mri.png", image[0, args.slice_index], None, args.norm)
    save_mri_panel(output_dir / f"{suffix}_mri_gt.png", image[0, args.slice_index], gt[args.slice_index], args.norm)
    save_panel(output_dir / f"{suffix}_feature_before_wfe.png", before[args.slice_index], "Feature Before WFE", args.norm)
    save_panel(output_dir / f"{suffix}_stem_conv1.png", stem_conv1[args.slice_index], "Stem Conv1 Feature", args.norm)
    save_panel(output_dir / f"{suffix}_feature_after_wfe.png", after[args.slice_index], "Feature After WFE", args.norm)

    print(f"Saved 5 panels for {args.case}, slice {args.slice_index} to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
