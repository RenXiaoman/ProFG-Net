#!/usr/bin/env python3
"""Visualize one representative probability-map slice per dataset/split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "130": "Dataset130_ProstateAHCDU",
    "141": "Dataset141_FullPICAI",
}


def read_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def load_foreground_probability(path: Path) -> np.ndarray:
    with np.load(path) as data:
        key = "probabilities" if "probabilities" in data else "softmax"
        values = np.asarray(data[key], dtype=np.float32)
    return values[1] if values.ndim == 4 else values


def normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, (0.5, 99.5))
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = int(pred.sum()) + int(gt.sum())
    return 1.0 if denom == 0 else 2.0 * float(np.logical_and(pred, gt).sum()) / denom


def choose_case(dataset_id: str, split: str, case_arg: str | None) -> str:
    if case_arg:
        return case_arg
    split_path = REPO_ROOT / "nnUNet/nnUNet_preprocessed" / DATASETS[dataset_id] / "splits_final.json"
    split_data = json.loads(split_path.read_text(encoding="utf-8"))[0]
    cases = split_data["train" if split == "labelsTr" else "val"]
    return sorted(cases)[0]


def draw_case(dataset_id: str, split: str, case: str, output_root: Path) -> Path:
    dataset = DATASETS[dataset_id]
    raw = REPO_ROOT / "nnUNet/nnUNet_raw" / dataset
    gt = read_array(raw / "labelsTr" / f"{case}.nii.gz") > 0
    image = normalize(read_array(raw / "imagesTr" / f"{case}_0000.nii.gz"))
    prob = load_foreground_probability(REPO_ROOT / "Prob" / dataset_id / split / f"{case}.npz")
    if image.shape != gt.shape or prob.shape != gt.shape:
        raise ValueError(f"Shape mismatch for {dataset_id}/{split}/{case}: image={image.shape}, gt={gt.shape}, prob={prob.shape}")

    areas = gt.sum(axis=(1, 2))
    slice_index = int(np.argmax(areas))
    image_slice, gt_slice, prob_slice = image[slice_index], gt[slice_index], prob[slice_index]
    pred_slice = prob_slice >= 0.5
    score = dice(pred_slice, gt_slice)

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.0), dpi=250, facecolor="white")
    axes[0].imshow(image_slice, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("MRI", fontsize=11)
    axes[1].imshow(image_slice, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].contour(gt_slice.astype(np.uint8), levels=[0.5], colors=["#E60000"], linewidths=1.5)
    axes[1].set_title("MRI + GT", fontsize=11)
    im = axes[2].imshow(prob_slice, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Foreground probability", fontsize=11)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    axes[3].imshow(image_slice, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[3].contour(gt_slice.astype(np.uint8), levels=[0.5], colors=["#E60000"], linewidths=1.5)
    axes[3].contour(pred_slice.astype(np.uint8), levels=[0.5], colors=["#FFD400"], linewidths=1.5)
    axes[3].set_title(f"GT + prediction (Dice={score:.3f})", fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Dataset {dataset_id} | {split} | {case} | slice {slice_index}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_dir = output_root / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{split}_{case}_slice{slice_index:03d}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "paper_tools/prob_visualization")
    parser.add_argument("--case_130_train", default=None)
    parser.add_argument("--case_130_val", default=None)
    parser.add_argument("--case_141_train", default=None)
    parser.add_argument("--case_141_val", default=None)
    args = parser.parse_args()

    for dataset_id in ("130", "141"):
        for split, suffix in (("labelsTr", "train"), ("labelsTs", "val")):
            case = getattr(args, f"case_{dataset_id}_{suffix}")
            output = draw_case(dataset_id, split, choose_case(dataset_id, split, case), args.output_root)
            print(f"Wrote {output}")


if __name__ == "__main__":
    main()
