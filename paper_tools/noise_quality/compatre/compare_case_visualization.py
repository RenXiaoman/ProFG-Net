#!/usr/bin/env python3
"""Generate multi-method contour visualization for selected PI-CAI cases."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, generate_binary_structure, label


METHOD_ORDER = ["UNet", "UNETR", "nnUNet", "TransUNet", "BMA", "ResGNet", "WaveFormer", "AWUNet", "WaTER", "FreGNet"]
DISPLAY_NAMES = {"UNet": "UNet", "UNETR": "UNETR", "nnUNet": "nnU-Net", "TransUNet": "TransUNet", "BMA": "BMA-Net", "ResGNet": "ResGNet", "WaveFormer": "WaveFormer", "AWUNet": "AWUNet", "WaTER": "WaTER", "FreGNet": "Ours"}
LEVELS = {
    "gaussian_noise": [0.01, 0.03, 0.05],
    "speckle_noise": [0.05, 0.10, 0.20],
    "contrast_reduction": [0.8, 0.6, 0.4],
    "inplane_gaussian_blur": [0.5, 1.0, 1.5],
}
DEGRADATIONS = tuple(LEVELS)


def read_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        array = np.load(path)
    else:
        array = np.asarray(nib.load(str(path)).get_fdata())
    if array.ndim == 3 and array.shape[-1] < array.shape[0]:
        array = np.transpose(array, (2, 0, 1))
    return array


def edge(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool) & ~binary_erosion(mask.astype(bool), generate_binary_structure(2, 1))


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 26-connected foreground component in a 3-D mask."""
    mask = mask.astype(bool)
    labeled, component_count = label(mask, structure=generate_binary_structure(3, 3))
    if component_count == 0:
        return mask
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignore background when selecting the largest region.
    return labeled == int(component_sizes.argmax())


def degrade(image: np.ndarray, kind: str, strength: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "gaussian_noise":
        result = image + rng.normal(0.0, strength, image.shape).astype(np.float32)
    elif kind == "speckle_noise":
        result = image * (1.0 + strength * rng.normal(0.0, 1.0, image.shape).astype(np.float32))
    elif kind == "contrast_reduction":
        mean = image.mean(dtype=np.float32)
        result = mean + strength * (image - mean)
    elif kind == "inplane_gaussian_blur":
        result = gaussian_filter(image, sigma=(0.0, strength, strength), mode="nearest")
    else:
        raise ValueError(kind)
    return result.astype(np.float32)


def combined_image(image: np.ndarray, level_index: int, case: str, base_seed: int = 20260831) -> np.ndarray:
    result = image.astype(np.float32, copy=True)
    for kind in DEGRADATIONS:
        token = f"{case}:combined:{level_index}:{kind}"
        seed = base_seed + int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
        result = degrade(result, kind, float(LEVELS[kind][level_index]), seed)
    return result


def draw(ax, image: np.ndarray, gt: np.ndarray, pred: np.ndarray | None, title: str | None, vmin: float, vmax: float, annotate: bool = True) -> None:
    ax.imshow(image.T, cmap="gray", vmin=vmin, vmax=vmax)
    yy, xx = np.where(edge(gt.T))
    ax.plot(xx, yy, ".", color="red", markersize=0.7, linewidth=0)
    if pred is not None:
        yy, xx = np.where(edge(pred.T))
        ax.plot(xx, yy, ".", color="yellow", markersize=0.7, linewidth=0)
    if annotate and title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_method_png(path: Path, image: np.ndarray, gt: np.ndarray, pred: np.ndarray, vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    draw(ax, image, gt, pred, None, vmin, vmax, annotate=False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_mri_png(path: Path, image: np.ndarray, gt: np.ndarray | None, vmin: float, vmax: float) -> None:
    """Save a clean MRI panel, optionally with the red GT contour."""
    fig, ax = plt.subplots(figsize=(4, 4))
    draw(ax, image, np.zeros_like(image, dtype=bool) if gt is None else gt, None, None, vmin, vmax, annotate=False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="10059_1000059")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--raw_dataset", type=Path, default=Path("/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw/Dataset141_FullPICAI"))
    parser.add_argument("--our_predictions", type=Path, default=Path("/home/Space/clib/Projects/nnUNetv2/paper_tools/noise_quality/predictions/a6a0ed2f9ae0/FreGNet"))
    parser.add_argument("--other_predictions", type=Path, default=Path("/home/Space/clib/Projects/CAD/paper_tools/noise_quality/noise_robustness/predictions"))
    parser.add_argument("--output_dir", type=Path, default=Path("/home/Space/clib/Projects/nnUNetv2/paper_tools/noise_quality/compatre/case_visual"))
    args = parser.parse_args()
    case = args.case
    case_output = args.output_dir / case
    case_output.mkdir(parents=True, exist_ok=True)
    image = read_array(args.raw_dataset / "imagesTr" / f"{case}_0000.nii.gz").astype(np.float32)
    gt = read_array(args.raw_dataset / "labelsTr" / f"{case}.nii.gz") > 0
    if image.shape != gt.shape:
        raise ValueError(f"Image/GT shape mismatch: {image.shape} vs {gt.shape}")

    condition_names = ["original", "combined_1", "combined_2", "combined_3"]
    predictions_by_condition = {}
    for condition in condition_names:
        our_dir = args.our_predictions / ({"original": "original", "combined_1": "combined1", "combined_2": "combined2", "combined_3": "combined3"}[condition])
        predictions = {"FreGNet": read_array(our_dir / f"{case}.npy") > 0}
        for method in METHOD_ORDER[:-1]:
            path = args.other_predictions / method / case / f"{condition}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            predictions[method] = read_array(path) > 0
        # Apply the same 3-D post-processing to every method before selecting
        # slices or drawing contours. Ground truth is intentionally unchanged.
        predictions = {
            method: keep_largest_connected_component(pred)
            for method, pred in predictions.items()
        }
        for method, pred in predictions.items():
            if pred.shape != gt.shape:
                raise ValueError(f"Shape mismatch for {method}/{condition}: {pred.shape} vs {gt.shape}")
        predictions_by_condition[condition] = predictions

    foreground_slices = np.where(gt.reshape(gt.shape[0], -1).any(axis=1))[0]
    if foreground_slices.size == 0:
        raise ValueError("Ground truth has no foreground")
    center = int(round(float(foreground_slices[0] + foreground_slices[-1]) / 2.0))
    vmin, vmax = float(np.percentile(image, 1)), float(np.percentile(image, 99))

    columns = ["MRI", "MRI + GT"] + [DISPLAY_NAMES[m] for m in METHOD_ORDER]
    displayed_images = [image] + [combined_image(image, i, case, args.seed) for i in range(3)]
    row_labels = ["Original", "Level-1", "Level-2", "Level-3"]
    fig, axes = plt.subplots(4, len(columns), figsize=(2.2 * len(columns), 2.6 * 4), squeeze=False, constrained_layout=True)
    for row, (condition, row_label, row_image) in enumerate(zip(condition_names, row_labels, displayed_images)):
        z = center
        draw(axes[row, 0], row_image[z], np.zeros_like(gt[z]), None, f"{row_label}\nMRI", vmin, vmax)
        draw(axes[row, 1], row_image[z], gt[z], None, f"{row_label}\nMRI + GT", vmin, vmax)
        for col, method in enumerate(METHOD_ORDER, start=2):
            draw(axes[row, col], row_image[z], gt[z], predictions_by_condition[condition][method][z], DISPLAY_NAMES[method], vmin, vmax)
    fig.suptitle(f"{case} | synthetic degradation comparison | z={center}", fontsize=13)
    output = case_output / f"{case}_all_conditions_multimethod.png"
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"Rows: {row_labels}; slice: {center}")

    condition_dirs = {"original": "original", "combined_1": "level1", "combined_2": "level2", "combined_3": "level3"}
    for condition, row_image in zip(condition_names, displayed_images):
        out_dir = case_output / condition_dirs[condition]
        out_dir.mkdir(parents=True, exist_ok=True)
        save_mri_png(out_dir / "MRI.png", row_image[center], None, vmin, vmax)
        save_mri_png(out_dir / "MRI_GT.png", row_image[center], gt[center], vmin, vmax)
        for method in METHOD_ORDER:
            save_method_png(out_dir / f"{DISPLAY_NAMES[method].replace('-', '')}.png", row_image[center], gt[center], predictions_by_condition[condition][method][center], vmin, vmax)
    print(f"Per-method panels saved under: {case_output}")


if __name__ == "__main__":
    main()
