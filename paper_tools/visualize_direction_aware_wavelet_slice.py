#!/usr/bin/env python3
"""Visualize DirectionAwareLearnableWavelet outputs on random Dataset141 slices."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCM import DirectionAwareLearnableWavelet


IMAGES_DIR = PROJECT_ROOT / "nnUNet/nnUNet_raw/Dataset141_FullPICAI/imagesTr"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paper_tools/DirectionAwareWavelet_Slice_FeatureMaps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-id", action="append", default=None)
    return parser.parse_args()


def robust_normalize(image: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Normalize an image to [0, 1] using robust percentiles."""
    lower, upper = np.percentile(image, (lo, hi))
    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def choose_slice(volume: np.ndarray) -> tuple[int, np.ndarray]:
    """Choose the slice with the largest robust intensity variation."""
    lower, upper = np.percentile(volume, (1.0, 99.0))
    clipped = np.clip(volume, lower, upper)
    slice_scores = clipped.std(axis=(0, 1))
    slice_index = int(np.argmax(slice_scores))
    return slice_index, volume[:, :, slice_index].astype(np.float32)


def plot_case(
    case_id: str,
    image_slice: np.ndarray,
    wavelet_outputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output_path: Path,
) -> None:
    s_ll, d_hl, d_lh, d_hh = wavelet_outputs
    titles = ["Input", "s_LL", "d_HL", "d_LH", "d_HH"]
    maps = [image_slice, s_ll, d_hl, d_lh, d_hh]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), constrained_layout=True)
    for ax, title, feature_map in zip(axes, titles, maps):
        if title == "Input":
            im = ax.imshow(feature_map, cmap="gray", vmin=0.0, vmax=1.0)
        elif title == "s_LL":
            im = ax.imshow(feature_map, cmap="viridis")
        else:
            limit = max(float(np.max(np.abs(feature_map))), 1e-8)
            im = ax.imshow(feature_map, cmap="RdBu_r", vmin=-limit, vmax=limit)

        ax.set_title(title, fontsize=12)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"DirectionAwareLearnableWavelet - {case_id}", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    image_files = sorted(args.images_dir.glob("*_0000.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    if args.case_id:
        selected_files = []
        for case_id in args.case_id:
            matches = [path for path in image_files if path.name.startswith(f"{case_id}_")]
            if matches:
                selected_files.append(matches[0])
        if not selected_files:
            raise FileNotFoundError("None of the requested case IDs were found.")
    else:
        selected_files = rng.sample(image_files, min(args.num_cases, len(image_files)))

    wavelet = DirectionAwareLearnableWavelet(channels=1).eval()

    for image_path in selected_files:
        case_id = image_path.name.split("_0000.nii.gz")[0]
        volume = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D image for {case_id}, got {volume.shape}")

        slice_index, image_slice = choose_slice(volume)
        image_slice_norm = robust_normalize(image_slice)
        x = torch.from_numpy(image_slice_norm).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            s_ll, d_hl, d_lh, d_hh = wavelet(x)

        outputs = (
            s_ll[0, 0].numpy(),
            d_hl[0, 0].numpy(),
            d_lh[0, 0].numpy(),
            d_hh[0, 0].numpy(),
        )
        output_path = args.output_dir / f"{case_id}_slice{slice_index:03d}.png"
        plot_case(case_id, image_slice_norm, outputs, output_path)
        print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
