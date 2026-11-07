#!/usr/bin/env python3
"""Plot raw and per-case normalized T2 intensity distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter1d


def image_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*_0000.nii.gz"))


def collect_samples(
    files: list[Path],
    rng: np.random.Generator,
    max_voxels_per_case: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    raw_parts: list[np.ndarray] = []
    standardized_parts: list[np.ndarray] = []
    case_means: list[float] = []
    case_stds: list[float] = []

    for path in files:
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32, copy=False)
        values = image[np.isfinite(image) & (image != 0)]
        if values.size == 0:
            continue
        if values.size > max_voxels_per_case:
            values = rng.choice(values, size=max_voxels_per_case, replace=False)

        mean = float(values.mean())
        std = float(values.std())
        case_means.append(mean)
        case_stds.append(std)
        raw_parts.append(values)
        standardized_parts.append((values - mean) / max(std, 1e-6))

    stats = {
        "case_mean_mean": float(np.mean(case_means)),
        "case_mean_std": float(np.std(case_means)),
        "case_std_mean": float(np.mean(case_stds)),
        "case_std_std": float(np.std(case_stds)),
    }
    return np.concatenate(raw_parts), np.concatenate(standardized_parts), stats


def density_curve(values: np.ndarray, lower: float, upper: float, bins: int = 500):
    histogram, edges = np.histogram(values, bins=bins, range=(lower, upper), density=True)
    curve = gaussian_filter1d(histogram, sigma=2)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, curve


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    p = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {f"p{q:g}": float(v) for q, v in zip([1, 5, 25, 50, 75, 95, 99], p)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset141",
        type=Path,
        default=Path("nnUNet/nnUNet_raw/Dataset141_FullPICAI/imagesTr"),
    )
    parser.add_argument(
        "--dataset130",
        type=Path,
        default=Path("nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU/imagesTr"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nnUNet/analysis/Dataset141_vs_Dataset130_intensity_distribution.png"),
    )
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--max_voxels_per_case", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=141130)
    args = parser.parse_args()

    files = {"Dataset141": image_files(args.dataset141), "Dataset130": image_files(args.dataset130)}
    if not files["Dataset141"] or not files["Dataset130"]:
        raise FileNotFoundError("Both image folders must contain *_0000.nii.gz files")

    rng = np.random.default_rng(args.seed)
    case_count = min(len(files["Dataset141"]), len(files["Dataset130"]))
    if args.max_cases is not None:
        case_count = min(case_count, args.max_cases)
    for name in files:
        files[name] = list(rng.choice(files[name], size=case_count, replace=False))

    samples = {}
    stats = {
        "case_count_used": case_count,
        "max_voxels_per_case": args.max_voxels_per_case,
        "normalization": "per-case z-score over sampled non-zero intensities",
    }
    for name in ("Dataset141", "Dataset130"):
        raw, standardized, case_stats = collect_samples(files[name], rng, args.max_voxels_per_case)
        samples[name] = (raw, standardized)
        stats[name] = {
            "all_nonzero_sampled_voxels": int(raw.size),
            "raw_percentiles": percentile_summary(raw),
            "standardized_percentiles": percentile_summary(standardized),
            **case_stats,
        }

    raw_lower, raw_upper = np.percentile(
        np.concatenate([samples[name][0] for name in samples]), [0.5, 99.5]
    )
    colors = {"Dataset141": "#1f77b4", "Dataset130": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for name in ("Dataset141", "Dataset130"):
        raw, standardized = samples[name]
        x, y = density_curve(raw, float(raw_lower), float(raw_upper))
        axes[0].plot(x, y, color=colors[name], linewidth=2, label=name)
        x, y = density_curve(standardized, -5, 5)
        axes[1].plot(x, y, color=colors[name], linewidth=2, label=name)

    axes[0].set_title("Raw non-zero T2 intensity")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(raw_lower, raw_upper)
    axes[1].set_title("Per-case z-score T2 intensity")
    axes[1].set_xlabel("Standardized intensity")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(-5, 5)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle(
        f"T2 intensity distribution: Dataset141 vs Dataset130 "
        f"(matched cases={case_count}, max {args.max_voxels_per_case:,} voxels/case)",
        fontsize=13,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    stats_path = args.output.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Saved plot: {args.output}")
    print(f"Saved stats: {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
