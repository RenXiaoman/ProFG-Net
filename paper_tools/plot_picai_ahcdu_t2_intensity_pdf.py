#!/usr/bin/env python3
"""Compare PI-CAI and AHCDU T2W intensity distributions.

The two panels show valid non-zero whole-image voxels and prostate-gland
voxels, respectively. Raw intensities are used by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


DATASETS = {
    "PI-CAI": "Dataset141_FullPICAI",
    "AHCDU": "Dataset130_ProstateAHCDU",
}
COLORS = {"PI-CAI": "#0072B2", "AHCDU": "#D55E00"}
LINE_STYLES = {"PI-CAI": "-", "AHCDU": "-"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot whole-image and prostate-gland T2W intensity PDFs for PI-CAI and AHCDU."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("nnUNet/nnUNet_raw"),
        help="Directory containing Dataset141_FullPICAI and Dataset130_ProstateAHCDU.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_tools/t2_intensity_distribution"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=("train", "test"),
        help="Dataset partitions to include. Both are included by default.",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Cases per dataset; 0 uses all cases.")
    parser.add_argument("--max-voxels-per-case", type=int, default=20000)
    parser.add_argument("--max-voxels-per-region", type=int, default=500000)
    parser.add_argument("--clip-percentiles", nargs=2, type=float, default=(0.5, 99.5), metavar=("LOW", "HIGH"))
    parser.add_argument("--bins", type=int, default=500)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=141130)
    parser.add_argument(
        "--normalization",
        choices=("none", "zscore", "minmax"),
        default="none",
        help="Optional per-case normalization over valid non-zero voxels.",
    )
    return parser.parse_args()


def strip_nii(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def normalize_case(image: np.ndarray, valid: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image
    values = image[valid]
    if values.size == 0:
        return image
    if mode == "zscore":
        return (image - float(values.mean())) / max(float(values.std()), 1e-6)
    lo, hi = float(values.min()), float(values.max())
    return (image - lo) / max(hi - lo, 1e-6)


def sample(values: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    values = values.reshape(-1)
    if limit <= 0 or values.size <= limit:
        return values.astype(np.float32, copy=False)
    indices = rng.choice(values.size, size=limit, replace=False)
    return values[indices].astype(np.float32, copy=False)


def case_pairs(dataset_root: Path, splits: list[str]) -> list[tuple[Path, Path]]:
    subdirs = {"train": ("imagesTr", "labelsTr"), "test": ("imagesTs", "labelsTs")}
    pairs: list[tuple[Path, Path]] = []
    for split in splits:
        image_dir, label_dir = (dataset_root / item for item in subdirs[split])
        for image_path in sorted(image_dir.glob("*_0000.nii.gz")):
            case = strip_nii(image_path.name).removesuffix("_0000")
            label_path = label_dir / f"{case}.nii.gz"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            pairs.append((image_path, label_path))
    return pairs


def collect_dataset(
    dataset_name: str,
    dataset_root: Path,
    splits: list[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict]:
    pairs = case_pairs(dataset_root, splits)
    if not pairs:
        raise FileNotFoundError(f"No matched T2W cases found under {dataset_root}")
    if args.max_cases > 0 and len(pairs) > args.max_cases:
        indices = np.sort(rng.choice(len(pairs), size=args.max_cases, replace=False))
        pairs = [pairs[int(i)] for i in indices]

    whole_parts: list[np.ndarray] = []
    gland_parts: list[np.ndarray] = []
    for image_path, label_path in tqdm(pairs, desc=f"Collecting {dataset_name}", unit="case"):
        image = load_nifti(image_path)
        label = load_nifti(label_path)
        if image.shape != label.shape:
            raise ValueError(f"Shape mismatch: {image_path} {image.shape} vs {label_path} {label.shape}")
        valid = np.isfinite(image) & (image != 0)
        image = normalize_case(image, valid, args.normalization)
        whole_parts.append(sample(image[valid], args.max_voxels_per_case, rng))
        gland_parts.append(sample(image[valid & (label > 0)], args.max_voxels_per_case, rng))

    values = {
        "whole": sample(np.concatenate(whole_parts), args.max_voxels_per_region, rng),
        "gland": sample(np.concatenate(gland_parts), args.max_voxels_per_region, rng),
    }
    stats = {
        "dataset_root": str(dataset_root),
        "cases": len(pairs),
        "whole_sampled_voxels": int(values["whole"].size),
        "gland_sampled_voxels": int(values["gland"].size),
    }
    return values, stats


def distribution_range(dataset_values: dict[str, dict[str, np.ndarray]], region: str, percentiles: tuple[float, float]) -> tuple[float, float]:
    pooled = np.concatenate([dataset_values[name][region] for name in DATASETS])
    low, high = np.percentile(pooled, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid plotting range for {region}: [{low}, {high}]")
    return float(low), float(high)


def density(values: np.ndarray, low: float, high: float, bins: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values) & (values >= low) & (values <= high)]
    histogram, edges = np.histogram(values, bins=bins, range=(low, high), density=True)
    curve = gaussian_filter1d(histogram.astype(np.float64), sigma=sigma)
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers, curve


def summary(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values, [0.5, 1, 25, 50, 75, 99, 99.5])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        **{f"p{label}": float(value) for label, value in zip(("0.5", "1", "25", "50", "75", "99", "99.5"), q)},
    }


def plot(dataset_values: dict[str, dict[str, np.ndarray]], args: argparse.Namespace) -> tuple[Path, Path, dict]:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    panels = (("whole", "(a) Whole-image intensity distributions"), ("gland", "(b) Gland intensity distributions"))
    ranges = {}
    for ax, (region, title) in zip(axes, panels):
        low, high = distribution_range(dataset_values, region, tuple(args.clip_percentiles))
        ranges[region] = [low, high]
        for dataset_name in DATASETS:
            x, y = density(dataset_values[dataset_name][region], low, high, args.bins, args.smooth_sigma)
            ax.plot(x, y, color=COLORS[dataset_name], linestyle=LINE_STYLES[dataset_name], linewidth=1.8, label=dataset_name)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Intensity" if args.normalization == "none" else f"Intensity ({args.normalization})")
        ax.set_ylabel("Probability density")
        ax.set_xlim(low, high)
        ax.legend(loc="best")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "picai_ahcdu_t2_intensity_distributions.pdf"
    png_path = args.output_dir / "picai_ahcdu_t2_intensity_distributions.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path, ranges


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dataset_values: dict[str, dict[str, np.ndarray]] = {}
    metadata = {
        "normalization": args.normalization,
        "splits": list(args.splits),
        "seed": args.seed,
        "clip_percentiles": list(args.clip_percentiles),
        "max_voxels_per_case": args.max_voxels_per_case,
        "max_voxels_per_region": args.max_voxels_per_region,
        "datasets": {},
    }
    for dataset_name, folder in DATASETS.items():
        values, stats = collect_dataset(dataset_name, args.raw_root / folder, list(args.splits), args, rng)
        dataset_values[dataset_name] = values
        metadata["datasets"][dataset_name] = {
            **stats,
            "whole": summary(values["whole"]),
            "gland": summary(values["gland"]),
        }

    pdf_path, png_path, ranges = plot(dataset_values, args)
    metadata["plot_ranges"] = ranges
    stats_path = args.output_dir / "picai_ahcdu_t2_intensity_statistics.json"
    stats_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved statistics: {stats_path}")


if __name__ == "__main__":
    main()
