#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


CASE_IDS = [
    "11190_1001213",
    "11041_1001061",
    "10012_1000012",
    "10059_1000059",
    "10451_1000459",
    "11345_1001368",
    "10220_1000224",
    "10558_1000570",
    "10484_1000492",
    "11063_1001085",
]

DATASET_ROOT = Path("nnUNet/nnUNet_raw/Dataset131_ProstatePI-CAI")
PRED_DIR = Path("nnUNet/infer/Dataset131_DIYBoundaryMSAGAuxTrainer/0")
OUTPUT_DIR = Path("paper_tools/badcase")
TOP_K = 5

GT_COLOR = "#E60000"
PRED_COLOR = "#FFD400"
CONTOUR_WIDTH = 1.5
DPI = 300
PANEL_WIDTH = 1.8
PANEL_HEIGHT = 1.8
TITLE_SIZE = 8
CROP_ENABLED = True
CROP_MARGIN = 24


Array = np.ndarray


@dataclass(frozen=True)
class Settings:
    case_ids: list[str]
    dataset_root: Path
    pred_dir: Path
    output_dir: Path
    top_k: int


@dataclass(frozen=True)
class SliceRow:
    case: str
    slice_index: int
    dice: float
    gt_pixels: int
    pred_pixels: int
    intersection_pixels: int


def read_nii(path: Path) -> Array:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def find_image(dataset_root: Path, case: str) -> Path:
    candidates = [
        dataset_root / "imagesTs" / f"{case}_0000.nii.gz",
        dataset_root / "imagesTs" / f"{case}.nii.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing image for case {case}: tried {candidates}")


def find_label(dataset_root: Path, case: str) -> Path:
    label_file = dataset_root / "labelsTs" / f"{case}.nii.gz"
    if label_file.is_file():
        return label_file
    raise FileNotFoundError(f"Missing GT for case {case}: {label_file}")


def find_prediction(pred_dir: Path, case: str) -> Path:
    candidates = [
        pred_dir / f"{case}.nii.gz",
        pred_dir / f"{case}_pred.nii.gz",
        pred_dir / f"{case}_0000.nii.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing prediction for case {case}: tried {candidates}")


def normalize_slice(image: Array) -> Array:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, (0.5, 99.5))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, lo, hi)
    return (image - lo) / (hi - lo)


def dice2d(pred: Array, gt: Array) -> tuple[float, int]:
    intersection = int(np.logical_and(pred, gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0, intersection
    return 2.0 * float(intersection) / float(pred_sum + gt_sum), intersection


def bbox_from_masks(masks: list[Array], margin: int, shape: tuple[int, int]) -> tuple[slice, slice] | None:
    coords = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if ys.size:
            coords.append(np.stack([ys, xs], axis=1))
    if not coords:
        return None

    merged = np.concatenate(coords, axis=0)
    y0, x0 = merged.min(axis=0)
    y1, x1 = merged.max(axis=0) + 1
    h, w = shape
    return (
        slice(max(0, int(y0) - margin), min(h, int(y1) + margin)),
        slice(max(0, int(x0) - margin), min(w, int(x1) + margin)),
    )


def overlay_contours(ax: plt.Axes, image: Array, contours: list[tuple[Array, str]]) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for mask, color in contours:
        if np.any(mask):
            ax.contour(
                mask.astype(np.uint8),
                levels=[0.5],
                colors=[color],
                linewidths=CONTOUR_WIDTH,
            )
    ax.axis("off")


def rank_slices(case: str, settings: Settings) -> tuple[list[SliceRow], Array, Array, Array]:
    image = read_nii(find_image(settings.dataset_root, case))
    gt = read_nii(find_label(settings.dataset_root, case)) > 0
    pred = read_nii(find_prediction(settings.pred_dir, case)) > 0
    if image.shape != gt.shape:
        raise ValueError(f"Image and GT shape mismatch for {case}: image={image.shape}, gt={gt.shape}")
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction and GT shape mismatch for {case}: pred={pred.shape}, gt={gt.shape}")

    rows = []
    for slice_index in range(gt.shape[0]):
        gt_slice = gt[slice_index]
        pred_slice = pred[slice_index]
        dice, intersection = dice2d(pred_slice, gt_slice)
        rows.append(
            SliceRow(
                case=case,
                slice_index=slice_index,
                dice=dice,
                gt_pixels=int(gt_slice.sum()),
                pred_pixels=int(pred_slice.sum()),
                intersection_pixels=intersection,
            )
        )
    rows.sort(key=lambda row: (row.dice, -row.gt_pixels, row.slice_index))
    return rows, image, gt, pred


def plot_triplet(case: str, row: SliceRow, image: Array, gt: Array, pred: Array, output_dir: Path) -> Path:
    slice_index = row.slice_index
    image_slice = normalize_slice(image[slice_index])
    gt_slice = gt[slice_index]
    pred_slice = pred[slice_index]
    panels: list[tuple[str, list[tuple[Array, str]]]] = [
        ("Image", []),
        ("GT", [(gt_slice, GT_COLOR)]),
        ("Ours", [(gt_slice, GT_COLOR), (pred_slice, PRED_COLOR)]),
    ]

    if CROP_ENABLED:
        bbox = bbox_from_masks([gt_slice, pred_slice], CROP_MARGIN, image_slice.shape)
        if bbox is not None:
            ys, xs = bbox
            image_slice = image_slice[ys, xs]
            panels = [
                (name, [(mask[ys, xs], color) for mask, color in contours])
                for name, contours in panels
            ]

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(PANEL_WIDTH * len(panels), PANEL_HEIGHT),
        dpi=DPI,
        facecolor="white",
    )
    for ax, (_name, contours) in zip(axes, panels):
        overlay_contours(ax, image_slice, contours)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.04)
    out_dir = output_dir / "visual"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case}_slice{slice_index:02d}_dice{row.dice:.4f}.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="Rank per-slice Dice and plot Image/GT/Ours bad-case panels.")
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--pred_dir", type=Path, default=PRED_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--cases", nargs="+", default=CASE_IDS)
    args = parser.parse_args()
    return Settings(
        case_ids=list(args.cases),
        dataset_root=args.dataset_root,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )


def main() -> None:
    settings = parse_args()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    top_rows = []
    visual_paths = []

    for case in tqdm(settings.case_ids, desc="Ranking bad-case slices", unit="case"):
        rows, image, gt, pred = rank_slices(case, settings)
        for rank, row in enumerate(rows, start=1):
            all_rows.append({**row.__dict__, "rank": rank})
        for rank, row in enumerate(rows[: settings.top_k], start=1):
            top_rows.append({**row.__dict__, "rank": rank})
            visual_paths.append(plot_triplet(case, row, image, gt, pred, settings.output_dir))

    all_df = pd.DataFrame(all_rows)
    top_df = pd.DataFrame(top_rows)
    for df in (all_df, top_df):
        if not df.empty:
            df["dice"] = df["dice"].round(4)

    all_csv = settings.output_dir / "ours_slice_dice_all.csv"
    top_csv = settings.output_dir / f"ours_badcase_top{settings.top_k}.csv"
    all_df.to_csv(all_csv, index=False)
    top_df.to_csv(top_csv, index=False)

    print("Top-5 lowest Dice slices per case:")
    for case in settings.case_ids:
        case_df = top_df[top_df["case"] == case]
        items = [
            f"slice {int(row.slice_index)} Dice={row.dice:.4f} GT={int(row.gt_pixels)} Pred={int(row.pred_pixels)}"
            for row in case_df.itertuples(index=False)
        ]
        print(f"{case}: " + "; ".join(items))

    print(f"Wrote {all_csv}")
    print(f"Wrote {top_csv}")
    print(f"Wrote {len(visual_paths)} visualizations to {settings.output_dir / 'visual'}")


if __name__ == "__main__":
    main()
