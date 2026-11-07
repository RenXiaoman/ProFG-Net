#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


# Manually edit these four values for each figure.
# EXPERIMENT_FILE = Path("paper_tools/comparative/experiments_AHCDU.txt")
# CASE = "WuZuoGui"


EXPERIMENT_FILE = Path("paper_tools/comparative/experiments_PICAI.txt")
CASE = "10975_1000994"
###############################################################
SLICE_INDEX: int | str = "11"
OUTPUT_PATH = Path("paper_tools/comparative/outputs/test/{case}_slice{slice_index}.png")

# Visual settings. Usually these do not need to change.
GT_COLOR = "#E60000"
PRED_COLOR = "#FFD400"
CONTOUR_WIDTH = 1.5
DPI = 300
PANEL_WIDTH = 1.7
PANEL_HEIGHT = 1.7
SHOW_TITLES = True
TITLE_SIZE = 8
BACKGROUND = "white"
TITLE_COLOR = "black"
CROP_ENABLED = False
CROP_MARGIN = 24


Array = np.ndarray


@dataclass(frozen=True)
class Experiment:
    name: str
    pred_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    gt_root: Path
    experiments: list[Experiment]

    @property
    def images_ts(self) -> Path:
        return self.gt_root / "imagesTs"

    @property
    def labels_ts(self) -> Path:
        return self.gt_root / "labelsTs"


def read_nii(path: Path) -> Array:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def parse_experiment_file(path: Path) -> ExperimentConfig:
    experiments: list[Experiment] = []
    gt_root: Path | None = None

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid line {line_no} in {path}: {raw_line!r}")

        name, value = parts
        if name == "gt":
            gt_root = Path(value)
        else:
            experiments.append(Experiment(name=name, pred_dir=Path(value)))

    if gt_root is None:
        raise ValueError(f"No gt line found in {path}")
    if not experiments:
        raise ValueError(f"No experiment entries found in {path}")
    return ExperimentConfig(gt_root=gt_root, experiments=experiments)


def normalize_slice(image: Array) -> Array:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, (0.5, 99.5))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, lo, hi)
    return (image - lo) / (hi - lo)


def choose_slice(mask: Array) -> int:
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")
    areas = (mask > 0).sum(axis=(1, 2))
    return int(np.argmax(areas))


def validate_slice_index(slice_index: int, depth: int) -> int:
    if slice_index < 0:
        slice_index += depth
    if not 0 <= slice_index < depth:
        raise IndexError(f"slice_index={slice_index} is outside volume depth {depth}")
    return slice_index


def find_image(images_ts: Path, case: str) -> Path:
    candidates = [
        images_ts / f"{case}_0000.nii.gz",
        images_ts / f"{case}.nii.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing image for case {case}: tried {candidates}")


def find_label(labels_ts: Path, case: str) -> Path:
    label_file = labels_ts / f"{case}.nii.gz"
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


def bbox_from_masks(masks: Iterable[Array], margin: int, shape: tuple[int, int]) -> tuple[slice, slice] | None:
    coords: list[Array] = []
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


def safe_filename(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def overlay_contours(
    ax: plt.Axes,
    image: Array,
    contours: list[tuple[Array, str]],
) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for mask, color in contours:
        if not np.any(mask):
            continue
        ax.contour(
            mask.astype(np.uint8),
            levels=[0.5],
            colors=[color],
            linewidths=CONTOUR_WIDTH,
        )
    ax.axis("off")


def save_cut_panels(
    panels: list[tuple[str, list[tuple[Array, str]]]],
    image_slice: Array,
    output_path: Path,
    show_titles: bool,
) -> Path:
    output_dir = output_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, contours) in enumerate(panels):
        fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=DPI, facecolor=BACKGROUND)
        overlay_contours(ax, image_slice, contours)
        if show_titles:
            ax.set_title(name, fontsize=TITLE_SIZE, color=TITLE_COLOR, pad=4)
        fig.subplots_adjust(left=0, right=1, top=0.94 if show_titles else 1, bottom=0)
        panel_path = output_dir / f"{index:02d}_{safe_filename(name)}.png"
        fig.savefig(panel_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor=BACKGROUND)
        plt.close(fig)
    return output_dir


def plot_comparison(
    experiment_file: Path,
    case: str,
    requested_slice: int | str,
    output_template: Path,
    cut: bool,
    show_titles: bool,
) -> Path:
    config = parse_experiment_file(experiment_file)

    gt = read_nii(find_label(config.labels_ts, case))
    if str(requested_slice).lower() == "auto":
        slice_index = choose_slice(gt)
    else:
        slice_index = validate_slice_index(int(requested_slice), gt.shape[0])

    image = read_nii(find_image(config.images_ts, case))
    if image.shape != gt.shape:
        raise ValueError(f"Image and GT shape mismatch: image={image.shape}, gt={gt.shape}")

    image_slice = normalize_slice(image[slice_index])
    gt_slice = gt[slice_index] > 0

    panels: list[tuple[str, list[tuple[Array, str]]]] = [
        ("Image", []),
        ("GT", [(gt_slice, GT_COLOR)]),
    ]
    for experiment in config.experiments:
        pred = read_nii(find_prediction(experiment.pred_dir, case))
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction shape mismatch for {experiment.name}: pred={pred.shape}, gt={gt.shape}")
        pred_slice = pred[slice_index] > 0
        panels.append(
            (experiment.name, [(gt_slice, GT_COLOR), (pred_slice, PRED_COLOR)])
        )

    if CROP_ENABLED:
        bbox = bbox_from_masks(
            [mask for _, contours in panels for mask, _ in contours],
            CROP_MARGIN,
            image_slice.shape,
        )
        if bbox is not None:
            ys, xs = bbox
            image_slice = image_slice[ys, xs]
            panels = [
                (name, [(mask[ys, xs], color) for mask, color in contours])
                for name, contours in panels
            ]

    output_path = Path(str(output_template).format(case=case, slice_index=slice_index))
    if cut:
        return save_cut_panels(panels, image_slice, output_path, show_titles)

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(PANEL_WIDTH * len(panels), PANEL_HEIGHT),
        dpi=DPI,
        facecolor=BACKGROUND,
    )

    for ax, (name, contours) in zip(axes, panels):
        overlay_contours(ax, image_slice, contours)
        if show_titles:
            ax.set_title(name, fontsize=TITLE_SIZE, color=TITLE_COLOR, pad=4)

    plt.subplots_adjust(left=0, right=1, top=0.90 if show_titles else 1, bottom=0, wspace=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2D MRI segmentation comparisons.")
    parser.add_argument("--experiment_file", type=Path, default=EXPERIMENT_FILE)
    parser.add_argument("--case", default=CASE)
    parser.add_argument("--slice", dest="slice_index", default=str(SLICE_INDEX))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--cut",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save Image, GT, and each method as separate high-resolution PNG files.",
    )
    parser.add_argument(
        "--show_titles",
        action=argparse.BooleanOptionalAction,
        default=SHOW_TITLES,
        help="Show titles above panels. Use --no-show_titles for clean standalone panels.",
    )
    args = parser.parse_args()
    output = plot_comparison(
        experiment_file=args.experiment_file,
        case=args.case,
        requested_slice=args.slice_index,
        output_template=args.output,
        cut=args.cut,
        show_titles=args.show_titles,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
