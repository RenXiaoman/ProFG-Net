#!/usr/bin/env python3
"""Render the new FreGNet (Adaptive CMSAG) panel for the comparison case folders.

This reuses the plotting helpers of 2_plot_comparison.py and writes exactly one
new file per case folder, 11_Ours_new.png. Existing panels (including the old
11_Ours.png) are left untouched.

Prediction sources:
  AHCDU cases  -> nnUNet/infer/Dataset130_DIYBoundaryAdaptiveCMSAGAuxTrainer
  PICAI cases  -> nnUNet/infer/Dataset141_DIYBoundaryAdaptiveCMSAGAuxTrainer
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_plot_module():
    path = Path(__file__).with_name("2_plot_comparison.py")
    spec = importlib.util.spec_from_file_location("plot_comparison", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLOT = _load_plot_module()

AHCDU_ROOT = Path("nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU")
PICAI_ROOT = Path("nnUNet/nnUNet_raw/Dataset131_ProstatePI-CAI")
AHCDU_PRED = Path("nnUNet/infer/Dataset130_DIYBoundaryAdaptiveCMSAGAuxTrainer")
PICAI_PRED = Path("nnUNet/infer/Dataset141_DIYBoundaryAdaptiveCMSAGAuxTrainer")
OUTPUT_ROOT = Path("paper_tools/comparative/outputs")


@dataclass(frozen=True)
class PanelJob:
    folder: str
    case: str
    slice_index: int
    gt_root: Path
    pred_dir: Path


JOBS = [
    PanelJob("LiangJiZhou_slice11", "LiangJiZhou", 11, AHCDU_ROOT, AHCDU_PRED),
    PanelJob("PICAI_10012_1000012_slice2", "10012_1000012", 2, PICAI_ROOT, PICAI_PRED),
    PanelJob("PICAI_10220_1000224_slice11", "10220_1000224", 11, PICAI_ROOT, PICAI_PRED),
    PanelJob("PICAI_10393_1000399_slice2", "10393_1000399", 2, PICAI_ROOT, PICAI_PRED),
    PanelJob("PICAI_10975_1000994_slice11", "10975_1000994", 11, PICAI_ROOT, PICAI_PRED),
    PanelJob("QinDengHua_slice7", "QinDengHua", 7, AHCDU_ROOT, AHCDU_PRED),
    PanelJob("TianGuiLong_slice15", "TianGuiLong", 15, AHCDU_ROOT, AHCDU_PRED),
    PanelJob("WangJinShu_slice13", "WangJinShu", 13, AHCDU_ROOT, AHCDU_PRED),
]


def build_panel(job: PanelJob, output_name: str, show_titles: bool) -> Path:
    gt = PLOT.read_nii(PLOT.find_label(job.gt_root / "labelsTs", job.case))
    image = PLOT.read_nii(PLOT.find_image(job.gt_root / "imagesTs", job.case))
    pred = PLOT.read_nii(PLOT.find_prediction(job.pred_dir, job.case))
    if image.shape != gt.shape or pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch for {job.case}: image={image.shape}, gt={gt.shape}, pred={pred.shape}"
        )

    slice_index = PLOT.validate_slice_index(job.slice_index, gt.shape[0])
    image_slice = PLOT.normalize_slice(image[slice_index])
    gt_slice = gt[slice_index] > 0
    pred_slice = pred[slice_index] > 0

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=PLOT.DPI, facecolor=PLOT.BACKGROUND)
    PLOT.overlay_contours(
        ax,
        image_slice,
        [(gt_slice, PLOT.GT_COLOR), (pred_slice, PLOT.PRED_COLOR)],
    )
    if show_titles:
        ax.set_title("Ours (new)", fontsize=PLOT.TITLE_SIZE, color=PLOT.TITLE_COLOR, pad=4)
    fig.subplots_adjust(left=0, right=1, top=0.94 if show_titles else 1, bottom=0)

    output_dir = OUTPUT_ROOT / job.folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    fig.savefig(
        output_path,
        dpi=PLOT.DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor=PLOT.BACKGROUND,
    )
    plt.close(fig)
    print(f"{job.folder}: gt={int(gt_slice.sum())} pred={int(pred_slice.sum())} -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_name", default="11_Ours_new.png")
    parser.add_argument(
        "--show_titles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="match the existing panels, which have no titles",
    )
    args = parser.parse_args()
    for job in JOBS:
        build_panel(job, args.output_name, args.show_titles)


if __name__ == "__main__":
    main()
