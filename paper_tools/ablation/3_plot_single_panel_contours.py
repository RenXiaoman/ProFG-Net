#!/usr/bin/env python3
"""Save one PNG per panel (Image, GT, each model) with contour overlays.

Companion to 2_plot_multirow_comparison.py: instead of one merged multi-row
figure, this writes a directory per case/slice containing one file per panel.
Overlays are contours, GT in red and prediction in yellow, matching the
comparative outputs.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


Array = np.ndarray

GT_COLOR = "#E60000"
PRED_COLOR = "#FFD400"
CONTOUR_WIDTH = 1.5
DPI = 300
PANEL_SIZE = 4.0
BACKGROUND = "white"


def read_nii(path: Path) -> Array:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def format_path(template: str, case: str, slice_index: Any) -> Path:
    return Path(template.format(case=case, slice_index=slice_index))


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
    return int(np.argmax((mask > 0).sum(axis=(1, 2))))


def validate_slice_index(slice_index: int, depth: int) -> int:
    if slice_index < 0:
        slice_index += depth
    if not 0 <= slice_index < depth:
        raise IndexError(f"slice_index={slice_index} is outside volume depth {depth}")
    return slice_index


def panel_name(model_name: str) -> str:
    """Drop the leading (a)/(b) ordering prefix for the file name."""
    return re.sub(r"^\([A-Za-z]\)\s*", "", model_name).strip()


def safe_filename(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def overlay_contours(ax: plt.Axes, image: Array, contours: list[tuple[Array, str]]) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for mask, color in contours:
        if not np.any(mask):
            continue
        ax.contour(mask.astype(np.uint8), levels=[0.5], colors=[color], linewidths=CONTOUR_WIDTH)
    ax.axis("off")


def save_panel(path: Path, image_slice: Array, contours: list[tuple[Array, str]]) -> None:
    fig, ax = plt.subplots(figsize=(PANEL_SIZE, PANEL_SIZE), dpi=DPI, facecolor=BACKGROUND)
    overlay_contours(ax, image_slice, contours)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor=BACKGROUND)
    plt.close(fig)


def case_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    if "cases" not in config:
        return [config]
    cases = []
    for case_cfg in config["cases"]:
        merged = dict(config)
        merged.update(case_cfg)
        merged.pop("cases", None)
        cases.append(merged)
    return cases


def render_case(config: dict[str, Any], output_root: Path) -> Path:
    case = config["case"]
    gt = read_nii(format_path(config["gt"], case, config.get("slice_index", "auto")))

    slice_spec = config.get("slice_index", "auto")
    slice_index = choose_slice(gt) if str(slice_spec).lower() == "auto" else validate_slice_index(
        int(slice_spec), gt.shape[0]
    )

    image = read_nii(format_path(config["image"], case, slice_index))
    if image.shape != gt.shape:
        raise ValueError(f"Image and GT shape mismatch for {case}: image={image.shape}, gt={gt.shape}")

    image_slice = normalize_slice(image[slice_index])
    gt_slice = gt[slice_index] > 0

    case_dir = output_root / f"{case}_slice{slice_index}"
    case_dir.mkdir(parents=True, exist_ok=True)

    save_panel(case_dir / "00_Image.png", image_slice, [])
    save_panel(case_dir / "01_GT.png", image_slice, [(gt_slice, GT_COLOR)])

    for offset, model in enumerate(config.get("models", []), start=2):
        pred = read_nii(format_path(model["pred"], case, slice_index))
        if pred.shape != gt.shape:
            raise ValueError(
                f"Prediction shape mismatch for {model['name']}: pred={pred.shape}, gt={gt.shape}"
            )
        pred_slice = pred[slice_index] > 0
        name = safe_filename(panel_name(model["name"]))
        save_panel(
            case_dir / f"{offset:02d}_{name}.png",
            image_slice,
            [(gt_slice, GT_COLOR), (pred_slice, PRED_COLOR)],
        )
    print(f"{case_dir}: {2 + len(config.get('models', []))} panels")
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Save one contour panel PNG per case and method.")
    parser.add_argument("--config", default="paper_tools/ablation/config_multirow_PICAI.json")
    parser.add_argument("--output-root", type=Path, default=Path("paper_tools/ablation/outputs/visual"))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    outputs = [render_case(case_cfg, args.output_root) for case_cfg in case_configs(config)]
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
