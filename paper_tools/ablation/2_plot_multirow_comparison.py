#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.gridspec import GridSpec


Array = np.ndarray


def read_nii(path: Path) -> Array:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def format_path(template: str, case: str, slice_index: Any, num_cases: int | None = None) -> Path:
    values = {"case": case, "slice_index": slice_index}
    if num_cases is not None:
        values["num_cases"] = num_cases
    return Path(template.format(**values))


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


def overlay_mask(ax: plt.Axes, image: Array, mask: Array | None, color: list[float], alpha: float) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if mask is not None:
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[..., :3] = color[:3]
        rgba[..., 3] = (mask > 0).astype(np.float32) * alpha
        ax.imshow(rgba, interpolation="nearest")
    ax.axis("off")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def build_panels(config: dict[str, Any]) -> tuple[list[str], Array, list[Array | None]]:
    case = config["case"]
    gt = read_nii(format_path(config["gt"], case, config.get("slice_index", "auto")))

    slice_spec = config.get("slice_index", "auto")
    if str(slice_spec).lower() == "auto":
        slice_index = choose_slice(gt)
    else:
        slice_index = validate_slice_index(int(slice_spec), gt.shape[0])

    image = read_nii(format_path(config["image"], case, slice_index))
    if image.shape != gt.shape:
        raise ValueError(f"Image and GT shape mismatch for {case}: image={image.shape}, gt={gt.shape}")

    image_slice = normalize_slice(image[slice_index])
    names = ["T2WI", "GT"]
    masks: list[Array | None] = [None, gt[slice_index] > 0]

    for model in config.get("models", []):
        pred = read_nii(format_path(model["pred"], case, slice_index))
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction shape mismatch for {case}/{model['name']}: pred={pred.shape}, gt={gt.shape}")
        names.append(model["name"])
        masks.append(pred[slice_index] > 0)

    return names, image_slice, masks


def plot_multirow(config: dict[str, Any], output_override: str | None) -> Path:
    rows = case_configs(config)
    if not rows:
        raise ValueError("No cases configured")

    panel_rows = [build_panels(row_cfg) for row_cfg in rows]
    column_names = panel_rows[0][0]
    for names, _, _ in panel_rows[1:]:
        if names != column_names:
            raise ValueError("All rows must have the same panel names")

    figure_cfg = config.get("figure", {})
    dpi = int(figure_cfg.get("dpi", 300))
    panel_width = float(figure_cfg.get("panel_width", 1.7))
    panel_height = float(figure_cfg.get("panel_height", 1.7))
    label_width = float(figure_cfg.get("case_label_width", 0.55))
    show_titles = bool(figure_cfg.get("show_titles", True))
    title_size = int(figure_cfg.get("title_size", 8))
    case_label_size = int(figure_cfg.get("case_label_size", 12))
    case_label_x = float(figure_cfg.get("case_label_x", 0.5))
    background = figure_cfg.get("background", "white")
    title_color = figure_cfg.get("title_color", "black")

    mask_cfg = config.get("mask", {})
    color = mask_cfg.get("color", [0.9, 0.05, 0.03])
    alpha = float(mask_cfg.get("alpha", 0.55))

    nrows = len(panel_rows)
    ncols = len(column_names)
    fig = plt.figure(
        figsize=(label_width + panel_width * ncols, panel_height * nrows),
        dpi=dpi,
        facecolor=background,
    )
    grid = GridSpec(nrows, ncols + 1, figure=fig, width_ratios=[label_width] + [panel_width] * ncols)

    for row_idx, (_, image_slice, masks) in enumerate(panel_rows):
        label_ax = fig.add_subplot(grid[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            case_label_x,
            0.5,
            f"Case {row_idx + 1}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=case_label_size,
            color=title_color,
        )

        for col_idx, (name, mask) in enumerate(zip(column_names, masks), start=1):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            overlay_mask(ax, image_slice, mask, color, alpha)
            if show_titles and row_idx == nrows - 1:
                ax.text(
                    0.5,
                    -0.08,
                    name,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=title_size,
                    color=title_color,
                    clip_on=False,
                )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.10 if show_titles else 0, wspace=0.04, hspace=0.04)

    output_template = output_override or config.get(
        "multirow_output",
        "paper_tools/ablation/outputs/visual/{num_cases}_case_ablation_comparison.png",
    )
    output_path = format_path(output_template, rows[0]["case"], rows[0].get("slice_index", "auto"), num_cases=nrows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw multi-row ablation comparison panels.")
    parser.add_argument("--config", default="paper_tools/ablation/config_multirow_PICAI.json")
    parser.add_argument("--output", default=None, help="Override output path template.")
    args = parser.parse_args()

    output = plot_multirow(load_config(Path(args.config)), args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
