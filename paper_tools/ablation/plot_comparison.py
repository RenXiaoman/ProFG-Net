#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


Array = np.ndarray


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
    areas = (mask > 0).sum(axis=(1, 2))
    return int(np.argmax(areas))


def validate_slice_index(slice_index: int, depth: int) -> int:
    if slice_index < 0:
        slice_index += depth
    if not 0 <= slice_index < depth:
        raise IndexError(f"slice_index={slice_index} is outside volume depth {depth}")
    return slice_index


def bbox_from_masks(masks: Iterable[Array], margin: int, shape: Tuple[int, int]) -> Optional[Tuple[slice, slice]]:
    coords: List[Array] = []
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
    y0 = max(0, int(y0) - margin)
    x0 = max(0, int(x0) - margin)
    y1 = min(h, int(y1) + margin)
    x1 = min(w, int(x1) + margin)
    return slice(y0, y1), slice(x0, x1)


def overlay_mask(ax: plt.Axes, image: Array, mask: Optional[Array], color: List[float], alpha: float) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if mask is not None:
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[..., :3] = color[:3]
        rgba[..., 3] = (mask > 0).astype(np.float32) * alpha
        ax.imshow(rgba, interpolation="nearest")
    ax.axis("off")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_cases(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if "cases" in config:
        for case_cfg in config["cases"]:
            merged = dict(config)
            merged.update(case_cfg)
            merged.pop("cases", None)
            yield merged
    else:
        yield config


def plot_one(config: Dict[str, Any], cli_case: Optional[str], cli_slice: Optional[str], cli_output: Optional[str]) -> Path:
    case = cli_case or config["case"]
    gt_path = format_path(config["gt"], case, config.get("slice_index", "auto"))
    gt = read_nii(gt_path)

    slice_spec = cli_slice if cli_slice is not None else config.get("slice_index", "auto")
    if str(slice_spec).lower() == "auto":
        slice_index = choose_slice(gt)
    else:
        slice_index = validate_slice_index(int(slice_spec), gt.shape[0])

    image_path = format_path(config["image"], case, slice_index)
    image = read_nii(image_path)
    if image.shape != gt.shape:
        raise ValueError(f"Image and GT shape mismatch: image={image.shape}, gt={gt.shape}")

    image_slice = normalize_slice(image[slice_index])
    gt_slice = gt[slice_index] > 0

    panels: List[Tuple[str, Optional[Array]]] = [("T2WI", None), ("GT", gt_slice)]
    for model in config.get("models", []):
        pred_path = format_path(model["pred"], case, slice_index)
        pred = read_nii(pred_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction shape mismatch for {model['name']}: pred={pred.shape}, gt={gt.shape}")
        panels.append((model["name"], pred[slice_index] > 0))

    crop_cfg = config.get("crop", {})
    if crop_cfg.get("enabled", False):
        masks = [mask for _, mask in panels if mask is not None]
        bbox = bbox_from_masks(masks, int(crop_cfg.get("margin", 24)), image_slice.shape)
        if bbox is not None:
            ys, xs = bbox
            image_slice = image_slice[ys, xs]
            panels = [(name, None if mask is None else mask[ys, xs]) for name, mask in panels]

    figure_cfg = config.get("figure", {})
    dpi = int(figure_cfg.get("dpi", 300))
    panel_width = float(figure_cfg.get("panel_width", 2.0))
    panel_height = float(figure_cfg.get("panel_height", 2.0))
    show_titles = bool(figure_cfg.get("show_titles", True))
    title_size = int(figure_cfg.get("title_size", 10))
    title_position = str(figure_cfg.get("title_position", "top")).lower()
    background = figure_cfg.get("background", "black")
    title_color = figure_cfg.get("title_color", "white")

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(panel_width * len(panels), panel_height),
        dpi=dpi,
        facecolor=background,
    )
    if len(panels) == 1:
        axes = [axes]

    mask_cfg = config.get("mask", {})
    color = mask_cfg.get("color", [0.9, 0.05, 0.03])
    alpha = float(mask_cfg.get("alpha", 0.55))

    for ax, (name, mask) in zip(axes, panels):
        overlay_mask(ax, image_slice, mask, color, alpha)
        if show_titles:
            if title_position == "bottom":
                ax.text(
                    0.5,
                    -0.06,
                    name,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=title_size,
                    color=title_color,
                    clip_on=False,
                )
            else:
                ax.set_title(name, fontsize=title_size, color=title_color, pad=4)

    top = 1 if show_titles and title_position == "bottom" else 0.92 if show_titles else 1
    bottom = 0.10 if show_titles and title_position == "bottom" else 0
    plt.subplots_adjust(left=0, right=1, top=top, bottom=bottom, wspace=0.04)

    output_template = cli_output or config.get("output", "paper_tools/comparative/outputs/{case}_slice{slice_index}.png")
    output_path = format_path(output_template, case, slice_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw T2WI/GT/model prediction comparison panels.")
    parser.add_argument("--config", default="paper_tools/comparative/config.json", help="Path to JSON config.")
    parser.add_argument("--case", default=None, help="Override case name in config.")
    parser.add_argument("--slice", default=None, help="Override slice index. Use 'auto' for largest GT slice.")
    parser.add_argument("--output", default=None, help="Override output path template.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    outputs = [plot_one(case_cfg, args.case, args.slice, args.output) for case_cfg in iter_cases(config)]
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
