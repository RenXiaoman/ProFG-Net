#!/usr/bin/env python3
"""Visualize the eight DWT sub-bands of the first WFE block (enhance[0]).

Uses the latest model (AdaptiveCMSAG). Cases are picked automatically from the
per-case metrics of the matching inference run, and the displayed slice is the
middle of the ground-truth prostate in the depth direction. Only cases whose
central slice reaches a minimum slice Dice are kept.

Each case gets its own directory with eight band PNGs plus the T2W slice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.training.nnUNetTrainer.FreGNetTrain.DIYBoundaryAdaptiveCMSAGAuxTrainer import (
    DIYBoundaryAdaptiveCMSAGAuxTrainer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = PROJECT_ROOT / "nnUNet/nnUNet_preprocessed/Dataset141_FullPICAI/nnUNetPlans_3d_fullres"
OUTPUT_DIR = PROJECT_ROOT / "paper_tools/frequncy"

# Two wavelet variants of the same model family.
#   fixed     : ptwt fixed wavelet inside wav_Enhance (ConvDWT)
#   learnable : LWTformer-style shared learnable filter bank
ARCHS = {
    "fixed": {
        "checkpoint": PROJECT_ROOT
        / "nnUNet/nnUNet_results/Dataset141_FullPICAI/"
        "DIYBoundaryAdaptiveCMSAGAuxTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
        "infer_dir": PROJECT_ROOT / "nnUNet/infer/Dataset141_DIYBoundaryAdaptiveCMSAGAuxTrainer",
    },
    "learnable": {
        "checkpoint": PROJECT_ROOT
        / "nnUNet/nnUNet_results/Dataset141_FullPICAI/"
        "LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
        "infer_dir": PROJECT_ROOT / "nnUNet/infer/Dataset141_LearnableAdaptiveCMSAG",
    },
}

# Order used by ConvDWT: low-frequency LLL first, then the seven detail bands.
BANDS = ("LLL", "aad", "ada", "add", "daa", "dad", "dda", "ddd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=None, help="Explicit case list; skips auto selection.")
    parser.add_argument(
        "--arch",
        choices=sorted(ARCHS),
        default="fixed",
        help="fixed = ptwt wavelet, learnable = learnable filter bank.",
    )
    parser.add_argument(
        "--input",
        choices=("stem", "image"),
        default="stem",
        help="stem = decompose the first encoder's feature map (original behaviour); "
        "image = decompose the preprocessed T2W volume itself.",
    )
    parser.add_argument("--num-cases", type=int, default=5, help="How many cases to select.")
    parser.add_argument("--min-case-dice", type=float, default=0.85, help="Minimum per-case Dice.")
    parser.add_argument(
        "--min-slice-dice",
        type=float,
        default=0.87,
        help="Minimum Dice of the displayed central slice.",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--pred-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--color-limit",
        type=float,
        default=5.0,
        help="Fixed symmetric display range [-A, A]; use 0 for matplotlib autoscale.",
    )
    parser.add_argument("--cmap", default="jet", help="Colormap for the band panels, e.g. jet or gray.")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = ARCHS[args.arch]["checkpoint"]
    if args.pred_dir is None:
        args.pred_dir = ARCHS[args.arch]["infer_dir"]
    if args.metrics is None:
        args.metrics = Path(args.pred_dir) / "1A.json"
    return args


def central_slice(gt: np.ndarray) -> int:
    """Middle slice of the prostate along the depth axis."""
    present = np.where((gt > 0).any(axis=(1, 2)))[0]
    if present.size == 0:
        raise ValueError("Ground truth is empty for this case")
    return int(round((int(present.min()) + int(present.max())) / 2))


def slice_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_mask = pred > 0
    gt_mask = gt > 0
    total = int(pred_mask.sum()) + int(gt_mask.sum())
    if total == 0:
        return 1.0
    return 2.0 * float(np.logical_and(pred_mask, gt_mask).sum()) / total


def select_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    """Rank cases by the Dice of the central slice and keep the best ones."""
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    candidates = [
        case
        for case in payload["cases"]
        if case.get("dice") is not None and case["dice"] > args.min_case_dice
    ]
    if not candidates:
        raise ValueError(f"No case with Dice > {args.min_case_dice} in {args.metrics}")

    dataset = nnUNetDatasetBlosc2(str(PREPROCESSED_DIR), identifiers=[c["case"] for c in candidates])
    scored: list[dict[str, object]] = []
    for entry in candidates:
        case = entry["case"]
        pred_file = args.pred_dir / f"{case}.nii.gz"
        if not pred_file.is_file():
            continue
        _, seg, _, _ = dataset.load_case(case)
        gt = np.asarray(seg[:])[0] > 0
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_file))) > 0
        if pred.shape != gt.shape:
            continue
        slice_index = central_slice(gt)
        scored.append(
            {
                "case": case,
                "case_dice": float(entry["dice"]),
                "slice_index": slice_index,
                "slice_dice": slice_dice(pred[slice_index], gt[slice_index]),
            }
        )

    kept = [row for row in scored if float(row["slice_dice"]) >= args.min_slice_dice]
    print(
        f"Scanned {len(scored)} cases; {len(kept)} reach slice Dice >= {args.min_slice_dice} "
        f"at their central slice"
    )
    kept.sort(key=lambda row: (-float(row["slice_dice"]), str(row["case"])))
    return kept[: args.num_cases]


def load_model(arch: str, checkpoint: Path, device: torch.device):
    if arch == "learnable":
        from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer import (
            LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer,
        )

        model = LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer.build_network_architecture(None, None, 1, 2, False)
    else:
        model = DIYBoundaryAdaptiveCMSAGAuxTrainer.build_network_architecture(None, None, 1, 2, False)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(state["network_weights"], strict=True)
    return model.to(device).eval()


def get_dwt_module(model, arch: str):
    """First WFE's DWT. The learnable bank is shared, so it lives on the backbone."""
    if arch == "learnable":
        return model.backbone.wavelet_bank.dwt
    return model.backbone.enhance[0].dwt


def upsample(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Display the half-resolution band maps at the original in-plane size."""
    if values.shape == size:
        return values
    tensor = torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0).unsqueeze(0).float()
    resized = torch.nn.functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def save_panel(path: Path, values: np.ndarray, cmap: str = "jet", color_limit: float | None = None) -> None:
    """Render values with an optional fixed range; values are never normalized or changed."""
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=300, facecolor="white")
    kwargs = {"cmap": cmap, "interpolation": "nearest"}
    if color_limit is not None:
        kwargs.update(vmin=-float(color_limit), vmax=float(color_limit))
    ax.imshow(values, **kwargs)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def render_case(
    model,
    arch: str,
    dataset,
    case: str,
    slice_index: int,
    output_dir: Path,
    input_mode: str,
    color_limit: float | None,
    cmap: str,
) -> Path:
    data, seg, _, _ = dataset.load_case(case)
    image = np.asarray(data[:], dtype=np.float32)
    gt = np.asarray(seg[:])[0] == 1

    device = next(model.parameters()).device
    if input_mode == "image":
        # Decompose the original T2W volume (1 channel) instead of features.
        source = torch.from_numpy(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    else:
        # Capture the stem output, then run the first WFE's DWT on it explicitly.
        # This works for both variants: the learnable bank is shared between blocks,
        # so hooking it directly would also catch the deeper WFE calls.
        captured: dict[str, torch.Tensor] = {}
        handle = model.backbone.stem.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("stem", output.detach())
        )
        try:
            with torch.no_grad():
                model(
                    torch.from_numpy(image)
                    .unsqueeze(0)
                    .to(device=device, dtype=torch.float32)
                )
        finally:
            handle.remove()
        source = captured["stem"].float()

    with torch.no_grad():
        dwt_out = get_dwt_module(model, arch)(source)
    channels = int(source.shape[1])
    if dwt_out.shape[1] != 8 * channels:
        raise ValueError(f"Expected 8*C channels from DWT, got {dwt_out.shape[1]} (C={channels})")

    band_slice = min(slice_index // 2, dwt_out.shape[2] - 1)
    display_shape = (int(image.shape[2]), int(image.shape[3]))
    case_dir = output_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"DWT output {tuple(dwt_out.shape)} -> band slice {band_slice} (original slice {slice_index} // 2)")
    print(f"{'band':6s} {'mean':>10s} {'max|.|':>10s}   file")
    for index, name in enumerate(BANDS):
        sub = dwt_out[:, index * channels : (index + 1) * channels]
        if input_mode == "image":
            # C == 1 for the original image: keep the signed coefficients.
            signed = sub[0, 0].float().cpu().numpy()
            values = upsample(signed[band_slice], display_shape)
            stat_mean, stat_max = float(signed.mean()), float(np.abs(signed).max())
        else:
            magnitude = torch.sqrt((sub.float() ** 2).mean(dim=1))[0].cpu().numpy()
            values = upsample(magnitude[band_slice], display_shape)
            stat_mean, stat_max = float(magnitude.mean()), float(magnitude.max())
        output = case_dir / f"{index:02d}_{name}.png"
        # Same rendering as the feature-based panels: jet + matplotlib autoscale,
        # no explicit vmin/vmax.
        save_panel(output, values, cmap=cmap, color_limit=color_limit)
        print(f"{name:6s} {stat_mean:10.4f} {stat_max:10.4f}   {output}")

    image_panel = case_dir / f"{len(BANDS):02d}_image.png"
    save_panel(image_panel, image[0, slice_index], cmap="gray")
    print(f"image  (T2W, original slice {slice_index})              {image_panel}")

    print(f"Wrote {len(BANDS) + 1} panels to {case_dir}\n")
    return case_dir


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.cases:
        dataset_all = nnUNetDatasetBlosc2(str(PREPROCESSED_DIR), identifiers=list(args.cases))
        selected = []
        for case in args.cases:
            _, seg, _, _ = dataset_all.load_case(case)
            selected.append(
                {
                    "case": case,
                    "case_dice": float("nan"),
                    "slice_index": central_slice(np.asarray(seg[:])[0] > 0),
                    "slice_dice": float("nan"),
                }
            )
    else:
        selected = select_cases(args)

    print("Selected cases:")
    print(f"{'case':16s} {'case Dice':>10s} {'slice':>6s} {'slice Dice':>11s}")
    for row in selected:
        print(
            f"{row['case']:16s} {row['case_dice']:10.4f} {row['slice_index']:6d} "
            f"{row['slice_dice']:11.4f}"
        )
    print()

    dataset = nnUNetDatasetBlosc2(str(PREPROCESSED_DIR), identifiers=[str(r["case"]) for r in selected])
    if args.color_limit < 0:
        raise ValueError("--color-limit must be >= 0 (0 = autoscale)")
    color_limit = args.color_limit if args.color_limit > 0 else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "display_scale.json").write_text(
        json.dumps(
            {
                "cmap": args.cmap,
                "vmin": None if color_limit is None else -color_limit,
                "vmax": None if color_limit is None else color_limit,
                "raw_coefficients_saved": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if color_limit is None:
        print(f"display: cmap={args.cmap}, matplotlib autoscale (raw coefficients unchanged)")
    else:
        print(
            f"display: cmap={args.cmap}, fixed range [{-color_limit:.6g}, {color_limit:.6g}] "
            "(raw coefficients unchanged)"
        )
    model = load_model(args.arch, args.checkpoint, device)
    for row in selected:
        render_case(
            model,
            args.arch,
            dataset,
            str(row["case"]),
            int(row["slice_index"]),
            args.output_dir,
            args.input,
            color_limit,
            args.cmap,
        )


if __name__ == "__main__":
    main()
