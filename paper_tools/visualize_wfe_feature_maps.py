#!/usr/bin/env python3
"""Visualize channel-RMS feature magnitude before and after the first WFE."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.DIY_boundary_msag import (
    DIYBoundaryCMSAG,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = (
    PROJECT_ROOT
    / "nnUNet/nnUNet_preprocessed/Dataset141_FullPICAI/nnUNetPlans_3d_fullres"
)
SPLITS_FILE = (
    PROJECT_ROOT / "nnUNet/nnUNet_preprocessed/Dataset141_FullPICAI/splits_final.json"
)
CHECKPOINT = (
    PROJECT_ROOT
    / "nnUNet/nnUNet_results/Dataset141_FullPICAI/"
    "DIYBoundaryCMSAGAuxTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "paper_tools/Dataset141_WFE_FeatureMaps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=2)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Normalize each displayed slice to [0, 1] using its 1st and 99th percentiles.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> DIYBoundaryCMSAG:
    model = DIYBoundaryCMSAG(
        in_channels=1,
        out_channels=2,
        spatial_dims=3,
        feature_size=16,
        norm_name="instance",
        dropout_rate=0.0,
        depth=4,
        boundary_channels=1,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["network_weights"], strict=True)
    model.eval()
    return model


def channel_rms(features: torch.Tensor) -> np.ndarray:
    magnitude = torch.sqrt(torch.mean(features.float().square(), dim=1))
    return magnitude[0].cpu().numpy().astype(np.float32, copy=False)


def resize_feature_map(feature_map: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize a D x H x W frequency map to the WFE feature resolution."""
    if feature_map.shape == target_shape:
        return feature_map
    tensor = torch.from_numpy(feature_map).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    return resized[0, 0].numpy().astype(np.float32, copy=False)


def extract_feature_magnitudes(
    model: DIYBoundaryCMSAG,
    image: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, ...]:
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output.detach()

        return hook

    def capture_input(name: str, position: int):
        def hook(_module, inputs, _output):
            captured[name] = inputs[position].detach()

        return hook

    handles = [
        model.backbone.stem.register_forward_hook(capture("before_wfe")),
        model.backbone.stem.layer.conv1.register_forward_hook(capture("stem_conv1")),
        model.backbone.enhance[0].register_forward_hook(capture("after_wfe")),
        model.backbone.enhance[0].dwt.register_forward_hook(capture("dwt_out")),
        # idwt(LL, Yh) is called with the attention-modulated high frequency.
        model.backbone.enhance[0].idwt.register_forward_hook(capture_input("modulated_high", 1)),
        model.backbone.decoder2.attention.register_forward_hook(capture("cmsag_output")),
        model.backbone.decoder2.conv_block.register_forward_hook(capture("decoder_output")),
    ]
    try:
        with torch.no_grad():
            model(image_tensor)
    finally:
        for handle in handles:
            handle.remove()

    enhancement = model.backbone.enhance[0]
    if enhancement.last_attended_low is None or enhancement.last_raw_high is None:
        raise RuntimeError("WFE intermediate tensors were not captured during forward.")

    before_wfe = channel_rms(captured["before_wfe"])
    # dwt_out = [B, 8C, D/2, H/2, W/2] or [B, 4C, 1, H/2, W/2]; the first C
    # channels are the raw low-frequency subband LL.
    channels = enhancement.in_dim
    raw_low = resize_feature_map(channel_rms(captured["dwt_out"][:, :channels]), before_wfe.shape)
    attended_low = resize_feature_map(channel_rms(enhancement.last_attended_low), before_wfe.shape)
    raw_high = resize_feature_map(channel_rms(enhancement.last_raw_high), before_wfe.shape)
    modulated_high = resize_feature_map(channel_rms(captured["modulated_high"]), before_wfe.shape)
    stem_conv1_feature = resize_feature_map(channel_rms(captured["stem_conv1"]), before_wfe.shape)
    after_wfe = channel_rms(captured["after_wfe"])
    cmsag_output = channel_rms(captured["cmsag_output"])
    decoder_output = channel_rms(captured["decoder_output"])
    return (
        before_wfe,
        raw_low,
        attended_low,
        raw_high,
        modulated_high,
        stem_conv1_feature,
        after_wfe,
        cmsag_output,
        decoder_output,
    )


def normalize_for_plot(image_slice: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image_slice, (1, 99))
    if hi <= lo:
        return np.zeros_like(image_slice, dtype=np.float32)
    return np.clip((image_slice - lo) / (hi - lo), 0, 1)


def save_case_figures(
    case_id: str,
    image: np.ndarray,
    gt: np.ndarray,
    before_wfe: np.ndarray,
    raw_low: np.ndarray,
    attended_low: np.ndarray,
    modulated_high: np.ndarray,
    stem_conv1_feature: np.ndarray,
    after_wfe: np.ndarray,
    cmsag_output: np.ndarray,
    decoder_output: np.ndarray,
    output_dir: Path,
    normalize: bool,
) -> None:
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    if any(
        feature_map.shape != image.shape[1:]
        for feature_map in (
            before_wfe,
            raw_low,
            attended_low,
            modulated_high,
            stem_conv1_feature,
            after_wfe,
            cmsag_output,
            decoder_output,
        )
    ):
        raise ValueError(
            f"Feature/image shape mismatch for {case_id}: image={image.shape}, "
            f"before={before_wfe.shape}, raw_low={raw_low.shape}, low={attended_low.shape}, "
            f"mod_high={modulated_high.shape}, stem_conv1={stem_conv1_feature.shape}, after={after_wfe.shape}, "
            f"cmsag={cmsag_output.shape}, decoder={decoder_output.shape}"
        )

    diff_wfe = after_wfe - before_wfe
    # "After + (After - Before)" = 2A - B, matching the panel title.
    after_plus_after_minus_before = after_wfe + (after_wfe - before_wfe)

    for slice_index in range(image.shape[1]):
        fig, axes = plt.subplots(1, 8, figsize=(28.0, 3.6), constrained_layout=True)
        image_slice = image[0, slice_index]
        axes[0].imshow(normalize_for_plot(image_slice) if normalize else image_slice, cmap="gray")
        axes[0].set_title("T2W", fontsize=11)
        def shown(values: np.ndarray) -> np.ndarray:
            return normalize_for_plot(values) if normalize else values

        feature_im = axes[1].imshow(shown(before_wfe[slice_index]), cmap="jet")
        axes[1].set_title("Feature Before WFE", fontsize=11)
        axes[2].imshow(shown(attended_low[slice_index]), cmap="jet")
        axes[2].set_title("Attended Low-Frequency", fontsize=11)
        axes[3].imshow(shown(raw_low[slice_index]), cmap="jet")
        axes[3].set_title("Modulated LF", fontsize=11)
        axes[4].imshow(shown(stem_conv1_feature[slice_index]), cmap="jet")
        axes[4].set_title("Stem Conv1 Feature", fontsize=11)
        axes[5].imshow(shown(after_wfe[slice_index]), cmap="jet")
        axes[5].set_title("Feature After WFE", fontsize=11)
        axes[6].imshow(shown(after_plus_after_minus_before[slice_index]), cmap="jet")
        axes[6].set_title("After + (After - Before)", fontsize=11)
        axes[7].imshow(shown(modulated_high[slice_index]), cmap="jet")
        axes[7].set_title("Modulated High-Frequency", fontsize=11)
        if np.any(gt[slice_index] > 0):
            axes[0].contour(
                gt[slice_index],
                levels=[0.5],
                colors=["magenta"],
                linewidths=1.5,
                alpha=0.5,
            )
            axes[7].contour(
                gt[slice_index],
                levels=[0.5],
                colors=["magenta"],
                linewidths=1.5,
                alpha=0.5,
            )
        for ax in axes:
            ax.axis("off")
        fig.colorbar(
            feature_im,
            ax=[axes[1], axes[2], axes[3], axes[4], axes[5], axes[6], axes[7]],
            fraction=0.035,
            pad=0.02,
            label="Normalized RMS activation" if normalize else "RMS activation",
        )
        fig.savefig(case_dir / f"slice_{slice_index:02d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if args.case_id is not None:
        selected_cases = args.case_id
    else:
        with SPLITS_FILE.open() as f:
            val_cases = json.load(f)[0]["val"]
        selected_cases = random.Random(args.seed).sample(val_cases, args.num_cases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": args.seed,
        "cases": selected_cases,
        "checkpoint": str(args.checkpoint.resolve()),
        "before_wfe_layer": "backbone.stem",
        "attended_low_frequency": "backbone.enhance.0.last_attended_low = LL * channel_attention(LL) * spatial_attention(channel_attention(LL) * LL)",
        "modulated_lf": "backbone.enhance.0.dwt output, first C channels = raw DWT low-frequency subband LL",
        "raw_high_frequency": "backbone.enhance.0.last_raw_high = concatenated seven high-frequency subbands before Gaussian enhancement",
        "modulated_high_frequency": "backbone.enhance.0.idwt input #1 = Yh after Yh * att.repeat_interleave(subbands)",
        "after_wfe_layer": "backbone.enhance.0",
        "cmsag_output_layer": "backbone.decoder2.attention",
        "decoder_output_layer": "backbone.decoder2.conv_block",
        "aggregation": "sqrt(mean(feature ** 2, axis=channel))",
        "normalization": "none" if not args.norm else "per-slice percentile normalization for display",
        "difference_map": "after_wfe - before_wfe",
        "after_plus_difference": "after_wfe + (after_wfe - before_wfe) = 2A - B",
        "plot_normalization": "per-slice percentile normalization to [0, 1]" if args.norm else "none; raw values passed to matplotlib",
    }
    with (args.output_dir / "selection.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    dataset = nnUNetDatasetBlosc2(str(PREPROCESSED_DIR), identifiers=selected_cases)
    model = load_model(args.checkpoint, device)
    for case_id in selected_cases:
        data, seg, _, _ = dataset.load_case(case_id)
        image = np.asarray(data[:], dtype=np.float32)
        gt = (np.asarray(seg[:])[0] == 1).astype(np.uint8)
        (
            before_wfe,
            raw_low,
            attended_low,
            raw_high,
            modulated_high,
            stem_conv1_feature,
            after_wfe,
            cmsag_output,
            decoder_output,
        ) = extract_feature_magnitudes(model, image, device)
        save_case_figures(
            case_id,
            image,
            gt,
            before_wfe,
            raw_low,
            attended_low,
            modulated_high,
            stem_conv1_feature,
            after_wfe,
            cmsag_output,
            decoder_output,
            args.output_dir,
            args.norm,
        )
        print(f"Saved {image.shape[1]} slices for {case_id}")

    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
