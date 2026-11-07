#!/usr/bin/env python3
"""Measure Params and THOP FLOPs for learnable-wavelet FreGNet variants."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from thop import profile

from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.DIY_boundary_msag_learnable import (
    LearnableDIYBoundaryCMSAG,
)
from nnunetv2.training.nnUNetTrainer.FreGNetTrain.DIYBoundaryAdaptiveCMSAGAuxTrainer import (
    replace_cmsag_w_x_with_adaptive_block,
)


WAVELETS = ("db2", "db3", "sym2", "sym3", "coif1", "coif2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_tools/learnable_wavelet_complexity.csv"),
    )
    return parser.parse_args()


def build_model(wavelet: str, device: torch.device) -> LearnableDIYBoundaryCMSAG:
    model = LearnableDIYBoundaryCMSAG(
        in_channels=1,
        out_channels=2,
        spatial_dims=3,
        feature_size=16,
        norm_name="instance",
        dropout_rate=0.0,
        depth=4,
        boundary_channels=1,
        wavelet=wavelet,
        mode="zero",
    )
    replaced = replace_cmsag_w_x_with_adaptive_block(model)
    if replaced != 4:
        raise RuntimeError(f"Expected 4 adaptive CMSAG gates, replaced {replaced}")
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    x = torch.randn(1, 1, 16, 256, 256, device=device)
    rows = []

    for wavelet in WAVELETS:
        model = build_model(wavelet, device)
        params = sum(parameter.numel() for parameter in model.parameters())
        try:
            with torch.no_grad():
                flops, thop_params = profile(model, inputs=(x,), verbose=False)
        except Exception as error:
            row = {
                "wavelet": wavelet,
                "params": params,
                "params_m": params / 1e6,
                "thop_params": "",
                "flops": "",
                "flops_g": "",
                "status": str(error),
            }
            rows.append(row)
            print(f"{wavelet:5s} unavailable: {error}")
            del model
            continue

        row = {
            "wavelet": wavelet,
            "params": params,
            "params_m": params / 1e6,
            "thop_params": int(thop_params),
            "flops": int(flops),
            "flops_g": flops / 1e9,
            "status": "ok",
        }
        rows.append(row)
        print(f"{wavelet:5s} Params={row['params_m']:.4f}M FLOPs={row['flops_g']:.4f}G")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
