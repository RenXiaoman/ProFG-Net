#!/usr/bin/env python3
"""Rank cases where FreGNet has a clear per-case DSC advantage."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        array = np.load(path)
    else:
        array = np.asarray(nib.load(str(path)).get_fdata())
    # Cache predictions use D x H x W; NIfTI files are H x W x D.
    if array.ndim == 3 and array.shape[-1] < array.shape[0]:
        array = np.transpose(array, (2, 0, 1))
    return array > 0


def dsc(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = int(pred.sum()) + int(gt.sum())
    if denom == 0:
        return 1.0
    return 2.0 * float(np.logical_and(pred, gt).sum()) / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--our_predictions", type=Path, default=Path("/home/Space/clib/Projects/nnUNetv2/paper_tools/noise_quality/predictions/a6a0ed2f9ae0/FreGNet"))
    parser.add_argument("--other_predictions", type=Path, default=Path("/home/Space/clib/Projects/CAD/paper_tools/noise_quality/noise_robustness/predictions"))
    parser.add_argument("--labels_dir", type=Path, default=Path("/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw/Dataset141_FullPICAI/labelsTr"))
    parser.add_argument("--output_dir", type=Path, default=Path("/home/Space/clib/Projects/nnUNetv2/paper_tools/noise_quality/compatre"))
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditions = ("original", "combined_1", "combined_2", "combined_3")
    our_condition_dirs = {"original": "original", "combined_1": "combined1", "combined_2": "combined2", "combined_3": "combined3"}
    our_cases = {p.stem for p in (args.our_predictions / "original").glob("*.npy")}
    method_dirs = sorted(p for p in args.other_predictions.iterdir() if p.is_dir())
    methods = ["FreGNet"] + [p.name for p in method_dirs]
    label_cases = {p.name.removesuffix(".nii.gz") for p in args.labels_dir.glob("*.nii.gz")}
    common_cases = sorted(our_cases & label_cases)
    if not common_cases:
        raise RuntimeError("No common cases between FreGNet cache and labels directory")

    rows = []
    for case in tqdm(common_cases, desc="Computing per-case DSC", unit="case"):
        gt = read_mask(args.labels_dir / f"{case}.nii.gz")
        scores = {}
        for condition in conditions:
            scores["FreGNet", condition] = dsc(read_mask(args.our_predictions / our_condition_dirs[condition] / f"{case}.npy"), gt)
            for method_dir in method_dirs:
                pred_file = method_dir / case / f"{condition}.nii.gz"
                if pred_file.is_file():
                    pred = read_mask(pred_file)
                    if pred.shape != gt.shape:
                        raise ValueError(f"Shape mismatch for {method_dir.name}/{case}/{condition}: {pred.shape} vs {gt.shape}")
                    scores[method_dir.name, condition] = dsc(pred, gt)
        available_baselines = [m for m in methods[1:] if (m, "original") in scores]
        if not available_baselines:
            continue
        row = {"case": case}
        for method in methods:
            for condition in conditions:
                value = scores.get((method, condition), np.nan)
                row[f"{method}_{condition}_DSC"] = round(100.0 * value, 2) if np.isfinite(value) else np.nan
        for condition in conditions:
            baseline_values = [scores[m, condition] for m in available_baselines if (m, condition) in scores]
            mean_gap = scores["FreGNet", condition] - float(np.mean(baseline_values)) if baseline_values else np.nan
            best_gap = scores["FreGNet", condition] - float(np.max(baseline_values)) if baseline_values else np.nan
            row[f"FreGNet_minus_baseline_mean_{condition}"] = round(100.0 * mean_gap, 2) if np.isfinite(mean_gap) else np.nan
            row[f"FreGNet_minus_best_baseline_{condition}"] = round(100.0 * best_gap, 2) if np.isfinite(best_gap) else np.nan
        gaps = [row[f"FreGNet_minus_baseline_mean_{c}"] for c in conditions if np.isfinite(row[f"FreGNet_minus_baseline_mean_{c}"])]
        row["mean_gap_across_conditions"] = round(float(np.mean(gaps)), 2) if gaps else np.nan
        row["min_gap_across_conditions"] = round(float(np.min(gaps)), 2) if gaps else np.nan
        row["original_gap_to_baseline_mean"] = row["FreGNet_minus_baseline_mean_original"]
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["mean_gap_across_conditions", "min_gap_across_conditions"], ascending=False
    ).reset_index(drop=True)
    result.to_csv(args.output_dir / "case_dsc_comparison.csv", index=False)
    result.head(args.top_k).to_csv(args.output_dir / "top_cases_by_dsc_advantage.csv", index=False)

    ranking_cols = [
        "case",
        "original_gap_to_baseline_mean",
        "mean_gap_across_conditions",
        "min_gap_across_conditions",
    ]
    ranking = result[ranking_cols].head(args.top_k)
    ranking.to_csv(args.output_dir / "case_ranking_summary.csv", index=False)
    print(f"Cases evaluated: {len(result)}")
    print(ranking.to_string(index=False))
    print(f"Saved ranking files to {args.output_dir}")


if __name__ == "__main__":
    main()
