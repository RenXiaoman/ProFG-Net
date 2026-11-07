#!/usr/bin/env python3
"""Evaluate FreGNet and its matched baseline under synthetic image degradations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.ndimage import binary_erosion, gaussian_filter, generate_binary_structure
from scipy.spatial import cKDTree


METRICS = ("DSC", "IoU", "HD95", "Sensitivity", "Precision")
# Edit all four lists together. Their lengths must remain identical so that
# every visualization contains one original panel plus the same number of
# degraded panels.
DEGRADATION_LEVELS = {
    "gaussian_noise": [0.01, 0.03, 0.05],
    "speckle_noise": [0.05, 0.10, 0.20],
    "contrast_reduction": [0.8, 0.6, 0.4],
    "inplane_gaussian_blur": [0.5, 1.0, 1.5],
}
COMBINED_DEGRADATIONS = (
    "gaussian_noise",
    "speckle_noise",
    "contrast_reduction",
    "inplane_gaussian_blur",
)


def validate_degradation_levels() -> None:
    lengths = {name: len(values) for name, values in DEGRADATION_LEVELS.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "All degradation level lists must have the same length; "
            f"got {lengths}. Add or remove levels from every list together."
        )


def load_nifti(path: Path) -> np.ndarray:
    data = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    # NIfTI arrays in this dataset are stored as H x W x D, while the
    # nnU-Net network consumes D x H x W tensors.
    if data.ndim == 3 and data.shape[-1] <= data.shape[0] and data.shape[0] > data.shape[-1]:
        data = np.transpose(data, (2, 0, 1))
    return data


def normalize01(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def degrade(image: np.ndarray, kind: str, strength: float, seed: int, clip_output: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "gaussian_noise":
        result = image + rng.normal(0.0, strength, image.shape).astype(np.float32)
    elif kind == "speckle_noise":
        result = image * (1.0 + strength * rng.normal(0.0, 1.0, image.shape).astype(np.float32))
    elif kind == "contrast_reduction":
        result = image.mean(dtype=np.float32) + strength * (image - image.mean(dtype=np.float32))
    elif kind == "inplane_gaussian_blur":
        result = gaussian_filter(image, sigma=(0.0, strength, strength), mode="nearest")
    else:
        raise ValueError(f"Unknown degradation: {kind}")
    if clip_output:
        result = np.clip(result, 0.0, 1.0)
    return result.astype(np.float32)


def combined_degrade(image: np.ndarray, level_index: int, case: str, base_seed: int, clip_output: bool) -> np.ndarray:
    """Apply the same-index level of all four degradations sequentially."""
    result = image.astype(np.float32, copy=True)
    for kind in COMBINED_DEGRADATIONS:
        strength = float(DEGRADATION_LEVELS[kind][level_index])
        seed_text = f"{case}:combined:{level_index}:{kind}"
        local_seed = base_seed + int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
        result = degrade(result, kind, strength, local_seed, clip_output)
    return result


def surface(mask: np.ndarray) -> np.ndarray:
    struct = generate_binary_structure(3, 3)
    return np.argwhere(mask.astype(bool) & ~binary_erosion(mask.astype(bool), struct))


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    ps, gs = surface(pred), surface(gt)
    if len(ps) == 0 and len(gs) == 0:
        return 0.0
    if len(ps) == 0 or len(gs) == 0:
        return 100.0
    distances = np.concatenate([cKDTree(gs).query(ps)[0], cKDTree(ps).query(gs)[0]])
    return float(np.percentile(distances, 95))


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    smooth = 1e-6
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.float64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.float64)
    return {
        "DSC": float((2 * tp + smooth) / (2 * tp + fp + fn + smooth)),
        "IoU": float((tp + smooth) / (tp + fp + fn + smooth)),
        "HD95": hd95(pred, gt),
        "Sensitivity": float((tp + smooth) / (tp + fn + smooth)),
        "Precision": float((tp + smooth) / (tp + fp + smooth)),
    }


def load_model(checkpoint: Path, baseline: bool, device: torch.device, arch: str = "cmsag") -> torch.nn.Module:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    if baseline:
        from nnunetv2.training.nnUNetTrainer.FreGNetTrain.ablation_trainers import FreGNetBaselineTrainer
        model = FreGNetBaselineTrainer.build_network_architecture(None, None, 1, 2, False)
    elif arch == "adaptive":
        # Previous FreGNet: CMSAG whose W_x is AdaptiveComplementaryMultiScaleDepthwiseBlock.
        from nnunetv2.training.nnUNetTrainer.FreGNetTrain.DIYBoundaryAdaptiveCMSAGAuxTrainer import (
            DIYBoundaryAdaptiveCMSAGAuxTrainer,
        )
        model = DIYBoundaryAdaptiveCMSAGAuxTrainer.build_network_architecture(None, None, 1, 2, False)
    elif arch == "learnable":
        # Newest FreGNet: learnable convolution wavelet + adaptive CMSAG.
        from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer import (
            LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer,
        )
        model = LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer.build_network_architecture(None, None, 1, 2, False)
    else:
        from nnunetv2.training.nnUNetTrainer.FreGNetTrain.DIYBoundaryCMSAGAuxTrainer import DIYBoundaryCMSAGAuxTrainer
        model = DIYBoundaryCMSAGAuxTrainer.build_network_architecture(None, None, 1, 2, False)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = state.get("network_weights", state.get("model_state_dict", state))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def contour(ax, image: np.ndarray, gt: np.ndarray, pred: np.ndarray | None, title: str, slice_index: int) -> None:
    # NIfTI data are converted to D x H x W for inference. Transpose only the
    # displayed slice so the in-plane H/W orientation matches the source image;
    # the depth index and all metric computations remain unchanged.
    image_slice = image[slice_index].T
    gt_slice = gt[slice_index].T
    display_kwargs = {"vmin": 0, "vmax": 1} if image.min() >= 0 and image.max() <= 1 else {}
    ax.imshow(image_slice, cmap="gray", **display_kwargs)
    gt_edge = gt_slice.astype(bool) & ~binary_erosion(gt_slice.astype(bool), generate_binary_structure(2, 1))
    yy, xx = np.where(gt_edge)
    ax.plot(xx, yy, ".", color="red", markersize=0.6, linewidth=0)
    if pred is not None:
        pred_slice = pred[slice_index].T
        pred_edge = pred_slice.astype(bool) & ~binary_erosion(pred_slice.astype(bool), generate_binary_structure(2, 1))
        yy, xx = np.where(pred_edge)
        ax.plot(xx, yy, ".", color="yellow", markersize=0.6, linewidth=0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_four_panel(path: Path, original: np.ndarray, degraded: list[np.ndarray], gt: np.ndarray, preds: list[np.ndarray], labels: list[str]) -> None:
    slice_index = int(np.argmax(gt.reshape(gt.shape[0], -1).sum(axis=1)))
    panel_count = len(degraded) + 1
    fig, axes = plt.subplots(1, panel_count, figsize=(4 * panel_count, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    contour(axes[0], original, gt, None, "Original (GT: red)", slice_index)
    for i in range(len(degraded)):
        contour(axes[i + 1], degraded[i], gt, preds[i], f"{labels[i]} (GT: red, Pred: yellow)", slice_index)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def metric_case_task(case: str, gt: np.ndarray, original_preds: dict[str, np.ndarray], degraded_preds: dict[str, dict[str, list[np.ndarray]]]) -> list[dict]:
    records = []
    original_metrics = {}
    for model_name, pred in original_preds.items():
        original_metrics[model_name] = metrics(pred, gt)
        for metric_name, value in original_metrics[model_name].items():
            records.append({"case": case, "model": model_name, "degradation": "original", "strength": 0.0, "metric": metric_name, "value": value, "drop": 0.0})
    for model_name, kinds in degraded_preds.items():
        for kind, entries in kinds.items():
            for strength, pred in entries:
                current = metrics(pred, gt)
                original = original_metrics[model_name]
                for metric_name, value in current.items():
                    direction = -1.0 if metric_name == "HD95" else 1.0
                    drop = direction * (value - original[metric_name])
                    records.append({"case": case, "model": model_name, "degradation": kind, "strength": float(strength), "metric": metric_name, "value": value, "drop": drop})
    return records


def plot_case_task(path: Path, original_slice: np.ndarray, gt_slice: np.ndarray, degraded_slices: list[np.ndarray], pred_slices: list[np.ndarray], labels: list[str]) -> None:
    save_four_panel(path, original_slice[None, ...], [item[None, ...] for item in degraded_slices], gt_slice[None, ...], [item[None, ...] for item in pred_slices], labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset", type=Path, default=Path("/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw/Dataset141_FullPICAI"))
    parser.add_argument("--splits", type=Path, default=Path("/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_preprocessed/Dataset141_FullPICAI/splits_final.json"))
    parser.add_argument("--freGnet_checkpoint", type=Path, default=Path("/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_results/Dataset141_FullPICAI/DIYBoundaryCMSAGAuxTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"))
    parser.add_argument("--baseline_checkpoint", type=Path, default=Path("/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_results/Dataset141_FullPICAI/FreGNetBaselineTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"))
    parser.add_argument("--model", choices=("all", "FreGNet", "Baseline"), default="all", help="Evaluate all models or only FreGNet/Baseline.")
    parser.add_argument(
        "--freGnet_arch",
        choices=("cmsag", "adaptive", "learnable"),
        default="cmsag",
        help="Architecture for the FreGNet entry: cmsag (original), adaptive (previous AdaptiveCMSAG), or learnable (newest learnable-wavelet AdaptiveCMSAG).",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("paper_tools/noise_quality"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max_cases", type=int, default=0, help="Smoke-test limit; 0 evaluates all validation cases.")
    parser.add_argument("--norm", action="store_true", help="Apply per-case Min-Max normalization to [0, 1].")
    parser.add_argument("--predict_only", action="store_true", help="Run Step 0 prediction caching only; skip metrics and plots.")
    args = parser.parse_args()
    validate_degradation_levels()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visualizations").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(args.gpu))

    val_cases = json.loads(args.splits.read_text())[0]["val"]
    if args.max_cases > 0:
        val_cases = val_cases[:args.max_cases]
    # The requested cases come from splits_final.json and therefore belong to the
    # labeled training pool (the validation fold is an internal validation split).
    image_dir, label_dir = args.raw_dataset / "imagesTr", args.raw_dataset / "labelsTr"
    available_models = {"FreGNet": (args.freGnet_checkpoint, False), "Baseline": (args.baseline_checkpoint, True)}
    models = available_models if args.model == "all" else {args.model: available_models[args.model]}
    cache_config = {
        "norm": bool(args.norm),
        "seed": int(args.seed),
        "levels": DEGRADATION_LEVELS,
        "freGnet_checkpoint": str(args.freGnet_checkpoint.resolve()),
        "freGnet_arch": str(args.freGnet_arch),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "freGnet_mtime_ns": args.freGnet_checkpoint.stat().st_mtime_ns,
        "baseline_mtime_ns": args.baseline_checkpoint.stat().st_mtime_ns,
        "model": args.model,
    }
    cache_signature = hashlib.sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()[:12]
    cache_root = args.output_dir / "predictions" / cache_signature
    level_count = len(next(iter(DEGRADATION_LEVELS.values())))

    def cache_file(model_name: str, condition: str, case: str, level_index: int | None = None) -> Path:
        if condition == "original":
            filename = f"{case}.npy"
        else:
            filename = f"{case}.npy"
        return cache_root / model_name / condition / filename

    # Step 0: run inference only for missing cached predictions.
    missing = []
    for model_name in models:
        for case in val_cases:
            missing.append((model_name, "original", case, None))
            for i in range(level_count):
                missing.append((model_name, f"combined{i + 1}", case, i))
    missing = [item for item in missing if not cache_file(*item).is_file()]
    if missing:
        print(f"Step 0/3: generating {len(missing)} missing cached predictions")
        loaded_models = {
            name: load_model(
                checkpoint,
                baseline,
                device,
                args.freGnet_arch if name == "FreGNet" else "cmsag",
            )
            for name, (checkpoint, baseline) in models.items()
        }
        with torch.no_grad():
            for model_name, condition, case, level_index in tqdm(missing, desc="Step 0/3: inference", unit="prediction"):
                image = load_nifti(image_dir / f"{case}_0000.nii.gz")
                if args.norm:
                    image = normalize01(image)
                if condition == "original":
                    degraded = image
                elif condition.startswith("combined"):
                    degraded = combined_degrade(image, level_index, case, args.seed, clip_output=args.norm)
                else:
                    degraded = combined_degrade(image, level_index, case, args.seed, clip_output=args.norm)
                logits = loaded_models[model_name](torch.from_numpy(degraded[None, None]).to(device))
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                prediction = logits.argmax(1).squeeze(0).cpu().numpy().astype(bool)
                output = cache_file(model_name, condition, case, level_index)
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(output, prediction)
        del loaded_models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("Step 0/3: all cached predictions found; skipping inference")

    if args.predict_only:
        print("Prediction-only mode: Step 1 metrics and Step 2 plots skipped.")
        return

    records: list[dict] = []
    metric_futures = []
    plot_futures = []
    spawn_context = get_context("spawn")
    # Step 1: compute all metrics exclusively from cached predictions.
    with ProcessPoolExecutor(max_workers=20, mp_context=spawn_context) as metric_pool:
        for case in tqdm(val_cases, desc="Step 1/3: metrics from cache", unit="case"):
            image = load_nifti(image_dir / f"{case}_0000.nii.gz")
            if args.norm:
                image = normalize01(image)
            gt = load_nifti(label_dir / f"{case}.nii.gz") > 0
            original_preds = {name: np.load(cache_file(name, "original", case)) for name in models}
            all_predictions = {name: {} for name in models}
            for name in models:
                all_predictions[name]["combined"] = [np.load(cache_file(name, f"combined{i + 1}", case, i)) for i in range(level_count)]
            degraded_payload = {
                name: {
                    "combined": list(zip(range(1, level_count + 1), all_predictions[name]["combined"])),
                }
                for name in models
            }
            metric_futures.append(metric_pool.submit(metric_case_task, case, gt, original_preds, degraded_payload))
        for future in metric_futures:
            records.extend(future.result())

    # Step 2: generate only combined visualizations from cached predictions.
    with ProcessPoolExecutor(max_workers=20, mp_context=spawn_context) as plot_pool:
        for case in tqdm(val_cases, desc="Step 2/3: plots from cache", unit="case"):
            image = load_nifti(image_dir / f"{case}_0000.nii.gz")
            if args.norm:
                image = normalize01(image)
            gt = load_nifti(label_dir / f"{case}.nii.gz") > 0
            combined_images = [combined_degrade(image, i, case, args.seed, clip_output=args.norm) for i in range(level_count)]
            combined_labels = [f"Combined level {i + 1}" for i in range(level_count)]
            slice_index = int(np.argmax(gt.reshape(gt.shape[0], -1).sum(axis=1)))
            for model_name in models:
                combined_preds = [np.load(cache_file(model_name, f"combined{i + 1}", case, i)) for i in range(level_count)]
                plot_futures.append(plot_pool.submit(
                    plot_case_task,
                    args.output_dir / "visualizations" / f"{case}_combined_{model_name}.png",
                    image[slice_index], gt[slice_index],
                    [item[slice_index] for item in combined_images],
                    [item[slice_index] for item in combined_preds], combined_labels,
                ))
        for future in plot_futures:
            future.result()
    case_df = pd.DataFrame(records)
    case_df.to_csv(args.output_dir / "case_metrics.csv", index=False)
    summary = (case_df.groupby(["model", "degradation", "strength", "metric"], as_index=False).agg(mean=("value", "mean"), std=("value", "std"), mean_drop=("drop", "mean")))
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    # Compact overall table for the original data and the three combined
    # degradation levels. Keep numeric mean/std columns for further analysis
    # and a formatted mean +/- std column for direct use in a manuscript.
    overall_conditions = [
        ("original", 0.0, "Original"),
        ("combined", 1.0, "Combined level 1"),
        ("combined", 2.0, "Combined level 2"),
        ("combined", 3.0, "Combined level 3"),
    ]
    overall_rows = []
    for model_name in models:
        for degradation, strength, condition in overall_conditions:
            row = {"model": model_name, "condition": condition}
            part = summary[
                (summary.model == model_name)
                & (summary.degradation == degradation)
                & (summary.strength == strength)
            ]
            for metric_name in METRICS:
                metric_part = part[part.metric == metric_name]
                if metric_part.empty:
                    row[f"{metric_name}_mean"] = np.nan
                    row[f"{metric_name}_std"] = np.nan
                    row[f"{metric_name}_mean_std"] = ""
                else:
                    mean_value = float(metric_part.iloc[0]["mean"])
                    std_value = float(metric_part.iloc[0]["std"]) if pd.notna(metric_part.iloc[0]["std"]) else 0.0
                    row[f"{metric_name}_mean"] = mean_value
                    row[f"{metric_name}_std"] = std_value
                    row[f"{metric_name}_mean_std"] = f"{mean_value:.4f} +/- {std_value:.4f}"
            overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(args.output_dir / "overall_combined_metrics.csv", index=False)
    try:
        with pd.ExcelWriter(args.output_dir / "noise_quality_summary.xlsx") as writer:
            case_df.to_excel(writer, sheet_name="case_metrics", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            overall.to_excel(writer, sheet_name="overall_combined", index=False)
    except Exception as exc:
        print(f"[WARN] Excel output unavailable: {exc}")
    for kind in ["combined"]:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
        for ax, metric_name in zip(axes, METRICS):
            for model_name, color in (("FreGNet", "tab:blue"), ("Baseline", "tab:orange")):
                if model_name not in models:
                    continue
                part = summary[(summary.model == model_name) & (summary.degradation == kind) & (summary.metric == metric_name)]
                ax.plot(part.strength, part["mean"], marker="o", label=model_name, color=color)
            ax.set_title(metric_name); ax.set_xlabel("Degradation strength"); ax.grid(alpha=0.3)
        axes[0].legend(); fig.savefig(args.output_dir / f"{kind}_curves.png", dpi=200); plt.close(fig)
    print("\nOverall metrics (original and combined degradation levels):")
    print(overall[["model", "condition"] + [f"{metric}_mean_std" for metric in METRICS]].to_string(index=False))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
