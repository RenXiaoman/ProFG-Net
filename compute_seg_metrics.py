from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def read_seg(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing_xyz = image.GetSpacing()
    spacing_zyx = tuple(float(i) for i in spacing_xyz[::-1])
    return array, spacing_zyx


def foreground_mask(seg: np.ndarray) -> np.ndarray:
    return seg > 0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum + gt_sum == 0:
        return 0.0
    return 2.0 * float(np.logical_and(pred, gt).sum()) / float(pred_sum + gt_sum)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum()) / float(union)


def surface_points(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 3)
    eroded = ndimage.binary_erosion(mask, structure)
    surface = mask.astype(np.uint8) - eroded.astype(np.uint8)
    return np.argwhere(surface > 0)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Match CAD/test_prostate_val.py: 26-neighbor surfaces, voxel distances."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float(np.sqrt(sum(dim**2 for dim in pred.shape)))

    pred_surface = surface_points(pred)
    gt_surface = surface_points(gt)
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return 0.0

    pred_to_gt = cKDTree(gt_surface).query(pred_surface, k=1)[0]
    gt_to_pred = cKDTree(pred_surface).query(gt_surface, k=1)[0]
    result = np.percentile(np.concatenate([pred_to_gt, gt_to_pred]), 95)
    return 0.0 if not np.isfinite(result) else float(result)


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = int(np.logical_and(pred, gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    if tp + fn == 0:
        return 1.0
    return float(tp) / float(tp + fn)


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    if tp + fp == 0:
        return 1.0
    return float(tp) / float(tp + fp)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return None, None
    return float(finite.mean()), float(finite.std(ddof=0))


def fmt_percent(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "nan±nan"
    return f"{mean * 100:.2f}±{std * 100:.2f}"


def fmt_mm(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "nan±nan"
    return f"{mean:.2f}±{std:.2f}"


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def metric_summary(cases: list[dict], key: str, is_distance: bool = False) -> dict:
    mean, std = mean_std([case[key] for case in cases])
    summary = {
        "mean": round_or_none(mean),
        "std": round_or_none(std),
    }
    if is_distance:
        summary["mean_std"] = fmt_mm(mean, std)
        summary["unit"] = "voxel"
    else:
        summary["mean_std_percent"] = fmt_percent(mean, std)
    return summary


def compute_case_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, ...],
) -> dict:
    case_dice = dice(pred, gt)
    case_iou = iou(pred, gt)
    case_hd95 = hd95(pred, gt)
    case_sensitivity = sensitivity(pred, gt)
    case_precision = precision(pred, gt)
    return {
        "dice": round_or_none(case_dice),
        "iou": round_or_none(case_iou),
        "hd95_voxel": round_or_none(case_hd95),
        "sensitivity": round_or_none(case_sensitivity),
        "precision": round_or_none(case_precision),
    }


def summarize_cases(cases: list[dict]) -> dict:
    return {
        "dice": metric_summary(cases, "dice"),
        "iou": metric_summary(cases, "iou"),
        "hd95_voxel": metric_summary(cases, "hd95_voxel", is_distance=True),
        "sensitivity": metric_summary(cases, "sensitivity"),
        "precision": metric_summary(cases, "precision"),
    }


def evaluate(pred_dir: Path, gt_dir: Path, output_name: str) -> dict:
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    cases = []
    missing_gt = []
    iterator = pred_files
    if tqdm is not None:
        iterator = tqdm(pred_files, desc="Computing metrics", unit="case")

    for pred_file in iterator:
        gt_file = gt_dir / pred_file.name
        if not gt_file.is_file():
            missing_gt.append(pred_file.name)
            continue

        pred_seg, pred_spacing = read_seg(pred_file)
        gt_seg, gt_spacing = read_seg(gt_file)
        pred = foreground_mask(pred_seg)
        gt = foreground_mask(gt_seg)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {pred_file.name}: pred {pred.shape}, gt {gt.shape}")

        cases.append(
            {
                "case": pred_file.stem.removesuffix(".nii"),
                "prediction": str(pred_file),
                "ground_truth": str(gt_file),
                **compute_case_metrics(pred, gt, gt_spacing),
                "spacing_zyx": list(gt_spacing),
            }
        )

    if missing_gt:
        raise FileNotFoundError(f"Missing ground-truth files for {len(missing_gt)} predictions: {missing_gt[:10]}")

    result = {
        "prediction_dir": str(pred_dir),
        "ground_truth_dir": str(gt_dir),
        "num_cases": len(cases),
        "summary": summarize_cases(cases),
        "cases": cases,
    }

    output_file = pred_dir / output_name
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=Path,
        # default=Path("nnUNet/infer/Dataset131_DIYBoundaryMSAGTrainer/4"),
        # default=Path("nnUNet/infer/Dataset130_DIYBoundaryMSAGAuxTrainer/2"),
        # default=Path("nnUNet/infer/Dataset112_DIYBoundaryMSAGAuxTrainer/2"),
        # default=Path("nnUNet/infer/Dataset158_DIYBoundaryMSAGTrainer/0"),
        # default=Path("nnUNet/infer/Dataset130_BoundaryTrainer/0"),
        # default=Path("nnUNet/infer/Dataset131_MSAGTrainer/3"),
        
        # default=Path("nnUNet/infer/Dataset141_DIYBoundaryAdaptiveCMSAGAuxTrainer/"),
        default=Path("nnUNet/infer/Dataset130_DIYBoundaryAdaptiveCMSAGAuxTrainer"),
        
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        # default=Path("nnUNet/nnUNet_raw/Dataset141_FullPICAI/labelsTs"),
        default=Path("nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU/labelsTs"),
        
        # default=Path("nnUNet/nnUNet_raw/Dataset131_ProstatePI-CAI/labelsTs"),
        # default=Path("nnUNet/nnUNet_raw/Dataset112_Promise12/labelsTs"),
        # default=Path("nnUNet/nnUNet_raw/Dataset158_prostate158/labelsTs"),
    )
    parser.add_argument("--output_name", default="1A.json")
    args = parser.parse_args()

    result = evaluate(args.pred_dir, args.gt_dir, args.output_name)
    print(f"Wrote {args.pred_dir / args.output_name}")
    print(f"Dice: {result['summary']['dice']['mean_std_percent']}")
    print(f"IoU: {result['summary']['iou']['mean_std_percent']}")
    print(f"HD95(voxel): {result['summary']['hd95_voxel']['mean_std']}")
    print(f"Sensitivity: {result['summary']['sensitivity']['mean_std_percent']}")
    print(f"Precision: {result['summary']['precision']['mean_std_percent']}")


if __name__ == "__main__":
    main()
