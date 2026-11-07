#!/usr/bin/env python3
"""Export nnUNet softmax probability maps for train/validation cases and score them.

The output layout is:
    Prob/{130,141}/{labelsTr,labelsTs}/*.npz

For Dataset130, labelsTr and labelsTs correspond to the raw dataset directories,
so all 463 labelsTr cases and all 104 labelsTs cases are processed. For Dataset141,
the fold-0 split is retained. The files are probability maps, not hard segmentations.
The reported segmentation metrics threshold the foreground probability at 0.5 only
for evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "130": {
        "name": "Dataset130_ProstateAHCDU",
        "trainer": "DIYBoundaryCMSAGAuxTrainer",
        "fold": 0,
    },
    "141": {
        "name": "Dataset141_FullPICAI",
        "trainer": "DIYBoundaryCMSAGAuxTrainer",
        "fold": 0,
    },
}


def read_binary(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def case_id_from_path(path: Path) -> str:
    return path.name.removesuffix(".nii.gz")


def load_probability(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "probabilities" in data:
            probabilities = data["probabilities"]
        elif "softmax" in data:
            probabilities = data["softmax"]
        else:
            raise KeyError(f"No probabilities/softmax array in {path}: {data.files}")
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W] probabilities in {path}, got {probabilities.shape}")
    if probabilities.shape[0] < 2:
        raise ValueError(f"Expected background and foreground channels in {path}, got {probabilities.shape}")
    return probabilities[1]


def metric_values(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    # Import after adding the repository root, so this script can be run from anywhere.
    sys.path.insert(0, str(REPO_ROOT))
    from compute_seg_metrics import compute_case_metrics  # pylint: disable=import-outside-toplevel

    return compute_case_metrics(pred, gt, spacing=(1.0, 1.0, 1.0))


def summarize(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    metrics = ["dice", "iou", "hd95_voxel", "sensitivity", "precision"]
    summary: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std(ddof=0)) if values.size else None,
        }
    return summary


def run_predictor(
    dataset_id: str,
    info: dict,
    images_dir: Path,
    output_dir: Path,
    checkpoint: str,
    device: str,
    disable_tta: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]

    cmd = [
        "nnUNetv2_predict",
        "-i", str(images_dir),
        "-o", str(output_dir),
        "-d", dataset_id,
        "-c", "3d_fullres",
        "-f", str(info["fold"]),
        "-tr", info["trainer"],
        "-chk", checkpoint,
        "--save_probabilities",
    ]
    if disable_tta:
        cmd.append("--disable_tta")

    print("[predict]", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def find_npz(directory: Path, case: str) -> Path:
    candidates = [directory / f"{case}.npz", directory / f"{case}_0000.npz"]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(directory.glob(f"{case}*.npz"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Probability map not found for {case} in {directory}")


def process_dataset(
    dataset_id: str,
    output_root: Path,
    checkpoint: str,
    device: str,
    disable_tta: bool,
    keep_tmp: bool,
) -> None:
    info = DATASETS[dataset_id]
    dataset_name = info["name"]
    raw_root = REPO_ROOT / "nnUNet" / "nnUNet_raw" / dataset_name

    # Dataset130 has an official raw labelsTs set. The fold split only subdivides
    # labelsTr into an internal train/validation split and must not replace it.
    if dataset_id == "130":
        groups = [
            {
                "split": "labelsTr",
                "images": raw_root / "imagesTr",
                "labels": raw_root / "labelsTr",
                "cases": sorted(case_id_from_path(p) for p in (raw_root / "labelsTr").glob("*.nii.gz")),
            },
            {
                "split": "labelsTs",
                "images": raw_root / "imagesTs",
                "labels": raw_root / "labelsTs",
                "cases": sorted(case_id_from_path(p) for p in (raw_root / "labelsTs").glob("*.nii.gz")),
            },
        ]
    else:
        split_path = REPO_ROOT / "nnUNet" / "nnUNet_preprocessed" / dataset_name / "splits_final.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))[info["fold"]]
        groups = [
            {
                "split": "labelsTr",
                "images": raw_root / "imagesTr",
                "labels": raw_root / "labelsTr",
                "cases": sorted(split["train"]),
            },
            {
                "split": "labelsTs",
                "images": raw_root / "imagesTr",
                "labels": raw_root / "labelsTr",
                "cases": sorted(split["val"]),
            },
        ]

    for group in groups:
        if not group["cases"]:
            raise RuntimeError(f"No cases found for {dataset_id}/{group['split']}")

    with tempfile.TemporaryDirectory(prefix=f"prob_{dataset_id}_", dir=str(REPO_ROOT)) as tmp:
        tmp_dir = Path(tmp)
        destination = output_root / dataset_id
        # Remove only this dataset's two generated groups so stale old split files
        # (e.g. the former 370/93 Dataset130 layout) cannot remain in the results.
        for group in groups:
            shutil.rmtree(destination / group["split"], ignore_errors=True)

        for group in groups:
            split_name = group["split"]
            cases = group["cases"]
            group_tmp = tmp_dir / split_name
            run_predictor(dataset_id, info, group["images"], group_tmp, checkpoint, device, disable_tta)
            split_dir = destination / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for case in tqdm(cases, desc=f"{dataset_id} save {split_name}", unit="case"):
                source = find_npz(group_tmp, case)
                shutil.copy2(source, split_dir / f"{case}.npz")

        rows: list[dict] = []
        for group in groups:
            split_name = group["split"]
            cases = group["cases"]
            labels_dir = group["labels"]
            for case in tqdm(cases, desc=f"{dataset_id} score {split_name}", unit="case"):
                probability = load_probability(destination / split_name / f"{case}.npz")
                gt_path = labels_dir / f"{case}.nii.gz"
                if not gt_path.is_file():
                    raise FileNotFoundError(f"Missing GT for {dataset_id}/{case}: {gt_path}")
                gt = read_binary(gt_path)
                if probability.shape != gt.shape:
                    raise ValueError(
                        f"Probability/GT shape mismatch for {dataset_id}/{case}: "
                        f"probability={probability.shape}, gt={gt.shape}"
                    )
                metrics = metric_values(probability >= 0.5, gt)
                rows.append({"dataset": dataset_id, "split": split_name, "case": case, **metrics})

        csv_path = destination / "case_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "dataset": dataset_id,
            "dataset_name": dataset_name,
            "trainer": info["trainer"],
            "fold": info["fold"],
            "checkpoint": checkpoint,
            "probability_threshold_for_metrics": 0.5,
            "num_train": len(groups[0]["cases"]),
            "num_val": len(groups[1]["cases"]),
            "split_definition": (
                "raw labelsTr/labelsTs directories"
                if dataset_id == "130"
                else "fold-0 split from splits_final.json; both subsets use imagesTr and labelsTr"
            ),
            "train": summarize([row for row in rows if row["split"] == "labelsTr"]),
            "val": summarize([row for row in rows if row["split"] == "labelsTs"]),
        }
        (destination / "summary_metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        if keep_tmp:
            retained = output_root / f"_tmp_predictions_{dataset_id}"
            if retained.exists():
                shutil.rmtree(retained)
            shutil.copytree(tmp_dir, retained)
            print(f"[saved] temporary predictor outputs: {retained}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "Prob")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--keep_tmp", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for dataset_id in args.dataset:
        process_dataset(
            dataset_id=dataset_id,
            output_root=args.output_root,
            checkpoint=args.checkpoint,
            device=args.device,
            disable_tta=args.disable_tta,
            keep_tmp=args.keep_tmp,
        )


if __name__ == "__main__":
    main()
