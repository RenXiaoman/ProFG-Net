#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compute_seg_metrics import compute_case_metrics, foreground_mask, read_seg


METRICS = ["dice", "iou", "hd95_voxel", "sensitivity", "precision"]


@dataclass(frozen=True)
class Experiment:
    name: str
    pred_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    config_path: Path
    gt_root: Path
    experiments: list[Experiment]

    @property
    def labels_ts(self) -> Path:
        return self.gt_root / "labelsTs"

    @property
    def images_ts(self) -> Path:
        return self.gt_root / "imagesTs"


def parse_experiment_file(path: Path) -> ExperimentConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment file: {path}")

    experiments: list[Experiment] = []
    gt_root: Path | None = None

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid line {line_no} in {path}: {raw_line!r}")

        name, value = parts
        if name == "gt":
            gt_root = Path(value)
        else:
            experiments.append(Experiment(name=name, pred_dir=Path(value)))

    if gt_root is None:
        raise ValueError(f"No gt line found in {path}")
    if not experiments:
        raise ValueError(f"No experiment entries found in {path}")

    dataset_name = path.stem.removeprefix("experiments_")
    return ExperimentConfig(name=dataset_name, config_path=path, gt_root=gt_root, experiments=experiments)


def case_name_from_label(label_file: Path) -> str:
    return label_file.name.removesuffix(".nii.gz")


def find_prediction(pred_dir: Path, case: str) -> Path | None:
    candidates = [
        pred_dir / f"{case}.nii.gz",
        pred_dir / f"{case}_pred.nii.gz",
        pred_dir / f"{case}_0000.nii.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def check_paths(config: ExperimentConfig) -> dict[str, list[str]]:
    problems: dict[str, list[str]] = {}

    if not config.gt_root.is_dir():
        problems.setdefault("missing_gt_root", []).append(str(config.gt_root))
    if not config.labels_ts.is_dir():
        problems.setdefault("missing_labelsTs", []).append(str(config.labels_ts))
    if not config.images_ts.is_dir():
        problems.setdefault("missing_imagesTs", []).append(str(config.images_ts))

    for experiment in config.experiments:
        if not experiment.pred_dir.is_dir():
            problems.setdefault("missing_prediction_dirs", []).append(f"{experiment.name}: {experiment.pred_dir}")

    if not problems and not list(config.labels_ts.glob("*.nii.gz")):
        problems.setdefault("empty_labelsTs", []).append(str(config.labels_ts))

    return problems


def print_path_report(configs: Iterable[ExperimentConfig]) -> bool:
    all_ok = True
    for config in configs:
        print(f"\n[{config.name}] {config.config_path}")
        problems = check_paths(config)
        if problems:
            all_ok = False
            for key, values in problems.items():
                for value in values:
                    print(f"  MISSING {key}: {value}")
            continue

        label_files = sorted(config.labels_ts.glob("*.nii.gz"))
        print(f"  OK gt_root: {config.gt_root}")
        print(f"  OK imagesTs: {config.images_ts}")
        print(f"  OK labelsTs: {config.labels_ts} ({len(label_files)} cases)")

        sample_cases = [case_name_from_label(i) for i in label_files[:5]]
        for experiment in config.experiments:
            pred_files = sorted(experiment.pred_dir.glob("*.nii.gz"))
            missing_samples = [case for case in sample_cases if find_prediction(experiment.pred_dir, case) is None]
            status = "OK" if not missing_samples else "CHECK"
            print(f"  {status} {experiment.name}: {experiment.pred_dir} ({len(pred_files)} nii.gz)")
            if missing_samples:
                all_ok = False
                print(f"    sample predictions not found: {missing_samples}")

    return all_ok


def evaluate_dataset(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    problems = check_paths(config)
    if problems:
        flattened = [f"{key}: {value}" for key, values in problems.items() for value in values]
        raise FileNotFoundError("; ".join(flattened))

    label_files = sorted(config.labels_ts.glob("*.nii.gz"))
    cases = [case_name_from_label(i) for i in label_files]
    rows_by_metric: dict[str, list[dict]] = {metric: [] for metric in METRICS}

    cache: dict[Path, tuple] = {}

    def read_cached(path: Path):
        if path not in cache:
            cache[path] = read_seg(path)
        return cache[path]

    for label_file in tqdm(label_files, desc=f"{config.name} cases", unit="case"):
        case = case_name_from_label(label_file)
        gt_seg, gt_spacing = read_cached(label_file)
        gt = foreground_mask(gt_seg)

        metric_rows = {metric: {"case": case} for metric in METRICS}
        for experiment in config.experiments:
            pred_file = find_prediction(experiment.pred_dir, case)
            if pred_file is None:
                for metric in METRICS:
                    metric_rows[metric][experiment.name] = None
                continue

            pred_seg, _ = read_cached(pred_file)
            pred = foreground_mask(pred_seg)
            if pred.shape != gt.shape:
                raise ValueError(
                    f"Shape mismatch for {config.name}/{experiment.name}/{case}: "
                    f"pred {pred.shape}, gt {gt.shape}"
                )

            metrics = compute_case_metrics(pred, gt, gt_spacing)
            for metric in METRICS:
                metric_rows[metric][experiment.name] = metrics[metric]

        for metric in METRICS:
            rows_by_metric[metric].append(metric_rows[metric])

    dataframes = {}
    columns = ["case"] + [experiment.name for experiment in config.experiments]
    for metric, rows in rows_by_metric.items():
        dataframes[metric] = pd.DataFrame(rows, columns=columns)
    return dataframes


def write_excel(config: ExperimentConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{config.name}_case_metrics.xlsx"
    dataframes = evaluate_dataset(config)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for metric in METRICS:
            dataframes[metric].to_excel(writer, sheet_name=metric, index=False)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-case comparative metrics to XLSX files.")
    parser.add_argument(
        "--experiment_files",
        nargs="+",
        type=Path,
        default=[
            Path("paper_tools/comparative/experiments_AHCDU.txt"),
            Path("paper_tools/comparative/experiments_PICAI.txt"),
        ],
    )
    parser.add_argument("--output_dir", type=Path, default=Path("paper_tools/comparative/metrics_xlsx"))
    parser.add_argument("--check_only", action="store_true", help="Only validate configured paths.")
    args = parser.parse_args()

    configs = [parse_experiment_file(path) for path in args.experiment_files]
    paths_ok = print_path_report(configs)
    if args.check_only:
        if not paths_ok:
            raise SystemExit(1)
        return

    for config in configs:
        output_file = write_excel(config, args.output_dir)
        print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
