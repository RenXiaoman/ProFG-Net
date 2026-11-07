#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


EXPERIMENT_FILE = Path("paper_tools/ablation/experiments_PICAI.txt")
OUTPUT_CSV = Path("paper_tools/ablation/outputs/PICAI_recommended_slices.csv")
CASE_IDS = []

OURS_NAME = "WFE+MSAG+Boundary"
TOP_K = 3
GLOBAL_TOP_K = 40
MIN_GT_PIXELS = 10
MIN_OURS_PIXELS = 10
EXCLUDE_EDGE_SLICES = 0


class Experiment:
    def __init__(self, name: str, pred_dir: Path):
        self.name = name
        self.pred_dir = pred_dir


class ExperimentConfig:
    def __init__(self, gt_root: Path, experiments: list[Experiment]):
        self.gt_root = gt_root
        self.experiments = experiments

    @property
    def labels_ts(self) -> Path:
        return self.gt_root / "labelsTs"


def parse_experiment_file(path: Path) -> ExperimentConfig:
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
    return ExperimentConfig(gt_root=gt_root, experiments=experiments)


def find_label(labels_ts: Path, case: str) -> Path:
    label_file = labels_ts / f"{case}.nii.gz"
    if label_file.is_file():
        return label_file
    raise FileNotFoundError(f"Missing GT for case {case}: {label_file}")


def find_prediction(pred_dir: Path, case: str) -> Path:
    candidates = [
        pred_dir / f"{case}.nii.gz",
        pred_dir / f"{case}_pred.nii.gz",
        pred_dir / f"{case}_0000.nii.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing prediction for case {case}: tried {candidates}")


def read_nii(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def dice2d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum + gt_sum == 0:
        return 0.0
    return 2.0 * float(np.logical_and(pred, gt).sum()) / float(pred_sum + gt_sum)


def foreground(seg: np.ndarray) -> np.ndarray:
    return seg > 0


def case_name_from_label(label_file: Path) -> str:
    return label_file.name.removesuffix(".nii.gz")


def collect_case_ids(config: ExperimentConfig) -> list[str]:
    label_files = sorted(config.labels_ts.glob("*.nii.gz"))
    if not label_files:
        raise FileNotFoundError(f"No labels found in {config.labels_ts}")
    return [case_name_from_label(label_file) for label_file in label_files]


def rank_case_slices(case: str, config: ExperimentConfig) -> list[dict]:
    experiments = config.experiments
    ours = next((experiment for experiment in experiments if experiment.name == OURS_NAME), None)
    if ours is None:
        raise ValueError(f"Cannot find Ours model named {OURS_NAME!r} in {EXPERIMENT_FILE}")

    gt = foreground(read_nii(find_label(config.labels_ts, case)))
    pred_by_model = {
        experiment.name: foreground(read_nii(find_prediction(experiment.pred_dir, case)))
        for experiment in experiments
    }

    for model_name, pred in pred_by_model.items():
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {case}/{model_name}: pred {pred.shape}, gt {gt.shape}")

    rows = []
    competitors = [name for name in pred_by_model if name != OURS_NAME]
    for slice_index in range(EXCLUDE_EDGE_SLICES, gt.shape[0] - EXCLUDE_EDGE_SLICES):
        gt_slice = gt[slice_index]
        ours_slice = pred_by_model[OURS_NAME][slice_index]
        gt_pixels = int(gt_slice.sum())
        ours_pixels = int(ours_slice.sum())
        if gt_pixels < MIN_GT_PIXELS or ours_pixels < MIN_OURS_PIXELS:
            continue

        ours_dice = dice2d(ours_slice, gt_slice)
        competitor_dices = [dice2d(pred_by_model[name][slice_index], gt_slice) for name in competitors]
        competitor_mean = float(np.mean(competitor_dices))
        competitor_best = float(np.max(competitor_dices))
        if ours_dice <= competitor_best:
            continue

        rows.append(
            {
                "case": case,
                "slice_index": slice_index,
                "score": ours_dice,
                "ours_dice": ours_dice,
                "competitor_mean_dice": competitor_mean,
                "competitor_best_dice": competitor_best,
                "ours_minus_best": ours_dice - competitor_best,
                "gt_pixels": gt_pixels,
                "ours_pixels": ours_pixels,
                **{f"{name}_dice": dice2d(pred_by_model[name][slice_index], gt_slice) for name in pred_by_model},
            }
        )

    rows.sort(
        key=lambda row: (
            row["ours_minus_best"],
            row["ours_dice"],
            row["score"],
            row["gt_pixels"],
        ),
        reverse=True,
    )
    return rows


def format_slice_dices(row: pd.Series, model_names: list[str]) -> str:
    dice_parts = [f"{name}={row[f'{name}_dice']:.4f}" for name in model_names]
    return ", ".join(dice_parts)


def main() -> None:
    config = parse_experiment_file(EXPERIMENT_FILE)
    model_names = [experiment.name for experiment in config.experiments]
    case_ids = CASE_IDS or collect_case_ids(config)
    if CASE_IDS:
        print(f"Using manually configured CASE_IDS: {len(case_ids)} cases")
    else:
        print(f"CASE_IDS is empty. Scanning all labelsTs cases: {len(case_ids)} cases")

    all_rows = []
    recommendations = {}
    for case in tqdm(case_ids, desc="Ranking cases", unit="case"):
        rows = rank_case_slices(case, config)
        if not rows:
            recommendations[case] = []
            continue
        all_rows.extend(rows)
        recommendations[case] = sorted(row["slice_index"] for row in rows[:TOP_K])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(all_rows)
    if not output_df.empty:
        float_columns = output_df.select_dtypes(include=["float"]).columns
        output_df[float_columns] = output_df[float_columns].round(4)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print("Recommended slices:")
    for case in case_ids:
        slices = recommendations[case]
        if slices:
            print(f"{case}: {','.join(str(i) for i in slices)}")
        else:
            print(f"{case}: no valid slice")

    if not output_df.empty:
        print(f"\nGlobal top {min(GLOBAL_TOP_K, len(output_df))} slices with per-experiment Dice:")
        top_rows = output_df.sort_values(
            by=["ours_minus_best", "ours_dice", "score", "gt_pixels"],
            ascending=False,
        ).head(GLOBAL_TOP_K)
        top_rows = top_rows.sort_values(by=["case", "slice_index"], ascending=True)
        for _, row in top_rows.iterrows():
            print(
                f"{row['case']}: slice {int(row['slice_index'])}, "
                f"{format_slice_dices(row, model_names)}"
            )
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
