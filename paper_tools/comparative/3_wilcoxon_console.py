#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


DATASETS = {
    "Dataset130_AHCDU": Path("paper_tools/comparative/metrics_xlsx/AHCDU_case_metrics.xlsx"),
    "Dataset131_PICAI": Path("paper_tools/comparative/metrics_xlsx/PICAI_case_metrics.xlsx"),
}
METRICS = ["dice", "iou", "hd95_voxel", "sensitivity", "precision"]
OURS_COL = "Ours"


def p_to_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def wilcoxon_stars(df: pd.DataFrame, method: str) -> str:
    paired = df[[OURS_COL, method]].dropna()
    if paired.empty:
        return ""

    diff = paired[OURS_COL].to_numpy() - paired[method].to_numpy()
    if (diff == 0).all():
        return ""

    _, p_value = wilcoxon(paired[OURS_COL], paired[method], alternative="two-sided")
    return p_to_stars(float(p_value))


def main() -> None:
    for dataset_name, excel_path in DATASETS.items():
        print(f"\n{dataset_name}")
        excel = pd.ExcelFile(excel_path)
        first_df = pd.read_excel(excel_path, sheet_name=METRICS[0])
        methods = [col for col in first_df.columns if col not in {"case", OURS_COL}]

        header = ["metric"] + methods
        print("\t".join(header))
        for metric in METRICS:
            if metric not in excel.sheet_names:
                raise ValueError(f"Missing sheet {metric!r} in {excel_path}")
            df = pd.read_excel(excel_path, sheet_name=metric)
            row = [metric] + [wilcoxon_stars(df, method) for method in methods]
            print("\t".join(row))


if __name__ == "__main__":
    main()
