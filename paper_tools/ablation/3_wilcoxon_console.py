#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


DATASETS = {
    "Dataset131_PICAI_Ablation": Path("paper_tools/ablation/metrics_xlsx/PICAI_ablation_case_metrics.xlsx"),
}
METRICS = ["dice", "iou", "hd95_voxel", "sensitivity", "precision"]
OURS_COL = "WFE+MSAG+Boundary"


def p_to_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def wilcoxon_stars(df: pd.DataFrame, method: str, metric: str) -> str:
    paired = df[[OURS_COL, method]].dropna()
    if paired.empty:
        return ""

    if metric == "hd95_voxel":
        diff = paired[method].to_numpy() - paired[OURS_COL].to_numpy()
    else:
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
        if OURS_COL not in first_df.columns:
            raise ValueError(f"Missing ours column {OURS_COL!r} in {excel_path}")
        methods = [col for col in first_df.columns if col not in {"case", OURS_COL}]

        header = ["metric"] + methods
        print("\t".join(header))
        for metric in METRICS:
            if metric not in excel.sheet_names:
                raise ValueError(f"Missing sheet {metric!r} in {excel_path}")
            df = pd.read_excel(excel_path, sheet_name=metric)
            row = [metric] + [wilcoxon_stars(df, method, metric) for method in methods]
            print("\t".join(row))


if __name__ == "__main__":
    main()
