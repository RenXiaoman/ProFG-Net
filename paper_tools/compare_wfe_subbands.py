#!/usr/bin/env python3
"""Side-by-side comparison of the fixed and learnable wavelet sub-bands.

Reads the per-panel PNGs produced by visualize_wfe_subbands.py and stacks them
into one figure per case: rows are the eight sub-bands, the left column is the
fixed-wavelet version (`old`) and the right column is the learnable one (`new`).

Panels are reused as-is, so the pixels match the individual PNGs exactly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "paper_tools/frequncy"
BANDS = ("LLL", "aad", "ada", "add", "daa", "dad", "dda", "ddd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Directory holding old/ and new/.")
    parser.add_argument("--old-name", default="old", help="Left column directory name.")
    parser.add_argument("--new-name", default="new", help="Right column directory name.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <root>/compare.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--panel-size", type=float, default=1.5, help="Inches per panel.")
    return parser.parse_args()


def compose(case: str, old_dir: Path, new_dir: Path, output: Path, dpi: int, panel: float) -> None:
    fig, axes = plt.subplots(
        len(BANDS),
        2,
        figsize=(panel * 2, panel * len(BANDS)),
        dpi=dpi,
        facecolor="white",
    )
    for row, band in enumerate(BANDS):
        for col, (directory, label) in enumerate(((old_dir, "old"), (new_dir, "new"))):
            panel_path = directory / case / f"{row:02d}_{band}.png"
            if not panel_path.is_file():
                raise FileNotFoundError(f"Missing panel: {panel_path}")
            ax = axes[row][col]
            ax.imshow(plt.imread(panel_path), interpolation="nearest")
            ax.axis("off")
            if col == 0:
                ax.text(
                    -0.02,
                    0.5,
                    band,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
            if row == 0:
                ax.text(
                    0.5,
                    1.04,
                    label,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
    fig.suptitle(case, fontsize=11, y=0.998)
    fig.subplots_adjust(left=0.06, right=1.0, top=0.965, bottom=0.0, wspace=0.02, hspace=0.02)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    old_dir = args.root / args.old_name
    new_dir = args.root / args.new_name
    output_dir = args.output_dir or (args.root / "compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    old_cases = sorted(d.name for d in old_dir.iterdir() if d.is_dir())
    new_cases = sorted(d.name for d in new_dir.iterdir() if d.is_dir())
    if old_cases != new_cases:
        raise ValueError(
            f"Case sets differ. only in {args.old_name}: {sorted(set(old_cases) - set(new_cases))}, "
            f"only in {args.new_name}: {sorted(set(new_cases) - set(old_cases))}"
        )
    if not old_cases:
        raise ValueError(f"No case directories found under {old_dir}")

    for case in old_cases:
        output = output_dir / f"{case}.png"
        compose(case, old_dir, new_dir, output, args.dpi, args.panel_size)
        print(f"{case}: {output}")
    print(f"\nWrote {len(old_cases)} comparison figures to {output_dir}")


if __name__ == "__main__":
    main()
