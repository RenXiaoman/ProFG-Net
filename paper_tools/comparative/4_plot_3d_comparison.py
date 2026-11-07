#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


DPI = 200
EXPERIMENT_FILES = {
    "PICAI": Path("paper_tools/comparative/experiments_PICAI.txt"),
    "AHCDU": Path("paper_tools/comparative/experiments_AHCDU.txt"),
}
DEFAULT_PICAI_CASES = ["10707_1000723", "10873_1000889", "11169_1001192", "11341_1001364"]
DEFAULT_AHCDU_CASES = ["XuYue", "YangChangZhu", "ZhongYueXian"]


@dataclass(frozen=True)
class Experiment:
    name: str
    pred_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    gt_root: Path
    experiments: list[Experiment]

    @property
    def labels_ts(self) -> Path:
        return self.gt_root / "labelsTs"


@dataclass(frozen=True)
class DatasetCases:
    dataset: str
    cases: list[str]
    config: ExperimentConfig


@dataclass(frozen=True)
class PanelJob:
    row: int
    col: int
    name: str
    path: Path


@dataclass(frozen=True)
class SurfacePanel:
    row: int
    col: int
    name: str
    path: Path
    verts: np.ndarray
    faces: np.ndarray


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


def read_image(path: Path) -> sitk.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return sitk.ReadImage(str(path))


def read_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = read_image(path)
    mask = sitk.GetArrayFromImage(image) > 0
    spacing_xyz = image.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    return mask, spacing_zyx


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


def surface_from_mask(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    step_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")
    if not np.any(mask):
        raise ValueError("Cannot render an empty mask")

    padded = np.pad(mask.astype(np.float32), 1, mode="constant", constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=spacing, step_size=step_size)
    verts -= np.asarray(spacing, dtype=np.float64)
    return verts, faces


def build_surface_panel(job: PanelJob, step_size: int) -> SurfacePanel:
    mask, spacing = read_mask(job.path)
    verts, faces = surface_from_mask(mask, spacing, step_size=step_size)
    return SurfacePanel(row=job.row, col=job.col, name=job.name, path=job.path, verts=verts, faces=faces)


def set_equal_axes(ax: plt.Axes, verts: np.ndarray) -> None:
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    centers = (mins + maxs) / 2
    radius = float((maxs - mins).max() / 2)
    radius = max(radius, 1.0)

    ax.set_xlim(centers[2] - radius, centers[2] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[0] - radius, centers[0] + radius)


def is_dark_background(color: str) -> bool:
    red, green, blue = to_rgb(color)
    return (0.299 * red + 0.587 * green + 0.114 * blue) < 0.5


def add_surface(
    ax: plt.Axes,
    verts: np.ndarray,
    faces: np.ndarray,
    color: tuple[float, float, float],
    alpha: float,
    edge_color: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.08),
) -> None:
    # marching_cubes returns vertices in z, y, x order. Matplotlib displays x, y, z.
    xyz = verts[:, [2, 1, 0]]
    mesh = Poly3DCollection(xyz[faces], linewidths=0.05, alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor(edge_color)
    ax.add_collection3d(mesh)
    set_equal_axes(ax, verts)


def style_axis(ax: plt.Axes, title: str, elev: float, azim: float, background: str = "white") -> None:
    if title:
        ax.set_title(title, fontsize=8, pad=2, color="white" if is_dark_background(background) else "black")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.set_facecolor(background)


def safe_filename(value: str) -> str:
    """Make a method/case name safe for use as a single path component."""
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_cut_figures(
    surfaces: list[SurfacePanel],
    case_names: list[str],
    output_root: Path,
    elev: float,
    azim: float,
    show_title: bool,
    background: str = "white",
) -> Path:
    """Save one high-resolution PNG per case and per method/GT surface."""
    if len(case_names) != len({panel.row for panel in surfaces}):
        raise ValueError("The number of case names does not match the surface rows")

    output_root.mkdir(parents=True, exist_ok=True)
    panels_by_row: dict[int, list[SurfacePanel]] = {}
    for panel in surfaces:
        panels_by_row.setdefault(panel.row, []).append(panel)

    for row, case in enumerate(case_names):
        case_dir = output_root / safe_filename(case)
        case_dir.mkdir(parents=True, exist_ok=True)
        row_panels = sorted(panels_by_row[row], key=lambda panel: panel.col)
        for panel in row_panels:
            fig = plt.figure(figsize=(4.0, 4.0), dpi=300, facecolor=background)
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            add_surface(
                ax,
                panel.verts,
                panel.faces,
                (0.90, 0.05, 0.03),
                alpha=0.90,
            )
            style_axis(ax, panel.name if show_title else "", elev=elev, azim=azim, background=background)
            output_file = case_dir / f"{panel.col:02d}_{safe_filename(panel.name)}.png"
            fig.subplots_adjust(left=0, right=1, top=0.92 if show_title else 1, bottom=0)
            fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor=background)
            plt.close(fig)
        print(f"[cut] {case}: {len(row_panels)} figures -> {case_dir}")
    return output_root


def plot_3d_comparison(
    dataset_cases: list[DatasetCases],
    output_dataset: str,
    output_path: Path,
    elev: float,
    azim: float,
    show_title: bool,
    num_workers: int,
    step_size: int,
    cut: bool,
    background: str = "white",
) -> Path:
    if not dataset_cases or not any(item.cases for item in dataset_cases):
        raise ValueError("At least one case is required")
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if step_size < 1:
        raise ValueError("step_size must be >= 1")

    reference_config = dataset_cases[0].config
    panel_names = ["GT"] + [experiment.name for experiment in reference_config.experiments]
    for item in dataset_cases[1:]:
        names = ["GT"] + [experiment.name for experiment in item.config.experiments]
        if names != panel_names:
            raise ValueError(f"Experiment methods differ for {item.dataset}: {names} != {panel_names}")

    surface_color = (0.90, 0.05, 0.03)

    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    ncols = len(panel_names)
    nrows = sum(len(item.cases) for item in dataset_cases)
    fig = plt.figure(figsize=(2.0 * ncols, 2.0 * nrows), dpi=DPI, facecolor=background)

    jobs: list[PanelJob] = []
    row = 0
    for item in dataset_cases:
        for case in item.cases:
            panels: list[tuple[str, Path]] = [("GT", find_label(item.config.labels_ts, case))]
            panels.extend((experiment.name, find_prediction(experiment.pred_dir, case)) for experiment in item.config.experiments)

            for col, (name, path) in enumerate(panels):
                jobs.append(PanelJob(row=row, col=col, name=name, path=path))
            row += 1
    prepare_seconds = time.perf_counter() - stage_start
    print(f"[profile] prepare jobs: {prepare_seconds:.2f} s ({len(jobs)} panels)")

    stage_start = time.perf_counter()
    if num_workers == 1:
        surfaces = [build_surface_panel(job, step_size) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            surfaces = list(executor.map(build_surface_panel, jobs, [step_size] * len(jobs)))
    mesh_seconds = time.perf_counter() - stage_start
    total_verts = sum(len(panel.verts) for panel in surfaces)
    total_faces = sum(len(panel.faces) for panel in surfaces)
    print(
        f"[profile] read masks + marching cubes: {mesh_seconds:.2f} s "
        f"(verts={total_verts:,}, faces={total_faces:,}, workers={num_workers}, step_size={step_size})"
    )
    for panel in sorted(surfaces, key=lambda item: len(item.faces), reverse=True)[:5]:
        print(
            f"[profile] heavy panel row={panel.row} col={panel.col} {panel.name}: "
            f"verts={len(panel.verts):,}, faces={len(panel.faces):,}, file={panel.path}"
        )

    output = Path(str(output_path).format(dataset=output_dataset, case=str(nrows)))
    if cut:
        case_names = [case for item in dataset_cases for case in item.cases]
        cut_root = output.with_suffix("")
        stage_start = time.perf_counter()
        result = save_cut_figures(
            surfaces=surfaces,
            case_names=case_names,
            output_root=cut_root,
            elev=elev,
            azim=azim,
            show_title=show_title,
            background=background,
        )
        save_seconds = time.perf_counter() - stage_start
        total_seconds = time.perf_counter() - total_start
        print(f"[profile] save cut figures: {save_seconds:.2f} s")
        print(f"[profile] plot_3d_comparison total: {total_seconds:.2f} s")
        return result

    stage_start = time.perf_counter()
    for panel in surfaces:
        index = panel.row * ncols + panel.col + 1
        ax = fig.add_subplot(nrows, ncols, index, projection="3d")
        add_surface(
            ax,
            panel.verts,
            panel.faces,
            surface_color,
            alpha=0.90,
        )
        title = panel.name if show_title and panel.row == 0 else ""
        style_axis(ax, title, elev=elev, azim=azim, background=background)
    draw_seconds = time.perf_counter() - stage_start
    print(f"[profile] draw matplotlib 3D axes: {draw_seconds:.2f} s")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0, right=1, top=0.88 if show_title else 1, bottom=0, wspace=0.0, hspace=0.0)
    stage_start = time.perf_counter()
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    save_seconds = time.perf_counter() - stage_start
    plt.close(fig)
    total_seconds = time.perf_counter() - total_start
    print(f"[profile] save figure: {save_seconds:.2f} s")
    print(f"[profile] plot_3d_comparison total: {total_seconds:.2f} s")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 3D mesh comparison for one or more cases.")
    parser.add_argument(
        "--mixed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render PI-CAI cases first and AHCDU cases second. Use --no-mixed for single-dataset mode.",
    )
    parser.add_argument("--picai_cases", nargs="+", default=DEFAULT_PICAI_CASES)
    parser.add_argument("--ahcdu_cases", nargs="+", default=DEFAULT_AHCDU_CASES)
    parser.add_argument(
        "--dataset",
        choices=sorted(EXPERIMENT_FILES),
        default="AHCDU",  # AHCDU PICAI
        help="Dataset preset used to select the experiment file.",
    )
    parser.add_argument(
        "--experiment_file",
        type=Path,
        default=None,
        help="Optional custom experiment file. Overrides --dataset when provided.",
    )
    parser.add_argument("--case", nargs="+", default=None, help="Single-dataset cases used with --no-mixed.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("paper_tools/comparative/3dvisual/{dataset}_{case}_3d_comparison.png"),
    )
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Parallel workers for reading masks and extracting meshes. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="marching_cubes step size. Larger values are faster but produce coarser surfaces.",
    )
    parser.add_argument(
        "--show_title",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show panel titles. Use --no-show_title to hide them.",
    )
    parser.add_argument(
        "--cut",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save one high-resolution PNG per case and per panel instead of one combined grid.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of method names to render, e.g. --methods UNet nnU-Net Ours. GT is always included. "
        "Order follows the experiment file, so the panel numbering stays stable.",
    )
    parser.add_argument(
        "--background",
        default="white",
        help="Background colour for the figure and the 3D axes, e.g. white or black.",
    )
    args = parser.parse_args()

    if args.mixed:
        dataset_cases = [
            DatasetCases(
                dataset="PICAI",
                cases=args.picai_cases,
                config=parse_experiment_file(EXPERIMENT_FILES["PICAI"]),
            ),
            DatasetCases(
                dataset="AHCDU",
                cases=args.ahcdu_cases,
                config=parse_experiment_file(EXPERIMENT_FILES["AHCDU"]),
            ),
        ]
        output_dataset = "Mixed"
    else:
        experiment_file = args.experiment_file or EXPERIMENT_FILES[args.dataset]
        default_cases = DEFAULT_PICAI_CASES if args.dataset == "PICAI" else DEFAULT_AHCDU_CASES
        dataset_cases = [
            DatasetCases(
                dataset=args.dataset,
                cases=args.case or default_cases,
                config=parse_experiment_file(experiment_file),
            )
        ]
        output_dataset = args.dataset

    if args.methods:
        wanted = set(args.methods)
        filtered: list[DatasetCases] = []
        for item in dataset_cases:
            available = {experiment.name for experiment in item.config.experiments}
            missing = sorted(name for name in wanted if name not in available)
            if missing:
                raise ValueError(f"Methods not found in the {item.dataset} experiment file: {missing}")
            kept = [experiment for experiment in item.config.experiments if experiment.name in wanted]
            filtered.append(replace(item, config=replace(item.config, experiments=kept)))
        dataset_cases = filtered
        panel_names = ["GT"] + [experiment.name for experiment in dataset_cases[0].config.experiments]
        print(f"Methods in use ({len(panel_names)} panels): {', '.join(panel_names)}")

    start_time = time.perf_counter()
    output = plot_3d_comparison(
        dataset_cases=dataset_cases,
        output_dataset=output_dataset,
        output_path=args.output_path,
        elev=args.elev,
        azim=args.azim,
        show_title=args.show_title,
        num_workers=args.num_workers,
        step_size=args.step_size,
        cut=args.cut,
        background=args.background,
    )
    elapsed = time.perf_counter() - start_time
    print(f"Wrote {output}")
    print(f"Elapsed: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
