#!/usr/bin/env python3
"""VTK/PyVista transparent 3D surfaces for the Mixed_7 comparison cases.

The script reuses the VTK mesh builder from 6_plot_3d_vtk_surfaces.py and the
experiment-file parsing from 4_plot_3d_comparison.py. It can either render one
translucent surface per case and panel, or overlay GT plus several predictions
into a single translucent scene.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))

compare4 = importlib.import_module("paper_tools.comparative.4_plot_3d_comparison")
vtk6 = importlib.import_module("paper_tools.comparative.6_plot_3d_vtk_surfaces")

EXPERIMENT_FILES = compare4.EXPERIMENT_FILES
find_label = compare4.find_label
find_prediction = compare4.find_prediction
parse_experiment_file = compare4.parse_experiment_file

build_mesh = vtk6.build_mesh
hex_to_rgb = vtk6.hex_to_rgb
read_mask = vtk6.read_mask


DEFAULT_PICAI_CASES = ["10707_1000723", "10873_1000889", "11169_1001192", "11341_1001364"]
DEFAULT_AHCDU_CASES = ["XuYue", "YangChangZhu", "ZhongYueXian"]
DEFAULT_METHODS = ["UNet", "nnU-Net", "TransUNet", "ResGNet", "AWUNet", "Ours"]
METHOD_COLORS = {
    "UNet": "#4E79A7",
    "nnU-Net": "#F28E2B",
    "TransUNet": "#E15759",
    "ResGNet": "#76B7B2",
    "AWUNet": "#EDC948",
    "Ours": "#B07AA1",
}
GT_COLOR = "#9AA7B0"


def safe_filename(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def load_panel_paths(
    dataset: str,
    case: str,
    methods: list[str],
    gt_dir: Path,
) -> list[tuple[str, Path]]:
    config = parse_experiment_file(EXPERIMENT_FILES[dataset])
    gt_file = gt_dir / f"{case}.nii.gz"
    if not gt_file.is_file():
        raise FileNotFoundError(f"Missing GT for case {case}: {gt_file}")
    panels = [("GT", gt_file)]
    for experiment in config.experiments:
        if experiment.name in methods:
            panels.append(
                (experiment.name, find_prediction(experiment.pred_dir, case))
            )
    return panels


def make_plotter(window_size: int, background: str) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=(window_size, window_size))
    plotter.background_color = background
    return plotter


def set_camera(plotter: pv.Plotter, elev: float, azim: float, zoom: float, up_axis: str) -> None:
    bounds = np.asarray(plotter.bounds, dtype=np.float64).reshape(3, 2)
    center = bounds.mean(axis=1)
    radius = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0])) / 2.0
    azimuth = np.deg2rad(azim)
    elevation = np.deg2rad(elev)
    direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    up = np.array(
        {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }[up_axis]
    )
    plotter.camera_position = [tuple(center + direction * radius * 3.0), tuple(center), tuple(up)]
    plotter.camera.zoom(zoom)


def add_surface(
    plotter: pv.Plotter,
    path: Path,
    color: str,
    opacity: float,
    sigma_mm: float,
    smooth_iters: int,
    smooth_passband: float,
    decimate: float,
    pad: int,
    specular: float,
    specular_power: float,
) -> None:
    mask, spacing_zyx, direction_xyz, origin_xyz = read_mask(path)
    sigma_vox = (
        tuple(float(sigma_mm) / float(s) for s in spacing_zyx)
        if sigma_mm > 0
        else None
    )
    mesh = build_mesh(
        mask > 0,
        spacing_zyx,
        sigma_vox,
        smooth_iters,
        smooth_passband,
        decimate,
        pad,
        direction_xyz,
        origin_xyz,
    )
    plotter.add_mesh(
        mesh,
        color=hex_to_rgb(color),
        opacity=opacity,
        smooth_shading=True,
        specular=specular,
        specular_power=specular_power,
        show_edges=False,
    )


def render_cut(
    output_root: Path,
    panels: list[tuple[str, Path]],
    case: str,
    args: argparse.Namespace,
) -> None:
    case_dir = output_root / safe_filename(case)
    case_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, path) in enumerate(panels):
        color = args.surface_color
        opacity = args.surface_opacity
        plotter = make_plotter(args.window_size, args.background)
        add_surface(
            plotter,
            path,
            color,
            opacity,
            args.sigma_mm,
            args.smooth_iters,
            args.smooth_passband,
            args.decimate,
            args.pad,
            args.specular,
            args.specular_power,
        )
        set_camera(plotter, args.elev, args.azim, args.zoom, args.up_axis)
        output_file = case_dir / f"{index:02d}_{safe_filename(name)}.png"
        plotter.screenshot(str(output_file), return_img=False)
        plotter.close()
        print(f"[cut] {output_file}")


def render_overlay(
    output_root: Path,
    panels: list[tuple[str, Path]],
    case: str,
    args: argparse.Namespace,
) -> None:
    plotter = make_plotter(args.window_size, args.background)
    for index, (name, path) in enumerate(panels):
        if name == "GT":
            # A light envelope plus a brighter GT surface makes the reference
            # visible while keeping the predictions transparent.
            color = GT_COLOR
            opacity = args.gt_envelope_opacity
            add_surface(
                plotter,
                path,
                color,
                opacity,
                args.sigma_mm,
                args.smooth_iters,
                args.smooth_passband,
                args.decimate,
                args.pad,
                args.specular,
                args.specular_power,
            )
            add_surface(
                plotter,
                path,
                "#FFFFFF",
                args.gt_surface_opacity,
                args.sigma_mm,
                args.smooth_iters,
                args.smooth_passband,
                args.decimate,
                args.pad,
                args.specular,
                args.specular_power,
            )
        else:
            color = args.surface_color
            add_surface(
                plotter,
                path,
                color,
                args.pred_opacity,
                args.sigma_mm,
                args.smooth_iters,
                args.smooth_passband,
                args.decimate,
                args.pad,
                args.specular,
                args.specular_power,
            )
    set_camera(plotter, args.elev, args.azim, args.zoom, args.up_axis)
    output_file = output_root / f"{safe_filename(case)}_overlay.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_file), return_img=False)
    plotter.close()
    print(f"[overlay] {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--picai_cases", nargs="+", default=DEFAULT_PICAI_CASES)
    parser.add_argument("--ahcdu_cases", nargs="+", default=DEFAULT_AHCDU_CASES)
    parser.add_argument(
        "--picai_gt_dir",
        type=Path,
        default=Path("nnUNet/nnUNet_raw/Dataset141_FullPICAI/labelsTs"),
    )
    parser.add_argument(
        "--ahcdu_gt_dir",
        type=Path,
        default=Path("nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU/labelsTs"),
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("paper_tools/comparative/3dvisual/Mixed_7_3d_comparison_black_vtk"),
    )
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--background", default="white")
    parser.add_argument("--surface_opacity", type=float, default=0.5)
    parser.add_argument("--surface_color", default="#EDC948")
    parser.add_argument("--pred_opacity", type=float, default=0.42)
    parser.add_argument("--gt_envelope_opacity", type=float, default=0.16)
    parser.add_argument("--gt_surface_opacity", type=float, default=0.5)
    parser.add_argument("--sigma_mm", type=float, default=1.0)
    parser.add_argument("--smooth_iters", type=int, default=60)
    parser.add_argument("--smooth_passband", type=float, default=0.1)
    parser.add_argument("--decimate", type=float, default=0.0)
    parser.add_argument("--pad", type=int, default=2)
    parser.add_argument("--specular", type=float, default=0.35)
    parser.add_argument("--specular_power", type=float, default=18.0)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--zoom", type=float, default=1.15)
    parser.add_argument("--up_axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--window_size", type=int, default=1200)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    case_specs = [("PICAI", case) for case in args.picai_cases] + [
        ("AHCDU", case) for case in args.ahcdu_cases
    ]

    for dataset, case in case_specs:
        gt_dir = args.picai_gt_dir if dataset == "PICAI" else args.ahcdu_gt_dir
        panels = load_panel_paths(dataset, case, args.methods, gt_dir)
        print(f"[case] {dataset}/{case}: {len(panels)} panels")
        if args.overlay:
            render_overlay(args.output_root, panels, case, args)
        else:
            render_cut(args.output_root, panels, case, args)


if __name__ == "__main__":
    main()
