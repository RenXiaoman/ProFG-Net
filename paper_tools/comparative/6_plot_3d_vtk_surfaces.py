#!/usr/bin/env python3
"""VTK/PyVista renderer for smooth, shaded 3D segmentation surfaces.

Compared with 5_plot_3d_label_surfaces.py (matplotlib), this version uses
VTK marching cubes plus Phong smooth shading and real translucency, which is
what the "3D reconstruction" panels in the reference figure were made with.

Example:
  python paper_tools/comparative/6_plot_3d_vtk_surfaces.py \
    --mask path/to/pred.nii.gz \
    --labels 1 --names Gland --colors "#C94F45" \
    --sigma_mm 1.0 --smooth_iters 60 \
    --elev 20 --azim -60 --output out.png
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv

pv.OFF_SCREEN = True


@dataclass(frozen=True)
class Layer:
    label: int
    name: str
    color: str
    opacity: float


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def read_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray, tuple[float, float, float]]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing_zyx = tuple(float(v) for v in image.GetSpacing()[::-1])
    direction_xyz = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin_xyz = tuple(float(v) for v in image.GetOrigin())
    return array, spacing_zyx, direction_xyz, origin_xyz


def build_mesh(
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    sigma_vox: tuple[float, float, float] | None,
    taubin_iters: int,
    taubin_passband: float,
    decimate: float,
    pad: int = 2,
    direction_xyz: np.ndarray | None = None,
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    field = mask.astype(np.float32)
    if sigma_vox is not None and float(np.max(sigma_vox)) > 0:
        field = ndimage.gaussian_filter(field, sigma=sigma_vox, mode="constant")

    # SimpleITK returns z, y, x. PyVista/VTK expects x, y, z.
    field_xyz = np.ascontiguousarray(field.transpose(2, 1, 0))
    spacing_xyz = tuple(reversed(spacing_zyx))
    if pad > 0:
        # Zero padding keeps surfaces closed when a mask touches the volume border.
        field_xyz = np.pad(field_xyz, pad, mode="constant")
        origin = tuple(-pad * s for s in spacing_xyz)
    else:
        origin = (0.0, 0.0, 0.0)

    grid = pv.ImageData(
        dimensions=field_xyz.shape,
        spacing=spacing_xyz,
        origin=origin,
    )
    grid.point_data["values"] = field_xyz.ravel(order="F")
    mesh = grid.contour([0.5], scalars="values", method="marching_cubes")

    if taubin_iters > 0:
        mesh = mesh.smooth_taubin(n_iter=taubin_iters, pass_band=taubin_passband)
    if decimate > 0:
        mesh = mesh.decimate(decimate)
    if direction_xyz is not None:
        # Map from the array frame into patient (physical) coordinates.
        mesh.points = mesh.points @ np.asarray(direction_xyz).T + np.asarray(origin_xyz)
    mesh = mesh.compute_normals(cell_normals=False, point_normals=True, consistent_normals=True)
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--labels", type=int, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="+", default=None)
    parser.add_argument("--colors", type=str, nargs="+", default=["#C94F45", "#BFD96B", "#6FA8C8"])
    parser.add_argument("--opacities", type=float, nargs="+", default=[1.0, 1.0, 1.0])
    parser.add_argument("--envelope", action="store_true")
    parser.add_argument("--envelope_color", type=str, default="#9AA7B0")
    parser.add_argument("--envelope_opacity", type=float, default=0.25)
    parser.add_argument("--sigma_mm", type=float, default=1.0)
    parser.add_argument("--smooth_iters", type=int, default=60, help="VTK Taubin iterations")
    parser.add_argument("--smooth_passband", type=float, default=0.1)
    parser.add_argument("--decimate", type=float, default=0.0, help="target reduction ratio, 0 disables")
    parser.add_argument("--pad", type=int, default=2, help="zero padding in voxels to close border surfaces")
    parser.add_argument(
        "--ignore_direction",
        action="store_true",
        help="render in the array frame instead of applying the NIfTI direction matrix",
    )
    parser.add_argument("--specular", type=float, default=0.35)
    parser.add_argument("--specular_power", type=float, default=18.0)
    parser.add_argument("--edges", action="store_true", help="draw thin feature edges")
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--zoom", type=float, default=1.15)
    parser.add_argument(
        "--up_axis",
        choices=("x", "y", "z"),
        default="z",
        help="which physical axis points up in the image, z = superior-inferior",
    )
    parser.add_argument("--parallel", action="store_true", help="parallel instead of perspective projection")
    parser.add_argument("--window_size", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mask, spacing_zyx, direction_xyz, origin_xyz = read_mask(args.mask)
    sigma_vox = tuple(float(args.sigma_mm) / s for s in spacing_zyx) if args.sigma_mm > 0 else None
    print(f"spacing_zyx={spacing_zyx}, sigma_vox={sigma_vox}")
    if args.ignore_direction:
        print("[warn] --ignore_direction: rendering in the array frame")
        direction_xyz = None
        origin_xyz = (0.0, 0.0, 0.0)
    else:
        print(f"origin_xyz={tuple(round(v, 2) for v in origin_xyz)}")
        print("direction_xyz=\n", np.round(direction_xyz, 3))

    names = args.names or [str(label) for label in args.labels]
    if len(names) != len(args.labels):
        raise ValueError("--names must have the same length as --labels")
    layers = [
        Layer(
            label=label,
            name=names[index],
            color=args.colors[index % len(args.colors)],
            opacity=args.opacities[index % len(args.opacities)],
        )
        for index, label in enumerate(args.labels)
    ]

    plotter = pv.Plotter(off_screen=True, window_size=(args.window_size, args.window_size))
    plotter.background_color = "white"

    if args.envelope:
        mesh = build_mesh(
            mask > 0,
            spacing_zyx,
            sigma_vox,
            args.smooth_iters,
            args.smooth_passband,
            args.decimate,
            args.pad,
            direction_xyz,
            origin_xyz,
        )
        plotter.add_mesh(
            mesh,
            color=hex_to_rgb(args.envelope_color),
            opacity=args.envelope_opacity,
            smooth_shading=True,
            specular=args.specular,
            specular_power=args.specular_power,
        )

    for layer in layers:
        selection = mask == layer.label
        if not selection.any():
            print(f"[skip] label {layer.label}: empty")
            continue
        mesh = build_mesh(
            selection,
            spacing_zyx,
            sigma_vox,
            args.smooth_iters,
            args.smooth_passband,
            args.decimate,
            args.pad,
            direction_xyz,
            origin_xyz,
        )
        print(
            f"[label {layer.label} {layer.name}] points={mesh.n_points:,} cells={mesh.n_cells:,}"
        )
        plotter.add_mesh(
            mesh,
            color=hex_to_rgb(layer.color),
            opacity=layer.opacity,
            smooth_shading=True,
            specular=args.specular,
            specular_power=args.specular_power,
            show_edges=False,
        )
        if args.edges:
            plotter.add_mesh(
                mesh.extract_feature_edges(45),
                color=(0.15, 0.15, 0.15),
                line_width=1.0,
                style="wireframe",
            )

    if args.parallel:
        plotter.enable_parallel_projection()

    bounds = np.array(plotter.bounds, dtype=np.float64).reshape(3, 2)
    center = bounds.mean(axis=1)
    radius = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0])) / 2.0
    azimuth = np.deg2rad(args.azim)
    elevation = np.deg2rad(args.elev)
    direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    up = np.array({"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[args.up_axis])
    camera_position = center + direction * radius * 3.0
    plotter.camera_position = [tuple(camera_position), tuple(center), tuple(up)]
    plotter.camera.zoom(args.zoom)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(args.output), return_img=False)
    plotter.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
