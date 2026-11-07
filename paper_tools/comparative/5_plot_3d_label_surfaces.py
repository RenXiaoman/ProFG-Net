#!/usr/bin/env python3
"""Render shaded 3D surfaces from a (multi-label) segmentation mask.

This is the label-aware counterpart of 4_plot_3d_comparison.py: instead of one
single-color surface per panel, each label of one mask gets its own color and
an optional translucent envelope can be drawn around the whole object.

Example:
  python paper_tools/comparative/5_plot_3d_label_surfaces.py \
    --mask path/to/pred.nii.gz \
    --labels 1 2 \
    --names PZ TZ \
    --colors "#BFD96B" "#6FA8C8" \
    --envelope \
    --output out.png
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage, sparse
from skimage import measure


@dataclass(frozen=True)
class Surface:
    name: str
    verts: np.ndarray  # z, y, x order, already scaled by spacing
    faces: np.ndarray
    color: tuple[float, float, float]
    alpha: float


def read_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing_zyx = tuple(float(v) for v in image.GetSpacing()[::-1])
    return array, spacing_zyx


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def build_surface(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    step_size: int,
    sigma_vox: tuple[float, float, float] | None = None,
    smooth_iters: int = 0,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not mask.any():
        return None
    field = mask.astype(np.float32)
    if sigma_vox is not None and float(np.max(sigma_vox)) > 0:
        field = ndimage.gaussian_filter(field, sigma=sigma_vox, mode="constant")
    padded = np.pad(field, 1, mode="constant")
    verts, faces, _, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=spacing,
        step_size=step_size,
        allow_degenerate=False,
    )
    verts = verts - np.asarray(spacing, dtype=np.float64)
    if smooth_iters > 0:
        verts = taubin_smooth(verts, faces, iterations=smooth_iters)
    return verts, faces


def _vertex_adjacency(n_verts: int, faces: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(rows.shape[0], dtype=np.float64)
    adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n_verts, n_verts)).tocsr()
    adjacency.data[:] = 1.0
    adjacency.sum_duplicates()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    return adjacency, degree


def taubin_smooth(
    verts: np.ndarray,
    faces: np.ndarray,
    iterations: int = 8,
    lam: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    """Alternating shrink/expand smoothing that removes marching-cubes staircase."""
    adjacency, degree = _vertex_adjacency(len(verts), faces)
    smoothed = verts.astype(np.float64)
    for _ in range(iterations):
        lap = adjacency @ smoothed / degree[:, None] - smoothed
        smoothed = smoothed + lam * lap
        lap = adjacency @ smoothed / degree[:, None] - smoothed
        smoothed = smoothed + mu * lap
    return smoothed


def face_shading(
    xyz: np.ndarray,
    faces: np.ndarray,
    specular: float = 0.35,
    shininess: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face diffuse and specular terms, with normals oriented outward."""
    tri = xyz[faces]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths

    centers = tri.mean(axis=1)
    mesh_center = xyz.mean(axis=0)
    outward = centers - mesh_center
    flip = (normals * outward).sum(axis=1) < 0
    normals[flip] *= -1.0

    light = np.array([0.35, -0.55, 0.75])
    light = light / np.linalg.norm(light)
    view = np.array([0.0, 0.0, 1.0])
    halfway = light + view
    halfway = halfway / np.linalg.norm(halfway)

    diffuse = 0.42 + 0.58 * np.clip(normals @ light, 0.0, 1.0)
    specular_term = specular * np.clip(normals @ halfway, 0.0, 1.0) ** shininess
    return diffuse, specular_term


def add_surface(
    ax: plt.Axes,
    surface: Surface,
    specular: float = 0.35,
    shininess: float = 24.0,
) -> None:
    # marching_cubes returns z, y, x. Matplotlib draws x, y, z.
    xyz = surface.verts[:, [2, 1, 0]]
    diffuse, spec = face_shading(xyz, surface.faces, specular=specular, shininess=shininess)
    rgba = np.empty((len(surface.faces), 4), dtype=np.float64)
    rgb = np.asarray(surface.color)[None, :] * diffuse[:, None] + spec[:, None]
    rgba[:, :3] = np.clip(rgb, 0.0, 1.0)
    rgba[:, 3] = surface.alpha
    mesh = Poly3DCollection(
        xyz[surface.faces],
        facecolors=rgba,
        edgecolors=(0.15, 0.15, 0.15, 0.06),
        linewidths=0.05,
    )
    ax.add_collection3d(mesh)


def set_equal_axes(ax: plt.Axes, surfaces: list[Surface]) -> None:
    verts = np.concatenate([surface.verts for surface in surfaces], axis=0)[:, [2, 1, 0]]
    mins, maxs = verts.min(axis=0), verts.max(axis=0)
    centers = (mins + maxs) / 2
    radius = max(float((maxs - mins).max() / 2), 1.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True, help="multi-label mask (.nii.gz)")
    parser.add_argument("--labels", type=int, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="+", default=None)
    parser.add_argument("--colors", type=str, nargs="+", default=["#D9483B", "#BFD96B", "#6FA8C8"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.95, 0.90, 0.85])
    parser.add_argument("--envelope", action="store_true", help="draw mask>0 as a translucent shell")
    parser.add_argument("--envelope_color", type=str, default="#9AA7B0")
    parser.add_argument("--envelope_alpha", type=float, default=0.22)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument(
        "--sigma_mm",
        type=float,
        default=1.0,
        help="isotropic gaussian smoothing of the mask in millimetres before meshing, 0 disables",
    )
    parser.add_argument(
        "--smooth_iters",
        type=int,
        default=8,
        help="Taubin mesh smoothing iterations, 0 disables",
    )
    parser.add_argument("--specular", type=float, default=0.35)
    parser.add_argument("--shininess", type=float, default=24.0)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument("--figsize", type=float, default=4.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mask, spacing = read_mask(args.mask)
    if args.sigma_mm > 0:
        sigma_vox = tuple(float(args.sigma_mm) / s for s in spacing)
    else:
        sigma_vox = None
    print(f"spacing_zyx={spacing}, sigma_vox={sigma_vox}, smooth_iters={args.smooth_iters}")
    names = args.names or [str(label) for label in args.labels]
    if len(names) != len(args.labels):
        raise ValueError("--names must have the same length as --labels")

    surfaces: list[Surface] = []
    if args.envelope:
        built = build_surface(mask > 0, spacing, args.step_size, sigma_vox, args.smooth_iters)
        if built is not None:
            surfaces.append(
                Surface(
                    name="envelope",
                    verts=built[0],
                    faces=built[1],
                    color=hex_to_rgb(args.envelope_color),
                    alpha=args.envelope_alpha,
                )
            )

    for index, label in enumerate(args.labels):
        built = build_surface(mask == label, spacing, args.step_size, sigma_vox, args.smooth_iters)
        if built is None:
            print(f"[skip] label {label}: empty")
            continue
        surfaces.append(
            Surface(
                name=names[index],
                verts=built[0],
                faces=built[1],
                color=hex_to_rgb(args.colors[index % len(args.colors)]),
                alpha=args.alphas[index % len(args.alphas)],
            )
        )
        print(f"[label {label} {names[index]}] verts={len(built[0]):,} faces={len(built[1]):,}")

    if not surfaces:
        raise SystemExit("No surface to draw")

    fig = plt.figure(figsize=(args.figsize, args.figsize), dpi=args.dpi, facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for surface in surfaces:
        add_surface(ax, surface, specular=args.specular, shininess=args.shininess)
    set_equal_axes(ax, surfaces)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
