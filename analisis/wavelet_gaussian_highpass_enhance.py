from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import ptwt


BAND_ORDER = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")


def robust_gray_range(arr: np.ndarray) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        hi = lo + 1e-6
    return float(lo), float(hi)


def robust_signed_abs(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    vmax = np.percentile(np.abs(finite), 99)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    return float(vmax if vmax > 0 else 1.0)


def gaussian_kernel3d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    padding = kernel_size // 2
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - padding
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size, kernel_size)


def clear_pngs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for png in path.glob("*.png"):
        png.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ptwt wavedec3 subbands after Gaussian high-pass enhancement."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "nnUNet/nnUNet_raw/Dataset131_ProstatePI-CAI/imagesTr/11095_1001118_0000.nii.gz"
        ),
    )
    parser.add_argument("--case-name", default="11095_1001118")
    parser.add_argument("--output", type=Path, default=Path("analisis/11095_enhance_alpha2"))
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--wavelet", default="db1")
    parser.add_argument("--mode", default="zero")
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show enhanced-original differences for LLL and all 7 high-frequency subbands.",
    )
    parser.add_argument(
        "--sum-high",
        action="store_true",
        help="Show image, LLL, and the direct sum of all 7 displayed high-frequency subbands.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kernel_size % 2 != 1:
        raise ValueError(f"kernel-size must be odd, got {args.kernel_size}.")

    out_dir = args.output
    recon_dir = out_dir / "reconstruction"
    clear_pngs(out_dir)
    clear_pngs(recon_dir)

    img = sitk.GetArrayFromImage(sitk.ReadImage(str(args.image))).astype(np.float32)
    x = torch.from_numpy(img)[None, None]
    coeffs = ptwt.wavedec3(x, wavelet=args.wavelet, level=1, mode=args.mode)
    lll = coeffs[0]
    detail = coeffs[1]

    yh = torch.stack([detail[band] for band in BAND_ORDER], dim=1)
    b, subbands, c, d, h, w = yh.shape
    yh = yh.reshape(b, subbands * c, d, h, w)

    padding = args.kernel_size // 2
    kernel = gaussian_kernel3d(args.kernel_size, args.sigma, yh.device, yh.dtype)
    weight = kernel.repeat(yh.shape[1], 1, 1, 1, 1)
    yh_pad = F.pad(
        yh,
        (padding, padding, padding, padding, padding, padding),
        mode="replicate",
    )
    yh_low = F.conv3d(yh_pad, weight, groups=yh.shape[1])
    yh_high = yh - yh_low
    yh_enh = yh + args.alpha * yh_high

    yh_enh_bands = yh_enh.reshape(b, subbands, c, d, h, w)
    detail_enh = {band: yh_enh_bands[:, idx] for idx, band in enumerate(BAND_ORDER)}
    x_rec = ptwt.waverec3((lll, detail_enh), wavelet=args.wavelet)

    lll_np = lll[0, 0].detach().cpu().numpy()
    yh_np = {
        band: detail[band][0, 0].detach().cpu().numpy() for band in BAND_ORDER
    }
    yh_enh_np = {
        band: detail_enh[band][0, 0].detach().cpu().numpy() for band in BAND_ORDER
    }
    lll_display_np = np.zeros_like(lll_np) if args.diff else lll_np
    yh_display_np = {
        band: yh_enh_np[band] - yh_np[band] if args.diff else yh_enh_np[band]
        for band in BAND_ORDER
    }
    rec_np = x_rec[0, 0].detach().cpu().numpy()[: img.shape[0], : img.shape[1], : img.shape[2]]

    print("image:", tuple(x.shape))
    print("LLL:", tuple(lll.shape))
    print("Yh_enh:", tuple(yh_enh.shape))
    print("bands:", ",".join(BAND_ORDER))
    print("reconstruction:", rec_np.shape)
    print("alpha:", args.alpha, "sigma:", args.sigma, "kernel_size:", args.kernel_size)

    img_vmin, img_vmax = robust_gray_range(img)
    if args.diff:
        lll_vmax = robust_signed_abs(lll_display_np)
        lll_vmin = -lll_vmax
    else:
        lll_vmin, lll_vmax = robust_gray_range(lll_display_np)
    band_vmax = {band: robust_signed_abs(yh_display_np[band]) for band in BAND_ORDER}
    sum_high_np = np.sum(
        np.stack([yh_display_np[band] for band in BAND_ORDER], axis=0),
        axis=0,
    )
    sum_high_vmax = robust_signed_abs(sum_high_np)

    for z in range(img.shape[0]):
        coeff_z = min(z // 2, lll_display_np.shape[0] - 1)
        lll_title = "LLL_diff" if args.diff else "LLL"
        if args.sum_high:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            suffix = "diff" if args.diff else "enh"
            tiles = [
                ("image", img[z], img_vmin, img_vmax),
                (lll_title, lll_display_np[coeff_z], lll_vmin, lll_vmax),
                (f"sum_high_{suffix}", sum_high_np[coeff_z], -sum_high_vmax, sum_high_vmax),
            ]
            for ax, (title, arr, vmin, vmax) in zip(axes, tiles):
                ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=13)
                ax.axis("off")
            fig.suptitle(
                (
                    f"{args.case_name} slice {z:02d} | sum of 7 high-frequency "
                    f"{suffix} subbands alpha={args.alpha} | coeff slice {coeff_z:02d}"
                ),
                fontsize=12,
            )
        else:
            fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
            tiles = [
                ("image", img[z], img_vmin, img_vmax),
                (lll_title, lll_display_np[coeff_z], lll_vmin, lll_vmax),
            ]
            for band in BAND_ORDER:
                vmax = band_vmax[band]
                suffix = "diff" if args.diff else "enh"
                tiles.append((f"{band}_{suffix}", yh_display_np[band][coeff_z], -vmax, vmax))

            for ax, (title, arr, vmin, vmax) in zip(axes.flat, tiles):
                ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=13)
                ax.axis("off")
            fig.suptitle(
                (
                    f"{args.case_name} slice {z:02d} | Gaussian high-pass "
                    f"{'difference' if args.diff else 'enhanced subbands'} "
                    f"alpha={args.alpha} | coeff slice {coeff_z:02d}"
                ),
                fontsize=12,
            )
        fig.savefig(out_dir / f"{z:02d}.png", dpi=160)
        plt.close(fig)

        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        for ax, title, arr in [
            (axes2[0], "image", img[z]),
            (axes2[1], f"enhanced_alpha{args.alpha:g}", rec_np[z]),
        ]:
            ax.imshow(arr, cmap="gray", vmin=img_vmin, vmax=img_vmax)
            ax.set_title(title, fontsize=13)
            ax.axis("off")
        fig2.suptitle(
            f"{args.case_name} slice {z:02d} | reconstructed enhanced image alpha={args.alpha}",
            fontsize=12,
        )
        fig2.savefig(recon_dir / f"{z:02d}.png", dpi=160)
        plt.close(fig2)

    print("saved:", out_dir.resolve())
    print("main_png_count:", len(list(out_dir.glob("*.png"))))
    print("reconstruction_png_count:", len(list(recon_dir.glob("*.png"))))


if __name__ == "__main__":
    main()
