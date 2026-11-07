from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def read_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def normalize_slice(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, (1, 99))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, lo, hi)
    return (image - lo) / (hi - lo)


def find_cases(images_dir: Path) -> list[str]:
    cases = []
    for path in sorted(images_dir.glob("*_0000.nii.gz")):
        cases.append(path.name.removesuffix("_0000.nii.gz"))
    return cases


def find_detection_path(detection_dir: Path, case: str) -> Path | None:
    candidates = [
        detection_dir / f"{case}_detection_map.nii.gz",
        detection_dir / f"{case}_softmax.nii.gz",
        detection_dir / f"{case}.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def default_gt_dir(images_dir: Path) -> Path:
    dataset_dir = images_dir.parent
    return dataset_dir / "labelsTr"


def choose_slice(gt: np.ndarray, detection: np.ndarray) -> int:
    gt_score = gt.reshape(gt.shape[0], -1).sum(axis=1)
    if gt_score.max() > 0:
        return int(np.argmax(gt_score))
    det_score = detection.reshape(detection.shape[0], -1).sum(axis=1)
    if det_score.max() > 0:
        return int(np.argmax(det_score))
    return gt.shape[0] // 2


def draw_overlay(ax, gt_slice: np.ndarray, detection_slice: np.ndarray | None = None) -> None:
    if detection_slice is not None and np.max(detection_slice) > 0:
        ax.imshow(detection_slice, cmap="magma", alpha=0.45)
    if np.max(gt_slice) > 0:
        ax.contour(gt_slice > 0, levels=[0.5], colors="yellow", linewidths=1.8, linestyles="--")


def plot_slice(
    t2w: np.ndarray,
    dwi: np.ndarray,
    adc: np.ndarray,
    gt: np.ndarray,
    detection: np.ndarray,
    detection_name: str,
    case: str,
    z: int,
    output_path: Path,
) -> None:
    panels = [
        ("T2W", normalize_slice(t2w[z])),
        ("DWI", normalize_slice(dwi[z])),
        ("ADC", normalize_slice(adc[z])),
        ("Detection", np.clip(detection[z].astype(np.float32), 0, 1)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
    detection_image = None
    for ax, (title, image) in zip(axes, panels):
        if title == "Detection":
            detection_image = ax.imshow(image, cmap="jet", vmin=0, vmax=1)
            draw_overlay(ax, gt[z])
        else:
            ax.imshow(image, cmap="gray")
            draw_overlay(ax, gt[z])
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    if detection_image is not None:
        colorbar = fig.colorbar(detection_image, ax=axes[3], fraction=0.046, pad=0.02)
        colorbar.set_ticks(np.linspace(0, 1, 6))
        colorbar.ax.tick_params(labelsize=11)

    fig.suptitle(f"{case} | slice {z} | {detection_name}", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def visualize_case(images_dir: Path, detection_dir: Path, output_dir: Path, gt_dir: Path, case: str) -> list[Path]:
    t2w = read_array(images_dir / f"{case}_0000.nii.gz")
    adc = read_array(images_dir / f"{case}_0001.nii.gz")
    dwi = read_array(images_dir / f"{case}_0002.nii.gz")
    gt = read_array(gt_dir / f"{case}.nii.gz")

    detection_path = find_detection_path(detection_dir, case)
    if detection_path is None:
        raise FileNotFoundError(f"No detection map found for case {case} in {detection_dir}")
    detection = read_array(detection_path)

    if not (t2w.shape == dwi.shape == adc.shape == gt.shape == detection.shape):
        raise ValueError(
            f"Shape mismatch for {case}: "
            f"T2W={t2w.shape}, DWI={dwi.shape}, ADC={adc.shape}, GT={gt.shape}, detection={detection.shape}"
        )

    case_dir = output_dir / case
    output_paths = []
    for z in range(t2w.shape[0]):
        output_path = case_dir / f"slice_{z:03d}.png"
        plot_slice(t2w, dwi, adc, gt, detection, detection_path.name, case, z, output_path)
        output_paths.append(output_path)
    return output_paths


def pick_case(images_dir: Path, detection_dir: Path, gt_dir: Path, requested_case: str | None) -> str:
    if requested_case is not None:
        return requested_case

    for case in find_cases(images_dir):
        if find_detection_path(detection_dir, case) is None:
            continue
        gt_path = gt_dir / f"{case}.nii.gz"
        if not gt_path.exists():
            continue
        gt = read_array(gt_path)
        if np.any(gt > 0):
            return case

    cases = find_cases(images_dir)
    if not cases:
        raise RuntimeError(f"No cases found in {images_dir}")
    return cases[0]


def visualize_case_job(args: tuple[Path, Path, Path, Path, str]) -> tuple[str, int, str | None]:
    images_dir, detection_dir, output_dir, gt_dir, case = args
    try:
        output_paths = visualize_case(images_dir, detection_dir, output_dir, gt_dir, case)
    except Exception as exc:
        return case, 0, str(exc)
    return case, len(output_paths), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("detection_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gt_dir", type=Path, default=None)
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=20)
    args = parser.parse_args()

    gt_dir = args.gt_dir if args.gt_dir is not None else default_gt_dir(args.images_dir)
    cases = [args.case] if args.case is not None else [
        case for case in find_cases(args.images_dir)
        if find_detection_path(args.detection_dir, case) is not None and (gt_dir / f"{case}.nii.gz").exists()
    ]
    if not cases:
        case = pick_case(args.images_dir, args.detection_dir, gt_dir, None)
        cases = [case]

    jobs = [(args.images_dir, args.detection_dir, args.output_dir, gt_dir, case) for case in cases]
    total = 0
    errors = []
    if args.num_workers <= 1 or len(jobs) == 1:
        iterator = (visualize_case_job(job) for job in jobs)
        for case, count, error in tqdm(iterator, total=len(jobs), desc="Visualizing cases"):
            if error is not None:
                errors.append((case, error))
                continue
            total += count
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for case, count, error in tqdm(executor.map(visualize_case_job, jobs), total=len(jobs), desc="Visualizing cases"):
                if error is not None:
                    errors.append((case, error))
                    continue
                total += count

    print(f"Done. Wrote {total} png files.")
    if errors:
        print(f"Failed {len(errors)} cases:")
        for case, error in errors[:20]:
            print(f"{case}: {error}")


if __name__ == "__main__":
    main()
