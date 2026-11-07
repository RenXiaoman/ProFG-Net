from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


@dataclass(frozen=True)
class PreprocessingSettings:
    matrix_size: tuple[int, int, int] = (16, 256, 256)  # z, y, x
    spacing: tuple[float, float, float] = (3.0, 0.4, 0.4)  # z, y, x


DEFAULT_ROOT = Path("nnUNet/nnUNet_raw/Dataset158_prostate158")
DEFAULT_OUTPUT = DEFAULT_ROOT / "preprocessed_3x04x04_16x256x256"
SETTINGS = PreprocessingSettings()


def collect_case_ids(images_dir: Path) -> list[str]:
    return sorted(path.name[:-12] for path in images_dir.glob("*_0000.nii.gz"))


def sitk_size_from_zyx(size_zyx: Iterable[int]) -> list[int]:
    return list(size_zyx)[::-1]


def sitk_spacing_from_zyx(spacing_zyx: Iterable[float]) -> list[float]:
    return list(spacing_zyx)[::-1]


def resample_to_spacing(
    image: sitk.Image,
    spacing_zyx: Iterable[float],
    interpolator: int,
    default_value: float = 0.0,
) -> sitk.Image:
    spacing_xyz = sitk_spacing_from_zyx(spacing_zyx)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    out_size = [
        int(round(size * spacing_in / spacing_out))
        for size, spacing_in, spacing_out in zip(original_size, original_spacing, spacing_xyz)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing_xyz)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image)


def center_crop_or_pad(
    image: sitk.Image,
    target_size_zyx: Iterable[int],
    pad_value: float = 0.0,
) -> sitk.Image:
    target_size_xyz = sitk_size_from_zyx(target_size_zyx)
    current_size = image.GetSize()
    lower_crop = []
    upper_crop = []
    lower_pad = []
    upper_pad = []

    for current, target in zip(current_size, target_size_xyz):
        if current > target:
            diff = current - target
            lower = diff // 2
            upper = diff - lower
            lower_crop.append(lower)
            upper_crop.append(upper)
            lower_pad.append(0)
            upper_pad.append(0)
        else:
            diff = target - current
            lower = diff // 2
            upper = diff - lower
            lower_crop.append(0)
            upper_crop.append(0)
            lower_pad.append(lower)
            upper_pad.append(upper)

    if any(lower_crop) or any(upper_crop):
        image = sitk.Crop(image, lower_crop, upper_crop)
    if any(lower_pad) or any(upper_pad):
        image = sitk.ConstantPad(image, lower_pad, upper_pad, pad_value)
    return image


def binarize_label(label_image: sitk.Image) -> sitk.Image:
    label_array = sitk.GetArrayFromImage(label_image)
    label_array = (label_array > 0).astype(np.uint8)
    output = sitk.GetImageFromArray(label_array)
    output.CopyInformation(label_image)
    return output


def preprocess_pair(
    image_path: Path,
    label_path: Path,
    settings: PreprocessingSettings,
    binarize: bool,
) -> tuple[sitk.Image, sitk.Image]:
    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    label = sitk.ReadImage(str(label_path), sitk.sitkUInt8)

    image = resample_to_spacing(
        image,
        settings.spacing,
        interpolator=sitk.sitkBSpline,
        default_value=0.0,
    )
    label = resample_to_spacing(
        label,
        settings.spacing,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
    )

    image = center_crop_or_pad(image, settings.matrix_size, pad_value=0.0)
    label = center_crop_or_pad(label, settings.matrix_size, pad_value=0)

    label.SetOrigin(image.GetOrigin())
    label.SetSpacing(image.GetSpacing())
    label.SetDirection(image.GetDirection())

    if binarize:
        label = binarize_label(label)

    return image, label


def process_split(
    root: Path,
    output_root: Path,
    split: str,
    settings: PreprocessingSettings,
    binarize: bool,
) -> int:
    images_dir = root / f"images{split}"
    labels_dir = root / f"labels{split}"
    output_images_dir = output_root / f"images{split}"
    output_labels_dir = output_root / f"labels{split}"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_images_dir.glob("*.nii.gz"):
        old_file.unlink()
    for old_file in output_labels_dir.glob("*.nii.gz"):
        old_file.unlink()

    case_ids = collect_case_ids(images_dir)
    for case_id in tqdm(case_ids, desc=f"{split} cases"):
        image_path = images_dir / f"{case_id}_0000.nii.gz"
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {case_id}: {label_path}")

        image, label = preprocess_pair(image_path, label_path, settings, binarize)
        sitk.WriteImage(image, str(output_images_dir / f"{case_id}_0000.nii.gz"))
        sitk.WriteImage(label, str(output_labels_dir / f"{case_id}.nii.gz"))

    return len(case_ids)


def write_dataset_json(
    root: Path,
    output_root: Path,
    num_training: int,
    binarize: bool,
) -> None:
    source_json = root / "dataset.json"
    if source_json.exists():
        dataset_json = json.loads(source_json.read_text())
    else:
        dataset_json = {
            "channel_names": {"0": "T2"},
            "labels": {"background": 0, "Prostate": 1},
            "file_ending": ".nii.gz",
        }

    dataset_json["numTraining"] = num_training
    if binarize:
        dataset_json["labels"] = {"background": 0, "Prostate": 1}

    (output_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Dataset158 T2W and gland labels to fixed spacing and matrix size."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--matrix-size", nargs=3, type=int, default=list(SETTINGS.matrix_size))
    parser.add_argument("--spacing", nargs=3, type=float, default=list(SETTINGS.spacing))
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Binarize labels to gland foreground. By default, original PZ/TZ labels are preserved.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Tr", "Ts"],
        choices=["Tr", "Ts"],
        help="nnU-Net splits to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = PreprocessingSettings(
        matrix_size=tuple(args.matrix_size),
        spacing=tuple(args.spacing),
    )
    binarize = args.binarize

    print(f"Input root: {args.root}")
    print(f"Output root: {args.output}")
    print(f"Target matrix_size (z,y,x): {settings.matrix_size}")
    print(f"Target spacing (z,y,x): {settings.spacing}")
    print(f"Label mode: {'binarize gland labels' if binarize else 'keep PZ/TZ zones'}")

    counts = {}
    for split in args.splits:
        counts[split] = process_split(args.root, args.output, split, settings, binarize)

    write_dataset_json(args.root, args.output, counts.get("Tr", 0), binarize)
    print(f"Processed counts: {counts}")
    print(f"Wrote: {args.output / 'dataset.json'}")


if __name__ == "__main__":
    main()
