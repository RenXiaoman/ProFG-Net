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
    spacing: tuple[float, float, float] = (3.3, 0.5, 0.5)  # z, y, x


DEFAULT_ROOT = Path("nnUNet/nnUNet_raw/Dataset112_Promise12")
SETTINGS = PreprocessingSettings()


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
    label = binarize_label(label)
    return image, label


def collect_cases(source_dir: Path) -> list[str]:
    return sorted(
        path.stem
        for path in source_dir.glob("Case*.mhd")
        if not path.name.endswith("_segmentation.mhd")
    )


def clear_nii_gz(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old_file in directory.glob("*.nii.gz"):
        old_file.unlink()


def process_source_split(
    root: Path,
    source_name: str,
    images_split: str,
    labels_split: str,
    settings: PreprocessingSettings,
) -> int:
    source_dir = root / source_name
    output_images_dir = root / images_split
    output_labels_dir = root / labels_split
    clear_nii_gz(output_images_dir)
    clear_nii_gz(output_labels_dir)

    cases = collect_cases(source_dir)
    for case in tqdm(cases, desc=source_name):
        image_path = source_dir / f"{case}.mhd"
        label_path = source_dir / f"{case}_segmentation.mhd"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing segmentation for {case}: {label_path}")

        case_id = f"Promise12_{case[4:]}"
        image, label = preprocess_pair(image_path, label_path, settings)
        sitk.WriteImage(image, str(output_images_dir / f"{case_id}_0000.nii.gz"))
        sitk.WriteImage(label, str(output_labels_dir / f"{case_id}.nii.gz"))

    return len(cases)


def write_dataset_json(root: Path, num_training: int) -> None:
    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "Prostate": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PROMISE12 mhd/raw files to nnU-Net nii.gz format with fixed spacing and matrix size."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--matrix-size", nargs=3, type=int, default=list(SETTINGS.matrix_size))
    parser.add_argument("--spacing", nargs=3, type=float, default=list(SETTINGS.spacing))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = PreprocessingSettings(
        matrix_size=tuple(args.matrix_size),
        spacing=tuple(args.spacing),
    )
    print(f"Root: {args.root}")
    print(f"Target matrix_size (z,y,x): {settings.matrix_size}")
    print(f"Target spacing (z,y,x): {settings.spacing}")
    print("Output format: nnU-Net .nii.gz")

    num_training = process_source_split(
        args.root,
        source_name="training_data",
        images_split="imagesTr",
        labels_split="labelsTr",
        settings=settings,
    )
    num_test = process_source_split(
        args.root,
        source_name="test_data",
        images_split="imagesTs",
        labels_split="labelsTs",
        settings=settings,
    )
    write_dataset_json(args.root, num_training)
    print(f"Processed training cases: {num_training}")
    print(f"Processed test cases: {num_test}")
    print(f"Wrote: {args.root / 'dataset.json'}")


if __name__ == "__main__":
    main()
