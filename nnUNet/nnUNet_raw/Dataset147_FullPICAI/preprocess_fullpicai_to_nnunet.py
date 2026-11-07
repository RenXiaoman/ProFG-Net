from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


@dataclass(frozen=True)
class PreprocessingSettings:
    matrix_size: tuple[int, int, int] = (16, 256, 256)  # z, y, x
    spacing: tuple[float, float, float] = (3.5, 0.5, 0.5)  # z, y, x


DEFAULT_ROOT = Path("nnUNet/nnUNet_raw/Dataset147_FullPICAI")
DEFAULT_IMAGE_ROOT = Path("/home/ikun_server/clib/PycharmProjects/Z-SSMNet/src/workdir/input/images")
DEFAULT_LABEL_ROOT = Path(
    "/home/ikun_server/clib/PycharmProjects/Z-SSMNet/src/workdir/input/labels/"
    "anatomical_delineations/whole_gland/AI/Bosma22b"
)
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
            upper_crop.append(0)
            upper_pad.append(0)
            upper_crop[-1] = upper
            lower_pad.append(0)
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


def case_id_from_label(label_path: Path) -> str:
    return label_path.name.removesuffix(".nii.gz")


def image_path_for_case(image_root: Path, case_id: str) -> Path:
    patient_id = case_id.split("_", maxsplit=1)[0]
    return image_root / patient_id / f"{case_id}_t2w.mha"


def collect_cases(
    image_root: Path,
    label_root: Path,
    skip_missing: bool,
) -> tuple[list[tuple[str, Path, Path]], list[dict[str, str]]]:
    label_files = sorted(label_root.glob("*.nii.gz"))
    cases = []
    missing_images: list[dict[str, str]] = []

    for label_path in label_files:
        case_id = case_id_from_label(label_path)
        image_path = image_path_for_case(image_root, case_id)
        if not image_path.is_file():
            missing_images.append({"case": case_id, "expected_image": str(image_path)})
            continue
        cases.append((case_id, image_path, label_path))

    if missing_images and not skip_missing:
        preview = "\n".join(f"  {i['case']}: {i['expected_image']}" for i in missing_images[:20])
        raise FileNotFoundError(
            f"Missing T2WI images for {len(missing_images)} labels. First missing entries:\n{preview}"
        )

    if not cases:
        raise RuntimeError(f"No paired cases found from image_root={image_root} and label_root={label_root}")

    return cases, missing_images


def clear_nii_gz(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old_file in directory.glob("*.nii.gz"):
        old_file.unlink()


def write_dataset_json(root: Path, num_training: int) -> None:
    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "Prostate": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")


def process_one_case(
    case_id: str,
    image_path: Path,
    label_path: Path,
    images_tr: Path,
    labels_tr: Path,
    settings: PreprocessingSettings,
    itk_threads: int,
) -> str:
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(itk_threads)
    image, label = preprocess_pair(image_path, label_path, settings)
    sitk.WriteImage(image, str(images_tr / f"{case_id}_0000.nii.gz"))
    sitk.WriteImage(label, str(labels_tr / f"{case_id}.nii.gz"))
    return case_id


def process_cases(
    root: Path,
    image_root: Path,
    label_root: Path,
    settings: PreprocessingSettings,
    skip_missing: bool,
    num_workers: int,
    itk_threads: int,
) -> int:
    images_tr = root / "imagesTr"
    labels_tr = root / "labelsTr"
    clear_nii_gz(images_tr)
    clear_nii_gz(labels_tr)

    cases, missing_images = collect_cases(image_root, label_root, skip_missing=skip_missing)
    if missing_images:
        (root / "missing_images.json").write_text(json.dumps(missing_images, indent=2) + "\n")
        print(f"Skipped missing T2WI cases: {len(missing_images)}")
        print(f"Wrote: {root / 'missing_images.json'}")
    if num_workers <= 1:
        for case_id, image_path, label_path in tqdm(cases, desc="FullPICAI"):
            process_one_case(case_id, image_path, label_path, images_tr, labels_tr, settings, itk_threads)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    process_one_case,
                    case_id,
                    image_path,
                    label_path,
                    images_tr,
                    labels_tr,
                    settings,
                    itk_threads,
                )
                for case_id, image_path, label_path in cases
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"FullPICAI x{num_workers}"):
                future.result()

    return len(cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Full PI-CAI T2WI and Bosma22b whole-gland labels to nnU-Net format."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--matrix-size", nargs=3, type=int, default=list(SETTINGS.matrix_size))
    parser.add_argument("--spacing", nargs=3, type=float, default=list(SETTINGS.spacing))
    parser.add_argument("--skip-missing", action="store_true", help="Skip labels without a matching T2WI image.")
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--itk-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = PreprocessingSettings(
        matrix_size=tuple(args.matrix_size),
        spacing=tuple(args.spacing),
    )

    print(f"Root: {args.root}")
    print(f"Image root: {args.image_root}")
    print(f"Label root: {args.label_root}")
    print(f"Target matrix_size (z,y,x): {settings.matrix_size}")
    print(f"Target spacing (z,y,x): {settings.spacing}")
    print(f"num_workers: {args.num_workers}")
    print(f"itk_threads per worker: {args.itk_threads}")
    print("Output format: nnU-Net .nii.gz")

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(args.itk_threads)
    num_training = process_cases(
        args.root,
        args.image_root,
        args.label_root,
        settings,
        skip_missing=args.skip_missing,
        num_workers=args.num_workers,
        itk_threads=args.itk_threads,
    )
    write_dataset_json(args.root, num_training)
    print(f"Processed training cases: {num_training}")
    print(f"Wrote: {args.root / 'dataset.json'}")


if __name__ == "__main__":
    main()
