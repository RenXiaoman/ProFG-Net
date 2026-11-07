from __future__ import annotations

import json
import random
import shutil
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "Dataset147_FullPICAI"
BASE_DATASET = Path(__file__).resolve().parents[1] / "Dataset131_ProstatePI-CAI"
DESTINATION = Path(__file__).resolve().parent
SEED = 141
NUM_TRAIN = 1199


def main() -> None:
    source_images = SOURCE / "imagesTr"
    source_labels = SOURCE / "labelsTr"
    cases = sorted(path.name.removesuffix(".nii.gz") for path in source_labels.glob("*.nii.gz"))

    if len(cases) != 1499:
        raise RuntimeError(f"Expected 1499 source cases, found {len(cases)}")

    for case_id in cases:
        image = source_images / f"{case_id}_0000.nii.gz"
        if not image.is_file():
            raise FileNotFoundError(f"Missing source image: {image}")

    source_cases = set(cases)
    base_train = {
        path.name.removesuffix(".nii.gz") for path in (BASE_DATASET / "labelsTr").glob("*.nii.gz")
    }
    base_val = {
        path.name.removesuffix(".nii.gz") for path in (BASE_DATASET / "labelsTs").glob("*.nii.gz")
    }
    if base_train & base_val:
        raise RuntimeError("Dataset131 train and val cases overlap")
    if not (base_train | base_val) <= source_cases:
        missing = sorted((base_train | base_val) - source_cases)
        raise RuntimeError(f"Dataset131 contains cases missing from Dataset147: {missing[:10]}")

    added_cases = sorted(source_cases - base_train - base_val)
    random.Random(SEED).shuffle(added_cases)
    num_added_train = NUM_TRAIN - len(base_train)
    train_cases = sorted(base_train | set(added_cases[:num_added_train]))
    val_cases = sorted(base_val | set(added_cases[num_added_train:]))

    if len(train_cases) != NUM_TRAIN or len(val_cases) != len(cases) - NUM_TRAIN:
        raise RuntimeError(f"Unexpected split sizes: train={len(train_cases)}, val={len(val_cases)}")
    if not base_train <= set(train_cases) or not base_val <= set(val_cases):
        raise RuntimeError("Dataset131 train/val membership was not preserved")

    output_dirs = {
        "train_images": DESTINATION / "imagesTr",
        "train_labels": DESTINATION / "labelsTr",
        "val_images": DESTINATION / "imagesTs",
        "val_labels": DESTINATION / "labelsTs",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
        for old_file in directory.glob("*.nii.gz"):
            old_file.unlink()

    # nnU-Net only preprocesses imagesTr/labelsTr, so the training pool must contain
    # both split groups. Keep a second copy of validation cases in imagesTs/labelsTs
    # for standalone inference and evaluation.
    for case_id in sorted(train_cases + val_cases):
        shutil.copy2(source_images / f"{case_id}_0000.nii.gz", output_dirs["train_images"])
        shutil.copy2(source_labels / f"{case_id}.nii.gz", output_dirs["train_labels"])
    for case_id in val_cases:
        shutil.copy2(source_images / f"{case_id}_0000.nii.gz", output_dirs["val_images"])
        shutil.copy2(source_labels / f"{case_id}.nii.gz", output_dirs["val_labels"])

    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "Prostate": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }
    split_manifest = {
        "seed": SEED,
        "source_dataset": SOURCE.name,
        "base_dataset": BASE_DATASET.name,
        "preserves_base_train_val_split": True,
        "base_train_count": len(base_train),
        "base_val_count": len(base_val),
        "added_train_count": len(train_cases) - len(base_train),
        "added_val_count": len(val_cases) - len(base_val),
        "train": train_cases,
        "val": val_cases,
    }
    (DESTINATION / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")
    (DESTINATION / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    (DESTINATION / "splits_final.json").write_text(
        json.dumps([{"train": train_cases, "val": val_cases}], indent=2) + "\n"
    )

    print(
        f"Created {DESTINATION.name}: training_pool={len(cases)}, "
        f"fold0_train={len(train_cases)}, fold0_val={len(val_cases)}"
    )


if __name__ == "__main__":
    main()
