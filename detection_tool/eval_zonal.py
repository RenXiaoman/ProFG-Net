# Copyright 2023 Yuan Yuan, Biomedical Data Analysis and Visualisation (BDAV) Lab, The University of Sydney, Sydney, Australia

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# nnUNetv2 usage note for this repository
# ---------------------------------------
# This script expects PI-CAI style foreground probability maps named
# <case>_softmax.nii.gz and prostate/gland masks named <case>.nii.gz.
# When predictions come from nnUNetv2_predict --save_probabilities, the script
# can auto-prepare the expected PI-CAI helper inputs for each evaluated fold:
#
# 1. Copy nnUNetv2 prediction outputs:
#    nnUNet/infer/<model>/fold_<k>/<case>.npz
#    -> nnUNet/infer/<model>_eval/<trainer>__nnUNetPlansv2.1/fold_<k>/
#       picai_pubtrain_predictions_checkpoint_best/<case>.npz
#
# 2. Convert nnUNetv2 probability npz files:
#    nnUNet/infer/<model>_eval/<trainer>__nnUNetPlansv2.1/fold_<k>/
#       picai_pubtrain_predictions_checkpoint_best/<case>.npz
#    -> nnUNet/infer/<model>_eval/<trainer>__nnUNetPlansv2.1/fold_<k>/
#       picai_pubtrain_predictions_checkpoint_best/<case>_softmax.nii.gz
#    Use probabilities[1] as the lesion foreground map and copy image geometry
#    from the same case's nnUNetv2 segmentation nifti.
#
# 3. Create a subject-list json from splits_final.json:
#    nnUNet/infer/<model>_eval/fold_<k>_subjects.json
#    {"subject_list": splits[k]["val"]}
#
# 4. Create fold-specific gland mask links:
#    nnUNet/infer/<model>_eval/zonal_masks/fold_<k>/<case>.nii.gz
#    -> nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/labelsTr_gland_Bosma22b/<case>.nii.gz
#
# 5. Pass an explicit detection_map_folder, for example:
#    --detection_map_folder nnUNet/infer/<model>/fold_<k>/detected_map
#
# The script will then generate <case>_detection_map.nii.gz by extracting
# lesion candidates from <case>_softmax.nii.gz and removing connected
# components that do not intersect the gland mask. Metrics are computed from
# these filtered detection maps, not directly from the raw softmax files.


import argparse
import json
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from picai_eval import evaluate_folder
from picai_prep.preprocessing import crop_or_pad
from report_guided_annotation import extract_lesion_candidates

from picai_baseline.nnunet.softmax_export import \
    convert_cropped_npz_to_original_nifty
from picai_baseline.splits.picai import valid_splits as picai_pub_valid_splits
from picai_baseline.splits.picai_nnunet import \
    valid_splits as picai_pub_nnunet_valid_splits
from picai_eval.image_utils import read_prediction
import SimpleITK as sitk


try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass


def round5(value):
    if value is None:
        return None
    return round(float(value), 5)


def build_summary_from_metrics_data(
    metrics_data: dict,
    task_name: str,
    fold: int,
    softmax_dir: Path,
) -> dict:
    case_pred = metrics_data.get("case_pred", {})
    case_target = metrics_data.get("case_target", {})
    lesion_results = metrics_data.get("lesion_results", {})

    cases = []
    for patient_name in sorted(case_target):
        target = int(case_target[patient_name])
        case_results = lesion_results.get(patient_name, [])
        candidate_confidences = [
            round5(item[1])
            for item in case_results
            if len(item) > 1 and float(item[1]) > 0
        ]
        detected = any(
            len(item) > 2 and int(item[0]) == 1 and float(item[1]) > 0 and float(item[2]) > 0
            for item in case_results
        )
        case_record = {
            "patient_name": patient_name,
            "case_confidence": round5(case_pred.get(patient_name, 0)),
            "case_target": target,
            "is_positive_gt": bool(target),
            "num_candidates": len(candidate_confidences),
            "candidate_confidences": candidate_confidences,
            "max_det": max(candidate_confidences) if candidate_confidences else 0.0,
        }
        if target:
            case_record["detected"] = detected
        cases.append(case_record)

    num_pos = sum(int(v == 1) for v in case_target.values())
    num_neg = sum(int(v == 0) for v in case_target.values())
    detected_positive_cases = sum(
        int(record["is_positive_gt"] and record.get("detected", False))
        for record in cases
    )
    ap = metrics_data.get("AP")
    auroc = metrics_data.get("auroc")
    score = None if ap is None or auroc is None else (float(ap) + float(auroc)) / 2

    return {
        "task_name": task_name,
        "fold": fold,
        "softmax_dir": str(softmax_dir),
        "num_cases": len(cases),
        "num_positive_cases": num_pos,
        "detected_positive_cases": detected_positive_cases,
        "num_negative_cases": num_neg,
        "AP": round5(ap),
        "AUROC": round5(auroc),
        "Score": round5(score),
        "cases": cases,
    }


def save_summary_json(
    metrics_path: Path,
    output_dir: Union[Path, str],
    task_name: str,
    fold: int,
    softmax_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    metrics_data = json.loads(metrics_path.read_text())
    summary = build_summary_from_metrics_data(metrics_data, task_name, fold, softmax_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "1A_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved summary to: {summary_path}")


def resolve_fold_path(path: Optional[Union[Path, str]], fold: int) -> Optional[Path]:
    if path is None:
        return None
    return Path(str(path).replace(r"{fold}", str(fold)))


def copy_nnunetv2_fold_predictions(src_dir: Path, dst_dir: Path, overwrite: bool = False) -> int:
    """Copy nnUNetv2_predict outputs from fold_<k> into the PI-CAI eval folder."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.is_dir():
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_path in sorted(src_dir.iterdir()):
        if not src_path.is_file():
            continue
        dst_path = dst_dir / src_path.name
        if dst_path.exists() and not overwrite:
            continue
        shutil.copy2(src_path, dst_path)
        copied += 1
    return copied


def _foreground_from_probability_npz(npz_path: Path) -> Optional[np.ndarray]:
    with np.load(npz_path) as data:
        if "probabilities" in data.files:
            probabilities = data["probabilities"]
        elif "softmax" in data.files:
            probabilities = data["softmax"]
        else:
            return None

    if probabilities.ndim == 4:
        if probabilities.shape[0] < 2:
            raise ValueError(f"Expected at least 2 channels in {npz_path}, got {probabilities.shape}")
        foreground = probabilities[1]
    elif probabilities.ndim == 3:
        foreground = probabilities
    else:
        raise ValueError(f"Unsupported probability shape in {npz_path}: {probabilities.shape}")
    return foreground.astype(np.float32, copy=False)


def convert_prediction_npz_to_softmax_nifti(npz_path: Union[Path, str]) -> None:
    """Create <case>_softmax.nii.gz from nnUNetv2 probability npz files.

    Falls back to the PI-CAI/nnU-Net v1 converter for older cropped npz files.
    """
    npz_path = Path(npz_path)
    dst_path = npz_path.with_name(f"{npz_path.stem}_softmax.nii.gz")
    if dst_path.exists():
        return

    foreground = _foreground_from_probability_npz(npz_path)
    reference_path = npz_path.with_suffix(".nii.gz")
    if foreground is not None and reference_path.exists():
        reference = sitk.ReadImage(str(reference_path))
        reference_shape = sitk.GetArrayFromImage(reference).shape
        if tuple(foreground.shape) == tuple(reference_shape):
            softmax_image = sitk.GetImageFromArray(foreground)
            softmax_image.CopyInformation(reference)
            sitk.WriteImage(softmax_image, str(dst_path))
            return

    convert_cropped_npz_to_original_nifty(npz_path)


def prepare_softmax_prediction_folder(task_dir: Path, softmax_dir: Path, fold: int, verbose: int = 2) -> None:
    """Ensure the expected PI-CAI softmax folder exists and is populated."""
    softmax_dir.mkdir(parents=True, exist_ok=True)
    has_predictions = any(softmax_dir.glob("*.npz")) or any(softmax_dir.glob("*_softmax.nii.gz"))
    source_dir = task_dir / f"fold_{fold}"

    if not has_predictions and source_dir.is_dir() and source_dir.resolve() != softmax_dir.resolve():
        copied = copy_nnunetv2_fold_predictions(source_dir, softmax_dir)
        if verbose >= 1:
            print(f"Copied {copied} files from {source_dir} to {softmax_dir}")


def extract_lesion_candidates_originated_from_prostate(
    softmax_path: Union[Path, str], 
    zonal_mask_path: Union[Path, str],
    dst_path: Optional[Union[Path, str]] = None,
    overwrite: bool = False,
) -> None:
    """Extract the candidate lesions that is originated from prostate"""
    softmax_path = Path(softmax_path)
    zonal_mask_path = Path(zonal_mask_path)

    if dst_path is None:
        dst_path = softmax_path.parent / softmax_path.name.replace("_softmax.nii.gz", "_detection_map.nii.gz")
    else:
        dst_path = Path(dst_path)
    if not overwrite and dst_path.exists():
        return

    # read softmax and zonal mask
    softmax_array = read_prediction(softmax_path)
    zonal_mask_array = read_prediction(zonal_mask_path)

    # generate the detection map
    detection_map = extract_lesion_candidates(softmax=softmax_array, threshold="dynamic")[0]
    
    # calculate the connected component of the detection map
    binary_detection_map = (detection_map > 0).astype(np.uint8)
    binary_detection_map: sitk.Image = sitk.GetImageFromArray(binary_detection_map)
    binary_detection_map.CopyInformation(sitk.ReadImage(str(softmax_path)))

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    out_mask = cc_filter.Execute(binary_detection_map)
    out_mask_array = sitk.GetArrayFromImage(out_mask)

    num_cc = cc_filter.GetObjectCount()
    for i in range(1, num_cc + 1):
        # select the connected component
        cc = (out_mask_array == i).astype(np.uint8)
        # calculate the intersection with zonal mask
        intersection = np.sum(cc * zonal_mask_array)
        if intersection == 0:
            # remove the connected component that without intersection with zonal mask
            detection_map[cc == 1] = 0
        else:
            continue
    
    # save the detection map
    detection_map: sitk.Image = sitk.GetImageFromArray(detection_map)
    detection_map.CopyInformation(sitk.ReadImage(str(softmax_path)))
    sitk.WriteImage(detection_map, str(dst_path))
    

def evaluate(
    task: str = "Task2302_z-nnmnet",
    trainer: str = "myTrainer_zonal",
    workdir: Union[Path, str] = "/workdir",
    task_dir: Union[Path, str] = "auto",
    checkpoints: str = ["model_best"],
    folds: str = list(range(5)),
    threshold: str = "dynamic",
    metrics_fn: str = "metrics-{checkpoint}-{threshold}-zonal.json",
    splits: str = "picai_pub",
    predictions_folder: str = r"predictions_{checkpoint}",
    labels_folder: Union[Path, str] = "auto",
    zonal_mask_folder: Union[Path, str] = "/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post/",
    detection_map_folder: Optional[Union[Path, str]] = None,
    save_summary: bool = True,
    output_dir: Optional[Union[Path, str]] = None,
    verbose: int = 2,
):
    # input validation
    workdir = Path(workdir)
    if task_dir == "auto":
        task_dir = workdir / "results" / "nnUNet" / "3d_fullres" / task
    else:
        task_dir = workdir / task_dir
    
    if isinstance(splits, str):
        if splits == "":
            splits = None
        else:
            # select splits
            predefined_splits = {
                "picai_pub": picai_pub_valid_splits,              
                "picai_pub_nnunet": picai_pub_nnunet_valid_splits,
            }
            if splits in predefined_splits:
                splits = predefined_splits[splits]
            else:
                # `splits` should be the path to a json file containing the splits
                print(f"Loading splits from {splits}")

    
    for fold in folds:
        print(f"Evaluating fold {fold}...")
        fold_zonal_mask_folder = resolve_fold_path(zonal_mask_folder, fold)
        fold_detection_map_folder = resolve_fold_path(detection_map_folder, fold)
        fold_output_dir = resolve_fold_path(output_dir, fold)
        if fold_output_dir is None and fold_detection_map_folder is not None:
            fold_output_dir = fold_detection_map_folder.parent
        if fold_detection_map_folder is not None:
            fold_detection_map_folder.mkdir(parents=True, exist_ok=True)

        for checkpoint in checkpoints:
            pred_folder = predictions_folder.replace(r"{checkpoint}", checkpoint)
            softmax_dir = task_dir / f"{trainer}__nnUNetPlansv2.1" / f"fold_{fold}" / pred_folder
            metrics_path = softmax_dir.parent / metrics_fn.replace(r"{checkpoint}", checkpoint).replace(r"{threshold}", threshold).replace("{fold}", str(fold))
 
            prepare_softmax_prediction_folder(task_dir, softmax_dir, fold, verbose=verbose)

            if metrics_path.exists():
                print(f"Metrics found at {metrics_path}, skipping..")
                if save_summary:
                    if fold_output_dir is None:
                        raise ValueError("Cannot save 1A_summary.json: pass --detection_map_folder or --summary_dir")
                    save_summary_json(metrics_path, fold_output_dir, task, fold, softmax_dir)
                continue
            else:
                print(f"Metrics will be saved to {metrics_path}.")

            original_softmax_prediction_paths = softmax_dir.glob("*.npz")

            if verbose >= 2:
                print(f"Predictions folder: {softmax_dir}")
                original_softmax_prediction_paths = list(original_softmax_prediction_paths)
                print(f"Found {len(original_softmax_prediction_paths)} predictions (e.g., {original_softmax_prediction_paths[0:2]})")

            # pad raw npz predictions to their original extent
            print("Converting nnU-Net predictions to original extent...")
            with Pool() as pool:
                pool.map(
                    func=convert_prediction_npz_to_softmax_nifti,
                    iterable=original_softmax_prediction_paths,
                )

            # postprocess softmax predictions
            print("Postprocessing softmax predictions...")
            softmax_prediction_paths = list(softmax_dir.glob("*_softmax.nii.gz"))
            for softmax_path in softmax_prediction_paths:
                dst_path = None if fold_detection_map_folder is None else fold_detection_map_folder / softmax_path.name.replace("_softmax.nii.gz", "_detection_map.nii.gz")
                extract_lesion_candidates_originated_from_prostate(
                    softmax_path,
                    Path(fold_zonal_mask_folder / softmax_path.name.replace("_softmax.nii.gz", ".nii.gz")),
                    dst_path=dst_path,
                )

            # select subject_list
            if splits is None:
                subject_list = None
            elif isinstance(splits, dict):
                if fold in splits:
                    subject_list = splits[fold]["subject_list"]
                elif str(fold) in splits:
                    subject_list = splits[str(fold)]["subject_list"]
                elif "subject_list" in splits:
                    subject_list = splits["subject_list"]
                else:
                    raise KeyError(f"Could not find fold {fold} in splits")
            elif isinstance(splits, str):
                path = Path(splits.replace(r"{fold}", str(fold)))
                with open(path, "r") as f:
                    split_data = json.load(f)
                subject_list = split_data["subject_list"]
            else:
                raise ValueError(f"Unrecognised splits: {splits}")

            # select labels folder
            if labels_folder == "auto":
                y_true_dir = workdir / "nnUNet_raw_data" / task / "labelsTr"
            else:
                y_true_dir = workdir / labels_folder.replace(r"{fold}", str(fold))
           
            # evaluate
            metrics = evaluate_folder(
                y_det_dir=softmax_dir if fold_detection_map_folder is None else fold_detection_map_folder,
                y_true_dir=y_true_dir,
                subject_list=subject_list,
                pred_extensions=['_detection_map.nii.gz'],
                num_parallel_calls=5,
            )

            # save and show metrics
            metrics.save(metrics_path)
            if save_summary:
                if fold_output_dir is None:
                    raise ValueError("Cannot save 1A_summary.json: pass --detection_map_folder or --summary_dir")
                save_summary_json(metrics_path, fold_output_dir, task, fold, softmax_dir)
            print(f"Results for checkpoint {checkpoint}:")
            print(metrics)


def main():
    parser = argparse.ArgumentParser(description='Command Line Arguments')
    parser.add_argument("--task", type=str, default="Task2302_z-nnmnet",
                        help="Task name of the Z-nnMNet experiment. Default: Task2302_z-nnmnet")
    parser.add_argument("--trainer", type=str, default="myTrainer_zonal",
                        help="Trainer of the Z-nnMNet experiment. Default: myTrainer_zonal")
    parser.add_argument("--workdir", type=str, default="/workdir",
                        help="Path to the workdir where 'results' and 'nnUNet_raw_data' are stored. Default: /workdir")
    parser.add_argument("--task_dir", type=str, default="auto",
                        help="Path to the task directory (relative to the workdir). Optional, will be constucted for default nnU-Net forlder structure")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=["model_best"],
                        help="Which checkpoints to evaluate. Multiple checkpoints can be passed at once. Default: model_best")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)),
                        help="Which folds to evaluate. Multiple folds can be evaluated at once. Default: 0, 1, 2, 3, 4  (all)")
    parser.add_argument("--threshold", type=str, default="dynamic",
                        help="Threshold for lesion extraction from softmax predictions. " + \
                             "Use dynamic-fast for quicker evaluation at almost equal performance." + \
                                "Default: dynamic.")
    parser.add_argument("--metrics_fn", type=str, default=r"metrics-{checkpoint}-{threshold}-zonal.json",
                        help=r"Filename to save metrics to. May contain {checkpoint} and {threshold} which are auto-filled. Default: metrics-{checkpoint}-{threshold}-zonal.json")
    parser.add_argument("--splits", type=str, default="picai_pub_nnunet",
                        help="Splits for cross-validation. Available predefined splits: picai_pub, picai_pub_nnunet. Alternatively, provide a path to a json file " +
                             "containing a dictiory with key 'subject_list', where `{fold}` in the path is replaced by " +
                             "the fold. Example: `/workdir/splits/val/fold_{fold}.json`. Default: picai_pub_nnunet.")
    parser.add_argument("--predictions_folder", type=str, default="predictions_{checkpoint}",
                        help=r"Folder with nnU-Net softmax predictions.")
    parser.add_argument("--labels_folder", type=str, default="auto",
                        help="Folder with annotations. Optional, will be constucted to labelsTr of the specified " +
                             "nnU-Net task.")
    parser.add_argument("--zonal_mask_folder", type=str, default="/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post/",
                        help="Folder with zonal masks.")
    parser.add_argument("--detection_map_folder", type=str, default=None,
                        help="Optional folder to save and evaluate zonal-filtered detection maps.")
    parser.add_argument("-o", "--save_summary", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to save 1A_summary.json. Default: true.")
    parser.add_argument("--summary_dir", type=str, default=None,
                        help="Optional directory for 1A_summary.json. Default: parent of --detection_map_folder.")
    args = parser.parse_args()

    # evaluate
    evaluate(
        task=args.task,
        trainer=args.trainer,
        workdir=args.workdir,
        task_dir=args.task_dir,
        checkpoints=args.checkpoints,
        folds=args.folds,
        threshold=args.threshold,
        metrics_fn=args.metrics_fn,
        splits=args.splits,
        predictions_folder=args.predictions_folder,
        labels_folder=args.labels_folder,
        zonal_mask_folder=args.zonal_mask_folder,
        detection_map_folder=args.detection_map_folder,
        save_summary=args.save_summary,
        output_dir=args.summary_dir,
    )

if __name__ == "__main__":
    main()
