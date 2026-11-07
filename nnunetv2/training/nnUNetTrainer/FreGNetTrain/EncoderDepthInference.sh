#!/usr/bin/env bash
set -euo pipefail

export nnUNet_preprocessed="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_results"
export nnUNet_raw="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw"

GPU_ID="${GPU_ID:-0}"
INPUT_DIR="${nnUNet_raw}/Dataset141_FullPICAI/imagesTs"
OUTPUT_ROOT="/home/Space/clib/Projects/nnUNetv2/nnUNet/infer"

# Depth-3 inference using the best fold-0 checkpoint.
CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_ROOT}/Dataset141_DIYBoundaryCMSAGAuxTrainerDepth3/0" \
    -d 141 \
    -c 3d_fullres \
    -tr DIYBoundaryCMSAGAuxTrainerDepth3 \
    -f 0 \
    -chk checkpoint_best.pth \
    -npp 2 \
    -nps 1

# Depth-5 inference using the best fold-0 checkpoint.
CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_ROOT}/Dataset141_DIYBoundaryCMSAGAuxTrainerDepth5/0" \
    -d 141 \
    -c 3d_fullres \
    -tr DIYBoundaryCMSAGAuxTrainerDepth5 \
    -f 0 \
    -chk checkpoint_best.pth \
    -npp 2 \
    -nps 1

# Adaptive W_x variants.
CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_ROOT}/Dataset141_DIYBoundaryAdaptiveCMSAGAuxTrainerDepth3/0" \
    -d 141 \
    -c 3d_fullres \
    -tr DIYBoundaryAdaptiveCMSAGAuxTrainerDepth3 \
    -f 0 \
    -chk checkpoint_best.pth \
    -npp 2 \
    -nps 1

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_ROOT}/Dataset141_DIYBoundaryAdaptiveCMSAGAuxTrainerDepth5/0" \
    -d 141 \
    -c 3d_fullres \
    -tr DIYBoundaryAdaptiveCMSAGAuxTrainerDepth5 \
    -f 0 \
    -chk checkpoint_best.pth \
    -npp 2 \
    -nps 1
