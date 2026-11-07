#!/usr/bin/env bash
set -euo pipefail

export nnUNet_preprocessed="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_results"
export nnUNet_raw="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw"
clear

GPU_ID="${GPU_ID:-0}"

# Encoder-depth ablation of WFE + CMSAG + boundary auxiliary supervision.
# The depth-4 reference trainer is DIYBoundaryCMSAGAuxTrainer.
CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr DIYBoundaryCMSAGAuxTrainerDepth3

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr DIYBoundaryCMSAGAuxTrainerDepth5

# Same encoder-depth ablation with the new adaptive W_x (AdaptiveComplementaryMultiScaleDepthwiseBlock).
# The depth-4 reference for this group is DIYBoundaryAdaptiveCMSAGAuxTrainer.
CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr DIYBoundaryAdaptiveCMSAGAuxTrainerDepth3

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr DIYBoundaryAdaptiveCMSAGAuxTrainerDepth5
