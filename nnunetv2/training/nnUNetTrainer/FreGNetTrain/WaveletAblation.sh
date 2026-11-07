#!/usr/bin/env bash
set -euo pipefail

export nnUNet_preprocessed="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_results"
export nnUNet_raw="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw"

GPU_ID="${GPU_ID:-0}"

# Wavelet ablation (Table 5) with the new adaptive W_x.
# db1 (Ours) is DIYBoundaryAdaptiveCMSAGAuxTrainer; the six variants below only
# change the wavelet name, everything else is identical.

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr DIYBoundaryAdaptiveCMSAGAuxTrainer_sym2

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr DIYBoundaryAdaptiveCMSAGAuxTrainer_sym3

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr DIYBoundaryAdaptiveCMSAGAuxTrainer_coif1

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr DIYBoundaryAdaptiveCMSAGAuxTrainer_coif2

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr DIYBoundaryAdaptiveCMSAGAuxTrainer_db2

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr DIYBoundaryAdaptiveCMSAGAuxTrainer_db3

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr DIYBoundaryAdaptiveCMSAGAuxTrainer_dmey
