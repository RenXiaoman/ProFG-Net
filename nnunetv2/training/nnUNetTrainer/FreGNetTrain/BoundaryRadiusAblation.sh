#!/usr/bin/env bash
set -euo pipefail

export nnUNet_preprocessed="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_results"
export nnUNet_raw="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw"

GPU_ID="${GPU_ID:-0}"

# Table 8 boundary-target radius sweep with the latest model
# (LearnableAdaptiveCMSAG).
# All five trainers use num_epochs=70.
# radius 1 uses _1 (same architecture as the latest model, 70 epochs) so the
# default epochs of LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer stay untouched.

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer_1

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer_2

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer_3

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer_4

CUDA_VISIBLE_DEVICES="${GPU_ID}" nnUNetv2_train 141 3d_fullres 0 \
    -tr LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer_5
