export nnUNet_preprocessed="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_results"
export nnUNet_raw="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw"
clear

# Baseline + 多尺度注意力门控
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 130 3d_fullres 3 -tr MSAGTrainer  --c # AHCDU
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 131 3d_fullres 3 -tr MSAGTrainer  --c # PICAI

# Baseline + 分割边界辅助任务
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 130 3d_fullres 2 -tr BoundaryTrainer  --c # AHCDU
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 131 3d_fullres 2 -tr BoundaryTrainer  --c # PICAI

# Baseline + 频域增强 + 分割边界辅助任务
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 130 3d_fullres 1 -tr FrequencyAuxTrainer
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 131 3d_fullres 0 -tr FrequencyAuxTrainer

# Baseline + 多尺度注意力门控 + 分割边界辅助任务
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 130 3d_fullres 1 -tr MSAGAuxTrainer
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 131 3d_fullres 0 -tr MSAGAuxTrainer

CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 141 3d_fullres 0 -tr DIYBoundaryCMSAGAuxTrainer

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -i nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU/imagesTs \
  -o nnUNet/infer/Dataset130_BoundaryTrainer/0 \
  -d 130 \
  -c 3d_fullres \
  -tr BoundaryTrainer \
  -f 0
#   -chk checkpoint_best.pth

CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
  -i nnUNet/nnUNet_raw/Dataset131_ProstatePI-CAI/imagesTs \
  -o nnUNet/infer/Dataset131_MSAGTrainer/3 \
  -d 131 \
  -c 3d_fullres \
  -tr MSAGTrainer \
  -f 3
  # -chk checkpoint_best.pth