export nnUNet_preprocessed="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_results"
export nnUNet_raw="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw"
clear


nnUNetv2_generate_dataset_json -i nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/imagesTr \
-l nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/labelsTr -o nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI \
--task_name DatasCUDA_VISIBLE_DEVICES=0 nnUNetv2_train 130 3d_fullres 0 -tr DIYBoundaryCMSAGAuxTrainer
et2302_FullPI-CAI --modalities 0 1 2



nnUNetv2_plan_and_preprocess -d 2302 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 130 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 131 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 112 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 147 --verify_dataset_integrity

nnUNetv2_plan_and_preprocess -d 141 --verify_dataset_integrity

# Baseline + 频域增强模块 + 多尺度注意力门控
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 112 3d_fullres 0 -tr DIYBoundaryMSAGTrainer 
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 130 3d_fullres 4 -tr DIYBoundaryMSAGTrainer  --c # AHCDU
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 131 3d_fullres 4 -tr DIYBoundaryMSAGTrainer  --c # PICAI
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 158 3d_fullres 0 -tr DIYBoundaryMSAGTrainer


# Baseline + 频域增强模块 + 多尺度注意力门控 + 分割边界辅助任务
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 112 3d_fullres 4 -tr DIYBoundaryMSAGAuxTrainer
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 130 3d_fullres 3 -tr DIYBoundaryMSAGAuxTrainer
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 131 3d_fullres 4 -tr DIYBoundaryMSAGAuxTrainer
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 131 3d_fullres 1 -tr DIYBoundaryMSAGAuxTrainerTest
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 158 3d_fullres 1 -tr DIYBoundaryMSAGAuxTrainer

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 147 3d_fullres 4 -tr DIYBoundaryMSAGAuxTrainerTest \
--val --val_best 

# 先把 plans.json 里的 batch_size 改成 8
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 130 3d_fullres 0 -tr DIYBoundaryMSAGAuxTrainer --c



# Test
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
  -i nnUNet/nnUNet_raw/Dataset112_Promise12/imagesTs \
  -o nnUNet/infer/Dataset112_DIYBoundaryMSAGAuxTrainer/0 \
  -d 147 \
  -c 3d_fullres \
  -f 0 \
  -tr DIYBoundaryMSAGAuxTrainer 
  # -chk checkpoint_best.pth
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 147 3d_fullres 0 -tr DIYBoundaryMSAGAuxTrainer

# Promise12
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
  -i nnUNet/nnUNet_raw/Dataset112_Promise12/imagesTs \
  -o nnUNet/infer/Dataset112_DIYBoundaryMSAGAuxTrainer/2 \
  -d 112 \
  -c 3d_fullres \
  -f 2 \
  -tr DIYBoundaryMSAGAuxTrainer \
  -chk checkpoint_best.pth


# AHCDU
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -i nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU/imagesTs \
  -o nnUNet/infer/Dataset130_DIYBoundaryMSAGAuxTrainer/3 \
  -d 130 \
  -c 3d_fullres \
  -tr DIYBoundaryMSAGAuxTrainer \
  -f 3
#   -chk checkpoint_best.pth

# AHCDU-新版网络
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -i nnUNet/nnUNet_raw/Dataset130_ProstateAHCDU/imagesTs \
  -o nnUNet/infer/Dataset130_DIYBoundaryCMSAGAuxTrainer/3 \
  -d 130 \
  -c 3d_fullres \
  -tr DIYBoundaryCMSAGAuxTrainer \
  -f 3

# PICAI
nnUNetv2_predict \
  -i nnUNet/nnUNet_raw/Dataset131_ProstatePI-CAI/imagesTs \
  -o nnUNet/infer/Dataset131_DIYBoundaryMSAGTrainer/4 \
  -d 131 \
  -c 3d_fullres \
  -tr DIYBoundaryMSAGAuxTrainer \
  -f 4 
  # -chk checkpoint_best.pth


# Prostate158
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
  -i nnUNet/nnUNet_raw/Dataset158_prostate158/imagesTs \
  -o nnUNet/infer/Dataset158_DIYBoundaryMSAGTrainer/0 \
  -d 158 \
  -c 3d_fullres \
  -f 0 \
  -tr DIYBoundaryMSAGTrainer \
  -chk checkpoint_best.pth


########################################################################



CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 2302 3d_fullres 0 -tr CDSATrainer --c

nnUNetv2_train 2302 3d_fullres 1 -tr CDSATrainer
nnUNetv2_train 2302 3d_fullres 2 -tr CDSATrainer
nnUNetv2_train 2302 3d_fullres 3 -tr CDSATrainer
nnUNetv2_train 2302 3d_fullres 4 -tr CDSATrainer

cp -r /home/Space/clib/Projects/CAD/dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU  nnUNet/nnUNet_raw
cp -r /home/Space/clib/Projects/CAD/dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI nnUNet/nnUNet_raw


nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 130 -c 3d_fullres --save_probabilities



/home/ikun_server/anaconda3/envs/rxm/bin/python compute_seg_metrics.py \
 --pred_dir nnUNet/nnUNet_results/Dataset147_FullPICAI/DIYBoundaryMSAGAuxTrainerTest__nnUNetPlans__3d_fullres/fold_0/validation \
 --gt_dir nnUNet/nnUNet_raw/Dataset147_FullPICAI/labelsTr \
 --output_name validation_metrics.json