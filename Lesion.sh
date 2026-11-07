export nnUNet_preprocessed="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_results"
export nnUNet_raw="/home/Space/clib/Projects/nnUNetv2/nnUNet/nnUNet_raw"
clear
# CDSA
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 2302 3d_fullres 1 -tr CDSATrainer

# UNet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 2302 3d_fullres 0 -tr UNetDetectionTrainer


# 第一步: 推理
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -d 2302 \
  -i nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/val0 \
  -o nnUNet/infer/Dataset2302_UNetDetectionTrainer_eval/fold_0 \
  -c 3d_fullres \
  -tr UNetDetectionTrainer \
  -f 0 \
  --save_probabilities \
  -npp 2 \
  -nps 1 \
  -chk checkpoint_best.pth 

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -d 2302 \
  -i nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/val1 \
  -o nnUNet/infer/Dataset2302_CDSATrainer_eval/fold_1 \
  -c 3d_fullres \
  -tr CDSATrainer \
  -f 1 \
  --save_probabilities \
  -npp 2 \
  -nps 1 \
  -chk checkpoint_best.pth \

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -d 2302 \
  -i nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/val2 \
  -o nnUNet/infer/Dataset2302_CDSATrainer_eval/fold_2 \
  -c 3d_fullres \
  -tr CDSATrainer \
  -f 2 \
  --save_probabilities \
  -npp 2 \
  -nps 1 \
  -chk checkpoint_best.pth 

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -d 2302 \
  -i nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/val3 \
  -o nnUNet/infer/Dataset2302_CDSATrainer_eval/fold_3 \
  -c 3d_fullres \
  -tr CDSATrainer \
  -f 3 \
  -chk checkpoint_best.pth \
  --save_probabilities \
  -npp 2 \
  -nps 1 \
  -chk checkpoint_best.pth 

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -d 2302 \
  -i nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/val4 \
  -o nnUNet/infer/Dataset2302_CDSATrainer_eval/fold_4 \
  -c 3d_fullres \
  -tr CDSATrainer \
  -f 4 \
  -chk checkpoint_best.pth \
  --save_probabilities \
  -npp 2 \
  -nps 1


# /home/ikun_server/anaconda3/envs/rxm/bin/python eval.py \
#   --task Dataset2302_FullPI-CAI \
#   --trainer CDSATrainer \
#   --workdir /home/Space/clib/Projects/nnUNetv2 \
#   --task_dir nnUNet/infer/Dataset2302_CDSATrainer_eval \
#   --checkpoints checkpoint_best \
#   --folds 1 \
#   --predictions_folder picai_pubtrain_predictions_checkpoint_best \
#   --labels_folder nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/labelsTr \
#   --splits nnUNet/infer/Dataset2302_CDSATrainer_eval/fold_{fold}_subjects.json \
#   --metrics_fn metrics-{checkpoint}-{fold}-{threshold}.json


# 第二步: 保存检测图
# /home/ikun_server/anaconda3/envs/rxm/bin/python eval_zonal.py \
#   --task Dataset2302_FullPI-CAI \
#   --trainer CDSATrainer \
#   --workdir /home/Space/clib/Projects/nnUNetv2 \
#   --task_dir nnUNet/infer/Dataset2302_CDSATrainer_eval \
#   --checkpoints checkpoint_best \
#   --folds 0 \
#   --predictions_folder picai_pubtrain_predictions_checkpoint_best \
#   --labels_folder nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/labelsTr \
#   --splits nnUNet/infer/Dataset2302_CDSATrainer_eval/fold_{fold}_subjects.json \
#   --zonal_mask_folder nnUNet/infer/Dataset2302_CDSATrainer_eval/zonal_masks/fold_0 \
#   --detection_map_folder nnUNet/infer/Dataset2302_CDSATrainer/fold_0/detected_map \
#   --metrics_fn metrics-{checkpoint}-{fold}-{threshold}-zonal.json



/home/ikun_server/anaconda3/envs/rxm/bin/python detection_tool/eval_zonal.py \
  --task Dataset2302_FullPI-CAI \
  --trainer CDSATrainer \
  --workdir /home/Space/clib/Projects/nnUNetv2 \
  --task_dir nnUNet/infer/Dataset2302_CDSATrainer_eval \
  --checkpoints checkpoint_best \
  --folds 4 \
  --predictions_folder picai_pubtrain_predictions_checkpoint_best \
  --labels_folder nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/labelsTr \
  --splits '/home/Space/clib/Projects/nnUNetv2/nnUNet/infer/Dataset2302_CDSATrainer_eval/fold_{fold}_subjects.json' \
  --zonal_mask_folder 'nnUNet/infer/Dataset2302_CDSATrainer_eval/zonal_masks/fold_{fold}' \
  --detection_map_folder 'nnUNet/infer/Dataset2302_CDSATrainer_eval/fold_{fold}/detected_map' \
  --metrics_fn 'metrics-{checkpoint}-{fold}-{threshold}-zonal.json'

/home/ikun_server/anaconda3/envs/rxm/bin/python detection_tool/eval_zonal.py \
  --task Dataset2302_FullPI-CAI \
  --trainer UNetDetectionTrainer \
  --workdir /home/Space/clib/Projects/nnUNetv2 \
  --task_dir nnUNet/infer/Dataset2302_UNetDetectionTrainer_eval \
  --checkpoints checkpoint_best \
  --folds 0 \
  --predictions_folder picai_pubtrain_predictions_checkpoint_best \
  --labels_folder nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/labelsTr \
  --splits '/home/Space/clib/Projects/nnUNetv2/nnUNet/infer/Dataset2302_UNetDetectionTrainer_eval/fold_{fold}_subjects.json' \
  --zonal_mask_folder 'nnUNet/infer/Dataset2302_UNetDetectionTrainer_eval/zonal_masks/fold_{fold}' \
  --detection_map_folder 'nnUNet/infer/Dataset2302_UNetDetectionTrainer_eval/fold_{fold}/detected_map' \
  --metrics_fn 'metrics-{checkpoint}-{fold}-{threshold}-zonal.json'


# 第三步: 可视化
python detection_tool/visualize_detection.py \
  nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/val4 \
  nnUNet/infer/Dataset2302_CDSATrainer_eval/CDSATrainer__nnUNetPlansv2.1/fold_4/picai_pubtrain_predictions_checkpoint_best \
  nnUNet/infer/Dataset2302_CDSATrainer_eval/fold_4/pred

# UNet
python detection_tool/visualize_detection.py \
  nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI/val0 \
  nnUNet/infer/Dataset2302_UNetDetectionTrainer_eval/UNetDetectionTrainer__nnUNetPlansv2.1/fold_0/picai_pubtrain_predictions_checkpoint_best \
  nnUNet/infer/Dataset2302_UNetDetectionTrainer_eval/fold_0/pred
