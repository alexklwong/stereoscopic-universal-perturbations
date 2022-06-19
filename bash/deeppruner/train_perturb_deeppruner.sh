#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_perturb_model.py \
--train_image0_path training/kitti/kitti_train_image0.txt \
--train_image1_path training/kitti/kitti_train_image1.txt \
--val_image0_path validation/kitti/kitti_scene_flow_val_image0.txt \
--val_image1_path validation/kitti/kitti_scene_flow_val_image1.txt \
--val_ground_truth_path validation/kitti/kitti_scene_flow_val_disparity.txt \
--n_image_height 256 \
--n_image_width 640 \
--output_norm 0.02 \
--gradient_scale 0.0001 \
--attack full \
--n_batch 2 \
--n_epoch 1 \
--stereo_method deeppruner \
--stereo_model_restore_path pretrained_stereo_models/deeppruner/DeepPruner-best-kitti.tar \
--n_checkpoint 500 \
--checkpoint_path trained_perturbations/deeppruner/full_norm002 \
--n_worker 2 \
--device gpu


# pretrained_models/DeepPruner/DeepPruner-best-kitti.tar