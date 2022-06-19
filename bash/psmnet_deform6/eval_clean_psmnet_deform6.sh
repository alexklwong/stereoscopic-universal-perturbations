#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SUPS_DIRPATH=pretrained_perturbations/psmnet_deform6/clean

python src/evaluate_perturb_model.py \
--image0_path validation/kitti/kitti_scene_flow_val_image0.txt \
--image1_path validation/kitti/kitti_scene_flow_val_image1.txt \
--ground_truth_path validation/kitti/kitti_scene_flow_val_disparity.txt \
--n_image_height 256 \
--n_image_width 640 \
--stereo_method psmnet \
--num_deform_layers 6 \
--stereo_model_restore_path pretrained_stereo_models/psmnet_deform6/pretrained_KITTI2015.tar \
--output_dirpath $SUPS_DIRPATH/evaluation_results/psmnet_deform6/kitti2015 \
--save_outputs \
--device gpu
