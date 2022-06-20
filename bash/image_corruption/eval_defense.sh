#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SUPS_DIRPATH=pretrained_perturbations/psmnet/gaussian_stddev2

python src/evaluate_perturb_model.py \
--image0_path validation/kitti/kitti_scene_flow_val_image0.txt \
--image1_path validation/kitti/kitti_scene_flow_val_image1.txt \
--ground_truth_path validation/kitti/kitti_scene_flow_val_disparity.txt \
--n_image_height 256 \
--n_image_width 640 \
--stereo_method psmnet \
--defense_type gaussian \
--stdev 2 \
--stereo_model_restore_path pretrained_stereo_models/psmnet/pretrained_KITTI2015.tar \
--output_dirpath $SUPS_DIRPATH/evaluation_results/psmnet/kitti2015 \
--device gpu