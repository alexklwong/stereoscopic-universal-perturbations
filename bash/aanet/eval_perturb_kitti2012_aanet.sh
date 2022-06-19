#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SUPS_DIRPATH=pretrained_perturbations/aanet/tile64_norm002
SUPS_FILENAME=sups_aanet_tile64_norm002.pth

python src/evaluate_perturb_model.py \
--image0_path validation/kitti/kitti_stereo_flow_val_image0.txt \
--image1_path validation/kitti/kitti_stereo_flow_val_image1.txt \
--ground_truth_path validation/kitti/kitti_stereo_flow_val_disparity.txt \
--n_image_height 256 \
--n_image_width 640 \
--attack tile \
--n_perturbation_height 64 \
--n_perturbation_width 64 \
--perturb_model_restore_path $SUPS_DIRPATH/$SUPS_FILENAME \
--stereo_method aanet \
--stereo_model_restore_path pretrained_stereo_models/aanet/aanet_kitti12-e20bb24d.pth \
--output_dirpath $SUPS_DIRPATH/evaluation_results/aanet/kitti2012 \
--save_outputs \
--device gpu \

