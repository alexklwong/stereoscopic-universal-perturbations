#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SUPS_DIRPATH=pretrained_perturbations/psmnet/clean

python src/evaluate_perturb_model.py \
--image0_path testing/flyingthings3d/flyingthings3d_test_image0_finalpass.txt \
--image1_path testing/flyingthings3d/flyingthings3d_test_image1_finalpass.txt \
--ground_truth_path testing/flyingthings3d/flyingthings3d_test_disparity0.txt \
--n_image_height 256 \
--n_image_width 640 \
--stereo_method psmnet \
--stereo_model_restore_path pretrained_stereo_models/psmnet/pretrained_sceneflow_retrained.tar \
--output_dirpath $SUPS_DIRPATH/evaluation_results/psmnet/flyingthings3d \
--save_outputs \
--device gpu \
