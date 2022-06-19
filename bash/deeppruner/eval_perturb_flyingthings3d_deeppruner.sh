#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SUPS_DIRPATH=pretrained_perturbations/deeppruner/tile64_norm002
SUPS_FILENAME=sups_deeppruner_tile64_norm002.pth

python src/evaluate_perturb_model.py \
--image0_path testing/flyingthings3d/flyingthings3d_test_image0_finalpass.txt \
--image1_path testing/flyingthings3d/flyingthings3d_test_image1_finalpass.txt \
--ground_truth_path testing/flyingthings3d/flyingthings3d_test_disparity0.txt \
--n_image_height 256 \
--n_image_width 640 \
--attack tile \
--n_perturbation_height 64 \
--n_perturbation_width 64 \
--perturb_model_restore_path $SUPS_DIRPATH/$SUPS_FILENAME \
--stereo_method deeppruner \
--stereo_model_restore_path pretrained_stereo_models/deeppruner/DeepPruner-best-sceneflow.tar \
--output_dirpath $SUPS_DIRPATH/evaluation_results/deeppruner/flyingthings3d \
--save_outputs \
--device gpu \

