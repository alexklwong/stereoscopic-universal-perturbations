#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

python src/finetune.py \
	--train_image0_path training/kitti/kitti_scene_flow_train_image0.txt \
	--train_image1_path training/kitti/kitti_scene_flow_train_image1.txt \
	--train_ground_truth_path training/kitti/kitti_scene_flow_train_disparity.txt \
	--val_image0_path validation/kitti/kitti_scene_flow_val_image0.txt \
	--val_image1_path validation/kitti/kitti_scene_flow_val_image1.txt \
	--val_ground_truth_path validation/kitti/kitti_scene_flow_val_disparity.txt \
	--learning_rates 1e-05 5e-06 1e-06 5e-7 \
	--learning_schedule 250 500 750 1000 \
	--stereo_method aanet \
	--stereo_model_restore_path pretrained_stereo_models/aanet/aanet_kitti15-fb2a0d23.pth \
	--n_image_height 256 \
	--n_image_width 640 \
	--output_norm 0.02 0.01 0.005 0.002 \
	--gradient_scale 0.0001 0.0001 0.00005 0.0004 \
	--attack tile \
	--n_perturbation_height 64 \
	--n_perturbation_width 64 \
	--perturb_paths \
	pretrained_perturbations/aanet/tile64_norm002/sups_aanet_tile64_norm002.pth \
	pretrained_perturbations/aanet/tile64_norm001/sups_aanet_tile64_norm001.pth \
	pretrained_perturbations/aanet/tile64_norm0005/sups_aanet_tile64_norm0005.pth \
	pretrained_perturbations/aanet/tile64_norm0002/sups_aanet_tile64_norm0002.pth \
	--n_checkpoint 500 \
	--checkpoint_path trained_stereo_models/aanet_finetuned \
	--n_batch 8 \
	--n_worker 8 \
	--device gpu \


