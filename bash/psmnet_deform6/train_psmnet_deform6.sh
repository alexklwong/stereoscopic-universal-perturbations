#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python external_src/PSMNet/main.py \
    --maxdisp 192 \
    --model stackhourglass \
    --datapath data/scene_flow_datasets/ \
    --epochs 10 \
    --num_deform_layers 6 \
    --savemodel trained_stereo_models/psmnet_deform6/



python external_src/PSMNet/finetune.py \
    --maxdisp 192 \
    --model stackhourglass \
    --datatype 2015 \
    --datapath data/kitti_scene_flow/training \
    --epochs 300 \
    --num_deform_layers 6 \
    --loadmodel trained_stereo_models/psmnet_deform6/sceneflow/checkpoint_10.tar \
    --savemodel trained_stereo_models/psmnet_deform6/kitti/

