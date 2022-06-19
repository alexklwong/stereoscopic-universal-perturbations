export CUDA_VISIBLE_DEVICES=0,1,2,3

python train_sceneflow.py \
    --datapath_monkaa ../../../data/scene_flow_datasets \
    --datapath_flying ../../../data/scene_flow_datasets \
    --datapath_driving ../../../data/scene_flow_datasets \
    --epochs 64 \
    --savemodel deeppruner_deform/sceneflow/ \
    --logging_filename deeppruner_deform/sceneflow/train_sceneflow.log 