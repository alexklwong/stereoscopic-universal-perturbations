export CUDA_VISIBLE_DEVICES=0,1,2,3

python finetune_kitti.py \
--loadmodel deeppruner_deform/sceneflow/sceneflow_63.tar \
--savemodel deeppruner_deform/kitti/ \
--train_datapath_2015 data/kitti_2015/training \
--datapath_2012 data/kitti_2012/training \
--logging_filename deeppruner_deform/kitti/finetune_kitti.log \
--epochs 1040


