

python external_src/DeepPruner/deeppruner/train_sceneflow.py \
	--save_dir psmnet_deform25_patchmatch/sceneflow_disparity/ \
	--savemodel trained_stereo_models/psmnet_deform25_patchmatch/sceneflow_models/ \
	--num_deform_layers 25 \
	--datapath_monkaa data/scene_flow_datasets \
	--datapath_flying data/scene_flow_datasets \
	--datapath_driving data/scene_flow_datasets


python external_src/DeepPruner/deeppruner/finetune_kitti.py \
	--loadmodel trained_stereo_models/psmnet_deform25_patchmatch/sceneflow_models/sceneflow_64.tar \
	--savemodel trained_stereo_models/psmnet_deform25_patchmatch/kitti_models/ \
	--epochs 1040 \
	--num_deform_layers 25 \
	--train_datapath_2015 data/kitti_2015/training \
	--val_datapath_2015 data/kitti_2015/training \
	--datapath_2012 data/kitti_2012/training
