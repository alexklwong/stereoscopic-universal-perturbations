'''
Author: Zachery Berger <zackeberger@g.ucla.edu>, Parth Agrawal <parthagrawal24@g.ucla.edu>, Tian Yu Liu <tianyu139@g.ucla.edu>, Alex Wong <alexw@cs.ucla.edu>
If you use this code, please cite the following paper:

Z. Berger, P. Agrawal, T. Liu, S. Soatto, and A. Wong. Stereoscopic Universal Perturbations across Different Architectures and Datasets.
https://arxiv.org/pdf/2112.06116.pdf

@inproceedings{berger2022stereoscopic,
  title={Stereoscopic Universal Perturbations across Different Architectures and Datasets},
  author={Berger, Zachery and Agrawal, Parth and Liu, Tian Yu and Soatto, Stefano and Wong, Alex},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
'''

import os, sys, gdown, zipfile, glob
sys.path.insert(0, 'src')
import data_utils


'''
Define URLs and paths to download pretrained models
'''
GOOGLE_DRIVE_BASE_URL = 'https://drive.google.com/uc?id={}'
PRETRAINED_AANET_ZIP_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1R_sv0mZ4zcFNOnRoVUlYY7VLqkwonORV')

PRETRAINED_MODELS_DIRPATH = 'pretrained_stereo_models'
PRETRAINED_MODELS_ZIP_DIRPATH = os.path.join(PRETRAINED_MODELS_DIRPATH, 'zips')

PRETRAINED_AANET_ZIP_PATH = \
    os.path.join(PRETRAINED_MODELS_ZIP_DIRPATH, 'pretrained_aanet.zip')

PRETRAINED_AANET_ZIP_PATHS = [
    PRETRAINED_AANET_ZIP_PATH
]

PRETRAINED_AANET_ZIP_URLS = [
    PRETRAINED_AANET_ZIP_URL
]

'''
Downloading and unzipping pretrained models
'''
# Create directories to store pretrained models and zips
for dirpath in [PRETRAINED_MODELS_DIRPATH, PRETRAINED_MODELS_ZIP_DIRPATH]:
    os.makedirs(dirpath, exist_ok=True)

for path, url in zip(PRETRAINED_AANET_ZIP_PATHS, PRETRAINED_AANET_ZIP_URLS):

    filename = os.path.basename(path)

    # Download the zip file if we don't already have it
    if not os.path.exists(path):
        print('Downloading {} to {}'.format(filename, path))
        gdown.download(url, path, quiet=False)
    else:
        print('Found {} at {}'.format(filename, path))

    # Unzip the zip file
    print('Unzipping to {}'.format(PRETRAINED_MODELS_DIRPATH))
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(PRETRAINED_MODELS_DIRPATH)

'''
Define URLs and paths to download pseudo ground truth from GANet
'''
TRAIN_REFS_DIRPATH = os.path.join('training', 'kitti')
TEST_REFS_DIRPATH = os.path.join('testing', 'kitti')

# KITTI 2012 dataset (stereo flow)
STEREO_FLOW_EXTRAS_DISPARITY_URL = GOOGLE_DRIVE_BASE_URL.format('1ZJhraqgY1sL4UfHBrVojttCbvNAXfdj0')

STEREO_FLOW_ROOT_DIRPATH = os.path.join('data', 'kitti_stereo_flow', 'training')

STEREO_FLOW_EXTRAS_ROOT_DIRPATH = os.path.join(
    'data', 'kitti_stereo_flow_extras', 'training')

STEREO_FLOW_EXTRAS_DISPARITY_DIRPATH = os.path.join(
    STEREO_FLOW_EXTRAS_ROOT_DIRPATH, 'disp_occ_pseudo_gt')

STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH = os.path.join(
    STEREO_FLOW_EXTRAS_ROOT_DIRPATH, 'kitti_2012_disp_occ_pseudo_gt.zip')

STEREO_FLOW_IMAGE0_DIRPATH = os.path.join(STEREO_FLOW_ROOT_DIRPATH, 'image_2')

STEREO_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH = \
    os.path.join(TRAIN_REFS_DIRPATH, 'kitti_stereo_flow_all_disparity_pseudo.txt')
STEREO_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH = \
    os.path.join(TRAIN_REFS_DIRPATH, 'kitti_stereo_flow_train_disparity_pseudo.txt')
STEREO_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'kitti_stereo_flow_test_disparity_pseudo.txt')

# KITTI 2015 dataset (scene flow)
SCENE_FLOW_EXTRAS_DISPARITY_URL = GOOGLE_DRIVE_BASE_URL.format('14NGQp9CwIVNAK8ZQ6GSNeGraFGtVGOce')

SCENE_FLOW_ROOT_DIRPATH = os.path.join('data', 'kitti_scene_flow', 'training')

SCENE_FLOW_EXTRAS_ROOT_DIRPATH = os.path.join(
    'data', 'kitti_scene_flow_extras', 'training')

SCENE_FLOW_EXTRAS_DISPARITY_DIRPATH = os.path.join(
    SCENE_FLOW_EXTRAS_ROOT_DIRPATH, 'disp_occ_0_pseudo_gt')

SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH = os.path.join(
    SCENE_FLOW_EXTRAS_ROOT_DIRPATH, 'kitti_2015_disp_occ_0_pseudo_gt.zip')

SCENE_FLOW_IMAGE0_DIRPATH = os.path.join(SCENE_FLOW_ROOT_DIRPATH, 'image_2')

SCENE_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH = \
    os.path.join(TRAIN_REFS_DIRPATH, 'kitti_scene_flow_all_disparity_pseudo.txt')
SCENE_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH = \
    os.path.join(TRAIN_REFS_DIRPATH, 'kitti_scene_flow_train_disparity_pseudo.txt')
SCENE_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'kitti_scene_flow_test_disparity_pseudo.txt')

'''
Downloading pseudo ground truth from GANet
'''
# KITTI 2012 dataset (stereo flow)
if not os.path.exists(STEREO_FLOW_EXTRAS_ROOT_DIRPATH):
    os.makedirs(STEREO_FLOW_EXTRAS_ROOT_DIRPATH)

if not os.path.exists(STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH):
    print('Downloading stereo flow pseudo groundtruth disparity to {}'.format(
        STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH))
    gdown.download(STEREO_FLOW_EXTRAS_DISPARITY_URL, STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH, quiet=False)
else:
    print('Found stereo flow pseudo groundtruthd disparity at {}'.format(
        STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH))

with zipfile.ZipFile(STEREO_FLOW_EXTRAS_DISPARITY_FILEPATH, 'r') as z:
    z.extractall(STEREO_FLOW_EXTRAS_ROOT_DIRPATH)

stereo_flow_image0_paths = sorted(glob.glob(os.path.join(STEREO_FLOW_IMAGE0_DIRPATH, '*_10.png')))
stereo_flow_extras_disparity_paths = sorted(glob.glob(os.path.join(STEREO_FLOW_EXTRAS_DISPARITY_DIRPATH, '*_10.png')))

assert len(stereo_flow_image0_paths) == len(stereo_flow_extras_disparity_paths)

stereo_flow_extras_train_disparity_paths = stereo_flow_extras_disparity_paths[0:160]
stereo_flow_extras_test_disparity_paths = stereo_flow_extras_disparity_paths[160:]

# Write all paths to disk
print('Storing all {} stereo flow pseduo ground truth disparity file paths into: {}'.format(
    len(stereo_flow_extras_disparity_paths),
    STEREO_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH))
data_utils.write_paths(
    STEREO_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH,
    stereo_flow_extras_disparity_paths)

# Write training paths to disk
print('Storing {} training stereo flow pseudo ground truth disparity file paths into: {}'.format(
    len(stereo_flow_extras_train_disparity_paths),
    STEREO_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH))
data_utils.write_paths(
    STEREO_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH,
    stereo_flow_extras_train_disparity_paths)

# Write testing paths to disk
print('Storing {} testing pseudo stereo flow ground truth disparity file paths into: {}'.format(
    len(stereo_flow_extras_test_disparity_paths),
    STEREO_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH))
data_utils.write_paths(
    STEREO_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH,
    stereo_flow_extras_test_disparity_paths)

# KITTI 2015 dataset (scene flow)
if not os.path.exists(SCENE_FLOW_EXTRAS_ROOT_DIRPATH):
    os.makedirs(SCENE_FLOW_EXTRAS_ROOT_DIRPATH)

if not os.path.exists(SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH):
    print('Downloading scene flow pseudo groundtruth disparity to {}'.format(
        SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH))
    gdown.download(SCENE_FLOW_EXTRAS_DISPARITY_URL, SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH, quiet=False)
else:
    print('Found scene flow pseudo groundtruthd disparity at {}'.format(
        SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH))

with zipfile.ZipFile(SCENE_FLOW_EXTRAS_DISPARITY_FILEPATH, 'r') as z:
    z.extractall(SCENE_FLOW_EXTRAS_ROOT_DIRPATH)

scene_flow_image0_paths = sorted(glob.glob(os.path.join(SCENE_FLOW_IMAGE0_DIRPATH, '*_10.png')))
scene_flow_extras_disparity_paths = sorted(glob.glob(os.path.join(SCENE_FLOW_EXTRAS_DISPARITY_DIRPATH, '*_10.png')))

assert len(scene_flow_image0_paths) == len(scene_flow_extras_disparity_paths)

scene_flow_extras_train_disparity_paths = scene_flow_extras_disparity_paths[0:160]
scene_flow_extras_test_disparity_paths = scene_flow_extras_disparity_paths[160:]

# Write all paths to disk
print('Storing all {} scene flow pseduo ground truth disparity file paths into: {}'.format(
    len(scene_flow_extras_disparity_paths),
    SCENE_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH))
data_utils.write_paths(
    SCENE_FLOW_EXTRAS_ALL_DISPARITY_FILEPATH,
    scene_flow_extras_disparity_paths)

# Write training paths to disk
print('Storing {} training scene flow pseudo ground truth disparity file paths into: {}'.format(
    len(scene_flow_extras_train_disparity_paths),
    SCENE_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH))
data_utils.write_paths(
    SCENE_FLOW_EXTRAS_TRAIN_DISPARITY_FILEPATH,
    scene_flow_extras_train_disparity_paths)

# Write testing paths to disk
print('Storing {} testing scene flow pseudo ground truth disparity file paths into: {}'.format(
    len(scene_flow_extras_test_disparity_paths),
    SCENE_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH))
data_utils.write_paths(
    SCENE_FLOW_EXTRAS_TEST_DISPARITY_FILEPATH,
    scene_flow_extras_test_disparity_paths)
