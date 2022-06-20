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

import os, sys, glob
sys.path.insert(0, 'src')
import data_utils
import numpy as np

'''
Paths for KITTI dataset
'''
KITTI_RAW_DATA_DIRPATH = os.path.join('data', 'kitti_raw_data')
KITTI_STEREO_FLOW_DIRPATH = os.path.join('data', 'kitti_stereo_flow', 'training')
KITTI_SCENE_FLOW_DIRPATH = os.path.join('data', 'kitti_scene_flow', 'training')
KITTI_SEGMENTATION_DIRPATH = os.path.join('data', 'kitti_segmentation')

'''
Calibration Paths
'''
KITTI_INTRINSICS_FILENAME = 'calib_cam_to_cam.txt'

'''
Output filepaths to hold lists of input data paths
'''
TRAIN_OUTPUT_REF_DIRPATH = os.path.join('training', 'kitti')
VAL_OUTPUT_REF_DIRPATH = os.path.join('validation', 'kitti')
TEST_OUTPUT_REF_DIRPATH = os.path.join('testing', 'kitti')

KITTI_STEREO_FLOW_EXTRAS_DIRPATH = os.path.join('data', 'kitti_stereo_flow_extras', 'training')
KITTI_SCENE_FLOW_EXTRAS_DIRPATH = os.path.join('data', 'kitti_scene_flow_extras', 'training')

# KITTI raw dataset
KITTI_TRAIN_IMAGE0_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_train_image0.txt')
KITTI_TRAIN_IMAGE1_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_train_image1.txt')

# KITTI stereo flow dataset
KITTI_STEREO_FLOW_ALL_IMAGE0_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_all_image0.txt')
KITTI_STEREO_FLOW_ALL_IMAGE1_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_all_image1.txt')
KITTI_STEREO_FLOW_ALL_DISPARITY_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_all_disparity.txt')
KITTI_STEREO_FLOW_ALL_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_all_intrinsics.txt')
KITTI_STEREO_FLOW_ALL_LEFT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_all_left_intrinsics.txt')
KITTI_STEREO_FLOW_ALL_RIGHT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_all_right_intrinsics.txt')
KITTI_STEREO_FLOW_ALL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_all_left_focal_length_baseline.txt')
KITTI_STEREO_FLOW_ALL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_all_right_focal_length_baseline.txt')

KITTI_STEREO_FLOW_TRAIN_IMAGE0_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_train_image0.txt')
KITTI_STEREO_FLOW_TRAIN_IMAGE1_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_train_image1.txt')
KITTI_STEREO_FLOW_TRAIN_DISPARITY_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_train_disparity.txt')
KITTI_STEREO_FLOW_TRAIN_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_train_intrinsics.txt')
KITTI_STEREO_FLOW_TRAIN_LEFT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_train_left_intrinsics.txt')
KITTI_STEREO_FLOW_TRAIN_RIGHT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_train_right_intrinsics.txt')
KITTI_STEREO_FLOW_TRAIN_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_train_left_focal_length_baseline.txt')
KITTI_STEREO_FLOW_TRAIN_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_train_right_focal_length_baseline.txt')

KITTI_STEREO_FLOW_VAL_IMAGE0_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_val_image0.txt')
KITTI_STEREO_FLOW_VAL_IMAGE1_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_val_image1.txt')
KITTI_STEREO_FLOW_VAL_DISPARITY_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_val_disparity.txt')
KITTI_STEREO_FLOW_VAL_LEFT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_val_left_intrinsics.txt')
KITTI_STEREO_FLOW_VAL_RIGHT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_val_right_intrinsics.txt')
KITTI_STEREO_FLOW_VAL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_val_left_focal_length_baseline.txt')
KITTI_STEREO_FLOW_VAL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_stereo_flow_val_right_focal_length_baseline.txt')


# KITTI scene flow dataset
KITTI_SCENE_FLOW_ALL_IMAGE0_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_all_image0.txt')
KITTI_SCENE_FLOW_ALL_IMAGE1_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_all_image1.txt')
KITTI_SCENE_FLOW_ALL_DISPARITY_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_all_disparity.txt')
KITTI_SCENE_FLOW_ALL_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_all_intrinsics.txt')
KITTI_SCENE_FLOW_ALL_LEFT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_all_left_intrinsics.txt')
KITTI_SCENE_FLOW_ALL_RIGHT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_all_right_intrinsics.txt')
KITTI_SCENE_FLOW_ALL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_all_left_focal_length_baseline.txt')
KITTI_SCENE_FLOW_ALL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_all_right_focal_length_baseline.txt')
KITTI_SCENE_FLOW_ALL_SEG0_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_all_seg0.txt')
KITTI_SCENE_FLOW_ALL_SEG1_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_all_seg1.txt')

KITTI_SCENE_FLOW_TRAIN_IMAGE0_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_train_image0.txt')
KITTI_SCENE_FLOW_TRAIN_IMAGE1_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_train_image1.txt')
KITTI_SCENE_FLOW_TRAIN_DISPARITY_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_train_disparity.txt')
KITTI_SCENE_FLOW_TRAIN_LEFT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_train_left_intrinsics.txt')
KITTI_SCENE_FLOW_TRAIN_RIGHT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_train_right_intrinsics.txt')
KITTI_SCENE_FLOW_TRAIN_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_train_left_focal_length_baseline.txt')
KITTI_SCENE_FLOW_TRAIN_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_train_right_focal_length_baseline.txt')
KITTI_SCENE_FLOW_TRAIN_SEG0_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_train_seg0.txt')
KITTI_SCENE_FLOW_TRAIN_SEG1_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_train_seg1.txt')

KITTI_SCENE_FLOW_VAL_IMAGE0_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_val_image0.txt')
KITTI_SCENE_FLOW_VAL_IMAGE1_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_val_image1.txt')
KITTI_SCENE_FLOW_VAL_DISPARITY_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_val_disparity.txt')
KITTI_SCENE_FLOW_VAL_LEFT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_val_left_intrinsics.txt')
KITTI_SCENE_FLOW_VAL_RIGHT_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_val_right_intrinsics.txt')
KITTI_SCENE_FLOW_VAL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_val_left_focal_length_baseline.txt')
KITTI_SCENE_FLOW_VAL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_val_right_focal_length_baseline.txt')
KITTI_SCENE_FLOW_VAL_SEG0_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_val_seg0.txt')
KITTI_SCENE_FLOW_VAL_SEG1_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'kitti_scene_flow_val_seg1.txt')


def get_seg(image_path):
    path, image_filename = os.path.split(image_path)
    seg_filename = 'pred_' + image_filename.split('.')[0] + '.npy'
    return os.path.join(KITTI_SEGMENTATION_DIRPATH, path, seg_filename)


'''
Create output directories
'''
for dirpath in [TRAIN_OUTPUT_REF_DIRPATH, VAL_OUTPUT_REF_DIRPATH, TEST_OUTPUT_REF_DIRPATH]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


'''
Generate dataset paths for KITTI raw
'''
kitti_train_image0_paths = sorted(
    glob.glob(os.path.join(KITTI_RAW_DATA_DIRPATH, '*', '*', 'image_02', 'data', '*.png')))

kitti_train_image1_paths = sorted(
    glob.glob(os.path.join(KITTI_RAW_DATA_DIRPATH, '*', '*', 'image_03', 'data', '*.png')))

for image0_path, image1_path in zip(kitti_train_image0_paths, kitti_train_image1_paths):

    image0_dirpath = os.path.join(*(image0_path.split(os.sep)[0:-3]))
    image1_dirpath = os.path.join(*(image1_path.split(os.sep)[0:-3]))

    image0_filename = os.path.basename(image0_path)
    image1_filename = os.path.basename(image1_path)

    assert image0_dirpath == image1_dirpath, \
        'Mis-matched directory {}, {}'.format(image0_dirpath, image1_dirpath)
    assert image0_filename == image1_filename,  \
        'Mis-matched filename {}, {}'.format(image0_filename, image1_filename)

print('Storing %d left KITTI training image file paths into: %s' %
    (len(kitti_train_image0_paths), KITTI_TRAIN_IMAGE0_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_TRAIN_IMAGE0_OUTPUT_FILEPATH, kitti_train_image0_paths)

print('Storing %d right KITTI training image file paths into: %s' %
    (len(kitti_train_image1_paths), KITTI_TRAIN_IMAGE1_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_TRAIN_IMAGE1_OUTPUT_FILEPATH, kitti_train_image1_paths)


'''
Generate dataset paths for KITTI stereo flow (KITTI 2012)
'''
kitti_stereo_flow_image0_paths = sorted(
    glob.glob(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'image_2', '*_10.png')))
kitti_stereo_flow_image1_paths = sorted(
    glob.glob(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'image_3', '*_10.png')))
kitti_stereo_flow_disparity_paths = sorted(
    glob.glob(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'disp_occ', '*_10.png')))
kitti_stereo_flow_intrinsics_paths = sorted(
    glob.glob(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'calib', '*.txt')))


if not os.path.exists(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'intrinsics_left')):
    os.makedirs(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'intrinsics_left'))

if not os.path.exists(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'intrinsics_right')):
    os.makedirs(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'intrinsics_right'))

if not os.path.exists(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'focal_length_baseline_left')):
    os.makedirs(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'focal_length_baseline_left'))

if not os.path.exists(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'focal_length_baseline_right')):
    os.makedirs(os.path.join(KITTI_STEREO_FLOW_DIRPATH, 'focal_length_baseline_right'))

kitti_stereo_flow_left_intrinsics_paths = []
kitti_stereo_flow_right_intrinsics_paths = []
kitti_stereo_flow_left_focal_length_baseline_paths = []
kitti_stereo_flow_right_focal_length_baseline_paths = []

for intrinsics_path in kitti_stereo_flow_intrinsics_paths:

    intrinsics_left_path = intrinsics_path \
        .replace('calib', 'intrinsics_left') \
        .replace(KITTI_INTRINSICS_FILENAME, 'intrinsics_left') \
        .replace('txt', 'npy')

    intrinsics_right_path = intrinsics_path \
        .replace('calib', 'intrinsics_right') \
        .replace(KITTI_INTRINSICS_FILENAME, 'intrinsics_right') \
        .replace('txt', 'npy')

    # Example: data/kitti_raw_data_unsupervised_segmentation/2011_09_26/focal_length_baseline2.npy
    focal_length_baseline_left_path = intrinsics_path \
        .replace('calib', 'focal_length_baseline_left') \
        .replace(KITTI_INTRINSICS_FILENAME, 'focal_length_baseline_left') \
        .replace('txt', 'npy')

    focal_length_baseline_right_path = intrinsics_path \
        .replace('calib', 'focal_length_baseline_right') \
        .replace(KITTI_INTRINSICS_FILENAME, 'focal_length_baseline_right') \
        .replace('txt', 'npy')

    calib = data_utils.load_calibration(intrinsics_path)
    camera_left = np.reshape(calib['P2'], [3, 4]).astype(np.float32)
    camera_right = np.reshape(calib['P3'], [3, 4]).astype(np.float32)

    # Focal length of the cameras
    focal_length_left = camera_left[0, 0]
    focal_length_right = camera_right[0, 0]

    # camera2 is left of camera0 (-6cm) camera3 is right of camera2 (+53.27cm)
    translation_left = camera_left[0, 3] / focal_length_left
    translation_right = camera_right[0, 3] / focal_length_right
    baseline = translation_left - translation_right

    position_left = camera_left[0:3, 3] / focal_length_left
    position_right = camera_right[0:3, 3] / focal_length_right

    # Baseline should be just translation along x
    error_baseline = np.abs(baseline - np.linalg.norm(position_left - position_right))
    assert error_baseline < 0.01, \
        'baseline={}'.format(baseline)

    # Concatenate together as fB
    focal_length_baseline_left = np.concatenate([
        np.expand_dims(focal_length_left, axis=-1),
        np.expand_dims(baseline, axis=-1)],
        axis=-1)

    focal_length_baseline_right = np.concatenate([
        np.expand_dims(focal_length_right, axis=-1),
        np.expand_dims(baseline, axis=-1)],
        axis=-1)

    # Extract camera parameters
    intrinsics_left = camera_left[:3, :3]
    intrinsics_right = camera_right[:3, :3]

    np.save(focal_length_baseline_left_path, focal_length_baseline_left)
    np.save(focal_length_baseline_right_path, focal_length_baseline_right)

    np.save(intrinsics_left_path, intrinsics_left)
    np.save(intrinsics_right_path, intrinsics_right)

    kitti_stereo_flow_left_intrinsics_paths.append(intrinsics_left_path)
    kitti_stereo_flow_right_intrinsics_paths.append(intrinsics_right_path)
    kitti_stereo_flow_left_focal_length_baseline_paths.append(focal_length_baseline_left_path)
    kitti_stereo_flow_right_focal_length_baseline_paths.append(focal_length_baseline_right_path)

kitti_stereo_flow_paths = zip(
    kitti_stereo_flow_image0_paths,
    kitti_stereo_flow_image1_paths,
    kitti_stereo_flow_disparity_paths,
    kitti_stereo_flow_left_intrinsics_paths,
    kitti_stereo_flow_right_intrinsics_paths,
    kitti_stereo_flow_left_focal_length_baseline_paths,
    kitti_stereo_flow_right_focal_length_baseline_paths)

for inputs in kitti_stereo_flow_paths:

    image0_path, \
        image1_path, \
        disparity_path, \
        left_intrinsics_path, \
        right_intrinsics_path, \
        left_focal_length_baseline_path, \
        right_focal_length_baseline_path = inputs

    image0_filename = os.path.basename(image0_path)
    image1_filename = os.path.basename(image1_path)
    disparity_filename = os.path.basename(disparity_path)
    left_intrinsics_filename = os.path.basename(left_intrinsics_path)
    right_intrinsics_filename = os.path.basename(right_intrinsics_path)
    left_focal_length_baseline_filename = os.path.basename(left_focal_length_baseline_path)
    right_focal_length_baseline_filename = os.path.basename(right_focal_length_baseline_path)

    assert image0_filename == image1_filename,  \
        'Mis-matched filename in stereo pair: {}, {}'.format(image0_filename, image1_filename)

    assert image0_filename == disparity_filename,  \
        'Mis-matched filename in disparity: {}, {}'.format(image0_filename, disparity_filename)

kitti_stereo_flow_train_image0_paths = kitti_stereo_flow_image0_paths[:160]
kitti_stereo_flow_train_image1_paths = kitti_stereo_flow_image1_paths[:160]
kitti_stereo_flow_train_disparity_paths = kitti_stereo_flow_disparity_paths[:160]
kitti_stereo_flow_train_left_intrinsics_paths = kitti_stereo_flow_left_intrinsics_paths[:160]
kitti_stereo_flow_train_right_intrinsics_paths = kitti_stereo_flow_right_intrinsics_paths[:160]
kitti_stereo_flow_train_left_focal_length_baseline_paths = kitti_stereo_flow_left_focal_length_baseline_paths[:160]
kitti_stereo_flow_train_right_focal_length_baseline_paths = kitti_stereo_flow_right_focal_length_baseline_paths[:160]

kitti_stereo_flow_val_image0_paths = kitti_stereo_flow_image0_paths[160:]
kitti_stereo_flow_val_image1_paths = kitti_stereo_flow_image1_paths[160:]
kitti_stereo_flow_val_disparity_paths = kitti_stereo_flow_disparity_paths[160:]
kitti_stereo_flow_val_left_intrinsics_paths = kitti_stereo_flow_left_intrinsics_paths[160:]
kitti_stereo_flow_val_right_intrinsics_paths = kitti_stereo_flow_right_intrinsics_paths[160:]
kitti_stereo_flow_val_left_focal_length_baseline_paths = kitti_stereo_flow_left_focal_length_baseline_paths[160:]
kitti_stereo_flow_val_right_focal_length_baseline_paths = kitti_stereo_flow_right_focal_length_baseline_paths[160:]


# Store all paths
print('Storing %d left KITTI stereo flow (KITTI 2012) image file paths into: %s' %
    (len(kitti_stereo_flow_image0_paths), KITTI_STEREO_FLOW_ALL_IMAGE0_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_ALL_IMAGE0_OUTPUT_FILEPATH, kitti_stereo_flow_image0_paths)

print('Storing %d right KITTI stereo flow (KITTI 2012) image file paths into: %s' %
    (len(kitti_stereo_flow_image1_paths), KITTI_STEREO_FLOW_ALL_IMAGE1_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_ALL_IMAGE1_OUTPUT_FILEPATH, kitti_stereo_flow_image1_paths)

print('Storing %d left KITTI stereo flow (KITTI 2012) disparity file paths into: %s' %
    (len(kitti_stereo_flow_disparity_paths), KITTI_STEREO_FLOW_ALL_DISPARITY_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_ALL_DISPARITY_OUTPUT_FILEPATH, kitti_stereo_flow_disparity_paths)

print('Storing %d left KITTI stereo flow (KITTI 2012) intrinsics file paths into: %s' %
    (len(kitti_stereo_flow_left_intrinsics_paths), KITTI_SCENE_FLOW_ALL_LEFT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_ALL_LEFT_INTRINSICS_OUTPUT_FILEPATH, kitti_stereo_flow_left_intrinsics_paths)

print('Storing %d right KITTI stereo flow (KITTI 2012) intrinsics file paths into: %s' %
    (len(kitti_stereo_flow_right_intrinsics_paths), KITTI_STEREO_FLOW_ALL_RIGHT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_ALL_RIGHT_INTRINSICS_OUTPUT_FILEPATH, kitti_stereo_flow_right_intrinsics_paths)

print('Storing %d left KITTI stereo flow (KITTI 2012) focal_length_baseline file paths into: %s' %
    (len(kitti_stereo_flow_left_focal_length_baseline_paths), KITTI_STEREO_FLOW_ALL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_ALL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_stereo_flow_left_focal_length_baseline_paths)

print('Storing %d right KITTI stereo flow (KITTI 2012) focal_length_baseline file paths into: %s' %
    (len(kitti_stereo_flow_right_focal_length_baseline_paths), KITTI_STEREO_FLOW_ALL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_ALL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_stereo_flow_right_focal_length_baseline_paths)

# Store training paths
print('Storing %d left KITTI stereo flow (KITTI 2012) training image file paths into: %s' %
    (len(kitti_stereo_flow_train_image0_paths), KITTI_STEREO_FLOW_TRAIN_IMAGE0_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_TRAIN_IMAGE0_OUTPUT_FILEPATH, kitti_stereo_flow_train_image0_paths)

print('Storing %d right KITTI stereo flow (KITTI 2012) training image file paths into: %s' %
    (len(kitti_stereo_flow_train_image1_paths), KITTI_STEREO_FLOW_TRAIN_IMAGE1_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_TRAIN_IMAGE1_OUTPUT_FILEPATH, kitti_stereo_flow_train_image1_paths)

print('Storing %d left KITTI stereo flow (KITTI 2012) training disparity paths into: %s' %
    (len(kitti_stereo_flow_train_disparity_paths), KITTI_STEREO_FLOW_TRAIN_DISPARITY_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_TRAIN_DISPARITY_OUTPUT_FILEPATH, kitti_stereo_flow_train_disparity_paths)

print('Storing %d left KITTI stereo flow (KITTI 2012) training intrinsics paths into: %s' %
    (len(kitti_stereo_flow_train_left_intrinsics_paths), KITTI_STEREO_FLOW_TRAIN_LEFT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_TRAIN_LEFT_INTRINSICS_OUTPUT_FILEPATH, kitti_stereo_flow_train_left_intrinsics_paths)

print('Storing %d right KITTI stereo flow (KITTI 2012) training intrinsics paths into: %s' %
    (len(kitti_stereo_flow_train_right_intrinsics_paths), KITTI_STEREO_FLOW_TRAIN_RIGHT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_TRAIN_RIGHT_INTRINSICS_OUTPUT_FILEPATH, kitti_stereo_flow_train_right_intrinsics_paths)

print('Storing %d left KITTI stereo flow (KITTI 2012) training focal_length_baseline paths into: %s' %
    (len(kitti_stereo_flow_train_left_focal_length_baseline_paths), KITTI_STEREO_FLOW_TRAIN_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_TRAIN_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_stereo_flow_train_left_focal_length_baseline_paths)

print('Storing %d right KITTI stereo flow (KITTI 2012) training focal_length_baseline paths into: %s' %
    (len(kitti_stereo_flow_train_right_focal_length_baseline_paths), KITTI_STEREO_FLOW_TRAIN_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_TRAIN_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_stereo_flow_train_right_focal_length_baseline_paths)

# Store validation paths
print('Storing %d left KITTI stereo flow (KITTI 2012) validation image file paths into: %s' %
    (len(kitti_stereo_flow_val_image0_paths), KITTI_STEREO_FLOW_VAL_IMAGE0_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_VAL_IMAGE0_OUTPUT_FILEPATH, kitti_stereo_flow_val_image0_paths)

print('Storing %d right KITTI stereo flow (KITTI 2012) validation image file paths into: %s' %
    (len(kitti_stereo_flow_val_image1_paths), KITTI_STEREO_FLOW_VAL_IMAGE1_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_VAL_IMAGE1_OUTPUT_FILEPATH, kitti_stereo_flow_val_image1_paths)

print('Storing %d left KITTI stereo flow (KITTI 2012) validation disparity paths into: %s' %
    (len(kitti_stereo_flow_val_disparity_paths), KITTI_STEREO_FLOW_VAL_DISPARITY_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_VAL_DISPARITY_OUTPUT_FILEPATH, kitti_stereo_flow_val_disparity_paths)

print('Storing %d left KITTI stereo flow (KITTI 2012) validation intrinsics paths into: %s' %
    (len(kitti_stereo_flow_val_left_intrinsics_paths), KITTI_STEREO_FLOW_VAL_LEFT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_VAL_LEFT_INTRINSICS_OUTPUT_FILEPATH, kitti_stereo_flow_val_left_intrinsics_paths)

print('Storing %d right KITTI stereo flow (KITTI 2012) validation intrinsics paths into: %s' %
    (len(kitti_stereo_flow_val_right_intrinsics_paths), KITTI_STEREO_FLOW_VAL_RIGHT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_VAL_RIGHT_INTRINSICS_OUTPUT_FILEPATH, kitti_stereo_flow_val_right_intrinsics_paths)

print('Storing %d left KITTI stereo flow (KITTI 2012) validation focal_length_baseline paths into: %s' %
    (len(kitti_stereo_flow_val_left_focal_length_baseline_paths), KITTI_STEREO_FLOW_VAL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_VAL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_stereo_flow_val_left_focal_length_baseline_paths)

print('Storing %d right KITTI stereo flow (KITTI 2012) validation focal_length_baseline paths into: %s' %
    (len(kitti_stereo_flow_val_right_focal_length_baseline_paths), KITTI_STEREO_FLOW_VAL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_STEREO_FLOW_VAL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_stereo_flow_val_right_focal_length_baseline_paths)


'''
Generate dataset paths for KITTI scene flow (KITTI 2015)
'''
kitti_scene_flow_image0_paths = sorted(
    glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'image_2', '*_10.png')))
kitti_scene_flow_image1_paths = sorted(
    glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'image_3', '*_10.png')))
kitti_scene_flow_disparity_paths = sorted(
    glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'disp_occ_0', '*_10.png')))
kitti_scene_flow_intrinsics_paths = sorted(
    glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'calib_cam_to_cam', '*.txt')))

kitti_scene_flow_seg0_paths = [get_seg(x) for x in kitti_scene_flow_image0_paths]
kitti_scene_flow_seg1_paths = [get_seg(x) for x in kitti_scene_flow_image1_paths]


if not os.path.exists(os.path.join(KITTI_SCENE_FLOW_EXTRAS_DIRPATH, 'intrinsics_left')):
    os.makedirs(os.path.join(KITTI_SCENE_FLOW_EXTRAS_DIRPATH, 'intrinsics_left'))

if not os.path.exists(os.path.join(KITTI_SCENE_FLOW_EXTRAS_DIRPATH, 'intrinsics_right')):
    os.makedirs(os.path.join(KITTI_SCENE_FLOW_EXTRAS_DIRPATH, 'intrinsics_right'))

if not os.path.exists(os.path.join(KITTI_SCENE_FLOW_EXTRAS_DIRPATH, 'focal_length_baseline_left')):
    os.makedirs(os.path.join(KITTI_SCENE_FLOW_EXTRAS_DIRPATH, 'focal_length_baseline_left'))

if not os.path.exists(os.path.join(KITTI_SCENE_FLOW_EXTRAS_DIRPATH, 'focal_length_baseline_right')):
    os.makedirs(os.path.join(KITTI_SCENE_FLOW_EXTRAS_DIRPATH, 'focal_length_baseline_right'))

kitti_scene_flow_left_intrinsics_paths = []
kitti_scene_flow_right_intrinsics_paths = []
kitti_scene_flow_left_focal_length_baseline_paths = []
kitti_scene_flow_right_focal_length_baseline_paths = []

for intrinsics_path in kitti_scene_flow_intrinsics_paths:

    intrinsics_left_path = intrinsics_path \
        .replace(KITTI_SCENE_FLOW_DIRPATH, KITTI_SCENE_FLOW_EXTRAS_DIRPATH) \
        .replace('calib_cam_to_cam', 'intrinsics_left') \
        .replace(KITTI_INTRINSICS_FILENAME, 'intrinsics_left') \
        .replace('txt', 'npy')

    intrinsics_right_path = intrinsics_path \
        .replace(KITTI_SCENE_FLOW_DIRPATH, KITTI_SCENE_FLOW_EXTRAS_DIRPATH) \
        .replace('calib_cam_to_cam', 'intrinsics_right') \
        .replace(KITTI_INTRINSICS_FILENAME, 'intrinsics_right') \
        .replace('txt', 'npy')

    focal_length_baseline_left_path = intrinsics_path \
        .replace(KITTI_SCENE_FLOW_DIRPATH, KITTI_SCENE_FLOW_EXTRAS_DIRPATH) \
        .replace('calib_cam_to_cam', 'focal_length_baseline_left') \
        .replace(KITTI_INTRINSICS_FILENAME, 'focal_length_baseline_left') \
        .replace('txt', 'npy')

    focal_length_baseline_right_path = intrinsics_path \
        .replace(KITTI_SCENE_FLOW_DIRPATH, KITTI_SCENE_FLOW_EXTRAS_DIRPATH) \
        .replace('calib_cam_to_cam', 'focal_length_baseline_right') \
        .replace(KITTI_INTRINSICS_FILENAME, 'focal_length_baseline_right') \
        .replace('txt', 'npy')

    calib = data_utils.load_calibration(intrinsics_path)
    camera_left = np.reshape(calib['P_rect_02'], [3, 4]).astype(np.float32)
    camera_right = np.reshape(calib['P_rect_03'], [3, 4]).astype(np.float32)

    # Focal length of the cameras
    focal_length_left = camera_left[0, 0]
    focal_length_right = camera_right[0, 0]

    # camera2 is left of camera0 (-6cm) camera3 is right of camera2 (+53.27cm)
    translation_left = camera_left[0, 3] / focal_length_left
    translation_right = camera_right[0, 3] / focal_length_right
    baseline = translation_left - translation_right

    position_left = camera_left[0:3, 3] / focal_length_left
    position_right = camera_right[0:3, 3] / focal_length_right

    # Baseline should be just translation along x
    error_baseline = np.abs(baseline - np.linalg.norm(position_left - position_right))
    assert error_baseline < 0.01, \
        'baseline={}'.format(baseline)

    # Concatenate together as fB
    focal_length_baseline_left = np.concatenate([
        np.expand_dims(focal_length_left, axis=-1),
        np.expand_dims(baseline, axis=-1)],
        axis=-1)

    focal_length_baseline_right = np.concatenate([
        np.expand_dims(focal_length_right, axis=-1),
        np.expand_dims(baseline, axis=-1)],
        axis=-1)

    # Extract camera parameters
    intrinsics_left = camera_left[:3, :3]
    intrinsics_right = camera_right[:3, :3]

    np.save(focal_length_baseline_left_path, focal_length_baseline_left)
    np.save(focal_length_baseline_right_path, focal_length_baseline_right)

    np.save(intrinsics_left_path, intrinsics_left)
    np.save(intrinsics_right_path, intrinsics_right)

    kitti_scene_flow_left_intrinsics_paths.append(intrinsics_left_path)
    kitti_scene_flow_right_intrinsics_paths.append(intrinsics_right_path)
    kitti_scene_flow_left_focal_length_baseline_paths.append(focal_length_baseline_left_path)
    kitti_scene_flow_right_focal_length_baseline_paths.append(focal_length_baseline_right_path)

kitti_scene_flow_paths = zip(
    kitti_scene_flow_image0_paths,
    kitti_scene_flow_image1_paths,
    kitti_scene_flow_disparity_paths,
    kitti_scene_flow_left_intrinsics_paths,
    kitti_scene_flow_right_intrinsics_paths,
    kitti_scene_flow_left_focal_length_baseline_paths,
    kitti_scene_flow_right_focal_length_baseline_paths)

for inputs in kitti_scene_flow_paths:

    image0_path, \
        image1_path, \
        disparity_path, \
        left_intrinsics_path, \
        right_intrinsics_path, \
        left_focal_length_baseline_path, \
        right_focal_length_baseline_path = inputs

    image0_filename = os.path.basename(image0_path)
    image1_filename = os.path.basename(image1_path)
    disparity_filename = os.path.basename(disparity_path)
    left_intrinsics_filename = os.path.basename(left_intrinsics_path)
    right_intrinsics_filename = os.path.basename(right_intrinsics_path)
    left_focal_length_baseline_filename = os.path.basename(left_focal_length_baseline_path)
    right_focal_length_baseline_filename = os.path.basename(right_focal_length_baseline_path)

    assert image0_filename == image1_filename,  \
        'Mis-matched filename in stereo pair: {}, {}'.format(image0_filename, image1_filename)

    assert image0_filename == disparity_filename,  \
        'Mis-matched filename in disparity: {}, {}'.format(image0_filename, disparity_filename)

kitti_scene_flow_train_image0_paths = kitti_scene_flow_image0_paths[:160]
kitti_scene_flow_train_image1_paths = kitti_scene_flow_image1_paths[:160]
kitti_scene_flow_train_disparity_paths = kitti_scene_flow_disparity_paths[:160]
kitti_scene_flow_train_left_intrinsics_paths = kitti_scene_flow_left_intrinsics_paths[:160]
kitti_scene_flow_train_right_intrinsics_paths = kitti_scene_flow_right_intrinsics_paths[:160]
kitti_scene_flow_train_left_focal_length_baseline_paths = kitti_scene_flow_left_focal_length_baseline_paths[:160]
kitti_scene_flow_train_right_focal_length_baseline_paths = kitti_scene_flow_right_focal_length_baseline_paths[:160]
kitti_scene_flow_train_seg0_paths = kitti_scene_flow_seg0_paths[:160]
kitti_scene_flow_train_seg1_paths = kitti_scene_flow_seg1_paths[:160]

kitti_scene_flow_val_image0_paths = kitti_scene_flow_image0_paths[160:]
kitti_scene_flow_val_image1_paths = kitti_scene_flow_image1_paths[160:]
kitti_scene_flow_val_disparity_paths = kitti_scene_flow_disparity_paths[160:]
kitti_scene_flow_val_left_intrinsics_paths = kitti_scene_flow_left_intrinsics_paths[160:]
kitti_scene_flow_val_right_intrinsics_paths = kitti_scene_flow_right_intrinsics_paths[160:]
kitti_scene_flow_val_left_focal_length_baseline_paths = kitti_scene_flow_left_focal_length_baseline_paths[160:]
kitti_scene_flow_val_right_focal_length_baseline_paths = kitti_scene_flow_right_focal_length_baseline_paths[160:]
kitti_scene_flow_val_seg0_paths = kitti_scene_flow_seg0_paths[160:]
kitti_scene_flow_val_seg1_paths = kitti_scene_flow_seg1_paths[160:]


# Store all paths
print('Storing %d left KITTI scene flow (KITTI 2015) image file paths into: %s' %
    (len(kitti_scene_flow_image0_paths), KITTI_SCENE_FLOW_ALL_IMAGE0_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_ALL_IMAGE0_OUTPUT_FILEPATH, kitti_scene_flow_image0_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) image file paths into: %s' %
    (len(kitti_scene_flow_image1_paths), KITTI_SCENE_FLOW_ALL_IMAGE1_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_ALL_IMAGE1_OUTPUT_FILEPATH, kitti_scene_flow_image1_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) disparity file paths into: %s' %
    (len(kitti_scene_flow_disparity_paths), KITTI_SCENE_FLOW_ALL_DISPARITY_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_ALL_DISPARITY_OUTPUT_FILEPATH, kitti_scene_flow_disparity_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) intrinsics file paths into: %s' %
    (len(kitti_scene_flow_left_intrinsics_paths), KITTI_SCENE_FLOW_ALL_LEFT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_ALL_LEFT_INTRINSICS_OUTPUT_FILEPATH, kitti_scene_flow_left_intrinsics_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) intrinsics file paths into: %s' %
    (len(kitti_scene_flow_right_intrinsics_paths), KITTI_SCENE_FLOW_ALL_RIGHT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_ALL_RIGHT_INTRINSICS_OUTPUT_FILEPATH, kitti_scene_flow_right_intrinsics_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) focal_length_baseline file paths into: %s' %
    (len(kitti_scene_flow_left_focal_length_baseline_paths), KITTI_SCENE_FLOW_ALL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_ALL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_scene_flow_left_focal_length_baseline_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) focal_length_baseline file paths into: %s' %
    (len(kitti_scene_flow_right_focal_length_baseline_paths), KITTI_SCENE_FLOW_ALL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_ALL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_scene_flow_right_focal_length_baseline_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) segmentation file paths into: %s' %
    (len(kitti_scene_flow_seg0_paths), KITTI_SCENE_FLOW_ALL_SEG0_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_ALL_SEG0_OUTPUT_FILEPATH, kitti_scene_flow_seg0_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) segmentation file paths into: %s' %
    (len(kitti_scene_flow_seg1_paths), KITTI_SCENE_FLOW_ALL_SEG1_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_ALL_SEG1_OUTPUT_FILEPATH, kitti_scene_flow_seg1_paths)

# Store training paths
print('Storing %d left KITTI scene flow (KITTI 2015) training image file paths into: %s' %
    (len(kitti_scene_flow_train_image0_paths), KITTI_SCENE_FLOW_TRAIN_IMAGE0_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_TRAIN_IMAGE0_OUTPUT_FILEPATH, kitti_scene_flow_train_image0_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) training image file paths into: %s' %
    (len(kitti_scene_flow_train_image1_paths), KITTI_SCENE_FLOW_TRAIN_IMAGE1_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_TRAIN_IMAGE1_OUTPUT_FILEPATH, kitti_scene_flow_train_image1_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) training disparity paths into: %s' %
    (len(kitti_scene_flow_train_disparity_paths), KITTI_SCENE_FLOW_TRAIN_DISPARITY_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_TRAIN_DISPARITY_OUTPUT_FILEPATH, kitti_scene_flow_train_disparity_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) training intrinsics paths into: %s' %
    (len(kitti_scene_flow_train_left_intrinsics_paths), KITTI_SCENE_FLOW_TRAIN_LEFT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_TRAIN_LEFT_INTRINSICS_OUTPUT_FILEPATH, kitti_scene_flow_train_left_intrinsics_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) training intrinsics paths into: %s' %
    (len(kitti_scene_flow_train_right_intrinsics_paths), KITTI_SCENE_FLOW_TRAIN_RIGHT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_TRAIN_RIGHT_INTRINSICS_OUTPUT_FILEPATH, kitti_scene_flow_train_right_intrinsics_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) training focal_length_baseline paths into: %s' %
    (len(kitti_scene_flow_train_left_focal_length_baseline_paths), KITTI_SCENE_FLOW_TRAIN_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_TRAIN_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_scene_flow_train_left_focal_length_baseline_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) training focal_length_baseline paths into: %s' %
    (len(kitti_scene_flow_train_right_focal_length_baseline_paths), KITTI_SCENE_FLOW_TRAIN_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_TRAIN_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_scene_flow_train_right_focal_length_baseline_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) training segmentation file paths into: %s' %
    (len(kitti_scene_flow_train_seg0_paths), KITTI_SCENE_FLOW_TRAIN_SEG0_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_TRAIN_SEG0_OUTPUT_FILEPATH, kitti_scene_flow_train_seg0_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) training segmentation file paths into: %s' %
    (len(kitti_scene_flow_val_seg1_paths), KITTI_SCENE_FLOW_VAL_SEG1_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_VAL_SEG1_OUTPUT_FILEPATH, kitti_scene_flow_val_seg1_paths)

# Store validation paths
print('Storing %d left KITTI scene flow (KITTI 2015) validation image file paths into: %s' %
    (len(kitti_scene_flow_val_image0_paths), KITTI_SCENE_FLOW_VAL_IMAGE0_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_VAL_IMAGE0_OUTPUT_FILEPATH, kitti_scene_flow_val_image0_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) validation image file paths into: %s' %
    (len(kitti_scene_flow_val_image1_paths), KITTI_SCENE_FLOW_VAL_IMAGE1_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_VAL_IMAGE1_OUTPUT_FILEPATH, kitti_scene_flow_val_image1_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) validation disparity paths into: %s' %
    (len(kitti_scene_flow_val_disparity_paths), KITTI_SCENE_FLOW_VAL_DISPARITY_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_VAL_DISPARITY_OUTPUT_FILEPATH, kitti_scene_flow_val_disparity_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) validation intrinsics paths into: %s' %
    (len(kitti_scene_flow_val_left_intrinsics_paths), KITTI_SCENE_FLOW_VAL_LEFT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_VAL_LEFT_INTRINSICS_OUTPUT_FILEPATH, kitti_scene_flow_val_left_intrinsics_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) validation intrinsics paths into: %s' %
    (len(kitti_scene_flow_val_right_intrinsics_paths), KITTI_SCENE_FLOW_VAL_RIGHT_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_VAL_RIGHT_INTRINSICS_OUTPUT_FILEPATH, kitti_scene_flow_val_right_intrinsics_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) validation focal_length_baseline paths into: %s' %
    (len(kitti_scene_flow_val_left_focal_length_baseline_paths), KITTI_SCENE_FLOW_VAL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_VAL_LEFT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_scene_flow_val_left_focal_length_baseline_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) validation focal_length_baseline paths into: %s' %
    (len(kitti_scene_flow_val_right_focal_length_baseline_paths), KITTI_SCENE_FLOW_VAL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_VAL_RIGHT_FOCALLENGTH_BASELINE_OUTPUT_FILEPATH, kitti_scene_flow_val_right_focal_length_baseline_paths)

print('Storing %d left KITTI scene flow (KITTI 2015) validation segmentation file paths into: %s' %
    (len(kitti_scene_flow_val_seg0_paths), KITTI_SCENE_FLOW_VAL_SEG0_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_VAL_SEG0_OUTPUT_FILEPATH, kitti_scene_flow_val_seg0_paths)

print('Storing %d right KITTI scene flow (KITTI 2015) validation segmentation file paths into: %s' %
    (len(kitti_scene_flow_val_seg1_paths), KITTI_SCENE_FLOW_VAL_SEG1_OUTPUT_FILEPATH))
data_utils.write_paths(
    KITTI_SCENE_FLOW_VAL_SEG1_OUTPUT_FILEPATH, kitti_scene_flow_val_seg1_paths)
