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

import os, sys, glob, argparse, shutil
import numpy as np
sys.path.insert(0, 'src')
import data_utils


'''
Paths for FlyingThings3D dataset
'''
SCENE_FLOW_ROOT_DIRPATH = os.path.join('data', 'scene_flow_datasets')

FLYINGTHINGS3D_ROOT_DIRPATH = \
    os.path.join(SCENE_FLOW_ROOT_DIRPATH, 'flyingthings3d')

FLYINGTHINGS3D_TEST_FRAMES_CLEANPASS_DIRPATH = \
    os.path.join(FLYINGTHINGS3D_ROOT_DIRPATH, 'frames_cleanpass', 'TEST')
FLYINGTHINGS3D_TEST_FRAMES_FINALPASS_DIRPATH = \
    os.path.join(FLYINGTHINGS3D_ROOT_DIRPATH, 'frames_finalpass', 'TEST')
FLYINGTHINGS3D_TEST_DISPARITY_DIRPATH = \
    os.path.join(FLYINGTHINGS3D_ROOT_DIRPATH, 'disparity', 'TEST')

FLYINGTHINGS3D_TEST_SEQUENCE_DIRPATH = [
    'A',
    'B',
    'C'
]

'''
Paths for outputs
'''
SCENE_FLOW_OUTPUT_DIRPATH = \
    os.path.join('data', 'scene_flow_datasets_extras')

FLYINGTHINGS3D_OUTPUT_DIRPATH = \
    os.path.join(SCENE_FLOW_OUTPUT_DIRPATH, 'flyingthings3d_extras')

FLYINGTHINGS3D_TEST_REF_DIRPATH = os.path.join('testing', 'flyingthings3d')

FLYINGTHINGS3D_TEST_IMAGE0_CLEANPASS_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_image0_cleanpass.txt')
FLYINGTHINGS3D_TEST_IMAGE1_CLEANPASS_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_image1_cleanpass.txt')
FLYINGTHINGS3D_TEST_IMAGE0_FINALPASS_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_image0_finalpass.txt')
FLYINGTHINGS3D_TEST_IMAGE1_FINALPASS_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_image1_finalpass.txt')
FLYINGTHINGS3D_TEST_DISPARITY0_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_disparity0.txt')
FLYINGTHINGS3D_TEST_DISPARITY1_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_disparity1.txt')

FLYINGTHINGS3D_TEST_IMAGE0_CLEANPASS_SUBSET_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_image0_cleanpass-{}.txt')
FLYINGTHINGS3D_TEST_IMAGE1_CLEANPASS_SUBSET_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_image1_cleanpass-{}.txt')
FLYINGTHINGS3D_TEST_IMAGE0_FINALPASS_SUBSET_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_image0_finalpass-{}.txt')
FLYINGTHINGS3D_TEST_IMAGE1_FINALPASS_SUBSET_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_image1_finalpass-{}.txt')
FLYINGTHINGS3D_TEST_DISPARITY0_SUBSET_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_disparity0-{}.txt')
FLYINGTHINGS3D_TEST_DISPARITY1_SUBSET_OUTPUT_FILEPATH = os.path.join(
    FLYINGTHINGS3D_TEST_REF_DIRPATH,
    'flyingthings3d_test_disparity1-{}.txt')


parser = argparse.ArgumentParser()

parser.add_argument('--paths_only',
    action='store_true', help='If set, then do not generate data, only paths')

args = parser.parse_args()


if not os.path.exists(FLYINGTHINGS3D_TEST_REF_DIRPATH):
    os.makedirs(FLYINGTHINGS3D_TEST_REF_DIRPATH)

flyingthings3d_test_image0_cleanpass_paths = []
flyingthings3d_test_image1_cleanpass_paths = []
flyingthings3d_test_image0_finalpass_paths = []
flyingthings3d_test_image1_finalpass_paths = []
flyingthings3d_test_disparity0_paths = []
flyingthings3d_test_disparity1_paths = []

for ref_dirpath in FLYINGTHINGS3D_TEST_SEQUENCE_DIRPATH:

    frames_cleanpass_dirpath = \
        os.path.join(FLYINGTHINGS3D_TEST_FRAMES_CLEANPASS_DIRPATH, ref_dirpath)
    frames_finalpass_dirpath = \
        os.path.join(FLYINGTHINGS3D_TEST_FRAMES_FINALPASS_DIRPATH, ref_dirpath)

    disparity_dirpath = os.path.join(FLYINGTHINGS3D_TEST_DISPARITY_DIRPATH, ref_dirpath)

    # Get image paths
    flyingthings3d_image0_cleanpass_paths = glob.glob(
        os.path.join(frames_cleanpass_dirpath, '*', 'left', '*.png'))
    flyingthings3d_image0_finalpass_paths = glob.glob(
        os.path.join(frames_finalpass_dirpath, '*', 'left', '*.png'))

    flyingthings3d_image0_cleanpass_paths = sorted(flyingthings3d_image0_cleanpass_paths)
    flyingthings3d_image0_finalpass_paths = sorted(flyingthings3d_image0_finalpass_paths)

    flyingthings3d_image1_cleanpass_paths = glob.glob(
        os.path.join(frames_cleanpass_dirpath, '*', 'right', '*.png'))
    flyingthings3d_image1_finalpass_paths = glob.glob(
        os.path.join(frames_finalpass_dirpath, '*', 'right', '*.png'))

    flyingthings3d_image1_cleanpass_paths = sorted(flyingthings3d_image1_cleanpass_paths)
    flyingthings3d_image1_finalpass_paths = sorted(flyingthings3d_image1_finalpass_paths)

    # Get disparity paths
    flyingthings3d_disparity0_paths = glob.glob(
        os.path.join(disparity_dirpath, '*', 'left', '*.pfm'))

    flyingthings3d_disparity0_paths = sorted(flyingthings3d_disparity0_paths)

    flyingthings3d_disparity1_paths = glob.glob(
        os.path.join(disparity_dirpath, '*', 'right', '*.pfm'))

    flyingthings3d_disparity1_paths = sorted(flyingthings3d_disparity1_paths)

    n_sample = len(flyingthings3d_disparity0_paths)

    # Error check
    data_paths = [
        flyingthings3d_image0_cleanpass_paths,
        flyingthings3d_image1_cleanpass_paths,
        flyingthings3d_image0_finalpass_paths,
        flyingthings3d_image1_finalpass_paths,
        flyingthings3d_disparity0_paths,
        flyingthings3d_disparity1_paths
    ]

    for paths in data_paths:
        assert len(paths) == n_sample

    data_paths = zip(
        flyingthings3d_image0_cleanpass_paths,
        flyingthings3d_image1_cleanpass_paths,
        flyingthings3d_image0_finalpass_paths,
        flyingthings3d_image1_finalpass_paths,
        flyingthings3d_disparity0_paths,
        flyingthings3d_disparity1_paths)

    print('Processing {} samples in sequence {}'.format(n_sample, ref_dirpath))

    flyingthings3d_test_image0_cleanpass_subset_paths = []
    flyingthings3d_test_image1_cleanpass_subset_paths = []
    flyingthings3d_test_image0_finalpass_subset_paths = []
    flyingthings3d_test_image1_finalpass_subset_paths = []
    flyingthings3d_test_disparity0_subset_paths = []
    flyingthings3d_test_disparity1_subset_paths = []

    for paths in data_paths:

        image0_cleanpass_path, \
            image1_cleanpass_path, \
            image0_finalpass_path, \
            image1_finalpass_path, \
            disparity0_path, \
            disparity1_path = paths

        if not args.paths_only:
            # Store disparity as PNG, set negative and infinity values to zero
            disparity0 = data_utils.read_pfm(disparity0_path)
            disparity1 = data_utils.read_pfm(disparity1_path)

            disparity0[np.isinf(disparity0)] = 0.0
            disparity1[np.isinf(disparity1)] = 0.0

            disparity0[disparity0 < 0] = 0.0
            disparity1[disparity1 < 0] = 0.0

        # Create output data paths
        image0_cleanpass_output_path = image0_cleanpass_path[:-4] + '.png'
        image0_cleanpass_output_path = image0_cleanpass_output_path.replace(
            FLYINGTHINGS3D_ROOT_DIRPATH, FLYINGTHINGS3D_OUTPUT_DIRPATH)

        image1_cleanpass_output_path = image1_cleanpass_path[:-4] + '.png'
        image1_cleanpass_output_path = image1_cleanpass_output_path.replace(
            FLYINGTHINGS3D_ROOT_DIRPATH, FLYINGTHINGS3D_OUTPUT_DIRPATH)

        image0_finalpass_output_path = image0_finalpass_path[:-4] + '.png'
        image0_finalpass_output_path = image0_finalpass_output_path.replace(
            FLYINGTHINGS3D_ROOT_DIRPATH, FLYINGTHINGS3D_OUTPUT_DIRPATH)

        image1_finalpass_output_path = image1_finalpass_path[:-4] + '.png'
        image1_finalpass_output_path = image1_finalpass_output_path.replace(
            FLYINGTHINGS3D_ROOT_DIRPATH, FLYINGTHINGS3D_OUTPUT_DIRPATH)

        disparity0_output_path = disparity0_path[:-4] + '.png'
        disparity0_output_path = disparity0_output_path.replace(
            FLYINGTHINGS3D_ROOT_DIRPATH, FLYINGTHINGS3D_OUTPUT_DIRPATH)

        disparity1_output_path = disparity1_path[:-4] + '.png'
        disparity1_output_path = disparity1_output_path.replace(
            FLYINGTHINGS3D_ROOT_DIRPATH, FLYINGTHINGS3D_OUTPUT_DIRPATH)

        paths = [
            image0_cleanpass_output_path,
            image1_cleanpass_output_path,
            image0_finalpass_output_path,
            image1_finalpass_output_path,
            disparity0_output_path,
            disparity1_output_path
        ]

        for path in paths:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

        # Write to disk
        if not args.paths_only:
            shutil.copyfile(image0_cleanpass_path, image0_cleanpass_output_path)
            shutil.copyfile(image1_cleanpass_path, image1_cleanpass_output_path)

            shutil.copyfile(image0_finalpass_path, image0_finalpass_output_path)
            shutil.copyfile(image1_finalpass_path, image1_finalpass_output_path)

            data_utils.save_disparity(disparity0, disparity0_output_path, multiplier=256.0)
            data_utils.save_disparity(disparity1, disparity1_output_path, multiplier=256.0)

        # Add paths to list
        flyingthings3d_test_image0_cleanpass_subset_paths.append(image0_cleanpass_output_path)
        flyingthings3d_test_image1_cleanpass_subset_paths.append(image1_cleanpass_output_path)

        flyingthings3d_test_image0_finalpass_subset_paths.append(image0_finalpass_output_path)
        flyingthings3d_test_image1_finalpass_subset_paths.append(image1_finalpass_output_path)

        flyingthings3d_test_disparity0_subset_paths.append(disparity0_output_path)
        flyingthings3d_test_disparity1_subset_paths.append(disparity1_output_path)

        flyingthings3d_test_image0_cleanpass_paths.append(image0_cleanpass_output_path)
        flyingthings3d_test_image1_cleanpass_paths.append(image1_cleanpass_output_path)

        flyingthings3d_test_image0_finalpass_paths.append(image0_finalpass_output_path)
        flyingthings3d_test_image1_finalpass_paths.append(image1_finalpass_output_path)

        flyingthings3d_test_disparity0_paths.append(disparity0_output_path)
        flyingthings3d_test_disparity1_paths.append(disparity1_output_path)

    print('Storing %d left FlyingThings3D clean pass subset %s test image file paths into: %s' %
        (len(flyingthings3d_test_image0_cleanpass_subset_paths),
        ref_dirpath,
        FLYINGTHINGS3D_TEST_IMAGE0_CLEANPASS_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath)))
    data_utils.write_paths(
        FLYINGTHINGS3D_TEST_IMAGE0_CLEANPASS_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath),
        flyingthings3d_test_image0_cleanpass_subset_paths)

    print('Storing %d right FlyingThings3D clean pass subset %s test image file paths into: %s' %
        (len(flyingthings3d_test_image1_cleanpass_subset_paths),
        ref_dirpath,
        FLYINGTHINGS3D_TEST_IMAGE1_CLEANPASS_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath)))
    data_utils.write_paths(
        FLYINGTHINGS3D_TEST_IMAGE1_CLEANPASS_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath),
        flyingthings3d_test_image1_cleanpass_subset_paths)

    print('Storing %d left FlyingThings3D final pass subset %s test image file paths into: %s' %
        (len(flyingthings3d_test_image0_finalpass_subset_paths),
        ref_dirpath,
        FLYINGTHINGS3D_TEST_IMAGE0_FINALPASS_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath)))
    data_utils.write_paths(
        FLYINGTHINGS3D_TEST_IMAGE0_FINALPASS_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath),
        flyingthings3d_test_image0_finalpass_subset_paths)

    print('Storing %d right FlyingThings3D final pass subset %s test image file paths into: %s' %
        (len(flyingthings3d_test_image1_finalpass_subset_paths),
        ref_dirpath,
        FLYINGTHINGS3D_TEST_IMAGE1_FINALPASS_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath)))
    data_utils.write_paths(
        FLYINGTHINGS3D_TEST_IMAGE1_FINALPASS_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath),
        flyingthings3d_test_image1_finalpass_subset_paths)

    print('Storing %d left FlyingThings3D subset %s test disparity file paths into: %s' %
        (len(flyingthings3d_test_disparity0_subset_paths),
        ref_dirpath,
        FLYINGTHINGS3D_TEST_DISPARITY0_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath)))
    data_utils.write_paths(
        FLYINGTHINGS3D_TEST_DISPARITY0_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath),
        flyingthings3d_test_disparity0_subset_paths)

    print('Storing %d right FlyingThings3D subset %s test disparity file paths into: %s' %
        (len(flyingthings3d_test_disparity1_subset_paths),
        ref_dirpath,
        FLYINGTHINGS3D_TEST_DISPARITY1_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath)))
    data_utils.write_paths(
        FLYINGTHINGS3D_TEST_DISPARITY1_SUBSET_OUTPUT_FILEPATH.format(ref_dirpath),
        flyingthings3d_test_disparity1_subset_paths)

print('Storing %d left FlyingThings3D clean pass test image file paths into: %s' %
    (len(flyingthings3d_test_image0_cleanpass_paths),
    FLYINGTHINGS3D_TEST_IMAGE0_CLEANPASS_OUTPUT_FILEPATH))
data_utils.write_paths(
    FLYINGTHINGS3D_TEST_IMAGE0_CLEANPASS_OUTPUT_FILEPATH,
    flyingthings3d_test_image0_cleanpass_paths)

print('Storing %d right FlyingThings3D clean pass test image file paths into: %s' %
    (len(flyingthings3d_test_image1_cleanpass_paths),
    FLYINGTHINGS3D_TEST_IMAGE1_CLEANPASS_OUTPUT_FILEPATH))
data_utils.write_paths(
    FLYINGTHINGS3D_TEST_IMAGE1_CLEANPASS_OUTPUT_FILEPATH,
    flyingthings3d_test_image1_cleanpass_paths)

print('Storing %d left FlyingThings3D final pass test image file paths into: %s' %
    (len(flyingthings3d_test_image0_finalpass_paths),
    FLYINGTHINGS3D_TEST_IMAGE0_FINALPASS_OUTPUT_FILEPATH))
data_utils.write_paths(
    FLYINGTHINGS3D_TEST_IMAGE0_FINALPASS_OUTPUT_FILEPATH,
    flyingthings3d_test_image0_finalpass_paths)

print('Storing %d right FlyingThings3D final pass test image file paths into: %s' %
    (len(flyingthings3d_test_image1_finalpass_paths),
    FLYINGTHINGS3D_TEST_IMAGE1_FINALPASS_OUTPUT_FILEPATH))
data_utils.write_paths(
    FLYINGTHINGS3D_TEST_IMAGE1_FINALPASS_OUTPUT_FILEPATH,
    flyingthings3d_test_image1_finalpass_paths)

print('Storing %d left FlyingThings3D test disparity file paths into: %s' %
    (len(flyingthings3d_test_disparity0_paths),
    FLYINGTHINGS3D_TEST_DISPARITY0_OUTPUT_FILEPATH))
data_utils.write_paths(
    FLYINGTHINGS3D_TEST_DISPARITY0_OUTPUT_FILEPATH,
    flyingthings3d_test_disparity0_paths)

print('Storing %d right FlyingThings3D test disparity file paths into: %s' %
    (len(flyingthings3d_test_disparity1_paths),
    FLYINGTHINGS3D_TEST_DISPARITY1_OUTPUT_FILEPATH))
data_utils.write_paths(
    FLYINGTHINGS3D_TEST_DISPARITY1_OUTPUT_FILEPATH,
    flyingthings3d_test_disparity1_paths)
