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

import os, glob, argparse
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2


MAX_NORM_PERTURBATION = 2e-2
COLORMAP_DISPARITY = 'plasma'
COLORMAP_ERROR = 'magma'


parser = argparse.ArgumentParser()

parser.add_argument('--model_output_dirpath', type=str, required=True)
parser.add_argument('--colormap_disparity', type=str, default=COLORMAP_DISPARITY)
parser.add_argument('--colormap_error', type=str, default=COLORMAP_ERROR)
parser.add_argument('--max_norm_perturbation', type=float, default=MAX_NORM_PERTURBATION)


args = parser.parse_args()

cmap_disparity = plt.cm.get_cmap(args.colormap_disparity)
cmap_error = plt.cm.get_cmap(args.colormap_error)

# Set up paths to directories
image0_dirpath = os.path.join(args.model_output_dirpath, 'image0')
image1_dirpath = os.path.join(args.model_output_dirpath, 'image1')

image0_output_dirpath = os.path.join(args.model_output_dirpath, 'image0_output')
image1_output_dirpath = os.path.join(args.model_output_dirpath, 'image1_output')

perturb0_dirpath = os.path.join(args.model_output_dirpath, 'perturb0')
perturb1_dirpath = os.path.join(args.model_output_dirpath, 'perturb1')

disparity_origin_dirpath = os.path.join(args.model_output_dirpath, 'disparity_origin')
disparity_output_dirpath = os.path.join(args.model_output_dirpath, 'disparity_output')
ground_truth_dirpath = os.path.join(args.model_output_dirpath, 'ground_truth')

# Fetch file paths from directories
image0_paths = sorted(glob.glob(os.path.join(image0_dirpath, '*.png')))
image1_paths = sorted(glob.glob(os.path.join(image1_dirpath, '*.png')))

image0_output_paths = sorted(glob.glob(os.path.join(image0_output_dirpath, '*.png')))
image1_output_paths = sorted(glob.glob(os.path.join(image1_output_dirpath, '*.png')))

perturb0_paths = sorted(glob.glob(os.path.join(perturb0_dirpath, '*.npy')))
perturb1_paths = sorted(glob.glob(os.path.join(perturb1_dirpath, '*.npy')))

disparity_origin_paths = sorted(glob.glob(os.path.join(disparity_origin_dirpath, '*.npy')))
disparity_output_paths = sorted(glob.glob(os.path.join(disparity_output_dirpath, '*.npy')))

n_sample = len(image0_paths)

# Sanity checks on input directories
assert len(image1_paths) == n_sample
assert len(image0_output_paths) == n_sample
assert len(image1_output_paths) == n_sample
assert len(perturb0_paths) == n_sample
assert len(perturb1_paths) == n_sample
assert len(disparity_origin_paths) == n_sample
assert len(disparity_output_paths) == n_sample

ground_truth_available = os.path.exists(ground_truth_dirpath)

if ground_truth_available:
    ground_truth_paths = sorted(glob.glob(os.path.join(ground_truth_dirpath, '*.npy')))
    assert len(ground_truth_paths) == n_sample

# Create visualization output directories
visualization_dirpath = os.path.join(args.model_output_dirpath, 'visualizations')

perturb0_visualization_dirpath = os.path.join(visualization_dirpath, 'perturb0')
perturb1_visualization_dirpath = os.path.join(visualization_dirpath, 'perturb1')

disparity_origin_visualization_dirpath = os.path.join(visualization_dirpath, 'disparity_origin')
disparity_output_visualization_dirpath = os.path.join(visualization_dirpath, 'disparity_output')
ground_truth_visualization_dirpath = os.path.join(visualization_dirpath, 'ground_truth')
error_visualization_dirpath = os.path.join(visualization_dirpath, 'error')

panel_visualization_dirpath = os.path.join(visualization_dirpath, 'panel')

visualization_output_dirpaths = [
    perturb0_visualization_dirpath,
    perturb1_visualization_dirpath,
    disparity_origin_visualization_dirpath,
    disparity_output_visualization_dirpath,
    ground_truth_visualization_dirpath,
    error_visualization_dirpath,
    panel_visualization_dirpath
]

for dirpath in visualization_output_dirpaths:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Generate visualization
for idx in range(n_sample):

    print('Processing {}/{} samples'.format(idx + 1, n_sample), end='\r')

    _, filename = os.path.split(image0_paths[idx])

    perturb0_output_path = os.path.join(perturb0_visualization_dirpath, filename)
    perturb1_output_path = os.path.join(perturb1_visualization_dirpath, filename)

    disparity_origin_path = os.path.join(disparity_origin_visualization_dirpath, filename)
    disparity_output_path = os.path.join(disparity_output_visualization_dirpath, filename)

    ground_truth_path = os.path.join(ground_truth_visualization_dirpath, filename)
    error_output_path = os.path.join(error_visualization_dirpath, filename)

    panel_output_path = os.path.join(panel_visualization_dirpath, filename)

    # Load images
    image0 = np.asarray(Image.open(image0_paths[idx]).convert('RGB'), np.uint8)
    image1 = np.asarray(Image.open(image1_paths[idx]).convert('RGB'), np.uint8)

    images_panel = np.concatenate([image0, image1], axis=1)

    # Load and visualize perturbations
    perturb0 = np.load(perturb0_paths[idx])

    if perturb0.shape != image0.shape:

        height_perturb, width_perturb, _ = perturb0.shape
        height_image, width_image, _ = image0.shape

        if height_image % height_perturb != 0 or width_image % width_perturb != 0:
            perturb0 = cv2.resize(perturb0, (image0.shape[1], image0.shape[0]))
        else:
            height_repeat = height_image // height_perturb
            width_repeat = width_image // width_perturb
            perturb0 = np.tile(perturb0, (height_repeat, width_repeat, 1))

    perturb0 = 255 * ((perturb0 / (2 * args.max_norm_perturbation)) + 0.50)
    perturb0 = perturb0.astype(np.uint8)

    perturb1 = np.load(perturb1_paths[idx])

    if perturb1.shape != image1.shape:

        height_perturb, width_perturb, _ = perturb1.shape
        height_image, width_image, _ = image1.shape

        if height_image % height_perturb != 0 or width_image % width_perturb != 0:
            perturb1 = cv2.resize(perturb1, (image1.shape[1], image1.shape[0]))
        else:
            height_repeat = height_image // height_perturb
            width_repeat = width_image // width_perturb
            perturb1 = np.tile(perturb1, (height_repeat, width_repeat, 1))

    perturb1 = 255 * ((perturb1 / (2 * args.max_norm_perturbation)) + 0.50)
    perturb1 = perturb1.astype(np.uint8)

    perturbs_panel = np.concatenate([perturb0, perturb1], axis=1)

    # Save perturbations as images
    Image.fromarray(perturb0).save(perturb0_output_path)
    Image.fromarray(perturb1).save(perturb1_output_path)

    # Load and visualize disparities
    disparity_origin = np.load(disparity_origin_paths[idx])

    if disparity_origin.shape != image0.shape:
        disparity_origin = cv2.resize(disparity_origin, (image0.shape[1], image0.shape[0]))

    disparity_origin = 255 * cmap_disparity(disparity_origin / 100.0)[..., 0:3]
    disparity_origin = disparity_origin.astype(np.uint8)

    disparity_output = np.load(disparity_output_paths[idx])

    if disparity_output.shape != image0.shape:
        disparity_output = cv2.resize(disparity_output, (image0.shape[1], image0.shape[0]))

    # If ground truth is available then compute error
    if ground_truth_available:
        ground_truth = np.load(ground_truth_paths[idx])
        error_output = np.abs(disparity_output - ground_truth)

        validity_map = np.where(ground_truth > 0, 1, 0)
        validity_map = np.repeat(np.expand_dims(validity_map, 2), 3, axis=2)

        ground_truth = 255 * cmap_disparity(ground_truth / 100.0)[..., 0:3]
        ground_truth = ground_truth * validity_map
        ground_truth = ground_truth.astype(np.uint8)

        error_output = 255 * cmap_error(error_output / 0.05)[..., 0:3]
        error_output = error_output.astype(np.uint8)

        Image.fromarray(ground_truth).save(ground_truth_path)
        Image.fromarray(error_output).save(error_output_path)

    disparity_output = 255 * cmap_disparity(disparity_output / 100.0)[..., 0:3]
    disparity_output = disparity_output.astype(np.uint8)

    disparities_panel = np.concatenate([disparity_origin, disparity_output], axis=1)

    # Save disparities as images
    Image.fromarray(disparity_origin).save(disparity_origin_path)
    Image.fromarray(disparity_output).save(disparity_output_path)

    # Create visualization
    panel = np.concatenate([images_panel, perturbs_panel, disparities_panel], axis=0)

    Image.fromarray(panel).save(panel_output_path)
