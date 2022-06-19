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

import argparse, os, warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import datasets, data_utils, eval_utils
import global_constants as settings
from stereo_model import StereoModel
from perturb_model import PerturbationsModel
from transforms import Transforms
from log_utils import log
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "external_src")

torch.multiprocessing.set_sharing_strategy('file_system')

MIN_CLASS_PIXELS_IN_IMAGE = 25

parser = argparse.ArgumentParser()

# Evaluation input filepaths
parser.add_argument('--image0_path',
    type=str, required=True, help='Path to list of left image paths')
parser.add_argument('--image1_path',
    type=str, required=True, help='Path to list of right image paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground truth disparity paths')
parser.add_argument('--seg0_path',
    type=str, default=None, help='Path to list of segmentation left image paths')

# Perturbation model settings
parser.add_argument('--stereo_model_perturb_trained_on',
    type=str, default=None, help='Stereo model perturbation was trained on')
parser.add_argument('--n_image_height',
    type=int, default=settings.N_IMAGE_HEIGHT, help='Height of each sample')
parser.add_argument('--n_image_width',
    type=int, default=settings.N_IMAGE_WIDTH, help='Width of each sample')
parser.add_argument('--attack',
    type=str, default=settings.ATTACK, help='Perturbation attack method: [full, tile, patch]')
parser.add_argument('--n_perturbation_height',
    type=int, default=settings.N_PERTURBATION_HEIGHT, help='Height of perturbation')
parser.add_argument('--n_perturbation_width',
    type=int, default=settings.N_PERTURBATION_WIDTH, help='Width of perturbation')
parser.add_argument('--output_norm',
    type=float, default=settings.OUTPUT_NORM, help='Output norm of noise')
parser.add_argument('--gradient_scale',
    type=float, default=settings.GRADIENT_SCALE, help='Value to scale gradients by')
parser.add_argument('--perturb_model_restore_path',
    type=str, default=None, help='Path to restore perturbations checkpoint')

# Stereo model settings
parser.add_argument('--stereo_method',
    type=str, default=settings.STEREO_METHOD, help='Stereo method available: %s' % settings.STEREO_METHOD_AVAILABLE)
parser.add_argument('--stereo_model_restore_path',
    type=str, default=True, help='Path to restore model checkpoint')

# Output settings
parser.add_argument('--output_dirpath',
    type=str, required=True, help='Directory to store outputs')

# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')


def run(dataloader,
        stereo_model,
        transforms,
        perturb_model=None,
        device=torch.device('cuda'),
        verbose=False,
        n_seg_classes=0):
    '''
    Runs semantic segmentation on images and map class to corresponding disparities

    Arg(s):
        dataloader : torch.utils.data.DataLoader
            DataLoader instance for StereoSegmentationDataset
        stereo_model : StereoModel
            StereoModel instance
        transforms : Transforms
            Transforms instance
        peturb_model : PerturbationsModel
            PerturbationsModel instance
        device : torch.device
            cpu or cuda device to run on
        verbose : bool
            if set, then print progress
        n_seg_classes : int
            number of semantic segmentation classes
    Returns:
        list[numpy[float32]] : list of N x 1 x H x W disparities of original images
        list[numpy[float32]] : list of N x 1 x H x W disparities of perturbed images
        list[numpy[float32]] : list of N x 1 x H x W ground truth
        dict[int, list[float32]] : D1 error of perturbed images w.r.t. original images
        dict[int, list[float32]] : end point error of perturbed images w.r.t. original images
    '''

    d1_clean_images_seg = {i: [] for i in range(n_seg_classes)}
    epe_clean_images_seg = {i: [] for i in range(n_seg_classes)}

    disparities_origin = []
    disparities_output = []
    ground_truths = []

    for idx, data in enumerate(dataloader):

        image0, image1, seg0, ground_truth, _ = data
        seg0 = seg0.cpu().numpy()

        image0 = image0.to(device)
        image1 = image1.to(device)

        [image0, image1] = transforms.transform(images_arr=[image0, image1])

        if verbose:
            print('Processed {}/{} samples'.format(idx, len(dataloader)), end='\r')

        # Get original disparity without any perturbations
        disparity_origin = stereo_model.forward(image0, image1)

        # Perturb the images
        if perturb_model is not None:
            image0_output, image1_output = perturb_model.forward(image0, image1)
        else:
            image0_output = image0
            image1_output = image1

        # Get output disparity after perturbations
        disparity_output = stereo_model.forward(image0_output, image1_output)

        # Move to numpy
        disparity_origin = np.squeeze(disparity_origin.cpu().numpy())
        disparity_output = np.squeeze(disparity_output.cpu().numpy())
        ground_truth = np.squeeze(ground_truth.cpu().numpy())

        disparities_origin.append(disparity_origin)
        disparities_output.append(disparity_output)
        ground_truths.append(ground_truth)

        seg0 = np.squeeze(seg0)
        for c in range(n_seg_classes):
            seg_mask = (seg0 == c)

            if np.sum(seg_mask) == 0:
                continue
            elif np.count_nonzero(seg_mask) < MIN_CLASS_PIXELS_IN_IMAGE:
                continue

            output_c = disparity_output[seg_mask]
            origin_c = disparity_origin[seg_mask]

            d1_clean_images_seg[c].append(
                eval_utils.d1_error(output_c, origin_c))

            epe_clean_images_seg[c].append(
                eval_utils.end_point_error(output_c, origin_c))

    return disparities_origin, disparities_output, ground_truths, d1_clean_images_seg, epe_clean_images_seg


if __name__ == '__main__':

    args = parser.parse_args()

    # Set device
    args.device = args.device.lower()
    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA

    args.device = settings.CUDA if args.device == settings.GPU else args.device
    if args.device == settings.CUDA or args.device == settings.GPU:
        args.device = torch.device(settings.CUDA)
    else:
        args.device = torch.device(settings.CPU)

    # Set up log path
    log_path = os.path.join(args.output_dirpath, 'results.txt')

    # Read input paths
    image0_paths = data_utils.read_paths(args.image0_path)
    image1_paths = data_utils.read_paths(args.image1_path)
    ground_truth_paths = data_utils.read_paths(args.ground_truth_path)

    assert len(image0_paths) == len(image1_paths)
    assert len(image0_paths) == len(ground_truth_paths)

    # Read segmentation files
    seg0_paths = data_utils.read_paths(args.seg0_path)
    n_seg_classes = len(settings.SEG_LABELS)

    dataloader = torch.utils.data.DataLoader(
        datasets.StereoSegmentationDataset(
            image0_paths,
            image1_paths,
            ground_truth_paths,
            seg0_paths=seg0_paths,
            shape=(args.n_image_height, args.n_image_width)),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    # Set up data transforms
    transforms = Transforms(normalized_image_range=[0, 1])

    # Build and restore stereo model
    args.stereo_method = args.stereo_method.lower()
    stereo_model = StereoModel(method=args.stereo_method, device=args.device)

    stereo_model.restore_model(args.stereo_model_restore_path)
    stereo_model.eval()

    # Build and restore perturbation
    args.stereo_model_perturb_trained_on = args.stereo_model_perturb_trained_on.lower()
    perturb_model = None

    if args.perturb_model_restore_path is not None:
        perturb_model = PerturbationsModel(
            n_image_height=args.n_image_height,
            n_image_width=args.n_image_width,
            n_image_channel=3,
            output_norm=args.output_norm,
            gradient_scale=args.gradient_scale,
            attack=args.attack,
            n_perturbation_height=args.n_perturbation_height,
            n_perturbation_width=args.n_perturbation_width,
            device=args.device)
        train_step = perturb_model.restore_model(args.perturb_model_restore_path)

    log('Finding class-wise error via segmentation', log_path)
    log('Perturb model trained on {}'.format(args.stereo_model_perturb_trained_on), log_path)
    log('Evaluating on {}'.format(args.stereo_method), log_path)
    log('Output dirpath: {}'.format(args.output_dirpath), log_path)

    # Run evaluation and save outputs
    with torch.no_grad():

        disparities_origin, disparities_output, ground_truths, d1_clean_images_seg, epe_clean_images_seg = run(
            dataloader=dataloader,
            transforms=transforms,
            stereo_model=stereo_model,
            perturb_model=perturb_model,
            device=args.device,
            verbose=True,
            n_seg_classes=n_seg_classes)

        log('Validation results @ step={}:'.format(train_step), log_path)
        log('Error w.r.t. clean images', log_path)
        eval_utils.evaluate(
            disparities=disparities_output,
            ground_truths=disparities_origin,
            step=train_step,
            log_path=log_path)

        log('Error w.r.t. ground truth', log_path)
        results = eval_utils.evaluate(
            disparities=disparities_output,
            ground_truths=ground_truths,
            step=train_step,
            log_path=log_path)

        log('Segmentation breakdown (Clean):', log_path)
        log('{:<10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
            'Class', 'N_Images', 'D1-Error ', '+/-', 'EPE', '+/-'), log_path)

        for c in range(n_seg_classes):
            log('{:<10}  {:>10}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}'.format(
                c,
                len(d1_clean_images_seg[c]),
                np.mean(np.array(d1_clean_images_seg[c]) * 100.0),
                np.std(np.array(d1_clean_images_seg[c]) * 100.0),
                np.mean(np.array(epe_clean_images_seg[c])),
                np.std(np.array(epe_clean_images_seg[c]))),
                log_path)

        model_mean = np.zeros(n_seg_classes)
        model_std = np.zeros(n_seg_classes)

        for c in range(n_seg_classes):
            if len(d1_clean_images_seg[c]) < 5:
                model_mean[c] = 0
                model_std[c] = 0
            else:
                model_mean[c] = np.mean(np.array(d1_clean_images_seg[c]) * 100.0)
                model_std[c] = np.std(np.array(d1_clean_images_seg[c]) * 100.0)

        if args.stereo_model_perturb_trained_on == 'psmnet':
            stereo_model_perturb_trained_on_name = 'PSMNet'
        elif args.stereo_model_perturb_trained_on == 'deeppruner':
            stereo_model_perturb_trained_on_name = 'DeepPruner'
        elif args.stereo_model_perturb_trained_on == 'aanet':
            stereo_model_perturb_trained_on_name = 'AANet'
        else:
            raise NotImplementedError(
                'Stereo model perturb trained on not supported: {}'.format(args.stereo_model_perturb_trained_on))

        if args.stereo_method == 'psmnet':
            color = 'blue'
            stereo_method_name = 'PSMNet'
        elif args.stereo_method == 'deeppruner':
            color = 'green'
            stereo_method_name = 'DeepPruner'
        elif args.stereo_method == 'aanet':
            color = 'crimson'
            stereo_method_name = 'AANet'
        else:
            raise NotImplementedError('Stereo method not supported: {}'.format(args.stereo_method))

        # Plot the error of each segmentation class
        plt.title(
            'KITTI 2015 Class Error: {} -> {}'.format(stereo_model_perturb_trained_on_name, stereo_method_name),
            fontsize=20)
        ind_sort = np.argsort(model_mean)[::-1]

        # Remove the classes 15, 16, 17 because there are only 1 or 2 instances of them
        ind_sort = ind_sort[:-5]

        plt.bar([settings.SEG_LABELS[x] for x in ind_sort],
                model_mean[ind_sort],
                color=color,
                yerr=model_std[ind_sort],
                error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2))
        plt.ylim(0)
        plt.xticks(rotation=90, fontsize=14)
        plt.xlabel('Class', fontsize=16)
        plt.ylabel('D1-Error', fontsize=16)

        if not os.path.exists(args.output_dirpath):
            os.makedirs(args.output_dirpath)

        IMAGE_PATH = os.path.join(
            args.output_dirpath,
            '{}_to_{}_segmentation_plot.png'.format(args.stereo_model_perturb_trained_on, args.stereo_method))

        plt.savefig(IMAGE_PATH, bbox_inches='tight')
