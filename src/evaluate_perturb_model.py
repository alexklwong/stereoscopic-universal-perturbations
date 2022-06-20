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
import datasets, data_utils, eval_utils
import global_constants as settings
from stereo_model import StereoModel
from perturb_model import PerturbationsModel
from transforms import Transforms
from perturb_main import run
from log_utils import log

torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser()

# Evaluation input filepaths
parser.add_argument('--image0_path',
    type=str, required=True, help='Path to list of left image paths')
parser.add_argument('--image1_path',
    type=str, required=True, help='Path to list of right image paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground truth disparity paths')

# Perturbation model settings
parser.add_argument('--n_image_height',
    type=int, default=settings.N_IMAGE_HEIGHT, help='Height of each sample')
parser.add_argument('--n_image_width',
    type=int, default=settings.N_IMAGE_WIDTH, help='Width of each sample')
parser.add_argument('--output_norm',
    type=float, default=settings.OUTPUT_NORM, help='Output norm of noise')
parser.add_argument('--attack',
    type=str, default=settings.ATTACK, help='Perturbation attack method: [full, tile]')
parser.add_argument('--n_perturbation_height',
    type=int, default=settings.N_PERTURBATION_HEIGHT, help='Height of perturbation')
parser.add_argument('--n_perturbation_width',
    type=int, default=settings.N_PERTURBATION_WIDTH, help='Width of perturbation')
parser.add_argument('--perturb_model_restore_path',
    type=str, default=None, help='Path to restore perturbations checkpoint')
parser.add_argument('--defense_type',
    type=str, default=None, help='Type of defense [jpeg, gaussian, quantization, brightness, contrast, gaussian_noise, shot_noise, impulse_noise, pixelate, defocus_blur, motion_blur]')
parser.add_argument('--ksize',
    type=int, default=settings.GAUSSIAN_KSIZE, help='Kernel size for gaussian filter')
parser.add_argument('--stdev',
    type=int, default=settings.GAUSSIAN_STDEV, help='Standard deviation for gaussian filter')

# Stereo model settings
parser.add_argument('--stereo_method',
    type=str, default=settings.STEREO_METHOD, help='Stereo method available: %s' % settings.STEREO_METHOD_AVAILABLE)
parser.add_argument('--stereo_model_restore_path',
    type=str, default=True, help='Path to restore model checkpoint')
parser.add_argument('--num_deform_layers',
    type=int, default=0, help='Number of deformable convolution layers [0, 6, 25]')

# Output settings
parser.add_argument('--output_dirpath',
    type=str, required=True, help='Directory to store outputs')
parser.add_argument('--save_outputs',
    action='store_true', help='If set, then save outputs to disk')

# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')


args = parser.parse_args()

if __name__ == '__main__':

    args.stereo_method = args.stereo_method.lower()

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

    dataloader = torch.utils.data.DataLoader(
        datasets.StereoDataset(
            image0_paths,
            image1_paths,
            ground_truth_paths,
            shape=(args.n_image_height, args.n_image_width)),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    # Set up data transforms
    transforms = Transforms(normalized_image_range=[0, 1])

    # Build and restore stereo model
    stereo_model = StereoModel(method=args.stereo_method, device=args.device, num_deform_layers=args.num_deform_layers)

    stereo_model.restore_model(args.stereo_model_restore_path)
    stereo_model.eval()

    perturb_model = None
    train_step = 'N/A'

    if args.perturb_model_restore_path is not None:

        # Build and restore perturbations
        perturb_model = PerturbationsModel(
            n_image_height=args.n_image_height,
            n_image_width=args.n_image_width,
            n_image_channel=3,
            output_norm=args.output_norm,
            gradient_scale=None,
            attack=args.attack,
            n_perturbation_height=args.n_perturbation_height,
            n_perturbation_width=args.n_perturbation_width,
            device=args.device)

        train_step = perturb_model.restore_model(args.perturb_model_restore_path)

    # Log the type of defense
    if args.defense_type == 'jpeg':
        log('Apply JPEG Compression', log_path)
    elif args.defense_type == 'gaussian':
        log('Apply Gaussian Filter', log_path)
        log('Kernel Size {}'.format(args.ksize), log_path)
        log('Standard Deviation {}'.format(args.stdev), log_path)
    elif args.defense_type == 'quantization':
        log('Apply quantization', log_path)
    elif args.defense_type == 'brightness':
        log('Apply brightness', log_path)
    elif args.defense_type == 'contrast':
        log('Apply contrast', log_path)
    elif args.defense_type == 'gaussian_noise':
        log('Apply gaussian_noise', log_path)
    elif args.defense_type == 'shot_noise':
        log('Apply shot_noise', log_path)
    elif args.defense_type == 'impulse_noise':
        log('Apply impulse_noise', log_path)
    elif args.defense_type == 'pixelate':
        log('Apply pixelate', log_path)
    elif args.defense_type == 'defocus_blur':
        log('Apply defocus_blur', log_path)
    elif args.defense_type == 'motion_blur':
        log('Apply motion_blur', log_path)

    # Run evaluation and save outputs
    with torch.no_grad():

        disparities_origin, disparities_output, ground_truths = run(
            dataloader=dataloader,
            transforms=transforms,
            stereo_model=stereo_model,
            perturb_model=perturb_model,
            device=args.device,
            output_dirpath=args.output_dirpath if args.save_outputs else None,
            defense_type=args.defense_type,
            ksize=args.ksize,
            stdev=args.stdev,
            verbose=True)

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
