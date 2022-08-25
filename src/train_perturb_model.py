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

import argparse
import global_constants as settings
from perturb_main import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_image0_path',
    type=str, required=True, help='Path to list of left image paths')
parser.add_argument('--train_image1_path',
    type=str, required=True, help='Path to list of right image paths')
parser.add_argument('--train_pseudo_ground_truth_path',
    type=str, default=None, help='Path to list of ground truth disparity paths')
parser.add_argument('--val_image0_path',
    type=str, default=None, help='Path to list of left image paths')
parser.add_argument('--val_image1_path',
    type=str, default=None, help='Path to list of right image paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default=None, help='Path to list of ground truth disparity paths')

# Perturbation model settings
parser.add_argument('--n_image_height',
    type=int, default=settings.N_IMAGE_HEIGHT, help='Height of each sample')
parser.add_argument('--n_image_width',
    type=int, default=settings.N_IMAGE_WIDTH, help='Width of each sample')
parser.add_argument('--output_norm',
    type=float, default=settings.OUTPUT_NORM, help='Output norm of noise')
parser.add_argument('--gradient_scale',
    type=float, default=settings.GRADIENT_SCALE, help='Value to scale gradients by')
parser.add_argument('--attack',
    type=str, default=settings.ATTACK, help='Perturbation attack method: [full, tile]')
parser.add_argument('--n_perturbation_height',
    type=int, default=settings.N_PERTURBATION_HEIGHT, help='Height of perturbation')
parser.add_argument('--n_perturbation_width',
    type=int, default=settings.N_PERTURBATION_WIDTH, help='Width of perturbation')

# Optimization settings
parser.add_argument('--n_batch',
    type=int, default=settings.N_BATCH, help='Number of samples per batch')
parser.add_argument('--n_epoch',
    type=int, default=settings.N_EPOCH, help='Number of samples per batch')

# Stereo model settings
parser.add_argument('--stereo_method',
    type=str, default=settings.STEREO_METHOD, help='Stereo method available: %s' % settings.STEREO_METHOD_AVAILABLE)
parser.add_argument('--stereo_model_restore_path',
    type=str, default='', help='Path to restore model checkpoint')
parser.add_argument('--num_deform_layers',
    type=int, default=0, help='Number of deformable convolution layers [0, 6, 25]')

# Checkpoint settings
parser.add_argument('--n_checkpoint',
    type=int, default=settings.N_CHECKPOINT, help='Number of steps before saving a checkpoint')
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoints')

# Hardware settings
parser.add_argument('--n_worker',
    type=int, default=settings.N_WORKER, help='Number of workers/threads to use')
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')


args = parser.parse_args()

if __name__ == '__main__':

    args.stereo_method = args.stereo_method.lower()

    args.device = args.device.lower()

    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    train(train_image0_path=args.train_image0_path,
          train_image1_path=args.train_image1_path,
          train_pseudo_ground_truth_path=args.train_pseudo_ground_truth_path,
          val_image0_path=args.val_image0_path,
          val_image1_path=args.val_image1_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Perturbation model settings
          n_image_height=args.n_image_height,
          n_image_width=args.n_image_width,
          output_norm=args.output_norm,
          gradient_scale=args.gradient_scale,
          attack=args.attack,
          n_perturbation_height=args.n_perturbation_height,
          n_perturbation_width=args.n_perturbation_width,
          # Optimization settings
          n_batch=args.n_batch,
          n_epoch=args.n_epoch,
          # Stereo model settings
          stereo_method=args.stereo_method,
          stereo_model_restore_path=args.stereo_model_restore_path,
          num_deform_layers=args.num_deform_layers,
          # Checkpoint settings
          n_checkpoint=args.n_checkpoint,
          checkpoint_path=args.checkpoint_path,
          # Hardware settings
          n_worker=args.n_worker,
          device=args.device)
