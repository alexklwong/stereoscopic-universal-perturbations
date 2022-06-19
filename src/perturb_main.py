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

import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils, eval_utils
from log_utils import log
from transforms import Transforms
from stereo_model import StereoModel
from perturb_model import PerturbationsModel

import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'external_src')


def train(train_image0_path,
          train_image1_path,
          train_ground_truth_path,
          train_pseudo_ground_truth_path,
          val_image0_path,
          val_image1_path,
          val_ground_truth_path,
          # Dataloader settings
          n_batch,
          n_image_height,
          n_image_width,
          # Perturbation model settings
          n_epoch,
          output_norm,
          gradient_scale,
          attack,
          n_perturbation_height,
          n_perturbation_width,
          # Stereo model settings
          stereo_method,
          stereo_model_restore_path,
          num_deform_layers,
          # Checkpoint settings
          checkpoint_path,
          n_checkpoint,
          # Hardware settings
          n_worker,
          device):

    # Set device: cpu or cuda
    device = torch.device(device)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and logging paths
    log_path = os.path.join(checkpoint_path, 'results.txt')
    perturb_model_checkpoint_path = os.path.join(checkpoint_path, 'perturb_model-{}.pth')

    # Set up summary output path
    event_path = os.path.join(checkpoint_path, 'events')
    train_summary_writer = SummaryWriter(event_path + '-train')

    # To keep track of step containing best perturbations
    best_results = {
        'step'                   : -1,
        'd1_ground_truth_mean'   : -1,
        'd1_ground_truth_std'    : -1,
        'epe_ground_truth_mean'  : -1,
        'epe_ground_truth_std'   : -1,
    }

    '''
    Load training and validation data paths
    '''
    # Read training input paths
    train_image0_paths = data_utils.read_paths(train_image0_path)
    train_image1_paths = data_utils.read_paths(train_image1_path)

    n_train_sample = len(train_image0_paths)

    n_train_step = \
        n_epoch * np.ceil(n_train_sample / n_batch).astype(np.int32)

    assert n_train_sample == len(train_image1_paths)

    train_ground_truth_available = train_ground_truth_path is not None
    train_pseudo_ground_truth_available = train_pseudo_ground_truth_path is not None

    if train_ground_truth_available:
        train_ground_truth_paths = data_utils.read_paths(train_ground_truth_path)

        assert n_train_sample == len(train_ground_truth_paths)
    else:
        train_ground_truth_paths = [None] * n_train_sample

    if train_pseudo_ground_truth_available:
        train_pseudo_ground_truth_paths = data_utils.read_paths(train_pseudo_ground_truth_path)

        assert n_train_sample == len(train_pseudo_ground_truth_paths)
    else:
        train_pseudo_ground_truth_paths = [None] * n_train_sample

    train_dataloader = torch.utils.data.DataLoader(
        datasets.StereoDataset(
            image0_paths=train_image0_paths,
            image1_paths=train_image1_paths,
            ground_truth_paths=train_ground_truth_paths,
            pseudo_ground_truth_paths=train_pseudo_ground_truth_paths,
            shape=(n_image_height, n_image_width)),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_worker,
        drop_last=False)

    # Read validation input paths
    validation_available = \
        val_image0_path is not None and \
        val_image1_path is not None and \
        val_ground_truth_path is not None and \
        val_image0_path != '' and \
        val_image1_path != '' and \
        val_ground_truth_path != ''

    if validation_available:
        val_image0_paths = data_utils.read_paths(val_image0_path)
        val_image1_paths = data_utils.read_paths(val_image1_path)
        val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

        assert len(val_image0_paths) == len(val_image1_paths)
        assert len(val_image0_paths) == len(val_ground_truth_paths)

        val_dataloader = torch.utils.data.DataLoader(
            datasets.StereoDataset(
                image0_paths=val_image0_paths,
                image1_paths=val_image1_paths,
                ground_truth_paths=val_ground_truth_paths,
                pseudo_ground_truth_paths=None,
                shape=(n_image_height, n_image_width)),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

    # Set up data transforms
    transforms = Transforms(normalized_image_range=[0, 1])

    '''
    Set up stereo and perturbations models
    '''
    # Build and restore stereo model
    stereo_model = StereoModel(
        method=stereo_method,
        num_deform_layers=num_deform_layers,
        device=device)

    stereo_model.restore_model(stereo_model_restore_path)
    stereo_model.eval()

    # Initialize perturbations
    perturb_model = PerturbationsModel(
        n_image_height=n_image_height,
        n_image_width=n_image_width,
        n_image_channel=3,
        output_norm=output_norm,
        gradient_scale=gradient_scale,
        attack=attack,
        n_perturbation_height=n_perturbation_height,
        n_perturbation_width=n_perturbation_width,
        device=device)

    '''
    Log settings and train
    '''
    # Log arguments
    log('Training input paths:', log_path)
    train_input_paths = [
        train_image0_path,
        train_image1_path,
        train_ground_truth_path,
        train_pseudo_ground_truth_path
    ]

    for path in train_input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_image0_path,
        val_image1_path,
        val_ground_truth_path
    ]

    for path in val_input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log('Perturbation model settings:', log_path)
    log('n_image_height=%d  n_image_width=%d' %
        (n_image_height, n_image_width),
        log_path)
    log('output_norm=%.3f  gradient_scale=%.1e' %
        (output_norm, gradient_scale),
        log_path)
    log('attack=%s' %
        (attack),
        log_path)

    if attack == 'tile':
        log('n_perturbation_height=%d  n_perturbation_width=%d' %
            (n_perturbation_height, n_perturbation_width),
            log_path)

    log('Optimization settings:', log_path)
    log('n_train_sample=%d  n_batch=%d  n_epoch=%d' %
        (n_train_sample, n_batch, n_epoch), log_path)

    log('Stereo model settings:', log_path)
    log('stereo_method=%s' %
        (stereo_method),
        log_path)
    log('stereo_model_restore_path=%s' %
        (stereo_model_restore_path),
        log_path)

    if num_deform_layers > 0:
        log('num_deform_layers=%d' %
            (num_deform_layers),
            log_path)

    log('Checkpoint settings:', log_path)
    log('checkpoint_path=%s' %
        (checkpoint_path),
        log_path)

    log('Training...', log_path)

    train_step = 0
    time_start = time.time()

    for epoch in range(1, n_epoch + 1):

        for image0, image1, ground_truth, pseudo_ground_truth in train_dataloader:

            train_step = train_step + 1

            # Move data to device
            if train_ground_truth_available:
                ground_truth = ground_truth.to(device)
            else:
                ground_truth = None

            if train_pseudo_ground_truth_available:
                pseudo_ground_truth = pseudo_ground_truth.to(device)
            else:
                pseudo_ground_truth = None

            image0 = image0.to(device)
            image1 = image1.to(device)

            # Normalize images
            [image0, image1] = transforms.transform(images_arr=[image0, image1])

            # Compute and aggregate perturbations
            loss = perturb_model.optimize_perturbations(
                stereo_model=stereo_model,
                image0=image0,
                image1=image1,
                ground_truth=ground_truth,
                pseudo_ground_truth=pseudo_ground_truth)

            # Log results
            if train_step and (train_step % n_checkpoint) == 0:

                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                with torch.no_grad():

                    # Forward through through stereo network
                    disparity_origin = stereo_model.forward(image0, image1)

                    # Apply perturbations to the images
                    image0_output, image1_output = perturb_model.forward(image0, image1)

                    # Forward through network again
                    disparity_output = stereo_model.forward(image0_output, image1_output)

                    # Log summary to tensorboard
                    perturb_model.log_summary(
                        image0=image0,
                        image1=image1,
                        disparity_origin=disparity_origin,
                        disparity_output=disparity_output,
                        step=train_step,
                        summary_writer=train_summary_writer,
                        tag='train')

                if validation_available:
                    # Switch to validation mode
                    perturb_model.eval()

                    with torch.no_grad():

                        val_disparities_origin, val_disparities_output, val_ground_truths = run(
                            dataloader=val_dataloader,
                            transforms=transforms,
                            stereo_model=stereo_model,
                            perturb_model=perturb_model,
                            device=device)

                        log('Validation results @ step={}:'.format(train_step), log_path)
                        log('Error w.r.t. clean images', log_path)
                        eval_utils.evaluate(
                            disparities=val_disparities_output,
                            ground_truths=val_disparities_origin,
                            step=train_step,
                            log_path=log_path)

                        log('Error w.r.t. ground truth', log_path)
                        results = eval_utils.evaluate(
                            disparities=val_disparities_output,
                            ground_truths=val_ground_truths,
                            step=train_step,
                            log_path=log_path)

                        best_results = compare_results(
                            results=results,
                            best_results=best_results,
                            log_path=log_path)

                    # Switch back to training
                    perturb_model.train()

                # Save perturbations to checkpoint
                perturb_model.save_model(
                    checkpoint_path=perturb_model_checkpoint_path.format(train_step),
                    step=train_step)

    # Perform logging and validation for last step
    log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
        train_step, n_train_step, loss.item(), time_elapse, time_remain),
        log_path)

    if validation_available:
        # Switch to validation mode
        perturb_model.eval()

        with torch.no_grad():

            val_disparities_origin, val_disparities_output, val_ground_truths = run(
                dataloader=val_dataloader,
                transforms=transforms,
                stereo_model=stereo_model,
                perturb_model=perturb_model,
                device=device)

            log('Validation results @ step={}:'.format(train_step), log_path)
            log('Error w.r.t. clean images', log_path)
            eval_utils.evaluate(
                disparities=val_disparities_output,
                ground_truths=val_disparities_origin,
                step=train_step,
                log_path=log_path)

            log('Error w.r.t. ground truth', log_path)
            results = eval_utils.evaluate(
                disparities=val_disparities_output,
                ground_truths=val_ground_truths,
                step=train_step,
                log_path=log_path)

            best_results = compare_results(
                results=results,
                best_results=best_results,
                log_path=log_path)

    # Save perturbations to checkpoint
    perturb_model.save_model(
        checkpoint_path=perturb_model_checkpoint_path.format(train_step),
        step=train_step)

def run(dataloader,
        stereo_model,
        transforms,
        perturb_model=None,
        device=torch.device('cuda'),
        output_dirpath=None,
        defense_type=None,
        ksize=None,
        stdev=None,
        verbose=False):
    '''
    Runs inputs through a stereo network

    Arg(s):
        dataloader : torch.utils.data.DataLoader
            Loads left, right images and ground truth
        stereo_model : StereoModel
            StereoModel class instance
        transforms : Transforms
            Transforms class instance
        perturb_model : PerturbationsModel
            PerturbationsModel class instance
        device : torch.device
            device to run on
        output_dirpath : str
            if not None, then store inputs and outputs to directory
        defense_type : str
            Type of defense [jpeg, gaussian, quantization, brightness, contrast]
        ksize : int
            Kernel size for gaussian filter
        stdev : int
            Standard deviation for gaussian filter
        verbose : bool
            Verbose output
    Returns:
        list[numpy] : list of H x W disparity maps before perturbing the input
        list[numpy] : list of H x W disparity maps after perturbing the input
        list[numpy] : list of H x W ground truth disparity maps
    '''

    # If we plan to store outputs
    if output_dirpath is not None:
        # Define output paths
        image0_dirpath = os.path.join(output_dirpath, 'image0')
        image1_dirpath = os.path.join(output_dirpath, 'image1')

        image0_output_dirpath = os.path.join(output_dirpath, 'image0_output')
        image1_output_dirpath = os.path.join(output_dirpath, 'image1_output')

        perturb0_dirpath = os.path.join(output_dirpath, 'perturb0')
        perturb1_dirpath = os.path.join(output_dirpath, 'perturb1')

        disparity_origin_dirpath = os.path.join(output_dirpath, 'disparity_origin')
        disparity_output_dirpath = os.path.join(output_dirpath, 'disparity_output')
        ground_truth_dirpath = os.path.join(output_dirpath, 'ground_truth')

        # Create output directories
        output_dirpaths = [
            image0_dirpath,
            image1_dirpath,
            image0_output_dirpath,
            image1_output_dirpath,
            perturb0_dirpath,
            perturb1_dirpath,
            disparity_origin_dirpath,
            disparity_output_dirpath,
            ground_truth_dirpath
        ]

        for dirpath in output_dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

        # Convert perturbations to numpy
        if perturb_model is not None:
            perturb0, perturb1 = perturb_model.numpy()
            perturb0 = np.transpose(np.squeeze(perturb0), (1, 2, 0))
            perturb1 = np.transpose(np.squeeze(perturb1), (1, 2, 0))

    disparities_origin = []
    disparities_output = []
    ground_truths = []

    for idx, data in enumerate(dataloader):

        image0, image1, ground_truth, _ = data

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

            perturb0 = np.transpose(
                np.squeeze(np.zeros_like(image0_output.cpu().numpy())),
                (1, 2, 0))
            perturb1 = np.transpose(
                np.squeeze(np.zeros_like(image0_output.cpu().numpy())),
                (1, 2, 0))

        if defense_type is not None:
            image0_output = eval_utils.defense(image0_output, defense_type, device, ksize, stdev)
            image1_output = eval_utils.defense(image1_output, defense_type, device, ksize, stdev)

        # Get output disparity after perturbations
        disparity_output = stereo_model.forward(image0_output, image1_output)

        # Move to numpy
        disparity_origin = np.squeeze(disparity_origin.cpu().numpy())
        disparity_output = np.squeeze(disparity_output.cpu().numpy())
        ground_truth = np.squeeze(ground_truth.cpu().numpy())

        disparities_origin.append(disparity_origin)
        disparities_output.append(disparity_output)
        ground_truths.append(ground_truth)

        # Convert images and disparity maps to numpy and store
        if output_dirpath is not None:
            image0 = \
                np.transpose(np.squeeze(image0.detach().cpu().numpy()), (1, 2, 0))
            image1 = \
                np.transpose(np.squeeze(image1.detach().cpu().numpy()), (1, 2, 0))

            image0_output = \
                np.transpose(np.squeeze(image0_output.detach().cpu().numpy()), (1, 2, 0))
            image1_output = \
                np.transpose(np.squeeze(image1_output.detach().cpu().numpy()), (1, 2, 0))

            image_filename = '{:05d}.png'.format(idx)
            numpy_filename = '{:05d}.npy'.format(idx)

            # Save images to disk.
            image0_path = os.path.join(image0_dirpath, image_filename)
            image1_path = os.path.join(image1_dirpath, image_filename)

            Image.fromarray(np.uint8(image0 * 255.0)).save(image0_path)
            Image.fromarray(np.uint8(image1 * 255.0)).save(image1_path)

            image0_output_path = os.path.join(image0_output_dirpath, image_filename)
            image1_output_path = os.path.join(image1_output_dirpath, image_filename)

            Image.fromarray(np.uint8(image0_output * 255.0)).save(image0_output_path)
            Image.fromarray(np.uint8(image1_output * 255.0)).save(image1_output_path)

            # Save perturbations to disk
            perturb0_path = os.path.join(perturb0_dirpath, numpy_filename)
            perturb1_path = os.path.join(perturb1_dirpath, numpy_filename)

            np.save(perturb0_path, perturb0)
            np.save(perturb1_path, perturb1)

            # Save disparity maps to disk
            disparity_origin_path = os.path.join(disparity_origin_dirpath, numpy_filename)
            disparity_output_path = os.path.join(disparity_output_dirpath, numpy_filename)
            ground_truth_path = os.path.join(ground_truth_dirpath, numpy_filename)

            np.save(disparity_origin_path, disparity_origin)
            np.save(disparity_output_path, disparity_output)
            np.save(ground_truth_path, ground_truth)

    return disparities_origin, disparities_output, ground_truths

def compare_results(results,
                    best_results,
                    log_path=None,
                    finetune=False):

    step = results['step']
    d1_ground_truth_mean = results['d1_ground_truth_mean']
    d1_ground_truth_std = results['d1_ground_truth_std']
    epe_ground_truth_mean = results['epe_ground_truth_mean']
    epe_ground_truth_std = results['epe_ground_truth_std']

    if finetune is True:
        if d1_ground_truth_mean < best_results['d1_ground_truth_mean']:
            best_results['step'] = step
            best_results['d1_ground_truth_mean'] = d1_ground_truth_mean
            best_results['d1_ground_truth_std'] = d1_ground_truth_std
            best_results['epe_ground_truth_mean'] = epe_ground_truth_mean
            best_results['epe_ground_truth_std'] = epe_ground_truth_std
    else:
        if d1_ground_truth_mean > best_results['d1_ground_truth_mean']:
            best_results['step'] = step
            best_results['d1_ground_truth_mean'] = d1_ground_truth_mean
            best_results['d1_ground_truth_std'] = d1_ground_truth_std
            best_results['epe_ground_truth_mean'] = epe_ground_truth_mean
            best_results['epe_ground_truth_std'] = epe_ground_truth_std

    log('Best results @ step={}:'.format(best_results['step']), log_path)
    log('{:<10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
        '', 'D1-Error ', '+/-', 'EPE', '+/-'),
        log_path)
    log('{:<10}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}'.format(
        '',
        best_results['d1_ground_truth_mean'],
        best_results['d1_ground_truth_std'],
        best_results['epe_ground_truth_mean'],
        best_results['epe_ground_truth_std']),
        log_path)

    return best_results
