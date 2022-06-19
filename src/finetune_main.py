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

import os, time, random, warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import data_utils
import datasets
import eval_utils
from transforms import Transforms
from stereo_model import StereoModel
from log_utils import log
from perturb_model import PerturbationsModel
from perturb_main import run, compare_results


def train(train_image0_path,
          train_image1_path,
          train_ground_truth_path,
          train_pseudo_ground_truth_path,
          val_image0_path,
          val_image1_path,
          val_ground_truth_path,
          # Stereo model settings
          stereo_method,
          stereo_model_restore_path,
          num_deform_layers,
          # Dataloader settings
          n_batch,
          n_image_height,
          n_image_width,
          # Perturbation model settings
          attack,
          output_norms,
          gradient_scales,
          n_perturbation_height,
          n_perturbation_width,
          perturb_paths,
          p_threshold,
          # Learning rates settings
          learning_rates,
          learning_schedule,
          # Checkpoint settings
          n_checkpoint,
          checkpoint_path,
          # Hardware settings
          n_worker,
          device):

    # Set device: cpu or cuda
    device = torch.device(device)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and logging paths
    log_path = os.path.join(checkpoint_path, 'results.txt')
    stereo_model_checkpoint_path = os.path.join(checkpoint_path, 'stereo_model-{}.pth')

    # To keep track of step containing best perturbations
    best_results = {
        'step'                   : -1,
        'd1_ground_truth_mean'   : np.inf,
        'd1_ground_truth_std'    : np.inf,
        'epe_ground_truth_mean'  : np.inf,
        'epe_ground_truth_std'   : np.inf
    }

    '''
    Load training and validation data paths
    '''
    # Read train input paths
    train_image0_paths = data_utils.read_paths(train_image0_path)
    train_image1_paths = data_utils.read_paths(train_image1_path)

    n_train_sample = len(train_image0_paths)

    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)

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

    # Set the shape for resizing or random cropping
    if stereo_method == 'aanet':
        shape = (n_image_height, n_image_width)
    elif stereo_method == 'deeppruner' or stereo_method == 'psmnet':
        shape = (375, 1242)
    else:
        raise ValueError('Unsupported stereo method: {}'.format(stereo_method))

    # Dataloader for training data
    train_dataloader = torch.utils.data.DataLoader(
        datasets.StereoDataset(
            image0_paths=train_image0_paths,
            image1_paths=train_image1_paths,
            ground_truth_paths=train_ground_truth_paths,
            pseudo_ground_truth_paths=train_pseudo_ground_truth_paths,
            shape=shape),
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
        val_ground_truth_path != '' \

    if validation_available:
        val_image0_paths = data_utils.read_paths(val_image0_path)
        val_image1_paths = data_utils.read_paths(val_image1_path)
        val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

        assert len(val_image0_paths) == len(val_image1_paths)
        assert len(val_image0_paths) == len(val_ground_truth_paths)

        # Dataloader for validation data
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

    # Data augmentation for each method
    if stereo_method == 'aanet':
        normalized_image_range = [0, 1]
        random_flip_type = ['vertical']
        random_transform_probability = 1
        random_resize_and_pad = [-1, -1]
        random_crop = [-1, -1]
    elif stereo_method == 'deeppruner' or stereo_method == 'psmnet':
        normalized_image_range = [0, 1]
        random_flip_type = ['none']
        random_transform_probability = 0.5
        random_resize_and_pad = [0.9, 0.9]
        random_crop = [n_image_height, n_image_width]
    else:
        raise ValueError('Unsupported stereo method: {}'.format(stereo_method))

    # Initialize transforms for training and validation loop
    train_transforms = Transforms(
        normalized_image_range=normalized_image_range,
        random_flip_type=random_flip_type,
        random_transform_probability=random_transform_probability,
        random_resize_and_pad=random_resize_and_pad,
        random_crop=random_crop)

    val_transforms = Transforms(
        normalized_image_range=normalized_image_range)

    # Load perturbations
    perturb_models = []
    perturb_model_eval = None

    for i, perturb_path in enumerate(perturb_paths):

        perturb_model = PerturbationsModel(
            n_image_height=n_image_height,
            n_image_width=n_image_width,
            n_image_channel=3,
            output_norm=output_norms[i],
            gradient_scale=gradient_scales[i],
            attack=attack,
            n_perturbation_height=n_perturbation_height,
            n_perturbation_width=n_perturbation_width,
            device=device)

        perturb_model.restore_model(perturb_path)
        perturb_model.eval()
        perturb_models.append(perturb_model)

        if output_norms[i] == 0.02:
            perturb_model_eval = perturb_model

    '''
    Set up stereo model
    '''
    # Build stereo model and restore weights
    stereo_model = StereoModel(stereo_method, num_deform_layers)
    stereo_model.restore_model(stereo_model_restore_path)
    stereo_model.train()

    # Create instance of optimizer
    optimizer = torch.optim.Adam(
        stereo_model.parameters(),
        lr=learning_rates[0],
        betas=(0.9, 0.999))

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
    log('output_norm={}  gradient_scale={}'.format(output_norms, gradient_scales),
        log_path)

    if attack == 'tile':
        log('n_perturbation_height=%d  n_perturbation_width=%d' %
            (n_perturbation_height, n_perturbation_width),
            log_path)

    log('Stereo model settings:', log_path)
    log('stereo_method=%s' %
        (stereo_method),
        log_path)
    log('stereo_model_restore_path=%s' %
        (stereo_model_restore_path),
        log_path)

    log('Training settings:', log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('p_threshold={}'.format(p_threshold),
        log_path)

    log('Checkpoint settings:', log_path)
    log('checkpoint_path=%s' %
        (checkpoint_path),
        log_path)

    log('Training...', log_path)

    # Start Training
    schedule_pos = 0
    train_step = 0
    time_start = time.time()

    for epoch in range(1, learning_schedule[-1] + 1):

        # Set learning rate based on schedule
        if epoch > learning_schedule[schedule_pos]:
            schedule_pos = schedule_pos + 1
            learning_rate = learning_rates[schedule_pos]

            # Update optimizer learning rates
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

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

            if train_ground_truth_available and train_pseudo_ground_truth_available:
                [image0, image1], \
                    [ground_truth, pseudo_ground_truth] = train_transforms.transform(
                        images_arr=[image0, image1],
                        range_maps_arr=[ground_truth, pseudo_ground_truth])
            elif train_ground_truth_available and not train_pseudo_ground_truth_available:
                [image0, image1], \
                    [ground_truth] = train_transforms.transform(
                        images_arr=[image0, image1],
                        range_maps_arr=[ground_truth])
            else:
                [image0, image1] = train_transforms.transform(
                    images_arr=[image0, image1])

            optimizer.zero_grad()

            # Apply the perturbation
            if perturb_models is not [] and random.random() > p_threshold:
                perturb_model = random.choice(perturb_models)
                image0_output, image1_output = perturb_model.forward(image0, image1)
            else:
                image0_output = image0
                image1_output = image1

            outputs = stereo_model.forward(image0_output, image1_output)

            # Compute loss
            loss = stereo_model.compute_loss(
                outputs=outputs,
                ground_truth=ground_truth,
                pseudo_ground_truth=pseudo_ground_truth)

            loss.backward()
            optimizer.step()

            # Log results
            if train_step and (train_step % n_checkpoint) == 0:

                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                if validation_available:
                    # Switch to validation mode
                    stereo_model.eval()

                    with torch.no_grad():

                        val_disparities_origin, val_disparities_output, val_ground_truths = run(
                            dataloader=val_dataloader,
                            transforms=val_transforms,
                            stereo_model=stereo_model,
                            perturb_model=perturb_model_eval,
                            device=device)

                        log('Validation results @ step={}:'.format(train_step), log_path)
                        log('Error of clean images', log_path)
                        eval_utils.evaluate(
                            disparities=val_disparities_origin,
                            ground_truths=val_ground_truths,
                            step=train_step,
                            log_path=log_path)

                        log('Error of perturbed images', log_path)
                        results = eval_utils.evaluate(
                            disparities=val_disparities_output,
                            ground_truths=val_ground_truths,
                            step=train_step,
                            log_path=log_path)

                        best_results = compare_results(
                            results=results,
                            best_results=best_results,
                            log_path=log_path,
                            finetune=True)

                    # Switch back to training
                    stereo_model.train()

                # Save finetuned model to checkpoint
                stereo_model.save_model(
                    save_path=stereo_model_checkpoint_path.format(train_step))

    time_elapse = (time.time() - time_start) / 3600
    time_remain = (n_train_step - train_step) * time_elapse / train_step

    # Perform logging and validation for last step
    log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
        train_step, n_train_step, loss.item(), time_elapse, time_remain),
        log_path)

    if validation_available:
        # Switch to validation mode
        stereo_model.eval()

        with torch.no_grad():

            val_disparities_origin, val_disparities_output, val_ground_truths = run(
                dataloader=val_dataloader,
                transforms=val_transforms,
                stereo_model=stereo_model,
                perturb_model=perturb_model_eval,
                device=device)

            log('Validation results @ step={}:'.format(train_step), log_path)
            log('Error of clean images', log_path)
            eval_utils.evaluate(
                disparities=val_disparities_origin,
                ground_truths=val_ground_truths,
                step=train_step,
                log_path=log_path)

            log('Validation results @ step={}:'.format(train_step), log_path)
            log('Error of perturbed images', log_path)
            results = eval_utils.evaluate(
                disparities=val_disparities_output,
                ground_truths=val_ground_truths,
                step=train_step,
                log_path=log_path)

            best_results = compare_results(
                results=results,
                best_results=best_results,
                log_path=log_path,
                finetune=True)

    # Save finetuned model to checkpoint
    stereo_model.save_model(
        save_path=stereo_model_checkpoint_path.format(train_step))
