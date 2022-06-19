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

import random, cv2
import torch
import numpy as np
from log_utils import log
from PIL import Image


def root_mean_sq_err(src, tgt):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error between source and target
    '''

    return np.sqrt(np.mean((src - tgt) ** 2))

def mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error between source and target
    '''

    return np.mean(np.abs(src - tgt))

def d1_error(src, tgt):
    '''
    D1 error reported for KITTI 2015.
    A pixel is considered to be correctly estimated if the disparity end-point error is < 3 px or < 5 %.

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : d1 error between source and target (percentage of pixels)
    '''

    E = np.abs(src - tgt)
    n_err = np.count_nonzero(np.logical_and((tgt > 0), np.logical_and(E > 3, (E/np.abs(tgt)) > 0.05)))
    n_total = np.count_nonzero(tgt > 0)
    return n_err/n_total

def end_point_error(src, tgt):
    '''
    Computes end point error for scene flow datasets

    Calls mean absolute error, separate function for ease of naming

     Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error between source and target
    '''

    return mean_abs_err(src, tgt)

def lp_norm(T, p=1.0, axis=None):
    '''
    Computes the Lp-norm of a tensor

    Arg(s):
        T : numpy[float32]
            tensor
        p : float
            norm to use
        axis : int
            axis/dim to compute norm
    Returns:
        float : Lp norm of tensor
    '''

    if p != 0 and axis is None:
        return np.mean(np.abs(T))
    else:
        if p != 0:
            return np.mean(np.sum(np.abs(T) ** p, axis=axis)**(1.0/p))
        else:
            return np.max(np.abs(T))

def absolute_relative_error(true, pred):
    '''
    Computes the absolute relative difference of two tensors,
    filtering out outlier values greater than 3 std deviations from the mean.

    Arg(s):
        true : numpy[float32]
            tensor
        pred : numpy[float32]
            tensor
    Returns:
        float : ARE between true and pred
    '''

    assert(true.shape == pred.shape)

    # Get elementwise difference
    diff = np.subtract(true, pred)

    # Normalize diff elementwise, setting entries to 0
    # that would otherwise experience / 0 error
    x = np.divide(diff, true, out=np.zeros_like(diff), where=(true != 0))

    # Absolute value
    x = np.abs(x)

    # Compute mean and std
    mean, std = np.mean(x), np.std(x)
    hi, lo = (mean + 3*std), (mean - 3*std)

    # Ignore outliers
    x = x[(x > lo) & (x < hi)]

    # Sum over all entries and normalize by number of pixels
    x = np.sum(x) / x.size

    return x

def defense(imgs, type, device,  ksize=None, stdev=None):
    '''
    Applies the specified defense on tensor

    Arg(s):
        img : torch.Tensor[float32]
            B x C x H x W tensor
        type : str
            [jpeg, gaussian]
        device : torch.device
            device to run on
        ksize : int
            kernel size for gaussian filter
        stdev : int
            standard deviation for gaussian filter
    Returns:
        torch.Tensor[float32] : B x C x H x W tensor
    '''

    b_size = imgs.shape[0]
    img_list = []

    for i in range(b_size):

        # Convert tensor to numpy
        img = imgs[i].permute(1, 2, 0)
        img_arr = img.detach().cpu().numpy()

        # Applies the defense
        if type == 'jpeg':
            import common_perturbations_utils
            img_arr = common_perturbations_utils.jpeg_compression(img_arr)

        elif type == 'gaussian':
            import common_perturbations_utils
            img_arr = common_perturbations_utils.gaussian_blur(img_arr, ksize, stdev)

        elif type == 'quantization':
            import common_perturbations_utils
            img_arr = common_perturbations_utils.quantization(img_arr)

        elif type == 'brightness':
            import common_perturbations_utils
            img_arr = common_perturbations_utils.random_brightness(img_arr)

        elif type == 'contrast':
            import common_perturbations_utils
            img_arr = common_perturbations_utils.random_contrast(img_arr)

        elif type == 'gaussian_noise':
            import common_perturbations_utils
            img_arr = common_perturbations_utils.gaussian_noise(img_arr)

        elif type == 'shot_noise':
            import common_perturbations_utils
            img_arr = common_perturbations_utils.shot_noise(img_arr)

        elif type == 'pixelate':
            import common_perturbations_utils
            img_arr = common_perturbations_utils.pixelate(img_arr)

        elif type == 'defocus_blur':
            import common_perturbations_utils
            img_arr = common_perturbations_utils.defocus_blur(img_arr)

        elif type == 'motion_blur':
            import common_perturbations_utils
            img_arr = common_perturbations_utils.motion_blur(img_arr)

        # convert numpy to tensor
        img = torch.from_numpy(img_arr)
        img = img.permute(2, 0, 1)
        img_list.append(img)

    imgs = torch.stack(img_list).float()
    imgs = imgs.to(device)
    return imgs


def evaluate(disparities,
             ground_truths,
             step=0,
             log_path=None):

    assert len(disparities) == len(ground_truths)

    n_sample = len(disparities)

    # Disparity metrics
    d1_ground_truth = np.zeros(n_sample)
    epe_ground_truth = np.zeros(n_sample)

    data = zip(disparities, ground_truths)

    for idx, (disparity, ground_truth) in enumerate(data):

        # Resize output disparity to size of ground truth
        height_output, width_output = disparity.shape[-2:]
        height_ground_truth, width_ground_truth = ground_truth.shape[-2:]

        if height_output != height_ground_truth or width_output != width_ground_truth:
            scale = float(width_ground_truth) / float(width_output)

            disparity = cv2.resize(
                disparity,
                dsize=(width_ground_truth, height_ground_truth),
                interpolation=cv2.INTER_LINEAR)

            disparity = disparity * scale

        # Mask out invalid ground truth
        mask = np.logical_and(ground_truth > 0.0, ~np.isnan(ground_truth))

        disparity = disparity[mask]
        ground_truth = ground_truth[mask]

        # Compute disparity metrics
        d1_ground_truth[idx] = d1_error(disparity, ground_truth)
        epe_ground_truth[idx] = end_point_error(disparity, ground_truth)

    # Disparity metrics
    d1_ground_truth_mean = np.mean(d1_ground_truth * 100.0)
    d1_ground_truth_std = np.std(d1_ground_truth * 100.0)

    epe_ground_truth_mean = np.mean(epe_ground_truth)
    epe_ground_truth_std = np.std(epe_ground_truth)

    log('{:<10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
        '', 'D1-Error ', '+/-', 'EPE', '+/-'),
        log_path)
    log('{:<10}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}'.format(
        '',
        d1_ground_truth_mean,
        d1_ground_truth_std,
        epe_ground_truth_mean,
        epe_ground_truth_std),
        log_path)

    results = {
        'step' : step,
        'd1_ground_truth_mean' : d1_ground_truth_mean,
        'd1_ground_truth_std' : d1_ground_truth_std,
        'epe_ground_truth_mean' : epe_ground_truth_mean,
        'epe_ground_truth_std' : epe_ground_truth_std
    }

    return results
