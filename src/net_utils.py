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

import torch


def lp_projection(v, output_norm, p='inf'):
    '''
    Project v onto lp ball centered at 0, radius output_norm

    Arg(s):
        v : torch.Tensor[float32]
            tensor
        output_norm : float
            radius of lp ball
        p : str
            specified lp ball
    Returns:
        torch.Tensor[float32] : tensor projected onto lp ball
    '''

    if p == 'inf':
        v = torch.sign(v) * torch.min(torch.abs(v), torch.tensor([output_norm], device=v.device))
    else:
        raise ValueError("Currently only supports p='inf'")

    return v

def warp1d_horizontal(image, disparity, padding_mode='border'):
    '''
    Performs horizontal 1d warping

    Arg(s):
        image : torch.Tensor[float32]
            N x C x H x W image
        disparity : torch.Tensor[float32]
            N x 1 x H x W disparity
    Returns:
        torch.Tensor[float32] : N x C x H x W image
    '''

    n_batch, _, n_height, n_width = image.shape

    # Original coordinates of pixels
    x = torch.linspace(0, 1, n_width, dtype=torch.float32, device=image.device) \
        .repeat(n_batch, n_height, 1)
    y = torch.linspace(0, 1, n_height, dtype=torch.float32, device=image.device) \
        .repeat(n_batch, n_width, 1) \
        .transpose(1, 2)

    # Apply shift in X direction
    dx = disparity[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x + dx, y), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    return torch.nn.functional.grid_sample(image,
        grid=(2 * flow_field - 1),
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True)
