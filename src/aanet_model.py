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

import os, sys
import torch, torchvision
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join('external_src', 'aanet'))
sys.path.insert(0, os.path.join('external_src', 'aanet', 'dataloader'))
sys.path.insert(0, os.path.join('external_src', 'aanet', 'nets'))
from external_src.aanet.nets import AANet
from external_src.aanet.utils import utils


class AANetModel(object):
    '''
    Wrapper class for AANet model

    Arg(s):
        variant : str
            aanet model to use: regular (AANet), plus (AANet+)
        device : torch.device
            cpu or cuda device to run on
    '''

    def __init__(self, variant='regular', device=torch.device('cuda')):

        self.max_disparity = 192
        self.variant = variant
        self.device = device

        # Restore depth prediction network
        if self.variant == 'regular':
            self.model = AANet(
                max_disp=self.max_disparity,
                num_downsample=2,
                feature_type='aanet',
                no_feature_mdconv=False,
                feature_pyramid=False,
                feature_pyramid_network=True,
                feature_similarity='correlation',
                aggregation_type='adaptive',
                num_scales=3,
                num_fusions=6,
                num_stage_blocks=1,
                num_deform_blocks=3,
                no_intermediate_supervision=True,
                refinement_type='stereodrnet',
                mdconv_dilation=2,
                deformable_groups=2)
        elif self.variant == 'plus':
            self.model = AANet(
                max_disp=self.max_disparity,
                num_downsample=2,
                feature_type='ganet',
                no_feature_mdconv=False,
                feature_pyramid=True,
                feature_pyramid_network=False,
                feature_similarity='correlation',
                aggregation_type='adaptive',
                num_scales=3,
                num_fusions=6,
                num_stage_blocks=1,
                num_deform_blocks=3,
                no_intermediate_supervision=True,
                refinement_type='hourglass',
                mdconv_dilation=2,
                deformable_groups=2)
        else:
            raise NotImplementedError('Specified AANet variant not implemented: {}'.format(self.variant))

        # Move to device
        self.to(self.device)
        self.eval()

    def forward(self, image0, image1):
        '''
        Forwards stereo pair through the network

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W left image
            image1 : torch.Tensor[float32]
                N x C x H x W right image
        Returns:
            torch.Tensor[float32] : N x 1 x H x W disparity if mode is 'eval'
            list[torch.Tensor[float32]] : N x 1 x H x W disparity if mode is 'train'
        '''

        if image1 is None:
            image0, image1 = image0

        # Transform inputs
        image0, image1, \
            padding_top, padding_right = self.transform_inputs(image0, image1)

        # Forward through network
        outputs = self.model(
            left_img=image0,
            right_img=image1)

        if self.mode == 'eval':
            # Get finest output
            output = torch.unsqueeze(outputs[-1], dim=1)

            # If we padded the input, then crop
            if padding_top != 0 or padding_right != 0:
                output = output[:, :, padding_top:, :-padding_right]

            return output

        elif self.mode == 'train':
            outputs = [
                torch.unsqueeze(output, dim=1) for output in outputs
            ]

            # If we padded the input, then crop
            if padding_top != 0 or padding_right != 0:
                outputs = [
                    output[:, :, padding_top:, :-padding_right]
                    for output in outputs
                ]

            return outputs

    def transform_inputs(self, image0, image1):
        '''
        Transforms the stereo pair using standard normalization as a preprocessing step

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W left image
            image1 : torch.Tensor[float32]
                N x C x H x W right image
        Returns:
            torch.Tensor[float32] : N x 3 x H x W left image
            torch.Tensor[float32] : N x 3 x H x W right image
        '''

        normal_mean_var = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }

        transform_func = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(**normal_mean_var)])

        n_batch, _, n_height, n_width = image0.shape

        image0 = torch.chunk(image0, chunks=n_batch, dim=0)
        image1 = torch.chunk(image1, chunks=n_batch, dim=0)

        image0 = torch.stack([
            transform_func(torch.squeeze(image)) for image in image0
        ], dim=0)
        image1 = torch.stack([
            transform_func(torch.squeeze(image)) for image in image1
        ], dim=0)

        if self.variant == 'regular':
            downsample_scale = 12
        elif self.variant == 'plus':
            downsample_scale = 32
        else:
            raise NotImplementedError('Specified AANet variant not implemented: {}'.format(self.variant))

        # Pad images along top and right dimensions
        padding_top = int(downsample_scale - (n_height % downsample_scale))
        padding_right = int(downsample_scale - (n_width % downsample_scale))

        image0 = torch.nn.functional.pad(
            image0,
            (0, padding_right, padding_top, 0, 0, 0),
            mode='constant',
            value=0)
        image1 = torch.nn.functional.pad(
            image1,
            (0, padding_right, padding_top, 0, 0, 0),
            mode='constant',
            value=0)

        return image0, image1, padding_top, padding_right

    def compute_loss(self, outputs, ground_truth, pseudo_ground_truth=None):
        '''
        Computes training loss

        Arg(s):
            outputs : list[torch.Tensor[float32]]
                list of N x 1 x H x W output disparity
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W  disparity
            pseudo_ground_truth : torch.Tensor[float32]
                N x 1 x H x W  disparity
        Returns:
            float : loss
        '''

        mask_ground_truth = \
            (ground_truth > 0) & (ground_truth < self.max_disparity)

        mask_ground_truth.detach_()

        output = outputs[-1]

        # Select outputs where disparity is defined
        loss = torch.nn.functional.smooth_l1_loss(
            output[mask_ground_truth],
            ground_truth[mask_ground_truth],
            reduction='mean')

        if pseudo_ground_truth is not None and torch.max(pseudo_ground_truth) > 0:

            mask_pseudo_ground_truth = ground_truth <= 0

            mask_pseudo_ground_truth.detach_()

            # Compute loss with pseudo groundtruth where groundtruth is not available
            loss_pseudo_ground_truth = torch.nn.functional.smooth_l1_loss(
                output[mask_pseudo_ground_truth],
                pseudo_ground_truth[mask_pseudo_ground_truth],
                reduction='mean')

            loss = loss + loss_pseudo_ground_truth

        return loss

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.model.parameters()

    def named_parameters(self):
        '''
        Returns the list of named parameters in the model

        Returns:
            dict[str, torch.Tensor[float32]] : list of parameters
        '''

        return self.model.named_parameters()

    def train(self, flag_only=False):
        '''
        Sets model to training mode

        Arg(s):
            flag_only : bool
                if set, then only sets the train flag, but not mode
        '''

        if not flag_only:
            self.model.train()

        self.mode = 'train'

    def eval(self, flag_only=False):
        '''
        Sets model to evaluation mode

        Arg(s):
            flag_only : bool
                if set, then only sets the eval flag, but not mode
        '''

        if not flag_only:
            self.model.eval()

        self.mode = 'eval'

    def to(self, device):
        '''
        Moves model to device

        Arg(s):
            device : torch.device
                cpu or cuda device to run on
        '''

        # Move to device
        self.model.to(device)

    def save_model(self, save_path):
        '''
        Stores weights into a checkpoint

        Arg(s):
            save_path : str
                path to model weights
        '''

        checkpoint = {
            'state_dict' : self.model.state_dict()
        }

        torch.save(checkpoint, save_path)

    def restore_model(self, restore_path):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                path to model weights
        '''

        utils.load_pretrained_net(self.model, restore_path, no_strict=True)
        self.model = torch.nn.DataParallel(self.model)
