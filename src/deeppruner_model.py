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
sys.path.insert(0, os.path.join('external_src', 'DeepPruner'))
sys.path.insert(0, os.path.join('external_src', 'DeepPruner', 'deeppruner'))
sys.path.insert(0, os.path.join('external_src', 'DeepPruner', 'deeppruner', 'models'))
from external_src.DeepPruner.deeppruner.models.deeppruner import DeepPruner
from external_src.DeepPruner.deeppruner.loss_evaluation import loss_evaluation


class DeepPrunerModel(object):
    '''
    Wrapper class for DeepPruner model

    Arg(s):
        method : str
            deeppruner model to use: best (DeepPruner-best), fast (DeepPruner-fast)
        num_deform_layers : int
            number of deformable convolution layers [0, 6, 25]
        device : torch.device
            cpu or cuda device to run on
    '''

    def __init__(self, method='best', num_deform_layers=0, device=torch.device('cuda')):

        self.method = method
        self.num_deform_layers = num_deform_layers
        self.device = device

        # Restore depth prediction network
        self.model = DeepPruner(
            method=self.method,
            num_deform_layers=self.num_deform_layers)

        # Set the cost aggregator scale based on method
        if method == 'fast':
            self.cost_aggregator_scale = 8
        else:
            self.cost_aggregator_scale = 4

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

        # Transform inputs
        image0, image1, \
            padding_top, padding_right = self.transform_inputs(image0, image1)

        # Forward through network
        outputs = self.model(
            left_input=image0,
            right_input=image1)

        if self.mode == 'eval':
            # Get finest output
            output = torch.unsqueeze(outputs[0], dim=1)

            # If we padded the input, then crop
            if padding_top != 0 or padding_right != 0:
                output = output[:, :, padding_top:, :-padding_right]

            return output

        elif self.mode == 'train':
            outputs = [
                torch.unsqueeze(output, dim=1) for output in outputs
            ]

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
            int : padding applied to top of images
            int : padding applied to right of images
        '''

        # Dataset mean and standard deviations
        normal_mean_var = {
            'mean' : [0.485, 0.456, 0.406],
            'std' : [0.229, 0.224, 0.225]
        }

        transform_func = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(**normal_mean_var)])

        n_batch, _, n_height, n_width = image0.shape

        # Apply transform to each image pair
        image0 = torch.chunk(image0, chunks=n_batch, dim=0)
        image1 = torch.chunk(image1, chunks=n_batch, dim=0)

        image0 = torch.stack([
            transform_func(torch.squeeze(image)) for image in image0
        ], dim=0)
        image1 = torch.stack([
            transform_func(torch.squeeze(image)) for image in image1
        ], dim=0)

        downsample_scale = 8.0 * self.cost_aggregator_scale

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

    def compute_loss(self, outputs, ground_truth):
        '''
        Computes training loss

        Arg(s):
            outputs : list[torch.Tensor[float32]]
                list of N x 1 x H x W disparity
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W disparity
        Returns:
            float : loss
        '''

        mask = ground_truth > 0
        mask.detach_()

        loss, _ = loss_evaluation(
            outputs,
            ground_truth,
            mask,
            self.cost_aggregator_scale)

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
            dict[str, torch.Tensor[float32]] : name, parameters pair
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

        self.model.mode = 'training'
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

        self.model.mode = 'evaluation'
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
        Loads weights from a checkpoint

        Arg(s):
            restore_path : str
                path to model weights
        '''

        self.model = torch.nn.DataParallel(self.model)
        state_dict = torch.load(restore_path)
        self.model.load_state_dict(state_dict['state_dict'], strict=True)
