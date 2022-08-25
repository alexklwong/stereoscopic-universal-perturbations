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

import torch, torchvision
import log_utils
import net_utils


class UniversalPerturbations(torch.nn.Module):
    '''
    Universal perturbations class

    Arg(s):
        n_image_height : int
            height of perturbation image
        n_image_width : int
            width of perturbation image
        n_image_channel : int
            number of channels in perturbation image
        output_norm : float
            infinity norm constraint for perturbation image
        attack : str
            perturbation method to use
        n_perturbation_height : int
            height of perturbation (for tiling)
        n_perturbation_width : int
            width of perturbation (for tiling)
    '''

    def __init__(self,
                 n_image_height,
                 n_image_width,
                 n_image_channel,
                 output_norm,
                 attack,
                 n_perturbation_height,
                 n_perturbation_width):
        super(UniversalPerturbations, self).__init__()

        self.__output_norm = output_norm

        self.__attack = attack

        # Keep track of the number of perturbations we have summed
        self.n_sample = 0

        # Track image dimensions
        self.n_image_height = n_image_height
        self.n_image_width = n_image_width
        self.n_image_channel = n_image_channel

        # Set size of perturbation depending on attck being used
        if attack == 'full':
            # Full-image perturbation; override n_perturbation_<dim> with image dimensions
            n_perturbation_height = n_image_height
            n_perturbation_width = n_image_width
        elif attack == 'tile':
            # Tile attack placed across image. The repeated tile must fit in the image dimension
            if n_image_height % n_perturbation_height != 0:
                raise ValueError('Tile will not fit into image height: {} % {} != 0'.format(n_image_height, n_perturbation_height))
            if n_image_width % n_perturbation_width != 0:
                raise ValueError('Tile will not fit into image width: {} % {} != 0'.format(n_image_width, n_perturbation_width))

            # If tiling, track number of times to repeat tile to form perturbation
            self.repeat_height = n_image_height // n_perturbation_height
            self.repeat_width = n_image_width // n_perturbation_width
        else:
            raise ValueError("Invalid attack: {}".format(attack))

        self.n_perturbation_height = n_perturbation_height
        self.n_perturbation_width = n_perturbation_width

        # Initialize and set weights
        weights = torch.zeros(
            [n_image_channel, n_perturbation_height, n_perturbation_width],
            requires_grad=True)

        self.weights = torch.nn.Parameter(weights)

    def forward(self, x):
        '''
        Applies norm-constraint to perturbation and adds to image

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W image
        Returns:
            torch.Tensor[float32] : adversarially perturbed image
        '''

        # Project it onto l-infinity ball centered at 0, radius self.output_norm.
        projected_weights = net_utils.lp_projection(self.weights, self.__output_norm, p='inf')

        # Set the perturbation attack
        if self.__attack == 'full':
            perturbation = projected_weights
        elif self.__attack == 'tile':
            perturbation = projected_weights.repeat(1, self.repeat_height, self.repeat_width)
        else:
            raise ValueError('Invalid attack: {}'.format(self.__attack))

        # Apply perturbation to image
        return x + perturbation


class PerturbationsModel(object):
    '''
    Adversarial perturbation model

    Arg(s):
        n_image_height : int
            height of perturbation image
        n_image_width : int
            width of perturbation image
        n_image_channel : int
            number of channels in perturbation image
        output_norm : float
            upper (infinity) norm of adversarial noise
        gradient_scale : float
            value to scale gradient tensor by
        attack : str
            perturbation method to use
        n_perturbation_height : int
            height of perturbation for tiling
        n_perturbation_width : int
            width of perturbation for tiling
        device : torch.device
            device to run optimization
    '''

    def __init__(self,
                 n_image_height,
                 n_image_width,
                 n_image_channel,
                 output_norm,
                 gradient_scale,
                 attack,
                 n_perturbation_height,
                 n_perturbation_width,
                 device=torch.device('cuda')):

        self.__output_norm = output_norm
        self.__gradient_scale = gradient_scale

        self.__attack = attack

        self.__n_tile = (n_image_height / float(n_perturbation_height)) * (n_image_width / float(n_perturbation_width))
        self.__n_tile = torch.tensor(self.__n_tile, device=device)

        # Initialize perturbations
        self.perturb0 = UniversalPerturbations(
            n_image_height=n_image_height,
            n_image_width=n_image_width,
            n_image_channel=n_image_channel,
            output_norm=output_norm,
            attack=attack,
            n_perturbation_height=n_perturbation_height,
            n_perturbation_width=n_perturbation_width)

        self.perturb1 = UniversalPerturbations(
            n_image_height=n_image_height,
            n_image_width=n_image_width,
            n_image_channel=n_image_channel,
            output_norm=output_norm,
            attack=attack,
            n_perturbation_height=n_perturbation_height,
            n_perturbation_width=n_perturbation_width)

        self.__device = device
        self.to(device)

    def forward(self, image0, image1):
        '''
        Adds perturbations to images and clamp

        Args:
            image0 : torch.Tensor[float32]
                N x C x H x W left image
            image1 : torch.Tensor[float32]
                N x C x H x W right image
        Returns:
            torch.Tensor[float32] : adversarially perturbed left image
            torch.Tensor[float32] : adversarially perturbed right image
        '''

        # Add to image and clamp between supports of image intensity to get perturbed image
        image0_output = torch.clamp(self.perturb0(image0), 0.0, 1.0)
        image1_output = torch.clamp(self.perturb1(image1), 0.0, 1.0)

        return image0_output, image1_output

    def optimize_perturbations(self,
                               stereo_model,
                               image0,
                               image1,
                               pseudo_ground_truth=None):
        '''
        Optimizes perturbations based on our method.

        Arg(s):
            stereo_model : StereoModel
                StereoModel instance
            image0 : torch.Tensor[float32]
                N x C x H x W left image
            image1 : torch.Tensor[float32]
                N x C x H x W right image
            pseudo_ground_truth : torch.Tensor[float32]
                N x 1 x H x W  pseudo (from another model) ground truth disparity
        Returns:
            float : loss
        '''

        n_batch, _, n_image_height, n_image_width = image0.shape

        # Obtain clean stereo features and original prediction as ground truth
        with torch.no_grad():
            ground_truth = stereo_model.forward(
                image0,
                image1)

            ground_truth += torch.randn(
                [n_batch, 1, n_image_height, n_image_width],
                device=self.__device)

        # Set gradients for image to be true
        image0 = torch.autograd.Variable(image0, requires_grad=True)
        image1 = torch.autograd.Variable(image1, requires_grad=True)

        # Get perturbed images and obtain perturbed stereo features
        image0_perturbed, image1_perturbed = self.forward(image0, image1)

        # Wrap forward in train and eval block using flag only (with frozen batch norms)
        stereo_model.train(flag_only=True)

        outputs = stereo_model.forward(
            image0_perturbed,
            image1_perturbed)

        stereo_model.eval(flag_only=True)

        loss = stereo_model.compute_loss(
            outputs=outputs,
            ground_truth=ground_truth,
            pseudo_ground_truth=pseudo_ground_truth)

        loss.backward()

        grad0 = torch.sum(image0.grad.data, dim=0, keepdim=True)
        grad1 = torch.sum(image1.grad.data, dim=0, keepdim=True)

        # Projection function
        grad0 = net_utils.lp_projection(grad0, self.__gradient_scale, p='inf')
        grad1 = net_utils.lp_projection(grad1, self.__gradient_scale, p='inf')

        # Update the weights
        if self.__attack == 'full':
            self.perturb0.weights.data = self.perturb0.weights.data + torch.squeeze(grad0, dim=0)
            self.perturb1.weights.data = self.perturb1.weights.data + torch.squeeze(grad1, dim=0)

        elif self.__attack == 'tile':
            # Sum tiles from grad0 and grad1 into weights
            h = 0
            while h < self.perturb0.n_image_height:
                w = 0
                while w < self.perturb0.n_image_width:
                    tile_grad0 = torch.squeeze(
                        grad0[:, :, h:(h+self.perturb0.n_perturbation_height), w:(w+self.perturb0.n_perturbation_width)],
                        dim=0)

                    tile_grad1 = torch.squeeze(
                        grad1[:, :, h:(h+self.perturb1.n_perturbation_height), w:(w+self.perturb1.n_perturbation_width)],
                        dim=0)

                    tile_grad0 = tile_grad0 / self.__n_tile
                    tile_grad1 = tile_grad1 / self.__n_tile

                    self.perturb0.weights.data = self.perturb0.weights.data + tile_grad0

                    self.perturb1.weights.data = self.perturb1.weights.data + tile_grad1

                    w += self.perturb0.n_perturbation_width

                h += self.perturb0.n_perturbation_height

        else:
            raise ValueError("Invalid attack: {}".format(self.__attack))

        # Project each perturbation onto the l-infinity ball to satisfy norm constraint (iteratively project)
        self.perturb0.weights.data = \
            net_utils.lp_projection(self.perturb0.weights.data, self.__output_norm, p='inf')
        self.perturb1.weights.data = \
            net_utils.lp_projection(self.perturb1.weights.data, self.__output_norm, p='inf')

        return loss

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = \
            list(self.perturb0.parameters()) + \
            list(self.perturb1.parameters())

        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''

        self.perturb0.train()
        self.perturb1.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.perturb0.eval()
        self.perturb1.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        self.perturb0.to(device)
        self.perturb1.to(device)

    def save_model(self, checkpoint_path, step):
        '''
        Save perturbations to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
        '''

        checkpoint = {}

        # Save training state
        checkpoint['train_step'] = step

        # Save encoder and decoder weights
        checkpoint['perturb0_state_dict'] = self.perturb0.state_dict()
        checkpoint['perturb1_state_dict'] = self.perturb1.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, checkpoint_path):
        '''
        Restore perturbations model

        Arg(s):
            checkpoint_path : str
                path to checkpoint
        Returns:
            int : checkpoint step
        '''

        # Load checkpoint and load state dictionaries
        checkpoint = torch.load(checkpoint_path, map_location=self.__device)

        try:
            self.perturb0.load_state_dict(checkpoint['perturb0_state_dict'])
            self.perturb1.load_state_dict(checkpoint['perturb1_state_dict'])
        except RuntimeError:
            self.perturb0.weights.data = checkpoint['perturb0_state_dict']['weights']
            self.perturb1.weights.data = checkpoint['perturb1_state_dict']['weights']

        return checkpoint['train_step']

    def numpy(self):
        '''
        Exports perturbations as numpy arrays

        Returns:
            numpy[float32] : C x H x W perturbations
            numpy[float32] : C x H x W perturbations
        '''

        # Add perturbations (including output function) to zeros
        if self.__attack == 'full':
            perturb0 = self.perturb0(torch.zeros_like(self.perturb0.weights.data))
            perturb1 = self.perturb1(torch.zeros_like(self.perturb1.weights.data))
        elif self.__attack == 'tile':
            # If tiling, then the output function repeats the tile to image size
            # We select a h x w tile from the overall H x W image dimensions
            shape = [self.perturb0.n_image_channel, self.perturb0.n_image_height, self.perturb0.n_image_width]

            perturb0 = self.perturb0(torch.zeros(shape, device=self.__device))
            perturb0 = perturb0[:, 0:self.perturb0.n_perturbation_height, 0:self.perturb0.n_perturbation_width]

            perturb1 = self.perturb1(torch.zeros(shape, device=self.__device))
            perturb1 = perturb1[:, 0:self.perturb1.n_perturbation_height, 0:self.perturb1.n_perturbation_width]
        else:
            raise ValueError("Invalid attack: {}".format(self.__attack))

        if 'cuda' in self.__device.type:
            perturb0 = perturb0.cpu()
            perturb1 = perturb1.cpu()

        return perturb0.detach().numpy(), perturb1.detach().numpy()

    def log_summary(self,
                    image0,
                    image1,
                    disparity_origin,
                    disparity_output,
                    step,
                    summary_writer,
                    tag='',
                    n_display=4,
                    max_disparity=120):
        '''
        Logs summary to Tensorboard

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W left image
            image1 : torch.Tensor[float32]
                N x C x H x W right image
            disparity_origin : torch.Tensor[float32]
                N x 1 x H x W disparity image
            disparity_output : torch.Tensor[float32]
                N x 1 x H x W disparity image
            step : int
                current step in training
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                prefix to label logged content
            n_display : int
                number of images to display
            max_disparity : int
                max disparity to normalize disparity maps
        '''

        with torch.no_grad():

            tag = tag + '_' if tag != '' else tag

            # Select a subset of images and outputs to visualize
            image0_summary = image0[0:n_display, ...]
            image1_summary = image1[0:n_display, ...]

            disparity_origin_summary = disparity_origin[0:n_display, ...]
            disparity_output_summary = disparity_output[0:n_display, ...]

            disparity_error_summary = \
                torch.abs(disparity_origin_summary - disparity_output_summary) / 0.10

            # Normalize disparity to [0, ~1] range
            disparity_origin_summary = disparity_origin_summary / max_disparity
            disparity_output_summary = disparity_output_summary / max_disparity

            n_batch = image0_summary.shape[0]

            # Load perturbations
            if self.__attack == 'full':
                perturb0_summary = self.perturb0.weights.data
                perturb1_summary = self.perturb1.weights.data
            elif self.__attack == 'tile':
                perturb0_summary = self.perturb0.weights.data.repeat(1, self.perturb0.repeat_height, self.perturb0.repeat_width)
                perturb1_summary = self.perturb1.weights.data.repeat(1, self.perturb1.repeat_height, self.perturb1.repeat_width)
            else:
                raise ValueError('Invalid attack: {}'.format(self.__attack))

            # Shift perturbations values from [1output_norm, output_norm] to [0, 1]
            perturb0_summary = perturb0_summary / (self.__output_norm / 2.0) + 1.0
            perturb1_summary = perturb1_summary / (self.__output_norm / 2.0) + 1.0

            perturb0_summary = perturb0_summary.repeat(n_batch, 1, 1, 1)
            perturb1_summary = perturb1_summary.repeat(n_batch, 1, 1, 1)

            # Move images to CPU and olorize disparity and error maps
            image0_summary = image0_summary.cpu()
            image1_summary = image1_summary.cpu()

            disparity_origin_summary = log_utils.colorize(disparity_origin_summary.cpu(), colormap='plasma')
            disparity_output_summary = log_utils.colorize(disparity_output_summary.cpu(), colormap='plasma')
            disparity_error_summary = log_utils.colorize(disparity_error_summary.cpu(), colormap='hot')

            perturb0_summary = perturb0_summary.cpu()
            perturb1_summary = perturb1_summary.cpu()

            display_summary_image = torch.cat([
                image0_summary,
                image1_summary,
                perturb0_summary,
                perturb1_summary,
                disparity_origin_summary,
                disparity_output_summary,
                disparity_error_summary], dim=2)

            summary_writer.add_image(
                tag + 'image01_perturb01_disparity-origin-output-error',
                torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                global_step=step)
