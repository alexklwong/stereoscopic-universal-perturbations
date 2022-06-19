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
import torchvision.transforms.functional as functional


class Transforms(object):
    '''
    Transforms and augmentation class (ground truths only affected by flips)

    Arg(s):
        random_flip_type : list[str]
            none, horizontal, vertical
        random_brightness : list[float]
            brightness adjustment [0, B], from 0 (black image) to B factor increase
        random_contrast : list[float]
            contrast adjustment [0, C], from 0 (gray image) to C factor increase
        random_gamma : list[float]
            gamma adjustment [0, G] from 0 dark to G bright
        random_hue : list[float]
            hue adjustment [-0.5, 0.5] where -0.5 reverses hue to its complement, 0 does nothing
        random_saturation : list[float]
            saturation adjustment [0, S], from 0 (black image) to S factor increase
        random_crop : list[float]
            crops images to fixed size
        random_resize_and_pad : list[float, float],
            minimum percentage of height and width to resize and pad
        random_transform_probability : float
            probability to perform transform
        normalized_image_range : list[float]
            intensity range after normalizing images
    '''

    def __init__(self,
                 random_flip_type=['none'],
                 random_brightness=[-1, -1],
                 random_contrast=[-1, -1],
                 random_gamma=[-1, -1],
                 random_hue=[-1, -1],
                 random_saturation=[-1, -1],
                 random_crop=[-1, -1],
                 random_resize_and_pad=[-1, -1],
                 random_transform_probability=0.50,
                 normalized_image_range=[0, 255]):

        self.do_random_horizontal_flip = True if 'horizontal' in random_flip_type else False
        self.do_random_vertical_flip = True if 'vertical' in random_flip_type else False

        self.do_random_brightness = True if -1 not in random_brightness else False
        self.do_random_contrast = True if -1 not in random_contrast else False
        self.do_random_gamma = True if -1 not in random_gamma else False
        self.do_random_hue = True if -1 not in random_hue else False
        self.do_random_saturation = True if -1 not in random_saturation else False

        self.do_random_crop = True if -1 not in random_crop else False
        self.do_random_resize_and_pad = True if -1 not in random_saturation else False

        self.random_brightness = random_brightness
        self.random_contrast = random_contrast
        self.random_gamma = random_gamma
        self.random_hue = random_hue
        self.random_saturation = random_saturation

        self.random_crop_height = random_crop[0]
        self.random_crop_width = random_crop[1]

        self.random_resize_and_pad_height = random_resize_and_pad[0]
        self.random_resize_and_pad_width = random_resize_and_pad[1]

        self.random_transform_probability = random_transform_probability
        self.normalized_image_range = normalized_image_range

    def transform(self, images_arr, range_maps_arr=[]):
        '''
        Applies transform to images and ground truth

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            range_maps_arr : list[torch.Tensor[float32]]
                list of N x c x H x W tensors
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
            list[torch.Tensor[float32]] : list of transformed N x c x H x W range maps tensors
        '''

        device = images_arr[0].device
        n_batch, _, n_height, n_width = images_arr[0].shape

        do_random_transform = \
            torch.rand(n_batch, device=device) >= self.random_transform_probability

        if self.do_random_crop:

            # Random crop factors
            start_y = torch.randint(
                low=0,
                high=n_height - self.random_crop_height,
                size=(n_batch,),
                device=device)

            start_x = torch.randint(
                low=0,
                high=n_width - self.random_crop_width,
                size=(n_batch,),
                device=device)

            end_y = start_y + self.random_crop_height
            end_x = start_x + self.random_crop_width

            start_yx = [start_y, start_x]
            end_yx = [end_y, end_x]

            images_arr = self.crop(
                images_arr,
                start_yx=start_yx,
                end_yx=end_yx)

            range_maps_arr = self.crop(
                range_maps_arr,
                start_yx=start_yx,
                end_yx=end_yx)

        # Geometric transformations are applied to both images and ground truths
        if self.do_random_horizontal_flip:

            do_horizontal_flip = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) >= 0.50)

            images_arr = self.horizontal_flip(
                images_arr,
                do_horizontal_flip)

            range_maps_arr = self.horizontal_flip(
                range_maps_arr,
                do_horizontal_flip)

        if self.do_random_vertical_flip:

            do_vertical_flip = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) >= 0.50)

            images_arr = self.vertical_flip(
                images_arr,
                do_vertical_flip)

            range_maps_arr = self.vertical_flip(
                range_maps_arr,
                do_vertical_flip)

        # Color transformations are only applied to images
        if self.do_random_brightness:

            do_brightness = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) >= 0.50)

            values = torch.rand(n_batch, device=device)

            brightness_min, brightness_max = self.random_brightness
            factors = (brightness_max - brightness_min) * values + brightness_min

            images_arr = self.adjust_brightness(images_arr, do_brightness, factors)

        if self.do_random_contrast:

            do_contrast = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) >= 0.50)

            values = torch.rand(n_batch, device=device)

            contrast_min, contrast_max = self.random_contrast
            factors = (contrast_max - contrast_min) * values + contrast_min

            images_arr = self.adjust_contrast(images_arr, do_contrast, factors)

        if self.do_random_gamma:

            do_gamma = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) >= 0.50)

            values = torch.rand(n_batch, device=device)

            gamma_min, gamma_max = self.random_gamma
            gammas = (gamma_max - gamma_min) * values + gamma_min

            images_arr = self.adjust_gamma(images_arr, do_gamma, gammas)

        if self.do_random_hue:

            do_hue = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) >= 0.50)

            values = torch.rand(n_batch, device=device)

            hue_min, hue_max = self.random_hue
            factors = (hue_max - hue_min) * values + hue_min

            images_arr = self.adjust_hue(images_arr, do_hue, factors)

        if self.do_random_saturation:

            do_saturation = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) >= 0.50)

            values = torch.rand(n_batch, device=device)

            saturation_min, saturation_max = self.random_saturation
            factors = (saturation_max - saturation_min) * values + saturation_min

            images_arr = self.adjust_saturation(images_arr, do_saturation, factors)

        # Convert all images to float
        images_arr = [
            images.float() for images in images_arr
        ]

        # Do erase transforms on images to simulate occlusions
        if self.do_random_resize_and_pad:

            do_resize_and_pad = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) >= 0.50)

            # Random resize factors
            r_height = torch.randint(
                low=n_height * self.random_resize_and_pad_height,
                high=n_height,
                size=(n_batch,),
                device=device)

            r_width = torch.randint(
                low=n_width * self.random_resize_and_pad_width,
                high=n_width,
                size=(n_batch,),
                device=device)

            shape = [r_height, r_width]

            # Random padding along all sizes
            d_height = (n_height - r_height).int()
            pad_top = (d_height * torch.rand(n_batch, device=device)).int()
            pad_bottom = d_height - pad_top

            d_width = (n_width - r_width).int()
            pad_left = (d_width * torch.rand(n_batch, device=device)).int()
            pad_right = d_width - pad_left

            padding = [pad_top, pad_bottom, pad_left, pad_right]

            images_arr = self.resize_and_pad(
                images_arr,
                do_resize_and_pad=do_resize_and_pad,
                shape=shape,
                padding=padding,
                interpolation='bilinear')

            range_maps_arr = self.resize_and_pad(
                images_arr,
                do_resize_and_pad=do_resize_and_pad,
                shape=shape,
                padding=padding,
                interpolation='nearest',
                rescale_values_by_width=True)

        # Normalize images to a given range
        images_arr = self.normalize_images(
            images_arr,
            normalized_image_range=self.normalized_image_range)

        if len(range_maps_arr) == 0:
            return images_arr
        else:
            return images_arr, range_maps_arr

    def normalize_images(self, images_arr, normalized_image_range=[0, 1]):
        '''
        Normalize image to a given range

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            normalized_image_range : list[float]
                intensity range after normalizing images
        Returns:
            list[torch.Tensor[float32]] : list of normalized N x C x H x W tensors
        '''

        if normalized_image_range == [0, 1]:
            images_arr = [
                images / 255.0 for images in images_arr
            ]
        elif normalized_image_range == [-1, 1]:
            images_arr = [
                2.0 * (images / 255.0) - 1.0 for images in images_arr
            ]
        elif normalized_image_range == [0, 255]:
            pass
        else:
            raise ValueError('Unsupported normalization range: {}'.format(
                normalized_image_range))

        return images_arr

    def horizontal_flip(self, images_arr, do_horizontal_flip):
        '''
        Perform horizontal flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_horizontal_flip : bool
                N booleans to determine if horizontal flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_horizontal_flip[b]:
                    images[b, ...] = functional.hflip(image)

                    if images[b, ...].shape[0] == 1:
                        # We have depth
                        pass
                    elif images[b, ...].shape[0] == 2:
                        # We have optical flow, so negate x-direction
                        images[b, 0, ...] = -1.0 * images[b, 0, ...]
                    else:
                        # We have RGB image
                        pass

            images_arr[i] = images

        return images_arr

    def vertical_flip(self, images_arr, do_vertical_flip):
        '''
        Perform vertical flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_vertical_flip : bool
                N booleans to determine if vertical flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_vertical_flip[b]:
                    images[b, ...] = functional.vflip(image)

                    if images[b, ...].shape[0] == 1:
                        # We have depth
                        pass
                    elif images[b, ...].shape[0] == 2:
                        # We have optical flow, so negate y-direction
                        images[b, 1, ...] = -1.0 * images[b, 1, ...]
                    else:
                        # We have RGB image
                        pass

            images_arr[i] = images

        return images_arr

    def adjust_brightness(self, images_arr, do_brightness, factors):
        '''
        Adjust brightness on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_brightness : bool
                N booleans to determine if brightness is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_brightness[b]:
                    images[b, ...] = functional.adjust_brightness(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_contrast(self, images_arr, do_contrast, factors):
        '''
        Adjust contrast on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_contrast : bool
                N booleans to determine if contrast is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_contrast[b]:
                    images[b, ...] = functional.adjust_contrast(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_gamma(self, images_arr, do_gamma, gammas):
        '''
        Adjust gamma on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_gamma : bool
                N booleans to determine if gamma is adjusted on each sample
            gammas : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_gamma[b]:
                    images[b, ...] = functional.adjust_gamma(image, gammas[b], gain=1)

            images_arr[i] = images

        return images_arr

    def adjust_hue(self, images_arr, do_hue, factors):
        '''
        Adjust hue on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_hue : bool
                N booleans to determine if hue is adjusted on each sample
            gammas : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_hue[b]:
                    images[b, ...] = functional.adjust_hue(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_saturation(self, images_arr, do_saturation, factors):
        '''
        Adjust saturation on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_saturation : bool
                N booleans to determine if saturation is adjusted on each sample
            gammas : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_saturation[b]:
                    images[b, ...] = functional.adjust_saturation(image, factors[b])

            images_arr[i] = images

        return images_arr

    def resize_and_pad(self,
                       images_arr,
                       do_resize_and_pad,
                       shape,
                       padding,
                       interpolation='bilinear',
                       rescale_values_by_width=False):
        '''
        Resizes images and pad them to retain input shape

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_resize_and_pad : bool
                N booleans to determine if image will be resized and padded
            shape : list[int, int]
                height and width to resize
            padding : list[int, int, int, int]
                list of padding for top, bottom, left, right sides
            interpolation : str
                bilinear, nearest
            rescale_values_by_width : bool
                if set, then rescale values based on change in width
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_resize_and_pad[b]:
                    image = images[b, ...]

                    n_width = image.shape[2]

                    r_height = shape[0][b]
                    r_width = shape[1][b]
                    pad_top = padding[0][b]
                    pad_bottom = padding[1][b]
                    pad_left = padding[2][b]
                    pad_right = padding[3][b]

                    # Resize image
                    image = torch.nn.functional.interpolate(
                        image,
                        size=(r_height, r_width),
                        mode=interpolation)

                    # Pad image
                    image = torch.nn.functional.pad(
                        image,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode='constant',
                        value=0)

                    if rescale_values_by_width:
                        image = image * (r_width.float() / n_width.float())

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr

    def crop(self, images_arr, start_yx, end_yx):
        '''
        Performs on on images

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            start_yx : list[int, int]
                top left corner y, x coordinate
            end_yx : list
                bottom right corner y, x coordinate
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            images_cropped = []

            for b, image in enumerate(images):

                start_y = start_yx[0][b]
                start_x = start_yx[1][b]
                end_y = end_yx[0][b]
                end_x = end_yx[1][b]

                # Crop image
                image = image[..., start_y:end_y, start_x:end_x]

                images_cropped.append(image)

            images_arr[i] = torch.stack(images_cropped, dim=0)

        return images_arr
