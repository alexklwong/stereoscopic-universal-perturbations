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

import numpy as np
import torch.utils.data
import data_utils


class StereoDataset(torch.utils.data.Dataset):
    '''
    Loads a stereo pair, and if available ground truth and pseudo ground truth

    Arg(s):
        image0_paths : list[str]
            path to left image
        image1_paths : list[str]
            path to right image
        ground_truth_paths : list[str]
            path to disparity ground truth
        pseudo_ground_truth_paths : list[str]
            path to pseudo (from another model) ground truth disparity
        shape : list[int]
            (H, W) to resize images
    '''

    def __init__(self,
                 image0_paths,
                 image1_paths,
                 ground_truth_paths=None,
                 pseudo_ground_truth_paths=None,
                 shape=(None, None)):

        self.image0_paths = image0_paths
        self.image1_paths = image1_paths

        assert len(self.image0_paths) == len(self.image1_paths)

        if ground_truth_paths is None:
            self.ground_truth_paths = [None] * len(self.image0_paths)
        else:
            self.ground_truth_paths = ground_truth_paths

        if pseudo_ground_truth_paths is None:
            self.pseudo_ground_truth_paths = [None] * len(self.image0_paths)
        else:
            self.pseudo_ground_truth_paths = pseudo_ground_truth_paths

        self.shape = shape

        if shape is not None:
            self.n_height = shape[0]
            self.n_width = shape[1]

        self.data_format = 'CHW'

    def __getitem__(self, index):
        # Load images
        image0 = data_utils.load_image(
            self.image0_paths[index],
            normalize=False,
            data_format=self.data_format)

        image1 = data_utils.load_image(
            self.image1_paths[index],
            normalize=False,
            data_format=self.data_format)

        # Load ground truth, if not available then return zeros
        ground_truth_path = self.ground_truth_paths[index]

        if ground_truth_path is not None:
            ground_truth = data_utils.load_disparity(
                ground_truth_path,
                data_format=self.data_format)
        else:
            ground_truth = np.zeros([1] + list(image0.shape[1:3]))

        pseudo_ground_truth_path = self.pseudo_ground_truth_paths[index]

        if pseudo_ground_truth_path is not None:
            pseudo_ground_truth = data_utils.load_disparity(
                pseudo_ground_truth_path,
                data_format=self.data_format)
        else:
            pseudo_ground_truth = np.zeros([1] + list(image0.shape[1:3]))

        # Resize images and disparities to specified size
        if self.shape is not None and image0.shape[-2:] != self.shape:
            image0 = data_utils.resize(
                image0,
                shape=self.shape,
                interp_type='lanczos',
                data_format=self.data_format)

        if self.shape is not None and image1.shape[-2:] != self.shape:
            image1 = data_utils.resize(
                image1,
                shape=self.shape,
                interp_type='lanczos',
                data_format=self.data_format)

        if self.shape is not None and ground_truth.shape[-2:] != self.shape:
            width = ground_truth.shape[-1]

            ground_truth = data_utils.resize(
                ground_truth,
                shape=self.shape,
                interp_type='nearest',
                data_format=self.data_format)

            # Scale disparity based on change in width
            scale = np.asarray(self.n_width, np.float32) / np.asarray(width, np.float32)
            ground_truth = ground_truth * scale

        if self.shape is not None and pseudo_ground_truth.shape[-2:] != self.shape:
            width = pseudo_ground_truth.shape[-1]

            pseudo_ground_truth = data_utils.resize(
                ground_truth,
                shape=self.shape,
                interp_type='nearest',
                data_format=self.data_format)

            # Scale disparity based on change in width
            scale = np.asarray(self.n_width, np.float32) / np.asarray(width, np.float32)
            pseudo_ground_truth = pseudo_ground_truth * scale

        return image0, image1, ground_truth, pseudo_ground_truth

    def __len__(self):
        return len(self.image0_paths)


class StereoSegmentationDataset(torch.utils.data.Dataset):
    '''
    Loads a stereo pair, and if available ground truth and pseudo ground truth

    Arg(s):
        image0_paths : list[str]
            path to left image
        image1_paths : list[str]
            path to right image
        seg0_paths : list[str]
            path to left segmentation
        ground_truth_paths : list[str]
            path to disparity ground truth
        pseudo_ground_truth_paths : list[str]
            path to pseudo (from another model) ground truth disparity
        shape : list[int]
            (H, W) to resize images
        normalize : bool
            if set, normalize images between 0 and 1
    '''

    def __init__(self,
                 image0_paths,
                 image1_paths,
                 ground_truth_paths=None,
                 pseudo_ground_truth_paths=None,
                 seg0_paths=None,
                 shape=(None, None)):

        self.image0_paths = image0_paths
        self.image1_paths = image1_paths

        assert len(self.image0_paths) == len(self.image1_paths)

        if ground_truth_paths is None:
            self.ground_truth_paths = [None] * len(self.image0_paths)
        else:
            self.ground_truth_paths = ground_truth_paths

        if pseudo_ground_truth_paths is None:
            self.pseudo_ground_truth_paths = [None] * len(self.image0_paths)
        else:
            self.pseudo_ground_truth_paths = pseudo_ground_truth_paths

        if seg0_paths is not None:
            self.seg0_paths = seg0_paths
        else:
            self.seg0_paths = [None] * len(self.image0_paths)

        self.shape = shape

        if shape is not None:
            self.n_height = shape[0]
            self.n_width = shape[1]

        self.data_format = 'CHW'

    def __getitem__(self, index):
        # Load images
        image0 = data_utils.load_image(
            self.image0_paths[index],
            normalize=False,
            data_format=self.data_format)

        image1 = data_utils.load_image(
            self.image1_paths[index],
            normalize=False,
            data_format=self.data_format)

        # Load ground truth, if not available then return zeros
        ground_truth_path = self.ground_truth_paths[index]

        if ground_truth_path is not None:
            ground_truth = data_utils.load_disparity(
                ground_truth_path,
                data_format=self.data_format)
        else:
            ground_truth = np.zeros([1] + list(image0.shape[1:3]))

        pseudo_ground_truth_path = self.pseudo_ground_truth_paths[index]

        if pseudo_ground_truth_path is not None:
            pseudo_ground_truth = data_utils.load_disparity(
                pseudo_ground_truth_path,
                data_format=self.data_format)
        else:
            pseudo_ground_truth = np.zeros([1] + list(image0.shape[1:3]))

        seg_path = self.seg0_paths[index]
        if seg_path is not None:
            seg0 = np.load(seg_path)
            seg0 = np.expand_dims(seg0, 0)

        # Resize images and disparities to specified size
        if self.shape is not None and image0.shape[-2:] != self.shape:
            image0 = data_utils.resize(
                image0,
                shape=self.shape,
                interp_type='lanczos',
                data_format=self.data_format)

        if self.shape is not None and image1.shape[-2:] != self.shape:
            image1 = data_utils.resize(
                image1,
                shape=self.shape,
                interp_type='lanczos',
                data_format=self.data_format)

        if self.shape is not None and ground_truth.shape[-2:] != self.shape:
            width = ground_truth.shape[-1]

            ground_truth = data_utils.resize(
                ground_truth,
                shape=self.shape,
                interp_type='nearest',
                data_format=self.data_format)

            # Scale disparity based on change in width
            scale = np.asarray(self.n_width, np.float32) / np.asarray(width, np.float32)
            ground_truth = ground_truth * scale

        if self.shape is not None and pseudo_ground_truth.shape[-2:] != self.shape:
            width = pseudo_ground_truth.shape[-1]

            pseudo_ground_truth = data_utils.resize(
                ground_truth,
                shape=self.shape,
                interp_type='nearest',
                data_format=self.data_format)

            # Scale disparity based on change in width
            scale = np.asarray(self.n_width, np.float32) / np.asarray(width, np.float32)
            pseudo_ground_truth = pseudo_ground_truth * scale

        if seg_path is not None:
            if self.shape is not None and seg0.shape[-2:] != self.shape:
                seg0 = data_utils.resize(
                    seg0,
                    shape=self.shape,
                    interp_type='nearest',
                    data_format=self.data_format)

        return image0, image1, seg0, ground_truth, pseudo_ground_truth

    def __len__(self):
        return len(self.image0_paths)
