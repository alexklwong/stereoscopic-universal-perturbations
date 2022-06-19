# ---------------------------------------------------------------------------
# DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch
#
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Shivam Duggal
# ---------------------------------------------------------------------------

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomSampler(nn.Module):
    def __init__(self, device, number_of_samples):
        super(RandomSampler, self).__init__()

        # Number of offset samples generated by this function: (number_of_samples+1) * (number_of_samples+1)
        # we generate (number_of_samples+1) samples in x direction, and (number_of_samples+1) samples in y direction,
        # and then perform meshgrid like opertaion to generate (number_of_samples+1) * (number_of_samples+1) samples
        self.number_of_samples = number_of_samples
        self.range_multiplier = torch.arange(0.0, number_of_samples + 1, 1, device=device).view(
            number_of_samples + 1, 1, 1)

    def forward(self, min_offset_x, max_offset_x, min_offset_y, max_offset_y):
        """
        Random Sampler:
            Given the search range per pixel (defined by: [[lx(i), ux(i)], [ly(i), uy(i)]]),
            where lx = lower_bound of the hoizontal offset,
                  ux = upper_bound of the horizontal offset,
                  ly = lower_bound of the vertical offset,
                  uy = upper_bound of teh vertical offset, for all pixel i. )
            random sampler generates samples from this search range.
            First the search range is discretized into `number_of_samples` buckets,
            then a random sample is generated from each random bucket.
            ** Discretization is done in both xy directions. ** (similar to meshgrid)

        Args:
            :min_offset_x: Min horizontal offset of the search range.
            :max_offset_x: Max horizontal offset of the search range.
            :min_offset_y: Min vertical offset of the search range.
            :max_offset_y: Max vertical offset of the search range.
        Returns:
            :offset_x: samples representing offset in the horizontal direction.
            :offset_y: samples representing offset in the vertical direction.
        """

        device = min_offset_x.get_device()
        noise = torch.rand(min_offset_x.repeat(1, self.number_of_samples + 1, 1, 1).size(), device=device)

        offset_x = min_offset_x + ((max_offset_x - min_offset_x) / (self.number_of_samples + 1)) * \
            (self.range_multiplier + noise)
        offset_y = min_offset_y + ((max_offset_y - min_offset_y) / (self.number_of_samples + 1)) * \
            (self.range_multiplier + noise)

        offset_x = offset_x.unsqueeze_(1).expand(-1, offset_y.size()[1], -1, -1, -1)
        offset_x = offset_x.contiguous().view(
            offset_x.size()[0], offset_x.size()[1] * offset_x.size()[2], offset_x.size()[3], offset_x.size()[4])

        offset_y = offset_y.unsqueeze_(2).expand(-1, -1, offset_y.size()[1], -1, -1)
        offset_y = offset_y.contiguous().view(
            offset_y.size()[0], offset_y.size()[1] * offset_y.size()[2], offset_y.size()[3], offset_y.size()[4])

        return offset_x, offset_y


class Evaluate(nn.Module):
    def __init__(self, left_features, filter_size, evaluation_type='softmax', temperature=10000):
        super(Evaluate, self).__init__()
        self.temperature = temperature
        self.filter_size = filter_size
        self.softmax = torch.nn.Softmax(dim=1)
        self.evaluation_type = evaluation_type

        device = left_features.get_device()
        self.left_x_coordinate = torch.arange(0.0, left_features.size()[3], device=device).repeat(
            left_features.size()[2]).view(left_features.size()[2], left_features.size()[3])

        self.left_x_coordinate = torch.clamp(self.left_x_coordinate, min=0, max=left_features.size()[3] - 1)
        self.left_x_coordinate = self.left_x_coordinate.expand(left_features.size()[0], -1, -1).unsqueeze(1)

        self.left_y_coordinate = torch.arange(0.0, left_features.size()[2], device=device).unsqueeze(1).repeat(
            1, left_features.size()[3]).view(left_features.size()[2], left_features.size()[3])

        self.left_y_coordinate = torch.clamp(self.left_y_coordinate, min=0, max=left_features.size()[3] - 1)
        self.left_y_coordinate = self.left_y_coordinate.expand(left_features.size()[0], -1, -1).unsqueeze(1)

    def forward(self, left_features, right_features, offset_x, offset_y):
        """
        PatchMatch Evaluation Block
        Description:    For each pixel i, matching scores are computed by taking the inner product between the
                left feature and the right feature: score(i,j) = feature_left(i), feature_right(i+disparity(i,j))
                for all candidates j. The best k disparity value for each pixel is carried towards the next iteration.

                As per implementation,
                the complete disparity search range is discretized into intervals in
                DisparityInitialization() function. Corresponding to each disparity interval, we have multiple samples
                to evaluate. The best disparity sample per interval is the output of the function.

        Args:
            :left_features: Left Image Feature Map
            :right_features: Right Image Feature Map
            :offset_x: samples representing offset in the horizontal direction.
            :offset_y: samples representing offset in the vertical direction.

        Returns:
            :offset_x: horizontal offset evaluated as the best offset to generate NNF.
            :offset_y: vertical offset evaluated as the best offset to generate NNF.

        """

        right_x_coordinate = torch.clamp(self.left_x_coordinate - offset_x, min=0, max=left_features.size()[3] - 1)
        right_y_coordinate = torch.clamp(self.left_y_coordinate - offset_y, min=0, max=left_features.size()[2] - 1)

        right_x_coordinate -= right_x_coordinate.size()[3] / 2
        right_x_coordinate /= (right_x_coordinate.size()[3] / 2)
        right_y_coordinate -= right_y_coordinate.size()[2] / 2
        right_y_coordinate /= (right_y_coordinate.size()[2] / 2)

        samples = torch.cat((right_x_coordinate.unsqueeze(4), right_y_coordinate.unsqueeze(4)), dim=4)
        samples = samples.view(samples.size()[0] * samples.size()[1],
                               samples.size()[2],
                               samples.size()[3],
                               samples.size()[4])

        offset_strength = torch.mean(-1.0 * (torch.abs(left_features.expand(
            offset_x.size()[1], -1, -1, -1) - F.grid_sample(right_features.expand(
                offset_x.size()[1], -1, -1, -1), samples))), dim=1) * self.temperature

        offset_strength = offset_strength.view(left_features.size()[0],
                                               offset_strength.size()[0] // left_features.size()[0],
                                               offset_strength.size()[1],
                                               offset_strength.size()[2])

        if self.evaluation_type == "softmax":
            offset_strength = torch.softmax(offset_strength, dim=1)
            offset_x = torch.sum(offset_x * offset_strength, dim=1).unsqueeze(1)
            offset_y = torch.sum(offset_y * offset_strength, dim=1).unsqueeze(1)
        else:
            offset_strength = torch.argmax(offset_strength, dim=1).unsqueeze(1)
            offset_x = torch.gather(offset_x, index=offset_strength, dim=1)
            offset_y = torch.gather(offset_y, index=offset_strength, dim=1)

        return offset_x, offset_y


class Propagation(nn.Module):
    def __init__(self, device, filter_size):
        super(Propagation, self).__init__()
        self.filter_size = filter_size
        label = torch.arange(0, self.filter_size, device=device).repeat(self.filter_size).view(
            self.filter_size, 1, 1, 1, self.filter_size)

        self.one_hot_filter_h = torch.zeros_like(label).scatter_(0, label, 1).float()

        label = torch.arange(0, self.filter_size, device=device).repeat(self.filter_size).view(
            self.filter_size, 1, 1, self.filter_size, 1).long()

        self.one_hot_filter_v = torch.zeros_like(label).scatter_(0, label, 1).float()

    def forward(self, offset_x, offset_y, propagation_type="horizontal"):
        """
        PatchMatch Propagation Block
        Description:    Particles from adjacent pixels are propagated together through convolution with a
                        one-hot filter, which en-codes the fact that we allow each pixel
                        to propagate particles to its 4-neighbours.
        Args:
            :offset_x: samples representing offset in the horizontal direction.
            :offset_y: samples representing offset in the vertical direction.
            :device: Cuda/ CPU device
            :propagation_type (default:"horizontal"): In order to be memory efficient, we use separable convolutions
                                                    for propagtaion.

        Returns:
            :aggregated_offset_x: Horizontal offset samples aggregated from the neighbours.
            :aggregated_offset_y: Vertical offset samples aggregated from the neighbours.

        """

        offset_x = offset_x.view(offset_x.size()[0], 1, offset_x.size()[1], offset_x.size()[2], offset_x.size()[3])
        offset_y = offset_y.view(offset_y.size()[0], 1, offset_y.size()[1], offset_y.size()[2], offset_y.size()[3])

        if propagation_type is "horizontal":
            aggregated_offset_x = F.conv3d(offset_x, self.one_hot_filter_h, padding=(0, 0, self.filter_size // 2))
            aggregated_offset_y = F.conv3d(offset_y, self.one_hot_filter_h, padding=(0, 0, self.filter_size // 2))

        else:
            aggregated_offset_x = F.conv3d(offset_x, self.one_hot_filter_v, padding=(0, self.filter_size // 2, 0))
            aggregated_offset_y = F.conv3d(offset_y, self.one_hot_filter_v, padding=(0, self.filter_size // 2, 0))

        aggregated_offset_x = aggregated_offset_x.permute([0, 2, 1, 3, 4])
        aggregated_offset_x = aggregated_offset_x.contiguous().view(
            aggregated_offset_x.size()[0],
            aggregated_offset_x.size()[1] * aggregated_offset_x.size()[2],
            aggregated_offset_x.size()[3],
            aggregated_offset_x.size()[4])

        aggregated_offset_y = aggregated_offset_y.permute([0, 2, 1, 3, 4])
        aggregated_offset_y = aggregated_offset_y.contiguous().view(
            aggregated_offset_y.size()[0],
            aggregated_offset_y.size()[1] * aggregated_offset_y.size()[2],
            aggregated_offset_y.size()[3],
            aggregated_offset_y.size()[4])

        return aggregated_offset_x, aggregated_offset_y


class PropagationFaster(nn.Module):
    def __init__(self):
        super(PropagationFaster, self).__init__()

    def forward(self, offset_x, offset_y, device, propagation_type="horizontal"):
        """
        Faster version of PatchMatch Propagation Block
        This version uses a fixed propagation filter size of size 3. This implementation is not recommended
        and is used only to do the propagation faster.

        Description:    Particles from adjacent pixels are propagated together through convolution with a
                        one-hot filter, which en-codes the fact that we allow each pixel
                        to propagate particles to its 4-neighbours.
        Args:
            :offset_x: samples representing offset in the horizontal direction.
            :offset_y: samples representing offset in the vertical direction.
            :device: Cuda/ CPU device
            :propagation_type (default:"horizontal"): In order to be memory efficient, we use separable convolutions
                                                    for propagtaion.

        Returns:
            :aggregated_offset_x: Horizontal offset samples aggregated from the neighbours.
            :aggregated_offset_y: Vertical offset samples aggregated from the neighbours.

        """

        self.vertical_zeros = torch.zeros((offset_x.size()[0], offset_x.size()[1], 1, offset_x.size()[3])).to(device)
        self.horizontal_zeros = torch.zeros((offset_x.size()[0], offset_x.size()[1], offset_x.size()[2], 1)).to(device)

        if propagation_type is "horizontal":
            offset_x = torch.cat((torch.cat((self.horizontal_zeros, offset_x[:, :, :, :-1]), dim=3),
                                  offset_x,
                                  torch.cat((offset_x[:, :, :, 1:], self.horizontal_zeros), dim=3)), dim=1)
            offset_y = torch.cat((torch.cat((self.horizontal_zeros, offset_y[:, :, :, :-1]), dim=3),
                                  offset_y,
                                  torch.cat((offset_y[:, :, :, 1:], self.horizontal_zeros), dim=3)), dim=1)

        else:
            offset_x = torch.cat((torch.cat((self.vertical_zeros, offset_x[:, :, :-1, :]), dim=2),
                                  offset_x,
                                  torch.cat((offset_x[:, :, 1:, :], self.vertical_zeros), dim=2)), dim=1)
            offset_y = torch.cat((torch.cat((self.vertical_zeros, offset_y[:, :, :-1, :]), dim=2),
                                  offset_y,
                                  torch.cat((offset_y[:, :, 1:, :], self.vertical_zeros), dim=2)), dim=1)

        return offset_x, offset_y


class PatchMatch(nn.Module):
    def __init__(self, patch_match_args):
        super(PatchMatch, self).__init__()
        self.propagation_filter_size = patch_match_args.propagation_filter_size
        self.number_of_samples = patch_match_args.sample_count
        self.iteration_count = patch_match_args.iteration_count
        self.evaluation_type = patch_match_args.evaluation_type
        self.softmax_temperature = patch_match_args.softmax_temperature
        self.propagation_type = patch_match_args.propagation_type

        self.window_size_x = patch_match_args.random_search_window_size[0]
        self.window_size_y = patch_match_args.random_search_window_size[1]

    def forward(self, left_features, right_features):
        """
        Differential PatchMatch Block
        Description:    In this work, we unroll generalized PatchMatch as a recurrent neural network,
                        where each unrolling step is equivalent to each iteration of the algorithm.
                        This is important as it allow us to train our full model end-to-end.
                        Specifically, we design the following layers:
                            - Initialization or Paticle Sampling
                            - Propagation
                            - Evaluation
        Args:
            :left_features: Left Image feature map
            :right_features: Right image feature map

        Returns:
            :offset_x: offset for each pixel in the left_features corresponding to the
                                                        right_features in the horizontal direction.
            :offset_y: offset for each pixel in the left_features corresponding to the
                                                        right_features in the vertical direction.

            :x_coordinate: X coordinate corresponding to each pxiel.
            :y_coordinate: Y coordinate corresponding to each pxiel.

            (Offsets and the xy_cooridnates returned are used to generated the NNF field later for reconstruction.)

        """

        device = left_features.get_device()
        if self.propagation_type is "faster_filter_3_propagation":
            self.propagation = PropagationFaster()
        else:
            self.propagation = Propagation(device, self.propagation_filter_size)

        self.evaluate = Evaluate(left_features, self.propagation_filter_size,
                                 self.evaluation_type, self.softmax_temperature)
        self.uniform_sampler = RandomSampler(device, self.number_of_samples)

        min_offset_x = torch.zeros((left_features.size()[0], 1, left_features.size()[2],
                                    left_features.size()[3])).to(device) - left_features.size()[3]
        max_offset_x = min_offset_x + 2 * left_features.size()[3]
        min_offset_y = min_offset_x + left_features.size()[3] - left_features.size()[2]
        max_offset_y = min_offset_y + 2 * left_features.size()[2]

        for prop_iter in range(self.iteration_count):
            offset_x, offset_y = self.uniform_sampler(min_offset_x, max_offset_x,
                                                      min_offset_y, max_offset_y)

            offset_x, offset_y = self.propagation(offset_x, offset_y, device, "horizontal")
            offset_x, offset_y = self.evaluate(left_features,
                                               right_features,
                                               offset_x, offset_y)

            offset_x, offset_y = self.propagation(offset_x, offset_y, device, "vertical")
            offset_x, offset_y = self.evaluate(left_features,
                                               right_features,
                                               offset_x, offset_y)

            min_offset_x = torch.clamp(offset_x - self.window_size_x // 2, min=-left_features.size()[3],
                                       max=left_features.size()[3])
            max_offset_x = torch.clamp(offset_x + self.window_size_x // 2, min=-left_features.size()[3],
                                       max=left_features.size()[3])
            min_offset_y = torch.clamp(offset_y - self.window_size_y // 2, min=-left_features.size()[2],
                                       max=left_features.size()[2])
            max_offset_y = torch.clamp(offset_y + self.window_size_y // 2, min=-left_features.size()[2],
                                       max=left_features.size()[2])

        return offset_x, offset_y, self.evaluate.left_x_coordinate, self.evaluate.left_y_coordinate