# simple implementation of style GAN 2
# StyleGAN 2 changes both the generator and the discriminator of StyleGAN.

"""
Main Changes in Style GAN2 in compare to Style GAN:

1. They remssove the Adaptive Instance Normalization and replace it with
   the weight modulation and demodulation step.
   This is supposed to improve what they call droplet artifacts that are present in generated images,
   which are caused by the normalization in AdaIN operator.
   Then the convolution weights w are modulated and it's demodulated by normalizing.
2. It uses Path length regularization encourages a fixed-size step in W to result in a non-zero,
   fixed-magnitude change in the generated image.

3. It have no progressive growing.
   StyleGAN2 uses residual connections (with down-sampling) in the discriminator and skip connections
   in the generator with up-sampling the RGB outputs from each layer are added - no residual connections in feature maps.
   They show that with experiments that the contribution of low-resolution layers is higher
   at beginning of the training and then high-resolution layers take over.
"""

import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


class MappingNetwork(nn.Module):
    """
    This is an MLP with 8 linear layers.
    The mapping network maps the latent vector z W to an intermediate latent space w in W.
    W space will be disentangled from the image space where the factors of variation become more linear.
    """

    def __init__(self, features: int, n_layers: int):
        """
        * `features` is the number of features in $z$ and $w$
        * `n_layers` is the number of layers in the mapping network.
        """
        super().__init__()

        # Create the MLP
        layers = []
        for i in range(n_layers):
            # [Equalized learning-rate linear layers](#equalized_linear)
            layers.append(EqualizedLinear(features, features))
            # Leaky Relu
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        # Normalize z
        z = F.normalize(z, dim=1)
        # Map z to w
        return self.net(z)


class Generator(nn.Module):
    """
    The generator starts with a learned constant.
    Then it has a series of blocks. The feature map resolution is doubled at each block
    Each block outputs an RGB image and they are scaled up and summed to get the final RGB image.
    """

    def __init__(self, log_resolution: int, d_latent: int, n_features: int = 32, max_features: int = 512):
        """
        * `log_resolution` is the log_2 of image resolution
        * `d_latent` is the dimensionality of w
        * `n_features` number of features in the convolution layer at the highest resolution (final block)
        * `max_features` maximum number of features in any generator block
        """
        super().__init__()

        pass


class GeneratorBlock(nn.Module):
    """
    The generator block consists of two style blocks - 3 x 3 convolutions with style modulation
    and an RGB output.
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()

        pass


class StyleBlock(nn.Module):
    """
    A denotes a linear layer.
    B denotes a broadcast and scaling operation (noise is single channel).
    Style block has a weight modulation convolution layer.
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()
        # Get style vector from w
        # an [equalized learning-rate linear layer]
        pass


class ToRGB(nn.Module):
    """
    Generates an RGB image from a feature map using 1 x 1 convolution.
    """

    def __init__(self, d_latent: int, features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `features` is the number of features in the feature map
        """
        super().__init__()
        pass


class Conv2dWeightModulate(nn.Module):
    """
    ### Convolution with Weight Modulation and Demodulation
    This layer scales the convolution weights by the style vector and demodulates by normalizing it.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int, demodulate: float = True,
                 eps: float = 1e-8):
        super().__init__()
        pass

class Discriminator(nn.Module):
    """
    : StyleGAN 2 Discriminator
    Discriminator first transforms the image to a feature map of the same resolution and then
    runs it through a series of blocks with residual connections.
    The resolution is down-sampled by 2 times at each block while doubling the number of features.
    """

    def __init__(self, log_resolution: int, n_features: int = 64, max_features: int = 512):
        """
        * `log_resolution` is the log_2 of image resolution
        * `n_features` number of features in the convolution layer at the highest resolution (first block)
        * `max_features` maximum number of features in any generator block
        """
        super().__init__()

        pass

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, 3, height, width]`
        """

        pass


class DiscriminatorBlock(nn.Module):
    """
    Discriminator block consists of two 3 * 3 convolutions with a residual connection.
    """
    def __init__(self, in_features, out_features):
        """
            * `in_features` is the number of features in the input feature map
            * `out_features` is the number of features in the output feature map
        """
        super().__init__()
        pass


class MiniBatchStdDev(nn.Module):
    """
    Mini-batch standard deviation calculates the standard deviation
    across a mini-batch (or a subgroups within the mini-batch)
    for each feature in the feature map. Then it takes the mean of all
    the standard deviations and appends it to the feature map as one extra feature.
    """

    def __init__(self, group_size: int = 4):
        """
        * `group_size` is the number of samples to calculate standard deviation across.
        """
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        """
        * `x` is the feature map
        """
        # Check if the batch size is divisible by the group size
        assert x.shape[0] % self.group_size == 0
        # Split the samples into groups of `group_size`, we flatten the feature map to a single dimension
        # since we want to calculate the standard deviation for each feature.
        grouped = x.view(self.group_size, -1)
        # Calculate the standard deviation for each feature among `group_size` samples
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        # Get the mean standard deviation
        std = std.mean().view(1, 1, 1, 1)
        # Expand the standard deviation to append to the feature map
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        # Append (concatenate) the standard deviations to the feature map
        return torch.cat([x, std], dim=1)


class DownSample(nn.Module):
    """
    The down-sample operation smoothens each feature channel and
     scale 2 times using bilinear interpolation.
    This is based on the paper : Making Convolutional Networks Shift-Invariant Again - https://papers.labml.ai/paper/1904.11486.
    """

    def __init__(self):
        super().__init__()
        # Smoothing layer
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Smoothing or blurring
        x = self.smooth(x)
        # Scaled down
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)


class UpSample(nn.Module):
    """
    The up-sample operation scales the image up by 2 times and smoothens each feature channel.
    This is based on the paper Making Convolutional Networks Shift-Invariant Again - https://papers.labml.ai/paper/1904.11486.
    """

    def __init__(self):
        super().__init__()
        # Up-sampling layer
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Smoothing layer
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Up-sample and smoothen
        return self.smooth(self.up_sample(x))


class Smooth(nn.Module):
    """
    ### Smoothing Layer
    This layer blurs each channel
    """

    def __init__(self):
        super().__init__()
        # Blurring kernel
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        # Convert the kernel to a PyTorch tensor
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        # Normalize the kernel
        kernel /= kernel.sum()
        # Save kernel as a fixed parameter (no gradient updates)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        # Padding layer
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        # Get shape of the input feature map
        b, c, h, w = x.shape
        # Reshape for smoothening
        x = x.view(-1, 1, h, w)

        # Add padding
        x = self.pad(x)

        # Smoothen (blur) with the kernel
        x = F.conv2d(x, self.kernel)

        # Reshape and return
        return x.view(b, c, h, w)


class EqualizedLinear(nn.Module):
    """
    ## Learning-rate Equalized Linear Layer
    This uses learning-rate equalized weights for a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `bias` is the bias initialization constant
        """

        super().__init__()
        # Learning-rate equalized weights
        self.weight = EqualizedWeight([out_features, in_features])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # Linear transformation
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    """
    ## Learning-rate Equalized 2D Convolution Layer
    This uses learning-rate equalized weights for a convolution layer.
    """

    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 0):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `padding` is the padding to be added on both sides of each size dimension
        """
        super().__init__()
        # Padding size
        self.padding = padding
        # Learning-rate equalized weights
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        # Convolution
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class EqualizedWeight(nn.Module):
    """
    Learning-rate Equalized Weights Parameter
    This is based on equalized learning rate introduced in the Progressive GAN paper.
    """

    def __init__(self, shape: List[int]):
        """
        * `shape` is the shape of the weight parameter
        """
        super().__init__()

        # He initialization constant
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        # Initialize the weights with $\mathcal{N}(0, 1)$
        self.weight = nn.Parameter(torch.randn(shape))
        # Weight multiplication coefficient

    def forward(self):
        # Multiply the weights by $c$ and return
        return self.weight * self.c


class GradientPenalty(nn.Module):
    "This is the R1 regularization penality from the paper"

    def forward(self, x: torch.Tensor, d: torch.Tensor):


        # Get batch size
        batch_size = x.shape[0]

        # Calculate gradients of D(x) with respect to x.
        # `grad_outputs` is set to 1 since we want the gradients of D(x),
        # and we need to create and retain graph since we have to compute gradients
        # with respect to weight on this loss.
        gradients, *_ = torch.autograd.grad(outputs=d,
                                            inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)

        # Reshape gradients to calculate the norm
        gradients = gradients.reshape(batch_size, -1)
        # Calculate the norm
        norm = gradients.norm(2, dim=-1)
        # Return the loss
        return torch.mean(norm ** 2)


class PathLengthPenalty(nn.Module):
    """
    This regularization encourages a fixed-size step in w to result in a fixed-magnitude
    change in the image.
    """

    def __init__(self, beta: float):
        """
        * `beta` is the constant used to calculate the exponential moving average `a`
        """
        super().__init__()

        # $\beta$
        self.beta = beta
        # Number of steps calculated $N$
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        """
        * `w` is the batch of $w$ of shape `[batch_size, d_latent]`
        * `x` are the generated images of shape `[batch_size, 3, height, width]`
        """

        # Get the device
        device = x.device
        # Get number of pixels
        image_size = x.shape[2] * x.shape[3]
        # Calculate y
        y = torch.randn(x.shape, device=device)
        # Normalize by the square root of image size.
        # This is scaling is not mentioned in the paper but was present in
        # their implementation - https://github.com/NVlabs/stylegan2/blob/master/training/loss.py#L167.
        output = (x * y).sum() / math.sqrt(image_size)

        # Calculate gradients
        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        # Calculate L2-norm
        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        # Regularize after first step
        if self.steps > 0:
            # Calculate a
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            # Calculate the penalty
            loss = torch.mean((norm - a) ** 2)
        else:
            # Return a dummy loss if we can't calculate $a$
            loss = norm.new_tensor(0)

        # Calculate the mean
        mean = norm.mean().detach()
        # Update exponential sum
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        # Increment N
        self.steps.add_(1.)

        # Return the penalty
        return loss