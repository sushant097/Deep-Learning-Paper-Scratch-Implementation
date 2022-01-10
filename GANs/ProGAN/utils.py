import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class WSConv2d(nn.Module):
    """
    Weighted Scale Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.scale = (2 / (in_channels * kernel_size**2 ))**0.5

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.conv2d(input=x, weight=self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)


class PixelNormalization(nn.Module):
    """
        Performs pixel normalization in each channel
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        # N x C x H x W
        denominator = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
        return x / denominator


class Minibatch_std(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        size = list(x.size())
        size[1] = 1

        std = torch.std(x, dim=0)
        mean = torch.mean(std)
        return torch.cat((x, mean.repeat(size)), dim=1)
