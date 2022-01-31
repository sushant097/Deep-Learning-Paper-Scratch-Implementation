import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

# We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)
# Input: [batch_size, in_channels, height, width]

# "explicitly scale the weights at runtime"


# EqualLR
class ScaleW:
    '''
    Constructor: name - name of attribute to be scaled
    '''

    def __init__(self, name):
        self.name = name

    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)
        return hook

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


# equal_lr
# Quick apply for scaled weight
def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module


# class WSConv2d(nn.Module):
#     """
#     Weighted Scale Convolution
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super().__init__()
#         self.padding = padding
#         self.stride = stride
#         self.scale = (2 / (in_channels * kernel_size ** 2)) ** 0.5
#
#         self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
#         self.bias = Parameter(torch.Tensor(out_channels))
#
#         nn.init.normal_(self.weight)
#         nn.init.zeros_(self.bias)
#
#     def forward(self, x):
#         return F.conv2d(input=x, weight=self.weight * self.scale, bias=self.bias, stride=self.stride,
#                         padding=self.padding)


# class FC(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.linear = nn.Linear(in_channels, out_channels)
#
#     def forward(self, x):
#         return F.leaky_relu(self.linear(x), 0.2, inplace=True)
#
#
# class MappingNetwork(nn.Module):
#     def __init__(self, z_dims=512):
#         super(MappingNetwork, self).__init__()
#         layers = []
#         in_channels = z_dims
#         out_channels = z_dims // 2
#         for i in range(0, 8):
#             layers.append(FC(in_channels, out_channels))
#             in_channels = out_channels
#             if i + 1 < 4:
#                 out_channels //= 2
#             else:
#                 out_channels *= 2
#
#         self.model = nn.Sequential(PixelNormalization(), *layers)
#
#     def forward(self, x):
#         return self.model(x)

class ScaledConv2d(nn.Module):
    """
    Uniformly set the parameters of Conv2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = quick_scale(conv)

    def forward(self, x):
        return self.conv(x)


class ScaledLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        linear = nn.Linear(in_channels, out_channels)

        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)


class AdaIn(nn.Module):
    def __init__(self, style_dim, channel):
        super().__init__()
        self.channel = channel
        self.instance_norm = nn.InstanceNorm2d(channel)
        self.linear = ScaledLinear(style_dim, channel * 2)

    def forward(self, image, style):
        style = self.linear(style).view(2, -1, self.channel, 1, 1)
        image = self.instance_norm(image)

        result = (image * (style[0] + 1)) + style[1] # affine transform with multiply with scale and adding bias
        return result


 # NoiseInjection_
class ScaleNoiseB_(nn.Module):
    """
    Learned per-channels scale factor, used to scale the noise - Inplace
    """
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight * noise


#  NoiseInjection
class ScaleNoiseB(nn.Module):
    """
    Learned per-channels scale factor, used to scale the noise
    """
    def __init__(self, channels):
        super().__init__()
        injection = ScaleNoiseB_(channels)
        self.injection = quick_scale(injection)

    def forward(self, x):
        return self.injection(x)


# Learned affine transform A
class LearnedAffineTrans_A(nn.Module):
    """
    Learned affine transform A, this module is used to transform
    intermediate vector w into a style vector
    """

    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = ScaledLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


class PixelNormalization(nn.Module):
    """
        Performs pixel normalization on every element of input vector
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        # N x C x H x W
        denominator = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
        return x / denominator


# different, ToDO: Debug
class Minibatch_std(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        size = list(x.size())
        size[1] = 1

        std = torch.std(x, dim=0)
        mean = torch.mean(std)
        return torch.cat((x, mean.repeat(size)), dim=1)




