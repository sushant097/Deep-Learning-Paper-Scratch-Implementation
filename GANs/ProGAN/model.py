import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
import numpy as np

from utils import WSConv2d, PixelNormalization, Minibatch_std

"""
Factors is used in Discrmininator and Generator for how much
the channels should be multiplied and expanded for each layer,
so specifically the first 5 layers the channels stay the same,
whereas when we increase the img_size (towards the later layers)
we decrease the number of chanels by 1/2, 1/4, etc.
"""
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class FromRGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = WSConv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
        self.lRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.lRelu(self.conv(x))


class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = WSConv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        return self.conv(x)


class ConvGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, initial_block=False):
        super().__init__()
        if initial_block:
            self.upsample = None
            self.conv1 = WSConv2d(in_channels, out_channels, kernel_size=(4,4), stride=(1,1), padding=(3,3))

        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv1 = WSConv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))

        self.conv2 = WSConv2d(out_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.leakyR = nn.LeakyReLU(0.2)
        self.pn = PixelNormalization()
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        # first convolution and pixel normalization
        x = self.leakyR(self.conv1(x))
        x = self.pn(x)

        # second convolution and pixel normalization
        x = self.leakyR(self.conv2(x))
        x = self.pn(x)
        return x


class ConvDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, initial_block=None):
        super().__init__()

        if initial_block:
            self.miniBatchStd = Minibatch_std()
            self.conv1 = WSConv2d(in_channels+1, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.conv2 = WSConv2d(out_channels, out_channels, kernel_size=(4,4), stride=(1,1)) # in_channels=out_channels
            self.outLayer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(out_channels, 1)
            )

        else:
            self.miniBatchStd = None
            self.conv1 = WSConv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.conv2 = WSConv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.outLayer = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)) # downsampling with avgpooling

        self.leakyR = nn.LeakyReLU(0.2)
        self.pn = PixelNormalization()
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        if self.miniBatchStd is not None:
            x = self.miniBatchStd

            # first convolution and leakyRelu
            x = self.leakyR(self.conv1(x))

            # second convolution and leakyRelu
            x = self.leakyR(self.conv2(x))

            # output layer
            x = self.outLayer(x)
            return x


class Generator(nn.Module):
    def __init__(self, z_dim, out_res):
        super().__init__()
        # initially
        self.depth = 1
        self.alpha = 1 # between 0 to 1, increasing later on
        self.fade_iters = 0
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.current_layers = nn.ModuleList([ConvGBlock(z_dim, z_dim, initial_block=True)])
        self.rgbs_layers = nn.ModuleList([ToRGB(z_dim, 3)])

        for layer in range(2, int(np.log2(out_res))): # np.log2(256) = 8
            if layer < 6:
                # All low resolution blocks 8x8, 16x16, 32x32 with same 512 channels
                in_channels, out_channels = 512, 512

            else:
                # layer > 6 : 5th block(64x64), the number of channels halved for each block
                in_channels, out_channels = int(512 / 2**(layer - 6)), int(512 / 2**(layer - 6))

            self.current_layers.append(ConvGBlock(in_channels, out_channels))
            self.rgbs_layers.append(ToRGB(out_channels, 3))

    def forward(self, x):
        for block in self.current_layers[:self.depth-1]:
            x = block(x)

        out = self.current_layers[self.depth - 1](x)
        x_rgb_out = self.rgbs_layers[self.depth - 1](out)
        if self.alpha < 1:
            x_old = self.upsample(x)
            old_rgb = self.rgbs_layers[self.depth - 2](x_old)
            x_rgb_out = (1-self.alpha) * old_rgb + self.alpha * x_rgb_out

            self.alpha += self.fade_iters

        return x_rgb_out

    def growing_net(self, num_iters):
            self.fade_iters = 1 / num_iters
            self.alpha = 1 / num_iters

            self.depth += 1


class Discriminator(nn.Module):
    def __init__(self, z_dim, out_res):
        super().__init__()
        # initially
        self.depth = 1
        self.alpha = 1  # between 0 to 1, increasing later on
        self.fade_iters = 0
        self.downsample = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))

        self.current_layers = nn.ModuleList([ConvDBlock(z_dim, z_dim, initial_block=True)])
        self.fromRgbLayers = nn.ModuleList([FromRGB(3, z_dim)])

        for layer in range(2, int(np.log2(out_res))):  # np.log2(256) = 8
            if layer < 6:
                # All low resolution blocks 8x8, 16x16, 32x32 with same 512 channels
                in_channels, out_channels = 512, 512

            else:
                # layer > 6 : 5th block(64x64), the number of channels halved for each block
                in_channels, out_channels = int(512 / 2 ** (layer - 6)), int(512 / 2 ** (layer - 6))

            self.current_layers.append(ConvDBlock(in_channels, out_channels))
            self.fromRgbLayers.append(FromRGB(3, in_channels))

    def forward(self, x_rgb):
        x = self.fromRgbLayers[self.depth - 1](x_rgb)

        x = self.current_layers[self.depth - 1](x)

        if self.alpha < 1:
            x_rgb = self.downsample(x_rgb)
            x_old = self.fromRgbLayers[self.depth - 2](x_rgb)
            x = (1 - self.alpha) * x_old + self.alpha * x
            self.alpha += self.fade_iters

        for block in reversed(self.current_net[:self.depth - 1]):
            x = block(x)

        return x

    def growing_net(self, num_iters):
        self.fade_iters = 1 / num_iters
        self.alpha = 1 / num_iters

        self.depth += 1



