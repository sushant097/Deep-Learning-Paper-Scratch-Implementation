import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

# features
features_dim = [512, 512, 512, 512, 256, 128, 64, 32, 16]


# Input: [batch_size, in_channels, height, width]

# Scaled weight - He initialization
# "explicitly scale the weights at runtime"
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

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


# Quick apply for scaled weight
def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module


# Uniformly set the hyperparameters of Linears
# "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)"
# apply scaled weights
class ScaledLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)


# Uniformly set the hyperparameters of Conv2d
# "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)"
class ScaledConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = quick_scale(conv)

    def forward(self, x):
        return self.conv(x)


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


# from:: https://github.com/SiskonEmilia/StyleGAN-PyTorch/blob/master/model.py
class AdaIn(nn.Module):
    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result


# Inject Noise
class InjectNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight * noise


# needs to be check
# debug
class ScaleNoise_B(nn.Module):
    """
    Learned per-channels scale factor, used to scale the noise
    """

    def __init__(self, n_channel):
        super().__init__()
        self.weight = Parameter(torch.zeros((1, n_channel, 1, 1)))

    def forward(self, noise):
        return noise * self.weight


# Initial Convolutional Block - With Constant Input
class StyleConvBlockInitial(nn.Module):
    "First block of generator that get the constant value as input "

    def __init__(self, in_channels, dim_latent, dim_input):
        super().__init__()
        # Constant input N x C x H x W
        self.constant = nn.Parameter(torch.randn(1, in_channels, dim_input, dim_input))
        # generator styles
        self.style1 = LearnedAffineTrans_A(dim_latent, in_channels)
        self.style2 = LearnedAffineTrans_A(dim_latent, in_channels)
        # Processing noise modules
        self.noise1 = quick_scale(ScaleNoise_B(in_channels))
        self.noise2 = quick_scale(ScaleNoise_B(in_channels))
        # AdaIn
        self.adaIn = AdaIn(in_channels)
        self.leakyRelu = nn.LeakyReLU(0.2)
        # Conv layer
        self.conv = ScaledConv2d(in_channels, in_channels, 3, padding=1)  # in_channels = out_channels

    def forward(self, latent_, noise):
        # Noise: Gaussian Noise
        # Adding noise -> Introduce Stochastic variation of gen output
        x = self.constant.repeat(noise.shape[0], 1, 1, 1)  # N x C x H x W
        # add noise to constant input
        x = x + self.noise1(noise)
        # Apply adaptive instance norm
        x = self.leakyRelu(self.adaIn(x, self.style1(latent_)))

        # perform conv (3,3)
        x = self.conv(x)
        # add noise to convolved output
        x = x + self.noise2(noise)
        # Apply adaptive instance norm
        x = self.leakyRelu(self.adaIn(x, self.style2(latent_)))

        return x


# General Style Convolutional Blocks
class StyleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super().__init__()
        # Style generators
        self.style1 = LearnedAffineTrans_A(latent_dim, out_channels)
        self.style2 = LearnedAffineTrans_A(latent_dim, out_channels)
        # Noise processing modules
        self.noise1 = quick_scale(ScaleNoise_B(out_channels))
        self.noise2 = quick_scale(ScaleNoise_B(out_channels))
        # AdaIn
        self.adain = AdaIn(out_channels)
        self.leakyRelu = nn.LeakyReLU(0.2)
        # Convolutional layers
        self.conv1 = ScaledConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = ScaledConv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x, prev_result, latent_, noise):
        # Upsample : by generator itself
        # Conv 3*3
        x = self.conv1(prev_result)
        # Gaussian Noise: torch.normal(mean=0,std=torch.ones(result.shape))

        # add noise to constant input
        x = x + self.noise1(noise)
        # Apply adaptive instance norm
        x = self.leakyRelu(self.adaIn(x, self.style1(latent_)))

        # perform conv (3,3)
        x = self.conv2(x)
        # add noise to convolved output
        x = x + self.noise2(noise)
        # Apply adaptive instance norm
        x = self.leakyRelu(self.adaIn(x, self.style2(latent_)))

        return x


# General Convolutional Block
class ConvBlock(nn.Module):
    """
        Used to construct progressive discriminator
    """

    def __init__(self, in_channels, out_channels, kernel1_size, padding1,
                 kernel2_size=None, padding2=None):
        super().__init__()

        if kernel2_size is None:
            kernel2_size = kernel1_size
        if padding2 is None:
            padding2 = padding1

        self.conv = nn.Sequential(
            ScaledConv2d(in_channels, out_channels, kernel1_size, padding=padding1),
            nn.LeakyReLU(0.2),
            ScaledConv2d(out_channels, out_channels, kernel2_size, padding=padding2),
            nn.LeakyReLU(0.2)
        )

    def forward(self, image):
        return self.conv(image)


class MappingNetwork(nn.Module):
    """
     A mapping consists of 8 fully connected layers.
    Used to map the input to an intermediate latent space W.
   Layer: in_c -> out_c
     l1: 512 -> 256
     l2: 256 -> 128
     l3: 128 -> 64
     l4: 64 -> 32
     l5: 32 -> 64
     l6: 64 -> 128
     l7: 128 -> 256
     l8: 256 -> 512
    """

    def __init__(self, z_dims=512):
        super().__init__()
        layers = [PixelNormalization()]
        in_channels = z_dims
        out_channels = z_dims // 2
        for i in range(0, 8):
            layers.append(ScaledLinear(in_channels, out_channels))
            layers.append(nn.LeakyReLU(0.2))
            in_channels = out_channels
            if i + 1 < 4:
                out_channels //= 2
            else:
                out_channels *= 2

        self.mapping = nn.Sequential(*layers)

    def forward(self, latent_z):
        return self.mapping(latent_z)


# Generator
class StyleSynthesisNetwork(nn.Module):
    """
    Main Style Synthesis Module
    """

    def __init__(self, n_fc, dim_latent, dim_input):
        super().__init__()
        self.fcs = MappingNetwork(dim_latent)
        # Style Convolutional Blocks
        layers_conv = [StyleConvBlockInitial(512, dim_latent, dim_input)]
        in_channels = 512
        for i, feature in enumerate(features_dim[1:]):
            if i < 3:
                layers_conv.append(StyleConvBlock(feature, feature, dim_latent))
            else:
                layers_conv.append(StyleConvBlock(in_channels, feature, dim_latent))
                in_channels = feature

        self.convs = nn.ModuleList([*layers_conv])

        # rgbs layers
        layers_rgbs = []
        for feature in features_dim:
            layers_rgbs.append(ScaledConv2d(feature, 3, 1))

        self.to_rgbs = nn.ModuleList([*layers_rgbs])

    def forward(self, latent_z,
                step=0,  # Step means how many layers (count from 4 x 4) are used to train
                alpha=-1,  # Alpha is the parameter of smooth conversion of resolution):
                noise=None,  # TODO: support none noise
                mix_steps=[],  # steps inside will use latent_z[1], else latent_z[0]
                latent_w_center=None,  # Truncation trick in W
                psi=0):  # parameter of truncation
        if type(latent_z) != type([]):
            print('You should use list to package your latent_z')
            latent_z = [latent_z]
        if (len(latent_z) != 2 and len(mix_steps) > 0) or type(mix_steps) != type([]):
            print('Warning: Style mixing disabled, possible reasons:')
            print('- Invalid number of latent vectors')
            print('- Invalid parameter type: mix_steps')
            mix_steps = []

        latent_w = [self.fcs(latent) for latent in latent_z]
        batch_size = latent_w[0].size(0)
        # Truncation trick in W
        # Only usable in estimation
        if latent_w_center is not None:
            latent_w = [latent_w_center + psi * (unscaled_latent_w - latent_w_center)
                        for unscaled_latent_w in latent_w]

            # Generate needed Gaussian noise
            # 5/22: Noise is now generated by outer module
            # noise = []
            result = 0
            current_latent = 0
            # for i in range(step + 1):
            #     size = 4 * 2 ** i # Due to the upsampling, size of noise will grow
            #     noise.append(torch.randn((batch_size, 1, size, size), device=torch.device('cuda:0')))

            for i, conv in enumerate(self.convs):
                # Choose current latent_w
                if i in mix_steps:
                    current_latent = latent_w[1]
                else:
                    current_latent = latent_w[0]

                # Not the first layer, need to upsample
                if i > 0 and step > 0:
                    result_upsample = nn.functional.interpolate(result, scale_factor=2, mode='bilinear',
                                                                align_corners=False)
                    result = conv(result_upsample, current_latent, noise[i])
                else:
                    result = conv(current_latent, noise[i])

                # Final layer, output rgb image
                if i == step:
                    result = self.to_rgbs[i](result)

                    if i > 0 and 0 <= alpha < 1:
                        result_prev = self.to_rgbs[i - 1](result_upsample)
                        result = alpha * result + (1 - alpha) * result_prev

                    # Finish and break
                    break

            return result

    def center_w(self, zs):
        '''
        To begin, we compute the center of mass of W
        '''
        latent_w_center = self.fcs(zs).mean(0, keepdim=True)
        return latent_w_center


# Discriminator
class Discriminator(nn.Module):
    '''
    Main Module
    '''

    def __init__(self):
        super().__init__()
        # from rgbs
        # rgbs layers
        layers_rgbs = []
        for feature in features_dim[::-1]:
            layers_rgbs.append(ScaledConv2d(3, feature, 1))

        self.from_rgbs = nn.ModuleList([*layers_rgbs])

        layers_conv = [ConvBlock(513, 512, 3, 1, 4, 0)]
        features_dim_temp = features_dim[::-1]
        for i, feature in enumerate(features_dim_temp[:-1]):
            if i > 4:
                layers_conv.insert(0, ConvBlock(feature, feature, 3, 1))
            else:
                layers_conv.insert(0, ConvBlock(feature, feature * 2, 3, 1))

        self.convs = nn.ModuleList([*layers_conv])
        self.fc = ScaledLinear(512, 1)

        self.n_layer = 9  # 9 layers network

    def forward(self, image,
                step=0,  # Step means how many layers (count from 4 x 4) are used to train
                alpha=-1):  # Alpha is the parameter of smooth conversion of resolution):
        global result
        for i in range(step, -1, -1):
            # Get the index of current layer
            # Count from the bottom layer (4 * 4)
            layer_index = self.n_layer - i - 1

            # First layer, need to use from_rgb to convert to n_channel data
            if i == step:
                result = self.from_rgbs[layer_index](image)

            # Before final layer, do minibatch stddev
            if i == 0:
                # In dim: [batch, channel(512), 4, 4]
                res_var = result.var(0, unbiased=False) + 1e-8  # Avoid zero
                # Out dim: [channel(512), 4, 4]
                res_std = torch.sqrt(res_var)
                # Out dim: [channel(512), 4, 4]
                mean_std = res_std.mean().expand(result.size(0), 1, 4, 4)
                # Out dim: [1] -> [batch, 1, 4, 4]
                result = torch.cat([result, mean_std], 1)
                # Out dim: [batch, 512 + 1, 4, 4]

            # Conv
            result = self.convs[layer_index](result)

            # Not the final layer
            if i > 0:
                # Downsample for further usage
                result = nn.functional.interpolate(result, scale_factor=0.5, mode='bilinear',
                                                   align_corners=False)
                # Alpha set, combine the result of different layers when input
                if i == step and 0 <= alpha < 1:
                    result_next = self.from_rgbs[layer_index + 1](image)
                    result_next = nn.functional.interpolate(result_next, scale_factor=0.5,
                                                            mode='bilinear', align_corners=False)

                    result = alpha * result + (1 - alpha) * result_next

        # Now, result is [batch, channel(512), 1, 1]
        # Convert it into [batch, channel(512)], so the fully-connetced layer
        # could process it.
        result = result.squeeze(2).squeeze(2)
        result = self.fc(result)
        return result

class FromRGB(nn.Module):
    """
    Input image through weighted scale convolution
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: number of input channels in conv.
        :param out_channels: number of output channels in conv.
        """
        super().__init__()
        self.conv = WSConv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
        self.lRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.lRelu(self.conv(x))


class ToRGB(nn.Module):
    """
    Convert given image to RGB image with weighted scale convolution - kernel_size(1,1), stride=(1,1).
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: number of input channels in conv.
        :param out_channels: number of output channels in conv.
        """
        super().__init__()
        self.conv = WSConv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.conv(x)


class WSConv2d(nn.Module):
    """
    Weighted Scale Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.scale = (2 / (in_channels * kernel_size ** 2)) ** 0.5

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.conv2d(input=x, weight=self.weight * self.scale, bias=self.bias, stride=self.stride,
                        padding=self.padding)


class PixelNormalization(nn.Module):
    """
        Performs pixel normalization in each channel
    """

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        # N x C x H x W
        denominator = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
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
