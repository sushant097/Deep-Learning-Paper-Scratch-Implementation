
import torch
import torchvision.transforms.functional
from torch import nn

"""
Each step in contraction path have two 3x3 conv layers followed by ReLU activations.

"""

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels:int) -> None:
        super().__init__()
        
        # first 3x3 conv layers
        self.first = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        
        # second 3x3 conv layers
        self.second = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        
    def forward(self, x:torch.Tensor):
        "Apply the two convolution layers and activations"
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)

class Downsample(nn.Module):
    "Reduce the feature with 2x2 maxpool"
    def __init__(self) -> None:
        super().__init__()
        
        self.pool = nn.MaxPool3d(2)
        
    def forward(self, x:torch.Tensor):
        return self.pool(x)

class Upsample(nn.Module):
    "Each step in expansion path up-samples the feature maps with 2x2 up-convolution."
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x:torch.Tensor):
        return self.up(x)
    
class CropAndConcat(nn.Module):
    "In every step of expansion path corresponding feature  map from the contracting path concatenated with the current feature map."
    def forward(self, x:torch.Tensor, contracting_x:torch.Tensor):
        """
        params:
              x: current feature map in the expansion paath
              contracting_x: corresponding feature map from the contracting path
        """
        # Crop the feature map from the contracting path to the size of the current feature map
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        
        return x


class Unet(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()
        
        # double conv layers for contracting path.
        # no. of features gets doubles at each step, starting from 64.
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        # downsample layer for contraction path
        self.down_sample = nn.ModuleList([Downsample() for _ in range(4)])
        # two bottom mid convolution layers at the lowest resolution
        self.middle_conv = DoubleConvolution(512, 1024)
        # upsampling layers for the expansion path. The no. of features is halved with up-sampling.
        self.up_sample = nn.ModuleList([Upsample(i, o) for i, o in 
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        
        # double convolutional layers for the expansion 
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in 
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])                               
        # crop and concate layers for the expansion path
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        
        # final 1x1 convolution layer to produce output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor):
        # array to store output of contracting path for later concatenation
        pass_through = []
        # contracting path
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            # collect the output
            pass_through.append(x)
            # downsample
            x = self.down_sample[i](x)
            
        # two conv 3x3 at bottom layer
        x = self.middle_conv(x)
        
        
        # Expansion path
        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
                
            # contracting the output of the contracting path
            x = self.concat[i](x, pass_through.pop())
            
            # two 3x3 up conv layers
            x = self.up_conv[i](x)
            
        x = self.final_conv(x)
        return x
        


if __name__ == "__main__":
    # x has shape of (batch_size, channels, depth, height, width)
    x_test = torch.randn(1, 1, 96, 96, 96)
    print("The shape of input: ", x_test.shape)

    unet = Unet()
    output = unet(x_test)
    print(output.shape)