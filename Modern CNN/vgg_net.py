import torch
from torch import nn
from torchsummary import summary

def vgg_block(n_convs,  out_channels, in_channels=3):
    layers = []
    for _ in range(n_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, arch, num_classes=10) -> None:
        super().__init__()
        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    vgg = VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)))
    summary(vgg, (3, 334, 224))