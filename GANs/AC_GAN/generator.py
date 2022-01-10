import torch
import torch.nn as nn
from utils import custom_init


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=0 if stride == 1 else 1,  # stride=1, padding=0 or stride=2, padding=1
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    #     self.weight_init()
    #
    # def weight_init(self):
    #     for block in self._modules:
    #         try:
    #             for m in self._modules[block]:
    #                 custom_init(m)
    #         except:
    #             custom_init(block)

    def forward(self, x):
        x = self.conv(x)
        # print(f"COnvBlockGen: {x.shape}")
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=110, features=[384, 192, 96, 48, 3]):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        # first linear layer
        self.linear = nn.Linear(self.in_channels, 384)  # 110 is a length latent input vector

        layers = []
        in_feature = features[0]
        for feature in features[1:-1]:
            # print(f"in_feature:{in_feature}, output_feature:{feature}")
            layers.append(Block(
                in_feature, feature, kernel_size=4, stride=1 if in_feature == features[0] else 2,
            ))
            in_feature = feature
        self.model = nn.Sequential(*layers)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(in_feature, features[-1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    #     self.weight_init()
    #
    # def weight_init(self):
    #     for block in self._modules:
    #         try:
    #             for m in self._modules[block]:
    #                 custom_init(m)
    #         except:
    #             custom_init(block)

    def forward(self, x):
        x = x.view(-1, self.in_channels)
        x = self.linear(x)
        # print(x.shape)
        # here error
        x = x.view(-1, 384, 1, 1)  # reshape to required shape
        # print(x.shape)
        x = self.model(x)
        output = self.last(x)
        return output
