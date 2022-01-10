import torch.nn as nn
from utils import custom_init

image_size = (32, 32)  # cifar 10
num_classes = 10


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[16, 32, 64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

        layers = []
        in_channels = features[0]
        for i, feature in enumerate(features[1:]):
            layers.append(
                Block(in_channels, feature, stride=1 if i % 2 == 0 else 2))  # even layer, stride=1, else 2
            in_channels = feature
        self.model = nn.Sequential(*layers)

        # The height and width of downsampled image
        # downsampled in conv_layer 1, 3, 5 with stride=2
        ds_size = image_size[0] // 2 ** 3
        # Output layer: discriminator fc
        self.fc_dis = nn.Sequential(nn.Linear(512 * ds_size**2, 1), nn.Sigmoid())
        # Ouput Layer: aux-classifier fc
        self.fc_aux = nn.Sequential(nn.Linear(512 * ds_size**2, num_classes), nn.Softmax())

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        x = x.view(x.shape[0], -1)  # flatten
        validity = self.fc_dis(x)
        label = self.fc_aux(x)

        return validity, label
