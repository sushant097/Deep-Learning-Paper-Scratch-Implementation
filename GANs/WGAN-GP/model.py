import torch
import torch.nn as nn

from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        # Input: NxCx1x1
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Block 1
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            # 1: 1024x4x4 :
            # Block 2
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            #2: 512x8x8
            # Block 3
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 3: 512x16x16
            # Block 4
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            #4: 256x32x32
            # Output layer
            nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1)


        )
        # Final Output: 3x64x64
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.model(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels):

        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input: N x channels_img x 64 x 64
            # Block 1
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Intermediate: 256x16x16
            # Block 2
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Intermediate: 512x8x8
            # Block 3
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Intermediate: 1024x4x4
        )

        self.output = nn.Sequential(
            # Sigmoid is not used as output in D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=2, padding=0)

        )
        # Final output:1024x1x1

    def forward(self, x):
        x = self.model(x)
        return self.output(x)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def denorm_image(image):
    return (1+image)/2


if __name__ == "__main__":
    # Test
    netD = Discriminator(
        in_channels=3,
    )
    print(netD)
    print("Input(=image) : ")
    print(torch.randn(128, 3, 32, 32).size())
    y = netD(Variable(torch.randn(128, 3, 32, 32)))  # Input should be a 4D tensor
    print("Output(batchsize, channels, width, height) : ")

    print(y.size())

    net = Generator(
        hidden_dim=100,
        out_channels=3,
    )
    print(net)
    print( "Input(=z) : ",)
    print(torch.randn(128, 100, 1, 1).size())
    y = net(Variable(torch.randn(25, 100, 1, 1)))  # Input should be a 4D tensor
    print("Generator Output:",y.size())
    print ("Output(batchsize, channels, width, height) : ",)
