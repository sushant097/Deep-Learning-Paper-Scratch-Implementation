import torch
import torch.nn as nn

from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, hidden_dim, out_channels, features_g):
        # Input: NxCx1x1
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Block 1
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=features_g*16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(features_g*16),
            nn.ReLU(inplace=True),

            # 1: Nx1024x4x4 :
            # Block 2
            nn.ConvTranspose2d(in_channels=features_g*16, out_channels=features_g*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*8),
            nn.ReLU(inplace=True),

            #2: Nx512x8x8
            # Block 3
            nn.ConvTranspose2d(in_channels=features_g*8, out_channels=features_g*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*4),
            nn.ReLU(inplace=True),

            # 3: Nx512x16x16
            # Block 4
            nn.ConvTranspose2d(in_channels=features_g*4, out_channels=features_g*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*2),
            nn.ReLU(inplace=True),

            #4: Nx256x32x32
            # Output layer
            nn.ConvTranspose2d(in_channels=features_g*2, out_channels=out_channels, kernel_size=4, stride=2, padding=1)


        )
        # Final Output: Cx64x64
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.model(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels, features_d):

        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input: N x channels_img x 64 x 64
            # Block 1
            nn.Conv2d(in_channels=in_channels, out_channels=features_d, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features_d, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Intermediate: features_d X 32 X 32
            # Block 2
            nn.Conv2d(in_channels=features_d, out_channels=features_d*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features_d*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Intermediate: features_d*2 X 16 X 16
            # Block 3
            nn.Conv2d(in_channels=features_d*2, out_channels=features_d*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features_d*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Intermediate: features_d*4 X 8 X 8
            # Block 4
            nn.Conv2d(in_channels=features_d*4, out_channels=features_d*8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features_d*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Intermediate: features_d*8 X 4 X 4
        )

        self.output = nn.Sequential(
            # Sigmoid is not used as output in D.
            nn.Conv2d(in_channels=features_d*8, out_channels=1, kernel_size=4, stride=2, padding=0)

        )
        # Final output:1x1x1

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
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100

    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, features_d=N)
    print(disc(x).shape)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, features_g=N)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"

    # print(netD)
    # print("Input(=image) : ")
    # print(torch.randn(N, in_channels, H, W).size())
    # y = netD(Variable(torch.randn(noise_dim, in_channels, H, W)))  # Input should be a 4D tensor
    # print("Output of discr(batchsize, channels, width, height) : ")
    #
    # print(y.size())
    # print(net)
    # print( "Input(=z) : ",)
    # print(torch.randn(128, 100, 1, 1).size())
    # y = net(Variable(torch.randn(25, 100, 1, 1)))  # Input should be a 4D tensor
    # print("Generator Output:",y.size())
    # print ("Output(batchsize, channels, width, height) : ",)
