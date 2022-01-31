import torch
import torch.nn as nn

from torch.autograd import Variable


class Generator(nn.Module):
    """
    LSGAN Generator Module:
    Generator module consists of series of deconvolution layers

    layer 1:
        # Input size : input latent vector 'z' with dimension (nz)*1*1
        # Output size: output feature vector with (512)*4*4
    layer 2:
        # Input size : input feature vector with (512)*8*8
        # Output size: output feature vector with (256)*16*16

    layer 3:
        # Input size : input feature vector with (256)*16*16
        # Output size: output feature vector with (128)*16*16

    layer 4:
        # Input size : input feature vector with (128)*16*16
        # Output size: output feature vector with (64)*32*32
    Output:
        # Input size : input feature vector with (64)*32*32
        # Output size: output image with 1x64x64


    """

    def __init__(self, in_channels=100, out_channels=512):
        # input: in_channels-> hidden_dim to fist layer and out_channels-> no.of output channels of first layer

        super().__init__()
        layers = []
        n_layers = 4  # five layer of convolution
        for i in range(n_layers, 0, -1):

            model = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=1 if  i== n_layers else 2,
                    padding=0 if i==n_layers else 1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)

            )

            layers.append(model)
            in_channels = out_channels
            out_channels = out_channels // 2

        self.model = nn.Sequential(*layers)

        self.output = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels= in_channels,
                out_channels=3,
                kernel_size = 4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return self.output(x)


class Discriminator(nn.Module):
    """
    LSGAN Discriminator module
    Discriminator will be consisted with a series of convolution networks
    input : (batch * in_channels * image width * image height)

    layer 1:
        # Input size : input image with dimension (in_channels)*64*64
        # Output size: output feature vector with (out_channels)*32*32
    layer 2:
        # Input size : input feature vector with (out_channels)*32*32
        # Output size: output feature vector with (out_channels*2)*16*16

    layer 3:
        # Input size : input feature vector with (out_channels*2)*16*16
        # Output size: output feature vector with (out_channels*4)*8*8

    layer 4:
        # Input size : input feature vector with (out_channels*4)*8*8
        # Output size: output feature vector with (out_channels*8)*4*4
    Output:
        # Input size : input feature vector with (out_channels*8)*4*4
        # Output size: output probability of fake/real image

    """

    def __init__(self, in_channels=3, out_channels=64):

        super().__init__()
        layers = []
        n_layers = 5  # five layer of convolution
        for i in range(n_layers):
            if i == n_layers - 1:
                model = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=1,  # Discriminator last output is 1 neuron
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=False
                    ),
                    # not sigmoid -- replaced with Squared error loss
                )
            else:
                model = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels * (2 ** i),  # output channel increase as a factor of 2
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels * (2 ** i)),
                    nn.LeakyReLU(0.2, inplace=True)

                )

            layers.append(model)
            in_channels = out_channels * (2 ** i)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).view(-1, 1)


if __name__ == "__main__":
    # Test
    net = Discriminator(
        in_channels=3,
        out_channels=64
    )
    print(net)
    print("Input(=image) : ")
    print(torch.randn(128, 3, 64, 64).size())
    y = net(Variable(torch.randn(128, 3, 64, 64)))  # Input should be a 4D tensor
    print("Output(batchsize, channels, width, height) : ")
    #     # output must be: [128, 1]
    print(y.size())

    net = Generator(
        in_channels=100,
        out_channels=512,
    )
    print(net)
    print( "Input(=z) : ",)
    print(torch.randn(128, 100, 1, 1).size())
    y = net(Variable(torch.randn(128, 100, 1, 1)))  # Input should be a 4D tensor
    print ("Output(batchsize, channels, width, height) : ",)
    # output must be: [128, 3, 64, 64]
    print(y.size())