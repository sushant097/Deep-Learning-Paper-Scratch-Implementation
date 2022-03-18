import torch
from torch import nn


class ConvBlock(nn.Module):
	""" G: Conv - BN - PReLU -
	    D: Conv - BN - LReLU """

	def __init__(
		self,
		in_channels,
		out_channels,
		discriminator = False,
		use_act = True,
		use_bn = True,
		**kwargs,
	):
		super().__init__()
		self.use_act = use_act
		self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
		self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
		self.act = (
			nn.LeakyReLU(0.2, inplace=True)
			if discriminator
			else nn.PReLU(num_parameters=out_channels)
		)

	def forward(self, x):
		return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpSampleBlock(nn.Module):
	"Upsample by 2 times: If input 128x128 -> 256x256"
	def __init__(self, in_c, scale_factor):
		super().__init__()
		self.conv = nn.Conv2d(in_c, in_c * scale_factor **2, 3, 1, 1) # scale_factor = 2
		self.ps = nn.PixelShuffle(scale_factor) # in_c * 4, H, W --> in_c, H*2, W*2
		self.act = nn.PReLU(num_parameters=in_c)


	def forward(self, x):
		return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
	"""
	Only For Generator
	Adding Skip Connection.
	Conv -> BN -> PReLU -> Conv -> BN -> Elementwise Sum

	"""
	def __init__(self, in_channels):
		super().__init__()
		# output channels = input channels
		self.block1 = ConvBlock(
			in_channels,
			in_channels,
			kernel_size=3,
			stride=1,
			padding=1
		)

		self.block2 = ConvBlock(
			in_channels,
			in_channels,
			kernel_size=3,
			stride=1,
			padding=1,
			use_act=False,
		)


	def forward(self, x):
		out = self.block1(x)
		out = self.block2(out)
		return out + x





class Generator(nn.Module):
	def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
		super().__init__()
		self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
		self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
		self.convblock  = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
		self.upsamples = nn.Sequential(UpSampleBlock(num_channels, scale_factor=2), UpSampleBlock(num_channels, scale_factor=2))
		self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

	def forward(self, x):
		initial = self.initial(x)
		x = self.residuals(initial)
		x = self.convblock(x) + initial # skip connection from intial convblock to output from 64 residuals block
		x = self.upsamples(x)
		return torch.tanh(self.final(x))


class Discriminator(nn.Module):
	def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
		super().__init__()
		blocks = []
		for idx, feature in enumerate(features):
			blocks.append(
				ConvBlock(
					in_channels, 
					feature, 
					kernel_size=3,
					stride = 1+idx % 2, # 1 + (1 % 2) = 1 + 1 =2, 2 if (idx+1)%2==0 else 1
					padding=1,
					discriminator=True,
					use_act = True,
					use_bn = False if idx == 0 else True,
				)
			)

			in_channels = feature


		self.blocks = nn.Sequential(*blocks)
		"""
		Output Shape Calculation:
		Input Image => 96 x96
		We have 4 blocks that have stride=2 which half the image size and others 4 block have stride=2 and padding=1 which not change image size.
		Block 2 - s=2 -> o/p -> 96/2 => 48 x 48
		Block 4 - s=2 -> o/p -> 48/2 => 24 x 24
		Block 6 - s=2 -> o/p -> 24/2 => 12 x 12
		Block 8 - s=2 -> o/p -> 12/2 => 6 x 6

		Total features after Block 8 => 512 x 6 x 6
		A Dense Layer -> Output to 1024.
		Final Dense Layer -> 1024 to 1.
		"""
		self.classifier = nn.Sequential(
			nn.AdaptiveAvgPool2d((6,6)),
			nn.Flatten(),
			nn.Linear(512*6*6, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 1),
		)
	
	def forward(self, x):
		x = self.blocks(x)
		return self.classifier(x)


# def test():
# 	low_resolution = 24 # scale by x4
#
# 	x = torch.randn((5,3, low_resolution, low_resolution))
# 	gen = Generator()
# 	gen_out = gen(x)
# 	disc = Discriminator()
# 	disc_out = disc(gen_out)
#
# 	print(gen_out.shape)
# 	print(disc_out.shape)
#
#
# if __name__ == "__main__":
# 	test()
