import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os

from model import Generator, Discriminator
from utils import get_default_device, to_device
import config as cf

# cifar10
# dataset preparation
print("\n[Phase 1] : Data Preperation")
print("| Preparing LSUN dataset ...")

# create directory to save images
dataset_path = os.path.join(cf.data_dir,)
os.makedirs(dataset_path, exist_ok=True)

# not able to download LSUN dataset in pytorch
dataset = datasets.CIFAR10(
    dataset_path, download=True,
    transform=transforms.Compose([
        transforms.Resize(cf.img_size),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ])
)

dataloader = DataLoader(
    dataset,
    batch_size=cf.batch_size,
    shuffle=True
)
device = get_default_device()  # check gpu if available else cpu
######################### Define
print("\n[Phase 2] : Define Model")

### Instantiating generator and discriminator
netGen = Generator(cf.hidden_dim).to(device)
print(netGen)
netDisc = Discriminator().to(device)
print(netDisc)

######################### Loss & Optimizer
# criterion
mseLoss = nn.MSELoss()
optimizerD = torch.optim.Adam(netDisc.parameters(), lr=cf.learning_rate, betas=(cf.beta, 0.999))
optimizerG = torch.optim.Adam(netGen.parameters(), lr=cf.learning_rate, betas=(cf.beta, 0.999))



# Training
print("\n[Phase 4] : Train model")
for epoch in range(cf.epochs):
    for i, (images,_) in enumerate(dataloader):
        images = to_device(images, device)
        print(images.shape)
        #------------------------
        # Train Discriminator
        #-------------------------
        # train with real images
        optimizerD.zero_grad()
        output = netDisc(images)  # # Forward propagation, this should result in '1'
        label = torch.ones_like(output)  # real label: 1
        errD_real = mseLoss(output, label)  # mse Loss

        # train with fake images
        # generating noise by random sampling
        noise = torch.normal(0, 1, (cf.batch_size, cf.hidden_dim, 1, 1), dtype=torch.float).to(device)
        fake_image = netGen(noise)  # generate fake image

        print(fake_image.shape) # ToDo: Debug

        output = netDisc(fake_image.detach())  # Forward propagation for fake, this should result in '0'
        label = torch.zeros_like(output)  # fake label : 0
        errD_fake = mseLoss(output, label)

        # total discriminator error
        errD = 0.5 * (errD_fake + errD_real)
        errD.backward()
        optimizerD.step()

        #------------------
        #Train Generator
        #------------------

        optimizerG.zero_grad()
        output = netDisc(fake_image)
        label = torch.ones_like(output)  # fool the discriminator, fake label as real label
        errG = mseLoss(output, label) # real label: 1
        errG.backward()
        optimizerG.step()

        print('| Epoch [%2d/%2d] Iter [%3d/%3d] Loss(D): %.4f Loss(G): %.4f ' % (
            epoch, cf.epochs, i + 1, len(dataloader), errD.item(), errG.item()
        ))

    print(": Saving current results...")
    vutils.save_image(
        fake_image.data,
        '%s/fake_samples_%03d.png' % (cf.out_dir, epoch),
        normalize=True
    )

########### Save model
torch.save(netGen.state_dict(), "%s/netG.pth" % (cf.out_dir))
torch.save(netDisc.state_dict(), "%s/netD.pth" % (cf.out_dir))
