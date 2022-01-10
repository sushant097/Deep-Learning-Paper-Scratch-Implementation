import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
from tqdm import tqdm

from discriminator import Discriminator
from generator import Generator
from utils import custom_init, compute_acc, to_device, get_default_device, denorm, show_images
from config import *

dataset = CIFAR10(
    root=data_dir, download=True,
    transform=transforms.Compose([
        transforms.Scale((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)  # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    ])
)

dataloader = DataLoader(dataset, batch_size=batch_size)
device = get_default_device()  # check gpu if available else cpu

# instantiate generator
netG = Generator(noise_dim).to(device)  # hidden latent vector length
netG.apply(custom_init)  # apply custom intitialization to generator
print(netG)

# instantiate discriminator
netD = Discriminator(in_channels=3)
netD = to_device(netD, device)
print(netD)

# defining Optimizer
optimD = optim.Adam(netD.parameters(), lr)
optimG = optim.Adam(netG.parameters(), lr)

# defining Loss
disc_criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()

# noise for evaluation
eval_noise = torch.FloatTensor(batch_size, noise_dim, 1, 1).normal_(0, 1)
eval_noise_ = np.random.normal(0, 1, (batch_size, noise_dim))
eval_label = np.random.randint(0, num_classes, batch_size)
eval_onehot = np.zeros((batch_size, num_classes))
eval_onehot[np.arange(batch_size), eval_label] = 1
eval_noise_[np.arange(batch_size), :num_classes] = eval_onehot[np.arange(batch_size)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(batch_size, noise_dim, 1, 1))
eval_noise.to(device)


def save_samples(index, latent_tensors):
    fake_images = netG(latent_tensors)
    fake_fname = 'generated=images-{0:0=4d}.png'.format(index)
    vutils.save_image(denorm(fake_images, mean, std), os.path.join(save_dir, fake_fname), nrow=8)
    print("Saving", fake_fname)


# create directory to save images
os.makedirs(save_dir, exist_ok=True)

# Training
for epoch in range(epochs):
    with tqdm(dataloader, unit="batch") as tepoch:
        for i, data in enumerate(tepoch):
            tepoch.set_description(f"Epoch--[ {epoch}/{epochs}]")
            image, label = to_device(data[0], device), to_device(data[1], device)

            # First train discriminator
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # zero gradient of optimizer in every epoch
            optimD.zero_grad()
            # feed the batch of real image into the discriminator
            disc_output, aux_output = netD(image)
            disc_error_real = disc_criterion(disc_output, torch.ones_like(disc_output))
            aux_error_real = aux_criterion(aux_output, label)

            total_error_real = disc_error_real + aux_error_real
            total_error_real.backward()
            optimD.step()
            D_x = disc_output.data.mean()

            # get the current classification accuracy
            accuracy = compute_acc(aux_output, label)

            # generating noise by random sampling
            noise = torch.normal(0, 1, (batch_size, noise_dim), dtype=torch.float).to(device)
            # generating label for entire batch
            fake_label = torch.randint(0, 10, (batch_size,), dtype=torch.long).to(
                device)  # num of classes in CIFAR10 is 10

            fake_image = netG(noise)  # generator generate fake image

            # passing fake image to the discriminator
            disc_output_fake, aux_output_fake = netD(fake_image.detach())  # we will be using this tensor later on
            disc_error_fake = disc_criterion(disc_output_fake, torch.zeros_like(
                disc_output_fake))  # Train discriminator that it is fake image
            aux_error_fake = aux_criterion(aux_output_fake, fake_label)
            total_error_fake = disc_error_fake + aux_error_fake
            total_error_fake.backward()
            optimD.step()

            # Now we train the generator as we have finished updating weights of the discriminator
            netG.zero_grad()
            disc_output_fake, aux_output_fake = netD(fake_image)
            disc_error_fake = disc_criterion(disc_output_fake, torch.ones_like(
                disc_output_fake))  # Fool the discriminator that it is real
            aux_error_fake = aux_criterion(aux_output_fake, fake_label)
            total_error_gen = disc_error_fake + aux_error_fake
            total_error_gen.backward()
            optimG.step()

            tepoch.set_postfix(Loss_Discriminator =total_error_fake.item(), Loss_Generator=total_error_gen.item(), Accuracy=accuracy)
            # if i % 100 == 0:
            #     print(
            #         "Epoch--[{} / {}], Loss_Discriminator--[{}], Loss_Generator--[{}],Accuracy--[{}]".format(epoch,
            #                                                                                                  epochs,
            #                                                                                                  total_error_fake,
            #                                                                                                  total_error_gen,
            #                                                                                                  accuracy))

    # save generated samples at each epoch
    save_samples(epoch, eval_noise)
