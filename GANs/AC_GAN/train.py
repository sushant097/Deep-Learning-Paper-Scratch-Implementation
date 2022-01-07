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
from utils import custom_init, compute_acc, to_device, get_default_device, denorm
from config import *

dataset = CIFAR10(
    root=data_dir, download=True,
    transform=transforms.Compose([
        transforms.Scale((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    ])
)

dataloader = DataLoader(dataset, batch_size=batch_size)
device = get_default_device()  # check gpu if available else cpu
# instantiate generator
gen = Generator()  # hidden latent vector length
gen = to_device(gen, device)  # move generator to that device
# instantiate discriminator
disc = Discriminator(in_channels=3)
disc = to_device(disc, device)


# apply initialization
# disc.apply(weights_init)
# gen.apply(weights_init)

# defining Optimizer
optimD = optim.Adam(disc.parameters(), lr)
optimG = optim.Adam(gen.parameters(), lr)

# defining Generator
source_obj = nn.BCELoss()  # source_loss
class_obj = nn.NLLLoss()  # class_loss

eval_noise = torch.FloatTensor(batch_size, 110, 1, 1).normal_(0, 1)
eval_noise_ = np.random.normal(0, 1, (batch_size, 110))
eval_label = np.random.randint(0, 10, batch_size)
eval_onehot = np.zeros((batch_size, 10))
eval_onehot[np.arange(batch_size), eval_label] = 1
eval_noise_[np.arange(batch_size), :10] = eval_onehot[np.arange(batch_size)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(batch_size, 110, 1, 1))
eval_noise=eval_noise.to(device)


def train_network(image, label):
    # First train discriminator
    # feed the batch of real image into the discriminator
    source_real, class_label = disc(image)
    source_loss_real = source_obj(source_real, torch.ones_like(source_real))
    class_loss_real = class_obj(class_label, label)

    total_loss_real = source_loss_real + class_loss_real
    total_loss_real.backward()
    optimD.step()  # note in every epoch, make sure optimD zero grad
    # get the current classification accuracy
    accuracy = compute_acc(class_label, label)


    "Now we train the generator as we have finished updating weights of the discriminator"
    # generating noise by random sampling
    noise = torch.randn((batch_size, 110), dtype=torch.float).to(device)
    # generating label for entire batch
    fake_label = torch.randint(0, 10, (batch_size,), dtype=torch.long).to(device)  # num of classes in CIFAR10 is 10
    fake_image = gen(noise) # generator generate fake image

    # passing fake image to the discriminator
    source_fake, class_fake = disc(fake_image.detach())  # we will be using this tensor later on
    source_fake_loss = source_obj(source_fake, torch.zeros_like(source_fake)) # for fake image, we use label -- 0
    class_fake_loss = class_obj(class_fake, fake_label)
    total_fake_loss = source_fake_loss + class_fake_loss
    total_fake_loss.backward()
    optimD.step()

    return total_loss_real, accuracy, total_fake_loss




def save_samples(index, latent_tensors):
  fake_images = gen(latent_tensors)
  fake_fname = 'generated=images-{0:0=4d}.png'.format(index)
  vutils.save_image(denorm(fake_images, mean, std), os.path.join(save_dir, fake_fname), nrow=8)
  print("Saving", fake_fname)

# create directory to save images
os.makedirs(save_dir, exist_ok=True)

# Training
for epoch in range(epochs):
    with tqdm(dataloader, unit="batch") as tepoch:
        for i, data in enumerate(tepoch):
            tepoch.set_description(f"Epoch--[{} / {}] {epoch}/{epochs}")
            # zero gradient of optimizer in every epoch
            optimD.zero_grad()
            image, label = to_device(data[0], device), to_device(data[1], device)
            loss_real, acc_real, fake_loss = train_network(image, label)

            # tepoch.set_postfix(Loss_Discriminator =loss_real.item(), Loss_Generator=fake_loss.item(), Accuracy=acc_real)
            if i % 100 == 0:
                print(
                    "Epoch--[{} / {}], Loss_Discriminator--[{}], Loss_Generator--[{}],Accuracy--[{}]".format(epoch, epochs,
                                                                                                         loss_real,
                                                                                                         fake_loss,
                                                                                                         acc_real))
    # save generated samples at each epoch
    save_samples(epoch, eval_noise)













