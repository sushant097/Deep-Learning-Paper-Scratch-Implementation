import torch
from torchvision.utils import save_image
from torchvision import datasets
from torchvision.transforms import transforms
import os

from model import Generator, Discriminator, init_weights, denorm_image
from config import *
from utils import gradient_penalty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a directory if not exists
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

transforms = transforms.Compose([
    transforms.Resize(Image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        data_dir,
        download=True,
        train=True,
        transform=transforms
    ),
    batch_size=batch_size,
    shuffle=True
)

# Step2: Define Generator and discriminator
gen = Generator(hidden_dim, img_channels).to(device)
critic = Discriminator(img_channels).to(device)
# intialize weights
init_weights(gen)
init_weights(critic)

# WGAN with gradient clipping uses RMSprop instead of ADAM
optimCritic = torch.optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))
optimGen = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.9))


# ------------
#  Training
# ------------

# set the training mode
gen.train()
critic.train()
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # to device
        real_imgs = imgs.to(device)

        # -------------------
        # Train Discriminator : Max -E[critic(real)] + E[critic(fake)]
        # --------------------
        for _ in range(critic_iter):

            optimCritic.zero_grad()

            # sample noise as generator input
            z = torch.randn((batch_size, hidden_dim, 1, 1), dtype=torch.float, device=device)

            # Generate a batch of fake images
            fake_imgs = gen(z)

            # compute gradient penalty
            gp = gradient_penalty(critic, real_imgs, fake_imgs, device=device)
            # Adversarial loss - Critic
            critic_loss = (-torch.mean(critic(real_imgs)) + torch.mean(critic(fake_imgs))
                           + lambda_GP * gp) # adding gradient penalty to loss

            critic_loss.backward(retain_graph=True)
            optimCritic.step()

        # --------------
        # Train Generator
        # ---------------

        # Train the generator Min - E[critic(fake)] ~ Max E[critic(fake)]

        optimGen.zero_grad() # zero the gradient

        # Adversarial Loss
        loss_G = -torch.mean(critic(fake_imgs))

        loss_G.backward()
        optimGen.step()

        print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}'
              .format(epoch, epochs, critic_loss.item(), loss_G.item()))

    save_image(denorm_image(fake_imgs), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

# Save the model checkpoints
torch.save(gen.state_dict(), 'G.ckpt')
torch.save(critic.state_dict(), 'D.ckpt')
