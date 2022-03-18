import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from utils import load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader
from dataset import MyImageFolder
import tqdm

from srgan import Generator, Discriminator
from utils import VGGLoss
import config

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = MyImageFolder(root_dir="new_data/")
loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=config.NUM_WORKERS,
)
gen = Generator(in_channels=3).to(config.DEVICE)
disc = Discriminator(in_channels=3).to(config.DEVICE)
opt_gen = torch.optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
opt_disc = torch.optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
vgg_loss = VGGLoss()

if config.LOAD_MODEL:
    load_checkpoint(
        config.CHECKPOINT_GEN,
        gen,
        opt_gen,
        config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
    )


for epoch in range(config.NUM_EPOCHS):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        # -----------------------------------------------
        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # --------------------------------------------------
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # -------------------------------------------------------
        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # -------------------------------------------------------
        disc_fake = disc(fake)
        # l2_loss = mse(fake, high_res)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()



    if config.SAVE_MODEL:
        save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

