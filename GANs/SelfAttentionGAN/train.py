import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sgan_model import Generator, Discriminator
import datetime
import time
import os

# Hyperparameters
batch_size = 64
steps = 100000
z_dim = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

# Define data transformer
img_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

# Read data and transform
dataset = MNIST(root='./data', download=True, train=True, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# fixed random latent for generator
latent_z = torch.randn(64, 100).to(device)

def train(steps = 100000, batch_size = 64, z_dim = 100, attn = True):

    # Initialize model
    G = Generator(batch_size, attn).to(device)
    D = Discriminator(batch_size, attn).to(device)

    # make directory for samples and models
    cwd = os.getcwd()
    post = '_attn' if attn else ''
    if not os.path.exists(cwd+'/samples_mnist'+post):
        os.makedirs(cwd+'/samples_mnist'+post)

    # Initialize optimizer with filter, lr and coefficients
    g_optimizer = torch.optim.Adam(filter(lambda p: p.requores_grad, G.parameters()), 0.001, [0.0, 0.9])
    d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D.paramters()), 0.004, [0.0, 0.9])

    # load data
    Iter  = iter(dataloader)
    
    # time start : training
    start_time = time.time()
    
    for step in range(steps):
        # =============== Train Discriminator =========== #
        D.train()
        G.train()
        try:
            real_images, _ = next(Iter)
        except:
            Iter = iter(dataloader)
            real_images, _ = next(Iter)
            
        # Compute loss with real images
        d_out_real = D(real_images.to(device))
        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        
        # Compute loss with fake images
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = G(z)
        d_out_fake = D(fake_images)
        d_loss_real = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        
        # Backward + Optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # =========Train G =========== #
        # Create random noise
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = G(z)
        g_out_fake = D(fake_images)
        g_loss_fake = - g_out_fake.mean()
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss_fake.backward()
        g_optimizer.step()
        
        # Print out log info
        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            expect = elapsed/(step + 1)*(steps-step-1)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            expect = str(datetime.timedelta(seconds=expect))
            clear_output(wait=True)
            print("Elapsed [{}], Expect [{}], step [{}/{}], D_real_loss: {:.4f}, "
                  " ave_generator_gamma: {:.4f}".
                  format(elapsed,expect,step + 1,steps,d_loss_real.item(),G.attn.gamma.mean().item()))
            
            
        
        # Sample images
        if (step + 1) % (100) == 0:
            fake_images= G(fixed_z)
            save_image(denorm(fake_images), os.path.join('./samples_mnist'+post, '{}_fake.png'.format(step + 1)))

#train(steps = 60000, attn = True)
#print('Done training part 1')
train(steps = 60000, attn = False)
print('Done training part 2')

# Generate gif files
from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype("./demo/arial.ttf", 18)
def create_image_with_text(img, wh, text):
    width, height = wh
    draw = ImageDraw.Draw(img)
    draw.text((width, height), text, font = font, fill="white")
    return img

frames = []

for i in range(100, 20001, 100):
    img = Image.open('samples_mnist/{}_fake.png'.format(str(i)))
    img1 = Image.open('samples_mnist_attn/{}_fake.png'.format(str(i)))
    width, height = img.size
    expand = Image.new(img.mode, (width*2 + 10, height + 40), "black")
    expand.paste(img, (0, 0))
    expand.paste(img1, (width + 10, 0))
    epoch = round(i*64/60000,2)
    new_frame = create_image_with_text(expand,(10,258), "After "+str(epoch)+" epoches")
    new_frame = create_image_with_text(new_frame,(10,238), "Without Attention")
    new_frame = create_image_with_text(new_frame,(width + 20,238), "With Attention")
    frames.append(new_frame)
    
frames[0].save('./demo/comparison_mnist.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=60, loop=0)