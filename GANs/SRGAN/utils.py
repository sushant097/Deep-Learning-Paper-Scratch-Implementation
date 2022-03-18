import torch
import torch.nn as nn
from torchvision.models import vgg19
import config


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


def gradient_penalty(critic, real_img, fake_img, device="cpu"):
    batch_size, C, H, W = real_img.shape
    epsilon = torch.rand(batch_size, 1, 1, 1)  # For all samples, taken from uniform distribution [0,1]
    epislon = epsilon.repeat(1, C, H, W).to(device) # repeat fo all C, H, W
    # compute interpolation
    interpolated_images = real_img * epislon + fake_img * (1 - epislon)
    # print(f"FakeImage: {fake_img.shape}, e:{epislon.shape}")
    print(f"RealImage: {real_img.shape}")
    # calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Get gradient w.r.t. interpolated images
    gradients = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), # set the labels as 1 i.e. fool the critic
        create_graph=True,
        retain_graph=True,
    )[0]

    # flatten
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


