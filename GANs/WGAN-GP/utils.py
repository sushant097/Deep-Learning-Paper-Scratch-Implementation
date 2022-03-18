import torch


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

