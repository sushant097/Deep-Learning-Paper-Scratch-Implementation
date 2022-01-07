import torch
import numpy as np

lr = 0.0002
epochs = 20
batch_size=100
# narmalization constraints
mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
data_dir = "data"
save_dir = "generated_images"

#  some GAN configuration
noise_dim = 110
# real_label = torch.FloatTensor(batch_size).cuda()
# real_label.fill_(1)
#
# fake_label = torch.FloatTensor(batch_size).cuda()
# fake_label.fill_(0)
#
#
