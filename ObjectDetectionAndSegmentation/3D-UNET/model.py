import torch

import torch.nn as nn

from model_utils import EncoderBlock, DecoderBlock


class UnetModel(nn.Module):
    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="sigmoid") -> None:
        super().__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)
            
    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.sigmoid(x)
        print("Final output shape: ", x.shape)
        return x
    
    