import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        self.encoder = nn.ModuleList()
        return x
