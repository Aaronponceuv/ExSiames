import torch
import torch.nn as nn
import torch.nn.functional as F

class SiamesNet(nn.Module):
    def __init__(self, embeddingnet):
        super(SiamesNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        return embedded_x, embedded_y