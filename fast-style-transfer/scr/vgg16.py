import torch
from torchvision import models
from collections import namedtuple

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1_2 = self.slice1(X)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        layers = namedtuple("layers", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = layers(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out