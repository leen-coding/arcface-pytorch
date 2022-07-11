import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from nets.ConvNext.convNext import convnext_tiny
from nets.ConvNext.convNext_cbam import convnext_tiny_cbam

from nets.mobilefacenet import get_mbf
from nets.mobilenet import get_mobilenet
from nets.mobilefacenet_cbam import get_mbf_cbam
from nets.mobilefacenet_cbam_modify import get_mbf_cbam_modify


class Arcface_Head(Module):
    def __init__(self, embedding_size=128, num_classes=10575, s=64., m=0.5):
        super(Arcface_Head, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine  = F.linear(input, F.normalize(self.weight))
        sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)

        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output  *= self.s
        return output

class Arcface(nn.Module):
    def __init__(self, num_classes=None, backbone="mobilefacenet", pretrained=False, mode="train"):
        super(Arcface, self).__init__()
        if backbone=="mobilefacenet":
            embedding_size  = 128
            s               = 32
            self.arcface    = get_mbf(embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "convNext":
            embedding_size = 512
            s = 64
            self.arcface = convnext_tiny(embedding_size=embedding_size)

        elif backbone == "convNext_cbam":
            embedding_size = 512
            s = 64
            self.arcface = convnext_tiny_cbam(embedding_size=embedding_size)

        elif backbone=="mobilefacenet_cbam":
            embedding_size  = 128
            s               = 32
            self.arcface    = get_mbf_cbam(embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="mobilefacenet_cbam_modify":
            embedding_size  = 128
            s               = 32
            self.arcface    = get_mbf_cbam_modify(embedding_size=embedding_size, pretrained=pretrained)


        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilefacenet, mobilenetv1.'.format(backbone))

        self.mode = mode

        if mode == "train":
            self.head = Arcface_Head(embedding_size=embedding_size, num_classes=num_classes, s=s)

    def forward(self, x, y = None, mode = "predict"):
        x = self.arcface(x)

        x = x.view(x.size()[0], -1)
        x = F.normalize(x)

        if mode == "predict":
            return x
        else:
            x = self.head(x, y)
            return x

