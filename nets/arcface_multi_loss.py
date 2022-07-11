# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Module, Parameter
# from nets.ConvNext.convNext import convnext_tiny
# from nets.ConvNext.convNext_cbam import convnext_tiny_cbam
# from nets.ConvNext.convNext_cbam_multiloss import convnext_tiny_cbam_multiloss
# from nets.iresnet import (iresnet18, iresnet34, iresnet50, iresnet100,
#                           iresnet200)
# from nets.mobilefacenet import get_mbf
# from nets.mobilenet import get_mobilenet
# from nets.mobilefacenet_cbam import get_mbf_cbam
# from nets.mobilefacenet_cbam_modify import get_mbf_cbam_modify
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
#                                 nn.ReLU(),
#                                 nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
#
# class cbam(Module):
#     def __init__(self, planes):
#         super(cbam, self).__init__()
#         self.ca = ChannelAttention(planes, ratio=16)
#         self.sa = SpatialAttention(kernel_size=7)
#
#     def forward(self, x):
#         ca_map = self.ca(x)
#         x = self.ca(x) * x  # 广播机制
#         sa_map = self.sa(x)
#         x = self.sa(x) * x  # 广播机制
#         return x, ca_map, sa_map
#
#
# class Attention_maps(Module):
#     def __init__(self, dims):
#         super(Attention_maps, self).__init__()
#         self.cbam = cbam(dims)
#         self.outs = []
#         self.ca_maps = []
#         self.sa_maps = []
#
#     def forward(self, x):
#         short_cut = x
#         for i in range(8):
#             out, ca_map, sa_map = self.cbam(x)
#             self.outs.append(out)
#             self.ca_maps.append(ca_map)
#             self.sa_maps.append(sa_map)
#         for ca_map in self.ca_maps:
#
#
#
#
#
#
#
# class Arcface_Head(Module):
#     def __init__(self, embedding_size=128, num_classes=10575, s=64., m=0.5):
#         super(Arcface_Head, self).__init__()
#         self.s = s
#         self.m = m
#         self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
#         nn.init.xavier_uniform_(self.weight)
#
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m
#
#     def forward(self, input, label):
#         cosine  = F.linear(input, F.normalize(self.weight))
#         sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
#         phi     = cosine * self.cos_m - sine * self.sin_m
#         phi     = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)
#
#         one_hot = torch.zeros(cosine.size()).type_as(phi).long()
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output  *= self.s
#         return output
#
# class Arcface(nn.Module):
#     def __init__(self, num_classes=None, backbone="mobilefacenet", pretrained=False, mode="train"):
#         super(Arcface, self).__init__()
#
#         if backbone == "convNext_cbam_multi_loss":
#             embedding_size = 768
#             self.backbone_stem = convnext_tiny_cbam_multiloss()
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilefacenet, mobilenetv1.'.format(backbone))
#
#         self.mode = mode
#
#         if mode == "train":
#             self.head = Arcface_Head(embedding_size=embedding_size, num_classes=num_classes, s=s)
#
#     def forward(self, x, y = None, mode = "predict"):
#         x = self.arcface(x)
#
#         x = x.view(x.size()[0], -1)
#         x = F.normalize(x)
#
#         if mode == "predict":
#             return x
#         else:
#             x = self.head(x, y)
#             return x
#
