import torch.nn as nn
import torch
from torch.nn import functional


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64, ifbranch = False, ifbranchnext = False):
        super(Bottleneck, self).__init__()
        self.ifbranch = ifbranch
        self.ifbranchnext = ifbranchnext
        padding = 1

        if self.ifbranch:
            kernelsize = (3, 10)

        elif self.ifbranchnext:
            kernelsize = (1, 1)
            padding = 1

        else:
            kernelsize = (3, 3)
        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=kernelsize, stride=stride, bias=False, padding=padding)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if not self.ifbranch and not self.ifbranchnext:
            out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.branch1 = Bottleneck
        self.branch2 = Bottleneck
        self.include_top = include_top
        self.in_channel = 64
        self.branch1_in_channel = 1024
        self.branch2_in_channel = 1024
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4g = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.layer41 = self._make_branchlayers1(self.branch1, 512, blocks_num[3], stride=2)
        self.layer42 = self._make_branchlayers2(self.branch2, 512, blocks_num[3], stride=2)


        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.fcb = nn.Linear(512 * block.expansion, int(num_classes/2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):


        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def _make_branchlayers1(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.branch1_in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.branch1_in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.branch1_in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group, ifbranch=True))
        self.branch1_in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.branch1_in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,ifbranchnext=True))
        return nn.Sequential(*layers)

    def _make_branchlayers2(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.branch2_in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.branch2_in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.branch2_in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group, ifbranch=True))
        self.branch2_in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.branch2_in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #224*224 -> 112*112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #112*112 -> 56*56
        x = self.layer1(x)
        x = self.layer2(x)
        # 56*56 -> 28*28
        x = self.layer3(x)
        # 28*28 -> 14*14
        x1, x2 = torch.split(x, 7, dim=2)
        xg = self.layer4g(x)
        x1 = self.layer41(x1)
        x2 = self.layer42(x2)
        # 14*14 -> 7*7
        if self.include_top:
            xg = self.avgpool(xg)
            x1 = self.avgpool(x1)
            x2 = self.avgpool(x2)
            #7*7->1*1
            xg = torch.flatten(xg, 1)
            x1 = torch.flatten(x1, 1)
            x2 = torch.flatten(x2, 1)
            xg = self.fc(xg)
            x1 = self.fcb(x1)
            x2 = self.fcb(x2)
            x_side = torch.cat([x1, x2], 1)
            out = xg + x_side

        return out


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50_2branch(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

if __name__ == "__main__":
    model = resnet50_2branch(num_classes=128, include_top=True)
    x = torch.zeros([2,3,224,224])

    out = model(x)
    print("test")