import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=True),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=True))
        self.sigmoid = nn.PReLU()
        # nn.PReLU()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)
        self.sigmoid = nn.PReLU()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam(nn.Module):
    def __init__(self, planes, ratio):
        super(cbam, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        # ca_map = self.ca(x)
        x = self.ca(x) * x  # 广播机制
        # sa_map = self.sa(x)
        x = self.sa(x) * x  # 广播机制
        return x

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
    
def conv_dw(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class MobileNetV1(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, dropout_keep_prob, embedding_size, pretrained):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 32, 1),    # 3
            conv_dw(32, 64, 1),   # 7

            conv_dw(64, 128, 2),  # 11
            conv_dw(128, 128, 1),  # 19

            conv_dw(128, 256, 2),  # 27
            conv_dw(256, 256, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(256, 512, 2),  # 43 + 16 = 59
            conv_dw(512, 512, 1), # 59 + 32 = 91
            conv_dw(512, 512, 1), # 91 + 32 = 123
            conv_dw(512, 512, 1), # 123 + 32 = 155
            conv_dw(512, 512, 1), # 155 + 32 = 187
            conv_dw(512, 512, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(512, 1024, 2), # 219 +3 2 = 241
            conv_dw(1024, 1024, 1), # 241 + 64 = 301
        )
        self.cbam = cbam(1024, 16)
        self.sep        = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.sep_bn     = nn.BatchNorm2d(512)
        self.prelu      = nn.PReLU(512)

        self.bn2        = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout    = nn.Dropout(p=dropout_keep_prob, inplace=True)
        self.linear     = nn.Linear(512 * self.fc_scale, embedding_size)
        self.features   = nn.BatchNorm1d(embedding_size, eps=1e-05)
        
        if pretrained:
            self.load_state_dict(torch.load("model_data/mobilenet_v1_backbone_weights.pth"), strict = False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, 0, 0.1)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.cbam(x)
        x = self.sep(x)
        x = self.sep_bn(x)
        x = self.prelu(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.features(x)
        return x

def get_mobilenet_cbam(dropout_keep_prob, embedding_size, pretrained):
    return MobileNetV1(dropout_keep_prob, embedding_size, pretrained)

if __name__ == "__main__":
    model = get_mobilenet_cbam(dropout_keep_prob=0.5,embedding_size=512,pretrained=False)

    x = torch.zeros([2,3,112,112])
    out = model(x)
    print("test")
