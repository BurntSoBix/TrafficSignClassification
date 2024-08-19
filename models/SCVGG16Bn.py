import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1, bias=False)
        self.silu = nn.SiLU(True)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.silu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.silu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

    
class CbamBlock(nn.Module):
    def __init__(self, in_planes):
        super(CbamBlock, self).__init__()
        self.channelattention = ChannelAttention(in_planes)
        self.spatialattention = SpatialAttention()

    def forward(self, X):
        X = X * self.channelattention(X)
        X = X * self.spatialattention(X)
        return X


class SkipContactBlock(nn.Module):
    def __init__(self, in_chnl, out_chnl):
        super(SkipContactBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_chnl, out_chnl, 3, 1, 1),
            nn.BatchNorm2d(out_chnl),
            nn.SiLU(True),
            nn.Conv2d(out_chnl, out_chnl, 3, 1, 1),
            nn.BatchNorm2d(out_chnl),
            nn.SiLU(True),
            nn.Conv2d(out_chnl, out_chnl, 3, 1, 1),
            nn.BatchNorm2d(out_chnl),
            nn.SiLU(True)
        )
        self.block2 = nn.Sequential(
            CbamBlock(in_chnl // 2),
            nn.Conv2d(in_chnl // 2, in_chnl, 4, 2, 1),
            nn.BatchNorm2d(in_chnl),
            nn.SiLU(True),
            nn.Conv2d(in_chnl, in_chnl, 4, 2, 1),
            nn.BatchNorm2d(in_chnl),
            nn.SiLU(True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_chnl + out_chnl, out_chnl, 1, 1, 0),
            nn.BatchNorm2d(out_chnl),
            nn.SiLU(True),
            nn.MaxPool2d(2, 2, 0)
        )

    def forward(self, x1, x2):
        y2 = self.block1(x2)
        y1 = self.block2(x1)
        y = self.block3(torch.cat([y1, y2], dim=1))
        return y


class SCVGG16(nn.Module):
    def __init__(self):
        super(SCVGG16, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2, 0)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(True),
        )
        self.block3 = SkipContactBlock(128, 256)
        self.block4 = SkipContactBlock(256, 512)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.SiLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.SiLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.SiLU(True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.SiLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.SiLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 6)
        )
        self.classifier = nn.LogSoftmax(dim=1)

    def forward(self, x):
        y1 = self.block1(x)
        y = self.maxpool(y1)
        y2 = self.block2(y)
        y = self.maxpool(y2)
        y = self.block3(y1, y)
        y = self.block4(y2, y)
        y = self.block5(y)
        y = self.maxpool(y)
        y = self.fc(y)
        return self.classifier(y)