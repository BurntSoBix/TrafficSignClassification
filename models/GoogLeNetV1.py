import torch
import torch.nn as nn


class InceptionV1(nn.Module):
    def __init__(self, in_chnl, pth1_chnl, pth2_in_chnl, pth2_out_chnl, pth3_in_chnl, pth3_out_chnl, pth4_chnl):
        super(InceptionV1, self).__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(in_chnl, pth1_chnl, 1, 1, 0),
            nn.ReLU(True)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_chnl, pth2_in_chnl, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(pth2_in_chnl, pth2_out_chnl, 3, 1, 1),
            nn.ReLU(True)
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(in_chnl, pth3_in_chnl, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(pth3_in_chnl, pth3_out_chnl, 5, 1, 2),
            nn.ReLU(True)
        )
        self.path4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_chnl, pth4_chnl, 1, 1, 0),
            nn.ReLU(True)
        )

    def forward(self, x):
        y1 = self.path1(x)
        y2 = self.path2(x)
        y3 = self.path3(x)
        y4 = self.path4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class AuxBlock(nn.Module):
    def __init__(self, in_chnl):
        super(AuxBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.AvgPool2d(5, 3, 0),
            nn.Conv2d(in_chnl, 128, 1, 1, 0),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 6)
        )

    def forward(self, x):
        y = self.block1(x)
        y = self.fc(y)
        return y


class GoogLeNetV1(nn.Module):
    def __init__(self):
        super(GoogLeNetV1, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block3 = nn.Sequential(
            InceptionV1(192, 64, 96, 128, 16, 32, 32),
            InceptionV1(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block4_part1 = nn.Sequential(
            InceptionV1(480, 192, 96, 208, 16, 48, 64)
        )
        self.aux_block1 = AuxBlock(512)
        self.block4_part2 = nn.Sequential(
            InceptionV1(512, 160, 112, 224, 24, 64, 64),
            InceptionV1(512, 128, 128, 256, 24, 64, 64),
            InceptionV1(512, 112, 144, 288, 32, 64, 64)
        )
        self.aux_block2 = AuxBlock(528)
        self.block4_part3 = nn.Sequential(
            InceptionV1(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block5 = nn.Sequential(
            InceptionV1(832, 256, 160, 320, 32, 128, 128),
            InceptionV1(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(7, 1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, 6)
        )
        self.claasifier = nn.LogSoftmax(dim=1)

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4_part1(y)
        aux1 = self.aux_block1(y)
        y = self.block4_part2(y)
        aux2 = self.aux_block2(y)
        y = self.block4_part3(y)
        y = self.block5(y)
        y = self.fc(y)
        return self.claasifier(y), self.claasifier(aux1), self.claasifier(aux2)


if __name__ == "__main__":
    x = torch.randn((64, 3, 224, 224)).cuda()
    model = GoogLeNetV1().cuda()
    y, aux1, aux2 = model(x)
    print(y.shape, aux1.shape, aux2.shape)