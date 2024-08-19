import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2, 0)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2, 0)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 6)
        )
        self.classifier = nn.LogSoftmax(dim=1)

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.fc(y)
        return self.classifier(y)


if __name__ == "__main__":
    x = torch.randn((64, 1, 28, 28)).cuda()
    model = LeNet().cuda()
    y = model(x)
    print(y.shape)