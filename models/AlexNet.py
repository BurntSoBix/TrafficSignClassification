import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 256, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 6)
        )
        self.classifier = nn.LogSoftmax(dim=1)

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.fc(y)
        return self.classifier(y)


if __name__ == "__main__":
    x = torch.randn((64, 3, 224, 224)).cuda()
    model = AlexNet().cuda()
    y = model(x)
    print(y.shape)