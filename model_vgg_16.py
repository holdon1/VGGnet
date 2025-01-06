import torch.nn as nn

import torch


class VggNet16(nn.Module):
    def __init__(self, num_classes):
        super(VggNet16, self).__init__()
        self.feature = nn.Sequential(
            # block1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # block2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # block3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # block4
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            # block5
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            # fc1
            nn.Linear(512*7*7,4096),nn.ReLU(inplace=True),
            # fc2
            nn.Linear(4096,4096),nn.ReLU(inplace=True),
            # fc3
            nn.Linear(4096,num_classes)

        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.size())  # 打印 x 的形状，检查展平之前的大小
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(x.size())  # 打印 x 的形状，检查展平之后的大小
        x = self.classifier(x)
        return x