import torch
import torch.nn as nn
from torch.nn import functional as F


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel=3):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.name = "ResNetBasicBlock-%d-%d-%d-%d-" % (in_channels, out_channels, stride, kernel)

    def __repr__(self):
        return self.name

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel):
        super(ResNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride[0], padding=kernel//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride[1], padding=kernel//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.name = "ResNetDownBlock-%d-%d-%d-%d-%d-" % (in_channels, out_channels, stride[0], stride[1], kernel)

    def __repr__(self):
        return self.name

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class ResNet18(nn.Module):
    def __init__(self, kernel_list):
        super(ResNet18, self).__init__()
        # 3,224,224
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 64,112,112
        self.layer1 = nn.Sequential(ResNetBasicBlock(64, 64, 1, kernel=kernel_list[0]),
                                    ResNetBasicBlock(64, 64, 1, kernel=kernel_list[1]))
        # 64,112,112
        self.layer2 = nn.Sequential(ResNetDownBlock(64, 128, [2, 1], kernel=kernel_list[2]),
                                    ResNetBasicBlock(128, 128, 1, kernel=kernel_list[3])) # 128,56,56
        self.layer3 = nn.Sequential(ResNetDownBlock(128, 256, [2, 1], kernel=kernel_list[4]),
                                    ResNetBasicBlock(256, 256, 1, kernel=kernel_list[5])) # 256,28,28
        self.layer4 = nn.Sequential(ResNetDownBlock(256, 512, [2, 1], kernel=kernel_list[6]),
                                    ResNetBasicBlock(512, 512, 1, kernel=kernel_list[7])) # 512,14,14

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=[1, 1])# 512,14,14

        self.fc = nn.Linear(512, 10) # (512)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

    def __repr__(self):
        return "ResNet18"


if __name__ == "__main__":
    kernel_list = [3, 3, 3, 5, 7, 7, 3, 5]
    model = ResNet18(kernel_list)
    for m in model.modules():
        m.register_forward_hook(lambda m, f_in, f_out: print(m.__repr__(), list(f_in[0].size())))
    x = torch.ones(1,3,224,224)
    _ = model(x)
