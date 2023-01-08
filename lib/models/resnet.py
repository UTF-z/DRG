import numpy
import cv2
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        identity = X
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.downsample is not None:  
            identity = self.downsample(X)

        return self.relu(Y + identity)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)  

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        identity = X

        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))

        if self.downsample is not None:  
            identity = self.downsample(X)

        return self.relu(Y + identity)


class ResNet(nn.Module):

    def __init__(self, residual, num_residuals, num_classes=10, include_top=True, input_channel=1):
        super(ResNet, self).__init__()

        self.out_channel = 64   
        self.include_top = include_top

        self.prev_conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.prev_conv2 = nn.Conv2d(input_channel, input_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.prev_bn1 = nn.BatchNorm2d(input_channel)
        self.prev_bn2 = nn.BatchNorm2d(input_channel)
        self.conv1 = nn.Conv2d(input_channel, self.out_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #[b, 64, 64, 64]

        self.conv2 = self.residual_block(residual, 64, num_residuals[0]) #[b, 64, 64, 64]
        self.conv3 = self.residual_block(residual, 128, num_residuals[1], stride=2) #[b, 128, 32, 32]
        self.conv4 = self.residual_block(residual, 256, num_residuals[2], stride=2) #[b, 256, 16, 16]
        self.conv5 = self.residual_block(residual, 512, num_residuals[3], stride=2) #[b, 512, 8, 8]
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output_size = (1, 1)
            self.fc = nn.Linear(512 * residual.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def residual_block(self, residual, channel, num_residuals, stride=1):
        downsample = None

        if stride != 1 or self.out_channel != channel * residual.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channel, channel * residual.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * residual.expansion))

        block = []  
        block.append(residual(self.out_channel, channel, downsample=downsample, stride=stride))
        self.out_channel = channel * residual.expansion  

        for _ in range(1, num_residuals):
            block.append(residual(self.out_channel, channel))

        return nn.Sequential(*block)

    def forward(self, X):
        Y = self.relu(self.prev_bn1(self.prev_conv1(X)))
        Y = self.relu(self.prev_bn2(self.prev_conv2(Y)))
        Y = self.relu(self.bn1(self.conv1(Y)))
        Y = self.maxpool(Y)
        Y = self.conv5(self.conv4(self.conv3(self.conv2(Y))))

        if self.include_top:
            Y = self.avgpool(Y)
            Y = torch.flatten(Y, 1)
            Y = self.fc(Y)

        return Y


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
