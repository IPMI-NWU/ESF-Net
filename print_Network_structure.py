import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block 初始化卷积块
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),  # 使用输入边界的反射填充输入张量。填充长度为channels。
            nn.Conv2d(channels, out_features, 7),  # 7为 kernal_size 卷积核大小
            nn.InstanceNorm2d(out_features),  # 归一化
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer  nn.Tanh()激活函数
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


input_shape = (3, 256, 256)
G_AB = GeneratorResNet(input_shape, 9)

print('model_summary', summary(G_AB, input_size=(3, 256, 256), batch_size=10, device='cpu'))

