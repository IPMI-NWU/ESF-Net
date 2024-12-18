import torch
from torch import nn


class conv_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_layer,self).__init__()

        self.convlayer = nn.Sequential(
         nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
         nn.BatchNorm2d(out_ch),
         nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.convlayer(x)
        return x


class BiFPNUnit(nn.Module):
    def __init__(self, n=1, channels=128):
        super(BiFPNUnit, self).__init__()

        self.conv21 = conv_layer(8 * channels, 2 * channels)
        self.conv31 = conv_layer(24 * channels, 4 * channels)

        self.conv12 = conv_layer(4 * channels, 1 * channels)
        self.conv22 = conv_layer(7 * channels, 2 * channels)
        self.conv32 = conv_layer(14 * channels, 4 * channels)
        self.conv42 = conv_layer(20 * channels, 8 * channels)

        self.upsample12 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample21 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample31 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.downsample22 = nn.MaxPool2d(kernel_size=2)
        self.downsample32 = nn.MaxPool2d(kernel_size=2)
        self.downsample42 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x1, x2, x3, x4):
        x31 = torch.cat([x3, self.upsample31(x4)], dim=1)
        x31 = self.conv31(x31)

        x21 = torch.cat([x2, self.upsample21(x31)], dim=1)
        x21 = self.conv21(x21)

        x12 = torch.cat([x1, self.upsample12(x21)], dim=1)
        x12 = self.conv12(x12)

        x22 = torch.cat([x2, x21, self.downsample22(x12)], dim=1)
        x22 = self.conv22(x22)

        x32 = torch.cat([x3, x31, self.downsample32(x22)], dim=1)
        x32 = self.conv32(x32)

        x42 = torch.cat([x4, self.downsample42(x32)], dim=1)
        x42 = self.conv42(x42)
        
        return x12, x22, x32, x42

