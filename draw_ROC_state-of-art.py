from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
start_time = time.time()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import warnings, os, copy, time
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from PIL import Image
from typing import ClassVar, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms, utils
import argparse
from data_loader_fusion import *
# from data_loader_fake_fake_fake import *
from torch.utils.data import DataLoader
import math
from Blocks import *
from backbone import *
from bifpn import BiFPNUnit

# Training settings
# 设置一些参数,每个都有默认值,输入python main.py -h可以获得帮助
parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--img_height', type=int, default=100)
parser.add_argument('--img_width', type=int, default=100)
parser.add_argument('--n_residual_blocks', type=int, default=7)
# 跑多少次batch进行一次日志记录
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

torch.cuda.set_device(3)  # 本来是2
# 这个是使用argparse模块时的必备行,将参数进行关联
args = parser.parse_args()
# 这个是在确认是否使用GPU的参数
args.cuda = not args.no_cuda and torch.cuda.is_available()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MixedFusion_Block0(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(MixedFusion_Block0, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(in_dim), act_fn, )
        # self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_dim), act_fn, )

    def forward(self, x1, x2):
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)  # sum
        fusion_mul = torch.mul(x1, x2)

        modal_in1 = torch.reshape(x1, [x1.shape[0], 1, x1.shape[1], x1.shape[2], x1.shape[3]])
        modal_in2 = torch.reshape(x2, [x2.shape[0], 1, x2.shape[1], x2.shape[2], x2.shape[3]])
        modal_cat = torch.cat((modal_in1, modal_in2), dim=1)
        fusion_max = modal_cat.max(dim=1)[0]

        out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)

        out1 = self.layer1(out_fusion)
        out2 = self.layer2(out1)

        return out2


class MixedFusion_Block(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(MixedFusion_Block, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(in_dim), act_fn, )

        # revised in 09/09/2019.
        # self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim,  kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_dim), act_fn, )

    def forward(self, x1, x2, xx):
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)  # sum
        fusion_mul = torch.mul(x1, x2)

        modal_in1 = torch.reshape(x1, [x1.shape[0], 1, x1.shape[1], x1.shape[2], x1.shape[3]])
        modal_in2 = torch.reshape(x2, [x2.shape[0], 1, x2.shape[1], x2.shape[2], x2.shape[3]])
        modal_cat = torch.cat((modal_in1, modal_in2), dim=1)
        fusion_max = modal_cat.max(dim=1)[0]

        out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)

        out1 = self.layer1(out_fusion)
        out2 = self.layer2(torch.cat((out1, xx), dim=1))

        return out2


class MixedFusion_Block_expand(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(MixedFusion_Block_expand, self).__init__()

        # self.layer1 = nn.Sequential(nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, stride=1, padding=1),
        #                             nn.BatchNorm2d(in_dim), act_fn, )
        self.layer1 = nn.Sequential(Conv_residual_conv_Inception_Dilation_asymmetric(in_dim * 3, in_dim, stride=1),
                                    nn.BatchNorm2d(in_dim), act_fn, )

        # revised in 09/09/2019.
        # self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim,  kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_dim), act_fn, )
        # self.layer2 = nn.Sequential(Conv_residual_conv_Inception_Dilation_asymmetric(in_dim * 2, out_dim, stride=1),
        #                             nn.BatchNorm2d(in_dim), act_fn, )

    def forward(self, x1, x2, xx):
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)  # sum
        fusion_mul = torch.mul(x1, x2)

        modal_in1 = torch.reshape(x1, [x1.shape[0], 1, x1.shape[1], x1.shape[2], x1.shape[3]])
        modal_in2 = torch.reshape(x2, [x2.shape[0], 1, x2.shape[1], x2.shape[2], x2.shape[3]])
        modal_cat = torch.cat((modal_in1, modal_in2), dim=1)
        fusion_max = modal_cat.max(dim=1)[0]

        out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)

        out1 = self.layer1(out_fusion)
        out2 = self.layer2(torch.cat((out1, xx), dim=1))

        return out2


class Bottleneck(nn.Module):
    expansion = 4  # expansion是BasicBlock和Bottleneck的核心区别之一

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


############################################# Dense_NetWork ###################################


def croppCenter(tensorToCrop,finalShape):

    org_shape = tensorToCrop.shape

    diff = np.zeros(3)
    diff[0] = org_shape[2] - finalShape[2]
    diff[1] = org_shape[3] - finalShape[3]


    croppBorders = np.zeros(2,dtype=int)
    croppBorders[0] = int(diff[0]/2)
    croppBorders[1] = int(diff[1]/2)

    return tensorToCrop[:,
                        :,
                        croppBorders[0]:org_shape[2]-croppBorders[0],
                        croppBorders[1]:org_shape[3]-croppBorders[1]]


def conv_block(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1 ):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation ),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_block_Asym_Inception(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernel_size,1],   padding=tuple([padding,0]), dilation = (dilation,1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernel_size], padding=tuple([0,padding]), dilation = (1,dilation)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
    )
    return model


class Conv_residual_conv_Inception_Dilation_asymmetric(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(self, in_dim, out_dim, stride=1):
        super(Conv_residual_conv_Inception_Dilation_asymmetric, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = nn.ReLU()

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)

        self.conv_2_1 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=1, stride=1,
                                                  padding=0, dilation=1)
        self.conv_2_2 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=1, dilation=1)
        self.conv_2_3 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=5, stride=1,
                                                  padding=2, dilation=1)
        self.conv_2_4 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=2, dilation=2)
        self.conv_2_5 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=4, dilation=4)

        self.conv_2_output = conv_block(self.out_dim * 5, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0,
                                        dilation=1)

        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        # identity = conv_1
        conv_2_1 = self.conv_2_1(conv_1)
        conv_2_2 = self.conv_2_2(conv_1)
        conv_2_3 = self.conv_2_3(conv_1)
        conv_2_4 = self.conv_2_4(conv_1)
        conv_2_5 = self.conv_2_5(conv_1)

        # print('input', conv_1.shape)
        # print('conv_2_1', conv_2_1.shape)
        # print('conv_2_2', conv_2_2.shape)
        # print('conv_2_3', conv_2_3.shape)
        # print('conv_2_4', conv_2_4.shape)
        # print('conv_2_5', conv_2_5.shape)
        # print('---------------------------------------input', input.shape)
        # print('inplanes', self.in_dim)
        # print('outplanes', self.out_dim)
        # if self.downsample is not None:
        #     identity = self.downsample(input)
        #     conv_2_1 = self.downsample(conv_2_1)
        #     conv_2_2 = self.downsample(conv_2_2)
        #     conv_2_3 = self.downsample(conv_2_3)
        #     conv_2_4 = self.downsample(conv_2_4)
        #     conv_2_5 = self.downsample(conv_2_5)
        #     print('identity', identity.shape)

        out1 = torch.cat([conv_2_1, conv_2_2, conv_2_3, conv_2_4, conv_2_5], 1)
        out1 = self.conv_2_output(out1)

        conv_3 = self.conv_3(out1 + conv_1)
        # conv_3 = self.conv_3(out1 + identity)

        # print('out', conv_3.shape)
        return conv_3


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(self, inplanes: int, planes: int,
                 stride: Optional[int] = 1,
                 downsample: Optional[bool] = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.conv1 = Conv_residual_conv_Inception_Dilation_asymmetric(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = Conv_residual_conv_Inception_Dilation_asymmetric(planes, planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride
        self.in_dim = inplanes
        self.out_dim = planes
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity: torch.Tensor = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        # print('---------------------------------------x', x.shape)
        # print('inplanes', self.in_dim)
        # print('outplanes', self.out_dim)
        '''
        if self.downsample is not None:
            identity = self.downsample(x)
            out = self.pool(out)
        '''

        if self.downsample is not None:
            identity = self.downsample(x)

        # print('identity', identity.shape)
        # print('out', out.shape)
        out += identity
        out = self.relu(out)

        # print('out', out.shape)
        return out


# @: Custom build ResNet18 Model
class ResNet(nn.Module):
    def __init__(self, block: object, layers,
                 num_classes: Optional[int] = 2,
                 zero_init_residual: Optional[bool] = False) -> None:
        super(ResNet, self).__init__()
        self.inplanes: int = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(0.1)

        self.out_dim = 64

        #######################################################################
        # fusion layer
        #######################################################################
        # down 1st layer
        self.down_fu_1 = MixedFusion_Block0(self.out_dim, self.out_dim * 2, nn.LeakyReLU(0.2, inplace=True))
        self.pool_fu_1 = maxpool()

        self.down_fu_2 = MixedFusion_Block_expand(self.out_dim * 2, self.out_dim * 4, nn.LeakyReLU(0.2, inplace=True))
        self.pool_fu_2 = maxpool()

        self.down_fu_3 = MixedFusion_Block(self.out_dim * 4, self.out_dim * 8, nn.LeakyReLU(0.2, inplace=True))
        self.pool_fu_3 = maxpool()

        self.down_fu_4 = MixedFusion_Block_expand(self.out_dim * 8, self.out_dim * 8, nn.LeakyReLU(0.2, inplace=True))

        # down 4th layer
        self.down_fu_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 8, out_channels=self.out_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 8), nn.LeakyReLU(0.2, inplace=True))

        ##########################################################################

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: object, planes: int, blocks: int, stride: Optional[int] = 1) -> nn.Sequential():
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    def forward(self, input0, input1):
    # def forward(self, input):
        # ############################# #
        # ~~~~~~ Encoding path ~~~~~~~  #
        x1 = input0
        x2 = input1
        # x1 = input[:, 0:1, :, :]
        # x2 = input[:, 1:2, :, :]

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        # 提取x1的特征
        x1_1 = self.layer1(x1)  # torch.Size([20, 64, 64, 64])
        x1_2 = self.layer2(x1_1)  # torch.Size([20, 128, 32, 32])
        x1_3 = self.layer3(x1_2)  # torch.Size([20, 256, 16, 16])
        x1_4 = self.layer4(x1_3)  # torch.Size([20, 512, 8, 8])

        # 提取x2的特征
        x2_1 = self.layer1(x2)
        x2_2 = self.layer2(x2_1)
        x2_3 = self.layer3(x2_2)
        x2_4 = self.layer4(x2_3)

        # 特征融合
        # ----------------------------------------
        # fusion layer
        down_fu_1 = self.down_fu_1(x1_1, x2_1)
        down_fu_1m = self.pool_fu_1(down_fu_1)

        down_fu_2 = self.down_fu_2(x1_2, x2_2, down_fu_1m)
        down_fu_2m = self.pool_fu_2(down_fu_2)

        down_fu_3 = self.down_fu_3(x1_3, x2_3, down_fu_2m)
        down_fu_3m = self.pool_fu_3(down_fu_3)

        down_fu_4 = self.down_fu_4(x1_4, x2_4, down_fu_3m)
        down_fu_5 = self.down_fu_5(down_fu_4)

        x_fu = self.avgpool(down_fu_5)
        x_fu = x_fu.view(x_fu.size(0), -1)
        x_fu = self.fc(x_fu)
        x_fu = self.dropout(x_fu)

        return x_fu

##############################HyperDenseNet################################################

def croppCenter(tensorToCrop, finalShape):
    org_shape = tensorToCrop.shape
    diff = org_shape[2] - finalShape[2]
    croppBorders = int(diff / 2)
    return tensorToCrop[:,
           :,
           croppBorders:org_shape[2] - croppBorders,
           croppBorders:org_shape[3] - croppBorders]


def convBlock(nin, nout, kernel_size=3, batchNorm=False, layer=nn.Conv2d, bias=True, dropout_rate=0.0, dilation=1):
    if batchNorm == False:
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias, dilation=dilation)
        )
    else:
        return nn.Sequential(
            nn.BatchNorm2d(nin),
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias, dilation=dilation)
        )


def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        # nn.LeakyReLU(0.2)
        nn.PReLU()
    )


class HyperDenseNet_2Mod(nn.Module):
    def __init__(self, nClasses):
        super(HyperDenseNet_2Mod, self).__init__()

        # Path-Top
        self.conv1_Top = convBlock(1, 25)
        self.conv2_Top = convBlock(50, 25, batchNorm=True)
        self.conv3_Top = convBlock(100, 25, batchNorm=True)
        self.conv4_Top = convBlock(150, 50, batchNorm=True)
        self.conv5_Top = convBlock(250, 50, batchNorm=True)
        self.conv6_Top = convBlock(350, 50, batchNorm=True)
        self.conv7_Top = convBlock(450, 75, batchNorm=True)
        self.conv8_Top = convBlock(600, 75, batchNorm=True)
        self.conv9_Top = convBlock(750, 75, batchNorm=True)

        # Path-Bottom
        self.conv1_Bottom = convBlock(1, 25)
        self.conv2_Bottom = convBlock(50, 25, batchNorm=True)
        self.conv3_Bottom = convBlock(100, 25, batchNorm=True)
        self.conv4_Bottom = convBlock(150, 50, batchNorm=True)
        self.conv5_Bottom = convBlock(250, 50, batchNorm=True)
        self.conv6_Bottom = convBlock(350, 50, batchNorm=True)
        self.conv7_Bottom = convBlock(450, 75, batchNorm=True)
        self.conv8_Bottom = convBlock(600, 75, batchNorm=True)
        self.conv9_Bottom = convBlock(750, 75, batchNorm=True)

        # self.fully_1 = nn.Conv2d(1800, 400, kernel_size=1)  # original
        self.fully_1 = nn.Conv2d(700, 400, kernel_size=1)  # 5
        # self.fully_1 = nn.Conv2d(1200, 400, kernel_size=1)
        self.fully_2 = nn.Conv2d(400, 200, kernel_size=1)
        self.fully_3 = nn.Conv2d(200, 150, kernel_size=1)
        self.final = nn.Conv2d(150, nClasses, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(150, 2)
        self.dropout = nn.Dropout(0.1)

    # def forward(self, input):
    def forward(self, input1, input2):
        # ----- First layer ------ #
        # get 2 of the channels as 5D tensors
        # pdb.set_trace()
        # y1t = self.conv1_Top(input[:, 0:1, :, :])
        # y1b = self.conv1_Bottom(input[:, 1:2, :, :])

        y1t = self.conv1_Top(input1)
        y1b = self.conv1_Bottom(input2)

        # ----- Second layer ------ #
        # concatenate
        y2t_i = torch.cat((y1t, y1b), dim=1)
        y2b_i = torch.cat((y1b, y1t), dim=1)

        y2t_o = self.conv2_Top(y2t_i)
        y2b_o = self.conv2_Bottom(y2b_i)

        # ----- Third layer ------ #
        y2t_i_cropped = croppCenter(y2t_i, y2t_o.shape)
        y2b_i_cropped = croppCenter(y2b_i, y2t_o.shape)

        # concatenate
        y3t_i = torch.cat((y2t_i_cropped, y2t_o, y2b_o), dim=1)
        y3b_i = torch.cat((y2b_i_cropped, y2b_o, y2t_o), dim=1)

        y3t_o = self.conv3_Top(y3t_i)
        y3b_o = self.conv3_Bottom(y3b_i)

        # ------ Fourth layer ----- #
        y3t_i_cropped = croppCenter(y3t_i, y3t_o.shape)
        y3b_i_cropped = croppCenter(y3b_i, y3t_o.shape)

        # concatenate
        y4t_i = torch.cat((y3t_i_cropped, y3t_o, y3b_o), dim=1)
        y4b_i = torch.cat((y3b_i_cropped, y3b_o, y3t_o), dim=1)

        y4t_o = self.conv4_Top(y4t_i)
        y4b_o = self.conv4_Bottom(y4b_i)

        # ------ Fifth layer ----- #
        y4t_i_cropped = croppCenter(y4t_i, y4t_o.shape)
        y4b_i_cropped = croppCenter(y4b_i, y4t_o.shape)

        # concatenate
        y5t_i = torch.cat((y4t_i_cropped, y4t_o, y4b_o), dim=1)
        y5b_i = torch.cat((y4b_i_cropped, y4b_o, y4t_o), dim=1)

        y5t_o = self.conv5_Top(y5t_i)
        y5b_o = self.conv5_Bottom(y5b_i)

        # ------ Sixth layer ----- #
        y5t_i_cropped = croppCenter(y5t_i, y5t_o.shape)
        y5b_i_cropped = croppCenter(y5b_i, y5t_o.shape)

        # concatenate
        # y6t_i = torch.cat((y5t_i_cropped, y5t_o, y5b_o), dim=1)
        # y6b_i = torch.cat((y5b_i_cropped, y5b_o, y5t_o), dim=1)

        # y6t_o = self.conv6_Top(y6t_i)
        # y6b_o = self.conv6_Bottom(y6b_i)
        #
        # # ------ Seventh layer ----- #
        # y6t_i_cropped = croppCenter(y6t_i, y6t_o.shape)
        # y6b_i_cropped = croppCenter(y6b_i, y6t_o.shape)
        #
        # # concatenate
        # y7t_i = torch.cat((y6t_i_cropped, y6t_o, y6b_o), dim=1)
        # y7b_i = torch.cat((y6b_i_cropped, y6b_o, y6t_o), dim=1)

        # y7t_o = self.conv7_Top(y7t_i)
        # y7b_o = self.conv7_Bottom(y7b_i)

        # # ------ Eight layer ----- #
        # y7t_i_cropped = croppCenter(y7t_i, y7t_o.shape)
        # y7b_i_cropped = croppCenter(y7b_i, y7t_o.shape)
        #
        # # concatenate
        # y8t_i = torch.cat((y7t_i_cropped, y7t_o, y7b_o), dim=1)
        # y8b_i = torch.cat((y7b_i_cropped, y7b_o, y7t_o), dim=1)
        #
        # y8t_o = self.conv8_Top(y8t_i)
        # y8b_o = self.conv8_Bottom(y8b_i)
        #
        # # ------ Ninth layer ----- #
        # y8t_i_cropped = croppCenter(y8t_i, y8t_o.shape)
        # y8b_i_cropped = croppCenter(y8b_i, y8t_o.shape)
        #
        # # concatenate
        # y9t_i = torch.cat((y8t_i_cropped, y8t_o, y8b_o), dim=1)
        # y9b_i = torch.cat((y8b_i_cropped, y8b_o, y8t_o), dim=1)
        #
        # y9t_o = self.conv9_Top(y9t_i)
        # y9b_o = self.conv9_Bottom(y9b_i)

        ##### Fully connected layers
        # y9t_i_cropped = croppCenter(y9t_i, y9t_o.shape)
        # y9b_i_cropped = croppCenter(y9b_i, y9t_o.shape)

        outputPath_top = torch.cat((y5t_i_cropped, y5t_o, y5b_o), dim=1)
        outputPath_bottom = torch.cat((y5b_i_cropped, y5b_o, y5t_o), dim=1)

        inputFully = torch.cat((outputPath_top, outputPath_bottom), dim=1)

        y = self.fully_1(inputFully)
        y = self.fully_2(y)
        y = self.fully_3(y)

        x_fu = self.avgpool(y)
        x_fu = x_fu.view(-1, 150)
        # x_fu = x_fu.view(x_fu.size(0), -1)
        # x_fu = torch.flatten(x_fu, start_dim=1)
        x_fu = self.fc(x_fu)
        x_fu = self.dropout(x_fu)

        return x_fu

        # return self.final(y)

######################################IVD-NET##########################################


class Conv_residual_conv_Inception_Dilation(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv_Inception_Dilation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)

        self.conv_2_1 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_2_2 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_2_3 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv_2_4 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_2_5 = conv_block(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1, padding=4, dilation=4)

        self.conv_2_output = conv_block(self.out_dim * 5, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0,
                                        dilation=1)

        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)

        conv_2_1 = self.conv_2_1(conv_1)
        conv_2_2 = self.conv_2_2(conv_1)
        conv_2_3 = self.conv_2_3(conv_1)
        conv_2_4 = self.conv_2_4(conv_1)
        conv_2_5 = self.conv_2_5(conv_1)

        out1 = torch.cat([conv_2_1, conv_2_2, conv_2_3, conv_2_4, conv_2_5], 1)
        out1 = self.conv_2_output(out1)

        conv_3 = self.conv_3(out1 + conv_1)
        return conv_3


class Conv_residual_conv_Inception_Dilation_asymmetric(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv_Inception_Dilation_asymmetric, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)

        self.conv_2_1 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=1, stride=1,
                                                  padding=0, dilation=1)
        self.conv_2_2 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=1, dilation=1)
        self.conv_2_3 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=5, stride=1,
                                                  padding=2, dilation=1)
        self.conv_2_4 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=2, dilation=2)
        self.conv_2_5 = conv_block_Asym_Inception(self.out_dim, self.out_dim, act_fn, kernel_size=3, stride=1,
                                                  padding=4, dilation=4)

        self.conv_2_output = conv_block(self.out_dim * 5, self.out_dim, act_fn, kernel_size=1, stride=1, padding=0,
                                        dilation=1)

        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)

        conv_2_1 = self.conv_2_1(conv_1)
        conv_2_2 = self.conv_2_2(conv_1)
        conv_2_3 = self.conv_2_3(conv_1)
        conv_2_4 = self.conv_2_4(conv_1)
        conv_2_5 = self.conv_2_5(conv_1)

        out1 = torch.cat([conv_2_1, conv_2_2, conv_2_3, conv_2_4, conv_2_5], 1)
        out1 = self.conv_2_output(out1)

        conv_3 = self.conv_3(out1 + conv_1)
        return conv_3


class IVD_Net_asym(nn.Module):

    def __init__(self, input_nc, output_nc, ngf):
        super(IVD_Net_asym, self).__init__()
        print('~' * 50)
        print(' ----- Creating FUSION_NET HD (Assymetric) network...')
        print('~' * 50)

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        # act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn = nn.ReLU()

        act_fn_2 = nn.ReLU()

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1)
        self.down_1_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)
        self.pool_1_0 = maxpool()
        self.down_2_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 2, self.out_dim * 1, act_fn)
        self.pool_2_0 = maxpool()
        self.down_3_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_3_0 = maxpool()
        self.down_4_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 8, self.out_dim * 4, act_fn)
        self.pool_4_0 = maxpool()

        # Encoder (Modality 2)
        self.down_1_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)
        self.pool_1_1 = maxpool()
        self.down_2_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 2, self.out_dim * 1, act_fn)
        self.pool_2_1 = maxpool()
        self.down_3_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_3_1 = maxpool()
        self.down_4_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 8, self.out_dim * 4, act_fn)
        self.pool_4_1 = maxpool()

        # Encoder (Modality 3)
        self.down_1_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)
        self.pool_1_2 = maxpool()
        self.down_2_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_2_2 = maxpool()
        self.down_3_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 12, self.out_dim * 4, act_fn)
        self.pool_3_2 = maxpool()
        self.down_4_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 28, self.out_dim * 8, act_fn)
        self.pool_4_2 = maxpool()

        # Encoder (Modality 4)
        self.down_1_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.in_dim, self.out_dim, act_fn)
        self.pool_1_3 = maxpool()
        self.down_2_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.pool_2_3 = maxpool()
        self.down_3_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 12, self.out_dim * 4, act_fn)
        self.pool_3_3 = maxpool()
        self.down_4_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 28, self.out_dim * 8, act_fn)
        self.pool_4_3 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 16, self.out_dim * 16, act_fn)

        # ~~~ Decoding Path ~~~~~~ #
        self.deconv_1 = conv_decod_block(self.out_dim * 16, self.out_dim * 4, act_fn_2)
        self.up_1 = Conv_residual_conv_Inception_Dilation(self.out_dim * 4, self.out_dim * 4, act_fn_2)

        self.deconv_2 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_2 = Conv_residual_conv_Inception_Dilation(self.out_dim * 2, self.out_dim * 2, act_fn_2)

        self.deconv_3 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn_2)
        self.up_3 = Conv_residual_conv_Inception_Dilation(self.out_dim * 1, self.out_dim * 1, act_fn_2)

        self.deconv_4 = conv_decod_block(self.out_dim * 1, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv_Inception_Dilation(self.out_dim, self.out_dim, act_fn_2)

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.1)

        # Params initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_uniform(m.weight.data)
                # init.xavier_uniform(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # def forward(self, input):
    def forward(self, input1, input2):

        # ############################# #
        # ~~~~~~ Encoding path ~~~~~~~  #

        # i0 = input[:, 0:1, :, :]
        # i1 = input[:, 1:2, :, :]
        i0 = input1
        i1 = input2

        # -----  First Level --------
        down_1_0 = self.down_1_0(i0)
        down_1_1 = self.down_1_1(i1)

        # -----  Second Level --------
        # input_2nd = torch.cat((down_1_0,down_1_1,down_1_2,down_1_3),dim=1)
        input_2nd_0 = torch.cat((self.pool_1_0(down_1_0), self.pool_1_1(down_1_1)), dim=1)

        input_2nd_1 = torch.cat((self.pool_1_1(down_1_1), self.pool_1_0(down_1_0)), dim=1)

        down_2_0 = self.down_2_0(input_2nd_0)
        down_2_1 = self.down_2_1(input_2nd_1)


        # -----  Third Level --------
        # Max-pool
        down_2_0m = self.pool_2_0(down_2_0)
        down_2_1m = self.pool_2_0(down_2_1)

        input_3rd_0 = torch.cat((down_2_0m, down_2_1m), dim=1)
        input_3rd_0 = torch.cat((input_3rd_0, croppCenter(input_2nd_0, input_3rd_0.shape)), dim=1)

        input_3rd_1 = torch.cat((down_2_1m, down_2_0m), dim=1)
        input_3rd_1 = torch.cat((input_3rd_1, croppCenter(input_2nd_1, input_3rd_1.shape)), dim=1)

        down_3_0 = self.down_3_0(input_3rd_0)
        down_3_1 = self.down_3_1(input_3rd_1)

        # -----  Fourth Level --------

        # Max-pool
        down_3_0m = self.pool_3_0(down_3_0)
        down_3_1m = self.pool_3_0(down_3_1)

        input_4th_0 = torch.cat((down_3_0m, down_3_1m), dim=1)
        input_4th_0 = torch.cat((input_4th_0, croppCenter(input_3rd_0, input_4th_0.shape)), dim=1)

        input_4th_1 = torch.cat((down_3_1m, down_3_0m), dim=1)
        input_4th_1 = torch.cat((input_4th_1, croppCenter(input_3rd_1, input_4th_1.shape)), dim=1)

        down_4_0 = self.down_4_0(input_4th_0)
        down_4_1 = self.down_4_1(input_4th_1)

        # ----- Bridge -----
        # Max-pool
        down_4_0m = self.pool_4_0(down_4_0)
        down_4_1m = self.pool_4_0(down_4_1)

        inputBridge = torch.cat((down_4_0m, down_4_1m), dim=1)
        inputBridge = torch.cat((inputBridge, croppCenter(input_4th_0, inputBridge.shape)), dim=1)
        bridge = self.bridge(inputBridge)

        x_fu = self.avgpool(bridge)
        x_fu = x_fu.view(-1, 512)
        # x_fu = x_fu.view(x_fu.size(0), -1)
        # x_fu = torch.flatten(x_fu, start_dim=1)
        x_fu = self.fc(x_fu)
        x_fu = self.dropout(x_fu)

        return x_fu
        #
        # ############################# #
        # ~~~~~~ Decoding path ~~~~~~~  #
        # deconv_1 = self.deconv_1(bridge)
        # skip_1 = (deconv_1 + down_4_0 + down_4_1) / 3  # Residual connection
        # up_1 = self.up_1(skip_1)
        # deconv_2 = self.deconv_2(up_1)
        # skip_2 = (deconv_2 + down_3_0 + down_3_1) / 3  # Residual connection
        # up_2 = self.up_2(skip_2)
        # deconv_3 = self.deconv_3(up_2)
        # skip_3 = (deconv_3 + down_2_0 + down_2_1) / 3  # Residual connection
        # up_3 = self.up_3(skip_3)
        # deconv_4 = self.deconv_4(up_3)
        # skip_4 = (deconv_4 + down_1_0 + down_1_1) / 3  # Residual connection
        # up_4 = self.up_4(skip_4)

        # Last output
        # return F.softmax(self.out(up_4))
        # return self.out(up_4)


######################################MMBIFPN##########################################


def croppCenter(tensorToCrop, finalShape):
    org_shape = tensorToCrop.shape

    diff = np.zeros(2)
    diff[0] = org_shape[2] - finalShape[2]
    diff[1] = org_shape[3] - finalShape[3]

    croppBorders = np.zeros(2, dtype=int)
    croppBorders[0] = int(diff[0] / 2)
    croppBorders[1] = int(diff[1] / 2)

    return tensorToCrop[:, :, croppBorders[0]:croppBorders[0] + finalShape[2],
           croppBorders[1]:croppBorders[1] + finalShape[3]]


class MM_BiFPN(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32):
        super(MM_BiFPN, self).__init__()
        print('~' * 50)
        print(' ---- Creating Multi UNet ---')
        print('~' * 50)

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        # Encoder (Modality 1) Flair 1
        self.down_1_0 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_0 = maxpool()
        self.down_2_0 = ConvBlock2d(self.out_dim, self.out_dim * 2)
        self.pool_2_0 = maxpool()
        self.down_3_0 = ConvBlock2d(self.out_dim * 2, self.out_dim * 4)
        self.pool_3_0 = maxpool()
        self.down_4_0 = ConvBlock2d(self.out_dim * 4, self.out_dim * 8)
        self.pool_4_0 = maxpool()

        # Encoder (Modality 2) T1
        self.down_1_1 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_1 = maxpool()
        self.down_2_1 = ConvBlock2d(self.out_dim, self.out_dim * 2)
        self.pool_2_1 = maxpool()
        self.down_3_1 = ConvBlock2d(self.out_dim * 2, self.out_dim * 4)
        self.pool_3_1 = maxpool()
        self.down_4_1 = ConvBlock2d(self.out_dim * 4, self.out_dim * 8)
        self.pool_4_1 = maxpool()

        # Encoder (Modality 3) T1c
        self.down_1_2 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_2 = maxpool()
        self.down_2_2 = ConvBlock2d(self.out_dim, self.out_dim * 2)
        self.pool_2_2 = maxpool()
        self.down_3_2 = ConvBlock2d(self.out_dim * 2, self.out_dim * 4)
        self.pool_3_2 = maxpool()
        self.down_4_2 = ConvBlock2d(self.out_dim * 4, self.out_dim * 8)
        self.pool_4_2 = maxpool()

        # Encoder (Modality 4) T2
        self.down_1_3 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_3 = maxpool()
        self.down_2_3 = ConvBlock2d(self.out_dim, self.out_dim * 2)
        self.pool_2_3 = maxpool()
        self.down_3_3 = ConvBlock2d(self.out_dim * 2, self.out_dim * 4)
        self.pool_3_3 = maxpool()
        self.down_4_3 = ConvBlock2d(self.out_dim * 4, self.out_dim * 8)
        self.pool_4_3 = maxpool()

        # bridge between encoder decoder
        self.bridge = ConvBlock2d(self.out_dim * 16, self.out_dim * 16)

        # bifpn
        self.bifpn = BiFPNUnit(n=self.in_dim, channels=self.out_dim)

        # ~~ Decoding Path ~~#

        self.upLayer1 = UpBlock2d(self.out_dim * 16, self.out_dim * 8)
        self.upLayer2 = UpBlock2d(self.out_dim * 8, self.out_dim * 4)
        self.upLayer3 = UpBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.upLayer4 = UpBlock2d(self.out_dim * 2, self.out_dim * 1)

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

        # self.fc = nn.Linear(4194304/2, 2)
        self.fc = nn.Linear(2097152, 2)
        self.dropout = nn.Dropout(0.1)

    # def forward(self, input):
    def forward(self, input0, input1):
        # ~~~ Encoding Path ~~

        # i0 = input[:, 0:1, :, :]  # comment to remove flair
        # i1 = input[:, 1:2, :, :]  # comment to remove t1
        i0 = input0
        i1 = input1
        # print('i0')
        # print(i0.shape)
        # print('i1')
        # print(i1.shape)

        down_1_0 = self.down_1_0(i0)
        down_1_1 = self.down_1_1(i1)
        # print('down_1_0')
        # print(down_1_0.shape)

        input_2nd_0 = self.pool_1_0(down_1_0)
        input_2nd_1 = self.pool_1_1(down_1_1)
        # print('input_2nd_0')
        # print(input_2nd_0.shape)

        down_2_0 = self.down_2_0(input_2nd_0)
        down_2_1 = self.down_2_1(input_2nd_1)
        # print('down_2_0')
        # print(down_2_0.shape)

        input_3rd_0 = self.pool_2_0(down_2_0)
        input_3rd_1 = self.pool_2_1(down_2_1)
        # print('input_3rd_0')
        # print(input_3rd_0.shape)

        down_3_0 = self.down_3_0(input_3rd_0)
        down_3_1 = self.down_3_1(input_3rd_1)
        # print('down_3_0')
        # print(down_3_0.shape)

        input_4th_0 = self.pool_3_0(down_3_0)
        input_4th_1 = self.pool_3_1(down_3_1)
        # print('input_4th_0')
        # print(input_4th_0.shape)

        down_4_0 = self.down_4_0(input_4th_0)  # 8C
        down_4_1 = self.down_4_1(input_4th_1)
        # print('down_4_0')
        # print(down_4_0.shape)

        down_4_0m = self.pool_4_0(down_4_0)
        down_4_1m = self.pool_4_0(down_4_1)
        # print('down_4_0m')
        # print(down_4_0m.shape)

        inputBridge = torch.cat((down_4_0m, down_4_1m), dim=1)
        # print('inputBridge')
        # print(inputBridge.shape)

        bridge = self.bridge(inputBridge)
        # print('bridge ')
        # print(bridge.shape)

        skip_1 = torch.cat((down_4_0, down_4_1), dim=1)
        # print('skip_1 ')
        # print(skip_1.shape)
        skip_2 = torch.cat((down_3_0, down_3_1), dim=1)
        # print('skip_2 ')
        # print(skip_2.shape)
        skip_3 = torch.cat((down_2_0, down_2_1), dim=1)
        # print('skip_3 ')
        # print(skip_3.shape)
        skip_4 = torch.cat((down_1_0, down_1_1), dim=1)
        # print('skip_4 ')
        # print(skip_4.shape)

        x12, x22, x32, x42 = self.bifpn(skip_4, skip_3, skip_2, skip_1)

        x = self.upLayer1(bridge, x42)
        # x = self.upLayer1(x42)
        # print('uplayer1')
        # print(x.shape)
        x = self.upLayer2(x, x32)
        # print('uplayer2')
        # print(x.shape)
        x = self.upLayer3(x, x22)
        # print('uplayer3')
        # print(x.shape)
        x = self.upLayer4(x, x12)
        # print('uplayer4')
        # print(x.shape)

        x_fu = x.view(-1, 2097152)
        # x_fu = x_fu.view(x_fu.size(0), -1)
        # x_fu = torch.flatten(x_fu, start_dim=1)
        x_fu = self.fc(x_fu)
        x_fu = self.dropout(x_fu)
        # return self.out(x)
        return x_fu

###################################O V E R############################################

print('获取测试数据和验证数据')
# X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, random_state=666)

phase_test = False
test_data = MultiModalityData_load(args, train=False, test=True, k=3)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

print('获取模型')

Y_pred_my = []
Y_pred_HyperDenseNet = []
Y_pred_IVDNET = []
Y_pred_MMBIFPN = []
Y_pred_Deepak = []
Y_pred_Naser = []

Y_valid = []

# MMBIFPN&HyperDenseNet change
model_my = torch.load("./conv4-12/net3_29.pkl", map_location='cpu')
print("model_my ok")
# model_HyperDenseNet = torch.load("./conv4-64/net3_29.pkl", map_location='cpu')
model_HyperDenseNet = torch.load("./conv4-66/net3_29.pkl", map_location='cpu')
print("model_HyperDenseNet ok")
model_IVDNET = torch.load("./conv4-62/net3_29.pkl", map_location='cpu')
print("model_IVDNET ok")
# model_MMBIFPN = torch.load("./conv4-66/net3_29.pkl", map_location='cpu')
model_MMBIFPN = torch.load("./conv4-69/net3_29.pkl", map_location='cpu')
print("model_MMBIFPN ok")
# model_Deepak = torch.load("./conv4-69/net3_29.pkl", map_location='cpu')
model_Deepak = torch.load("./conv4-64/net3_29.pkl", map_location='cpu')
print("model_Deepak ok")
model_Naser = torch.load("./conv4-57/net3_29.pkl", map_location='cpu')
print("model_Naser ok")

model_my.eval()
model_HyperDenseNet.eval()
model_IVDNET.eval()
model_MMBIFPN.eval()
model_Deepak.eval()
model_Naser.eval()

model_my.cuda()
model_HyperDenseNet.cuda()
model_IVDNET.cuda()
model_MMBIFPN.cuda()
model_Naser.cuda()
model_Deepak.cuda()

for ii, (image1, image2, labels) in enumerate(test_loader):
    image1 = image1.cuda()
    image2 = image2.cuda()
    labels = labels[0].cuda()

    with torch.no_grad():
        outputs_my = model_my(image1, image2)  # outputs为得分情况
        outputs_HyperDenseNet = model_HyperDenseNet(image1, image2)  # outputs为得分情况
        outputs_IVDNET = model_IVDNET(image1, image2)  # outputs为得分情况
        outputs_MMBIFPN = model_MMBIFPN(image1, image2)  # outputs为得分情况
        outputs_Deepak = model_Deepak(image1, image2)  # outputs为得分情况
        outputs_Naser = model_Naser(image1, image2)  # outputs为得分情况

        outputs_my = torch.sigmoid_(outputs_my)
        outputs_HyperDenseNet = torch.sigmoid_(outputs_HyperDenseNet)
        outputs_IVDNET = torch.sigmoid_(outputs_IVDNET)
        outputs_MMBIFPN = torch.sigmoid_(outputs_MMBIFPN)
        outputs_Deepak = torch.sigmoid_(outputs_Deepak)
        outputs_Naser = torch.sigmoid_(outputs_Naser)

        output_np_my = outputs_my.cpu().numpy()
        output_np_HyperDenseNet = outputs_HyperDenseNet.cpu().numpy()
        output_np_IVDNET = outputs_IVDNET.cpu().numpy()
        output_np_MMBIFPN = outputs_MMBIFPN.cpu().numpy()
        output_np_Deepak = outputs_Deepak.cpu().numpy()
        output_np_Naser = outputs_Naser.cpu().numpy()

        label_np = labels.cpu().numpy()
        Y_pred_batch_my = output_np_my[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_HyperDenseNet = output_np_HyperDenseNet[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_IVDNET = output_np_IVDNET[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_MMBIFPN = output_np_MMBIFPN[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_Deepak = output_np_Deepak[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_Naser = output_np_Naser[:, -1]  # 预测为正例的得分情况

        for i in range(len(Y_pred_batch_my)):
            Y_pred_my.append(Y_pred_batch_my[i])
            Y_pred_HyperDenseNet.append(Y_pred_batch_HyperDenseNet[i])
            Y_pred_IVDNET.append(Y_pred_batch_IVDNET[i])
            Y_pred_MMBIFPN.append(Y_pred_batch_MMBIFPN[i])
            Y_pred_Deepak.append(Y_pred_batch_Deepak[i])
            Y_pred_Naser.append(Y_pred_batch_Naser[i])
        for i in range(len(label_np)):
            Y_valid.append(label_np[i])

# micro：多分类　　
# weighted：不均衡数量的类来说，计算二分类metrics的平均
# macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
# precision = precision_score(Y_valid, Y_pred, average='weighted')
# recall = recall_score(Y_valid, Y_pred, average='weighted')
# f1_score = f1_score(Y_valid, Y_pred, average='weighted')
# accuracy_score = accuracy_score(Y_valid, Y_pred)
# print("Precision_score:", precision)
# print("Recall_score:", recall)
# print("F1_score:", f1_score)
# print("Accuracy_score:", accuracy_score)

# 二分类　ＲＯＣ曲线
# roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
# 横坐标：假正率（False Positive Rate , FPR）

fpr_my, tpr_my, thresholds_keras_my = roc_curve(np.array(Y_valid), np.array(Y_pred_my), pos_label=1)
fpr_HyperDenseNet, tpr_HyperDenseNet, thresholds_keras_HyperDenseNet = roc_curve(np.array(Y_valid), np.array(Y_pred_HyperDenseNet), pos_label=1)
fpr_IVDNET, tpr_IVDNET, thresholds_keras_IVDNET = roc_curve(np.array(Y_valid), np.array(Y_pred_IVDNET), pos_label=1)
fpr_MMBIFPN, tpr_MMBIFPN, thresholds_keras_MMBIFPN = roc_curve(np.array(Y_valid), np.array(Y_pred_MMBIFPN), pos_label=1)
fpr_Deepak, tpr_Deepak, thresholds_keras_Deepak = roc_curve(np.array(Y_valid), np.array(Y_pred_Deepak), pos_label=1)
fpr_Naser, tpr_Naser, thresholds_keras_Naser = roc_curve(np.array(Y_valid), np.array(Y_pred_Naser), pos_label=1)

auc_my = auc(fpr_my, tpr_my)
auc_HyperDenseNet = auc(fpr_HyperDenseNet, tpr_HyperDenseNet)
auc_IVDNET = auc(fpr_IVDNET, tpr_IVDNET)
auc_MMBIFPN = auc(fpr_MMBIFPN, tpr_MMBIFPN)
auc_Deepak = auc(fpr_Deepak, tpr_Deepak)
auc_Naser = auc(fpr_Naser, tpr_Naser)

print("AUC_my : ", auc_my)
print("AUC_HyperDenseNet : ", auc_HyperDenseNet)
print("AUC_IVDNET : ", auc_IVDNET)
print("AUC_MMBIFPN : ", auc_MMBIFPN)
print("AUC_Deepak : ", auc_Deepak)
print("AUC_Naser : ", auc_Naser)

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_my, tpr_my, label='Ours (auc = {:.3f})'.format(auc_my), color='#8d6ab8')
plt.plot(fpr_HyperDenseNet, tpr_HyperDenseNet, label='HyperDenseNet (auc = {:.3f})'.format(auc_HyperDenseNet), color='#B3B8BC')
plt.plot(fpr_IVDNET, tpr_IVDNET, label='IVDNET (auc = {:.3f})'.format(auc_IVDNET), color='#76ba80')
plt.plot(fpr_MMBIFPN, tpr_MMBIFPN, label='MMBIFPN (auc = {:.3f})'.format(auc_MMBIFPN), color='#8EA3C2')
plt.plot(fpr_Deepak, tpr_Deepak, label='Deepak at el (auc = {:.3f})'.format(auc_Deepak), color='#F1DF82')
plt.plot(fpr_Naser, tpr_Naser, label='Naser at el (auc = {:.3f})'.format(auc_Naser), color='#EDB17F')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
# plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig("ROC_curve.svg", format='svg', dpi=256)
# plt.show()

print("--- %s seconds ---" % (time.time() - start_time))