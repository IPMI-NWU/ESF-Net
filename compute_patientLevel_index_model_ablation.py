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
# from data_loader_fusion import *
# from data_loader_fusion_others import *
from data_loader_fake_fake_fake import *
from torch.utils.data import DataLoader
import math

# Training settings
# 设置一些参数,每个都有默认值,输入python main.py -h可以获得帮助
parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=30, metavar='N',
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

# torch.cuda.set_device(3)
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


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


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

        # out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)
        out_fusion = torch.cat((fusion_sum, fusion_max, fusion_max), dim=1)

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

        # out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)
        out_fusion = torch.cat((fusion_sum, fusion_max, fusion_max), dim=1)

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

        # out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)
        out_fusion = torch.cat((fusion_sum, fusion_max, fusion_max), dim=1)

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


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


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


def getResult():
    print('获取测试数据和验证数据')
    # X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, random_state=666)

    phase_test = False
    test_data_0 = MultiModalityData_load(args, train=False, test=True, k=0)
    test_data_1 = MultiModalityData_load(args, train=False, test=True, k=1)
    test_data_2 = MultiModalityData_load(args, train=False, test=True, k=2)
    test_data_3 = MultiModalityData_load(args, train=False, test=True, k=3)

    test_loader_0 = DataLoader(test_data_0, batch_size=args.batch_size, shuffle=False)
    test_loader_1 = DataLoader(test_data_1, batch_size=args.batch_size, shuffle=False)
    test_loader_2 = DataLoader(test_data_2, batch_size=args.batch_size, shuffle=False)
    test_loader_3 = DataLoader(test_data_3, batch_size=args.batch_size, shuffle=False)

    print('获取模型')

    Y_pred_8 = []
    Y_valid = []


    def sigmoid_function(z):
        fz = []
        for num in z:
            fz.append(1/(1 + math.exp(-num)))
        return fz


    # 9&8 change and 5&4 change
    # model_10 = torch.load("./conv4-49/net3_29.pkl", map_location='cpu')
    # print('model_10 ok')
    model_8 = torch.load("./conv4-70/net0_29.pkl", map_location='cpu')
    print('model_8 ok')
    # model_9 = torch.load("./conv4-12/net3_29.pkl", map_location='cpu')
    # print('model_8 ok')
    # model_7 = torch.load("./conv4-54/net3_29.pkl", map_location='cpu')
    # print('model_7 ok')
    # model_6 = torch.load("./conv4-55/net3_29.pkl", map_location='cpu')
    # print('model_6 ok')
    # model_4 = torch.load("./conv4-56/net3_29.pkl", map_location='cpu')
    # print('model_5 ok')
    # model_5 = torch.load("./conv4-58/net3_29.pkl", map_location='cpu')
    # print('model_4 ok')
    # model_3 = torch.load("./conv4-59/net3_29.pkl", map_location='cpu')
    # print('model_3 ok')
    # model_2 = torch.load("./conv4-60/net3_29.pkl", map_location='cpu')
    # print('model_2 ok')

    model_8.eval()
    model_8.cuda()

    for ii, (image1, image2, labels) in enumerate(test_loader_0):
        image1 = image1.cuda()
        image2 = image2.cuda()
        labels = labels[0].cuda()

        with torch.no_grad():
            outputs_8 = model_8(image1, image2)  # outputs为得分情况
            _, pred_labels = torch.max(outputs_8, 1)
            # outputs_8 = torch.sigmoid(outputs_8)
            # output_np_8 = outputs_8.cpu().numpy()

            pred_np = pred_labels.cpu().numpy()
            label_np = labels.cpu().numpy()

            Txt = open("./patient_level.txt", "a")
            for i in pred_np:
                Txt.write(str(i) + 'w')
            Txt.close()


def computeIndex():
    fileA = 's_test3_GPU.txt'
    fileB = 's_test3_result.txt'
    fhA = open(fileA, 'r')
    fhB = open(fileB, 'r')
    imgs = []
    stand = []

    for line in fhB:
        result = line.split('w')

    for line in fhA:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        pre_index = words[0].split('\\')
        # window10 上运行
        imgs.append((pre_index[1], words[1]))

    mark = imgs[0][0]
    stand.append(imgs[0][1])
    count0 = 0
    count1 = 0
    patient_result = []
    i = 0
    # for i in range(len(imgs)):
    while i < len(imgs):
        if imgs[i][0] == mark:
            if result[i] == "0":
                count0 = count0 + 1
                i = i + 1
            else:
                count1 = count1 + 1
                i = i + 1
        else:
            mark = imgs[i][0]
            stand.append(imgs[i][1])
            if count1 >= count0:
                patient_result.append('1')
                count1 = 0
                count0 = 0
            else:
                patient_result.append('0')
                count1 = 0
                count0 = 0

    if count1 >= count0:
        patient_result.append('1')
    else:
        patient_result.append('0')

    print('patient_result', patient_result)
    print('stand', stand)
    # 计算TP TN FP FN
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(patient_result)):
        if patient_result[i] == stand[i] == '1':
            TP = TP + 1
        elif patient_result[i] == stand[i] == '0':
            TN = TN + 1
        elif patient_result[i] == '1' and stand[i] == '0':
            FP = FP + 1
        elif patient_result[i] == '0' and stand[i] == '1':
            FN = FN + 1

    print(str(TP) + "  " + str(FN) + "  " + str(FP) + "  " + str(TN))


if __name__ == '__main__':
    # getResult()
    computeIndex()