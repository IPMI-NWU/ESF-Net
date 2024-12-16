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
from data_loader_fake_fake_fake import *
from torch.utils.data import DataLoader
import math

# Training settings
# 设置一些参数,每个都有默认值,输入python main.py -h可以获得帮助
parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
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

torch.cuda.set_device(2)
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


print('获取测试数据和验证数据')
# X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, random_state=666)

phase_test = False
test_data_0 = MultiModalityData_load(args, train=False, test=True, k=0)
test_data_1 = MultiModalityData_load(args, train=False, test=True, k=1)
test_data_2 = MultiModalityData_load(args, train=False, test=True, k=2)
test_data_3 = MultiModalityData_load(args, train=False, test=True, k=3)

test_loader_0 = DataLoader(test_data_0, batch_size=args.batch_size, shuffle=True)
test_loader_1 = DataLoader(test_data_1, batch_size=args.batch_size, shuffle=True)
test_loader_2 = DataLoader(test_data_2, batch_size=args.batch_size, shuffle=True)
test_loader_3 = DataLoader(test_data_3, batch_size=args.batch_size, shuffle=True)

print('获取模型')

Y_pred_10 = []
Y_pred_9 = []
Y_pred_8 = []
Y_pred_7 = []
Y_pred_6 = []
Y_pred_5 = []
Y_pred_4 = []
Y_pred_3 = []
Y_pred_2 = []

Y_valid = []

def sigmoid_function(z):
    fz = []
    for num in z:
        fz.append(1/(1 + math.exp(-num)))
    return fz


# 9&8 change and 5&4 change
# model_10 = torch.load("./conv4-49/net3_29.pkl", map_location='cpu')
# print('model_10 ok')
model_8 = torch.load("./conv4-57/net3_29.pkl", map_location='cpu')
print('model_9 ok')
model_9 = torch.load("./conv4-12/net3_29.pkl", map_location='cpu')
print('model_8 ok')
model_7 = torch.load("./conv4-54/net3_29.pkl", map_location='cpu')
print('model_7 ok')
model_6 = torch.load("./conv4-55/net3_29.pkl", map_location='cpu')
print('model_6 ok')
model_4 = torch.load("./conv4-56/net3_29.pkl", map_location='cpu')
print('model_5 ok')
model_5 = torch.load("./conv4-58/net3_29.pkl", map_location='cpu')
print('model_4 ok')
model_3 = torch.load("./conv4-59/net3_29.pkl", map_location='cpu')
print('model_3 ok')
model_2 = torch.load("./conv4-60/net3_29.pkl", map_location='cpu')
print('model_2 ok')

# model_10.eval()
model_9.eval()
model_8.eval()
model_7.eval()
model_6.eval()
model_5.eval()
model_4.eval()
model_3.eval()
model_2.eval()

# model_10.cuda()
model_9.cuda()
model_8.cuda()
model_7.cuda()
model_6.cuda()
model_5.cuda()
model_4.cuda()
model_3.cuda()
model_2.cuda()
# opt = torch.optim.Adam(model.parameters(), lr=0.001)

for ii, (image1, image2, labels) in enumerate(test_loader_3):
    image1 = image1.cuda()
    image2 = image2.cuda()
    labels = labels[0].cuda()

    with torch.no_grad():
        # outputs_10 = model_10(image1, image2)  # outputs为得分情况
        outputs_9 = model_9(image1, image2)  # outputs为得分情况
        outputs_8 = model_8(image1, image2)  # outputs为得分情况
        outputs_7 = model_7(image1, image2)  # outputs为得分情况
        outputs_6 = model_6(image1, image2)  # outputs为得分情况
        outputs_5 = model_5(image1, image2)  # outputs为得分情况
        outputs_4 = model_4(image1, image2)  # outputs为得分情况
        outputs_3 = model_3(image1, image2)  # outputs为得分情况
        outputs_2 = model_2(image1, image2)  # outputs为得分情况

        # outputs_10 = torch.sigmoid(outputs_10)
        outputs_9 = torch.sigmoid(outputs_9)
        outputs_8 = torch.sigmoid(outputs_8)
        outputs_7 = torch.sigmoid(outputs_7)
        outputs_6 = torch.sigmoid(outputs_6)
        outputs_5 = torch.sigmoid(outputs_5)
        outputs_4 = torch.sigmoid(outputs_4)
        outputs_3 = torch.sigmoid(outputs_3)
        outputs_2 = torch.sigmoid(outputs_2)

        # output_np_10 = outputs_10.cpu().numpy()
        output_np_9 = outputs_9.cpu().numpy()
        output_np_8 = outputs_8.cpu().numpy()
        output_np_7 = outputs_7.cpu().numpy()
        output_np_6 = outputs_6.cpu().numpy()
        output_np_5 = outputs_5.cpu().numpy()
        output_np_4 = outputs_4.cpu().numpy()
        output_np_3 = outputs_3.cpu().numpy()
        output_np_2 = outputs_2.cpu().numpy()

        label_np = labels.cpu().numpy()

        # Y_pred_batch_10 = output_np_10[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_9 = output_np_9[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_8 = output_np_8[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_7 = output_np_7[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_6 = output_np_6[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_5 = output_np_5[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_4 = output_np_4[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_3 = output_np_3[:, -1]  # 预测为正例的得分情况
        Y_pred_batch_2 = output_np_2[:, -1]  # 预测为正例的得分情况

        for i in range(len(Y_pred_batch_5)):
            # Y_pred_10.append(Y_pred_batch_10[i])
            Y_pred_9.append(Y_pred_batch_9[i])
            Y_pred_8.append(Y_pred_batch_8[i])
            Y_pred_7.append(Y_pred_batch_7[i])
            Y_pred_6.append(Y_pred_batch_6[i])
            Y_pred_5.append(Y_pred_batch_5[i])
            Y_pred_4.append(Y_pred_batch_4[i])
            Y_pred_3.append(Y_pred_batch_3[i])
            Y_pred_2.append(Y_pred_batch_2[i])
        for i in range(len(label_np)):
            Y_valid.append(label_np[i])

print("计算完成")
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

# fpr_10, tpr_10, thresholds_keras_10 = roc_curve(np.array(Y_valid), np.array(Y_pred_10))
fpr_9, tpr_9, thresholds_keras_9 = roc_curve(np.array(Y_valid), np.array(Y_pred_9))
fpr_8, tpr_8, thresholds_keras_8 = roc_curve(np.array(Y_valid), np.array(Y_pred_8))
fpr_7, tpr_7, thresholds_keras_7 = roc_curve(np.array(Y_valid), np.array(Y_pred_7))
fpr_6, tpr_6, thresholds_keras_6 = roc_curve(np.array(Y_valid), np.array(Y_pred_6))
fpr_5, tpr_5, thresholds_keras_5 = roc_curve(np.array(Y_valid), np.array(Y_pred_5))
fpr_4, tpr_4, thresholds_keras_4 = roc_curve(np.array(Y_valid), np.array(Y_pred_4))
fpr_3, tpr_3, thresholds_keras_3 = roc_curve(np.array(Y_valid), np.array(Y_pred_3))
fpr_2, tpr_2, thresholds_keras_2 = roc_curve(np.array(Y_valid), np.array(Y_pred_2))
# auc_10 = auc(fpr_10, tpr_10)
auc_9 = auc(fpr_9, tpr_9)
auc_8 = auc(fpr_8, tpr_8)
auc_7 = auc(fpr_7, tpr_7)
auc_6 = auc(fpr_6, tpr_6)
auc_5 = auc(fpr_5, tpr_5)
auc_4 = auc(fpr_4, tpr_4)
auc_3 = auc(fpr_3, tpr_3)
auc_2 = auc(fpr_2, tpr_2)
# print("AUC_10 : ", auc_10)
print("AUC_9 : ", auc_9)
print("AUC_8 : ", auc_8)
print("AUC_7 : ", auc_7)
print("AUC_6 : ", auc_6)
print("AUC_5 : ", auc_5)
print("AUC_4 : ", auc_4)
print("AUC_3 : ", auc_3)
print("AUC_2 : ", auc_2)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_10, tpr_10, label='R = 0.0 (auc = {:.3f})'.format(auc_10))
plt.plot(fpr_9, tpr_9, label='R = 0.1 (auc = {:.3f})'.format(auc_9))
plt.plot(fpr_8, tpr_8, label='R = 0.2 (auc = {:.3f})'.format(auc_8))
plt.plot(fpr_7, tpr_7, label='R = 0.3 (auc = {:.3f})'.format(auc_7))
plt.plot(fpr_6, tpr_6, label='R = 0.4 (auc = {:.3f})'.format(auc_6))
plt.plot(fpr_5, tpr_5, label='R = 0.5 (auc = {:.3f})'.format(auc_5))
plt.plot(fpr_4, tpr_4, label='R = 0.6 (auc = {:.3f})'.format(auc_4))
plt.plot(fpr_3, tpr_3, label='R = 0.7 (auc = {:.3f})'.format(auc_3))
plt.plot(fpr_2, tpr_2, label='R = 0.8 (auc = {:.3f})'.format(auc_2))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
# plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig("ROC_curve.png")
# plt.show()

print("--- %s seconds ---" % (time.time() - start_time))