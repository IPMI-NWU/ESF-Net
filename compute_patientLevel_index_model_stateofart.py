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
from Blocks import *
from backbone import *
from bifpn import BiFPNUnit

# Training settings
# 设置一些参数,每个都有默认值,输入python main.py -h可以获得帮助
parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
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
    model_8 = torch.load("./conv4-69/net3_29.pkl", map_location='cpu')
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

    for ii, (image1, image2, labels) in enumerate(test_loader_3):
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
    fileA = 's_test0_GPU.txt'
    fileB = 's_test0_result.txt'
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
    getResult()
    # computeIndex()