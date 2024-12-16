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
# from data_loader_fusion_others import *
# from data_loader_fake_fake_fake import *
from torch.utils.data import DataLoader
import math
from torchsummary import summary

# Training settings
# 设置一些参数,每个都有默认值,输入python main.py -h可以获得帮助
parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=30, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--channels', type=int, default=1)
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

# 设置一个随机数种子
torch.manual_seed(args.seed)
if args.cuda:
    # 为GPU设置一个随机数种子
    torch.cuda.manual_seed(args.seed)


input_shape = (args.channels, args.img_height, args.img_width)


class ResidualBlock(nn.Module):
    """
    每一个ResidualBlock,需要保证输入和输出的维度不变
    所以卷积核的通道数都设置成一样
    """
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1)


    def forward(self, x):
        """
        ResidualBlock中有跳跃连接;
        在得到第二次卷积结果时,需要加上该残差块的输入,
        再将结果进行激活,实现跳跃连接 ==> 可以避免梯度消失
        在求导时,因为有加上原始的输入x,所以梯度为: dy + 1,在1附近
        """
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)

        # return x + self.block(x)


def maxpool():
    pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
    return pool

def maxpool2():
    pool = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
    return pool


class MixedFusion_Block0(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(MixedFusion_Block0, self).__init__()

        # self.layer1 = nn.Sequential(nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, stride=1, padding=1),
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


class Net(nn.Module):
    # def __init__(self, input_shape, num_residual_blocks):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # 输入为灰度图
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5)  #输入为RGB 3通道图像
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5)
        self.res_block_1 = ResidualBlock(32)
        self.res_block_2 = ResidualBlock(64)
        self.res_block_3 = ResidualBlock(128)
        self.res_block_4 = ResidualBlock(256)
        self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(15488, 2)  # 1*100*100/3*100*100
        # self.fc1 = nn.Linear(33088, 2)  # concat(a, b, 1)  200*100
        # self.fc1 = nn.Linear(244000, 2)  # concat(a, b, 1)  512*256
        # self.fc1 = nn.Linear(119072, 2)  # 1*256*256
        self.fc1 = nn.Linear(36864, 2)  # SFB融合
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(num_features, num_classes)

        self.dropout = nn.Dropout(0.1)

        self.out_dim = 32

        self.down_fu_1 = MixedFusion_Block0(self.out_dim, self.out_dim * 2, nn.LeakyReLU(0.2, inplace=True))
        self.pool_fu_1 = maxpool()

        self.down_fu_2 = MixedFusion_Block_expand(self.out_dim * 2, self.out_dim * 4, nn.LeakyReLU(0.2, inplace=True))
        self.pool_fu_2 = maxpool2()

        self.down_fu_3 = MixedFusion_Block(self.out_dim * 4, self.out_dim * 8, nn.LeakyReLU(0.2, inplace=True))
        self.pool_fu_3 = maxpool()

        self.down_fu_4 = MixedFusion_Block_expand(self.out_dim * 8, self.out_dim * 8, nn.LeakyReLU(0.2, inplace=True))

        # down 4th layer
        self.down_fu_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 8, out_channels=self.out_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 8), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x1, x2):
    # def forward(self, input):
    #     x1 = input[:, 0:1, :, :]
    #     x2 = input[:, 1:2, :, :]

        in_size = x1.size(0)

        x1_1 = F.max_pool2d(F.relu(self.conv1(x1)), 2)
        x1_1 = self.res_block_1(x1_1)
        x1_2 = F.max_pool2d(F.relu(self.conv2(x1_1)), 2)
        x1_2 = self.res_block_2(x1_2)
        x1_3 = F.max_pool2d(F.relu(self.conv3(x1_2)), 2)
        x1_3 = self.res_block_3(x1_3)
        x1_4 = F.max_pool2d(F.relu(self.conv4(x1_3)), 2)
        x1_4 = self.res_block_4(x1_4)

        x2_1 = F.max_pool2d(F.relu(self.conv1(x2)), 2)
        x2_1 = self.res_block_1(x2_1)
        x2_2 = F.max_pool2d(F.relu(self.conv2(x2_1)), 2)
        x2_2 = self.res_block_2(x2_2)
        x2_3 = F.max_pool2d(F.relu(self.conv3(x2_2)), 2)
        x2_3 = self.res_block_3(x2_3)
        x2_4 = F.max_pool2d(F.relu(self.conv4(x2_3)), 2)
        x2_4 = self.res_block_4(x2_4)

        # print('x1_1.shape', x1_1.shape)
        # print('x2_1.shape', x2_1.shape)
        down_fu_1 = self.down_fu_1(x1_1, x2_1)
        down_fu_1m = self.pool_fu_1(down_fu_1)

        # print('x1_2.shape', x1_2.shape)
        # print('x2_2.shape', x2_2.shape)
        # print('down_fu_1m', down_fu_1m.shape)
        down_fu_2 = self.down_fu_2(x1_2, x2_2, down_fu_1m)
        down_fu_2m = self.pool_fu_2(down_fu_2)

        # print('x1_3', x1_3.shape)
        # print('x2_3', x2_3.shape)
        # print('down_fu_2m', down_fu_2m.shape)
        down_fu_3 = self.down_fu_3(x1_3, x2_3, down_fu_2m)
        down_fu_3m = self.pool_fu_3(down_fu_3)

        # print('x1_4', x1_4.shape)
        # print('x2_4', x2_4.shape)
        # print('down_fu_3m', down_fu_3m.shape)
        # print('down_fu_3', down_fu_3.shape)
        down_fu_4 = self.down_fu_4(x1_4, x2_4, down_fu_3m)
        down_fu_5 = self.down_fu_5(down_fu_4)

        # x_fu = self.avgpool(down_fu_2m)
        # x_fu = torch.flatten(x_fu, start_dim=1)
        # x_fu = self.fc1(x_fu)
        #
        # return x_fu

        down_fu_5 = down_fu_5.view(in_size, -1)
        down_fu_5 = self.fc1(down_fu_5)
        return F.log_softmax(down_fu_5, dim=1)


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

    # model_10 = torch.load("./conv4-49/net3_29.pkl", map_location='cpu')
    # print('model_10 ok')
    model_8 = torch.load("./conv4-34/net0_29.pkl", map_location='cpu')
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
    getResult()
    # computeIndex()