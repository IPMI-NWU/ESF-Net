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

torch.cuda.set_device(1)
# 这个是使用argparse模块时的必备行,将参数进行关联
args = parser.parse_args()
# 这个是在确认是否使用GPU的参数
args.cuda = not args.no_cuda and torch.cuda.is_available()

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def croppCenter(tensorToCrop, finalShape):
    org_shape = tensorToCrop.shape

    diff = np.zeros(3)
    diff[0] = org_shape[2] - finalShape[2]
    diff[1] = org_shape[3] - finalShape[3]

    croppBorders = np.zeros(2, dtype=int)
    croppBorders[0] = int(diff[0] / 2)
    croppBorders[1] = int(diff[1] / 2)

    return tensorToCrop[:,
           :,
           croppBorders[0]:org_shape[2] - croppBorders[0],
           croppBorders[1]:org_shape[3] - croppBorders[1]]


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


print('获取测试数据和验证数据')
# X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, random_state=666)

phase_test = False
test_data = MultiModalityData_load(args, train=False, test=True, k=1)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

print('获取模型')

Y_pred = []
Y_valid = []
for ii, (image1, image2, labels) in enumerate(test_loader):
    image1 = image1.cuda()
    image2 = image2.cuda()
    labels = labels[0].cuda()

    with torch.no_grad():
    # with torch.set_grad_enabled(False):
        model = torch.load("./conv4-62/net1_29.pkl", map_location='cpu')
        model.eval()
        model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        outputs = model(image1, image2)  # outputs为得分情况
        outputs = torch.sigmoid_(outputs)
        # print("outputs", outputs)
        output_np = outputs.cpu().numpy()
        label_np = labels.cpu().numpy()
        Y_pred_batch = output_np[:, -1]  # 预测为正例的得分情况
        # print(Y_pred_batch)
        for i in range(len(Y_pred_batch)):
            Y_pred.append(Y_pred_batch[i])
        for i in range(len(label_np)):
            Y_valid.append(label_np[i])

print("Y_pred", Y_pred)
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

print(len(Y_valid))
print(len(Y_pred))
fpr, tpr, thresholds_keras = roc_curve(np.array(Y_valid), np.array(Y_pred), pos_label=1)
auc = auc(fpr, tpr)
print("AUC : ", auc)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig("ROC_2分类.png")
# plt.show()

print("--- %s seconds ---" % (time.time() - start_time))