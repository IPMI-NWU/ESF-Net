import warnings, os, copy, time
from tqdm import tqdm

warnings.filterwarnings('ignore')
from typing import ClassVar, Optional
from torchsummary import summary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms, utils
import argparse
# from data_loader_fusion import *
# from data_loader_fake import *
from data_loader_fake_fake_fake import *
from torch.utils.data import DataLoader


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

torch.cuda.set_device(3)
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


class BasicBlock(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(self, inplanes: int, planes: int,
                 stride: Optional[int] = 1,
                 downsample: Optional[bool] = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity: torch.Tensor = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


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

        self.down_fu_2 = MixedFusion_Block(self.out_dim * 2, self.out_dim * 4, nn.LeakyReLU(0.2, inplace=True))
        self.pool_fu_2 = maxpool()

        self.down_fu_3 = MixedFusion_Block(self.out_dim * 4, self.out_dim * 8, nn.LeakyReLU(0.2, inplace=True))
        self.pool_fu_3 = maxpool()

        self.down_fu_4 = MixedFusion_Block(self.out_dim * 8, self.out_dim * 8, nn.LeakyReLU(0.2, inplace=True))

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

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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


# model = ResNet(BasicBlock, [2, 2, 2, 2])
# print('model_summary', summary(model, input_size=(1, 100, 100), batch_size=10, device='cpu'))

# 读取之前保存的模型
model = torch.load('net16.pkl')

# 判断是否调用GPU模式
if args.cuda:
    model.cuda()
# 初始化优化器 model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def test(phase):
    ############################ 测 试 网 络 ###################################
    # Load data

    phase_test = False
    test_data = MultiModalityData_load(args, train=False, test=True, k=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    best_acc: float = 0.0

    # print(f'Epoch {epoch + 1}/{args.epochs}')
    # print('-' * 10)
    model.eval()

    test_loss: float = 0.0
    test_corrects: int = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for ii, (image1, image2, labels) in enumerate(test_loader):
        # images = images.cuda()
        image1 = image1.cuda()
        image2 = image2.cuda()
        labels = labels[0].cuda()


        with torch.set_grad_enabled(phase_test == 'train'):
            outputs = model(image1, image2)

            _, pred_labels = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        pre_np = pred_labels.cpu().numpy()
        lab_np = labels.cpu().numpy()
        for i in range(len(pre_np)):
            if pre_np[i] == lab_np[i] == 1:
                TP = TP + 1
            elif pre_np[i] == lab_np[i] == 0:
                TN = TN + 1
            elif pre_np[i] == 1 and lab_np[i] == 0:
                FP = FP + 1
            elif pre_np[i] == 0 and lab_np[i] == 1:
                FN = FN + 1

        test_loss += loss.item() * image1.size(0)
        test_corrects += torch.sum(pred_labels == labels.data)

        print('labels', labels)
        print('pred_labels', pred_labels)

        # train_acc = running_corrects / ((ii+1) * args.batch_size)

        # train_acc = torch.floor_divide(running_corrects, ((ii+1) * args.batch_size))
        Txt = open("./test_log.txt", "a")
        Txt.write('  Loss: ' + str(loss.item()) + '\n')
        Txt.close()

        # print('loss', loss.item())
        # print('running_loss', running_loss)
        # print('running_corretcs', running_corrects)
        # print('train_ACC', running_corrects / ((ii+1) * args.batch_size))

    epoch_test_loss: float = test_loss / len(test_data)
    epoch_test_acc: float = test_corrects.double() / len(test_data)

    print(f'{phase_test} Loss: {epoch_test_loss:.4f} Acc: {epoch_test_acc:.4f}')

    my_precision = TP / (TP + FP)
    my_recall = TP / (TP + FN)
    my_f1 = (2 * my_recall * my_precision) / (my_precision + my_recall)

    Txt = open("./test_log.txt", "a")
    Txt.write(
        '\n' + '  Loss: ' + str(epoch_test_loss) + '   Acc: ' + str(epoch_test_acc)
        + '   Precision: ' + str(my_precision) + '   Recall: ' + str(my_recall) + '   F1-score: ' + str(my_f1)
        + '\n' + '\n')
    Txt.write('TP: ' + str(TP) + '  FN: ' + str(FN) + '  FP: ' + str(FP) + '  TN: ' + str(TN) + '\n')
    Txt.close()

    print('Finished test!')


if __name__ == '__main__':
    test(phase='train')
