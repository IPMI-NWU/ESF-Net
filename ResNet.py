from __future__ import print_function
# 使得我们能够手动输入命令行参数
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
import torch.utils.data as Data
import numpy as np

from data_loader_fusion import *
from torch.utils.data import DataLoader

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


torch.cuda.set_device(0)
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


class Net(nn.Module):
    # def __init__(self, input_shape, num_residual_blocks):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=5)  # 输入为灰度图
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5)  #输入为RGB 3通道图像
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.res_block_1 = ResidualBlock(16)
        self.res_block_2 = ResidualBlock(32)
        self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(15488, 2)  # 1*100*100/3*100*100
        # self.fc1 = nn.Linear(33088, 2)  # concat(a, b, 1)  200*100
        # self.fc1 = nn.Linear(244000, 2)  # concat(a, b, 1)  512*256
        self.fc1 = nn.Linear(119072, 2)  # 1*256*256


    def forward(self, x):
        in_size = x.size(0)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.res_block_1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.res_block_2(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def test(model, zhe):
    # Test data loader
    # dirroot = r'../chest_data'
    # train_path = os.path.join(dirroot, 's_test0.txt')
    # train_path = os.path.join(dirroot, Test_txt)
    # pdb.set_trace()  # 程序运行到这里就会暂停。
    # test_datasets = MyDataset_test(dirroot, train_path, mode='test', wName=True)
    # test_loader = Data.DataLoader(dataset=test_datasets, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    test_data = MultiModalityData_load(args, train=False, test=True, k=zhe)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # 设置为test模式
    model.eval()
    # 初始化测试损失值为0
    test_loss = 0
    # 初始化预测正确的数据个数为0
    correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    SUM = 0

    for data, target in test_loader:
        # if args.cuda:
        #     data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        # output = model(data)

        data = data.cuda()
        data = data.type(torch.cuda.FloatTensor)
        target = target[0].cuda()

        with torch.no_grad():
            output = model(data.to(torch.float32))
            # 把所有loss值进行累加
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # 获取最大对数概率值的索引
            pred = output.data.max(1, keepdim=True)[1]
            # 对预测正确的个数进行累加
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        target1 = target.cpu()
        target1 = target1.numpy()
        pred1 = pred.cpu()
        # pred1 = torch.squeeze(pred1)
        pred1 = pred1.numpy()

        for i in range(len(pred1)):
            if target1[i] == 1 and pred1[i] == 1:
                TP = TP + 1
                SUM = SUM + 1
            elif target1[i] == 0 and pred1[i] == 0:
                TN = TN + 1
                SUM = SUM + 1
            elif target1[i] == 1 and pred1[i] == 0:
                FN = FN + 1
                SUM = SUM + 1
            elif target1[i] == 0 and pred1[i] == 1:
                FP = FP + 1
                SUM = SUM + 1

        # 混淆矩阵
        r = confusion_matrix(target1, pred1)
        # print("混淆矩阵为:", r)
        # 利用混淆矩阵求准确率
        acc = accuracy_score(target1, pred1)
        # print("准确率为:", acc)
        # precision = precision_score(target1, pred1, pos_label=1, zero_division=0)
        precision = precision_score(target1, pred1, pos_label=1)
        # print("精确率为:", precision)  # 反映了被分类器判定的正例中真正的正例样本的比重
        # recall = recall_score(target1, pred1, pos_label=1, zero_division=0)
        recall = recall_score(target1, pred1, pos_label=1)
        # print("召回率为:", recall)  # 反映了被正确判定的正例占总的正例的比重

        # Txt1 = open("./0812result_output.txt", "a")
        # Txt1.write('Test Set: \n')
        # Txt1.write('output:   ' + str(target1) + '\n')
        # Txt1.write('prediction:   ' + str(pred1) + '\n')
        # Txt1.close()

    if TP + FP != 0:
        precision_my = TP / (TP + FP)
    else:
        precision_my = TP / 0.0001
    if TP + FN != 0:
        recall_my = TP / (TP + FN)
    else:
        recall_my = TP / 0.0001

        # 因为把所有loss值进行累加,所以最后要除以总的数据长度才能得到平均loss
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) '.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ), end='')
    print('Test_Presion: ' + str(precision_my) + ', Test_Recall: ' + str(recall_my))

    Txt = open("./test_log.txt", "a")
    Txt.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) '.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))
    Txt.write('Test_Presion: ' + str(precision_my) + ', Test_Recall: ' + str(recall_my) + '\n')
    Txt.write('TP: ' + str(TP) + ', TN: ' + str(TN) + ', FP: ' + str(FP) + ', FN: ' + str(FN) + '\n')
    Txt.close()


def train(zhe):
    """
    定义每个epoch的训练细节
    """

    model = Net()
    # 输出torch模型每一层的输出
    # print('model_summary', summary(model, input_size=(1, 256, 256), batch_size=64, device='cpu'))

    # 判断是否调用GPU模式
    if args.cuda:
        model.cuda()
    # 初始化优化器 model.train()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training data loader
    # dirroot = r'../chest_data'
    # train_txt_path = os.path.join(dirroot, Train_txt)

    # pdb.set_trace()  # 程序运行到这里就会暂停。
    # train_datasets = MyDataset_train(dirroot, train_txt_path, mode='train', wName=True)
    # train_loader = Data.DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_data = MultiModalityData_load(args, train=True, k=zhe)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    print('len(dadaloader)', len(train_loader))


    for epoch in range(args.epochs):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        SUM = 0

        # 设置为training模式
        model.train()

        for batch_idx, (data, target0) in enumerate(train_loader):
            # target_n = np.array(target0)
            # target = torch.from_numpy(target_n)
            # print('data', data.shape)  # 64*1*28*28
            # print('target.size', target.shape) # torch.Size([64])
            # 如果要调用GPU模式,就把数据转存到GPU
            # if args.cuda:
            #     data, target = data.cuda(), target.cuda()
            # data, target = Variable(data), Variable(target)

            data = data.cuda()
            data = data.type(torch.cuda.FloatTensor)
            target = target0[0].cuda()

            # 优化器梯度初始化为零
            optimizer.zero_grad()
            # output = model(data)
            output = model(data.to(torch.float32))
            # output = torch.tensor(output)
            output = torch.squeeze(output)

            # 输出训练的准确率
            train_correct = 0
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大对数概率值的索引
            train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的个数进行累加
            train_correct = 100. * train_correct / args.batch_size

            target1 = target.cpu()
            target1 = target1.numpy()
            pred1 = pred.cpu()
            # pred1 = torch.squeeze(pred1)
            pred1 = pred1.numpy()
            # print('target1', target1)
            # print('pred1', pred1)
            # print('len(pred1)', len(pred1))

            for i in range(len(pred1)):
                if target1[i] == 1 and pred1[i] == 1:
                    TP = TP + 1
                    SUM = SUM + 1
                elif target1[i] == 0 and pred1[i] == 0:
                    TN = TN + 1
                    SUM = SUM + 1
                elif target1[i] == 1 and pred1[i] == 0:
                    FN = FN + 1
                    SUM = SUM + 1
                elif target1[i] == 0 and pred1[i] == 1:
                    FP = FP + 1
                    SUM = SUM + 1

            r = confusion_matrix(target1, pred1)
            # print("混淆矩阵为:", r)
            # 利用混淆矩阵求准确率S
            acc = accuracy_score(target1, pred1)
            # print("准确率为:", acc)
            # print("准确率——my为:", acc_my)
            # precision = precision_score(target1, pred1, pos_label=1, zero_division=0)
            precision = precision_score(target1, pred1, pos_label=1)
            # print("精确率为:", precision)  # 反映了被分类器判定的正例中真正的正例样本的比重
            # print("精确率为_my:", precision_my)  # 反映了被分类器判定的正例中真正的正例样本的比重
            # recall = recall_score(target1, pred1, pos_label=1, zero_division=0)
            recall = recall_score(target1, pred1, pos_label=1)
            # print("召回率为:", recall)  # 反映了被正确判定的正例占总的正例的比重
            # print("召回率_my为:", recall_my)  # 反映了被正确判定的正例占总的正例的比重


            # 负对数似然函数损失
            # print('output', output)
            # print('output.shape', output.shape)
            # print('target', target)
            # print('target.shape', target.shape)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            Txt = open("./run_log.txt", "a")

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}  Accuracy:({:.0f}%) '.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), train_correct
                ), end='')
                print(' Presion: ' + str(precision) + '  Recall: ' + str(recall))
            Txt.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}  Accuracy:({:.0f}%) '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), train_correct
            ))
            Txt.write(' Precision: ' + str(precision) + '  Recall: ' + str(recall) + '\n')
            Txt.close()
            # Txt1 = open("./0812result_output.txt", "a")
            # Txt1.write('Train Epoch: {} [{}/{} ({:.0f}%)]\n'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
            #                                                         100. * batch_idx / len(train_loader)))
            # Txt1.write('output:   ' + str(target1) + '\n')
            # Txt1.write('prediction:   ' + str(pred1) + '\n')
            # Txt1.close()

        acc_my = (TP + TN) / SUM
        if TP + FP != 0:
            precision_my = TP / (TP + FP)
        else:
            precision_my = TP / 0.0001
        if TP + FN != 0:
            recall_my = TP / (TP + FN)
        else:
            recall_my = TP / 0.0001
        print('Train_acc = ', acc_my)
        print('Train_precision = ', precision_my)
        print('Train_recall = ', recall_my)

        Txt = open("./run_log.txt", "a")
        Txt.write('\n' + 'Train_Presion: ' + str(precision_my) + '  Train_Recall: ' + str(recall_my) + '\n')
        Txt.close()

        # 训练完后返回model
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model_name = "net" + str(zhe) + "_" + str(epoch) + ".pkl"
            # model_name_paras = "net" + str(epoch) + "_paras.pkl"
            torch.save(model, model_name)  # 保存整个神经网络到net1.pkl中
            # torch.save(model.state_dict(), model_name_paras)  # 保存网络里的参数到net1_paras.pkl中


    ##################训练完了测试网络############################
    test(model, zhe)



# 进行每个epoch的训练
if __name__ == '__main__':
    for i in range(4):
    # for i in [0, 3]:
        print('--------------------第' + str(i) + '折---------------------')
        train(zhe=i)
    print('Finished!')