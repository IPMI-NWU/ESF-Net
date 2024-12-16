import os
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torch
import scipy.io as scio
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import MaxAbsScaler
import random
import pywt
import functools

def pic_loader(path, IorM='rgb'):
    # if IorM == 'rgb':
    #     return Image.open(path)
    # else:
    return Image.open(path).convert('L')


# 作用是将测试集按顺序输出，以便于后续计算病人层次的相关指标
def compare1(sf1, sf2):
    words1 = sf1[0].split('_')
    words2 = sf2[0].split('_')
    if int(words1[1]) > int(words2[1]):
        return 1
    elif int(words1[1]) == int(words2[1]):
        return 0
    else:
        return -1


def loadSubjectData(k, TrainOrTest):

    # dirroot = r"D:\PythonProgram\PyTorch-GAN\chest_data"  # cpu
    dirroot = r"../chest_data"  # GPU
    Train_dirroot = r'../implementations/cyclegan/images_epoch30_new_train/chest'  # GPU
    # Train_dirroot = r'../implementations/cyclegan/images_epoch30/chest'  # GPU
    Test_dirroot = r'../implementations/cyclegan/images_epoch30_new_test/chest'  # GPU
    Valid_dirroot = r'../implementations/cyclegan/images_epoch30_valid_result/chest'  # GPU
    # Valid_dirroot = r'../implementations/cyclegan/images_epoch30_test_result/chest'  # GPU
    if TrainOrTest == 1:  # 训练集取fake图像
        imgs = []
        path = Train_dirroot
        path_list = os.listdir(path)

        '''
        length = len(path_list)
        # number = len(path_list)/2
        number = 0
        '''

        for line in path_list:
            words = line.split('_')
            # if words[0] != 'data':  # 不分折
            # if words[0] == 'data' and words[1] == str(flag_num): # 对train中的数据分折训练
            if words[0] == str(k):  # 只取第一折数据
            # if words[0] == str(k) and number <= length/2:  # 只取第一折数据的一半
                pic_name = line
                label = words[-1].split('.')[0]
                # imgs.append((pic_name, label))
                imgs.append((pic_name, label, random.random()))
                # number = number + 1

        random.shuffle(imgs)
        length = int(len(imgs) / 2)
        imgs = imgs[:length]

        return imgs

    elif TrainOrTest == 0:  # 测试集fake图像
        imgs = []
        path = Test_dirroot
        path_list = os.listdir(path)

        # length = len(path_list)
        # number = 0

        for line in path_list:
            words = line.split('_')
            # if words[0] != 'data':  # 不分折
            # if words[0] == 'data' and words[1] == str(flag_num): # 对train中的数据分折训练
            if words[0] == str(k):  # 只取第一折数据
            # if words[0] == str(k) and number < length/2:  # 只取第一折数据
                pic_name = line
                label = words[-1].split('.')[0]
                imgs.append((pic_name, label))
                # imgs.append((pic_name, label, random.random()))
                # number = number + 1

        imgs.sort(key=functools.cmp_to_key(compare1))
        return imgs
    else:  # 验证集fake图像
        imgs = []
        path = Valid_dirroot
        path_list = os.listdir(path)
        for line in path_list:
            words = line.split('_')
            # if words[0] != 'data':  # 不分折
            # if words[0] == 'data' and words[1] == str(flag_num): # 对train中的数据分折训练
            if words[0] == str(0):  # 只取第一折数据
                pic_name = line
                label = words[-1].split('.')[0]
                # imgs.append((pic_name, label))
                imgs.append((pic_name, label, random.random()))

        return imgs


def loadChestData(imgs, index, TrainOrTest):

    if TrainOrTest == 1:  # 训练数据集##########################################################
        Train_dirroot = r'../implementations/cyclegan/images_epoch30_new_train/chest'  # GPU
        # Train_dirroot = r'../implementations/cyclegan/images_epoch30/chest'  # GPU
        Resize_train = transforms.Resize((256 * 4, 256))  # resize的参数顺序是h, w
        pic_name, label, R = imgs[index]
        image = pic_loader(os.path.join(Train_dirroot, pic_name))  # 512*256
        image = Resize_train(image)

        # if R < 0.8:
        #     # 大量数据为全真实图片
        #     box1 = (0, 256 * 2, 256, 256 * 3)
        #     box2 = (0, 256 * 0, 256, 256 * 1)
        # else:
        #     box1 = (0, 256 * 1, 256, 256 * 2)
        #     box2 = (0, 256 * 0, 256, 256 * 1)

        # 全真实图片
        box1 = (0, 256 * 0, 256, 256 * 1)
        box2 = (0, 256 * 2, 256, 256 * 3)
        #
        # 带生成图像
        # box1 = (0, 256 * 2, 256, 256 * 3)  # real_dce
        # box2 = (0, 256 * 3, 256, 256 * 4)  # fake_dwi

        # box1 = (0, 256 * 1, 256, 256 * 2)  # fake_dce
        # box2 = (0, 256 * 0, 256, 256 * 1)  # real_dwi

        # 全用real_dec
        # box1 = (0, 256 * 2, 256, 256 * 3)  # real_dce
        # box2 = (0, 256 * 2, 256, 256 * 3)  # real_dce

        # 全用real_dwi
        # box1 = (0, 256 * 0, 256, 256 * 1)
        # box2 = (0, 256 * 0, 256, 256 * 1)

        image_1 = image.crop(box1)  # real_shrink
        image_2 = image.crop(box2)  # real_DWI

        if label == '0':
            label = [0]
        else:
            label = [1]

        my_degrees = random.uniform(0, 10)
        my_HorizontalFlip = transforms.RandomHorizontalFlip(p=2)  # 依概率p垂直翻转
        my_RandomAffine = transforms.RandomAffine(degrees=my_degrees)  # 仿射变换
        my_ColorJitter = transforms.ColorJitter(brightness=0.1)  # 修改亮度

        my_norm = transforms.Normalize((0.4914,), (0.2023,))
        my_toTensor = transforms.ToTensor()

        for i in range(10):
            if random.random() >= 0.5:  # 水平翻转
                image_1 = my_HorizontalFlip(image_1)
                image_2 = my_HorizontalFlip(image_2)
            if random.random() >= 0.6:  # 仿射变换
                image_1 = my_RandomAffine(image_1)
                image_2 = my_RandomAffine(image_2)
            if random.random() >= 0.6:  # 对比度变换
                image_1 = my_ColorJitter(image_1)
                image_2 = my_ColorJitter(image_2)

        image_A_cA, image_A_cD = pywt.dwt(image_1, 'db2')
        image_B_cA, image_B_cD = pywt.dwt(image_2, 'db2')
        # print('image_A_cA', image_A_cA) # (100, 51)
        # print('image_A_cD', image_A_cD) # (100, 51)
        h, v = image_B_cA.shape
        image_c_cA = np.zeros((h, v))
        image_c_cD = np.zeros((h, v))
        for i in range(h):
            for j in range(v):
                image_c_cA[i][j] = (image_A_cA[i][j] + image_B_cA[i][j]) / 2
                image_c_cD[i][j] = image_A_cD[i][j] if image_A_cD[i][j] >= image_B_cD[i][j] else image_B_cD[i][j]
        # print('image_c_cA', image_c_cA)
        # print('image_c_cD', image_c_cD)
        image_c = pywt.idwt(image_c_cA, image_c_cD, 'db2')

        image_blend = Image.blend(image_1, image_2, 0.5)

        img1 = np.array(image_1)
        img2 = np.array(image_2)
        img_empty = np.zeros((256, 256))

        img1 = my_toTensor(img1)
        img2 = my_toTensor(img2)
        imgc = my_toTensor(image_c)
        img_empty = my_toTensor(img_empty)
        image_blend = my_toTensor(image_blend)
        image_concat = torch.cat([img1, img1])  # 图像的通道数加一 3*100*100
        img1 = my_norm(img1)
        img2 = my_norm(img2)
        image_concat = my_norm(image_concat)
        imgc = my_norm(imgc)
        image_blend = my_norm(image_blend)
        img_empty = my_norm(img_empty)

        return img2, label
        # return image_concat, label
        # return imgc, label
        # return image_blend, label
        # return img1, img2, label

    elif TrainOrTest == 0:  # 测试数据集#######################################################
        Test_dirroot = r'../implementations/cyclegan/images_epoch30_new_test/chest'  # GPU
        Resize_train = transforms.Resize((256 * 4, 256))  # resize的参数顺序是h, w
        pic_name, label = imgs[index]
        image = pic_loader(os.path.join(Test_dirroot, pic_name))  # 512*256
        image = Resize_train(image)

        # 全真实图片
        box1 = (0, 256 * 0, 256, 256 * 1)
        box2 = (0, 256 * 2, 256, 256 * 3)

        # 带生成图像
        # box1 = (0, 256 * 2, 256, 256 * 3)
        # box2 = (0, 256 * 3, 256, 256 * 4)

        # box1 = (0, 256 * 1, 256, 256 * 2)  # fake_dce
        # box2 = (0, 256 * 0, 256, 256 * 1)  # real_dwi

        # 全用real_dec
        # box1 = (0, 256 * 2, 256, 256 * 3)  # real_dce
        # box2 = (0, 256 * 2, 256, 256 * 3)  # real_dce

        # 全用real_dwi
        # box1 = (0, 256 * 0, 256, 256 * 1)
        # box2 = (0, 256 * 0, 256, 256 * 1)

        image_1 = image.crop(box1)  # real_shrink
        image_2 = image.crop(box2)  # real_DWI

        if label == '0':
            label = [0]
        else:
            label = [1]

        my_norm = transforms.Normalize((0.4914,), (0.2023,))
        my_toTensor = transforms.ToTensor()

        image_A_cA, image_A_cD = pywt.dwt(image_1, 'db2')
        image_B_cA, image_B_cD = pywt.dwt(image_2, 'db2')
        # print('image_A_cA', image_A_cA) # (100, 51)
        # print('image_A_cD', image_A_cD) # (100, 51)
        h, v = image_B_cA.shape
        image_c_cA = np.zeros((h, v))
        image_c_cD = np.zeros((h, v))
        for i in range(h):
            for j in range(v):
                image_c_cA[i][j] = (image_A_cA[i][j] + image_B_cA[i][j]) / 2
                image_c_cD[i][j] = image_A_cD[i][j] if image_A_cD[i][j] >= image_B_cD[i][j] else image_B_cD[i][j]
        # print('image_c_cA', image_c_cA)
        # print('image_c_cD', image_c_cD)
        image_c = pywt.idwt(image_c_cA, image_c_cD, 'db2')

        image_blend = Image.blend(image_1, image_2, 0.5)

        img1 = np.array(image_1)
        img2 = np.array(image_2)
        img_empty = np.zeros((256, 256))

        img1 = my_toTensor(img1)
        img2 = my_toTensor(img2)
        image_concat = torch.cat([img1, img2])  # 图像的通道数加一 3*100*100
        imgc = my_toTensor(image_c)
        image_blend = my_toTensor(image_blend)
        img_empty = my_toTensor(img_empty)

        img1 = my_norm(img1)
        img2 = my_norm(img2)
        image_concat = my_norm(image_concat)
        imgc = my_norm(imgc)
        image_blend = my_norm(image_blend)
        img_empty = my_norm(img_empty)

        return img2, label
        # return image_concat, label
        # return imgc, label
        # return image_blend, label
        # return img2, img_empty, label

    else:  # 验证数据集#######################################################
        Valid_dirroot = r'../implementations/cyclegan/images_epoch30_valid_result/chest'  # GPU
        # Valid_dirroot = r'../implementations/cyclegan/images_epoch30_test_result/chest'  # GPU
        Resize_train = transforms.Resize((256 * 4, 256))  # resize的参数顺序是h, w
        pic_name, label = imgs[index]
        image = pic_loader(os.path.join(Valid_dirroot, pic_name))  # 512*256
        image = Resize_train(image)

        # 全真实图片
        box1 = (0, 256 * 0, 256, 256 * 1)
        box2 = (0, 256 * 2, 256, 256 * 3)

        # 带生成图像
        # box1 = (0, 256 * 2, 256, 256 * 3)
        # box2 = (0, 256 * 3, 256, 256 * 4)

        # box1 = (0, 256 * 1, 256, 256 * 2)  # fake_dce
        # box2 = (0, 256 * 0, 256, 256 * 1)  # real_dwi

        # 全用real_dec
        # box1 = (0, 256 * 2, 256, 256 * 3)  # real_dce
        # box2 = (0, 256 * 2, 256, 256 * 3)  # real_dce

        # 全用real_dwi
        # box1 = (0, 256 * 0, 256, 256 * 1)
        # box2 = (0, 256 * 0, 256, 256 * 1)

        image_1 = image.crop(box1)  # real_shrink
        image_2 = image.crop(box2)  # real_DWI

        if label == '0':
            label = [0]
        else:
            label = [1]

        my_norm = transforms.Normalize((0.4914,), (0.2023,))
        my_toTensor = transforms.ToTensor()

        image_A_cA, image_A_cD = pywt.dwt(image_1, 'db2')
        image_B_cA, image_B_cD = pywt.dwt(image_2, 'db2')
        # print('image_A_cA', image_A_cA) # (100, 51)
        # print('image_A_cD', image_A_cD) # (100, 51)
        h, v = image_B_cA.shape
        image_c_cA = np.zeros((h, v))
        image_c_cD = np.zeros((h, v))
        for i in range(h):
            for j in range(v):
                image_c_cA[i][j] = (image_A_cA[i][j] + image_B_cA[i][j]) / 2
                image_c_cD[i][j] = image_A_cD[i][j] if image_A_cD[i][j] >= image_B_cD[i][j] else image_B_cD[i][j]
        # print('image_c_cA', image_c_cA)
        # print('image_c_cD', image_c_cD)
        image_c = pywt.idwt(image_c_cA, image_c_cD, 'db2')

        image_blend = Image.blend(image_1, image_2, 0.5)

        img1 = np.array(image_1)
        img2 = np.array(image_2)
        img_empty = np.zeros((256, 256))

        img1 = my_toTensor(img1)
        img2 = my_toTensor(img2)
        image_concat = torch.cat([img1, img2])  # 图像的通道数加一 3*100*100
        imgc = my_toTensor(image_c)
        image_blend = my_toTensor(image_blend)
        img_empty = my_toTensor(img_empty)

        img1 = my_norm(img1)
        img2 = my_norm(img2)
        image_concat = my_norm(image_concat)
        imgc = my_norm(imgc)
        image_blend = my_norm(image_blend)
        img_empty = my_norm(img_empty)

        return img2, label
        # return image_concat, label
        # return imgc, label
        # return image_blend, label
        # return img2, img_empty, label


class MultiModalityData_load(data.Dataset):

    def __init__(self, opt, transforms=None, train=True, test=False, k=0):

        self.opt = opt
        self.test = test
        self.train = train
        if self.train:
            self.imgs = loadSubjectData(k, TrainOrTest=1)  # 读取train0.txt
        if self.test:
            self.imgs = loadSubjectData(k, TrainOrTest=0)  # 读取test0.txt
        if self.train == False and self.test == False:
            self.imgs = loadSubjectData(k, TrainOrTest=-1)  # 读取valid.txt

        '''
        if self.test:
            path_test = opt.data_path + 'test/'
            data_paths = [os.path.join(path_test, i) for i in os.listdir(path_test)]

        if self.train:
            path_train = opt.data_path + 'train/'
            data_paths = [os.path.join(path_train, i) for i in os.listdir(path_train)]

        data_paths = sorted(data_paths, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.data_paths = np.array(data_paths)
        '''

    def __getitem__(self, index):

        # path
        # cur_path = self.data_paths[index]

        # get images
        # img_t1, img_t1ce, img_t2, img_flair = loadSubjectData(cur_path)  # 尺寸均为160*180 矩阵
        # imgs = loadSubjectData(self.imgs)
        # print('imgs', len(self.imgs))
        if self.train:
            # img_1, img_2, label = loadChestData(self.imgs, index, TrainOrTest=1)
            img, label = loadChestData(self.imgs, index, TrainOrTest=1)
        elif self.test:
            # img_1, img_2, label = loadChestData(self.imgs, index, TrainOrTest=0)  # 尺寸均为160*180 矩阵
            img, label = loadChestData(self.imgs, index, TrainOrTest=0)  # 尺寸均为160*180 矩阵
        else:
            # img_1, img_2, label = loadChestData(self.imgs, index, TrainOrTest=-1)  # 尺寸均为160*180 矩阵
            img, label = loadChestData(self.imgs, index, TrainOrTest=-1)  # 尺寸均为160*180 矩阵

        return img, label
        # return img_1, img_2, label

    def __len__(self):
        return len(self.imgs)


