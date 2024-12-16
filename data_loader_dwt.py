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


def pic_loader(path, IorM='rgb'):
    # if IorM == 'rgb':
    #     return Image.open(path)
    # else:
    return Image.open(path).convert('L')


def loadSubjectData(k, TrainOrTest):

    # dirroot = r"D:\PythonProgram\PyTorch-GAN\chest_data"  # cpu
    dirroot = r"../chest_data"  # GPU
    if TrainOrTest == 1:
        Txt = "s_train" + str(k) + ".txt"
    elif TrainOrTest == 0:
        Txt = "s_test" + str(k) + ".txt"
    else:
        Txt = "valid.txt"

    # Train_txt = 's_train0000.txt'
    txt_file = os.path.join(dirroot, Txt)
    fh = open(txt_file, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()

        '''
        # window10 上运行
        imgs.append((words[0], words[1]))
        '''


        # ubuntu 上运行需要修改路径的表示形式
        num = words[0].split('\\')
        f = num[0] + '/' + num[1] + '/' + num[2]
        imgs.append((f, words[1]))



    # print('imgs', imgs)

    # crop 160*180 images
    # img_t1 = img_t1[40:200, 20:200]
    # img_t1ce = img_t1ce[40:200, 20:200]
    # img_t2 = img_t2[40:200, 20:200]
    # img_flair = img_flair[40:200, 20:200]
    return imgs


def loadChestData(imgs, index, TrainOrTest):
    # dirroot = r"D:\PythonProgram\PyTorch-GAN\chest_data"  # cpu
    dirroot = r"../chest_data"  # GPU
    f, label = imgs[index]
    slice = os.path.split(f)[1]
    # print('f', f)  # 0\0panfuxia3406505\76
    # print('slice', slice)  # 76
    fn = os.path.join(dirroot, f)
    prefix_name = os.path.split(fn)[1]
    # print('fn', fn)  # D:\PythonProgram\PyTorch-GAN\chest_data\0\0panfuxia3406505\76
    # print('prefix_name', prefix_name)  # 76

    if label == '0':
        label = [0]
    else:
        label = [1]


    # 读取未裁减后的图片进行测试
    picDCE1_name = prefix_name + "_4.png"
    picDCE2_name = prefix_name + "_6.png"
    picDWI_name = "DWI_" + prefix_name + ".png"


    image_1 = pic_loader(os.path.join(fn, picDCE1_name))
    image_DWI = pic_loader(os.path.join(fn, picDCE2_name))
    image_2 = pic_loader(os.path.join(fn, picDWI_name))


    # 读取裁减的图片进行测试
    # image_1 = pic_loader(os.path.join(fn, 'cut_shrink.png'))
    # image_2 = pic_loader(os.path.join(fn, 'cut_DWI.png'))
    # image_DWI = pic_loader(os.path.join(fn, picDCE2_name))


    Resize_my = transforms.Resize((256, 256))  # resize的参数顺序是h, w
    image_1 = Resize_my(image_1)
    image_2 = Resize_my(image_2)
    image_DWI = Resize_my(image_DWI)

    my_degrees = random.uniform(0, 10)
    my_HorizontalFlip = transforms.RandomHorizontalFlip(p=2)
    my_RandomAffine = transforms.RandomAffine(degrees=my_degrees)
    my_ColorJitter = transforms.ColorJitter(brightness=0.1)
    my_norm = transforms.Normalize((0.4914,), (0.2023,))
    my_toTensor = transforms.ToTensor()

    if TrainOrTest == 1:  # train时进行数据增强操作
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

    img1 = np.array(image_1)
    img2 = np.array(image_2)
    imgDWI = np.array(image_DWI)
    image_c = np.array(image_c)


    # print(img1)
    # print(np.where(np.max(img1)))
    # print(np.where(np.min(img1)))

    # img1 = MaxAbsScaler().fit_transform(img1)  # 将数组中的值归一化至(-1,1)
    # img2 = MaxAbsScaler().fit_transform(img2)
    # imgDWI = MaxAbsScaler().fit_transform(imgDWI)

    img1 = torch.FloatTensor(img1)
    img2 = torch.FloatTensor(img2)
    imgDWI = torch.FloatTensor(imgDWI)
    # image_c = my_toTensor(image_c)
    image_c = torch.FloatTensor(image_c)

    img1 = torch.unsqueeze(img1, dim=0)
    img2 = torch.unsqueeze(img2, dim=0)
    imgDWI = torch.unsqueeze(imgDWI, dim=0)
    image_c = torch.unsqueeze(image_c, dim=0)

    img1 = my_norm(img1)
    img2 = my_norm(img2)
    imgDWI = my_norm(imgDWI)
    image_c = my_norm(image_c)

    # return img1, image_c, label
    return img1, img2, image_c, label


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
            img_1, img_2, img_3, label = loadChestData(self.imgs, index, TrainOrTest=1)
            # img_1, img_2, label = loadChestData(self.imgs, index, TrainOrTest=1)
        else:
            img_1, img_2, img_3, label = loadChestData(self.imgs, index, TrainOrTest=0)  # 尺寸均为160*180 矩阵
            # img_1, img_2, label = loadChestData(self.imgs, index, TrainOrTest=0)  # 尺寸均为160*180 矩阵


        # split into patches (128*128)
        # img_1_patches = generate_all_2D_patches(img_1)
        # img_2_patches = generate_all_2D_patches(img_2)
        # img_DWI_patches = generate_all_2D_patches(img_DWI)

        # return img_1_patches, img_2_patches, img_DWI_patches, label
        # return img_1, img_2, label
        return img_1, img_2, img_3, label

    def __len__(self):
        return len(self.imgs)


