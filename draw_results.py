# 要从line字符串中提取1.598912和2.104217两个数据
# line = 'step 0: dis loss 1.598912, gan loss 2.104217'
# temp = line.split('loss ')
# print(temp)
# t = temp[1].split(',')
# print(t[0])
# print(temp[2])


filename = 'log_epoch30_1.txt'

D_loss, G_loss, adv_loss, cycle_loss = [], [], [], []
SSIMA2B, SSIMB2A, PSNRA2B, PSNRB2A = [], [], [], []
step1, step2, step3, step4 = [], [], [], []
# 相比open(),with open()不用手动调用close()方法

# 第一折
with open(filename, 'r') as f:
    # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
    # 然后将每个元素中的不同信息提取出来
    lines = f.readlines()  # 下标从1开始
    # i变量，由于这个txt存储时有空行，所以增只读偶数行，主要看txt文件的格式，一般不需要
    # j用于判断读了多少条，step为画图的X轴
    i = 0
    j = 0
    w = 0
    print(lines[1])  # 单数为各种loss
    print(lines[2])  # 偶数为各种指标
    for line in lines:
        mark = line.split(' ')
        if mark[0] == '[Epoch':
            flag = 0
        elif mark[0] == '[SSIMa2b:':
            flag = 1
        else:
            flag = 2

        # if flag == 0 and (i + 1) % 4 == 0:
        if flag == 0 and i < 509100:
            t1 = line.split('D loss: ')
            dloss = t1[1].split(']')[0]
            D_loss.append(float(dloss))
            step1.append(j)
            j = j + 1
            # print(dloss)

            t2 = line.split('G loss: ')
            gloss = t2[1].split(',')[0]
            G_loss.append(float(gloss))
            # print(gloss)

            t3 = line.split('adv: ')
            adv = t3[1].split(',')[0]
            adv_loss.append(float(adv))
            # print(adv)

            t4 = line.split('cycle: ')
            cycle = t4[1].split(',')[0]
            cycle_loss.append(float(cycle))
            # print(cycle)
            i = i + 1

        # elif flag == 1 and i % 4 == 0:
        elif flag == 1 and i < 509100:
            t1 = line.split('SSIMa2b: ')
            ssim1 = t1[1].split(',')[0]
            SSIMA2B.append(float(ssim1))
            step2.append(w)
            w = w + 1
            # print(ssim1)

            t2 = line.split('SSIMb2a: ')
            ssim2 = t2[1].split(',')[0]
            SSIMB2A.append(float(ssim2))
            # print(ssim2)

            t3 = line.split('PSNRa2b: ')
            psnr1 = t3[1].split(',')[0]
            PSNRA2B.append(float(psnr1))
            # print(psnr1)

            t4 = line.split('PSNRb2a: ')
            psnr2 = t4[1].split(',')[0]
            PSNRB2A.append(float(psnr2))
            # print(psnr2)

            i = i + 1
        else:
            i = i + 1


print(max(SSIMA2B))
print(max(SSIMB2A))
print(max(PSNRA2B))
print(max(PSNRB2A))
'''
from matplotlib import pyplot as plt

fig = plt.figure()  # 创建绘图窗口，并设置窗口大小
# 画第一张图
ax1 = fig.add_subplot(221)  # 将画面分割为2行1列选第一个
ax1.plot(step2, SSIMA2B, 'red', label='loss')  # 画dis-loss的值，颜色红
ax1.legend(loc='upper right')  # 绘制图例，plot()中的label值
ax1.set_xlabel('step')  # 设置X轴名称
ax1.set_ylabel('G_LOSS')  # 设置Y轴名称
ax1.set_xticklabels(['0', '0', '15', '30', '45', '60', '75', '90', '105'], rotation=30, fontsize='small')

# 画第二张图
ax2 = fig.add_subplot(222)  # 将画面分割为2行1列选第二个
ax2.plot(step2, SSIMB2A, 'blue', label='loss')  # 画gan-loss的值，颜色蓝
ax2.legend(loc='upper right')  # loc为图例位置，设置在右上方，（右下方为lower right）
ax2.set_xlabel('step')
ax2.set_ylabel('D_LOSS')
ax2.set_xticklabels(['0', '0', '15', '30', '45', '60', '75', '90', '105'], rotation=30, fontsize='small')

# 画第三张图
ax3 = fig.add_subplot(223)  # 将画面分割为2行1列选第二个
ax3.plot(step2, PSNRA2B, 'orange', label='loss')  # 画gan-loss的值，颜色蓝
ax3.legend(loc='upper right')  # loc为图例位置，设置在右上方，（右下方为lower right）
ax3.set_xlabel('step')
ax3.set_ylabel('ADV_LOSS')
ax3.set_xticklabels(['0', '0', '15', '30', '45', '60', '75', '90', '105'], rotation=30, fontsize='small')

# 画第四张图
ax4 = fig.add_subplot(224)  # 将画面分割为2行1列选第二个
ax4.plot(step2, PSNRB2A, 'black', label='loss')  # 画gan-loss的值，颜色蓝
ax4.legend(loc='upper right')  # loc为图例位置，设置在右上方，（右下方为lower right）
ax4.set_xlabel('step')
ax4.set_ylabel('CYCLE_LOSS')
ax4.set_xticklabels(['0', '0', '15', '30', '45', '60', '75', '90', '105'], rotation=30, fontsize='small')

plt.show()  # 显示绘制的图
'''

'''
from matplotlib import pyplot as plt

fig = plt.figure()  # 创建绘图窗口，并设置窗口大小
# 画第一张图
ax1 = fig.add_subplot(221)  # 将画面分割为2行1列选第一个
ax1.plot(step1, G_loss, 'red', label='loss')  # 画dis-loss的值，颜色红
ax1.legend(loc='upper right')  # 绘制图例，plot()中的label值
ax1.set_xlabel('step')  # 设置X轴名称
ax1.set_ylabel('G_LOSS')  # 设置Y轴名称
ax1.set_xticklabels(['0', '0', '15', '30', '45', '60', '75', '90', '105'], rotation=30, fontsize='small')

# 画第二张图
ax2 = fig.add_subplot(222)  # 将画面分割为2行1列选第二个
ax2.plot(step1, D_loss, 'blue', label='loss')  # 画gan-loss的值，颜色蓝
ax2.legend(loc='upper right')  # loc为图例位置，设置在右上方，（右下方为lower right）
ax2.set_xlabel('step')
ax2.set_ylabel('D_LOSS')
ax2.set_xticklabels(['0', '0', '15', '30', '45', '60', '75', '90', '105'], rotation=30, fontsize='small')

# 画第三张图
ax3 = fig.add_subplot(223)  # 将画面分割为2行1列选第二个
ax3.plot(step1, adv_loss, 'orange', label='loss')  # 画gan-loss的值，颜色蓝
ax3.legend(loc='upper right')  # loc为图例位置，设置在右上方，（右下方为lower right）
ax3.set_xlabel('step')
ax3.set_ylabel('ADV_LOSS')
ax3.set_xticklabels(['0', '0', '15', '30', '45', '60', '75', '90', '105'], rotation=30, fontsize='small')

# 画第四张图
ax4 = fig.add_subplot(224)  # 将画面分割为2行1列选第二个
ax4.plot(step1, cycle_loss, 'black', label='loss')  # 画gan-loss的值，颜色蓝
ax4.legend(loc='upper right')  # loc为图例位置，设置在右上方，（右下方为lower right）
ax4.set_xlabel('step')
ax4.set_ylabel('CYCLE_LOSS')
ax4.set_xticklabels(['0', '0', '15', '30', '45', '60', '75', '90', '105'], rotation=30, fontsize='small')

plt.show()  # 显示绘制的图
'''