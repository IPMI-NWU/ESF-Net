
def computeIndex():
    fileA = 's_test0_GPU.txt'
    fileB = 's_test0_result.txt'
    fhA = open(fileA, 'r')
    fhB = open(fileB, 'r')
    imgs = []

    for line in fhB:
        result = line.split('w')

    for line in fhA:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        pre_index = words[0].split('\\')
        # window10 上运行
        imgs.append(words[1])

    # 计算TP TN FP FN
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(imgs)):
        if result[i] == imgs[i] == '1':
            TP = TP + 1
        elif result[i] == imgs[i] == '0':
            TN = TN + 1
        elif result[i] == '1' and imgs[i] == '0':
            FP = FP + 1
        elif result[i] == '0' and imgs[i] == '1':
            FN = FN + 1

    print(str(TP) + "  " + str(FN) + "  " + str(FP) + "  " + str(TN))


# 将读取到的文件名按照其中的某个数字排序
def compare1(sf1, sf2):
    words1 = sf1[0].split('_')
    words2 = sf2[0].split('_')
    if int(words1[1]) > int(words2[1]):
        return 1
    elif int(words1[1]) == int(words2[1]):
        return 0
    else:
        return -1


if __name__ == '__main__':
    import functools
    computeIndex()
    list = [('0_285_1.png', 'A', 10), ('0_16_1.png', 'A', 9), ('0_28_0.png', 'B', 9), ('0_205_1.png', 'B', 13)]
    # list1 = ['2_285_1.png', '3_16_1.png', '0_28_0.png', '0_205_1.png']
    list.sort(key=functools.cmp_to_key(compare1))
    print(list)