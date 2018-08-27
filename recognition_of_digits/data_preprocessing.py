import numpy as np


# 查看数据信息
def show_data(filename):
    #打开文件
    file = open(filename)
    # 按行读取文件
    lines = file.readlines()
    # 计算文件的行数
    count = len(lines)
    # 初始化数据数组
    data = np.zeros((count,8,8),dtype=np.int)

    for line in lines:
        temp = line.split(",")
        i = 0
        for j in range(8):
            for k in range(8):
                data[i][j][k] = temp[k + 8 * j]
                print(data[i][j][k],end="")
            print("")
        i += 1
    return data

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i * j] = int(lineStr[j])

    return returnVect


