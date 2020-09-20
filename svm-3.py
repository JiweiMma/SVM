import matplotlib.pyplot as plt
import numpy as np

#读取数据
def loadDataSet(fileName):
    # dataMat - 数据矩阵
    # labelMat - 数据标签
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        # 逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        # 添加数据
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        # 添加标签
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#数据可视化
#dataMat - 数据矩阵
#labelMat - 数据标签
def showDataSet(dataMat, labelMat):
    # 正样本
    data_plus = []
    # 负样本
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 转换为numpy矩阵
    data_plus_np = np.array(data_plus)
    # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)
    # 正样本散点图
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    # 负样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()

if __name__ == '__main__':
    # 加载训练集
    dataArr,labelArr = loadDataSet('D:\DataSetRBF.txt')
    showDataSet(dataArr, labelArr)
