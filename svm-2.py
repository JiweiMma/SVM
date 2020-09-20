import matplotlib.pyplot as plt
import numpy as np
import random

#数据结构，维护所有需要操作的值
class optStruct:
#dataMatIn - 数据矩阵
# #classLabels - 数据标签
#C - 松弛变量
#toler - 容错率
    def __init__(self, dataMatIn, classLabels, C, toler):
        # 数据矩阵
        self.X = dataMatIn
        # 数据标签
        self.labelMat = classLabels
        # 松弛变量
        self.C = C
        # 容错率
        self.tol = toler
        # 数据矩阵行数
        self.m = np.shape(dataMatIn)[0]
        # 根据矩阵行数初始化alpha参数为0
        self.alphas = np.mat(np.zeros((self.m,1)))
        # 初始化b参数为0
        self.b = 0
        # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.eCache = np.mat(np.zeros((self.m,2)))

#读取数据
def loadDataSet(fileName):
# dataMat - 数据矩阵
#labelMat - 数据标签
    dataMat = []; labelMat = []
    fr = open(fileName)
    # 逐行读取，滤除空格等
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        # 添加数据
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        # 添加标签
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#计算误差
#oS - 数据结构
#k - 标号为k的数据
def calcEk(oS, k):
    #Ek - 标号为k的数据误差
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T) + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#随机选择alpha_j的索引值
#i - alpha_i的索引值
#m - alpha参数个数
def selectJrand(i, m):
    #j - alpha_j的索引值
    #选择一个不等于i的j
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

#内循环启发方式
#i - 标号为i的数据的索引值
#oS - 数据结构
#Ei - 标号为i的数据误差
def selectJ(i, oS, Ei):
#j, maxK - 标号为j或maxK的数据的索引值
    #初始化
    #Ej - 标号为j的数据误差
    maxK = -1; maxDeltaE = 0; Ej = 0
    #根据Ei更新误差缓存
    oS.eCache[i] = [1,Ei]
    #返回误差不为0的数据的索引值
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    # 有不为0的误差
    if (len(validEcacheList)) > 1:
        # 遍历,找到最大的Ek
        for k in validEcacheList:
            # 不计算i,浪费时间
            if k == i: continue
            # 计算Ek
            Ek = calcEk(oS, k)
            # 计算|Ei-Ek|
            deltaE = abs(Ei - Ek)
            # 找到maxDeltaE
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        # 返回maxK,Ej
        return maxK, Ej
    # 没有不为0的误差
    else:
        # 随机选择alpha_j的索引值
        j = selectJrand(i, oS.m)
        # 计算Ej
        Ej = calcEk(oS, j)
    return j, Ej

#计算Ek,并更新误差缓存
#oS - 数据结构
#k - 标号为k的数据的索引值
def updateEk(oS, k):
    # 计算Ek
    Ek = calcEk(oS, k)
    # 更新误差缓存
    oS.eCache[k] = [1,Ek]

#修剪alpha_j
#aj - alpha_j的值
#H - alpha上限
#L - alpha下限
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    # aj - 修剪后的alpah_j的值
    return aj

#优化的SMO算法
#i - 标号为i的数据的索引值
#oS - 数据结构
def innerL(i, oS):
    #步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    #优化alpha,设定一定的容错率。
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        #使用内循环启发方式选择alpha_j,并计算Ej
        j,Ej = selectJ(i, oS, Ei)
        #保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        #步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        #步骤3：计算eta
        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
        if eta >= 0:
            print("eta>=0")
            return 0
        #步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        #步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        #更新Ej至误差缓存
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        #步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        #更新Ei至误差缓存
        updateEk(oS, i)
        #步骤7：更新b_1和b_2
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        #步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        # 1 - 有任意一对alpha值发生变化
        # 0 - 没有任意一对alpha值发生变化或变化太小
        return 1
    else:
        return 0

#完整的线性SMO算法
#dataMatIn - 数据矩阵
#classLabels - 数据标签
#C - 松弛变量
#toler - 容错率
#maxIter - 最大迭代次数
def smoP(dataMatIn, classLabels, C, toler, maxIter):
#oS.b - SMO算法计算的b
#oS.alphas - SMO算法计算的alphas
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)                    #初始化数据结构
    iter = 0                                                                                         #初始化当前迭代次数
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):                            #遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
        alphaPairsChanged = 0
        if entireSet:                                                                                #遍历整个数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)                                                    #使用优化的SMO算法
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:                                                                                         #遍历非边界值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]                        #遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:                                                                                #遍历一次后改为非边界遍历
            entireSet = False
        elif (alphaPairsChanged == 0):                                                                #如果alpha没有更新,计算全样本遍历
            entireSet = True
        print("迭代次数: %d" % iter)
    return oS.b,oS.alphas                                                                             #返回SMO算法计算的b和alphas

#分类结果可视化
#dataMat - 数据矩阵
#w - 直线法向量
#b - 直线解决
def showClassifer(dataMat, classLabels, w, b):
    #绘制样本点
    # 正样本
    data_plus = []
    # 负样本
    data_minus = []
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 转换为numpy矩阵
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    # 正样本散点图
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    # 负样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)

    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

#计算w
#dataArr - 数据矩阵
#classLabels - 数据标签
def calcWs(alphas,dataArr,classLabels):
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

if __name__ == '__main__':
    dataArr, classLabels = loadDataSet('D:\Data.txt')
    b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 40)
    w = calcWs(alphas,dataArr, classLabels)
    showClassifer(dataArr, classLabels, w, b)
