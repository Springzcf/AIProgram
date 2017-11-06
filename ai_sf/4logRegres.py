# _*_coding:utf-8_*_
from math import *
from numpy import *
'''Logistic回归梯度上升算法'''


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn, classLabels):
    # mat(list) 将list转化为numpy中的mat类型(矩阵)
    dataMatrix = mat(dataMatIn)
    # transpose矩阵转置
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        # TODO:计算weight值?
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xCord1 = []; yCord1 = []
    xCord2 = []; yCord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xCord1.append(dataArr[i, 1]); yCord1.append(dataArr[i, 2])
        else:
            xCord2.append(dataArr[i, 1]); yCord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xCord1, yCord1, s=30, c='red', marker='s')
    ax.scatter(xCord2, yCord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# Test@gradAscnet
dataMatIn, classLabel = loadDataSet()
plotBestFit(gradAscent(dataMatIn, classLabel).getA())


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights +alpha*error*dataMatrix[i]
    return weights

# plotBestFit(stocGradAscent0(array(dataMatIn), classLabel))