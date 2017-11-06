# _*_coding:utf-8_*_
from numpy import *
import operator

'''k-近邻算法'''


# 创建数据训练数据集
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 分类算法
# 入参:分类数据\样本数据\样本类型\K值
# 出参:分类类型
def classify0(inx, dataset, labels, k):
    dataSetSize = dataset.shape[0]
    diffMat = tile(inx, (dataSetSize, 1)) - dataset
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) +1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


g, l = createDataSet()
print(classify0([110, 0], g, l, 3))
