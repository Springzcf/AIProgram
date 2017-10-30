# _*_coding:utf-8_*_
from numpy import *
import operator

'''完成了对k-近邻算法'''
# 创建数据训练数据集
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, dataset, labels, k):
    print(dataset)
    dataSetSize = dataset.shape[0]
    print(dataSetSize)
    diffMat = tile(inx, (dataSetSize, 1)) - dataset
    print('*'*20)
    print(tile(inx, (dataSetSize, 1)))
    print(diffMat)
    sqDiffMat = diffMat**2
    print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    print(sqDistances)
    distances = sqDistances**0.5
    print(distances)
    sortedDistIndicies = distances.argsort()
    print(sortedDistIndicies)
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) +1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
g, l = createDataSet()
print(classify0([110, 0], g, l, 3))
