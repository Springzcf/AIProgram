# _*_coding:utf-8_*_
from math import log
import operator

#计算结果集中的香农熵
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		#获取数据集最后一列值,作为key
		currentLabel = featVec[-1]
		#遍历数据集,将类型作为key,将value作为出现的次数
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		#计算出现的概率
		prob = float(labelCounts[key])/numEntries
		#计算香农熵
		shannonEnt -= prob*log(prob,2)
	return shannonEnt

def createDataSet():
	dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
	labels = ['no surfacing','flippers']
	return dataSet,labels

#按照给定特征划分数据集
#params:数据集	划分数据集的特征  需要返回的特征值
def splitDataSet(dataSet,axis,value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value :
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend( featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet


#choose the best way to split dataset
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		print(featList)
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet,i,value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		print(infoGain)
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFuture = i
	return bestFuture
#what is it doing?
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys() : classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(),\
	key = operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

#create Tree function
def createTree(dataSet,labels):
	classList = [example[-1] for example in dataSet]
	print(classList)
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	print(bestFeat)
	bestFeatLabel = labels[bestFeat]
	print(bestFeatLabel)
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	print(featValues)
	uniqueVals = set(featValues)
	print(uniqueVals)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
	return myTree
