#_*_coding:utf-8_*_
import matplotlib.pyplot as plt

'''绘制决策树  '''
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode("决策节点", (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode("叶节点", (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

# 获取节点的数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.key()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.key():
        if type(secondDict[key].__name__) == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

# 获取输的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict[firstStr]:
        if type(secondDict[key].__name__) == 'dict':
                thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
    return maxDepth
