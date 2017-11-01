# _*_coding:utf-8_*_
from numpy import *
import feedparser
'''朴素贝叶斯算法'''

# 初始化样本数据
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],\
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],\
                 ['my', 'dalmation', 'is', 'so', 'cute', 'i', 'love', 'him'],\
                 ['stop', 'postiong', 'sutpid', 'worthless', 'garbage'],\
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],\
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList,classVec


# 将样本中的词条放入set中,用于样本向量化
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet|set(document)
    return list(vocabSet)


# 样本向量化(词集模型)
def setOfwords2Vec(vocabList, inputSet):
    # 全部置为0
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary!" % word)
    return returnVec


# 计算样本中,类型的条件概率
# 入参:样本,类型
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 类型1在所有类型中的总概率 p(c1)
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 防止出现概率0的出现
    # p0Num = zeros(numWords) p0Denom = 0.0
    # p1Num = zeros(numWords) p1Denom = 0.0
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 计算类型为1 每个词条出现的次数
            p1Num += trainMatrix[i]
            # 计算类型为1 所有词条的总次数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #
    # p1Vect = p1Num/p1Denom
    # p0Vect = p0Num/p0Denom
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    # 在类别1\2中每个词条的条件概率p(w|c0)  p(w|c1)
    # p(c1)
    return  p0Vect,p1Vect,pAbusive


# 分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # TODO 这里为什么不是p(w0|1)*p(w1*1)*p(w2*1)...
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# TestMain
def testingNB():
    # 加载数据(所有的样本,样本对应的类型)
    listOPosts, listClasses = loadDataSet()
    # 将样本进行去重(放入set集合中)
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfwords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfwords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfwords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

# 测试朴素叶贝思分类算法
#testingNB()

'''************************************************************************************************'''


# 样本向量化(词袋模型)
def bagOfwords2VecMN(vocabList, inputSet):
    returnVec = [0]*len*(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range():
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfwords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfwords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
        print("the error rate is :",float(errorCount)/len(testSet))


'''************************************************************************************************'''


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen)
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfwords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append((classList[docIndex]))
    p0V, p1V, pSPam = trainNB0(array(trainMat, array(trainClasses)))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfwords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is :", float(errorCount)/len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    import operator
    vocabLost, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -0.6:
            topSF.append((vocabLost[i], p0V[i]))
        if p1V[i] > -0.6 :
            topNY.append((vocabLost[i],p1V[i]))
    sortedSF = sort(topSF, key=lambda pair:pair[1], reverse=True)
    print("SF**"*10)
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**" * 10)
    for item in sortedNY:
        print(item[0])
# Test
# ny = feedparser.parse("http://newyork.craigslist.org.stp/index.rss")
# sf = feedparser.parse("http://sfbay.craigslist.org/stp/index.rss")
# getTopWords(ny, sf)

