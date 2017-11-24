from numpy import *

'''
1 K-均值聚类函数
2 二分K-均值算法(优化)
'''

# 加载数据
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat


# 计算平分差和
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)


# 返回K个随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


# K-均值
# 入参:
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 总行数
    m = shape(dataSet)[0]
    # 生成空簇,记录(所属哪个质点,距离质点的位置)
    clusterAssment = mat(zeros((m, 2)))#create mat to assign data points
                                      #to a centroid, also holds SE of each point
    # 生成随机质心
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # for each data point assign it to the closest centroid
        for i in range(m):
            minDist = inf
            minIndex = -1
            # 寻找最近的质心,将样本分质点,放入到簇clusterAssment中表示(所属哪个质点,距离质点的位置)
            for j in range(k):
                # 计算平方差
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 记录举例质心最近的点的位置和距离平方差
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        # 更新质心的位置
        for cent in range(k):
            # 获取所有第cent个质心的数据
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]#get all the point in this cluster
            # 更新质心(列方向的平均值)
            centroids[cent, :] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment


# 利用K-均值获取质点
# datMat = mat(loadDataSet('9testSet.txt'))
# myCentroids, clustAssing = kMeans(datMat, 4)



# 将所有点看成一个簇
# 当簇数目小于K时
# 对于每一个簇
#     计算总误差
#     在给定的簇上面进行K-均值聚类(K=2)
#     计算将该簇一分为二之后的总误差

# 二分K-均值算法
# 入参：数据集、需要划分种类K，误差计算函数
# 出参：质心列表，簇集合（样本的属于第几个质点，距离该质点的SSE）
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    # 生成空簇,记录当前样本(所属哪个质点,距离质点的距离)
    clusterAssment = mat(zeros((m, 2)))
    # 创建一个初始簇(列方向的平均值)
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    # 记录所有的质点
    centList =[centroid0] #create a list with one centroid
    # 计算初始化簇的误差,即到质点的距离
    for j in range(m):#calc initial Error
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
    # 对簇划分个数,直到到达要求K
    while (len(centList) < k):
        lowestSSE = inf
        # 遍历每个簇,尝试将每个簇一分为二,每次查找误差最小的
        # 比如当前有0,1两个簇,尝试将0进行分成两个簇,计算误差.与将1分成两个簇的误差进行比较
        # 如果将1分成的误差比0分成两个簇的误差小,则将1分成1和len()
        for i in range(len(centList)):
            # 获取质点是i的样本
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]#get the data points currently in cluster i
            # 该簇用K-均值,一分为二
            # 质心的位置，距离质心的SSE
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算总误差:sseSplit当前划分的误差(第i个簇),sseNotSplit没有划分的误差(除去第i个簇)
            sseSplit = sum(splitClustAss[:, 1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            # 找到划分最小的误差
            if (sseSplit + sseNotSplit) < lowestSSE:
                # 记录划分误差最小的簇是第几个
                bestCentToSplit = i
                # 记录划分误差最小的簇生成的二分簇的质心
                bestNewCents = centroidMat
                # 记录划分误差最小的簇生成的二分簇的距离该质心的距离
                bestClustAss = splitClustAss.copy()
                # 记录划分误差最小的SSE
                lowestSSE = sseSplit + sseNotSplit
        # 变换划分最优的二分簇的质心位置(如果是划分的簇是第2个,则将这个二分簇的质心位置变为当前质心列表的长度，和当前质心的位置)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        # 将簇集合中，最优划分的簇，进行改变为最优化分的二分簇
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment


# 利用K-均值获取质点
# datMat = mat(loadDataSet('9testSet.txt'))
# myCentroids, clustAssing = biKmeans(datMat, 4)


import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
