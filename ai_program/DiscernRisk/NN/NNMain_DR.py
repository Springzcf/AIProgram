'''
@author:Springzcf
风险识别主入口
通过利用TensorFlow来进行构建神经网络进行数据分类
'''

import tensorflow as tf
import numpy as np

import csv
import random

# 加载数据格式特征之间用tab相隔
# 入参:文件名(绝对路径)
# 出参:特征值mat,标签值mat
def loadDataByCSV(fileName):
    csvReader = csv.reader(open(fileName, encoding='utf-8'))
    dataList = []
    for row in csvReader:
        dataList.append(row)
    return dataList

def classifyData(dataList):
    d1 = []
    d0 = []
    for line in dataList:
        if line[-1] == '0':
            d0.append(line)
        else:
            d1.append(line)
    return d0,d1

def randomMiniDate(dataMini, Mini_len):
    dataRandomSet = set()
    dataLen = len(dataMini)
    randomList = []
    while True:
        if len(dataRandomSet) == Mini_len:
            break
        dataRandomSet.add(random.randint(0,dataLen-1))
    for index in dataRandomSet:
        randomList.append(dataMini[index])
    return randomList



def randomData(dataList, mini_batch):
    mini_len = mini_batch/2
    data = []
    d0, d1 = classifyData(dataList)
    data1 = randomMiniDate(d1,mini_len)
    data0 = randomMiniDate(d0, mini_len)
    data.extend(data0)
    data.extend(data1)
    return data


# 处理样本不均衡问题
def loadDataRandom(fileName='D:\\WorkSpace\\AI\\AIProgram\\ai_program\\DiscernRisk\\dataSet\\train.csv', mini_batch=256, time = 2):
    # 加载数据
    dataList = loadDataByCSV(fileName)
    x_len = len(dataList)
    bitchList = []
    for i in range(time):
        tmp = randomData(dataList, mini_batch)
        bitchList.append(tmp)
    return bitchList, x_len



def getXY(dataList):
    x_data = []
    y_data = []
    for data in dataList:
        x_data.append(data[0:-1])
        y_data.append(data[-1])
    return x_data, y_data


'''--------------------------Test----------------------------'''

testList = loadDataByCSV('D:\\WorkSpace\\AI\\AIProgram\\ai_program\\DiscernRisk\\dataSet\\Test.csv')

def getAccuracy(yList, y_List):
    AccNum = 0
    length = len(yList)
    for i in range(length):
        if y_List[i][0] > 0 :
            y_ = '1'
        else:
            y_ = '0'
        # print(yList[i], y_)
        if yList[i] == y_:
            AccNum = AccNum + 1
    return AccNum/len(yList)



'''--------------------------Run----------------------------'''

bitchList, x_len = loadDataRandom()

# 构建一层神经网络
def addLayer(inputs, in_size, out_size, acF=None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    out = tf.matmul(inputs, W) + b
    if acF is None:
        outputs = out
    else:
        outputs = acF(out)
    return outputs

xn = 30

# 声明变量空间
xs = tf.placeholder(tf.float32, [None, xn])
ys = tf.placeholder(tf.float32, [None, 1])

lay1 = addLayer(xs, xn, 32, acF=tf.nn.relu)
lay2 = addLayer(lay1, 32, 64, acF=tf.nn.relu)
lay3 = addLayer(lay2, 64, 48, acF=tf.nn.relu)
prediction = addLayer(lay3, 48, 1, acF=tf.nn.sigmoid)
# loss = tf.reduce_mean(tf.pow(ys - prediction,2))
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# yText = tf.placeholder(tf.float32, [None, 1])
# xText=np.mat([16,25,36])#49
# xText2=np.mat([64,81,100])#121
# xText3=np.mat([1,4,9])#16
# xText4=np.mat([144, 169, 196])# 225
# print(bitchList)

# for mini_bitch in bitchList:
#     x_data, y_data = getXY(mini_bitch)
#     mat1 = np.mat(y_data)
#     print(mat1)
lastAcc = 0
for i in range(5000000):
    for mini_bitch in bitchList:
        x_data, y_data = getXY(mini_bitch)
        sess.run(train_step, feed_dict={xs: np.mat(x_data), ys: np.mat(y_data).T})
        # y_T = sess.run(prediction, feed_dict={xs: np.mat(x_data)})
        # print(i, getAccuracy(y_data, y_T.tolist()))

    if i % 1000 == 0:
        x_dataT, y_dataT = getXY(testList)
        # sess.run(train_step, feed_dict={xs: np.mat(x_dataT), ys: np.mat(y_dataT).T})
        # print(sess.run(prediction, feed_dict={xs: np.mat(x_dataT)}))
        y_T = sess.run(prediction, feed_dict={xs: np.mat(x_dataT)})
        print(i,getAccuracy(y_dataT, y_T.tolist()),'训练正确率提升', getAccuracy(y_dataT, y_T.tolist())-lastAcc)
        lastAcc = getAccuracy(y_dataT, y_T.tolist())
