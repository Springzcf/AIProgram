# 通过tensorflow构建神经网络进行找规律

import tensorflow as tf
import numpy as np
import time,csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV


'''-----------------------------------------------SR's help function-------------------------------------------------------'''
# 进行数据的加载,从文件获取
def getDate(fileName,isLabel=True):
    x_date = []
    y_date = []
    fr = open(fileName)

    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float, curLine))  # map all elements to float()
        # normalizeData(fltLine)
        if isLabel:
            x_date.append(fltLine[0:-1])
            y_date.append(fltLine[-1])
        else:
            x_date.append(fltLine)
    return x_date, y_date


def wirteDate(testList,resultList,fileName):
    if len(testList)!= len(resultList):
        return 0
    fr = open(fileName,'w',newline='')
    fr_write = csv.writer(fr)
    for i in range(len(testList)):
        testList[i].extend(resultList[i])
        fr_write.writerow(testList[i])
    fr.close()


# 归一化
def normalizeData(data):
    global minX, maxX
    i = 0
    for x in data:
        data[i] = (x-minX)/(maxX-minX)
        i+=1
    return data

# 获取最大最小值
def getMaxMinNum(data):
    maxNum=0
    minNum=0
    for x in data:
        if max(x) > maxNum:
            maxNum = max(x)
        if min(x) < minNum:
            minNum = min(x)
    return maxNum,minNum


# 特征转换(升维)
def changeData(x_data,powNum=3):
    new_xData =PolynomialFeatures(powNum).fit_transform(x_data)
    return new_xData

# 去除无效特征
def delValuelessFeature(x_data,y_data):
    laModel = LassoCV(alphas=np.logspace(-3, 2, 50),fit_intercept=True)
    laModel.fit(x_data,y_data)
    wIndex = laModel.coef_.ravel()
    listindex=[]
    d=0
    for i in wIndex:
        if not (i<1e-03 and i>=0)or(i<0 and i>-1e-03):
            listindex.append(d)
        d+=1
    return listindex

# 筛选特征
def selectFuture(wIndex,x_data):
    new_data = []
    for data in x_data:
        new_data.append(np.array(data)[wIndex].tolist())
    print(new_data)
    return new_data

# 写日志
def writeLog(msg):
    pass
'''------------------------------------------------SR's NN function----------------------------------------------------'''
# 构建一层神经网络
def addLayer(inputs, in_size, out_size, acF=None):
    # W = tf.Variable(tf.truncated_normal([in_size, out_size]))
    with tf.name_scope("layer"):
        with tf.name_scope("W"):
            W = tf.Variable(tf.random_normal([in_size, out_size],mean=0.0, stddev=0.1),name='W')
        with tf.name_scope("b"):
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='b')
    # W = tf.truncated_normal([in_size,out_size],mean=0.0, stddev=0.1)
        with tf.name_scope('Wx_plus_b'):
            out = tf.add(tf.matmul(inputs, W), b)
        if acF is None:
            outputs = out
        else:
            outputs = acF(out)
        return outputs


def testNormalize(testList):
    retList = []
    for dataLine in testList:
        retList.append(normalizeData(dataLine))
    return retList

'''------------------------------------------------SR's main function---------------------------------------------------'''
# 训练文件名,测试文件名,学习步长,训练次数,允许MSE最小误差
def SRNN(trainFileName,testFileName,learningstep=0.0000001,trainTime=500000,MSELoss=1e-3):
    # 数据的获取
    x_data, y_data = getDate(trainFileName)
    maxNum, minNum = getMaxMinNum(x_data)
    xn = len(x_data[0])
    # 声明变量空间
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, xn])
        ys = tf.placeholder(tf.float32, [None, 1])
    keep_drop = tf.placeholder(tf.float32)
    # 构建曲线模型
    # lay1 = addLayer(xs, xn, 4, acF=tf.nn.relu)
    # lay2 = addLayer(lay1, 4, 6, acF=tf.nn.relu)
    # lay3= addLayer(lay2, 6, 4, acF=tf.nn.relu)
    lay1 = addLayer(xs, xn, 12, acF=tf.nn.relu)
    lay2 = addLayer(lay1, 12, 12, acF=tf.nn.relu)
    lay3 = addLayer(lay2, 12, 4, acF=tf.nn.relu)
    # lay4= addLayer(lay3, 12, 4, acF=tf.nn.relu)
    prediction = addLayer(lay3, 4, 1, acF=None)
    # 学习率优化
    step = learningstep/(maxNum)
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(step, global_step, decay_steps=2000, decay_rate=0.99, staircase=True)

    # loss = tf.reduce_mean(tf.reduce_sum(tf.pow((ys - prediction),2)))
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.pow((ys - prediction), 2))
    with tf.name_scope('train'):
        # train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # 创建训练变量
    yText = tf.placeholder(tf.float32, [None, 1])

    # 初始化变量
    init = tf.global_variables_initializer()
    sess = tf.Session()
    # sess.run(init)
    # writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
    progressP = 0
    beg = time.time()
    with tf.Session() as sess:
        sess.run(init)
        testList = getDate(testFileName,isLabel=False)[0]
        for i in range(trainTime):
            sess.run(train_step, feed_dict={xs: x_data, ys: np.mat(y_data).T.tolist(), global_step: i})
            H = sess.run(loss, feed_dict={xs: x_data, ys: np.mat(y_data).T.tolist()})
            lr = sess.run(learning_rate, feed_dict={global_step: i})
            # print(sess.run(prediction, feed_dict={xs: np.mat(testList)}), end='  ')
            if i % 1000 == 0:
                print("MSE: ", H, "\t", "learning_rate: ", lr)
                print('训练花费时间:', (time.time() - beg), '训练次数:', i, end=' ')
                print(sess.run(prediction, feed_dict={xs: np.mat(testList)}), end='  ')
                progressP = i/trainTime*100
                print("%s%%" % progressP)
            if H < MSELoss:
                break
        progressP=100
        print("%s%%" %progressP)
        # print(sess.run(prediction, feed_dict={xs: np.mat(testList)}), end='  ')
        resultList = sess.run(prediction, feed_dict={xs: np.mat(testList)})
        wirteDate(testList,resultList.tolist(),testFileName)
    pass

# SRNN("D:\\WorkSpace\\AI\\AIProgram\\ai_program\\SearchRule\\NN\\dataSet\\ruleFile.csv","D:\\WorkSpace\\AI\\AIProgram\\ai_program\\SearchRule\\NN\\dataSet\\test.csv",learningstep=0.0001,trainTime=100000)
'''
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return '<h1>Home</h1>'

@app.route('/signin', methods=['GET'])
def signin_form():
    return <form action="/signin" method="post">
              <p><input name="username"></p>
              <p><input name="password" type="password"></p>
              <p><button type="submit">Sign In</button></p>
              </form>

@app.route('/signin', methods=['POST'])
def signin():
    # 需要从request对象读取表单内容：
    if request.form['username']=='admin' and request.form['password']=='password':
        return '<h3>Hello, admin!</h3>'
    return '<h3>Bad username or password.</h3>'

if __name__ == '__main__':
    app.run()
'''

