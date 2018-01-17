'''
本模块主要应用了Logistic回归\Ridge回归\LASSO回归\ElasticNet回归进行解决问题
首先自己实现,后边调用了模块
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import csv
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Lasso, ElasticNetCV,LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2,SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb



def Logistic():
    pass


def Ridge():
    pass


# def Lasso():
#     pass


def ElasticNet():
    pass
'''
--------------------------------------------------通用方法实现--------------------------------------------------------
'''
testFileName = "D:\\WorkSpace\\AI\\AIProgram\\ai_program\\DiscernRisk\\dataSet\\train.csv"
testFileNameB = "D:\\WorkSpace\\AI\\AIProgram\\ai_program\\DiscernRisk\\dataSet\\trainBalance.csv"
predFileName = "D:\\WorkSpace\\AI\\AIProgram\\ai_program\\DiscernRisk\\dataSet\\pred.csv"

# 数据的加载(从CVS文件加载数据)
def getDataFromCSV(fileName):
    data = pd.read_csv(fileName)
    featureKeyList = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15",
                      "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "V29",
                      "V30","Label","ID"]
    # dataAll = data[featureKeyList]
    # data_0 = []
    # data_1 = []
    # dataAll = np.array(dataAll)
    #
    # for dataI in dataAll:
    #     if dataI[-1] == 0 :
    #         data_0.append(dataI)
    #     else:
    #         data_1.append(dataI)
    data_x = data[featureKeyList[0:-2]]
    data_y = data[featureKeyList[-2]]
    data_id = data[featureKeyList[-1]]
    return data_x, data_y,data_id

def writeDataCSV(fileName,data):
    out  = open(fileName,'w',newline='')
    csv_writer = csv.writer(out)
    for lineCon in data:
        csv_writer.writerow(lineCon)


# 处理样本不均衡问题
def dealImbalanceOnData(data_0, data_1):
    dataBalance = []
    dataBalance.append(data_1)
    dataBalance.append(data_0[0:len(data_1)])
    return dataBalance[0:-1], dataBalance[-1]



# 1 数据的处理(数据的获取\归一化\标准化)
def processingData(fileName):
    # 1 通过panda进行csv数据的获取
    data_0, data_1,data_id = getDataFromCSV(fileName)
    # 2 处理样本不均衡问题
    # x_data, y_data =dealImbalanceOnData(data_0, data_1)
    # 3 返回测试样本\训练样本\验证样本
    # x_train,x_test,y_train, y_test = train_test_split(data_0, data_1,train_size=0.8, random_state=1)

    # return x_train, y_train, x_test, y_test
    return data_0, data_1,data_id




# 2 特征处理(将元特征中的无效特征去除)
def selectFuture():
    f1score = 0
    futureNum = 0
    data_0, data_1, data_id = processingData(testFileNameB)

    #卡方检验
    # for i in range(1,len(data_0)):
        # NData_x= SelectKBest(chi2,i).fit_transform(data_0, data_1)
    NData_x= SelectFromModel(GradientBoostingClassifier()).fit_transform(data_0, data_1)

    x_train, x_test, y_train, y_test = train_test_split(NData_x, data_1, train_size=0.8, random_state=1)
    niceTreeModel = RandomForestClassifier(n_estimators=30, max_depth=19)
    dtcBag = BaggingClassifier(niceTreeModel, n_estimators=300, max_samples=0.6)
    dtcBag.fit(x_train, y_train)
    y_predict = dtcBag.predict(x_test)
    accuracyRet = getAccuracy(y_predict, y_test)
    # if f1score< accuracyRet[3]:
    #     futureNum = i
    # print(futureNum)
    return futureNum


# 测试结果正确率
def getAccuracy(y_prediction,y_real):
    count = (y_prediction == y_real)
    accuracy = np.mean(count)
    mse = np.average((y_prediction - np.array(y_real)) ** 2)
    rmse = np.sqrt(mse)
    f1score =f1_score(y_prediction, y_real,average='binary')
    print('正确率', accuracy, 'MSE', mse, 'RMSE', rmse,'f1-score',f1score)
    return accuracy, mse, rmse, f1score



'''
--------------------------------------------------调用sklearn库线性回归实现--------------------------------------------------------
'''

def getRegressionModels():
    models=[Pipeline([('poly',PolynomialFeatures(2)),('linear',LassoCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]),
            Pipeline([('poly',PolynomialFeatures(2)),('linear',RidgeCV(alphas=np.logspace(-3,2,100), fit_intercept=False))]),
            Pipeline([('poly', PolynomialFeatures(2)),('linear',ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1],alphas=np.logspace(-3,2,100), fit_intercept=False, max_iter=1e3, cv=3))]),
            Pipeline([('poly',PolynomialFeatures(2)),('clf',LogisticRegression())])
    ]
    return models



def regression():
    data_0, data_1, data_id = processingData(testFileName)
    x_train, x_test, y_train, y_test = train_test_split(data_0, data_1, train_size=0.8, random_state=1)
    # print(x_train,y_test)

    # linreg = LinearRegression()
    # model = linreg.fit(x_train, y_train)
    model = getRegressionModels()[3]
    model.fit(x_train, y_train)
    '''
    model = Lasso()
    alpha_can = np.logspace(-3, 2,10)
    lasso_model = GridSearchCV(model,param_grid={'alpha':alpha_can},cv=5)
    lasso_model.fit(x_train, y_train)
    '''
    print(model)

    y_hat = model.predict(x_test)
    accuracyRet= getAccuracy(y_hat,y_test)
    # 正确率 0.917355371901 MSE 0.0826446280992 RMSE 0.287479787288
    data_x, data_y, data_id = getDataFromCSV(predFileName)
    y_hat = model.predict(data_x)
    dataPred=[]
    data_id = np.array(data_id).tolist()
    y_hat = np.array(y_hat).tolist()
    for i in range(len(data_id)):
        dataPred.append([data_id[i],y_hat[i]])
    writeDataCSV('submit.csv',data=dataPred)

'''--------------------------------------------------调用sklearn库决策树&随机森林实现--------------------------------------------'''
def getTreeModel():
    dtc = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3)
    dtcBag = BaggingClassifier(dtc, n_estimators=100, max_samples=0.3)

    model = [dtc, dtcBag]
    return model

def tree():
    # futureNum = selectFuture()
    data_0, data_1, data_id = processingData(testFileName)
    NData_x= SelectFromModel(GradientBoostingClassifier()).fit_transform(data_0, data_1)

    print(NData_x)
    x_train, x_test, y_train, y_test = train_test_split(NData_x, data_1, train_size=0.8, random_state=1)
    # make model
    # for treeDeep in range(1, 40):
    #     model = DecisionTreeClassifier(criterion="entropy", max_depth=treeDeep)
    #     model.fit(x_train, y_train)
    #     y_predict = model.predict(x_test)
    #     accuracyRet = getAccuracy(y_predict, y_test)
    #     print('model',model,'\n正确率', accuracyRet[0], 'MSE', accuracyRet[1], 'RMSE', accuracyRet[2])
    #     if accuracy < accuracyRet[0] :
    #         niceTreeDeep = treeDeep
    # print('niceTreeDeep',treeDeep)
    # model = DecisionTreeClassifier(criterion="entropy", max_depth=19)
    niceTreeModel = RandomForestClassifier(n_estimators=30,max_depth=19)
    # niceTreeModel = RandomForestClassifier(criterion="entropy", max_depth=19)

    dtcBag = BaggingClassifier(niceTreeModel, n_estimators=300, max_samples=0.6)
    dtcBag.fit(x_train, y_train)
    y_predict = dtcBag.predict(x_test)
    accuracyRet = getAccuracy(y_predict, y_test)
    # f1 - score:0.808823529412
    data_x, data_y, data_id = getDataFromCSV(predFileName)
    dataPred = []
    y_predict = dtcBag.predict(data_x)
    data_id = np.array(data_id).tolist()
    y_hat = np.array(y_predict).tolist()
    for i in range(len(data_id)):
        dataPred.append([data_id[i], y_hat[i]])
    writeDataCSV('submitTree.csv', data=dataPred)

'''--------------------------------------------------调用xgboost实现--------------------------------------------'''
def boostting():
    data_0, data_1, data_id = processingData(testFileName)
    x_train, x_test, y_train, y_test = train_test_split(data_0, data_1, train_size=0.8, random_state=1)
    trainData = xgb.DMatrix(x_train,label=y_train)
    testData = xgb.DMatrix(x_test,label=y_test)
    param = {'max_depth': 19, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    watch_list = [(testData, 'eval'), (trainData, 'train')]
    bst = xgb.train(param, trainData, num_boost_round=6, evals=watch_list)
    y_hat = bst.predict(testData)
    getAccuracy(y_hat, y_test)
    # 正确率 0.933884297521 MSE 0.0661157024793 RMSE 0.257129738613

'''--------------------------------------------------调用tensorflow利用神经网络实现--------------------------------------------'''
# 构建一层神经网络
def addLayer(inputs, in_size, out_size, acF=None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    out = tf.matmul(inputs, W) + b
    if acF is None:
        outputs = tf.nn.sigmoid(out)
    else:
        outputs = acF(out)
    return outputs

def NN():
    # PolynomialFeatures(2)
    data_0, data_1, data_id = processingData(testFileName)
    x_train, x_test, y_train, y_test = train_test_split(data_0, data_1, train_size=0.8, random_state=1)

    xn = 30
    # 声明变量空间
    xs = tf.placeholder(tf.float32, [None, xn])
    ys = tf.placeholder(tf.float32, [None, 1])
    keep_drop = tf.placeholder(tf.float32)

    lay1 = addLayer(xs, xn, 6, acF=tf.nn.relu)
    lay2 = addLayer(lay1, 6, 12, acF=tf.nn.relu)
    # lay3 = addLayer(lay2, 12, 24, acF=tf.nn.relu)
    # lay4 = addLayer(lay3, 24, 12, acF=tf.nn.relu)
    prediction =addLayer(lay2, 12, 1,acF=None)
    loss = tf.reduce_mean(tf.pow(ys - prediction,2))
    # loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(50000):
        sess.run(train_step, feed_dict={xs: np.mat(x_train), ys: np.mat(y_train).T,keep_drop:0.7})
        # y_T = sess.run(prediction, feed_dict={xs: np.mat(x_data)})
        # print(i, getAccuracy(y_data, y_T.tolist()))

        if i % 100000== 0:
            # sess.run(train_step, feed_dict={xs: np.mat(x_dataT), ys: np.mat(y_dataT).T})
            # print(sess.run(prediction, feed_dict={xs: np.mat(x_dataT)}))
            y_T = sess.run(prediction, feed_dict={xs: np.mat(x_test)})
            print(y_T)
            # getAccuracy(np.mat(y_T), y_test)
            # lastAcc = getAccuracy( np.mat(tf.argmax(y_T,1)), np.mat(y_test))



if __name__ == "__main__":
    # regression()
    # tree()
    # boostting()
    NN()
    pass
