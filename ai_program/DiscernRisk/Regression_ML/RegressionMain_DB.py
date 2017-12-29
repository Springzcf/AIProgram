'''
本模块主要应用了Logistic回归\Ridge回归\LASSO回归\ElasticNet回归进行解决问题
首先自己实现,后边调用了模块
'''

import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Lasso, ElasticNetCV,LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures



def Logistic():
    pass


def Ridge():
    pass


# def Lasso():
#     pass


def ElasticNet():
    pass




'''
--------------------------------------------------调用sklearn库实现--------------------------------------------------------
'''
testFileName = "D:\\WorkSpace\\AI\\AIProgram\\ai_program\\DiscernRisk\\dataSet\\trainBalance.csv"
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


def dealImbalanceOnData(data_0, data_1):
    dataBalance = []
    dataBalance.append(data_1)
    dataBalance.append(data_0[0:len(data_1)])
    return dataBalance[0:-1], dataBalance[-1]



# 数据的处理(数据的获取)
def processingData(fileName):
    # 1 通过panda进行csv数据的获取
    data_0, data_1,data_id = getDataFromCSV(fileName)
    # 2 处理样本不均衡问题
    # x_data, y_data =dealImbalanceOnData(data_0, data_1)
    # 3 返回测试样本\训练样本\验证样本
    x_train,x_test,y_train, y_test = train_test_split(data_0, data_1,train_size=0.8, random_state=1)

    return x_train, y_train, x_test, y_test

def getModels():
    models=[Pipeline([('poly',PolynomialFeatures(2)),('linear',LassoCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]),
            Pipeline([('poly',PolynomialFeatures(2)),('linear',RidgeCV(alphas=np.logspace(-3,2,100), fit_intercept=False))]),
            Pipeline([('poly', PolynomialFeatures(2)),('linear',ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1],alphas=np.logspace(-3,2,100), fit_intercept=False, max_iter=1e3, cv=3))]),
            Pipeline([('poly',PolynomialFeatures(2)),('clf',LogisticRegression())])
    ]
    return models


# 将元特征中的无效特征去除
def selectFuture():
    pass



if __name__ == "__main__":
    x_train, y_train, x_test, y_test = processingData(testFileName)
    # print(x_train,y_test)

    # linreg = LinearRegression()
    # model = linreg.fit(x_train, y_train)
    model = getModels()[3]
    model.fit(x_train, y_train)
    '''
    model = Lasso()
    alpha_can = np.logspace(-3, 2,10)
    lasso_model = GridSearchCV(model,param_grid={'alpha':alpha_can},cv=5)
    lasso_model.fit(x_train, y_train)
    '''
    print(model)

    y_hat = model.predict(x_test)
    corsum = 0
    mse = np.average((y_hat - np.array(y_test))**2)
    rmse = np.sqrt(mse)
    print(mse, rmse)
    # 0.0826446280992
    # 0.287479787288
    data_x, data_y, data_id = getDataFromCSV(predFileName)
    y_hat = model.predict(data_x)
    dataPred=[]
    data_id = np.array(data_id).tolist()
    y_hat = np.array(y_hat).tolist()
    for i in range(len(data_id)):
        dataPred.append([data_id[i],y_hat[i]])
    writeDataCSV('submit.csv',data=dataPred)

    pass

