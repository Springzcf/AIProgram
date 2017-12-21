'''
利用神经网络进行拟合曲线
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



def getData():
    # 创建等差数列
    xTrain = np.linspace(-20, 20, 401).reshape([1, -1])
    # normal创建正态分布(均值, 标准差, shape)
    noise = np.random.normal(-100, 100, xTrain.shape)
    return xTrain, 400*np.sin(xTrain) + 2*xTrain*xTrain + noise


def drawData(xTrain, yTrain):
    # 保存初始化训练的图片
    # 清空plt画板
    plt.clf()
    # 将训练数据用圆点显示
    plt.plot(xTrain[0], yTrain[0], 'ro', label=u'训练数据')
    # 显示图片中的图例说明
    plt.legend()
    plt.savefig('curve_data.png', dpi=200)


xTrain, yTrain = getData()
drawData(xTrain, yTrain)


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


hiddenDim = 400

x = tf.placeholder(tf.float32, [1, 401])
w = weight_variable([hiddenDim, 1])
b = bias_variable([hiddenDim, 1])

w2 = weight_variable([1, hiddenDim])
b2 = bias_variable([1])

w3 = weight_variable([401, 401])
b3 = bias_variable([1, 401])

hidden = tf.nn.sigmoid(tf.matmul(w, x)+b)
y = tf.matmul(w2, hidden) +b2


#损失函数
loss = tf.reduce_mean(tf.square(y - yTrain))
step = tf.Variable(0, trainable=False)
rate = tf.train.exponential_decay(0.15, step, 1, 0.9999)
optimizer = tf.train.AdamOptimizer(rate)
train = optimizer.minimize(loss, global_step=step)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for time in range(0, 10000):
    train.run({x:xTrain}, sess)
    if time % 1000 == 0:
        # print("训练次数", time, ",训练平均损失值", loss.eval({x:xTrain,sess}))
        print("训练次数", time)
        plt.clf()
        plt.plot(xTrain[0], yTrain[0], 'ro', label=u'训练数据')
        plt.plot(xTrain[0], y.eval({x:xTrain}, sess)[0], label = u'拟合曲线')
        plt.legend()
        plt.savefig('curve_fitting_'+ str(int(time/1000))+'.png',dpi=200)

xTest = np.linspace(-40, 40, 401).reshape([1, -1])
yTest = 400*np.sin(xTrain) + 2*xTrain*xTrain + np.random.normal(-100, 100, xTest.shape)

# 在坐标图上,显示最终训练的结果
plt.clf()
plt.plot(xTest[0], yTest[0], 'mo', label=u'测试数据')
plt.legend()
plt.savefig('curve_fitting_test.png', dpi = 200)
plt.show()


