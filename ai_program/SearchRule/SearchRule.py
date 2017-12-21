# 通过tensorflow构建神经网络进行找规律

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 进行数据的加载,从文件获取
def getDate(fileName):
    x_date = []
    y_date = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # map all elements to float()
        x_date.append(fltLine[0:-1])
        y_date.append(fltLine[-1])
    return np.mat(x_date), np.mat(y_date).T



# 进行图形的绘制
# def drawPic():


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


xn = 3

# 声明变量空间
xs = tf.placeholder(tf.float32, [None, xn])
ys = tf.placeholder(tf.float32, [None, 1])
x_date, y_date = getDate("ruleFile.txt")
# 构建曲线模型
# -2 <元<2
# 指数<3
lay1 = addLayer(xs, xn, 6, acF=tf.nn.relu)
lay2 = addLayer(lay1, 6, 12, acF=tf.nn.relu)
# lay3 = addLayer(lay2, 12, 6, acF=tf.nn.relu)
prediction = addLayer(lay2, 12, 1, acF=None)
loss = tf.reduce_mean(tf.square(ys - prediction))
train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

yText = tf.placeholder(tf.float32, [None, 1])
xText=np.mat([16,25,36])#49
xText2=np.mat([64,81,100])#121
xText3=np.mat([1,4,9])#16
xText4=np.mat([144, 169, 196])# 225

print(sess.run(prediction, feed_dict={xs: xText}))


for i in range(5000000):
    sess.run(train_step, feed_dict={xs: x_date, ys: y_date})
    if i % 1000 == 0:
        print(i, sess.run(prediction, feed_dict={xs: xText}),sess.run(prediction, feed_dict={xs: xText2}), sess.run(prediction, feed_dict={xs: xText3}), sess.run(prediction, feed_dict={xs: xText4}))


# 保存模型
# tf.train.Saver.save()

#
# yText=np.mat([676.0])
print(sess.run(prediction, feed_dict={xs: xText}))



