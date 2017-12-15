'''
Created on 2017年12月15日

@author: zhangcf17306
'''

import tensorflow as tf


# 占位符placeholder，我们在TensorFlow运行计算时输入这个值
# placeholder params:类型,二维张量[任意长度,]
x = tf.placeholder("float", [None, 784])

# Variable代表一个可修改的张量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 创建softmax模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

#
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = data_set.next_batch(100)
    sess.run(train_step, feed_dict={})


