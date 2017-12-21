'''
主要
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



# 占位符placeholder，我们在TensorFlow运行计算时输入这个值
# placeholder params:类型,二维张量[任意长度,]
x = tf.placeholder(tf.float32, [None, 784])

# Variable代表一个可修改的张量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 创建softmax模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

#
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



