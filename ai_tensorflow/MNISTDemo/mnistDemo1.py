'''
主要
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#计算准确率
def compute_accuracy(v_xs, v_ys):
    global prediction
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

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
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={batch_xs: batch_xs, batch_ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels),
            compute_accuracy(
                batch_xs, batch_ys)
        )


