# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial
# pylint: disable=C0103, I0011, C0303, E1101
"""
Please note, this code is only for python 3+. 
If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
# 居然还有这个... 自带数据包...
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None,):
    '''add layer'''
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs, v_ys):
    '''定义accuracy的功能'''
    global prediction
    # 这里prediction是全局变量，不需要参数传进来
    # vx和vy则是test用的数据
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # 得到test data对应prediction的结果
    # 之后检查预测是否一致
    # 这里y的结果是离散的所以可以用equal这种函数
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # 定义accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 在这里，vx,vy作为correct_prediction中placeholder的内容
    # run accuracy得到结果
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network
# 输入是28*28的图片
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
# 结果是有10个分类
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
# 这里只有一层layer，输入即是xs，对应的输出就是0-9的十种分类
# classification这里用softmax这个activation function
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and real data
# 这里loss使用cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
# training 还是采用gradient descent，使得loss（cross entropy）达到最小
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    # batch 提取一部分sample
    # mnist提取其中的100个就可以了
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 这里run train step就可以了，在计算这一段的过程中
    # 会调用loss和predictin的内容
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))

