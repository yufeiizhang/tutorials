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


def add_layer(inputs, in_size, out_size, activation_function=None):
    '''add layer'''
    # add one more layer and return the output of this layer
    # 在layer框架的函数中，直接对框架命名
    # 这样每次引用add_layer所显示的内容都是一致的
    with tf.name_scope('layer'):
        # 单独为每个weight/bias/regression命名
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        # 这里activation function，特别是relu这样内置的function
        # 自己有内置的名字，不需要在这里单独进行设置
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs


# define placeholder for inputs to network
# 这里with的作用是在Tensorboard中，对应上一级的name
# 即含有x_input和y_input的框架
with tf.name_scope('inputs'):
    # 关于input部分，name参数本身是可有可无的
    # 即便是生成graph，name参数也不是必须的
    # 但name参数中的内容最终会出现在tensorboard上
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

# 定义session之后定义writer
# tf.train.SummaryWriter soon be deprecated, use following
if int((tf.__version__).split('.')[1]) < 12:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('logs/', sess.graph)
else: # tensorflow version >= 0.12
    # 整个框架加载到一起，这里loading（更应该说是写入）到logs的文件夹中
    # sess.graph即是整个框架
    # 这里writer似乎没有被使用，估计filewriter的返回值不是必须的吧？
    writer = tf.summary.FileWriter("logs/", sess.graph)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs
# 之后需要转到log所在的文件夹
# 运行tensorboard --logdir=logs
# 只要索引到文件夹就可以了
# 这里需要跳到graph上...

