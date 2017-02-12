# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial
# pylint: disable=C0103, I0011, C0303, E1101
"""
Please note, this code is only for python 3+. 
If you are using python 2+, please modify the code accordingly.
可视化训练过程
tensor board中event的内容
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    '''add layer'''
    # add one more layer and return the output of this layer
    # 给出layer name 
    # n_layer作为参数在函数中给出了
    layer_name = 'layer%s' % n_layer

    # for tensorflow version < 0.12
    if int((tf.__version__).split('.')[1]) < 12:
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
                # 准备histogram
                # 第一项为图表名 layer_name 和 str 形式的weights...
                # 这里的/似乎只是分隔符的作用
                # 反正在histogram中输出的结果是hayer2/weights 
                # 后一个参数是histogram的内容
                tf.histogram_summary(layer_name + '/weights', Weights)
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
                # 这里改bias就好了
                tf.histogram_summary(layer_name + '/biases', biases)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b, )
            # 同样的将output输出...
            tf.histogram_summary(layer_name + '/outputs', outputs)
    else:   # tensorflow version >= 0.12
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
                tf.summary.histogram(layer_name + '/weights', Weights)
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
                tf.summary.histogram(layer_name + '/biases', biases)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b, )
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    # loss是在event中显示出来... 
    # 这里是scalar_summary
    if int((tf.__version__).split('.')[1]) < 12:    # tensorflow version < 0.12
        tf.scalar_summary('loss', loss)
    else:   # tensorflow version >= 0.12
        tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# 这里需要将所有的summary合并
if int((tf.__version__).split('.')[1]) < 12:  # tensorflow version < 0.12
    merged = tf.merge_all_summaries()
else:   # tensorflow version >= 0.12
    merged = tf.summary.merge_all()

# tf.train.SummaryWriter soon be deprecated, use following
if int((tf.__version__).split('.')[1]) < 12:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('logs/', sess.graph)
else: # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    # train
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # 每50步输出结果
    if i % 50 == 0:
        # merged也需要run
        # 也要feed内容
        result = sess.run(merged,
                          feed_dict={xs: x_data, ys: y_data})
        # 返回的result，放到writer中
        # 这里i 是记录的step
        # 这里writer即是summary writer的结果，之前生成graph的返回值
        writer.add_summary(result, i)

# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs

'''
events 和 histogram 
好像只有两种不同的记录方式... 
应该没有什么太大的差别
'''