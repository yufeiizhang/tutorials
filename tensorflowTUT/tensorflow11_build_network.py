# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial
# pylint: disable=C0103, I0011, C0303, E1101
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
    '''
    增加神经网络层...
    这里本质上就是一个regression
    再加上一个activation function的组合
    '''
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
# -1到1区间有300个单位（相当于一个vector）
# 后半部分[:, np.newaxis]是新增加维度
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 引入noise，服从normal distribution（mean=0, std=0.05，格式和x.data一致）
noise = np.random.normal(0, 0.05, x_data.shape)

# 这里设定y=x^200.5+\epsilon
y_data = np.square(x_data) - 0.5 + noise


'''
关于神经网路
输入层 1个神经元
隐藏层 10个神经元
输出层 1个神经元
'''
# define placeholder for inputs to network
# 这个位置不能用variable么？
# 这里实际上被feed了x_data和y_data
# none对应的是sample的数量
# placeholder是没有预定义type的
# 这里 tf.float32是必须的...
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
# 定义隐藏层，利用之前预定义好的add_layer
# 这里，关于隐藏层，输入即x，输入层一个神经元。所以input size为1
# 隐藏层10个神经元，所以output size是10
# 选择一个activation function（TensorFlow中预定义好的nonlinear function）
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
# 定义输出层，接收的即是l1layer的结果
# input output size同理
# 这里激励方程选择为linear，即none
prediction = add_layer(l1, 10, 1, activation_function=None)

# 定义好两层神经层

'''
prediction 和 training
'''
# 开始预测，首先考虑loss（误差）
# the error between prediciton and real data
# (y-predict)^2 对于每个例子
# 这里对于所有例子还需要外层一个sum
# reduction_indices...
# 再求平均值，得到平均误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 进行training
# 选择一个optimizer，最基础的即是gradient descent 
# 需要给出learning rate...e.g.0.1
# optimizer的作用即是减少loss（误差）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
# 对所有Variables进行初始化
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
# 定义session
sess = tf.Session()
# run 初始化
sess.run(init)

# 学习，重复1000次
for i in range(1000):
    # training
    # 学习过程首先run train_step
    # 即之前gradient descent的内容
    # 是不是这里的意思就是，在run train step的过程中
    # 因为需要loss所以就不需要单独进行run session了呢？
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        # 也就是这里...转换为batch的方法更方便一些...
        # 例如这里可以被feed x_batch和y_batch
        # 这里输出误差来检查...training的效果
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

