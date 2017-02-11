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

# 定义 add_layer 函数
# 传入数据和传入数据大小，输出，和activation函数（none即linear）
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 定义权重，normal distribution？
    # 如果这里是placeholder会如何？
    # 形状即对应输入输出
    # insize行和outsize列
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 定义bias，矢量，一行，outsize列
    # bias初始值尽量不为零所以加上0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # weights和bias每次都会改变

    # regression，神经网络一层中的内容
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # 如果是linear（参数为none的情况）
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        # 如果不是，则采用activationfunction
        # 值作为参数传入
        outputs = activation_function(Wx_plus_b)
    # 函数返回
    return outputs
