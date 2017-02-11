# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# pylint: disable=C0103, I0011, E1101

"""
Please note, this code is only for python 3+. 
If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
# 一维结构，范围是-1到1
# 感觉这里weights其实只是一个初始值的样子
# 从这一段开始才出现tensor
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
# 预测的y
y = Weights*x_data + biases
# 定义loss fuction
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
### create tensorflow structure end ###
# 定义session
sess = tf.Session()
# session 运行init 所指向内容，这里是initial的步骤


# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
# 这里检查了tensorflow的版本号，但都是initialize的内容
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
# 运行variable初始化的内容
sess.run(init)

for step in range(401):
    # 开始训练...
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
        # sess.run(Weights)...session指向weight，给出weights的结果...


