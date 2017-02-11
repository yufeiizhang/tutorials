# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial
# pylint: disable=C0103, I0011, C0303, E1101
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf

state = tf.Variable(0, name='counter')
# 在这里，定义了变量state
# 这里初始化了值0和名字counter...用以表征变量的作用
#print(state.name)
# 这里输出counter:0（第一个变量

# 定义了一个新常量
one = tf.constant(1)

# new_value = state + one
new_value = tf.add(state, one)
# state = state + newvalue
# 这段好别扭... 
update = tf.assign(state, new_value)

# 依旧是初始化
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    # 针对variable，似乎对constant没影响的样子
    init = tf.global_variables_initializer()


with tf.Session() as sess:
    # session run 激活变量
    sess.run(init)
    # 三次循环...'_'是什么鬼畜啊！！！！
    for _ in range(3):
        # 每个循环，运行一次update....
        sess.run(update)
        # 似乎run更有session指针的意思...
        # 感觉这里state并没有真正的运行啊，
        # 毕竟之前一步update应该已经将state assign了啊！
        # 很想知道这个计算是在哪里完成运行的...
        print(sess.run(state))

