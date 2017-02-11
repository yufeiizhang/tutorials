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

# placeholder似乎需要在这里指定float32的类型
input1 = tf.placeholder(tf.float32)
# 也可以给定结构：
# input1 = tf.placeholder(tf.float32, [2, 2])
input2 = tf.placeholder(tf.float32)
ouput = tf.mul(input1, input2)

# 没有variable所以也就不需要initial的内容了

with tf.Session() as sess:
    # session.run运行，还是和之前一样，
    # 但placeholder的值，需要feed以字典的形式传入
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
