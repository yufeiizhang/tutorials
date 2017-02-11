# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial
# pylint: disable=C0103, I0011, C0303
"""
Session的两种打开方式
Please note, this code is only for python 3+. 
If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf

# 这里分别是2*1和1*2的两个矩阵...不是3*3或者2*2的
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)

# method 1
# 建立session
sess = tf.Session()
# 执行product指向的内容
result = sess.run(product)
print(result)
# 并不完全需要
sess.close()

# method 2
# with语句
# 相当于sess = tf.Session，在with语句范围内
# 定义了session
# 这里运行到最后，自动关闭，和for循环类似
with tf.Session() as sess:
    # 运行product指向的内容
    result2 = sess.run(product)
    print(result2)





