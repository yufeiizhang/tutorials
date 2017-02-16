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
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
# from sklearn datasets
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
# 这里转为[0,1,0,0,0,0,0,0,0,0]类似的data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    # 貌似，histogram summary是tensorboard必须的。
    tf.histogram_summary(layer_name + '/outputs', outputs)
    return outputs


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
# 这layer的50个输出是为了表征overfitting的问题
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
# prediction以l1为输出
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# the loss between prediction and real data
# 以cross entroy作为loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
# 输出到tensorboard上
tf.scalar_summary('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.merge_all_summaries()
# summary writer goes in here
train_writer = tf.train.SummaryWriter("logs/train", sess.graph)
test_writer = tf.train.SummaryWriter("logs/test", sess.graph)

# initial all variables
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# 500次learning
for i in range(500):
    # here to determine the keeping probability
    # feed training的数据
    # 百分之五十不被drop
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        # record loss
        # 在计算loss的这一部分，则不需要drop掉任何东西，是和train过程中drop百分之五十不同的 
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
