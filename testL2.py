import time
import numpy as np
import tensorflow as tf


def get_train_data(n):
    x = np.random.random((n, 3))
    w = np.array([[0.1], [0.2], [0.3]])
    y = np.dot(x, w)
    return x, y


def train():
    r = 400
    n = 200
    y_ = tf.placeholder(tf.float32, shape=[None, 1])
    X = tf.placeholder(tf.float32, shape=[None, 3])
    w = tf.Variable(tf.random_normal([3, 1], seed=1))
    Y = tf.matmul(X, w)
    loss = tf.reduce_mean(tf.square(y_ - Y)) + \
        tf.contrib.layers.l2_regularizer(0.1)(w)
    # loss = tf.reduce_mean(tf.square(y_ - Y))
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)
    with tf.Session() as sess:
        init_variables = tf.global_variables_initializer()
        sess.run(init_variables)
        train_x, train_y = get_train_data(n)
        print(train_x, train_y)
        for i in range(r):
            train_x, train_y = get_train_data(n)
            _, loss_, w_ = sess.run([train_step, loss, w], feed_dict={
                X: train_x,
                y_: train_y})
            print(loss_, '---', w_)
            time.sleep(0.1)


if __name__ == "__main__":
    train()
