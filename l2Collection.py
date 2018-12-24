import time
import numpy as np
import tensorflow as tf


def get_train_data(n):
    x = np.random.random((n, 3))
    w = np.array([[0.1], [0.2], [0.3]])
    y = np.dot(x, w)
    return x, y


def get_w(shape, lumbda):
    '''
    lumbda 其实就是lambda
    '''
    w = tf.Variable(tf.random_normal(shape, seed=1), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lumbda)(w))
    return w


def train():
    r = 800
    n = 200
    y_ = tf.placeholder(tf.float32, shape=[None, 1])
    X = tf.placeholder(tf.float32, shape=[None, 3])

    layers_nodes = [3, 10, 10, 1]
    n_layers = len(layers_nodes)
    in_node = layers_nodes[0]
    Y = X
    for i in range(1, n_layers):
        out_node = layers_nodes[i]
        w = get_w([in_node, out_node], 0.0001)
        Y = tf.matmul(Y, w)
        in_node = out_node

    loss_end = tf.reduce_mean(tf.square(y_ - Y))
    tf.add_to_collection('losses', loss_end)
    loss = tf.add_n(tf.get_collection("losses"))
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
