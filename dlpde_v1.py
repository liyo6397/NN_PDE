import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import symbols, diff
import tensorflow as tf
from tensorflow.python.keras import layers

TF_DTYPE = tf.float64


def assert_shape(x, shape):
    S = x.get_shape().as_list()
    if len(S) != len(shape):
        raise Exception("Shape mismatch: {} -- {}".format(S, shape))
    for i in range(len(S)):
        if S[i] != shape[i]:
            raise Exception("Shape mismatch: {} -- {}".format(S, shape))


def f(x, u_set, pred_dx1, pred_dx2=0):
    # equation 1
    # f =  pred_dx1 + (1/5)*u_set + tf.math.exp(x/5.)*tf.math.cos(x)
    # equation 2
    val = pred_dx2 + (1 / 5.) * pred_dx1 + u_set + (1. / 5) * tf.math.exp(-x / 5.) * tf.math.cos(x)

    return val


class DNNPDE:

    def __init__(self, n_inputs, num_lyr, d, weights, biases, input_size, start, end,
                 n_hidden=10, static_layer_initializer=False):
        # layer option
        self.static_layer_initializer = static_layer_initializer

        # inputs
        self.n_inputs = n_inputs
        self.x = tf.placeholder(shape=[None, 1], dtype=TF_DTYPE)
        self.x_space = np.linspace(start, end, self.n_inputs)
        self.input_size = input_size

        # hidden
        self.n_layers = num_lyr
        self.n_hidden = n_hidden

        # outputs
        self.u = self.u_network(self.x)
        self.loss = self.loss_function()
        self.error = np.zeros(self.n_inputs)
        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "forward")
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, var_list=var_list1)

    def u_network(self, x):
        with tf.variable_scope('forward'):
            for i in range(self.n_layers - 1):
                x = self.layer(x, self.n_hidden, i, activation_fn=tf.nn.relu, name='layer_{}'.format(i))
            output = self.layer(x, self.input_size, self.n_layers - 1, activation_fn=None, name='output_layer')
        assert_shape(x, (None, self.n_hidden))

        return output

    def compute_dx(self, u, x):
        grad_1 = tf.gradients(u, x)[0]
        g1 = grad_1[:, 0]
        pred_dx1 = tf.reshape(g1, (-1, self.input_size))
        grad_2 = tf.gradients(pred_dx1, x)[0]
        g2 = grad_2[:, 0]
        pred_dx2 = tf.reshape(g2, (-1, self.input_size))
        assert_shape(pred_dx1, (None, self.input_size))

        return pred_dx1, pred_dx2

    def evaluation(self):
        pred_dx1, pred_dx2 = self.compute_dx(self.u, self.x)
        return f(self.x, self.u, pred_dx1, pred_dx2)

    def loss_function(self):

        ls = tf.math.abs(self.evaluation())
        loss = tf.reduce_mean(ls)

        return loss

    def layer(self, input, output_size, nth_layer, activation_fn=None, name='linear', stddev=5.0):
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            if self.static_layer_initializer:
                initializer = tf.ones_initializer
            else:
                initializer = tf.random_normal_initializer(stddev=stddev / np.sqrt(shape[1] + output_size))
            weight = tf.get_variable('Matrix', [shape[1], output_size], TF_DTYPE, initializer)
            hiddens = tf.matmul(input, weight)

        if activation_fn:
            # print('creating hidden layer %d' % nth_layer)
            return activation_fn(hiddens)

        else:
            # print('creating output layer %d' % nth_layer)
            return hiddens

    def train(self, sess, i):

        domain_x = self.x_space.reshape((-1, self.input_size))
        _, loss, u = sess.run([self.opt, self.loss, self.u], feed_dict={self.x: domain_x})
        # loss = sess.run([self.loss], feed_dict={self.x: domain_x})

        # Z = uh.reshape((self.input_size, self.refn))

        if i % 1000 == 0:
            print("Iteration={}, loss= {}".format(i, loss))

        return loss, u  # res


def main():
    # Network Parameter
    input_num_units = 1
    h2_num_units = 10
    output_num_units = 1

    # Function
    start, end = 0, 1
    n_inputs = 10
    input_size = 1
    num_lyr, d = 10, 1
    d = 1
    x_space = np.linspace(start, end, n_inputs)

    # Tensorflow Variable
    seed = 100
    seq_length_batch = np.array([n_inputs, 1])
    weights = {'output': tf.Variable(tf.random_normal([h2_num_units, output_num_units], seed=seed))}
    biases = {'output': tf.Variable(tf.random_normal([h2_num_units], seed=seed))}

    # Structure of deep learning
    DNN = DNNPDE(n_inputs, num_lyr, d, weights, biases, input_size, start, end)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(10000):
            loss, u = DNN.train(sess, step)
        print(u)

    for x in x_space:
        exact = np.exp(-x / 5.) * np.sin(x)
        print(exact)


if __name__ == "__main__":
    main()