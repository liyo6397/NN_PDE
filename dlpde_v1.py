import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import symbols, diff
import tensorflow as tf
from tensorflow.python.keras import layers

MOMENTUM = 0.99
EPSILON = 1e-6

TF_DTYPE = tf.float32


def print_exact(x_space):
    #print("exact", [-x*2 for x in x_space])
    print("exact=", [np.exp(-x / 5.) * np.sin(x) for x in x_space])

def print_domain(x_space):
    print("domain:",x_space)


def transform(results):
    return [r[0] for r in results]


def assert_shape(x, shape):
    S = x.get_shape().as_list()
    if len(S) != len(shape):
        raise Exception("Shape mismatch: {} -- {}".format(S, shape))
    for i in range(len(S)):
        if S[i] != shape[i]:
            raise Exception("Shape mismatch: {} -- {}".format(S, shape))


def f(x, u_set, pred_dx1=0, pred_dx2=0):
    # sample equation: y = 2x --> y - 2x = 0
    #val = u_set + 2 * x
    # exact equation: y = e^(-x/5) * sin(x) --> y - e^(-x/5) * sin(x) = 0
    #val = u_set - tf.math.exp(-x / 5.) * tf.math.sin(x)

    # equation 1
    val =  pred_dx1 + (1/5)*u_set + tf.math.exp(x/5.)*tf.math.cos(x)
    # equation 2
    # val = pred_dx2 + (1 / 5.) * pred_dx1 + u_set + (1. / 5) * tf.math.exp(-x / 5.) * tf.math.cos(x)

    return val


class DNNPDE:

    def __init__(self, n_inputs, start, end, static_layer_initializer=False):
        # layer option
        self.static_layer_initializer = static_layer_initializer

        # inputs
        self.n_inputs = n_inputs
        self.x = tf.placeholder(shape=[None, 1], dtype=TF_DTYPE)
        self.x_space = np.linspace(start, end, self.n_inputs)
        self.input_size = 1

        # hidden
        self.n_layers = 2
        self.n_hidden = 20
        self.lstm_cell = 50
        self.step_boundaries = [2000, 4000]
        self.step_values = [1.0, 0.5, 0.1]
        self.drop_out_rate = 0.2
        self.learning_rate = 0.001

        # outputs
        self.output_size = 1
        self.u = self.u_dense_network(self.x)
        self.loss = self.loss_function()
        #var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "forward")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        #self.optimizer = self.optimizing(self.loss)

        # method by Raissi
        self.u_bn = self.u_boundary_network(self.x)


    def u_network(self, x):
        with tf.variable_scope('forward'):
            for i in range(self.n_layers - 1):
                x = self.layer(x, self.n_hidden, i, activation_fn=tf.nn.relu, name='layer_{}'.format(i))
            output = self.layer(x, self.input_size, self.n_layers - 1, activation_fn=None, name='output_layer')
        assert_shape(x, (None, self.n_hidden))

        return output

    def u_dense_network(self, inputs):

        for i in range(self.n_layers - 1):
            inputs = tf.keras.layers.Dense(self.n_hidden, activation=None)(inputs)
            inputs = tf.keras.layers.BatchNormalization()(inputs)
            inputs = tf.keras.activations.tanh(inputs)
        output = tf.keras.layers.Dense(self.output_size, activation=None)(inputs)
        output = tf.keras.layers.BatchNormalization()(output)


        return output



    def u_lstm_network(self, inputs):

        shape = inputs.get_shape().as_list()

        inputs = tf.keras.layers.Embedding(self.n_inputs, self.input_size, input_length=self.input_size)(inputs)
        inputs = tf.keras.layers.LSTM(self.lstm_cell, input_shape=(shape[1], self.lstm_cell), return_sequences=True)(inputs)
        for i in range(self.n_layers - 1):
            inputs = tf.keras.layers.LSTM(self.lstm_cell, return_sequences=True)(inputs)
            inputs = tf.keras.layers.Dropout(0.2)(inputs)
        output = tf.keras.layers.Dense(self.output_size, activation=None)(inputs)
        #output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.output_size))(inputs)

        return output

    



    def optimizing(self, loss):

        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #self.learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
                                                         #boundaries=self.step_boundaries, values=self.step_values)

        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        with tf.control_dependencies(extra_ops):
            train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

        return train_op



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
        return f(self.x, self.u, pred_dx1, pred_dx2=0)

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
            hiddens_bn = self._batch_norm(hiddens)

        if activation_fn:
            # print('creating hidden layer %d' % nth_layer)
            return activation_fn(hiddens_bn)

        else:
            # print('creating output layer %d' % nth_layer)
            return hiddens_bn

    def _batch_norm(self, x, name='batch_norm'):
        """Batch normalization"""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable('beta', params_shape, TF_DTYPE,
                                   initializer=tf.random_normal_initializer(
                                       0.0, stddev=0.1, dtype=TF_DTYPE))
            gamma = tf.get_variable('gamma', params_shape, TF_DTYPE,
                                    initializer=tf.random_uniform_initializer(
                                        0.1, 0.5, dtype=TF_DTYPE))
            # These ops will only be preformed when training
            mean, variance = tf.nn.moments(x, [0], name='moments')
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, EPSILON)
            y.set_shape(x.get_shape())
            return y

    def train(self, sess):

        domain_x = self.x_space.reshape((-1, self.input_size))
        _, loss, u = sess.run([self.optimizer, self.loss, self.u], feed_dict={self.x: domain_x})
        # loss = sess.run([self.loss], feed_dict={self.x: domain_x})

        # Z = uh.reshape((self.input_size, self.refn))
        return loss, u  # res






def main():

    # Function
    start, end = 0, 1
    n_inputs = 10
    x_space = np.linspace(start, end, n_inputs)

    # Structure of deep learning
    DNN = DNNPDE(n_inputs, start, end)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss, u = (0, None)
        for step in range(10000):
            loss, u = DNN.train(sess)

            if step % 1000 == 0:
                print("Step:%6d loss:%.4e" % (step, loss))

        print("loss=%.4e" % loss)
        print("u=", list(map(lambda n: "%.4e" % n, transform(u))))

    print_exact(x_space)
    print_domain(x_space)


if __name__ == "__main__":
    main()
