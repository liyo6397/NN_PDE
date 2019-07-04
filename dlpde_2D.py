import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import symbols, diff
import tensorflow as tf
from tensorflow.python.keras import layers

MOMENTUM = 0.99
EPSILON = 1e-6

TF_DTYPE = tf.float32
class dnnpde_berg:

    def __init__(self,n_inputs):
        # inputs variable
        self.n_inputs = n_inputs
        self.input_size = 2
        self.inputs = tf.placeholder(shape=[None, 2], dtype=TF_DTYPE)
        self.b_inputs = tf.placeholder(shape=[None, 2], dtype=TF_DTYPE)

        # inputs domain
        self.x_space = np.linspace(0, 1, n_inputs)
        self.t_space = np.linspace(0, 1, n_inputs)
        self.b_t_space = np.zeros(self.n_inputs)
        self.X, self.Y = np.meshgrid(self.x_space, self.t_space)
        self.domain = np.concatenate([self.X.reshape((-1, 1)), self.Y.reshape((-1, 1))], axis=1)

        # hidden
        self.n_layers = 2
        self.n_hidden = 5
        self.step_boundaries = [2000, 4000]
        self.step_values = [1.0, 0.5, 0.1]
        self.drop_out_rate = 0.2
        self.learning_rate = 0.001

        #ouput variable
        self.output_size = 1

        #boundary condition
        self.b_u = self.u_network(self.b_inputs)
        self.b_evaluate = self.B(self.b_inputs)
        self.b_loss = self.loss_function(self.b_evaluate)

        #pde
        #self.u = self.u_network(self.inputs)
        #self.evaluate = self.f(self.inputs, self.u)
        #self.loss = self.loss_function(self.evaluate)



    def u_network(self, inputs):

        for i in range(self.n_layers - 1):
            inputs = tf.keras.layers.Dense(self.n_hidden, activation=None)(inputs)
            inputs = tf.keras.layers.BatchNormalization()(inputs)
            inputs = tf.keras.activations.tanh(inputs)
        output = tf.keras.layers.Dense(self.output_size, activation=None)(inputs)
        output = tf.keras.layers.BatchNormalization()(output)

        return output

    def B(self, x):

        boundary_values = np.zeros(len(x))
        return boundary_values
        #raise NotImplementedError


    def loss_function(self,y):

        ls = tf.math.abs(y)
        loss = tf.reduce_mean(ls)

        return loss


    def train(self, sess):

        bX = np.zeros((4 * self.batch_size, 2))
        bX[:self.batch_size, 0] = np.random.rand(self.batch_size)
        bX[:self.batch_size, 1] = 0.0

        bX[self.batch_size:2*self.batch_size, 0] = np.random.rand(self.batch_size)
        bX[self.batch_size:2*self.batch_size, 1] = 1.0

        bX[2*self.batch_size:3*self.batch_size, 0] = 0.0
        bX[2*self.batch_size:3*self.batch_size, 1] = np.random.rand(self.batch_size)

        bX[3*self.batch_size:4*self.batch_size, 0] = 1.0
        bX[3*self.batch_size:4*self.batch_size, 1] = np.random.rand(self.batch_size)

        #domain_x = self.x_space.reshape((-1, self.input_size))
        #_, loss, u = sess.run([self.optimizer, self.loss, self.u], feed_dict={self.x: domain_x})
        u = sess.run([self.b_u], feed_dict={self.b_inputs: bX})

        return u
