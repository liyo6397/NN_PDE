# > 2 dim >2 order PDE
import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import symbols, diff
import tensorflow as tf
from tensorflow.keras import layers

TF_DTYPE = tf.float64


def compute_grad_dx(u,x):

    grad_1 = tf.gradients(u, x)[0]
    g1 = grad_1[:,0]
    pred_dx1 = tf.reshape(g1,(-1,input_size))
    grad_2 = tf.gradients(pred_dx1, x)[0]
    g2 = grad_2[:,0]
    pred_dx2 = tf.reshape(g2,(-1,input_size))
    return pred_dx1,pred_dx2

def compute_dx(u,mu,nx,order):

    grad_1 = tf.gradients(u, mu)[0]
    dx = grad_1[:,0]
    #ux = tf.reshape(dx,(-1,1))

    return dx

def f(x,u_set,pred_dx1,pred_dx2=0):

    #equation 1
    #f =  pred_dx1 + (1/5)*u_set + tf.math.exp(x/5.)*tf.math.cos(x)
    #equation 2
    f = pred_dx2 + (1/5)*pred_dx1 + u_set + tf.math.exp(x/5.)*tf.math.cos(x)
    return f

class DNNPDE:

    def __init__(self, nx, num_lyr,d, order, weights,biases,input_size):
        # inputs
        self.mu = tf.placeholder(shape=[None, input_size], dtype=TF_DTYPE)
        self.x_space = np.linspace(start_x, end_x, n_inputs)
        self.t_space = np.linspace(start_t, end_t, n_inputs)
        self.bX, self.bY = np.meshgrid(self.x_space, self.t_space)
        self.domain = np.concatenate([self.bX.reshape((-1, 1)), self.bY.reshape((-1, 1))], axis=1)
        self.input_size = input_size
        self.output_size = 1
        self.n_inputs= nx
        self.dim = d
        self.order = order
        #hidden
        self.n_layers = num_lyr
        self.n_hidden = 10
        #outputs
        self.u=self.u_network(self.mu)
        self.loss = self.loss_function()
        self.error = np.zeros(nx)
        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "forward")
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, var_list=var_list1)

    def u_network(self,inputs):
        with tf.variable_scope('forward'):
            for i in range(self.n_layers-1):
                inputs = self.layer(inputs,self.n_hidden,i,activation_fn=tf.nn.tanh,name='layer_{}'.format(i))
            output = self.layer(inputs,self.output_size,i,activation_fn=None,name='output_layer')


        return output

    def loss_function(self):


        ux = compute_dx(self.u,self.mu,self.n_inputs,self.order)
        uxx = compute_dx(ux,self.mu,self.n_inputs,self.order)

        #Equation dim=1 order=1
        #loss = f(self.x,self.u,pred_dx)
        #Equation dim=1 order=2
        #loss = f(self.mu,self.u,pred_dx1,pred_dx2)
        #loss = tf.reduce_mean(loss**2)

        return ux

    def layer(self,input,output_size,nth_layer,activation_fn=None,name='linear',stddev=5.0):
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            weight = tf.get_variable('Matrix', [shape[1], output_size], TF_DTYPE,
                                     tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+output_size)))
            hiddens = tf.matmul(input, weight)

        if activation_fn:
            return activation_fn(hiddens)
        else:
            return hiddens

    def train(self,sess,step):

        #domain_x = self.x_space.reshape((-1,input_size))

        u = sess.run([self.u], feed_dict={self.mu: self.domain})

        #if step % 1000 == 0:
        #    print("Iteration={}, loss= {}".format(step, loss))

        return u

#Network Parameter
input_num_units=1
h2_num_units = 10
output_num_units=1
#Function
start_x, end_x = 0, 1
start_t, end_t = 0, 1
n_inputs=10
input_size=2
num_lyr, d= 3, 1
d = input_size
order = 2
#Tensorflow Variable
seed=100
seq_length_batch = np.array([n_inputs, 1])
weights = {'output': tf.Variable(tf.random_normal([h2_num_units, output_num_units], seed=seed))}
biases = {'output': tf.Variable(tf.random_normal([h2_num_units], seed=seed))}



# Structure of deep learning
DNN = DNNPDE(n_inputs, num_lyr, d, order, weights, biases,input_size)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1):
        u = DNN.train(sess,step)
    print(np.shape(u))
