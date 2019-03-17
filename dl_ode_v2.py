# Euler+DL and using tensorflow to implement deepl
import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import symbols, diff
import tensorflow as tf

def compute_dx(u,x):
    grad = tf.gradients(u, x)[0]
    dudx = grad[:,0]
    return dudx

def f(x,fst_ts,sec_ts=0):
    #right= x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))
    right = tf.math.pow(x,3)+2*x+tf.math.pow(x,2) *((1. + 3.*tf.math.pow(x,2)) / (1. + x + tf.math.pow(x,3)))
    #coef = x + (1. + 3.*x**2) / (1. + x + x**3)
    coef = x + (1. + 3.*tf.math.pow(x,2)) / (1. + x + tf.math.pow(x,3))
    return (right-fst_ts)/coef


class DNNPDE:

    def __init__(self,x_placeholder,nx, num_lyr,d,biases,input_size):
        # Parameters
        self.n_inputs= nx
        self.n_layers = num_lyr
        self.n_hidden = 10
        self.input_size = input_size
        self.x = x_placeholder  # inner data
        self.r= 0.05
        self.n_outputs = 1

        #Solutions
        self.error = np.zeros(nx)

    def loss_function(self, u_set):

        #pred_du = compute_dx(u_set,self.x)
        pred_du = tf.gradients(u_set,self.x)
        y = f(self.x,pred_du)
        y = tf.reshape(y,[-1])


        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=u_set, labels=tf.reshape(y,[-1]))
        total_loss = tf.math.reduce_sum(loss)

        return total_loss

    def get_a_cell(self, lstm_size, keep_prob):

        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop


    def RNN(self,weights,biases):
        # reshape to [1, n_input]
        inputs_series = tf.reshape(self.x, [-1,self.input_size, self.input_size])
        # 1-layer LSTM with n_hidden units.
        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell(self.n_hidden, 0.5) for _ in range(self.n_layers)])

        initial_state = cell.zero_state(self.input_size, dtype=tf.float32)

        # generate prediction
        state_series, states = tf.nn.dynamic_rnn(cell, inputs_series, initial_state=initial_state, dtype=tf.float32)
        layer = tf.reshape(state_series, [-1, self.n_hidden])
        prediction_series = tf.matmul(layer, weights['output']) + biases['output']

        total_loss = self.loss_function(prediction_series)
        optimizer = tf.train.AdamOptimizer(0.05).minimize(total_loss)


        return prediction_series, total_loss,optimizer


#Network Parameter
input_num_units=1
h1_num_units=10
h2_num_units=10
output_num_units=1
loss_total = 0
#Function
start, end = 0, 1
n_inputs=10
input_size = 1
num_lyr, d= 3, 1
d=1
x_space = np.linspace(start, end, n_inputs)
#Tensorflow Variable
seed=100
#x_placeholder = tf.placeholder(tf.float32,[None,input_size])
x_placeholder = tf.placeholder(tf.float32,[input_size,None])
#x_placeholder = tf.placeholder(tf.float32,[None,n_inputs])
#x_placeholder = tf.placeholder(tf.float32,[None,input_size,input_num_units])
seq_length_batch = np.array([n_inputs, 1])
weights = {'output': tf.Variable(tf.random_normal([h2_num_units, output_num_units], seed=seed))}
biases = {'output': tf.Variable(tf.random_normal([h2_num_units], seed=seed))}
u0 = tf.Variable(np.zeros(1), dtype=tf.float32)

# Structure of deep learning
DNN = DNNPDE(x_placeholder,n_inputs, num_lyr, d, biases,input_size)
pred_u, total_loss = DNN.RNN(weights,biases)
#total_loss = DNN.loss_function(pred_u)
#weight_update = tf.train.AdagradOptimizer(0.05).minimize(total_loss)

# Initializing the variables
init = tf.global_variables_initializer()
print("Network Initialized!")
with tf.Session() as session:
    session.run(init)
    print("Run Session")
    step = 0
    training_iters = 10
    offset = 0
    x = x_space.reshape((-1,input_size))
    meshX = x

    end_idx = input_size

    while step < training_iters:

        #for batch_idx in range(n_inputs):

        #meshX = x[:,batch_idx:end_idx]
        _, loss, pred = session.run([pred_u],feed_dict={x_placeholder: meshX})
        end_idx += 1

        loss_total += loss
        if batch_idx%5 == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/n_inputs))
        step += 1

    print("Optimization Finished!")
