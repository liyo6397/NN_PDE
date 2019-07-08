'''''
@author : Maziar Raissi
'''''
import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
import pandas as pd
from mpl_toolkits.mplot3d import axes3d, Axes3D

np.random.seed(1234)
tf.set_random_seed(1234)

class boundaryLearnNN:

    def __init__(self, data, u, layers, lb, ub, bc_idx, period_idx):

        self.bc_idx = bc_idx
        self.period_idx = period_idx
        self.lb = lb
        self.ub = ub
        self.data = data
        # BC Domain
        self.data_bc = self.data[self.bc_idx,:]
        self.x = self.data_bc[:, 0:1]
        self.y = self.data_bc[:, 1:2]
        self.u = u[self.bc_idx]
        # Period Domain
        self.data_prd = self.data[self.period_idx, :]
        self.x_prd = self.data_prd[:, 0:1]
        self.y_prd = self.data_prd[:, 1:2]

        # tf placeholders and graph
        self.layers = layers
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([-6.0], dtype=tf.float32)

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.x_prd_tf = tf.placeholder(tf.float32, shape=[None, self.x_prd.shape[1]])
        self.y_prd_tf = tf.placeholder(tf.float32, shape=[None, self.y_prd.shape[1]])

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN()

        # Training: boundary
        self.u_pred = self.net_u(self.x_tf, self.y_tf)
        self.u_x_pred = tf.Variable([0.0], dtype=tf.float32)
        self.u_y_pred = tf.Variable([0.0], dtype=tf.float32)
        self.u_xx_pred = tf.Variable([0.0], dtype=tf.float32)
        self.u_yy_pred = tf.Variable([0.0], dtype=tf.float32)
        self.f_pred = self.net_f_bc(self.x_tf, self.y_tf, self.u_pred)
        self.f_bc = self.f(self.x_tf, self.y_tf)
        #Training: period
        self.u_prd_pred = self.net_u(self.x_prd_tf, self.y_prd_tf)
        self.u_x_prd_pred = tf.Variable([0.0], dtype=tf.float32)
        self.u_y_prd_pred = tf.Variable([0.0], dtype=tf.float32)
        self.u_xx_prd_pred = tf.Variable([0.0], dtype=tf.float32)
        self.u_yy_prd_pred = tf.Variable([0.0], dtype=tf.float32)
        self.f_prd_pred = self.net_f_prd(self.x_prd_tf, self.y_prd_tf, self.u_prd_pred)
        self.f_prd = self.f(self.x_prd_tf, self.y_prd_tf)


        self.loss = tf.reduce_mean(tf.square(self.f_prd - self.f_prd_pred))+\
                    tf.reduce_mean(tf.square(self.u_x_prd_pred)) + \
                    tf.reduce_mean(tf.square(self.u_y_prd_pred)) +\
                    tf.reduce_mean(tf.square(self.u_x_pred)) + \
                    tf.reduce_mean(tf.square(self.u_y_pred)) +\
                    tf.reduce_mean(tf.square(self.f_bc - self.f_pred)) #+ \
                    #tf.reduce_mean(tf.square(self.u_tf - self.u_pred))




        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                        'maxfun': 50000,
                                                                        'maxcor': 50,
                                                                        'maxls': 50,
                                                                        'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self):
        weights = []
        biases = []
        num_layers = len(self.layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[self.layers[l], self.layers[l+1]])
            b = tf.Variable(tf.zeros([1,self.layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, data, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(data - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, y):
        u = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
        return u

    def net_f_bc(self, x, y, u):

        u_y = tf.gradients(u, y)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        self.u_x_pred = u_x
        self.u_y_pred = u_y

        f = -(u_xx+u_yy)

        #f = u_t + lambda_1*u*u_x - lambda_2*u_xx

        return f

    def net_f_prd(self, x, y, u):

        u_y = tf.gradients(u, y)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        self.u_x_prd_pred = u_x
        self.u_y_prd_pred = u_y
        self.u_xx_prd_pred = u_xx
        self.u_yy_prd_pred = u_yy
        f = -(u_xx+u_yy)

        return f


    def f(self,x,y):

        #f = np.ones((self.u.shape[0], 1))
        f = 30*np.pi**2*tf.math.sin(5*np.pi*x)*tf.math.sin(6*np.pi*y)

        return f


    def callback(self, loss):
        print('Loss: %e' % (loss))


    def train(self, nIter):
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.u_tf: self.u,\
                   self.x_prd_tf: self.x_prd, self.y_prd_tf: self.y_prd}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = np.exp(self.sess.run(self.lambda_2))
                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, Time: %.2f' %
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss])
                                #loss_callback = self.callback)

    def predict(self, X_star):

        tf_dict = {self.x_tf: X_star[:, 0:1], self.y_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)

        return u_star, f_star



class pdeNN:
    # Initialize the class
    def __init__(self, data, u, layers, lb, ub, bc_idx, weights, biases):

        self.lb = lb
        self.ub = ub
        self.bc_idx = bc_idx

        self.data = data
        self.data_bc = data[self.bc_idx, :]

        # Domain
        self.x = data[:, 0:1]
        self.y = data[:, 1:2]
        self.u = u
        self.bc_points = self.x.shape[0]

        # tf placeholders and graph
        self.layers = layers
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([-6.0], dtype=tf.float32)

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])


        # Initialize NNs
        self.weights = weights
        self.biases = biases
        #self.weights, self.biases = self.initialize_NN()
        #self.bc_learn = boundaryLearnNN(self.data, self.u, self.layers, self.lb, self.ub, self.bc_idx)
        #self.bc_learn.train(100)
        #self.weights = self.bc_learn.weights
        #self.biases = self.bc_learn.biases

        #Training: f
        self.u_pred = self.net_u(self.x_tf, self.y_tf)
        self.u_x_pred = tf.Variable([0.0], dtype=tf.float32)
        self.u_y_pred = tf.Variable([0.0], dtype=tf.float32)
        self.f_pred = self.net_f(self.x_tf, self.y_tf)
        self.f = self.f()

        self.loss = tf.reduce_mean(tf.square(self.f - self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self):
        weights = []
        biases = []
        num_layers = len(self.layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[self.layers[l], self.layers[l+1]])
            b = tf.Variable(tf.zeros([1,self.layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, data, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(data - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, y):
        u = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
        return u

    def net_f(self, x, y):
        lambda_1 = self.lambda_1
        lambda_2 = tf.exp(self.lambda_2)
        #u = self.net_u(x,y)
        u_y = tf.gradients(self.u_pred, y)[0]
        u_x = tf.gradients(self.u_pred, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        self.u_x_pred = u_x
        self.u_y_pred = u_y
        f = -(u_xx+u_yy)

        #f = u_t + lambda_1*u*u_x - lambda_2*u_xx

        return f

    def f(self):

        #f = np.ones((self.u.shape[0], 1))
        f = 30 * np.pi ** 2 * tf.math.sin(5 * np.pi * x) * tf.math.sin(6 * np.pi * y)
        return f

    def loss(self):

        f_loss = tf.reduce_mean(tf.square(self.f_pred-self.f))
        #bc_u = self.u
        #bc_u_pred = self.u_pred
        #bc_loss = tf.reduce_mean(tf.square(bc_u - bc_u_pred))
        #loss = f_loss + bc_loss

        return f_loss



    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, np.exp(lambda_2)))


    def train(self, nIter):
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.u_tf: self.u}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = np.exp(self.sess.run(self.lambda_2))
                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, Time: %.2f' %
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.lambda_1, self.lambda_2])
                                #loss_callback = self.callback)


    def predict(self, X_star):

        tf_dict = {self.x_tf: X_star[:, 0:1], self.y_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)

        return u_star, f_star


class preprocessing:

    def __init__(self, xs, ys, exact, nus):

        self.xs = xs.flatten()[:,None]
        self.ys = ys.flatten()[:,None]
        self.exact = exact.T
        self.N_u = nus
        self.npoints = 100
        self.data, self.u_train = self.produce_matrix()
        # Doman bounds
        self.lb = self.data.min(0)
        self.ub = self.data.max(0)
        self.bc_idx = self.find_bc_idx()
        self.period_idx_bg, self.period_idx_fn = self.find_period_idx()
        # Generate training data
        #self.idx, self.data_train, self.u_exact = self.training_data()

    def produce_matrix(self):

        data = np.hstack((self.xs, self.ys))
        exact = self.exact[:, None]

        return data, exact

    def find_bc_idx(self):

        #idx = np.zeros(4*self.npoints-4)
        idx = []
        #left:
        for i in range(self.npoints):
            idx.append(i)
        #top
        for i in range(1,self.npoints):
            idx.append(self.npoints*i)
        #bottom
        for i in range(1,self.npoints):
            idx.append(self.npoints*i+self.npoints-1)
        #right
        for i in range(1,self.npoints-1):
            idx.append(self.npoints*(self.npoints-1)+i)

        return np.array(idx)

    def find_period_idx(self):

        idx_1 = []
        idx_2 = []
        element = 0

        for i in range(self.data.shape[0]):
            if self.data[i, 0] <= 2/5. or self.data[i, 1] <= 2/6.:
                idx_1.append(element)
            if self.data[i, 0] >= 4/5. or self.data[i, 1] >= 5/6.:
                idx_2.append(element)

            element += 1

        return np.array(idx_1), np.array(idx_2)



    def training_data(self):

        idx = np.random.choice(self.data.shape[0], self.N_u, replace=False)
        data_train = self.data[idx, :]
        u = self.u_train[idx]

        return idx, data_train, u




if __name__ == "__main__":

    nu = 0.01/np.pi

    N_u = 2000
    #layers = [2, 10, 10, 1]
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]


    data = pd.read_csv("Data/poisson_2_1.csv")

    y = np.array(data['y'])
    x = np.array(data['x'])
    exact = np.array(data['fdm'])
    data_process = preprocessing(x, y, exact, N_u)

    data, u_train = data_process.produce_matrix()
    lb = data_process.lb
    ub = data_process.ub
    #bc_idx = data_process.bc_idx
    period_idx, period_idx_fn = data_process.find_period_idx()
    bc_idx = period_idx_fn

    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    noise = 0.0

    bc_learn = boundaryLearnNN(data, u_train, layers, lb, ub, bc_idx, period_idx)
    bc_learn.train(10000)
    weights = bc_learn.weights
    biases = bc_learn.biases
    u_pred, f_pred = bc_learn.predict(data)
    #u_pred[bc_idx] = 1e-06



    #model = pdeNN(data, u_train, layers, lb, ub, period_idx, weights, biases)
    #model.train(10000)

    ###############

    #u_pred, f_pred = model.predict(data)

    error_u = np.linalg.norm(u_train-u_pred, 2)/np.linalg.norm(u_train, 2)
    #error_u = (u_train - u_pred) / (u_train)

    X, Y = np.meshgrid(x, y)
    U_pred = griddata(data, u_pred.flatten(), (X, Y), method='cubic')

    #lambda_1_value = model.sess.run(model.lambda_1)
    #lambda_2_value = model.sess.run(model.lambda_2)
    #lambda_2_value = np.exp(lambda_2_value)

    #error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    #error_lambda_2 = np.abs(lambda_2_value - nu)/nu * 100



    print('Error u: %e' % (error_u))
   # print('Error l1: %.5f%%' % (error_lambda_1))
   # print('Error l2: %.5f%%' % (error_lambda_2))


    u = np.reshape(u_pred, (100,100))
    x = np.reshape(x, (100, 100))
    y = np.reshape(y, (100, 100))


    h = plt.contourf(x, y, u)
    plt.show()

    #fig = plt.figure()
    #ax = Axes3D(fig)

    #cset = ax.contour(X, Y, u, 16, extend3d=True)
    #ax.clabel(cset, fontsize=9, inline=1)
    #plt.show()


