import unittest
import pandas as pd
import numpy as np
from continuous_time_Possion import preprocessing, pdeNN, boundaryLearnNN
import tensorflow as tf


class Test_preprocessing(unittest.TestCase):

    def setUp(self):
        data = pd.read_csv("Data/poisson.csv")
        self.N_u = 5
        self.xs = np.array(data['x']).flatten()[:,None]
        self.ys = np.array(data['y']).flatten()[:,None]
        self.exact = np.array(data['fdm'])
        self.process = preprocessing(self.xs, self.ys, self.exact, self.N_u)

    def test_variable(self):

        lb = self.process.lb
        ub = self.process.ub
        exact = self.process.u_train
        npoints = self.process.npoints

        print(npoints)

    def test_produce_matrix(self):

        data, u_train = self.process.produce_matrix()

        print(np.shape(data))
        print(np.shape(u_train))


    def test_training_data(self):

        idx, data_train, u_train = self.process.training_data()

        print(idx)

    def test_idx(self):

        idx = self.process.find_bc_idx()
        idx = np.array(idx)
        data, u_train = self.process.produce_matrix()

        print(u_train[idx])

class Test_pdedNN(unittest.TestCase):

    def setUp(self):
        # set up data
        data = pd.read_csv("Data/poisson.csv")
        y = np.array(data['y'])
        x = np.array(data['x'])
        exact = (np.array(data['fdm']).T)

        self.N_u = 10
        data_process = preprocessing(x, y, exact, self.N_u)
        self.bc_idx = data_process.bc_idx
        self.data, self.u_train = data_process.produce_matrix()
        self.bc_idx = data_process.find_bc_idx()
        self.x = self.data[:, 0:1]
        self.y = self.data[:, 1:2]
        self.lb = data_process.lb
        self.ub = data_process.ub
        # tf placeholders and graph

        # test models
        self.layers = [2, 5, 1]
        self.model = pdeNN(self.data, self.u_train, self.layers, self.lb, self.ub, self.bc_idx)

        # sess run
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.sess.run(tf.global_variables_initializer())

    def test_variable(self):


        x = self.model.x
        y = self.model.y
        u = self.model.u

        print(u)


    def test_tf_placeholder(self):
        # given
        tf_dict = {self.model.x_tf: self.x, self.model.y_tf: self.y, self.model.u_tf: self.model.u}
        x = self.model.x_tf
        y = self.model.y_tf
        u = self.model.u_tf
        #out = tf.concat([x, y], 1)

        # when
        result = self.sess.run(u, tf_dict)

        # then
        print('data=', result)
        print(np.shape(result))

    def test_initialize_NN(self):

        #tf_dict= {self.model.x_tf: self.x, self.model.y_tf: self.y}
        weight, bias = self.model.initialize_NN()
        #weight = self.sess.run(weight, tf_dict)

        print(weight)
        print(bias)

    def test_network(self):
        # given
        tf_dict = {self.model.x_tf: self.x, self.model.y_tf: self.y, self.model.u_tf: self.model.u}
        out = self.model.u_pred

        # when
        result = self.sess.run(out, tf_dict)


        # then
        print(result[self.bc_idx])

    def test_f_pred(self):
        # given
        tf_dict = {self.model.x_tf: self.x, self.model.y_tf: self.y, self.model.u_tf: self.model.u}
        out = self.model.f_pred

        # when
        result = self.sess.run(out, tf_dict)

        # then
        print(result)

    def test_f(self):
        out = self.model.f

        print(out)

    def test_u_bc(self):
        # given
        tf_dict = {self.model.x_tf: self.x, self.model.y_tf: self.y, self.model.u_tf: self.model.u}
        out = self.model.u_pred_bc

        # when
        result = self.sess.run(out, tf_dict)

        # then
        print(result)


    def test_loss(self):
        # given
        tf_dict = {self.model.x_tf: self.x, self.model.y_tf: self.y, self.model.u_tf: self.model.u}
        out = self.model.loss

        # when
        result = self.sess.run(out, tf_dict)

        # then
        print(result)



    def test_opt(self):
        # given
        tf_dict = {self.model.x_tf: self.x, self.model.y_tf: self.y, self.model.u_tf: self.model.u}
        out = self.model.optimizer

        # when
        result = self.sess.run(out, tf_dict)

        # then
        print(result)

    def test_bc_net(self):
        # given
        tf_dict = {self.model.x_bc_tf: self.x_bc, self.model.y_bc_tf: self.y, self.model.u_tf: self.model.u}
        out = self.model.u_pred

        # when
        result = self.sess.run(out, tf_dict)

        # then
        print(result[self.bc_idx])

class Test_boundaryLearnNN(unittest.TestCase):


    def setUp(self):
        # set up data
        data = pd.read_csv("Data/poisson.csv")
        y = np.array(data['y'])
        x = np.array(data['x'])
        exact = (np.array(data['fdm']).T)
        self.N_u = 10
        self.layers = [2, 5, 1]

        data_process = preprocessing(x, y, exact, self.N_u)
        self.data, self.u = data_process.produce_matrix()
        self.bc_idx = data_process.find_bc_idx()
        self.lb = data_process.lb
        self.ub = data_process.ub

        self.bc_model = boundaryLearnNN(self.data, self.u, self.layers, self.lb, self.ub, self.bc_idx)
        self.x = self.bc_model.x
        self.y = self.bc_model.y
        self.u = self.bc_model.u
        self.lb = data_process.lb
        self.ub = data_process.ub
        # tf placeholders and graph



        # sess run
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.sess.run(tf.global_variables_initializer())

    def test_tf_variable(self):
        # given
        tf_dict = {self.bc_model.x_tf: self.x, self.bc_model.y_tf: self.y, self.bc_model.u_tf: self.bc_model.u}
        x = self.bc_model.x_tf
        y = self.bc_model.y_tf
        u = self.bc_model.u_tf
        out = tf.concat([x, y], 1)

        # when
        result = self.sess.run(out, tf_dict)

        # then
        print(result)
        print(np.shape(self.bc_model.data_bc))

    def test_network(self):
        # given
        tf_dict = {self.bc_model.x_tf: self.x, self.bc_model.y_tf: self.y, self.bc_model.u_tf: self.u}
        out = self.bc_model.u_pred

        # when
        result = self.sess.run(out, tf_dict)


        # then
        print(np.shape(result))
        print(result)

    def test_loss(self):
        # given
        tf_dict = {self.bc_model.x_tf: self.x, self.bc_model.y_tf: self.y, self.bc_model.u_tf: self.u}
        out = self.bc_model.loss

        # when
        result = self.sess.run(out, tf_dict)

        # then
        print(result)

    def test_weight(self):
        tf_dict = {self.bc_model.x_tf: self.x, self.bc_model.y_tf: self.y, self.bc_model.u_tf: self.u}
        print(self.bc_model.weights)
        self.bc_model.train(5)
        print(self.bc_model.weights)








