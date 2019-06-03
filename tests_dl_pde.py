import unittest
import numpy as np
import tensorflow as tf


from dlpde_v1 import DNNPDE_Cai, dnnpde_Rassi

def transform(results):
    return [r[0] for r in results]

class TestDNNPDE(unittest.TestCase):
    def setUp(self):
        self.N_INPUTS = 3
        self.sess = tf.Session()
        self.x = np.linspace(0, 1, self.N_INPUTS).reshape((-1, 1))
        #self.dnn = DNNPDE(self.N_INPUTS, 2, 0, None, None, 1, 0, 1, n_hidden=3, static_layer_initializer=True)
        self.dnn = DNNPDE_Cai(self.N_INPUTS, 0, 1, static_layer_initializer=True)
        self.sess.run(tf.global_variables_initializer())

    def tearDown(self):
        self.sess.close()
        tf.reset_default_graph()

    def test_print_exact(self):
        # exact= [0.0, 0.4338021664911263, 0.6889381730850401]
        print_exact(np.linspace(0, 1, self.N_INPUTS))

    def test_tf_placeholder(self):
        # given
        out = self.dnn.x

        # when
        result = self.sess.run(out, feed_dict={self.dnn.x: self.x})

        # then
        print('x=', transform(result))

    def test_network(self):
        # given
        out = self.dnn.u

        # when
        result = self.sess.run(out, feed_dict={self.dnn.x: self.x})


        # then
        #print("u_network=", transform(result))
        print(result)

    def test_compute_dx(self):
        # given
        out = self.dnn.compute_dx(self.dnn.u, self.dnn.x)

        # when
        result = self.sess.run(out, feed_dict={self.dnn.x: self.x})

        # then
        # print(result)
        pred_dx1, pred_dx2 = result
        print("pred_dx1=", transform(pred_dx1))
        print("pred_dx2=", transform(pred_dx2))

    def test_evaluation(self):
        # given
        out = self.dnn.evaluation()

        # when
        result = self.sess.run(out, feed_dict={self.dnn.x: self.x})

        # then
        print("evaluation=", transform(result))

    def test_loss(self):
        # given

        # when
        result = self.sess.run(self.dnn.loss, feed_dict={self.dnn.x: self.x})

        # then
        print("loss=", result)

    def test_opt(self):
        # given

        # when
        result = self.sess.run(self.dnn.opt, feed_dict={self.dnn.x: self.x})

        # then
        print("train_op=", result)



class Testdnnpde_rassi(unittest.TestCase):

    def setUp(self):
        self.n_inputs = 5
        self.sess = tf.Session()
        self.batch_size = 1

        #inputs variable
        #self.inputs = tf.placeholder(shape=[None, 2], dtype=TF_DTYPE)
        #self.b_inputs = tf.placeholder(shape=[None, 2], dtype=TF_DTYPE)
        self.dnn = dnnpde_Rassi(self.n_inputs)
        self.sess.run(tf.global_variables_initializer())

        # inputs domain
        self.x_space = np.linspace(0, 1, self.n_inputs)
        self.t_space = np.linspace(0, 1, self.n_inputs)
        self.b_t_space = np.zeros(self.n_inputs)
        self.bX, self.bY = np.meshgrid(self.x_space, self.t_space)
        self.domain = np.concatenate([self.bX.reshape((-1, 1)), self.bY.reshape((-1, 1))], axis=1)

    def test_boundary_input(self):
        # given
        out = self.dnn.b_inputs
        bX = np.zeros((4 * self.batch_size, 2))
        bX[:self.batch_size, 0] = np.random.rand(self.batch_size)
        bX[:self.batch_size, 1] = 0.0

        bX[self.batch_size:2 * self.batch_size, 0] = np.random.rand(self.batch_size)
        bX[self.batch_size:2 * self.batch_size, 1] = 1.0

        bX[2 * self.batch_size:3 * self.batch_size, 0] = 0.0
        bX[2 * self.batch_size:3 * self.batch_size, 1] = np.random.rand(self.batch_size)

        bX[3 * self.batch_size:4 * self.batch_size, 0] = 1.0
        bX[3 * self.batch_size:4 * self.batch_size, 1] = np.random.rand(self.batch_size)

        # when
        result = self.sess.run(out, feed_dict={self.dnn.b_inputs: bX})

        # then
        print("shape=", result.shape)
        print("boundary_inputs=", result)

    def test_u_network(self):
        # given
        out = self.dnn.b_u

        # when
        result = self.sess.run([out], feed_dict={self.dnn.b_inputs: self.domain})

        # then
        result = np.array(result)
        print("shape=", result.shape)
        print("boundary_inputs=", result)

    def test_u_f(self):
        # given
        out = self.dnn.b_evaluate
        bX = np.zeros((4 * self.batch_size, 2))
        bX[:self.batch_size, 0] = np.random.rand(self.batch_size)
        bX[:self.batch_size, 1] = 0.0

        bX[self.batch_size:2 * self.batch_size, 0] = np.random.rand(self.batch_size)
        bX[self.batch_size:2 * self.batch_size, 1] = 1.0

        bX[2 * self.batch_size:3 * self.batch_size, 0] = 0.0
        bX[2 * self.batch_size:3 * self.batch_size, 1] = np.random.rand(self.batch_size)

        bX[3 * self.batch_size:4 * self.batch_size, 0] = 1.0
        bX[3 * self.batch_size:4 * self.batch_size, 1] = np.random.rand(self.batch_size)

        # when
        result = self.sess.run([out], feed_dict={self.dnn.b_inputs: bX})



