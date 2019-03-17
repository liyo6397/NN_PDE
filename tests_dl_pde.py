import unittest
import numpy as np
import tensorflow as tf

from dlpde_v1 import DNNPDE, transform, print_exact


class TestDNNPDE(unittest.TestCase):
    def setUp(self):
        self.N_INPUTS = 3
        self.sess = tf.Session()
        self.x = np.linspace(0, 1, self.N_INPUTS).reshape((-1, 1))
        self.dnn = DNNPDE(self.N_INPUTS, 2, 0, None, None, 1, 0, 1, n_hidden=3, static_layer_initializer=True)
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
        print("u_network=", transform(result))

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
