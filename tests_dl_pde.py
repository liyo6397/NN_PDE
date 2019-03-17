import unittest
import numpy as np
import tensorflow as tf

from dlpde_v1 import DNNPDE


class TestDNNPDE(unittest.TestCase):
    def test_compute_dx(self):
        # given
        x = tf.constant([[-2.25], [-3.25]], dtype=tf.float64)
        u = tf.constant([[-2.25], [-3.25]], dtype=tf.float64)
        # x = tf.Tensor(op=None, value_index=0, dtype=tf.float64)

        dnn = DNNPDE(10, 2, 0, None, None, 1, 0, 1)
        # x = np.linspace(0, 1, 10).reshape((-1, 1))
        # u = dnn.u_network(x)

        # when

        # then

        pass

    def test_tf_placeholder(self):
        sess = tf.InteractiveSession()
        with sess.as_default():
            # given
            # real_x = [1, 2, 3, 4]
            real_x = np.linspace(0, 1, 11).reshape((-1, 1))
            x = tf.placeholder(tf.float32, name='x')
            out = tf.add(x, x)

            # when
            result = sess.run(out, feed_dict={x: real_x})

            # then
            print(result)

    def test_interactive(self):
        sess = tf.InteractiveSession()
        with sess.as_default():
            real_x = np.linspace(0, 1, 11).reshape((-1, 1))

            x = tf.placeholder(tf.float32, name='x')
            out = tf.add(x, x)
            result = sess.run(out, feed_dict={x: real_x})
            print(result)

    def test_compute(self):
        sess = tf.InteractiveSession()
        with sess.as_default():
            x = np.linspace(0, 1, 10).reshape((-1, 1))
            dnn = DNNPDE(10, 2, 0, None, None, 1, 0, 1)
            out = dnn.compute_dx(dnn.u, dnn.x)
            sess.run(out, feed_dict={dnn.x: x})

    def test_network(self):
        sess = tf.InteractiveSession()
        with sess.as_default():
            # given
            real_x = np.linspace(0, 1, 11).reshape((-1, 1))
            dnn = DNNPDE(10, 2, 0, None, None, 1, 0, 1)
            out = dnn.u_network(dnn.x)
            sess.run(out, feed_dict={dnn.x: real_x})
