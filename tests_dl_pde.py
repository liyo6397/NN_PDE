import unittest
import numpy as np

from dlpde_v1 import DNNPDE


class TestDNNPDE(unittest.TestCase):
    def test_compute_dx(self):
        # given
        x = np.linspace(0, 1, 10)
        dnn = DNNPDE(10, 3, 0, None, None, 1, 0, 1)

        pass
