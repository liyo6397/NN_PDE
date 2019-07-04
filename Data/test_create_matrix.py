import unittest
import numpy as np
from create_matrix import FDM_2D


class TestFDM_2D(unittest.TestCase):

    def setUp(self):
        self.xs = (0, 1.)
        self.ts = (0, 1.)
        self.npoints = 5
        self.fdm2d = FDM_2D(self.xs, self.ts, self.npoints)

    def test_xs(self):

        nx = self.fdm2d.xs

        print(nx)

    def test_generate_grids(self):

        grid_x, grid_y = self.fdm2d.generate_grids()

        print(grid_x)
        print(grid_y)

    def test_elements(self):

        E = self.fdm2d.elements

        print(E)

    def test_second_div(self):

        D = self.fdm2d.D2

        print(D)

    def test_B(self):

        B = self.fdm2d.fdm_B()

        print(B)

    def test_A(self):

        A =self.fdm2d.fdm_A()

        print(A)

    def test_f(self):
        grid_x, grid_y = self.fdm2d.generate_grids()
        f, inner_f = self.fdm2d.fdm_f(grid_x, grid_y)

        print(f)
        print(inner_f)
        print(np.shape(inner_f))


    def test_fmd_right(self):

        b = self.fdm2d.fdm_right()

        print(b)

    def test_u(self):

        u, inner = self.fdm2d.fdm_u()

        print(np.shape(u))
        print(u)
        print(inner)



    def test_analytic_solution(self):

        r = 1
        h = np.linspace(0, 2*np.pi, 10)

        u = np.sqrt(r)*np.sin(h/2)

        print(u)




