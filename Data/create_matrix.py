from numpy import *
from numpy.matlib import *
from numpy.linalg import *
from matplotlib.pyplot import *

from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay
import scipy.sparse as sparse
import scipy.sparse.linalg
import csv
import matplotlib.pyplot as plt

class FDM_2D:
    def __init__(self, xs, ts, npoints):
        #grids
        self.xs = xs
        self.ts = ts
        self.npoints = npoints
        self.inner_size = npoints-2
        self.grids_x, self.grids_y = self.generate_grids()
        #parameters
        self.spetial_step = (self.xs[1]-self.xs[0])/(self.npoints-1)
        self.time_step = self.spetial_step/2.
        self.D2 = self.second_div()
        self.I = sparse.identity(self.inner_size).toarray()
        #solutions
        self.B = self.fdm_B()
        self.A = self.fdm_A()
        self.g = self.fdm_bc()
        self.f, self.inner_f = self.fdm_f(self.grids_x,self.grids_y)
        self.bc_idx = self.find_bc_idx()
        self.b = self.fdm_right()
        self.u, self.inner = self.fdm_u()


        #self.vertics = self.grid_vertics(self.mesh_pts)

    def generate_grids(self):

        a = self.xs[0]
        b = self.xs[1]
        c = self.ts[0]
        d = self.ts[1]

        x_space = np.linspace(a, b, self.npoints)
        t_space = np.linspace(c, d, self.npoints)

        X, T = np.meshgrid(x_space, t_space)
        #grids = np.concatenate([X.reshape((-1, 1)), T.reshape((-1, 1))], axis=1)

        return X, T

    def fdm_B(self):

        data = -1*np.ones((3, self.inner_size))
        data[1] = -2 * data[1]
        diags = [-1, 0, 1]
        B = sparse.spdiags(data, diags, self.inner_size, self.inner_size)

        return B.toarray()

    def fdm_A(self):

        A = np.kron(self.B, self.I) + np.kron(self.I, self.B)

        return A

    def fdm_f(self,x,y):

        # (1)
        #f = np.ones((self.inner_size)**2)
        #inner_f = 1

        # (2)
        f = 30*np.pi**2*np.sin(5*np.pi*x)*np.sin(6*np.pi*y)
        #f = f.reshape((-1,1))

        inner_f = np.zeros((self.inner_size)**2)
        element = 0

        for i in range(1, f.shape[0]-1):
            for j in range(1, f.shape[1]-1):
                inner_f[element] = f[i, j]
                element += 1


        return f.reshape(-1,1), inner_f


    def fdm_bc(self):

        bc = np.zeros(((self.npoints)**2,(self.npoints)**2))

        return bc

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

    def fdm_right(self):

        G = np.zeros((self.inner_size)**2)
        top_bottom = np.zeros(self.inner_size)
        top_bottom[0] = 0
        top_bottom[-1] = 0

        for i in range(1,self.inner_size):
            top_bottom[i] = 0

        G[:self.inner_size] = self.g[1:self.inner_size+1, 0]+top_bottom

        for m in range(1,self.inner_size-1):
            G[self.inner_size*m:self.inner_size*(m+1)] = top_bottom

        G[-self.inner_size:] = self.g[1:self.inner_size+1, -1]+top_bottom



        b = self.spetial_step**2*self.inner_f+G
        #b = self.spetial_step ** 2 * self.f + G

        return b

    def fdm_u(self):

        u = np.zeros(((self.npoints),(self.npoints)))
        inner = np.transpose(np.mat(sparse.linalg.spsolve(self.A, self.b)))
        element_u = 0

        #for j in range(self.inner_size):
        #    for i in range(self.inner_size):

        #        if j == 0:
        #            u[i+1, j+1] = inner[element_u]

        #        else:
        #            u[i + 1, j + 1] = inner[element_u]
        #        element_u += 1

        for i in range(self.inner_size):
            for j in range(self.inner_size):
                u[i+1,j+1] = inner[element_u]
                element_u += 1



        return u, inner


    def second_div(self):

        data = np.ones((3, self.npoints))
        data[1] = -2 * data[1]
        diags = [-1, 0, 1]
        D2 = sparse.spdiags(data, diags, self.npoints, self.npoints) / (self.spetial_step ** 2)
        D2 = D2.toarray()

        return D2


    def heat(self):

        data = []
        u = 0
        numOftimestp = int(1 / self.time_step)
        for i in range(numOftimestp):
            A = (self.I - self.time_step/2*self.D2)
            b = (self.I + self.time_step / 2 * self.D2) * u
            u = np.transpose(np.mat(sparse.linalg.spsolve(A, b)))

        data.append(u)

        return data


def main():
    xs = (0, 1.)
    ys = (0, 1.)
    npoints = 10

    fdm = FDM_2D(xs, ys, npoints)

    # generate mesh and determine boundary vertices
    #n_xy = fem.mesh(xs, ys, npoints)

    # solve Poisson equation
    u = fdm.u

    X, Y = fdm.generate_grids()
    print(np.shape(u))
    print(np.shape(X))
    print(np.shape(Y))

    f = open('poisson_2.csv', 'w')
    #csv_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    f.write("x")
    f.write(",")
    f.write("y")
    f.write(",")
    f.write("fdm")
    f.write("\n")
    for i in range(npoints):
        for e in range(npoints):
            f.write(str(X[i, e]))
            f.write(",")
            f.write(str(Y[i, e]))
            f.write(",")
            f.write(str(u[i, e]))
            f.write("\n")
    f.close()


    h = plt.contourf(X, Y, u)
#    plt.pcolor(X, Y, u)
    plt.show()




if __name__ == "__main__":
    main()
