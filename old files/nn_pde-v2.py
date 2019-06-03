import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import symbols, diff

class Neural_Network(object):

    def left(self,x, d):
        if d ==1:
            return 1 #(3)
        if d ==2:
            return 1/5 #(3)
        #return  1/5.0#(2)
        #return x + (1. + 3.*x**2) / (1. + x + x**3)#(1)

    def right(self,x):
        return (-1/5.0)*np.exp(-x/5.0)*np.cos(x)#(3)
        #return np.exp(-1*x/5)*np.cos(x)#(2)
        #return x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))#(1)

    def func(self,x, y,u,u_x=0, u_y=0):
        return np.exp(-x)*(x-2+y^3+6*y)
        #return self.right(x)-ts*self.left(x,1)-d_ts * self.left(x,2) #(3)
        #return self.right(x) - ts * self.left(x)#(2)
        #return self.right(x) - psy * self.left(x) #(1)


    def forward(self, x, y, W1,W2):
        #forward propagation through our network
        input = [x,y]
        self.z = np.dot(
            input,
            W1)  # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(W2.T,self.z2)

        return self.z3

    def dy_forward(self,h,x,y,W1,W2):

        return W1[h]*self.sigmoidPrime(self.z[h],1)*W2[h]

    def dx_forward(self,h,x,y,W1,W2):

        return W1[h]*self.sigmoidPrime(self.z[h],1)*W2[h]

    def sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s, k):
        #derivative of sigmoid
        if k == 1:
            return self.sigmoid(s) * (1 - self.sigmoid(s))
        if k == 2:
            return self.sigmoidPrime(s,1)-2*self.sigmoidPrime(s,1)*self.sigmoid(s)

    def F(self,a,g0,g1):
        return (1-a)*g0+a*g1

    def X0(self,a):
        uy0=self.u_a(0,0)
        uy1=self.u_a(1,0)
        return self.u_a(a,0)-self.F(a,u_a(uy0,uy1))

    def Y0(self,a):
        uy0=self.u_a(0,1)
        uy1=self.u_a(1,1)
        return self.u_a(a,0)-self.F(a,u_a(uy0,uy1))



    def dirich(self,x,y,u_a):

        return self.u_a(x,0)-self.F(x,u_a(0,y),u_a(1,y))+ \
               self.F(y,X0(x),Y0(y))

    def mixed(self,x,y):

        return self.F(x,u_a(0,y),u_a(1,y))+X0(x)+Y0(y)

    def dxy_trial_sol(self,x,y,ana_space):

        # for dirichlet
        return self.dirich(x,y,ana_space)+x*(1-x)*y*(1-y)*self.forward(x,y)
        # for mixed
        return self.mixed(x,y,ana_space)+x*(1-x)*y*(self.forward(x,y,W1,W2)-self.forward(x,1,W1,W2)-diff(self.forward(x,1,W1,W2),y)
    def trial_sol(self,x,y,o,ana_space):

        # for dirichlet
        return self.dirich(x,y,ana_space)+x*(1-x)*y*(1-y)*o
        # for mixed
        return self.mixed(x,y,ana_space)+x*(1-x)*y*(o-o-diff(self.dxy_forward(x,1,W1,W2),y)


    def dy_trial_sol(self, xi, o, d_oxi,ana, total_K):
        #d_Psyt = np.zeros(total_K)

        if total_K == 1:
            d_Psyt = o + xi * d_oxi
        if total_K == 2:
            d_Psyt= -ana[0] + psy_analytic(1,0) + o +xi*d_oxi -2*xi*o -xi**2*d_oxi


        return d_Psyt

    def dd_trial_sol(self, xi, o, d_oxi, dd_oxi):
        return -2*o + 2*d_oxi -4*xi*d_oxi + xi*dd_oxi - xi**2*dd_oxi

    def backward(self, X, ana_space, W1, W2, r, total_K):

        loss_sum = 0
        err_sqr = 0
        err = np.zeros(len(X))
        d_out_xi = np.zeros_like(X)
        d_trial_sol  = np.zeros(total_K)
        trial_psy = np.zeros(total_K)
        e=0

        for i in xrange(nx):
            xi = x_space[i]
            yi = y_space[i]
            o = self.forward(xi,yi, W1,W2)

            u_trial = self.trial_sol(xi, yi, o, ana_space)

            d_oxi = 0
            d_oyi = 0
            dd_oxi = 0
            dd_oyi = 0
            d_ux_trial = np.zeros_like(X)
            d_uy_trial = np.zeros_like(Y)
            dd_ux_trial = np.zeros_like(X)
            dd_uy_trial = np.zeros_like(Y)
            '''for h in xrange(hiddenSize):
                d_oxi += W1[0][h]*diff(self.sigmoid(self.z[h]),self.z[h])*W2[h]
                d_oyi += W1[0][h]*diff(self.sigmoid(self.z[h]),self.z[h])*W2[h]

            for h in xrange(hiddenSize):
                dd_oxi += W1[0][h]*W1[0][h]*W2[h]*diff(self.sigmoid(self.z[h]),self.z[h])*diff(self.z[h],x)
                dd_oyi += W1[0][h]*W1[0][h]*W2[h]*diff(self.sigmoid(self.z[h]),self.z[h])*diff(self.z[h],x)'''

            d_ux_trial[i] = diff(self.dxy_trial_sol(xi, o, ana_space),x)
            d_uy_trial[i] = diff(self.dxy_trial_sol(xi, o, ana_space),y)
            dd_ux_trial[i] = diff(self.dxy_trial_sol(xi, o, ana_space),x,2)
            dd_uy_trial[i]= diff(self.dxy_trial_sol(xi, o, ana_space),y,2)
            d_out_xi[i] = d_oxi

            f = self.func(xi, yi, u_trial, d_ux_trial=0, d_uy_trial=0)
            err[i] = (dd_ux_trial+dd_uy_trial - f)
            e += np.square(err[i])

            loss_sum += np.square(trial_psy - ana_space[i])

        d_E_out = np.zeros_like(err)
        #update weights
        for i in xrange(hiddenSize):
            #iterate input space
            d_E_wo = 0
            d_E_wh = 0
            for j in xrange(len(x_space)):
                xi = x_space[j]
                yi = y_space[j]
                dN_err = diff(self.trial_sol(xi, yi, ana_space),o,2)+diff(self.trial_sol(xi, yi, ana_space),o,2) - \
                self.func(xi, yi, u_trial, d_ux_trial=0, d_uy_trial=0).diff(o)
                d_E_out[j] = (err[j])*dN_err

                d_E_wo += np.dot(d_E_out[j],self.sigmoid(self.z[i]))
                h_units = np.dot(np.dot(x,W1[0][i]),np.dot(y,W1[1][i])
                d_out_v = W2[i].dot(self.sigmoid(np.dot(h_units)).diff(x)*xi)
                d_E_wh += d_E_out[j]*d_out_v

            W2[i] -= r*d_E_wo
            W1[0][i] -= r*d_E_wh

        bound = trial_psy

        return W1, W2, loss_sum/nx*1.0, bound, e
        #loss_sum/nx*1.0

    def saveWeights(self):
        np.savetxt("w1.txt", W1, fmt="%s")
        np.savetxt("w2.txt", W2, fmt="%s")

    def predict(self):
        print "Predicted data based on trained weights: "
        print "Input (scaled): \n" + str(xPredicted)
        print "Output: \n" + str(self.forward(xPredicted))

def psy_analytic(x,y):

    return np.exp(-x)*(x+y**3)#(pde1)
    #return np.exp(-x/5.0)*np.sin(x)#(3)
        #return np.exp(-x/5)*np.sin(x) #(2)
    #return (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2 #(1)
    #return np.exp(-x/5)*np.sin(x)*(-1/5)+ np.exp(-x/5)*np.cos(x)#(3)



NN = Neural_Network()

ittr = 0
max_ittr =500

nx=10
total_K = 2
hiddenSize = 5
x_space = np.linspace(0, 1, nx) #discretize space
y_space = np.linspace(0, 1, nx)
ana_space = psy_analytic(x_space,y_space)
bc = psy_analytic(x_space[-1],0)
total_err = np.zeros(1000)

W1 = np.random.randn(
    1,hiddenSize)# 1xhidden) weight matrix from input to hidden layer
W2 = np.random.randn(
    hiddenSize,1) # (hiddenx1) weight matrix from hidden to output layer

for i in xrange(1000):  # trains the NN 1,000 times
    #if i <= 250:
    #    r = 0.5
    #else:
    #    r = 0.005
    r = 0.01
    W1,W2, loss_square, measure_bc, sum_err = NN.backward(x_space, ana_space,W1,W2, r, 2)
    total_err[i] = sum_err


    if (np.abs(measure_bc - bc) <= 1e-4) and (ittr >= max_ittr):

        print i
        break
    ittr += 1


print "Loss: \n" +str(loss_square)
print sum_err
NN.saveWeights()

#nn_result = [ini[0] + xi * NN.forward(xi,W1,W2) for xi in x_space ]
nn_result = np.zeros(nx)

for i in xrange(len(x_space)):
    xi = x_space[i]
    o = NN.forward(xi,W1,W2)
    nn_result[i] = NN.trial_sol(xi,o,ana_space,total_K)

plt.figure(1)
plt.plot(x_space, ana_space, 'b--')
plt.plot(x_space, nn_result,'r--')
plt.legend(('Actural Solution','NN Solution'),loc='upper left')
plt.xlabel("Input space")
plt.ylabel("The value of trial solution.")

plt.figure(2)
plt.plot(total_err)
plt.xlabel("The number of iteration times")
plt.ylabel("The value of sum square error")

plt.show()
