import numpy as np
from matplotlib import pyplot as plt
import math

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

    def func(self,x,ts, d_ts, d):
        return self.right(x)-ts*self.left(x,1)-d_ts * self.left(x,2) #(3)
        #return self.right(x) - ts * self.left(x)#(2)
        #return self.right(x) - psy * self.left(x) #(1)


    def forward(self, xi,W1,W2):
        #forward propagation through our network

        self.z = np.dot(
            xi,
            W1[0])  # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(W2.T,self.z2)

        return self.z3

    def sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s, k):
        #derivative of sigmoid
        if k == 1:
            return self.sigmoid(s) * (1 - self.sigmoid(s))
        if k == 2:
            return self.sigmoidPrime(s,1)-2*self.sigmoidPrime(s,1)*self.sigmoid(s)

    def trial_sol(self,xi,o,ana,total_K):
        if total_K == 1:
            return ana[0] +o*xi
        if total_K == 2:

            return ana[0] - ana[1]*xi + psy_analytic(1,0)*xi + xi*o - o*xi**2

    def d_trial_sol(self, xi, o, d_oxi,ana, total_K):
        #d_Psyt = np.zeros(total_K)

        if total_K == 1:
            d_Psyt = o + xi * d_oxi
        if total_K == 2:
            #d = ini[0] + 2*xi*o + xi**2 * d_oxi
            #d = -ini[0] + KN.psy_analytic(1,1) + o - 2*o + 2*xi* d_oxi
            d_Psyt= -ana[0] + psy_analytic(1,0) + o +xi*d_oxi -2*xi*o -xi**2*d_oxi
        #d_Psyt[total_K-1] = d

        return d_Psyt

    def dd_trial_sol(self, xi, o, d_oxi, dd_oxi):
        return -2*o + 2*d_oxi -4*xi*d_oxi + xi*dd_oxi - xi**2*dd_oxi





    def Echain(self, xi, d_oxi, o, total_K):
        if total_K == 1:
            return(1-(-xi*self.left(xi)))
        if total_K == 2:
            #return 2*xi + 2*xi*o + np.square(xi)*d_oxi
            return -2+xi-xi**2-1/5.0


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
            o = self.forward(xi,W1,W2)

            trial_psy = self.trial_sol(xi, o, ana_space, total_K)

            d_oxi = 0
            dd_oxi = 0
            for h in xrange(hiddenSize):
                d_oxi += W1[0][h]*self.sigmoidPrime(self.z[h],1)*W2[h]

            for h in xrange(hiddenSize):
                dd_oxi += W1[0][h]*W1[0][h]*W2[h]*self.sigmoidPrime(self.z[h],2)

            d_trial_psy = self.d_trial_sol(xi,o,d_oxi,ana_space, total_K)
            dd_trial_psy = self.dd_trial_sol(xi,o,d_oxi, dd_oxi)
            d_out_xi[i] = d_oxi

            f = self.func(xi, trial_psy, d_trial_psy, total_K)
            err[i] = (dd_trial_psy - f)
            e += np.square(err[i])
            #err.append(er)
            #err_sqr += np.square(d_trial_psy - f)

            loss_sum += np.square(trial_psy - ana_space[i])

        #error = np.array(err)
        d_E_out = np.zeros_like(err)
        #update weights
        for i in xrange(hiddenSize):
            #iterate input space
            d_E_wo = 0
            d_E_wh = 0
            for j in xrange(len(x_space)):
                xi = x_space[j]
                d_E_out[j] = (err[j])*self.Echain(xi,d_out_xi[j],o, total_K)
                d_E_wo += np.dot(d_E_out[j],self.sigmoid(np.dot(xi,W1[0][i])))
                d_out_v = W2[i].dot(self.sigmoidPrime(np.dot(xi,W1[0][i]),1)*xi)
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

def psy_analytic(x, k):
    if k == 0:
        return np.exp(-x/5.0)*np.sin(x)#(3)
        #return np.exp(-x/5)*np.sin(x) #(2)
    #return (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2 #(1)
    if k == 1:
        return np.exp(-x/5)*np.sin(x)*(-1/5)+ np.exp(-x/5)*np.cos(x)#(3)



NN = Neural_Network()

ittr = 0
max_ittr =500

nx=10
total_K = 2
hiddenSize = 5
x_space = np.linspace(0, 2, nx) #discretize space
ana_space = psy_analytic(x_space,0)
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
