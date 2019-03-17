# Sobolev training neural networks for second derivative: NN for ddu
import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import symbols, diff

class Neural_Network(object):

    def coefficient(self,x):

            #return 1 #(3)
        return x + (1. + 3.*x**2) / (1. + x + x**3)#(1)
        #return 1./5#(2)
            #return 1/5 #(3)
            #return  1/5.0#(2)


    def right(self,x):
        return (-1/5.0)*np.exp(-x/5.0)*np.cos(x)#(3)
        #return np.exp(-1*x/5)*np.cos(x)#(2)
        #return x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))#(1)

    def func(self,x, ts):
        #return self.right(x)-ts*self.left(x,1)-d_ts * self.left(x,2) #(3)
        #return (self.right(x)-ts)/self.coefficient(x)#(2)
        return (self.right(x)-ts)/self.coefficient(x) #(1)

    def u_func(self,x, fst_ts, snd_ts):

        return self.right(x)-fst_ts*(1./5)-snd_ts #(3)

    def du_func(self,x, ts, snd_ts):

        return (-ts + self.right(x) -snd_ts)*5. #(3)

    def ddu_func(self,x, ts, fst_ts):

        return self.right(x)-ts - fst_ts*(1./5) #(3)



    def forward(self, xi,W1,W2):
        #forward propagation through our network

        self.z2 = np.zeros(len(W2))
        self.dz2 = np.zeros(len(W2))
        self.z = np.dot(
            xi,
            W1)  # dot product of X (input) and first set of 3x2 weights
        #self.z2 = self.sigmoid(self.z)  # activation function
        for i in xrange(len(W2)):
            if self.z[i] > 0:
                self.dz2[i] = 1
                self.z2[i] = self.z[i]
            else:
                self.dz2[i] = 0.01
                self.z2[i] = 0.01*float(self.z[i])
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
            return self.sigmoid(s) * (1 - self.sigmoid(s))*(1 - 2*self.sigmoid(s))
            #return self.sigmoidPrime(s,1)-2*self.sigmoid(s)
        if k ==3:
            return self.sigmoid(s) * (1 - self.sigmoid(s))*(6*self.sigmoid(s)**2-6*self.sigmoid(s)+1)

    def trial_sol(self,du,x1,x2,u0,h):

        return u0+du*h

    def backward(self, X, ana_space, W1, W2, r,h):

        loss_sum = 0
        err_sqr = 0

        err = np.zeros(len(X))
        du_x = np.zeros(len(X))
        ddu_x = np.zeros(len(X))

        u_x = np.zeros(len(X))
        trial_u_x = np.zeros(len(X))
        e=0


        for i in xrange(nx):
            xi = x_space[i]
            ddu_x[i] = self.forward(xi,W1,W2)

            if i == 0:
                du_x[i]=1
                u_x[i] = 0
            else:

                du_x[i] = h*ddu_x[i]
                u_x[i] = ddu_x[i]*1/2.*h**2
                #trial_u_x[i] = np.sqrt((np.square(ddu_x[i])*h**3)*1/3. + h*(ddu_x[i])**2)
                trial_u_x[i] = np.sqrt((np.square(ddu_x[i])*h**5)*1/4.*1/5. + 1/2.*np.square(ddu_x[i])*h**2 + np.square(ddu_x[i])*h)




            ddf = self.ddu_func(xi, trial_u_x[i], du_x[i])
            #df = self.du_func(xi, u_x[i], ddu_x[i])
            #f = self.u_func(xi, du_x[i],ddu_x[i])
            #err[i] = (trial_u[i] - u) #(1,2)

            err[i] = (ddf - ddu_x[i])
            #err[i] = (du_x[i] - du)
            e += np.square(err[i])#(3)

            #loss_sum += (np.abs(trial_u[i] - ana_space[i])/ana_space[i])

        d_E_out = np.zeros_like(err)
        #update weights
        for j in xrange(hiddenSize):
            #iterate input space
            d_E_wo = 0
            d_E_wh = 0
            for i in xrange(1,len(x_space)):
                xi = x_space[j]

                d_E_out[i] = (err[i])
                d_E_wo += -d_E_out[i]*self.z2[j]
                d_E_wh += -d_E_out[i]*self.dz2[j]*W2[j]*xi

            W2[j] -= r*d_E_wo
            W1[j] -= r*d_E_wh



        return W1, W2, e, u_x
        #loss_sum/nx*1.0

    def saveWeights(self):
        np.savetxt("w1.txt", W1, fmt="%s")
        np.savetxt("w2.txt", W2, fmt="%s")

    def predict(self):
        print "Predicted data based on trained weights: "
        print "Input (scaled): \n" + str(xPredicted)
        print "Output: \n" + str(self.forward(xPredicted))

def psy_analytic(x):

    return np.exp(-x/5.0)*np.sin(x)#(3)
    #return np.exp(-x/5)*np.sin(x) #(2)
    #return (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2 #(1)




NN = Neural_Network()

ittr = 0
max_ittr =1000

nx=100

hiddenSize = 10
x_space = np.linspace(0, 2, nx) #discretize space
h= 2./100
ana_space = psy_analytic(x_space)
bc = psy_analytic(x_space[-1])
sqr_err = np.zeros(max_ittr)
ave_err = np.zeros(nx)

W1 = np.random.uniform(
    -0.5,0.5,hiddenSize)# 1xhidden) weight matrix from input to hidden layer
W2 = np.random.uniform(
    -0.5,0.5,hiddenSize) # (hiddenx1) weight matrix from hidden to output layer

Mitr = 500

for i in xrange(max_ittr):  # trains the NN 1,000 times
#    if i <= 250:
#        r = 0.5
#    else:
#        r = 0.005
    r = 0.05
    W1,W2, loss_square, u_x = NN.backward(x_space, ana_space,W1,W2, r, h)
    sqr_err[i] = loss_square

    ittr += 1



NN.saveWeights()

#nn_result = [ini[0] + xi * NN.forward(xi,W1,W2) for xi in x_space ]
nn_result = np.zeros(nx)
u_x = np.zeros(nx)
d2u=0


for i in xrange(len(x_space)):
    xi = x_space[i]
    if i == 0:
        #du_x[i] = 0
        u = psy_analytic(xi)
    else:

        #u = NN.forward(xi,W1,W2)
        d2u = NN.forward(xi,W1,W2)
        du = h*d2u
        #u = np.sqrt((d2u**2)*(h**3)*1/3. + h*(d2u)**2)
        u = np.sqrt((np.square(d2u)*h**5)*1/20. + 1/2.*np.square(d2u)*h**2 + + np.square(d2u)*h)
        #u = d2u*1/2.*h**2
        ave_err[i] = np.abs(u-psy_analytic(xi))/psy_analytic(xi)

    nn_result[i] = u
print np.mean(ave_err[1:])*100, "%"
show_space = np.linspace(0, 20, nx)
plt.figure(1)
plt.plot(x_space, ana_space, 'b--')
plt.plot(x_space, nn_result,'r--')
plt.legend(('Actural Solution','NN Solution'),loc='upper left')
plt.xlabel("Input space")
plt.ylabel("The value of trial solution.")

plt.figure(2)
plt.plot(sqr_err)
plt.xlabel("The number of iteration times")
plt.ylabel("The value of sum square error")

'''plt.figure(3)
plt.plot(show_space, ana_space, 'b--')
plt.xlabel("The number of iteration times")
plt.ylabel("The value of sum square error")'''

plt.show()
