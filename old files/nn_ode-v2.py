import numpy as np
from matplotlib import pyplot as plt
import math
# X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input data for prediction)


nx=10
XS=[]
A = []
def psy_analytic(x):
    #return np.exp(-x/5)*np.sin(x) #(2)
    return (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2 #(1)

x_space = np.linspace(0, 1, nx) #discretize space

analytic_space = psy_analytic(x_space)
psy_d = np.zeros_like(analytic_space)
#psy_d[0] = 1.
psy_d[0] = analytic_space[0]
psy_d[-1] = analytic_space[-1]
hiddenSize = 5
#weights
W1 = np.random.randn(
    1,hiddenSize)# 1x10) weight matrix from input to hidden layer
W2 = np.random.randn(
    hiddenSize,1) # (10x1) weight matrix from hidden to output layer


class Neural_Network(object):

    def left(self,x):
        #return  1/5.0#(2)
        return x + (1. + 3.*x**2) / (1. + x + x**3)#(1)

    def right(self,x):
        #return np.exp(-1*x/5)*np.cos(x)#(2)
        return x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))#(1)

    def func(self,x,psy):
        #return self.right(x) - psy * self.left(x)#(2)
        return self.right(x) - psy * self.left(x) #(1)

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

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return self.sigmoid(s) * (1 - self.sigmoid(s))

    def backward(self, X, ana_space, W1, W2):


        Func =[]
        d_E_net = 0
        err = []
        loss_sum = 0
        err_sqr = 0


        for i in xrange(nx):
            xi = x_space[i]
            o = self.forward(xi,W1,W2)
            psy_t = psy_d[0] + xi*o

            d_out_xi = 0
            for h in xrange(hiddenSize):
                d_out_xi += W1[0][h]*self.sigmoidPrime(self.z[h])*W2[h]

            d_psy_t = o + xi * d_out_xi

            f = self.func(xi, psy_t)
            er = (d_psy_t - f)
            err.append(er)
            err_sqr += np.square(d_psy_t - f)

            loss_sum += np.square(psy_t - ana_space[i])

        error = np.array(err)
        d_E_out = np.zeros_like(error)
        #update weights
        for i in xrange(hiddenSize):
            #iterate input space
            d_E_wo = 0
            d_E_wh = 0
            for j in xrange(len(x_space)):
                xi = x_space[j]
                d_E_out[j] = (error[j])*(1-(-xi*self.left(xi)))
                d_E_wo += np.dot(d_E_out[j],self.sigmoid(np.dot(xi,W1[0][i])))
                d_out_v = W2[i].dot(self.sigmoidPrime(np.dot(xi,W1[0][i]))*xi)
                d_E_wh += d_E_out[j]*d_out_v


            W2[i] -= 0.01*d_E_wo
            W1[0][i] -= 0.01*d_E_wh




        return W1, W2, loss_sum/nx*1.0, psy_t, np.sum(np.square(error))
        #loss_sum/nx*1.0

    def saveWeights(self):
        np.savetxt("w1.txt", W1, fmt="%s")
        np.savetxt("w2.txt", W2, fmt="%s")

    def predict(self):
        print "Predicted data based on trained weights: "
        print "Input (scaled): \n" + str(xPredicted)
        print "Output: \n" + str(self.forward(xPredicted))





NN = Neural_Network()

ittr = 0
max_ittr =500
total_err = np.zeros(2000)
for i in xrange(2000):  # trains the NN 1,000 times

    W1,W2, loss_square, bc, mean_err = NN.backward(x_space,analytic_space,W1,W2)
    total_err[i] = np.log(mean_err)

    if (np.abs(bc-psy_d[-1]) <= 1e-5) & (ittr >= max_ittr):
        print i
        break
    ittr += 1


print "Loss: \n" +str(loss_square)


nn_result = [psy_d[0] + xi * NN.forward(xi,W1,W2) for xi in x_space ]
NN.saveWeights()


plt.figure(1)
plt.plot(x_space, analytic_space, 'b--')
plt.plot(x_space, nn_result,'r--')
plt.legend(('Actural Solution','NN Solution'),loc='upper left')
plt.xlabel("Input space")
plt.ylabel("The value of trial solution.")

plt.figure(2)
plt.plot(total_err)
plt.xlabel("The number of iteration times")
plt.ylabel("The quantity of sum square error")

plt.show()
