import numpy as np
from matplotlib import pyplot as plt
# X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input data for prediction)

def psy_analytic(x):
    return (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2

nx=10

XS=[]
A = []
x_space = np.linspace(0, 1, nx) #discretize space
analytic_space = psy_analytic(x_space)

#for i in range(nx):
#    XS.append([ini_x_space[i]])
#    A.append([ini_analytic_space[i]])
#x_space = np.array(XS,dtype=float)
#analytic_space = np.array(A,dtype=float)


psy_d = np.zeros_like(analytic_space)
psy_d[0] = 1.

#xPredicted = np.array(([4, 8]), dtype=float)

# scale units
#x_space = x_space / np.amax(x_space, axis=0)  # maximum of X array
#xPredicted = xPredicted / np.amax(
    #xPredicted,
    #axis=0)  # maximum of xPredicted (our input data for the prediction)
#analytic_space = analytic_space / 100  # max test score is 100'''


class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 1
        self.outputSize = 1
        self.hiddenSize = 10

        #weights
        self.W1 = np.random.randn(
            self.inputSize,
            self.hiddenSize)  # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(
            self.hiddenSize,
            self.outputSize)  # (3x1) weight matrix from hidden to output layer


    def left(self,x):
        return x + (1. + 3.*x**2) / (1. + x + x**3)

    def right(self,x):
        return x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))

    def func(self,x,psy):
        return self.right(x) - psy * self.left(x)

    def forward(self, X):
        #forward propagation through our network
        f_z3 = []

        self.z = np.dot(
            x_space,
            self.W1[0])  # dot product of X (input) and first set of 3x2 weights


        self.z2 = self.sigmoid(self.z)  # activation function


        for w2 in self.W2:
            f_z3.append(self.z2*w2)
        self.z3 = np.array(f_z3)
        #self.z3 = np.dot(
        #    self.z2, self.W2
        #)  # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3)  # final activation function


        return o

    def sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)


    def backward(self, X, ana_space, o, def_psy, def_psyx,loss_sum):
        # backward propgate through the network
        #self.o_error = ana_space-trial_sol  # error in output
        self.o_error = loss_sum

        o_d = []
        z2_d = []
        for i in xrange(nx):
            o_del = self.o_error * self.sigmoidPrime(
                o[i])  # applying derivative of sigmoid to error
            o_d.append(o_del)
            z2_e = o_del*self.W2[i]

            z2_del = z2_e*self.sigmoidPrime(
                self.z2)
            z2_d.append(z2_del)



        self.o_delta = np.array(o_d)
        self.z2_delta = np.array(z2_d)

        #self.z2_error = np.array(z2_e)
        #self.z2_error = self.o_delta.dot(
        #    self.W2.T)
          # z2 error: how much our hidden layer weights contributed to output error


        #self.z2_delta = self.z2_error * self.sigmoidPrime(
            #self.z2)  # applying derivative of sigmoid to z2 error

        #self.W1 += X.T.dot(
            #self.z2_delta)  # adjusting first set (input --> hidden) weights



        for i in range(nx):
            self.W1[0][i] += -0.001*self.z2_delta[i]
            #self.W2[i] += -self.z2*self.o_delta[i]
            self.W2[i] += -0.001*self.o_delta[i]
            #for j in xrange(self.hiddenSize):
                #self.W2[i] += self.z2*self.o_delta[i]




        #self.W2 += self.z2.T.dot(
            #self.o_delta)  # adjusting second set (hidden --> output) weights


    def loss_func(self, X, ana_space):
        loss_sum = 0
        o = self.forward(x_space)

        Dpsy = []
        Func =[]
        sol = []
        #loss_func(X)
        for i in xrange(nx):
            xi = x_space[i]
            psy_t = psy_d[0] + xi*o[i]
            sol.append(psy_t)

            weightS = self.W1[0][i]*self.W2[i]
            #d_psy_t = o[i] + xi * np.dot(np.dot(self.W1, self.W2),(self.sigmoidPrime(o[i])))
            d_psy_t = o[i] + xi * np.dot(np.dot(self.W1[0][i], self.W2[i]),(self.sigmoidPrime(o[i])))
            Dpsy.append(d_psy_t)

            f = self.func(xi, psy_t)
            Func.append(f)
            err_sqr = (d_psy_t - f)**2
            loss_sum += err_sqr
        trial_sol = np.array(sol,dtype=float)
        def_psy = np.array(Dpsy,dtype=float)
        def_psyx = np.array(Func,dtype=float)

        self.backward(x_space, ana_space, o, def_psy, def_psyx,loss_sum)

        return loss_sum/nx, trial_sol

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def predict(self):
        print "Predicted data based on trained weights: "
        print "Input (scaled): \n" + str(xPredicted)
        print "Output: \n" + str(self.forward(xPredicted))





NN = Neural_Network()
for i in xrange(10):  # trains the NN 1,000 times

    print " #" + str(i) + "\n"
    print "Input (scaled): \n" + str(x_space)
    print "Actual Output: \n" + str(analytic_space)
    loss_square, trial_sol = NN.loss_func(x_space,analytic_space)
    print "Predicted Output: \n" + str(np.array(trial_sol))
    #print "Loss: \n" + str(np.mean(
    #    np.square(analytic_space - NN.forward(x_space))))  # mean sum squared loss
    print "Loss: \n" +str(loss_square)
    print "\n"

nn_result = [1 + x_space[i] * NN.forward(x_space)[i] for i in range(10) ]



print nn_result

plt.figure()
plt.plot(x_space, analytic_space, 'b--')
plt.plot(x_space, nn_result,'r--')
plt.show()

NN.saveWeights()
#NN.predict()

# full tutorial: https://enlight.nyc/neural-network
