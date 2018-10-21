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
        self.W1 = np.random.rand(
            1,nx)# 1x10) weight matrix from input to hidden layer
        #print self.W1
        #self.W1 = [np.linspace(-0.1, 0.1, nx)]
        #print self.W1
        self.W2 = np.random.rand(
            nx,1) # (10x1) weight matrix from hidden to output layer



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
            X,
            self.W1[0])  # dot product of X (input) and first set of 3x2 weights


        self.z2 = self.sigmoid(self.z)  # activation function

        for i in xrange(nx):
            f_z3.append(self.z2*self.W2[i])
        self.z3 = np.array(f_z3)

        o = self.z3

        return o

    def sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return self.sigmoid(s) * (1 - self.sigmoid(s))

    def backward(self, X, ana_space):
        loss_sum = 0
        o = self.forward(x_space)

        Func =[]
        d_E_net = 0

        for i in xrange(nx):
            xi = x_space[i]
            psy_t = psy_d[0] + xi*o[i]
            #O_neural= self.forward(xi)[0][0]
            #psy_t = psy_d[0] + xi*O_neural

            #sol.append(psy_t)
            d_psy_t = o[i] + xi * np.dot(np.dot(self.W2[:].T,self.W1[0].T),(self.sigmoidPrime(self.z)))
            #d_psy_t = O_neural + xi * np.dot(np.dot(self.W2[:].T,self.W1[0].T),(self.sigmoidPrime(self.z)))
            #d_psy_t = o[i] + xi * (self.W1[0][i] * self.sigmoidPrime(self.z) * self.W2[i])

            f = self.func(xi, psy_t)
            Func.append(f)

            #d_psy_t = o[i] + xi * np.dot(xi,(self.sigmoidPrime(o[i])))
            d_E_psy = 2*(d_psy_t-f)
            d_psy_out = 1-(-xi*self.left(xi))
            #d_out_net = self.sigmoidPrime(o[i])
            d_out_wo = self.z2
            d_E_wo = d_E_psy*d_psy_out*d_out_wo
            d_E_net += d_E_psy*d_psy_out*self.W2[i]
            self.W2[i] -= 0.01*d_E_wo
            d_ho_hn = self.sigmoidPrime(self.z)
            err_sqr = (psy_t - ana_space[i])**2
            loss_sum += err_sqr

        for i in xrange(nx):
            xi = x_space[i]
            d_ho_hn = self.sigmoidPrime(self.z)
            d_hn_w = xi
            d_E_wh=d_E_net*d_ho_hn*d_hn_w
            #d_E_wh=self.W2[i]*d_ho_hn*d_hn_w
            self.W1[0][i] -= 0.01*d_E_wh




        return 1.0*(loss_sum)/nx

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def predict(self):
        print "Predicted data based on trained weights: "
        print "Input (scaled): \n" + str(xPredicted)
        print "Output: \n" + str(self.forward(xPredicted))





NN = Neural_Network()


#for i in xrange(1000):  # trains the NN 1,000 times
bc = psy_analytic(x_space[-1])
count = 0
end_psy = psy_d[-1]
diff = abs(end_psy - bc*1.0)

while count <= 1000:

    final_o = NN.forward(x_space)
    loss_square = NN.backward(x_space,analytic_space)
    end_psy = psy_d[0] + x_space[nx-1] * final_o[nx-1]
    diff = abs(end_psy - bc*1.0)
    count += 1

print end_psy
print "Loss: \n" +str(loss_square)

nn_result = [psy_d[0] + x_space[i] * final_o[i] for i in xrange(nx) ]

#print "Actual Result", analytic_space.T
#print "Predicted Result", nn_result





#print nn_result

plt.figure()
plt.plot(x_space, analytic_space, 'b--')
plt.plot(x_space, nn_result,'r--')
plt.legend(('Actural Solution','NN Solution'),loc='upper left')
plt.xlabel("Input space")
plt.ylabel("The value of trial solution.")

plt.show()

NN.saveWeights()
#NN.predict()

# full tutorial: https://enlight.nyc/neural-network
