# Euler+DL to implement deepl
import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import symbols, diff



class Function(object):

    def coefficient(self,x):

                #return 1 #(3)
        return x + (1. + 3.*x**2) / (1. + x + x**3)#(1)
            #return 1./5#(2)
                #return 1/5 #(3)
                #return  1/5.0#(2)

    def right(self,x):
            #return (-1/5.0)*np.exp(-x/5.0)*np.cos(x)#(3)
            #return np.exp(-1*x/5)*np.cos(x)#(2)
        return x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))#(1)

    def func(self,x, ts):
            #return self.right(x)-ts*self.left(x,1)-d_ts * self.left(x,2) #(3)
            #return (self.right(x)-ts)/self.coefficient(x)#(2)
        return (self.right(x)-ts)/self.coefficient(x) #(1)

def hiddenLayer_size(num_ly):

    num_layer = np.zeros(int(num_ly))

    for l in xrange(num_ly):
        num_unit = raw_input("Define the number of hidden units in Layer {}:".format(l))
        num_layer[l]=num_unit

    return num_layer



def add_hidden(layer, dim, W,activation='relu'):

    layer = np.dot(layer,W)



    if activation == 'relu':
        for l in np.nditer(layer):

            lr = np.maximum(0.01*l,l)

            if lr < 0:
                deriv=0.01
            else:
                deriv = 1

    return layer,deriv

    if activation == 'sigmoid':
        for lr in layer:
            lr = 1/(1+np.exp(-lr))

        return layer

def output(layer,dim,W):
    return np.dot(layer,W).sum()

def update_weight(x_space, layer1, der1,layer2,der2,error,num_hid,W0,W1,OutW):


    OutW_delta = np.dot(error.T,layer2) #(0x3 and 1x3)


    w1_ly2 = np.dot(error.T,np.dot(der2,OutW))

    w1_delta = w1_ly2.T.dot(der1)

    w0_ly1 = np.dot(w1_ly2,np.dot(der1,W1))
    w0_delta = np.dot(x_space.T,w0_ly1)

    return w0_delta, w1_delta, OutW_delta

def trial_sol(u0,du,h):
    return u0+du*h


nx=10
start = 0
end = 2.
hidden_ly = 2
init_con=0
r=0.05
max_ittr = 10
x_space = np.linspace(start, end, nx)
euler_learning_rate = (end-start)/nx
error = np.zeros(nx)
sqr_err = np.zeros(max_ittr)

Func = Function()

for i in xrange(max_ittr):  # trains the NN 1,000 times

    if i == 0 :
        output_unit = hiddenLayer_size(hidden_ly)
        sum_layer = np.zeros(hidden_ly)
        W0 = np.random.randn(1,int(output_unit[0]))
        W1 = np.random.randn(int(output_unit[0]),int(output_unit[1]))

        OutW= np.random.randn(int(output_unit[1]),1)
        layer1 = np.zeros((nx,int(output_unit[0])))
        der1 = np.zeros((nx,int(output_unit[0])))
        layer2 = np.zeros((nx,int(output_unit[1])))
        der2 = np.zeros((nx,int(output_unit[1])))
    for j in xrange(nx):

        input = x_space[j]
        r = 0.05
        layer1[j],der1[j] = add_hidden(input, output_unit[0],W0 , activation='relu')
        layer2[j],der2[j] = add_hidden(layer1[j], output_unit[1],W1 , activation='relu')
        deriv1 = output(layer2[j], 1, OutW)

        if j == 0:
            measured = init_con
        else:
            measured = trial_sol(input,deriv1,euler_learning_rate)

        error[j] = Func.func(input,deriv1)-measured

    d_W0,d_W1,d_OutW = update_weight(x_space, layer1, der1,layer2,der2,error,hidden_ly,W0,W1,OutW)
    sqr_err[i] = np.mean(np.square(error))

    W0 -= r*-h*d_W0
    W1 -= r*-h*d_W1
    OutW -= r*-H*D_OutW
