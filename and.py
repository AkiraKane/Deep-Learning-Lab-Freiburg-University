__author__ = 'mohamed'
import numpy as np

# 1) Randomly initialize Weights
# 2) Implement fprop to get vector of outputs to any vector of inputs
# 3) Implement Code to compute cost function (vector of outputs from (2) - real outputs from the data itself)
# 4) Implemet backprop to get partial derivatives for each parameter theta
# 5) use gradient checking to compare between partial derivatives from backprop vs. using numerical  estimate  of gradient of cost function
# then disable gradient checking code as it takes time each time you will train the network
# 6) use gradient descent/stochastic gd  with backprop to try and minimize the cost function



def sigmoid(x):
    return (1/(1+np.exp(-x)))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


def weights_initialize(dim,epsilon):
    W = np.random.randn(1,dim)*2*epsilon - epsilon
    return W

def fprop(x_inputs,W):
    Z_2 = W.dot( x_inputs.T)
    a_2 = sigmoid(Z_2)
    a_2 = np.asarray(a_2)
    return a_2.T,Z_2.T

def cost_function(y,output):
    error = (((y.T).dot)(np.log(output))) + (1-y.T).dot(((np.log(1-output))))
    return -(1/m)*error


def bprop(y,output_fprop,Z_2,a1):
    dJ_da2 = -(output_fprop-y)
    delta2 = np.multiply(dJ_da2,sigmoid_der(Z_2))
    dJ_dW1 = a1.T.dot(delta2)

    return dJ_dW1


#and data
x = np.array([0, 0, 0, 1, 1, 0, 1, 1])
x = x.reshape(4,2)

y = np.array([0,0,0,1])
y = y.reshape(4,1)
#adding x0
x_zero = np.ones(shape=(x.shape[0],1))
x = np.append(x_zero,x,axis=1)

m = x.shape[0] #num of examples
n = x.shape[1] # num of features


#1st initialize weights
W = weights_initialize(x.shape[1],0.2)

#2nd fprop
output_fprop,Z_2 = fprop(x,W) #output from all examples


#3rd Cost function
print(cost_function(y,output_fprop))


#4th backprop
dJ_dW1 = bprop(y,output_fprop,Z_2,x)


#5th gradient checking


#6th gradient descent









