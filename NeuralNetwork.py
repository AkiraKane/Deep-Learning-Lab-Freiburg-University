__author__ = 'mohamed'
import numpy as np
from matplotlib import pyplot as plt


# 1) Randomly initialize Weights
# 2) Implement fprop to get vector of outputs to any vector of inputs
# 3) Implement Code to compute cost function (vector of outputs from (2) - real outputs from the data itself)
# 4) Implemet backprop to get partial derivatives for each parameter theta
# 5) use gradient checking to compare between partial derivatives from backprop vs. using numerical  estimate  of gradient of cost function
# then disable gradient checking code as it takes time each time you will train the network
# 6) use gradient descent/stochastic gd  with backprop to try and minimize the cost function


# define Activation functions
def sigmoid(x):
    return 1/(1+ np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_d(x):
    return 1-pow(tanh(x),2)

def relu(x): #rectified linear unit
    return np.maximum(0.0,x)

def relu_d(x):
    if x > 0:
        return 1
    else:
        return 0

#don't forget feature scaling
class NeuralNetwork(object):
    def __init__(self,x,y,activation_name):
        self.inputLayerSize = x.shape[1]
        self.numOfHiddenLayers = 1
        self.hiddenLayerSize = 2
        self.outputLayerSize = y.shape[1]
        self.numOfExamples = x.shape[0]

        #1st Define weights
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    #2nd forward prop
    def forward(self,X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = sigmoid(self.z3)
        return yHat

    #3rd Cost function
    def J(self,X,y):
        self.yHat = self.forward(X)
        error = sum((self.yHat-y)**2)#np.dot(y.T,np.log(self.yHat)) + np.dot((1-y.T),np.log(1-self.yHat))
        return 0.5*error #(-1/self.numOfExamples) * error


    #4th back prop to
    def dJ_dW(self,X,y):
        delta3 = np.multiply(self.yHat-y,sigmoid_d(self.z3)) # element wise multiplication
        dJ_dW2 = np.dot(self.a2.T,delta3) # matrix maltiplication

        delta2 = np.dot(delta3,self.W2.T)*sigmoid_d(self.z2)
        dJ_dW1 = np.dot(X.T,delta2)

        return dJ_dW1,dJ_dW2

    def get_weights(self):
        #get W1 and W2 unrolled into a vector
        weights = np.concatenate((self.W1.ravel(),self.W2.ravel()))
        return weights

    def set_weights(self,weights):
        #set the weights of the neural network
        W1_start = 0
        W1_end = self.inputLayerSize*self.hiddenLayerSize
        self.W1 = np.reshape(weights[W1_start:W1_end],self.W1.shape)

        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(weights[W1_end:W2_end],self.W2.shape)

    def unroll_dJdW(self,X,y):
        dJW1,dJW2 = self.dJ_dW(X,y)
        return np.concatenate((dJW1.ravel(),dJW2.ravel()))


#5th gradient check (not in the class)
def grad_check(N,X,y):
    e = 1e-4
    weights_initial = N.get_weights()

    numerical_der = np.zeros(weights_initial.shape)
    perturb = np.zeros(weights_initial.shape)


    for i in range(len(weights_initial)):
        perturb[i] = e
        N.set_weights(weights_initial + perturb)
        plus = N.J(X,y)
        N.set_weights(weights_initial - perturb)
        minus = N.J(X,y)

        numerical_der[i] = (plus - minus)/(2*e)

        perturb[i] = 0
    N.set_weights(weights_initial)

    return numerical_der



#6th Gradient descent
def gd(N,x,y,alpha,error = 1e-10,max_iter = 100000):
    i = 0
    converged = False
    costs = []
    cost_old = N.J(x,y)
    costs.append(cost_old)

    while not converged:


        partial_deravatives = N.unroll_dJdW(x,y)
        W = N.get_weights()
        W = W - alpha*partial_deravatives
        N.set_weights(W)
        cost_new = N.J(x,y)
        costs.append(cost_new)
        i+=1

        if((cost_old-cost_new)**2 < error):
            print("Converged, iterations: ", i)
            converged = True


        if(i > max_iter):
            print("Maximum iterations exceeded !")
            converged = True

        cost_old = cost_new
    return  costs


x = np.array(([0,0],[0,1],[1,0],[1,1]),dtype=float)
y = np.array(([0],[1],[1],[0]),dtype=float)
alpha = 0.3
max_iter = 5000
NN = NeuralNetwork(x,y,"sigmoid")
cost_init = NN.J(x,y)

#5th gradient check
# print(NN.unroll_dJdW(x,y))
# print(grad_check(NN,x,y))
# print(sum(NN.unroll_dJdW(x,y) - grad_check(NN,x,y)))

#gradient decent
costs= gd(NN,x,y,alpha)
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

