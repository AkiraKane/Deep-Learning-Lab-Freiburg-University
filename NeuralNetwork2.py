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
    return 1-((tanh(x))**2)

def relu(x): #rectified linear unit
    return np.maximum(0.0,x) #needs ADJUSTMENT

def relu_d(x):
    if x > 0:  #NEEDS adjustment
        return 1
    else:
        return 0

def softmax(x, axis=1):
    # to make the softmax a "safe" operation we will
    # first subtract the maximum along the specified axis
    # so that np.exp(x) does not blow up!
    # Note that this does not change the output.
    x_max = np.max(x, axis=axis, keepdims=True)
    x_safe = x - x_max
    e_x = np.exp(x_safe)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels

def unhot(one_hot_labels):
    """ Invert a one hot encoding, creating a flat vector """
    return np.argmax(one_hot_labels, axis=-1)





# then define an activation function class
class Activation(object):

    def __init__(self, tname):
        if tname == 'sigmoid':
            self.act = sigmoid
            self.act_d = sigmoid_d
        elif tname == 'tanh':
            self.act = tanh
            self.act_d = tanh_d
        elif tname == 'relu':
            self.act = relu
            self.act_d = relu_d
        else:
            raise ValueError('Invalid activation function.')

    def fprop(self, input):
        # we need to remember the last input
        # so that we can calculate the derivative with respect
        # to it later on
        self.last_input = input
        return self.act(input)

    def bprop(self, output_grad):
        return output_grad * self.act_d(self.last_input)





#don't forget feature scaling

# define a base class for layers
class Layer(object):

    def fprop(self, input):
        """ Calculate layer output for given input
            (forward propagation).
        """
        raise NotImplementedError('This is an interface class, please use a derived instance')

    def bprop(self, output_grad):
        """ Calculate input gradient and gradient
            with respect to weights and bias (backpropagation).
        """
        raise NotImplementedError('This is an interface class, please use a derived instance')

    def output_size(self):
        """ Calculate size of this layer's output.
        input_shape[0] is the number of samples in the input.
        input_shape[1:] is the shape of the feature.
        """
        raise NotImplementedError('This is an interface class, please use a derived instance')





# define a base class for loss outputs
# an output layer can then simply be derived
# from both Layer and Loss
class Loss(object):

    def loss(self, output, output_net):
        """ Calculate mean loss given real output and network output. """
        raise NotImplementedError('This is an interface class, please use a derived instance')

    def input_grad(self, output, output_net):
        """ Calculate input gradient real output and network output. """
        raise NotImplementedError('This is an interface class, please use a derived instance')





# define a base class for parameterized things
class Parameterized(object):

    def params(self):
        """ Return parameters (by reference) """
        raise NotImplementedError('This is an interface class, please use a derived instance')

    def grad_params(self):
        """ Return accumulated gradient with respect to params. """
        raise NotImplementedError('This is an interface class, please use a derived instance')






class InputLayer(Layer):
    def __init__(self,input_shape):
        if not isinstance(input_shape,tuple):
            raise ValueError("Input layer requires input shape as tuple")
        self.input_shape = input_shape

    def output_size(self):
        return self.input_shape

    def fprop(self, input):
        return input

    def bprop(self, output_grad):
        return output_grad








class FullyConnectedLayer(Layer,Parameterized):
    def __init__(self,input_layer,num_units,init_stddev,activation_fun = Activation('sigmoid')):
        self.num_units = num_units
        self.activation_fun = activation_fun
        # the input shape will be of size (batch_size, num_units_prev)
        # where num_units_prev is the number of units in the input
        # (previous) layer
        self.input_shape = input_layer.output_size()
        # this is the weight matrix it should have shape: (num_units_prev, num_units)
        W = np.random.normal(0, init_stddev, self.input_shape[1]*self.num_units)
        W.reshape(self.input_shape[1]*self.num_units)
        self.W = W
        b = np.random.normal(0, init_stddev, self.num_units)
        self.b = b
        self.dW = None
        self.db = None

    def output_size(self):
        return (self.input_shape[0], self.num_units)


    def fprop(self, input): ########input X or Z's
        # TODO ################################################
        # TODO: implement forward propagation
        ############################################# NOTE: you should also handle the case were
        #       activation_fun is None (meaning no activation)
        #       then this is simply a linear layer
        z = np.dot(input,self.W) + self.b
        a = self.activation_fun.fprop(z)
        # you again want to cache the last_input for the bprop
        # implementation below!
        self.last_input = z
        return a


    def bprop(self, output_grad):
        """ Calculate input gradient (backpropagation). """
        delta = self.activation_fun.bprop(output_grad)

        # HINT: you may have to divide the weights by n
        #       to make gradient checking work
        #       (since you want to divide the loss by number of inputs)
        n = output_grad.shape[0]

        self.dW = np.dot(,delta)
        self.db = delta
        grad_input = np.dot(delta,self.W.T)
        return grad_input



    def params(self):
        return self.W, self.b

    def grad_params(self):
        return self.dW, self.db



class NeuralNetwork(object):
    def __init__(self,x,y,activation_name):
        self.inputLayerSize = x.shape[1]
        self.numOfHiddenLayers = 1
        self.hiddenLayerSize = 2
        self.outputLayerSize = y.shape[1]
        self.numOfExamples = x.shape[0]
        if(activation_name == 'sigmoid'):
            self.act = sigmoid
            self.act_d = sigmoid_d
        elif(activation_name == 'tanh'):
            self.act = tanh
            self.act_d = tanh_d
        elif(activation_name == 'relu'):
            self.act = relu
            self.act_d = relu_d
        else:
            raise ValueError("Invalid activation function")


        #1st initialize the weights
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    #2nd forward prop
    def fprop(self,X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.act(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.act(self.z3)
        return yHat

    #3rd Cost function
    def J(self,X,y):
        self.yHat = self.fprop(X)
        error = sum((self.yHat-y)**2)#np.dot(y.T,np.log(self.yHat)) + np.dot((1-y.T),np.log(1-self.yHat))
        return 0.5*error #(-1/self.numOfExamples) * error


    #4th back prop
    def dJ_dW(self,X,y):
        delta3 = np.multiply(self.yHat-y,self.act_d(self.z3)) # element wise multiplication
        dJ_dW2 = np.dot(self.a2.T,delta3) # matrix maltiplication

        delta2 = np.dot(delta3,self.W2.T)*self.act_d(self.z2)
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
        print("Iteration ", i, ", Cost: ", cost_new)
        if((cost_old-cost_new)**2 < error):
            print("Converged, iterations: ", i)
            converged = True


        if(i > max_iter):
            print("Maximum iterations exceeded !")
            converged = True

        cost_old = cost_new
    return  costs

#7th predict
def target_check(N, x, y):
    pred = N.fprop(x)
    print("========================================")
    print("Patterns:")
    for i in range(0,y.shape[0]):
        print(" pattern",i+1,":",x[i],y[i],"prediction ->",pred[i])
    print("========================================")


x = np.array(([0,0],[0,1],[1,0],[1,1]),dtype=float)
y = np.array(([0],[1],[1],[1]),dtype=float)
alpha = 0.3
max_iter = 5000
NN = NeuralNetwork(x,y,"sigmoid")
cost_init = NN.J(x,y)

#5th gradient check
# print(NN.unroll_dJdW(x,y))
# print(grad_check(NN,x,y))
# print(sum(NN.unroll_dJdW(x,y) - grad_check(NN,x,y)))

#gradient decent
# costs= gd(NN,x,y,alpha)
# plt.plot(costs)
# plt.xlabel("Iterations")
# plt.ylabel("Cost")
# plt.show()
#
# prediction = target_check(NN,x,y)

layers = [InputLayer(x.shape)]
