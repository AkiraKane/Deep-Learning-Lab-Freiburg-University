__author__ = 'mohamed'
import numpy as np
import _pickle as cPickle
import os
import gzip
import math
import sys
import time
from matplotlib import pyplot as plt

# Loading data from MNIST
def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 1, 28, 28)
    test_y = test_y.astype('int32')

    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 1, 28, 28)
    valid_y = valid_y.astype('int32')

    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 1, 28, 28)
    train_y = train_y.astype('int32')

    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    print('... done loading data')
    return rval



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
    return np.maximum(0.0,x)

def relu_d(x):
    dx = np.zeros(x.shape)
    dx[x > 0 ] = 1
    return dx

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
        self.tname = tname
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
        self.z = input
        # print("Z activated")
        return self.act(input)

    def bprop(self, output_grad):

        return output_grad * self.act_d(self.z)





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
        # print("fprop Input layer")
        return input

    def bprop(self, output_grad):
        # print("bprop Input layer")
        return output_grad








class FullyConnectedLayer(Layer,Parameterized):
    def __init__(self,input_layer,num_units,init_stddev,activation_fun = Activation('sigmoid')):
        self.num_units = num_units
        self.activation_fun = activation_fun
        # the input shape will be of size (batch_size, num_units_prev)
        # where num_units_prev is the number of units in the input
        # (previous) layer
        self.input_shape = input_layer.output_size()
        # print("Input shape =", self.input_shape)

        # this is the weight matrix it should have shape: (num_units_prev, num_units)
        W = np.random.normal(0, init_stddev, self.input_shape[1]*self.num_units)
        W = W.reshape(self.input_shape[1],self.num_units)
        self.W = W
        # print("W shapes:",self.W.shape)
        b = np.random.normal(0, init_stddev, self.num_units)
        self.b = b
        self.dW = None
        self.db = None

    def output_size(self):
        return (self.input_shape[0], self.num_units)

    def fprop(self, input): ########input is the a or z in previous layer and output is the z of this layer
        #
        # implement forward propagation
        ######  NOTE: you should also handle the case were
        #       activation_fun is None (meaning no activation)
        #       then this is simply a linear layer
        #calculate net (z)
        output = np.dot(input,self.W) + self.b
        # print("Calculate Z")
        if self.activation_fun is not None:
            #activation function (a) of net(z)
            # print("Go activate Z to be a")
            output = self.activation_fun.fprop(output)

        # you again want to cache the last_input for the bprop
        # implementation below!
        self.last_input = input #
        # print("last input after activation",self.last_input.shape)

        return output


    def bprop(self, output_grad): #output grad is delta of next layer
        """ Calculate input gradient (backpropagation). """
        # HINT: you may have to divide the weights by n
        #       to make gradient checking work
        #       (since you want to divide the loss by number of inputs)
        if self.activation_fun is not None:
            output_grad = self.activation_fun.bprop(output_grad)

        n = output_grad.shape[0]
        self.dW = np.dot(self.last_input.T, output_grad)/n
        self.db = np.mean(output_grad,axis=0)
        grad_input = np.dot(output_grad,self.W.T)
        # print("last input  = ", self.last_input.shape)
        # print("W shape:",self.W.shape)
        # print("dW shape:",self.dW.shape)
        # print("b shape",self.b.shape)
        # print("db shape",self.db.shape)
        # print("grad output shape =",output_grad.shape)
        # print("grad input shape =",grad_input.shape)

        return grad_input


    def params(self):
        return self.W, self.b

    def grad_params(self):
        return self.dW, self.db


# finally we specify the interface for output layers
# which are layers that also have a loss function
# we will implement two output layers:
#  a Linear, and Softmax (Logistic Regression) layer
# The difference between output layers and and normal
# layers is that they will be called to compute the gradient
# of the loss through input_grad(). bprop will never
# be called on them!
class LinearOutput(Layer, Loss):
    """ A simple linear output layer that
        uses a squared loss (e.g. should be used for regression)
    """
    def __init__(self, input_layer):
        self.input_size = input_layer.output_size()

    def output_size(self):
        return (1,)

    def fprop(self, input):
        #print("Fprop Linear output")
        return input

    def bprop(self, output_grad):
        raise NotImplementedError(
            'LinearOutput should only be used as the last layer of a Network'
            + ' bprop() should thus never be called on it!'
        )

    def input_grad(self, Y, Y_pred):
        #implement gradient of squared loss
        return Y_pred - Y

    def loss(self, Y, Y_pred):
        loss = 0.5 * np.square(Y - Y_pred)
        return np.mean(np.sum(loss, axis=1))





class SoftmaxOutput(Layer, Loss):
    """ A softmax output layer that calculates
        the negative log likelihood as loss
        and should be used for classification.
    """

    def __init__(self, input_layer):
        self.input_size = input_layer.output_size()

    def output_size(self):
        return (1,)

    def fprop(self, input):
       # print("Fprop Softmax output")
        return softmax(input)

    def bprop(self, output_grad):
        raise NotImplementedError(
            'SoftmaxOutput should only be used as the last layer of a Network'
            + ' bprop() should thus never be called on it!'
        )

    def input_grad(self, Y, Y_pred):
        # HINT: since this would involve taking the log
        #       of the softmax (which is np.exp(x)/np.sum(x, axis=1))
        #       this gradient computation can be simplified a lot!
        return (Y_pred - Y)

    def loss(self, Y, Y_pred):
        # Assume one-hot encoding of Y
        # calculate softmax first
        out = softmax(Y_pred)   # WHY ??? Y_pred is already a softmax function
        # to make the loss numerically stable
        # you may want to add an epsilon in the log ;)
        eps = 1e-10

        # calculate negative log likelihood
        # sum of the each elements in every examples
        loss = - np.sum(Y * np.log(Y_pred+eps),axis=1)

        #sum of all the examples / number of examples
        return np.mean(loss)




class NeuralNetwork:
    """ Our Neural Network container class.
    """
    def __init__(self, layers):
        self.layers = layers


    def _loss(self, X, Y):
        Y_pred = self.predict(X)
        return self.layers[-1].loss(Y, Y_pred)


    def predict(self, X):
       # print("Entered predict (X)")
        """ Calculate an output Y for the given input X. """
        # forward pass through all layers
        input = X
        #print("Start fprop through all layers")
        for l,layer in enumerate(self.layers):
         #   print("========================")
          #  print("Go fprop for layer", l+1)
            input = layer.fprop(input)


        Y_pred = input
        return Y_pred


    def backpropagate(self, Y, Y_pred, upto=0):
        """ Backpropagation of partial derivatives through
            the complete network up to layer 'upto'
        """
        next_grad = self.layers[-1].input_grad(Y, Y_pred)
        #i = 4
        for layer in reversed((self.layers[upto:-1])):
           # print("=================================")
            #print("layer",i)
            #i-=1
            next_grad = layer.bprop(next_grad)

        return next_grad


    def classification_error(self, X, Y):
        """ Calculate error on the given data
            assuming they are classes that should be predicted.
        """
        Y_pred = unhot(self.predict(X))
        error = Y_pred != Y
        return np.mean(error)


    def get_all_params(self):
        params_all = np.array([])
        for layer in self.layers:
            if isinstance(layer,Parameterized):
                for params in layer.params():
                    #first get the parameters
                    # print(params.shape)
                    # print(params)
                    #add W first then b of each layer
                    params_all = np.append(params_all,np.ravel(params))
        # print("all")
        #print(params_all)
        #print(params_all.shape)
        return params_all

    def get_all_grads(self):
        grads_all = np.array([])
        for layer in self.layers:
            if isinstance(layer,Parameterized):
                for grads in layer.grad_params():
                    #first get the grads
                    # print(grads.shape)
                    # print(grads)
                    #add W first then b of each layer
                    grads_all = np.append(grads_all,np.ravel(grads))
        # print("all")
        # print(grads_all)
        # print(grads_all.shape)
        return grads_all


    def set_all_params(self,params_all):
        num_prev_weights = 0
        for layer in self.layers:
             if isinstance(layer,Parameterized):
                 for p,params in enumerate(layer.params()):
                     if params.ndim == 2:
                         size = params.shape[0] * params.shape[1]
                     else:
                        size = params.shape[0]

                     # print(params.shape)
                     params[:] = np.reshape(params_all[num_prev_weights:size+num_prev_weights]
                                            ,params.shape)
                     num_prev_weights+=size
                     # print(params)


    def sgd_epoch(self, X, Y, learning_rate, batch_size):
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        for b in range(n_batches):

            # Implement stochastic gradient descent here
            # start by extracting a batch from X and Y
            # (you can assume the inputs are already shuffled)
            X_batch = X[b*batch_size:b*batch_size + batch_size,:]
            Y_batch = Y[b*batch_size:b*batch_size + batch_size,:]
            # print("batch number",b)
            # print("X batch",X_batch)
            # print("Y batch",Y_batch)

            # full forward propagation
            Y_pred = self.predict(X_batch)

            # full backward propagation
            self.backpropagate(Y_batch,Y_pred)
            params_all = self.get_all_params()
            grads_all = self.get_all_grads()
            params_all = params_all - learning_rate*grads_all
            self.set_all_params(params_all)





    def gd_epoch(self, X, Y, learning_rate,error,max_iter = 100000):
        # TODO ##################################################
        # Implement batch gradient descent here
        # A few hints:
        #   There are two strategies you can follow:
        #   Either shove the whole dataset throught the network
        #   at once (which can be problematic for large datasets)
        #   or run through it batch  ise as in the sgd approach
        #   and accumulate the gradients for all parameters as
        #   you go through the data. Either way you should then
        #   do one gradient step after you went through the
        #   complete dataset!
        # i = 0
        # converged = False
        # costs = []
        # cost_old = self._loss(X,Y)
        # costs.append(cost_old)
        #
        # while not converged:
        #
        #     partial_deravatives = self.layers
        #     W = N.get_weights()
        #     W = W - alpha*partial_deravatives
        #     N.set_weights(W)
        #     cost_new = N.J(x,y)
        #     costs.append(cost_new)
        #     i+=1
        #     print("Iteration ", i, ", Cost: ", cost_new)
        #     if((cost_old-cost_new)**2 < error):
        #         print("Converged, iterations: ", i)
        #         converged = True
        #
        #
        #     if(i > max_iter):
        #         print("Maximum iterations exceeded !")
        #         converged = True
        #
        #     cost_old = cost_new
        # return  costs
        pass
        # TODO ##################################################




    def train(self, X, Y, learning_rate=0.1, max_epochs=100,
              batch_size=64, descent_type="sgd", y_one_hot=True):
        """ Train network on the given data. """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        if y_one_hot:
            Y_train = one_hot(Y)
        else:
            Y_train = Y

        print("... starting training")
        for e in range(max_epochs+1):
            if descent_type == "sgd":
                self.sgd_epoch(X, Y_train, learning_rate, batch_size)
            elif descent_type == "gd":
                self.gd_epoch(X, Y_train, learning_rate)
            else:
                raise NotImplementedError("Unknown gradient descent type {}".format(descent_type))

            # Output error on the training data
            train_loss = self._loss(X, Y_train)
            train_error = self.classification_error(X, Y)
            print('epoch {:.4f}, loss {:.4f}, train error {:.4f}'.format(e, train_loss, train_error))
            # TODO ##################################################
            # compute error on validation data:
            # simply make the function take validation data as input
            # and then compute errors here and print them
            # TODO ##################################################


    def test(self,X,Y):
        Y_predict = self.predict(X)
        Y_predict = unhot(Y_predict)
        print("====================")
        print("Examples :")
        for i in range(0,len(Y)):
            print("Example number",i+1,"| Real Target:",Y[i],
                  "| Predicted:", Y_predict[i])

        h = np.zeros(Y.shape)
        h[Y != Y_predict] = 1

        j = 0
        for i in range(0,len(Y)):
            if(h[i] == 1):
                j+=1
        print("j",j)
        print(j/Y.shape[0])

        error = Y!= Y_predict
        error = np.mean(error)
        print("error",error)
        print("Examples correctly predicted: ", (1-error )*100,"%")


    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for
        correctness. """
        for l, layer in enumerate(self.layers):

            if isinstance(layer, Parameterized):
                print("=====================")
                print('checking gradient for layer {}'.format(l))
                for p, param in enumerate(layer.params()):
                     # 1st iter is the W, second is the b
                    # we iterate through all parameters
                    param_shape = param.shape

                    # define functions for conveniently swapping
                    # out parameters of this specific layer and
                    # computing loss and gradient with these
                    # changed parametrs
                    def output_given_params(param_new):
                        """ A function that will compute the output
                            of the network given a set of parameters
                        """
                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # return computed loss
                        return self._loss(X, Y)

                    def grad_given_params(param_new):
                        """A function that will compute the gradient
                           of the network given a set of parameters
                        """

                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # Forward propagation through the net
                        #print("Go enter dict(X)")
                        Y_pred = self.predict(X)
                        # Backpropagation of partial derivatives
                        self.backpropagate(Y, Y_pred, upto=l)
                        # return the computed gradient
                        return np.ravel(self.layers[l].grad_params()[p])

                    # let the initial parameters be the ones that
                    # are currently placed in the network and flatten them
                    # to a vector for convenient comparisons, printing etc.
                    param_init = np.ravel(np.copy(param))

                    #
                    #       compute the gradient with respect to
                    #      the initial parameters in two ways:
                    #      1) with grad_given_params()
                    #get the derivatives
                    grad_param_init = grad_given_params(param_init)

                    #      2) with finite differences
                    #         using output_given_params()
                    #         (as discussed in the lecture)
                    #      if your implementation is correct
                    #      both results should be epsilon close
                    #      to each other!

                    epsilon = 1e-5
                    # making sure your gradient checking routine itself
                    # has no errors can be a bit tricky. To debug it
                    # you can "cheat" by using scipy which implements
                    # gradient checking exactly the way you should!
                    # To do that simply run the following here:
                    import scipy.optimize
                    err = scipy.optimize.check_grad(output_given_params,
                                                    grad_given_params, param_init)
                    print("Cheat gradient check error =",err)


                    # finite diff
                    gparam_fd = np.zeros_like(param_init)
                    perturb = np.zeros(param_init.shape)
                    for i in range(len(param_init)):

                        perturb[i] = epsilon
                        loss_plus = output_given_params(param_init + perturb)
                        loss_minus = output_given_params(param_init - perturb)
                        gparam_fd[i] = (loss_plus - loss_minus) / (2*epsilon)
                        perturb[i] = 0
                    gparam_fd = gparam_fd.reshape(param_shape)

                    # gradient as calculated through bprop
                    gparam_bprop = grad_param_init.reshape(param_shape)
                    # calculate difference between them
                    err = np.mean(np.abs(gparam_bprop - gparam_fd))
                    print('Implemented Gradient check error {:.2e}'.format(err))
                    assert(err < epsilon)

                    # reset the parameters to their initial values
                    param[:] = np.reshape(param_init, param_shape)



# Setup a small MLP / Neural Network
# we can set the first shape to None here to indicate that
# we will input a variable number inputs to the network
# load
# Dtrain, Dval, Dtest = mnist()
# X_train, y_train = Dtrain
# # Downsample training data to make it a bit faster for testing this code
# n_train_samples = 10000
# train_idxs = np.random.permutation(X_train.shape[0])[:n_train_samples]
# X_train = X_train[train_idxs]
# y_train = y_train[train_idxs]
#
# print("X_train shape: {}".format(np.shape(X_train)))
# print("y_train shape: {}".format(np.shape(y_train)))
#
# X_train = X_train.reshape(X_train.shape[0], -1)
# print("Reshaped X_train size: {}".format(X_train.shape))


#Gradient Checking

# input_shape = (5, 10)
# n_labels = 6
# layers = [InputLayer(input_shape)]
#
# layers.append(FullyConnectedLayer(
#                 layers[-1],
#                 num_units=15,
#                 init_stddev=0.1,
#                 activation_fun=Activation('sigmoid')
# ))
# layers.append(FullyConnectedLayer(
#                 layers[-1],
#                 num_units=6,
#                 init_stddev=0.1,
#                 activation_fun=Activation('sigmoid')
# ))
# layers.append(FullyConnectedLayer(
#                 layers[-1],
#                 num_units=n_labels,
#                 init_stddev=0.1,
#                 activation_fun= None
# ))
# layers.append(SoftmaxOutput(layers[-1]))
# nn = NeuralNetwork(layers)
#
# # create random data
# X = np.random.normal(size=input_shape)
# # and random labels
# Y = np.zeros((input_shape[0], n_labels))
#
# for i in range(Y.shape[0]):
#     idx = np.random.randint(n_labels)
#     Y[i, idx] = 1.
#
#
# nn.check_gradients(X,Y)
# nn.train(X, Y, learning_rate=0.1,
#           max_epochs=20, batch_size=64, y_one_hot=False)



# #Training on MNIST
# # load
Dtrain, Dval, Dtest = mnist()
X_train, y_train = Dtrain
# Downsample training data to make it a bit faster for testing this code
n_train_samples = 10000
train_idxs = np.random.permutation(X_train.shape[0])[:n_train_samples]
X_train = X_train[train_idxs]
y_train = y_train[train_idxs]
print("X_train shape: {}".format(np.shape(X_train)))
print("y_train shape: {}".format(np.shape(y_train)))
X_train = X_train.reshape(X_train.shape[0], -1)
print("Reshaped X_train size: {}".format(X_train.shape))

# Setup a small MLP / Neural Network
# we can set the first shape to None here to indicate that
# we will input a variable number inputs to the network
input_shape = (None, 28*28)
layers = [InputLayer(input_shape)]
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=100,
                init_stddev=0.01,
                activation_fun=Activation('relu')
))
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=100,
                init_stddev=0.01,
                activation_fun=Activation('relu')
))
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=10,
                init_stddev=0.01,
                # last layer has no nonlinearity
                # (softmax will be applied in the output layer)
                activation_fun= None
))
layers.append(SoftmaxOutput(layers[-1]))
nn = NeuralNetwork(layers)

#nn.check_gradients(X_train, one_hot(y_train))
# Train neural network
t0 = time.time()
nn.train(X_train, y_train, learning_rate=0.1,
        max_epochs=20, batch_size=20, y_one_hot=True)
t1 = time.time()
print('Duration: {:.1f}s'.format(t1-t0))

nn.test(X_train,y_train)


# 7war el gradient checking shwyt printing
