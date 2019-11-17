import numpy as np
import random
import math

def sigmoid(z,gamma): # sigmoid function, gamma is hyperparameter
    return 1.0/(1.0+ np.exp(-gamma*z))
def sigmoid_prime (z,gamma): # derivation of sigmoid function
    """Derivative of the sigmoid function."""
    return gamma*(sigmoid(gamma*z)*(1- sigmoid(gamma*z)))
def tanh(z,gamma): # tangent hyperbolic function, gamma is hyperparameter
    return (np.exp(2.0*gamma*z) - 1.0)/(np.exp(2.0*gamma*z) + 1.0)
def derivative_tanh(z,gamma): # derivation of tanh function, gamma is hyperparameter
    return gamma*(1.0 - (tanh(gamma*z))**2.0)


class Network():

    def __init__(self , sizes, biases = None, weights = None , gamma=None,gamma_sigmoid = 6):
        self. num_layers = len(sizes)
        self.sizes = sizes
        self.gamma_sigmoid = gamma_sigmoid
        self.gamma = gamma
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [np.random.randn(y, 1) for y in sizes [1:]]
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes [:-1], sizes [1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""

        for b, w, number in zip(self.biases, self.weights, range(len(self.weights))):
            if (number == 0 or number == 1):
                a = tanh(np.dot(w, a) + b,gamma=self.gamma[number]) # gamma is hyperparameter
            else:
                a = sigmoid(np.dot(w, a) + b,self.gamma_sigmoid) # here can be also some hyperparameter, for now we leave it without any hyperparameter
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini -batch stochastic gradient descent. The
        "training_data" is a list of tuples "(x, y)" representing the training
        inputs and the desired outputs. The other non-optional parameters are self -
        explanatory. If "test_data" is provided then the network will be evaluated
        against the test data after each epoch , and partial progress printed out.
        This is useful for tracking progress , but slows things down substantially.
        """

        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network’s weights and biases by applying gradient descent using
        backpropagation to a single mini batch. The "mini_batch" is a list of tuples
        "(x, y)", and "eta" is the learning rate.
        @param:mini_batch ----- randomly chosen mini-batch for stochastic gradient descent
        @param:eta ----- learning rate"""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights,nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""

        return (output_activations - y) # this may change with respect to the cost function

    def evaluate(self, test_data): # this really depends on settings of the task, for other neural architecture this could be modified
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network’s output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        test_results = [(np.argmax(self.feedforward(x)), y)for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def backprop(self, x, y): #
        """Return a tuple ‘‘(nabla_b , nabla_w)‘‘ representing the
        gradient for the cost function C_x. ‘‘nabla_b ‘‘ and
        ‘‘nabla_w ‘‘ are layer -by-layer lists of numpy arrays , similar
        to ‘‘self.biases ‘‘ and ‘‘self.weights ‘‘.
        THIS SHOULD BE REWRITTEN IN MATRIX FORM IN ORDER TO ACHIEVE FASTER TRAINING PERFORMANCE --- do this!"""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations , layer by layer
        zs = []  # list to store all the z vectors , layer by layer
        for b, w, number in zip(self.biases, self.weights, range(len(self.weights))):
            z = np.dot(w, activation) + b
            zs.append(z)
            if (number == 0 or number == 1):
                activation = tanh(z,self.gamma[number])
            else:
                activation = sigmoid(z,self.gamma_sigmoid)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1],self.gamma_sigmoid)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            if l == 2:
                sp = sigmoid_prime(z,self.gamma_sigmoid)
            else:
                sp = derivative_tanh(z,self.gamma[self.num_layers - l - 1])
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)







