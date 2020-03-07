import numpy as np
import random
from CostFunctions import *
import math


# proti Boosted klasickému je zde přidán Nesterov momentum v SGD

class Network():

    def __init__(self , sizes, biases = None, weights = None , gamma=None,gamma_output = 6,
                 weight_initilizer = 'new', cost = None, output_func = 'sigmoid', momentum = 0.9):
        self. num_layers = len(sizes)
        self.sizes = sizes
        self.gamma_output = gamma_output
        self.gamma = gamma
        self.cost = cost
        self.output_func = output_func
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [np.random.randn(y, 1) for y in sizes [1:]]
        if weights is not None:
            self.weights = weights
        else:
            if weight_initilizer == 'new':
                self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes [:-1], sizes [1:])] # here we divide with sqrt of number of connections to the respective neuron
            else:
                self.weights = [np.random.randn(y, x) for x, y in zip(sizes [:-1], sizes [1:])] # here we divide with sqrt of number of connections to the respective neuron

        self.velocity_weights = [] # nastavíme na 0
        self.velocity_biases = [] # nastavíme na 0
        for w in self.weights:
            self.velocity_weights.append(np.zeros(w.shape))
        for b in self.biases:
            self.velocity_biases.append(np.zeros(b.shape))
        self.momentum = momentum


    def feedforward(self, a):
        """Return the output of the network if "a" is input."""

        for b, w, number in zip(self.biases, self.weights, range(len(self.weights))):
            if (number == 0 or number == 1):
                a = tanh(np.dot(w, a) + b,gamma=self.gamma[number]) # gamma is hyperparameter
            else:
                if self.output_func == 'sigmoid':
                    a = sigmoid(np.dot(w, a) + b,self.gamma_output)
                elif self.output_func == 'softmax':
                    a = softmax(np.dot(w, a) + b,self.gamma_output)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, num_classes, lmbda = 0.0,
            evaluation_data=None, monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False, monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch
        stochastic gradient descent. The ‘‘training_data‘‘ is a list of tuples ‘‘(x, y)‘‘ representing the training
         inputs and the desired outputs. The other non-optional parameters are self-explanatory,
         as is the regularization parameter ‘‘lmbda‘‘. The method also accepts ‘‘evaluation_data‘‘,
          usually either the validation or test data. We can monitor the cost and accuracy on either the evaluation
           data or the training data, by setting the appropriate flags. The method returns a tuple containing four lists:
            the (per-epoch) costs on the evaluation data, the accuracies on the evaluation data, the costs on the training data,
             and the accuracies on the training data. All values are evaluated at the end of each training epoch.
             So, for example, if we train for 30 epochs, then the first element of the tuple will be a 30-element
             list containing the cost on the evaluation data at the end of each epoch. Note that the lists are empty
             if the corresponding flag is not set.
        """

        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda, num_classes)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, num_classes,convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy


    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network’s weights and biases by applying gradient descent using backpropagation
        to a single mini batch. The ‘‘mini_batch‘‘ is a list of tuples ‘‘(x, y)‘‘, ‘‘eta‘‘ is the learning rate,
        ‘‘lmbda‘‘ is the regularization parameter, and ‘‘n‘‘ is the total size of the training data set.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, eta)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        nabla_b = [(1 / (len(mini_batch))) * nb for nb in nabla_b]
        nabla_w = [(1 / (len(mini_batch))) * nw for nw in nabla_w]

        self.velocity_weights = [self.momentum*v + nw for v,nw in zip(self.velocity_weights,nabla_w)]
        self.velocity_biases = [self.momentum*b + nb for b,nb in zip(self.velocity_biases,nabla_b)]

        self.weights = [(1-eta*(lmbda/n))*w - eta * v for w, v in zip(self.weights,self.velocity_weights)] # added regularization term to the cost function
        self.biases = [b - eta*vb for b, vb in zip(self.biases, self.velocity_biases)]

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

    def backprop(self, x, y, eta): # zde použijeme nesterova a Adam
        """Return a tuple ‘‘(nabla_b , nabla_w)‘‘ representing the
        gradient for the cost function C_x. ‘‘nabla_b ‘‘ and
        ‘‘nabla_w ‘‘ are layer -by-layer lists of numpy arrays , similar
        to ‘‘self.biases ‘‘ and ‘‘self.weights ‘‘.
        THIS SHOULD BE REWRITTEN IN MATRIX FORM IN ORDER TO ACHIEVE FASTER TRAINING PERFORMANCE --- do this!"""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        weights = [w - eta*self.momentum*v for w,v in zip(self.weights,self.velocity_weights)]
        biases = [b - eta*self.momentum*v for b,v in zip(self.biases,self.velocity_biases)]


        # feedforward
        activation = x
        activations = [x]  # list to store all the activations , layer by layer
        zs = []  # list to store all the z vectors , layer by layer
        for b, w, number in zip(biases, weights, range(len(weights))):
            z = np.dot(w, activation) + b
            #print(z)
            zs.append(z)
            if (number == 0 or number == 1):
                activation = tanh(z,self.gamma[number])
            else:
                if self.output_func == 'sigmoid':
                    activation = sigmoid(z,self.gamma_output)
                elif self.output_func == 'softmax':
                    activation = softmax(z,self.gamma_output)
            activations.append(activation)
        # backward pass
        delta = self.cost.delta(zs[-1],activations[-1],y,self.gamma_output)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = derivative_tanh(z,self.gamma[self.num_layers - l - 1])
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self,data,lmbda,num_classes,convert=False):
        """Return the total cost for the data set ‘‘data‘‘.
        The flag ‘‘convert‘‘ should be set to False if the data set is the training data (the usual case),
        and to True if the data set is the validation or test data. See comments on the similar (but reversed)
        convention for the ‘‘accuracy‘‘ method, above. """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y,num_classes)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost
