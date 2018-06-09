#!/usr/bin/env python3

"""A neural network to classify handwritten digits"""

# Standard
import random

# Third-party
import numpy as np

class Network(object):
    """Represent a neural net"""

    def __init__(self, sizes):
        """Initialize a neural network with neurons as described by the 'size' list.
        Each list entry is an integer representing the number of neurons in the specific layer."""
        self.sizes = sizes
        self.num_layers = len(sizes) # the number of layers in the network

        # for each layer from 2nd to last, create a vector(matrix) 
        # of random floats representing the biases in the layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # connect a layer to the next layer via weights
        # take the number of neurons in the current layer and in the next layer
        # then for each neuron in the next layer, generate weights going from all the neurons of the current layer
        self.weights = [np.random.randn(y, x) for (x, y) in zip(sizes[:-1], sizes[1:])]

    # @staticmethod
    # def sigmoid(z):
    #     """Apply the sigmoid activation function to a (vector) input z"""
    #     return 1/(1+np.exp(-z))

    # def activate(f, z):
    #     """Apply the activation function f on the input z"""
    #     return f(z)

    def feedforward(self, a):
        """Given the input vector a (from input layer) ((n, 1) numpy array), calculate the output of a neural network (from the output layer), given an arbitrary number of in-between hidden layers"""
        for (b, w) in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Let a neural network learn with Stochastic Gradient Descent. Use Gradient Descent (minimise cost function), but approximate the cost gradient with mini-batches (with mini_batch_size integer number of training data) of the training data (list of (x, y) tuples, where x is the input and y is its desired output). Once learned from all the batches, start over, for a number of epochs (such repetitions). eta is the learning rate with which to update the network. test_data is optional progress display."""
        
        training_data = list(training_data) # python3 implementation
        n = len(training_data) # number of training_data points

        if test_data: # some pre-processing to be more efficiency when testing afterwards
            test_data = list(test_data) # python3 detail 
            n_test = len(test_data) # number of test_data points

        # execute epochs
        for j in range(epochs):
            # randomly get mini-batches
            random.shuffle(training_data) # random points
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] # get slices
            # learn from each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print('Epoch {}: {} out of {} correct'.format(j, self.evaluate(test_data), n_test))
            else:
                print('Epoch {} completed'.format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the weights and biases of a network using a mini-batch training data (list of (x, y) tuples) supplied and gradient descent (with learning rate eta) and backpropagation to compute the gradients."""
        
        # get shape for updates (gradients) (zero at first)
        nabla_b = [np.zeros(b.shape) for b in self.biases] # get list and shape of each bias layer
        nabla_w = [np.zeros(w.shape) for w in self.weights] # get list and shape of each weight layer pair

        # know the difference between:
        # - a python list (use a list comprehension to manipulate elements)
        # - a numpy array (vector operations easily applicable)
        # - a numpy array *shape*

        # get a gradient for an input, and add to the total gradient; for each input point
        for (x, y) in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # 'delta' = difference/change (to be added)
            # iterate over the original and change in gradient lists, adding their component gradients
            nabla_b = [nb+dnb for (nb, dnb) in zip(nabla_b, delta_nabla_b)] 
            nabla_w = [nw+dnw for (nw, dnw) in zip(nabla_w, delta_nabla_w)]

        # set the new biases and weights
        m = len(mini_batch)
        self.biases = [b - (eta/m)*nb for (b, nb) in zip(self.biases, nabla_b)]
        self.weights = [w - (eta/m)*nw for (w, nw) in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        """Backpropagation algorithm
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        # gradients, filled with zeros and of shape of the neural net's parameters
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activations = [] # store all activations (activation function applied)
        zs = [] # store all z vectors (dot + bias)

        # initial
        activation = x
        activations.append(x)

        # go over each layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b # calculate z-vector
            zs.append(z) # save
            activation = sigmoid(z) # set the new activation for the next layer
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # (different notation)
        # do for each layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        # finished with the gradients for (x, y)
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def evaluate(self, test_data):
        """Evaluate a network's accuracy against the test_data supplied. Returns the number of accurate classifications (number of inputs for which the output is correct) (printing is up to you)
        An 'output' is assumed to be the neuron in the final layer with the highest activation. The label y in the test_data is assumed to be the index of the neuron with the highest activation."""

        # normalise - activation vectors to actual 'outputs', comparable to labels
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        # for (x, y) in test_results:
        #     if x == y:
        #         print(x, y)
        # calculate number of matches
        return sum(int(x == y) for (x, y) in test_results)

# Generic maths functions

def sigmoid(z):
    """Sigmoid activation function to (vector) input z (dot product + bias)"""
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    # direct or alternate form: sigmoid(z)*(1-sigmoid(z))
    return sigmoid(z)*(1-sigmoid(z))

# Tests

def main():
    net = Network([3, 5, 2])
    # print(net.sigmoid(-10))
    # print(net.weights)
    # print(net.biases)
    # print(net.feedforward(np.array([0.1, 0.75, 0.5]).reshape(3, 1)))
    net.SGD([([0, 1, 0.5],[0, 0.2]), ([0, 0.7, 0.5],[0.3, 0.2]), ([1, 1, 0.5],[1, 0.2]), ([0.5, 1, 0.5],[0.4, 0.2])], 1, 2, 0.2, [([0, 0, 0],[0, 0]), ([1, 0.5, 1],[0.2, 1]), ([0.2, 0.3, 0.5],[0.4, 0.5])])

if __name__ == '__main__':
    main()