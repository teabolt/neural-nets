# neural-nets

## Description
My attempt at learning about neural networks and associated topics.
Most of the code and understanding comes from Michael Nielsen's online book 'Neural networks and deep learning'.
The problem is to classify handwritten digits from the MNIST dataset.
The code is to be tinkered with, modified, and extended.

## Key functionality and programs
* 'net_utils.py/get_net()' generates, trains, and saves a neural net (just a shortcut to typing out many commands in the command line)
* 'data/own' contains own handwritten digits, outside of the MNIST data

## Desired future functionality / to-do
* pure non-numpy implementation of the core neural net code - understand the data structures behind biases, weights, gradients, etc. better
* more power over the mnist_loader.py - eg: load only parts of the data (test or training or validation only), parts of the 'sub-data' (eg: 10,000 instead of all the 50,000 images of the net)
* save the weights and biases, besides saving the entire network object
* be informed of and actually see the test images for which the network fails
* take the "globally" best set of weights and biases, out of the training epochs, instead of the final set of parameters
* print change in accuracy (difference) over epochs, see when learning slows down
* concurrent/parallel/multi-threaded implementation of this - eg: train many networks simultaneously
* 'timeit' information, between epochs, between start and finish of training
* a neural net / algorithm / program to determine optimal hyper-parameters (a genetic algorithm?) (automatic logarithmic-scale tests?)
* non-random / configurable initialisation of a network's weights and biases (zeroed at first?)
* calculate the variance, etc of the set of all the evaluations of a network (eg: how spread out are the network's accuracies over epochs)
* combine multiple neural networks (for non-overlapping inputs)
* go beyond MNIST to recognise handwritten digits - train with images of your own handwritten digits, with digits on different backgrounds, coloured instead of grayscale, etc
* finer control over the network - which neurons connect to which, recurrent instead of only feedforward networks
* choose which activation function to apply, to which layer/neuron furthermore
* optimise how data is loaded for use, eg: now the 'live' data is saved into a python zip object, which is then explictitly converted to a list (as of python3). There might be better ways to go about this?
* port the network to learn from the data in the original MNIST format instead of M. Nielsen's, to understand how the data was reformatted better
* more control over how the gradient is calculated, eg: not just with backpropagation, but also apply other differentiation ways?
