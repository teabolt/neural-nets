#!/usr/bin/env python3

"""Utility neural network functions"""

# Standard library
import datetime
import random
import pickle

# Own
import mnist_loader
import network
import genetic

# Note that the code needs to be executed in the 'src' directory, to ensure relative paths are correct

def optimise_hparams():
    """Use a genetic algorithm to find the optimal hyperparameters for a neural network"""
    pass

def get_net(data_path=None, net_path=None, hidden_layers=None, epochs=30, mini_batch_size=10, eta=3, more_results=True, best_in_session=True, stats=True):
    """Create, train, and save a neural network from arguments. Contains indicative output for the user.
    data_path is the directory path of the image data,
    net_path is the path to which to save the neural net (the path is assumed to exist) and the name of the file,
    hidden_layer is a list of integers, representing the number of neurons per layer, going from input to output layers,
    epochs, mini_batch_size, and eta are the training hyper-parameters."""
    if hidden_layers == None:
        hidden_layers = [30]

    if data_path == None:
        dir_path = './'
        data_path = dir_path + './../data/mnielsen-mnist/mnist.pkl.gz'
    
    print('set up...')
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(data_path)
    net = network.Network([784]+hidden_layers+[10])

    print('training...')
    net.SGD(training_data, epochs, mini_batch_size, eta, test_data, more_results, best_in_session, stats)
    print('training finished')

    print('saving...')
    if net_path == None:
        # default filename contains dash-separated data on: current date and time, some hyperparameters used, network's accuracy (integer percent), a random integer between 0 and 10000 (decrease likelihood of colliding filenames)
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper(data_path)
        time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        hyperparams = '{0}-{1}-{2}-{3}-{4}'.format(epochs, len(hidden_layers), 
            max(hidden_layers), mini_batch_size, int(eta))
        test_data = list(test_data)
        n_test = len(test_data)
        n_correct = net.evaluate(test_data)
        accuracy = int(n_correct/n_test*100)
        rand_num = random.randint(0, 1000)

        # filenames are systematic so that later on the names could be used to extract information, 
        # eg: best accuracy network out of many
        net_path = dir_path + '../trained_nets/net-{0}-{1}-{2}-{3}.pkl'.format(time, hyperparams, accuracy, rand_num)

    # truncate file if already exists (in the unlikely case, save both nets in one file)
    with open(net_path, 'w+b') as f_net: 
        pickle.dump(net, f_net)

    print('finished')

def main():
    get_net(epochs=15, more_results=True, best_in_session=True, stats=True)

if __name__ == '__main__':
    main()