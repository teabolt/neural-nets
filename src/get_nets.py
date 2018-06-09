#!/usr/bin/env python3

# Own
import mnist_loader
import nn

# Standard library
import pickle
import datetime

def main():
    dir_path = './'
    data_path = dir_path + '../data/mnielsen/mnist.pkl.gz'
    
    print('set up...')
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(data_path)
    net = nn.Network([784, 15, 10])

    print('training...')
    net.SGD(training_data, 1, 30, 3, test_data)
    print('training finished')

    print('saving...')
    # file includes time; to-do: meta-data (number of hidden layer neurons, other hyperparameters, etc)
    curr_time = datetime.datetime.now()
    net_path = dir_path + '../trained_nets/net-{}.pkl'.format(curr_time.strftime('%Y-%m-%d-%H-%M-%S-%f'))
    with open(net_path, 'wb') as f_net:
        pickle.dump(net, f_net)

    print('finished')

if __name__ == '__main__':
    main()