import neural_net as nn
import data_utils
import os
import numpy as np
import time
import sys

def build_and_train_nn(Xtr, Ytr, Xval, Yval, Xte = None, Yte = None, input_size = 3072, hidden_layer_size = 500, output_size = 10, learning_rate = 1, num_iters = 5000, learning_rate_decay = .9, reg =1e-3, verbose = True):
    start_time = time.time()
    our_net = nn.TwoLayerNet(input_size, hidden_layer_size, output_size)
    results = our_net.train(Xtr, Ytr, Xval, Yval, learning_rate = learning_rate, reg = reg, verbose = verbose, num_iters = num_iters, learning_rate_decay = learning_rate_decay)
    end_time = time.time()
    results['time'] = end_time - start_time
    print('Training Accuracy: '+str(results['train_acc_history'][-1]))
    print('Validation Accuracy: '+str(results['val_acc_history'][-1]))
    if Xte is not None and Yte is not None:
        print('Testing Accuracy: '+str(our_net.accuracy(Xte,Yte)))
    return results

def load_and_process(location = None):
    Xtr, Ytr, Xte, Yte = None, None, None, None,
    if location is None:
    	Xtr, Ytr, Xte, Yte = data_utils.load_CIFAR10(os.path.join(os.getcwd(),'cifar-10-batches-py'))
    else:
	Xtr, Ytr, Xte, Yte = data_utils.load_CIFAR10(os.path.join(os.getcwd(),location,'cifar-10-batches-py'))
    Xtr, Xte = np.reshape(Xtr,(Xtr.shape[0], 3072)), np.reshape(Xte, (Xte.shape[0], 3072))
    #preprocessing
    feature_maxes = np.abs(Xtr).max(axis = 0)
    Xtr = Xtr/feature_maxes
    Xte = Xte/feature_maxes
    mean_image = np.mean(Xtr, axis = 0)
    Xtr -= mean_image
    Xte -= mean_image
    #end preprocessing
    Xtr, Ytr = nn.shuffle_training_sets(Xtr,Ytr)
    training_set_size = Xtr.shape[0]
    Xtrain, Xval = Xtr[:int(training_set_size*.9)],Xtr[int(training_set_size*.9):]
    Ytrain, Yval = Ytr[:int(training_set_size*.9)], Ytr[int(training_set_size*.9):]
    return Xtrain, Ytrain, Xval, Yval, Xte, Yte

if __name__ == '__main__':
    Xtr, Ytr, Xval, Yval, Xte, Yte = None, None, None, None, None, None
    if len(sys.argv) > 1:
    	Xtr, Ytr, Xval, Yval, Xte, Yte = load_and_process(sys.argv[1])
    else:
	Xtr, Ytr, Xval, Yval, Xte, Yte = load_and_process()
    results = build_and_train_nn(Xtr,Ytr, Xval, Yval, Xte = Xte, Yte = Yte)


