# Author:   Michael Stepzinski
# Date:     October 9, 2021
# Purpose:  CS422 Machine Learning - Perceptron implementation file

import numpy as np

def perceptron_train(X,Y):

    # define problem size
    num_features = len(X[0])

    # define weights, bias, activation function
    w = [0] * num_features
    b = 0
    activation = 0

    # define convergence flag
    convergence = False

    # train for 100 iterations, only trains this much when data is not linearly separable
    for i in range(0, 100):
        # convergence is true until update is used
        convergence = True 
        # for each entry in X and Y, calculate activation
        for sample, y in zip(X, Y):
            for j in range(num_features):
                activation = activation + w[j]*sample[j]
            activation = (activation + b)*y
            # if activation function <= 0 adjust weights and continue training
            if activation <= 0:
                # update
                for j in range(num_features):
                    w[j] = w[j] + sample[j]*y
                b = b + y
                convergence = False
            activation = 0
        # if convergence is still true, then quit loop
        if convergence:
            break

    return w, b


def perceptron_test(X_test, Y_test, w, b):
    
    # define problem size
    num_samples = len(X_test)
    num_features = len(X_test[0])

    # declare activation and accuracy
    activation = 0
    accuracy = float(0)

    # if activation function is > 0 for a sample 
    # then that sample is classified correctly and accuracy + 1
    # otherwise, classified incorrectly
    for sample, y in zip(X_test, Y_test):
        for j in range(num_features):
            activation = activation + w[j]*sample[j]
        activation = (activation + b)*y
        if activation > 0:
            accuracy += 1
        activation = 0
        
    return accuracy/num_samples