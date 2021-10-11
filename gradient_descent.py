# Author:   Michael Stepzinski
# Date:     October 9, 2021
# Purpose:  CS422 Machine Learning - Gradient Descent implementation file


import numpy as np

def gradient_descent(grad_f, x_init, learning_rate):
    # Define number of features, initial gradient of f at x_init values, iters, and magnitude
    num_features = len(x_init)
    grad_f_value = x_init
    #iters = 0 # used for debugging
    magnitude = 1

    # while the magnitude of grad_f at value is greater than (randomly selected) 0.0001
    # do gradient descent
    while magnitude > 0.0001:
        # do gradient descent on 1 variable at a time, ex. grad_f[0] is x, grad_f[1] is y and so on
        for x in range(num_features):
            grad_f_value[x] = grad_f_value[x] - (learning_rate * grad_f(grad_f_value)[x])
        # find magnitude to get the loop to stop
        magnitude = np.linalg.norm(grad_f_value)
        #iters += 1
    return grad_f_value