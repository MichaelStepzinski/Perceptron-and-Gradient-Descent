# Author:   Michael Stepzinski
# Date:     October 9, 2021
# Purpose:  CS422 Machine Learning - Gradient Descent implementation file

import numpy as np

def gradient_descent(grad_f, x_init, learning_rate):
    
    features = len(x_init)
    grad_f_value = x_init
    iters = 0
    magnitude = 1

    while magnitude > 0.0001:
        for x in range(features):
            grad_f_value[x] = grad_f_value[x] - (learning_rate * grad_f(grad_f_value)[x])
        magnitude = np.linalg.norm(grad_f_value)
        iters += 1
    return grad_f_value