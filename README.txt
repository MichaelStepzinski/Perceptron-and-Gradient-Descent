Author:  Michael Stepzinski
Date:    October 11, 2021
Purpose: CS422 - Machine Learning Project 2 Gradient Descent Writeup

    perceptron_train
My implementation of the perceptron training algorithm 
first defines problem size by taking the length of the first sample, 
then defines weights, bias, the activation function, and a convergence flag. 
My algorithm loops for 100 iterations and checks if the activation function 
has been updated in this loop. If not, then it breaks. 
The function stops and returns w and b after 100 iterations regardless.
The algorithm takes a sample of X with its result Y together.
Then calculates the activation function by looping over each column of sample.
If activation <= 0 then w1 through num_features are updated and b is updated.
w and b are then returned.

    perceptron_test
My implementation of the perceptron testing algorithm
first defines problem size by taking the length of the sample set
and the length of the first sample. Using the same process as above,
the activation function is calculated for each sample.
If the activation is less than or equal to 0 then that point is missed
and not included in the final accuracy calculation.
Otherwise, the num of correct samples is divided by num total samples
to return the total accuracy

    gradient_descent
My implementation of the gradient_descent algorithm
first defines problem size by taking the len of the x_init array
