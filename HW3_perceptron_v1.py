# Copyright (C) 2024 Shipeng Liu
# 
# This file is part of EE559 HW3.
# 
# Purpose: Implementation of perceptron algorithms and methods required for EE559 Homework 3.
# Please fill in function perceptron_learning and perceptron_testing
# TA: Shipeng Liu

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

def perceptron_learning(x, labels, w_int=np.array([1.0,1.0,1.0]),
                        eta=1, max_iterations=1000):
    """
    Implements the batch gd perceptron learning algorithm.

    Args:
        x: (N, D) data array, non-augmented format.
        labels: Array of length N with labels 1, 2.
        w_int: Initial weights as a numpy array. Defaults to numpy.array([1.0,1.0,1.0]).
        eta: Learning rate as a float. Defaults to 1.
        max_iterations: Maximum number of iterations as an int. 
                        Defaults to 1000(here refer to epochs for batch gd).

    Returns:
        w_hats: Weights after each iteration.
                The array size should be (epochs, D+1) when augmented
        Js: Criterion function values after each iteration, size: (epochs, )
        error_rates: Misclassification rates after each iteration, size: (epochs, )
    """

    ########################
    # your code comes here
    ########################

    return w_hats, Js, error_rates


def plot_linear_learning(x, labels, w_hat, Js, error_rates):
    """
    Plot training curve, and visualize the decision boundry
    Args:
        x: (N, D) data array, non-augmented format
        labels:  lengnth array with labels 1, 2
        w_hat: final learned weights.
        Js: length N_epochs of Loss values (the learning curve)
        error_rates: length N_epochs of misclassification rates as the algorithm learned
    """
    N_epochs = len(Js)
    x_1 = x[labels==1]
    x_2 = x[labels==2]

    fig, ax = plt.subplots(1, 2, sharex=False, figsize=(12, 6))

    ## scatter plot with deciscion boundaries
    LIMIT = np.max(x)
    x_plot = np.arange(-1 * LIMIT, LIMIT, 0.01)
    ax[0].scatter(x_1.T[0], x_1.T[1], fc=(0, 0, 1, 0.5), label='class 1')
    ax[0].scatter(x_2.T[0], x_2.T[1], fc=(1, 0, 0, 0.5), label='class 2')
    #plot 2-class linear decision boundary
   
    ax[0].plot( x_plot, -1 * ( w_hat[1] *  x_plot  + w_hat[0] )
                                     / w_hat[2], linewidth=2, label='boundry')
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_xlim([-LIMIT, LIMIT])
    ax[0].set_ylim([-LIMIT, LIMIT])
    ax[0].legend()
    ax[0].grid(':')

    ## Learning curve
    epochs = np.arange(N_epochs)
    ax[1].plot(epochs, Js, marker='o', color='green', label='J (Loss)')
    # add second y-axis for the metric of error rate: https://pythonguides.com/matplotlib-two-y-axes/
    ax2 = ax[1].twinx() 
    ax2.plot(epochs, error_rates * 100, marker='x', color='purple', label='Error Rate')
    ax[1].set_ylabel('J (Loss)', color = 'green') 
    ax[1].tick_params(axis ='y', labelcolor = 'green') 
    ax2.set_ylabel('Error rate (%)', color = 'purple') 
    ax2.tick_params(axis ='y', labelcolor = 'purple') 

    # ax[1].set_xlabel('epoch')
    # ax[1].set_ylabel('J (Loss)')
    ax2.set_ylabel('error rate (%)')
    ax[0].legend()
    ax[1].grid(':')

def perceptron_testing(x, labels, w_hat):
    '''
    Implements the perceptron testing.

    Args:
        x: (N, D) data array, non-augmented format.
        labels: Array of length N with labels 1, 2.
        w_hat: final learned weights as a numpy array.
    Returns:
        error_rates: Misclassification rates.
        
    '''
    ########################
    # your code comes here
    ########################
    return error_rate

def plot_linear_testing(x, labels, w_hat):
    """
    Visualize the decision boundary of testing sets
    Args:
    x: (N, D) data array, non-augmented format
    labels:  lengnth array with labels 1, 2
    w_hat: final learned weights as a numpy array.
    """
  
    x_1 = x[labels==1]
    x_2 = x[labels==2]

    fig, ax = plt.subplots(1, 2, sharex=False, figsize=(12, 6))

    ## scatter plot with deciscion boundaries
    
    LIMIT = np.max(x)

    
    x_plot = np.arange(-1 * LIMIT, LIMIT, 0.01)
    ax[0].scatter(x_1.T[0], x_1.T[1], fc=(0, 0, 1, 0.5), label='class 1')
    ax[0].scatter(x_2.T[0], x_2.T[1], fc=(1, 0, 0, 0.5), label='class 2')
    #plot 2-class linear decision boundary
    ax[0].plot( x_plot, -1 * ( w_hat[1] *  x_plot  + w_hat[0] ) 
                                    / w_hat[2], linewidth=2, label='boundry')
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_xlim([-LIMIT, LIMIT])
    ax[0].set_ylim([-LIMIT, LIMIT])
    ax[0].legend()
    ax[0].grid(':')


data = np.genfromtxt("ee559_dataset/dataset1_train.csv",
                    delimiter=",", dtype=float)
x = data[1:,0:2]
labels = data[1:,2]
print("--------------start training----------------------")
w_hats, Js, error_rates = perceptron_learning(x, labels)
plot_linear_learning(x, labels, w_hats[-1], Js, error_rates)

test_data = np.genfromtxt("ee559_dataset/dataset1_test.csv",
                    delimiter=",", dtype=float)
x = test_data[1:,0:2]
labels = test_data[1:,2]
print("--------------start testing----------------------")
# print(x, labels, w_hats[-1])
error_rate = perceptron_testing(x, labels, w_hats[-1])
plot_linear_testing(x, labels, w_hats[-1])
plt.show()