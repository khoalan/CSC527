import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import random

def generate_data():
    var_ = 0.02
    x0 = 0
    maxSize = 10000
    coef = 0.99
    epsilon = np.random.normal(0, math.sqrt(var_), size = maxSize)
    x = np.zeros(maxSize)
    x[0] =  coef * x0 + epsilon[0]

    for i in range(1, maxSize):
      x[i] = coef * x[i-1] + epsilon[i]

    lowBound = random.randint(4999)
    x = x[lowBound:lowBound+5000]

    return x.reshape(len(x),1)


def linear_train(epochs, learningRate):
    x0 = 0
    mse_arr = []

    for iter in range(epochs):

        x = generate_data()
        w = 0
        err_arr = []

        n = 0
        x_pred1 = x0 * w
        error1 = x[n] - x_pred1
        w = w + learningRate * error1 * x[0]
        err_arr.append(error1)

        for n in range(1, len(x)):
            x_pred = x[n - 1] * w
            error = x[n] - x_pred
            w = w + learningRate * error * x[n - 1]
            err_arr.append(error)

        mse_arr.append(err_arr)
    mse_arr = np.array(mse_arr).reshape(epochs, 5000)

    return w, mse_arr


def show_results(mse, j_theory, eta):
    plt.title('Learning-rate parameter eta =' + str(eta))
    plt.semilogy(np.mean(np.square(mse), axis=0), 'k-', label='Experiment', linewidth=0.6)
    plt.semilogy(j_theory, 'b--', label='Theory', linewidth=0.6)
    plt.xlabel("Number of iterations")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig('Task1_Eta=' + str(eta) + '.png')
    plt.show()

def runT1(eta = 0.001):
    coef = 0.99
    sig2 = 0.995
    t = np.array(range(1, 5001))
    w, mse = linear_train(100, eta)

    j_theory = sig2*(1-coef**2)*(1+(eta/2)*sig2) + sig2*(coef**2+(eta/2)*(coef**2)*sig2-0.5*eta*sig2)*(1-eta*sig2)**(2*t)
    show_results(mse, j_theory, eta)


runT1(0.02)