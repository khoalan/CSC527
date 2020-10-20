# CSC/ECE/DA 427/527
# Fall 2020
# Lan Nguyen
# HW4 - d = 0

from random import random
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import math
import numpy as np

def moon(num_points, distance, radius, width):
    '''
    Function to create the double moon
    '''

    points = num_points

    x1 = [0 for _ in range(points)]
    y1 = [0 for _ in range(points)]
    x2 = [0 for _ in range(points)]
    y2 = [0 for _ in range(points)]

    for i in range(points):
        d = distance
        r = radius
        w = width
        a = random() * math.pi
        x1[i] = math.sqrt(random()) * math.cos(a) * (w / 2) + (
                    (-(r + w / 2) if (random() < 0.5) else (r + w / 2)) * math.cos(a))
        y1[i] = math.sqrt(random()) * math.sin(a) * (w) + (r * math.sin(a)) + d

        a = random() * math.pi + math.pi
        x2[i] = (r + w / 2) + math.sqrt(random()) * math.cos(a) * (w / 2) + (
            (-(r + w / 2)) if (random() < 0.5) else (r + w / 2)) * math.cos(a)
        y2[i] = -(math.sqrt(random()) * math.sin(a) * (-w) + (-r * math.sin(a))) - d
    return ([x1, x2, y1, y2])


def costFunc(X, y, w, lamda):
    '''
    Calculate the cost function using formula 2.36 on page 77 of the text book.
    '''
    m = X.shape[0]
    epsilon = (1. / (2. * m)) * \
        (np.sum((np.dot(X, w) - y) ** 2.) + lamda * np.dot(w.T, w))

    return epsilon

def updateWeight(X, y, w, epochs, learningRate, lamda):
    '''
    Update function for weights using gradient descent for regularized least squares.
    Gradient = The derivative of the cost function.
    '''

    m = X.shape[0]
    #Keep a history of Costs (for visualisation)
    epsilon = np.zeros((epochs, 1))

    for i in range(epochs):
        epsilon[i] = costFunc(X, y, w, lamda)

        #Using equation 1.42 on page 65
        w = w - (learningRate / m) * \
            (np.dot(X.T, (X.dot(w) - y[:, np.newaxis])) + lamda * w)

    return w, epsilon

def train(X, y, epochs, learningRate, lamda):
    """
    Trains all the vector in data.
    """
    Xn = np.ndarray.copy(X)
    yn = np.ndarray.copy(y)

    #Initial weight vector as w = [0., 0., 0.]
    w = np.zeros((Xn.shape[1] + 1, 1))

    #Normalise the X
    X_mean = np.mean(Xn, axis=0)
    X_std = np.std(Xn, axis=0)
    Xn -= X_mean
    X_std[X_std == 0] = 1
    Xn /= X_std

    y_mean = yn.mean(axis=0)
    yn -= y_mean

    # add ones for intercept term
    Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

    w, epsilon = updateWeight(
        Xn, yn, w, epochs, learningRate, lamda)

    return w, epsilon

def getDataSet(num_points, distance, radius, width):
    '''
    Consider X as a 2D array with 2 columns and num_point rows.
    Consider y as desired output for each X[0] and X[1].
    '''
    x1, x2, y1, y2 = moon(num_points, distance, radius, width)

    x1 = np.array(x1)
    x2 = np.array(x2)
    x = concatenate((x1, x2))
    output1 = np.ones(num_points)

    y1 = np.array(y1)
    y2 = np.array(y2)
    y = concatenate((y1, y2))
    output2 = np.zeros((num_points))

    XX = np.vstack([x, y])
    YY = np.concatenate((output1, output2))

    return XX.T, YY


def draw_line(w, xx):
    '''
    Draw decision boundary
    w0 + w1x + w2y = 0 => y = -(w0 + w1x)/w2
    '''
    x = np.linspace(np.amin(xx),np.amax(xx),100)
    y = -(w[0]+x*w[1])/w[2]
    plt.plot(x, y, '--k',label="DB")


#Create dataset for training
x_train, y_train = getDataSet(800, -2, 10, 6)

#Create dataset for testing
x_test, y_test = getDataSet(1000, -2, 10, 6)

print("Running...")
w, epsilon = train(x_train, y_train, 200, 0.1, 1)

#Draw decision boundary

figure(0)
draw_line(w, x_train)

x1, x2, y1, y2 = moon(500, 1, 10, 6)

#Plot the data points with decision boundary

for i in range(len(y_test)):
    if x_test[i][1] > (-(w[0]+x_test[i][0]*w[1])/w[2]):
        plt.scatter(x_test[i][0], x_test[i][1], marker='x', color='r')
    else:
        plt.scatter(x_test[i][0], x_test[i][1], marker='o', color='b')
plt.savefig("Least squares with dist = -2.png")
plt.show()



iter = np.linspace(0, 200, 200)
figure(1)
plt.plot(iter, epsilon)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()
print("Done!")



