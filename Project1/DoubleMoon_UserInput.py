# CSC/ECE/DA 427/527
# Fall 2020
# Lan Nguyen
# Modified mode

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



def sgnFunc(x, w):
    '''
    Signum function, return 1 if sum(w*x)+b >= 0 else return -1
    '''

    activation = w[0]
    for i in range(2):
        activation += sum([i * j for i, j in zip(w[1:],x[0:2])])
    if activation >= 0:
        return 1
    else:
        return -1


def updateWeight(x, learningRate, error, w):
    '''
    Update function for weights
    w(n+1) = w(n) + learningRate * (d(n) - y(n)) * x(n)
    iterError is (d(n) - y(n))
    '''

    return [i + learningRate * error * j for i, j in zip(w[1:],x[0:2])]

def train(data, learningRate, w):
    """
    Trains all the vector in data.
    """

    epochs = 0
    MSE = []
    while True:
        sumError = 0.0
        for x in data:

            predicted = sgnFunc(x, w)
            expected = x[2]
            if expected != predicted:
                error = expected - predicted
                w[1:] = updateWeight(x, learningRate, error, w)
                w[0] = w[0] + learningRate*error
                sumError += error**2
        epochs += 1
        MSE.append(sumError/50)
        if epochs >= 50 or sumError == 0.0:
            break
    return w, MSE

def getDataSet(num_points, distance, radius, width):
    '''
    Add third element as desired output in each data point
    Return the dataset.
    '''

    x1, x2, y1, y2 = moon(num_points, distance, radius, width)
    data = []
    data.extend([x1[i], y1[i], 1] for i in range(num_points))
    data.extend([x2[i], y2[i], -1] for i in range(num_points))
    return data

def draw_line(w, data):
    '''
    Draw decision boundary
    w0 + w1x + w2y = 0 => y = -(w0 + w1x)/w2
    '''

    x = np.linspace(np.amin(data),np.amax(data),100)

   # x = np.linspace(-10,10, 100)
    y = -(w[0] + x*w[1])/w[2]
    plt.plot(x, y, '--k',label="DB")
numOfPoints_Train = input("Enter number of points for training dataset: ")
numOfPoints_Test  = input("Enter number of points for testing dataset: ")

distance = input("Enter the distance between 2 moons: ")
learningRate = input("Enter the learning rate: ")
bias = int(input("Enter the bias: "))

w = [bias] + [0 for _ in range(2)]


dataTrain2 = getDataSet(int(numOfPoints_Train), int(distance), 10, 6)

#Create dataset for testing
dataTest2 = getDataSet(int(numOfPoints_Test), int(distance), 10, 6)

#Train the model
result2, MSE2 = train(dataTrain2, float(learningRate), w)

#Draw all the points in testing dataset
for x in dataTest2:
    plt.figure(5)
    predict = sgnFunc(x, result2)
    if predict == 1:
        plt.plot(x[0], x[1], marker='x', color='r',label="bl")
    else:
        plt.plot(x[0], x[1], marker='o', color='b',label='rd')


#Draw decision boundary
plt.figure(5)
draw_line(result2, dataTest2)
plt.axis([-20, 30, -20, 20])
plt.title("Classification using perceptron with distance = " +distance+ ", radius = 10, and width = 6")
plt.show()

iters = [i for i in range(50)]
if len(MSE2) < 50:
  for i in range(50-len(MSE2)):
    MSE2.append(0)

plt.figure(6)
plt.plot(iters, MSE2, color="red")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Learning curve")
plt.savefig("LearningCurve_Modified.png")
plt.show()