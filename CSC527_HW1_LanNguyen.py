#The Catholic University of America
#CSC527
#Instructor: Dr. Hieu Bui
#Student: Lan Nguyen
#HW1

import numpy as np


def func(y):
    if y >= 0:
        return 1
    else:
        return 0


def perceptron(x, w, bk):
    keepConstraint = input("Do you want to keep the constraint (bk > -1 = firing state) || (bk <-3 = quiesent state) (y/n)?: ")
    if keepConstraint == 'y':
        if bk > -1:
            return 1
        if bk <= -3:
            return 0
        sum = np.dot(w, x) + bk
        y = func(sum)
        return y
    else:
        sum = np.dot(w, x) + bk
        y = func(sum)
        return y


def logicGate(inputValue, numOfInput):
    weights = np.ones((numOfInput,), dtype=int)
    bk = float(input('Enter bk: '))
    return perceptron(inputValue, weights, bk)


while True:
    while True:
        try:
            numOfInput = int(input("Enter number of input: "))
            while numOfInput <= 1:
                print("Your number of input must be an int > 1!")
                numOfInput = int(input("Enter number of input: "))
                if numOfInput > 1:
                    break
            break
        except ValueError:
            print("Please input integer only!")
            continue

    iArray = []
    for i in range(numOfInput):
        while True:
            i1 = input("Enter value of input " + str(i + 1) + " (0 or 1): ")
            if i1 == "0" or i1 == "1":
                break
        iArray.append(int(i1))

    input1 = np.array(iArray)
    print("Input = ",input1, " Predicted result = ", logicGate(input1, numOfInput))

    again = input("Try again ? (y/n): ")
    if again == "n":
        break

print("=====================================================================")
print("For the answer of question 2d please open the README.md on the github")
print("=====================================================================")

