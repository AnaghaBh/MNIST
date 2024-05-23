import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from keras.datasets import mnist

data = pd.read_csv('C:\Users\Anagha\Downloads\mnist_test.csv')

data = np.array(data)
m,n = data.shape #dimensions
np.random.shuffle(data)

data_dev = data[0:1000].T #first 1000 examples transpose
Y_dev = data_dev[0]# first row 
X_dev = data_dev[1:n]

data_train = data[1000:m].T #training data (rest of the data)
Y_train = data_train[0]
X_train = data_train[1:n]

def init_paramaters():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0,Z)

def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z)) #preseves the amount of columns and collapses the amount of rows to one 
    
def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(A1)
    
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.MAX()+1)) #matrix of zeroes, assumes classes  0-9 therefore + 1
    one_hot_Y[np.arange(Y.size),Y] = 1 #fro each row go to the column specified by label Y and set it to 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
def back_propagation(Z1, A1, A2, W2, Y):
    one_hot_Y = one_hot(Y)
    