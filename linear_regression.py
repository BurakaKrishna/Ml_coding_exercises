# Linear regression from scratch
# Equation : w = (XtX)^-1(XtY)
# where X = (r,c) matrix and Y = (r,1) matrix
# and w are the weights to be found
# What
# How
# When
# Limitations

import numpy as np

# inputs : X,y
# X is a matrix
# y is a matrix
# dot product is used ? question yourself why
# inverse is used to find the x transpose and x inverse

from numpy.linalg import inv

def train_model(X,y):
	weights = np.dot(inv(np.dot(X.T,X)),np.dot(X.T,y))
	return weights


# example
import pandas as pd
data_url = 'https://raw.githubusercontent.com/cmdlinetips/data/master/cars.tsv'
cars = pd.read_csv(data_url, sep="\t")
# X is speed and y is dist taken to stop
# We are finding function to approximate distance a car moves when applied brakes from a certain speed
X = cars.dist.values
y = cars.speed.values
# print(X)

X_mat=np.vstack((np.ones(len(X)), X)).T
# print(X_mat)
# len(X),2
y = np.array(y)
coefficients = train_model(X_mat,y)
print(coefficients)
# [8.28390564 0.16556757] supposed to get
