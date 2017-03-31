from Network import *
from Layers import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def binary(x):
	return 1. if x >= 0.5 else 0.

#Generate 100 random samples on the unit square
n_samples = 800
X = np.random.uniform(0,1,size=(n_samples,2))
Y = np.zeros((n_samples, 2))
#Given parameters for the circle on the unit square
a = 0.5
b = 0.6
r = 0.4
y = (((X[:,0] - a)**2 + (X[:,1] - b)**2) < r**2)
for i in range(len(y)):
	if (y[i] == 0):
		Y[i,0] = 1.
	else:
		Y[i,1] = 1.

#Test set of 100 samples
X_test = np.random.uniform(0,1,size=(n_samples,2))
y_test = (((X_test[:,0] - a)**2 + (X_test[:,1] - b)**2) < r**2)
Y_test = np.zeros((n_samples, 2))
for i in range(len(y_test)):
	if (y_test[i] == 0):
		Y_test[i,0] = 1.
	else:
		Y_test[i,1] = 1.

#Hyperparameters
hidden_nodes = 10
output_nodes = 1
learning_rate = 0.03
number_epochs = 40000


colors = ['red','blue']
plt.figure(figsize=(8,8))
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title("Training Set")

#Model architecture
layer_1 = Relu((X.shape[1], 1) , hidden_nodes)
layer_2 = Softmax((hidden_nodes, 1), 2)

model = Network([layer_1, layer_2],
					learning_rate = learning_rate,
					func = "categorical crossentropy"
				)

model.train(X, Y, number_epochs)
outputs = model.predict(X_test, Y_test)
#print outputs
maxes = outputs.argmax(axis = 1)

#print prediction

plt.figure(figsize=(8,8))
plt.scatter(X_test[:,0],X_test[:,1],c=maxes,cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title("Training Set")
plt.show()

