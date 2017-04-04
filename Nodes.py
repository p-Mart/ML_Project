import numpy as np

def sigmoid(x):
	return (1.0 / (1.0 + np.exp(-x)))	

def relu(x):
	return np.maximum(0., x)

def smoothRelu(x):
	return np.log(1 + np.exp(x))

def softmax(x):
	#Note the x - np.max(x).
	#This is for numerical stability in case np.exp encounters
	#large values of x.
	f = np.exp(x - np.max(x))
	#print f / np.sum(f)
	return f / np.sum(f) #Return normalized probabilities
