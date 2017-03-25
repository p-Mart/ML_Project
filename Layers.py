import numpy as np
from Nodes import *

class Sigmoid:

	def __init__(self, input_size, nodes):
		self.input_size = input_size
		self.nodes = nodes

		self.weights = np.random.rand(nodes, input_size+1)

	def output(self, x):
		'''Returns the outputs of the nodes in this layer, of
		shape {nodes x 1}'''

		#Assuming that the input vector x comes in
		#with a shape of {1 x input_size}

		return sigmoid(np.dot(self.weights, x.T))

	def derivative(self, x):
		return (self.output(x)*(1 - self.output(x)))

class Relu:

	def __init__(self, input_size, nodes):
		self.input_size = input_size
		self.nodes = nodes

		self.weights = np.random.rand(nodes, input_size+1)

	def output(self, x):
		'''Returns the outputs of the nodes in this layer, of
		shape {nodes x 1}'''

		#Assuming that the input vector x comes in
		#with a shape of {1 x input_size}

		return relu(np.dot(self.weights, x.T))

	def derivative(self, x):
		outputs = self.output(x)
		for i in range(len(outputs)):
			if(outputs[i] <= 0):
				outputs[i] = 0.
			else:
				outputs[i] = 1.

		return outputs

class Softmax:
	'''Note: softmax must be used for categorical crossentropy'''
	def __init__(self, input_size, nodes):
		self.input_size = input_size
		self.nodes = nodes

		self.weights = np.random.rand(nodes, input_size+1)

	def output(self, x):
		'''Returns the outputs of the nodes in this layer, of
		shape {nodes x 1}'''

		#Assuming that the input vector x comes in
		#with a shape of {1 x input_size}
	
		return softmax(np.dot(self.weights, x.T))

	def derivative(self, x):
		#return (self.output(x)*(1 - self.output(x)))#This MIGHT be wrong
		return 1.

class MaxPool:

	def __init__(self, receptive_field, stride):
		self.f = receptive_field
		self.s = stride

	def maxPool(self, x):
		'''Assuming x is some tensor of shape [h x w x d]'''
		f = self.f #Size of partitions being looked at
		s = self.s #Stride over the input
		
		#Compute the output dimensions of the max pool layer
		h, w, d = x.shape
		output_h = (h - f)/s + 1
		output_w = (w - f)/s + 1
		output_d = d

		pool = np.empty((output_h,output_w,output_d))
		#This could probably be significantly optimized	
		#Max pooling operation
		for k in range(d):
			for i in range(0,h-f+1,s):
				for j in range(0,w-f+1,s):
					pool[i/s,j/s,k] = np.max(x[i:(i+f),j:(j+f),k])

		return pool



