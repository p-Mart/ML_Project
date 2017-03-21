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
'''
class Softmax:

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
		derivatives = self.output(x)
		for i in range(len(outputs)):
			if(outputs[i] <= 0):
				outputs[i] = 0.
			else:
				outputs[i] = 1.

		return outputs
'''