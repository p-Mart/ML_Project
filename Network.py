import numpy as np
from Layers import *

class Network:

	def __init__(self, layers, learning_rate = 1):
		self.layers = layers
		self.depth = len(layers)

		self.learning_rate = learning_rate

	def getOutputs(self, sample):
		outputs = []
		x = sample
		for i in range(self.depth):
			
			if(i < self.depth - 1):
				outputs.append(np.hstack((1,self.layers[i].output(x))))
				x = outputs[i]
			else:
				outputs.append(self.layers[i].output(x))
			
			#outputs.append(self.layers[i].output(x))
			#x = outputs[i]


		return outputs

	def getGradients(self, x, y):

		gradients = []
		outputs = self.getOutputs(x)


		for i in range(self.depth):
			gradients.append(np.ones((outputs[i].shape[0] - 1,1)))
			

		#Backpropagation algorithm
		for i in reversed(range(self.depth)):
			#Initial gradient computed at the output layer
			if (i == self.depth - 1):
				gradients[i] = (y - outputs[i])*self.layers[i].derivative(outputs[i-1])
				#print gradients[i].shape

			#Gradients computed backwards from the output layer to the input layer
			elif(i < self.depth - 1 and i > 0):
				#print i, self.layers[i+1].weights.shape
				#print outputs[i].shape
				#print gradients[i].shape
				#print gradients[i+1].shape
				gradients[i]  = (np.dot(self.layers[i+1].weights.T[1:,:], gradients[i+1]) * 
								self.layers[i].derivative(outputs[i-1]))
			else:
				gradients[i]  = (np.dot(self.layers[i+1].weights.T[1:,:], gradients[i+1]) * 
								self.layers[i].derivative(x))
				

		return gradients

	def train(self, X, Y, number_epochs):

		#Initialize biases along with inputs to the network
		X = np.hstack((np.ones((X.shape[0],1)), X))

		#Stochastic Gradient Descent
		for i in range(number_epochs):

			sample_number = np.random.randint(X.shape[0])
			x = X[sample_number,:]
			y = Y[sample_number]

			outputs = self.getOutputs(x)
			gradients = self.getGradients(x,y)
			#x = x.reshape((X.shape[1], 1))
			
			for i in range(self.depth):
				#print self.layers[i].weights.shape
				#print i
				if (i == 0):
					self.layers[i].weights =self.layers[i].weights + (self.learning_rate*
										np.outer(gradients[i],x.T))
				#elif (i > 0 and i < self.depth - 1):
				#	self.layers[i].weights =self.layers[i].weights + (self.learning_rate*
				#						np.outer(gradients[i],outputs[i-1]))
				else:
					self.layers[i].weights =self.layers[i].weights + (self.learning_rate*
										np.outer(gradients[i],outputs[i-1]))

		

	def predict(self, X, Y):

		#Initialize biases along with inputs to the network
		X = np.hstack((np.ones((X.shape[0],1)), X))
		outputs = np.ones(Y.shape)

		for i in range(outputs.shape[0]):
			x = X[i,:]
			outputs[i] = self.getOutputs(x)[self.depth-1]
		
		loss = 0.5 * np.sum(np.power((Y - outputs),2))
		print "Loss:", loss
		return outputs

if __name__ == "__main__":
	layers = []

	layer_1 = Sigmoid(nodes = 10, input_size = 11)
	layer_2 = Sigmoid(nodes = 9, input_size = 10)
	layer_3 = Sigmoid(nodes = 10, input_size = 9)

	layers.extend((layer_1,layer_2,layer_3))

	#model = Network(layers)