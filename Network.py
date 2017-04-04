import numpy as np
import sys
from Layers import *

class Network:

	def __init__(self, layers, learning_rate = 1, func = "squared error"):
		self.layers = layers
		self.depth = len(layers)

		self.learning_rate = learning_rate
		self.func = func
		self.outputs = []

	def lossFunction(self, y, output):
		
		if (self.func == "squared error"):
			return (0.5 * np.sum(np.power((y - output),2)))
		elif(self.func == "categorical crossentropy"):
			#Might need to do 1 / N
			i = list(y).index(1.)
			if(output[i] == 0):
				loss = np.inf
			else:
				loss = -np.log(output[i])

			return loss

	def lossDerivative(self, y, output):
		if (self.func == "squared error"):
			return (-(y - output))
		elif (self.func == "categorical crossentropy"):
			derivative = np.array(output)
			i = list(y).index(1.) #Get the index containing the correct class
			derivative[i] = (derivative[i] - 1.)
			return derivative

	def getOutputs(self, sample):
		outputs = []
		x = sample

		#Feedforward calculation of the outputs at each layer.
		'''
		for i in range(self.depth):
			print i
			if(i < self.depth - 1):
				outputs.append(np.hstack((1,self.layers[i].output(x).flatten())))
				x = outputs[i]
			else:
				outputs.append(self.layers[i].output(x))
		'''
		for i in range(self.depth):
			outputs.append(self.layers[i].output(x))
			x = outputs[i]

		return outputs

	def getGradients(self, x, y):

		gradients = []
		#Calculate the backwards pass of gradients.
		#The number of gradients per layer is the number of nodes(excluding
		#the bias unit) in that layer.

		for i in range(self.depth):
			if(i < self.depth - 1):
				gradients.append(np.ones((self.outputs[i].flatten().shape[0],1)))
			else:
				#There's no bias unit on the final layer.
				gradients.append(np.ones((self.outputs[i].flatten().shape[0],1)))
		#print gradients[0].shape
		#print gradients[1].shape

		#Backpropagation algorithm
		for i in reversed(range(self.depth)):
			#Initial gradient computed at the output layer
			if (i == self.depth - 1):
				#print self.outputs[i-1].shape
				gradients[i] = (self.lossDerivative(y, self.outputs[i]) *
								self.layers[i].derivative(self.outputs[i-1].flatten()))
				#print gradients[i].shape
				#print y.shape
				#print self.lossDerivative(y,self.outputs[i]).shape
				#print i, self.outputs[i-1].flatten().shape
				#print self.layers[i].derivative(self.outputs[i-1].flatten()).shape

			#Gradients computed backwards from the output layer to the input layer
			elif(i < self.depth - 1 and i > 0):
				#print i, self.layers[i+1].weights.T[1:,:].shape
				#print gradients[i+1].shape
				#print self.layers[i].derivative(self.outputs[i-1].flatten()).shape
				#print self.outputs[i-1].flatten().shape

				gradients[i]  = (np.dot(self.layers[i+1].weights.T[1:,:], gradients[i+1]) * 
								self.layers[i].derivative(self.outputs[i-1].flatten()))
			else:
				#print i, self.layers[i+1].weights.shape
				#print outputs[i].shape
				#print gradients[i].shape
				#print gradients[i+1].shape
				gradients[i]  = (np.dot(self.layers[i+1].weights.T[1:,:], gradients[i+1]) * 
								self.layers[i].derivative(x.flatten()))
				

		return gradients

	def train(self, X, Y, number_epochs):

		#Initialize biases along with inputs to the network
		#X = np.hstack((np.ones((X.shape[0],1)), X))
		
		#Keeping track of the best result
		best_loss = np.inf

		#Stochastic Gradient Descent
		for i in range(number_epochs):
			
			#sys.stdout.write("Training progress: [%d / %d] \r" %(i + 1, number_epochs))
			#sys.stdout.flush()

			sample_number = np.random.randint(X.shape[0])
			x = X[sample_number,:]
			#print x.shape
			y = Y[sample_number]
			y = y.reshape((len(y), 1))

			self.outputs = self.getOutputs(x)
			gradients = self.getGradients(x,y)
			#print outputs
			#print gradients
			print self.layers[0].weights.shape

			#Weight update
			for i in range(self.depth):
				#print self.layers[i].weights.shape
				#print gradients[i].shape
				#print x.shape
				#print i
				if (i == 0):
					if(self.layers[i].__class__.__name__ == "Convolutional"):
						print self.layers[i].im2col(x).flatten().shape
						temp_weights = (self.learning_rate*np.outer(gradients[i],
							 	self.layers[i].im2col(x).flatten()
							 	)
							 )

						weights = self.layers[i].weightUpdate(temp_weights)
						self.layers[i].weights = self.layers[i].weights - weights

					else:
						self.layers[i].weights =self.layers[i].weights - (self.learning_rate*
										np.outer(gradients[i],
											np.hstack(([1.], x.flatten()))
											)
										)
				else:
					#print self.outputs[i-1].shape
					if(self.layers[i].__class__.__name__ == "MaxPool"):
						continue
					elif(self.layers[i].__class__.__name__ == "Convolutional"):
						temp_weights = (self.learning_rate*np.outer(gradients[i],
							 	self.layers[i].im2col(outputs[i-1]).flatten()
							 	)
							 )

						weights = self.layers[i].weightUpdate(temp_weights)
						self.layers[i].weights = self.layers[i].weights - weights
					else:
						self.layers[i].weights = (self.layers[i].weights  -
							 self.learning_rate*np.outer(gradients[i],
							 	np.hstack(([1.], self.outputs[i-1].flatten()))
							 	)
							 )
						 
			
			#Calculate the loss on this example.
			loss = self.lossFunction(y, self.outputs[self.depth-1])
			if(loss < best_loss):
				best_loss = loss
			
		#sys.stdout.write("\nBest Loss: %f\n" % (best_loss))
		#sys.stdout.flush()


	def predict(self, X, Y):

		#Initialize biases along with inputs to the network
		#X = np.hstack((np.ones((X.shape[0],1)), X))
		predictions = np.ones(Y.shape)

		for i in range(Y.shape[0]):
			x = X[i,:]
			predictions[i] = self.getOutputs(x)[self.depth-1].flatten()
		
		return predictions


#Debugging
if __name__ == '__main__':
	x = np.random.rand(32,32,3)
	x = x.flatten()
	x = x.reshape((1, len(x)))
	y = np.array([[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.]])

	layer_1 = Convolutional((32,32,3),12,1,1,0)
	#layer_1_os = layer_1.output(x).shape
	#print layer_1.weights.shape
	layer_2 = Relu((32,32,12), np.prod(np.array([32,32,12])))
	#layer_2_os = layer_2.output(layer_1.output(x)).shape
	#print layer_2.weights.shape
	layer_3 = MaxPool((32,32,12),2,2)
	#layer_3_os = layer_3.output(layer_2.output(layer_1.output(x))).shape

	layer_4 = Softmax((16,16,12), 10)
	#layer_4_os = layer_4.output(layer_3.output(layer_2.output(layer_1.output(x))))
	#print layer_4.weights.shape
	model = Network([layer_1, layer_2,layer_3,layer_4],
					learning_rate = 0.02,
					func = "categorical crossentropy"
				)
	'''
	#print "\n"
	model.outputs = model.getOutputs(x)
	#print model.outputs[0].flatten().shape
	#print model.outputs[1].flatten().shape
	#print model.outputs[2].flatten().shape
	#print model.outputs[3].flatten().shape

	print  "WEIGHTS"
	print layer_1.weights.shape
	print layer_2.weights.shape
	print layer_3.weights.shape
	print layer_4.weights.shape
	gradients = model.getGradients(x[0], y[0])
	print "GRADIENTS"
	print gradients[0].shape
	print gradients[1].shape
	print gradients[2].shape
	print gradients[3].shape

	
	print gradients[1]
	print gradients[2]
	'''
	model.train(x, y, number_epochs = 100)



