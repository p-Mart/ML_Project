import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from Layers import *

class Network:

	def __init__(self, layers, learning_rate = 1, func = "squared error"):
		self.layers = layers
		self.depth = len(layers)

		self.learning_rate = learning_rate
		self.func = func
		self.outputs = []

	def lossFunction(self, y, output):
		'''Computes the loss on a prediction relative to the 
		correct prediction.'''

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
		'''Compute the derivative of the loss function.'''
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
		for i in range(self.depth):
			outputs.append(self.layers[i].output(x))
			x = outputs[i]

		return outputs

	def getGradients(self, x, y):
		'''
		Calculate the backwards pass of gradients.
		The number of gradients per layer is the number of nodes(excluding
		the bias unit) in that layer.
		'''

		#Construct a list containing all the gradients
		gradients = []
		for i in range(self.depth):
			gradients.append(np.ones((self.outputs[i].flatten().shape[0],1)))

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
				#print i,  [i].shape
				#print np.count_nonzero(np.dot(self.layers[i+1].weights.T[1:,:], gradients[i+1]))
				#print self.layers[i].derivative(self.outputs[i-1].flatten()).shape
				#print self.outputs[i-1].flatten().shape
				if(self.layers[i+1].__class__.__name__ == "Convolutional"):
					gradients[i] = (self.layers[i+1].gradient(gradients[i+1]).flatten()
									* self.layers[i].derivative(self.outputs[i-1].flatten()))
				else:					
					gradients[i]  = (np.dot(self.layers[i+1].weights.T[1:,:], gradients[i+1])  
									* self.layers[i].derivative(self.outputs[i-1].flatten()))
				
				gradients[i] = gradients[i].reshape((gradients[i].shape[0], 1))

				#print gradients[i].shape
			else:
				#print i, self.layers[i+1].weights.shape
				#print outputs[i].shape
				#print i, gradients[i].shape
				#print gradients[i+1].shape
				#print self.layers[i+1].weights.shape
				gradients[i]  = (np.dot(self.layers[i+1].weights.T[1:,:], gradients[i+1]) * 
								self.layers[i].derivative(x.flatten()))

				#print gradients[i].shape
				

		return gradients

	def train(self, X, Y, number_epochs):
		'''
		Trains the network for a given number of epochs.
		'''

		#Keeping track of the best result
		best_loss = np.inf

		#Stochastic Gradient Descent
		for i in range(number_epochs):
			
			sys.stdout.write("Training progress: [%d / %d] \r" %(i + 1, number_epochs))
			sys.stdout.flush()

			sample_number = np.random.randint(X.shape[0])
			x = X[sample_number,:]
			#print x.shape
			y = Y[sample_number]
			y = y.reshape((len(y), 1))

			self.outputs = self.getOutputs(x)
			gradients = self.getGradients(x,y)

			#print gradients[0]
			#print outputs
			#print gradients
			#print self.layers[0].weights.shape

			#Weight update
			for i in range(self.depth):
				#print "Weights", i, self.layers[i].weights.shape
				#print "Gradients", i, gradients[i]
				#print np.count_nonzero(gradients[i])
				#print x.shape
				#print i
				if (i == 0):
					if(self.layers[i].__class__.__name__ == "Convolutional"):
						#print self.layers[i].im2col(x).flatten().shape
						'''
						temp_weights = (self.learning_rate*np.outer(gradients[i],
							 	self.layers[i].im2col(x).flatten()
							 	)
							 )
						'''

						dweights = (self.learning_rate 
								* self.layers[i].weightUpdate(gradients[i], x))

						self.layers[i].weights = self.layers[i].weights - dweights

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
						dweights = (self.learning_rate 
								* self.layers[i].weightUpdate(gradients[i], self.outputs[i-1].flatten()))

						self.layers[i].weights = self.layers[i].weights - dweights
					else:
						self.layers[i].weights = (self.layers[i].weights  -
							 self.learning_rate*np.outer(gradients[i],
							 	np.hstack(([1.], self.outputs[i-1].flatten()))
							 	)
							 )
						 
			time.sleep(0.1) #Take this out if you can run at 100% CPU usage
			
			#Calculate the loss on this example.
			loss = self.lossFunction(y, self.outputs[self.depth-1])
			if(loss < best_loss):
				best_loss = loss
		
		#print y	
		sys.stdout.write("\nBest Loss: %f\n" % (best_loss))
		sys.stdout.flush()


	def predict(self, X, Y):
		'''Outputs predictions for a given dataset.'''
		predictions = np.ones(Y.shape)

		for i in range(Y.shape[0]):
			x = X[i,:]
			predictions[i, :] = self.getOutputs(x)[self.depth-1].flatten()
		
			time.sleep(0.1) #Take this out if you can run at 100% CPU usage

		return predictions


#Debugging
if __name__ == '__main__':
	x = np.random.rand(20,20,1)
	plt.imshow(x[:,:,0])
	plt.show()
	x = x.flatten()
	x = x.reshape((1, len(x)))
	y = np.array([[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.]])

	layer_1 = Convolutional((20,20,1),6,1,1,0)
	#layer_1_os = layer_1.output(x).shape
	#print layer_1.weights.shape
	layer_2 = Relu((20,20,6), np.prod(np.array([20,20,6])))
	#layer_2_os = layer_2.output(layer_1.output(x)).shape
	#print layer_2.weights.shape
	layer_3 = MaxPool((20,20,6),2,2)
	#layer_3_os = layer_3.output(layer_2.output(layer_1.output(x))).shape

	layer_4 = Softmax((10,10,6), 10)
	#layer_4_os = layer_4.output(layer_3.output(layer_2.output(layer_1.output(x))))
	#print layer_4.weights.shape
	model = Network([layer_1, layer_2,layer_3,layer_4],
					learning_rate = 0.1,
					func = "categorical crossentropy"
				)

	model.train(x, y, number_epochs = 1)
	predictions = model.predict(x, y)
	plt.imshow(model.outputs[0][:,:,0])
	plt.show()
	print predictions

