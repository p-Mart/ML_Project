import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from Layers import *

class Network:

	def __init__(self, layers, 
			learning_rate=0.01, reg=0.0001, 
			batches=1, func="categorical crossentropy"):

		self.layers = layers
		self.depth = len(layers)

		#Hyperparameters
		self.reg = reg #Regularization term
		self.learning_rate = learning_rate
		#self.mu = mu #Momentum
		self.batches = batches

		self.func = func
		self.outputs = []

	##############################
	#Functions responsible for the cost function of the
	#network
	##############################

	def lossFunction(self, y, output):
		'''Computes the loss on a prediction relative to the 
		correct prediction.'''

		#Regularization term
		reg_term = 0.5 * self.reg
		sum_weights = 0.
		#Get the sum of all W^T  * W terms in the network.
		#Not including the weights on bias terms.
		for i in range(self.depth):
			if(self.layers[i].__class__.__name__ == "MaxPool"):
				continue
			else:
				sum_weights += (np.sum(
							    np.power(self.layers[i].weights[:, 1:], 2)
							    ))

		reg_term = reg_term * sum_weights


		#Error functions
		if (self.func == "squared error"):
			return (0.5 * np.sum(np.power((y - output),2))
					+ reg_term)

		elif(self.func == "categorical crossentropy"):
			i = list(y).index(1.)
			if(output[i] == 0):
				loss = np.inf
			else:
				loss = (-np.log(output[i])
					 	+ reg_term)

			return loss

	def lossDerivative(self, y, output):
		'''Computes the derivative of the loss function.'''
		if (self.func == "squared error"):
			return (-(y - output))
		elif (self.func == "categorical crossentropy"):
			derivative = np.array(output)
			i = list(y).index(1.) #Get the index containing the correct class
			derivative[i] = (derivative[i] - 1.)

			return derivative

	##############################
	#Functions responsible for getting the forward 
	#and backwards pass of terms in the network
	##############################

	def getOutputs(self, sample):
		'''Calculates the outputs of the network.'''
		outputs = []
		x = sample

		#Feedforward calculation of the outputs at each layer.
		for i in range(self.depth):
			outputs.append(self.layers[i].output(x))
			x = outputs[i]

		return outputs

	def getGradients(self, x, y):
		'''
		Calculate the backwards pass of delta/error terms.
		The number of gradients per layer is the number of nodes(excluding
		the bias unit) in that layer.
		'''
		#Holds the gradients in each layer
		gradients = []
		for i in range(self.depth):
			gradients.append(np.zeros((self.layers[i].output_size,1)))

		#Backpropagation algorithm for delta terms
		for i in reversed(range(self.depth)):
			#Initial gradient computed at the output layer
			if (i == self.depth - 1):
				gradients[i] = (1./self.batches)*(self.lossDerivative(y, self.outputs[i]) *
								self.layers[i].derivative(self.outputs[i-1].flatten()))

			#Gradients computed backwards from the output layer to the
			#first hidden layer
			elif(i < self.depth - 1 and i > 0):
				#Delta term is calculated internally when propagating through
				#a convolutional layer
				if(self.layers[i+1].__class__.__name__ == "Convolutional"):
					gradients[i] = (self.layers[i+1].gradient(gradients[i+1]).flatten()
									* self.layers[i].derivative(self.outputs[i-1].flatten()))
				#Delta term for non-convolutional hidden layers
				else:					
					gradients[i]  = (np.dot(self.layers[i+1].weights.T[1:,:], gradients[i+1])  
									* self.layers[i].derivative(self.outputs[i-1].flatten()))

			#Gradients computed at the input layer
			else:
				gradients[i]  = (np.dot(self.layers[i+1].weights.T[1:,:], gradients[i+1]) * 
								self.layers[i].derivative(x.flatten()))

			gradients[i] = gradients[i].reshape((gradients[i].shape[0], 1))	

		return gradients

	##############################
	#Functions responsible for training the network
	#and using it to predict on a new dataset
	##############################

	def train(self, X, Y, number_epochs):
		'''
		Trains the network for a given number of epochs.
		'''

		num_samples = X.shape[0]

		#Keeping track of the best result
		best_loss = np.inf
		loss = 0.
		losses = [] #Keep track of all losses

		#v = [] #Momentum of each layer
		#for i in range(self.depth):
		#	v.append(np.zeros(self.layers[i].weights.shape))

		#Accumulated gradients in each layer
		gradients = []
		for i in range(self.depth):
			gradients.append(np.zeros((self.layers[i].output_size,1)))

		#Batch Gradient Descent
		for epoch in range(number_epochs):
			
			sys.stdout.write("Epoch: %d\n" % (epoch+1))

			#Iterate over batches
			for j in range(0, num_samples, self.batches):

				#Reset the accumulated gradient to zero
				for i in range(self.depth):
					gradients[i] *= 0

				#Sum the gradients across all the examples in the batch
				for k in range(self.batches):

					sys.stdout.write("Training Progress: [%d / %d] \r" %(k+j+1, num_samples))
					sys.stdout.flush()

					#Get a sample
					x = X[j+k,:]
					y = Y[j+k]
					y = y.reshape((len(y), 1))

					#Get outputs for this sample
					self.outputs = self.getOutputs(x)
					current_grad = self.getGradients(x, y)

					#Sum gradient contribution through the batch
					for d in range(self.depth):
						gradients[d]+= current_grad[d]

					#time.sleep(0.12) #Take this out if you can run at 100% CPU usage

				#Average the gradients
				for i in range(self.depth):
					gradients[i] /= self.batches

				#Weight update algorithm from input layer to output layer
				for i in range(self.depth):
					#Gradient at the input layer
					if (i == 0):
						#Gradient is calculated internally for convolutional layer
						if(self.layers[i].__class__.__name__ == "Convolutional"):
							dweights =(self.learning_rate 
									* self.layers[i].weightUpdate(gradients[i], x))
						#Gradient of a non-convolutional layer w.r.t. input layer
						else:
							dweights = (self.learning_rate
										* np.outer(gradients[i], np.hstack(([1.], x.flatten()))
										))
					#Gradient at all the other layers
					else:
						#Dont update weights on max pool layer, they update internally
						if(self.layers[i].__class__.__name__ == "MaxPool"):
							continue
						#Gradient is calculated internally for convolutional layer
						elif(self.layers[i].__class__.__name__ == "Convolutional"):
							dweights = (self.learning_rate 
									* self.layers[i].weightUpdate(gradients[i], self.outputs[i-1].flatten()))
						#Gradient with respect to the output of a hidden layer
						else:
							dweights = self.learning_rate*np.outer(gradients[i],
								 	np.hstack(([1.], self.outputs[i-1].flatten()))
								 	)

					#Calculate momentum
					#if(epoch == 0):
						#v[i] = dweights
					#else:
						#v[i] = v[i]*self.mu + dweights
						#v[i] = dweights

					#Update weights at current layer
					self.layers[i].weights = self.layers[i].weights - dweights

					#Weight decay at current layer
					self.layers[i].weights[:, 1:] -= (self.reg* self.layers[i].weights[:, 1:])
						
				
				#Calculate the loss on this batch.
				for k in range(self.batches):
					loss += self.lossFunction(Y[j+k], self.outputs[self.depth-1])

			#Average the loss over the number of samples
			#Store best loss per epoch
			loss /= num_samples	
			if(loss < best_loss):
				best_loss = loss

			print "\nLoss: ", loss
			losses.append(loss[0])			
		
		#print y	
		sys.stdout.write("\nBest Loss: %f\n" % (best_loss))
		sys.stdout.flush()

		return losses

	def predict(self, X, Y):
		'''Outputs predictions for a given dataset.'''
		predictions = np.ones(Y.shape)

		#Take off dropout when predicting
		for i in range(self.depth):
			if(self.layers[i].__class__.__name__== "Relu"):
				self.layers[i].dropout = False

		#Predict on dataset
		for i in range(Y.shape[0]):
			sys.stdout.write("Progress: [%d / %d]\r" % (i+1, Y.shape[0]))
			sys.stdout.flush()

			x = X[i,:]
			self.outputs = self.getOutputs(x)
			predictions[i, :] = self.outputs[self.depth-1].flatten()
		
			#time.sleep(0.08) #Take this out if you can run at 100% CPU usage

		return predictions

	##############################
	#Weight saving and loading
	##############################	

	def save(self, file_name):
		for i in range(self.depth):
			f = file_name + '_' + str(i)
			np.save(f, self.layers[i].weights)

	def load(self, file_name):
		for i in range(self.depth):
			f = file_name + '_' + str(i) + '.npy'
			print self.layers[i].weights.shape
			self.layers[i].weights = np.load(f)