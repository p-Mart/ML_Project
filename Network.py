import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from Layers import *

class Network:

	def __init__(self, layers, 
			learning_rate=0.01, reg=0.0001, mu=0,
			batches=10, func="squared error"):

		self.layers = layers
		self.depth = len(layers)

		#Hyperparameters
		self.reg = reg#Regularization term
		self.learning_rate = learning_rate
		self.mu = mu#Momentum
		self.batches = batches

		self.func = func
		self.outputs = []

	def lossFunction(self, y, output):
		'''Computes the loss on a prediction relative to the 
		correct prediction.'''

		#Regularization term
		reg_term = 0.5 * self.reg
		sum_weights = 0.
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
			#Might need to do 1 / N
			i = list(y).index(1.)
			if(output[i] == 0):
				loss = np.inf
			else:
				loss = (-np.log(output[i])
					 	+ reg_term)

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
		gradients = []
		for i in range(self.depth):
			gradients.append(np.zeros((self.layers[i].output_size,1)))

		#Backpropagation algorithm
		for i in reversed(range(self.depth)):
			#Initial gradient computed at the output layer
			if (i == self.depth - 1):
				#print self.outputs[i-1].shape
				gradients[i] = (1./self.batches)*(self.lossDerivative(y, self.outputs[i]) *
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

		num_samples = X.shape[0]

		#Keeping track of the best result
		best_loss = np.inf
		loss = 0.
		losses = [] #Keep track of all losses

		v = [] #Momentum of each layer
		for i in range(self.depth):
			v.append(np.zeros(self.layers[i].weights.shape))

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

					x = X[j+k,:]
					#print x.shape
					y = Y[j+k]
					y = y.reshape((len(y), 1))

					#Get outputs for this batch
					self.outputs = self.getOutputs(x)
					current_grad = self.getGradients(x, y)
					#Sum Gradient contribution
					for d in range(self.depth):
						gradients[d]+= current_grad[d]

					time.sleep(0.1) #Take this out if you can run at 100% CPU usage

				#Average the gradients
				for i in range(self.depth):
					gradients[i] /= self.batches

				#Weight update
				for i in range(self.depth):
					#print "Weights", i, self.layers[i].weights.shape
					#print "Gradients", i, gradients[i]
					#print np.count_nonzero(gradients[i])
					#print x.shape
					#print i
					if (i == 0):
						if(self.layers[i].__class__.__name__ == "Convolutional"):
							#Gradient of the cost function with respect to weight
							dweights =(self.learning_rate 
									* self.layers[i].weightUpdate(gradients[i], x))

							#Calculate momentum
							if(epoch == 0):
								v[i] = dweights
							else:
								v[i] = v[i]*self.mu + dweights
								#v[i] = dweights

							#Update weights
							self.layers[i].weights = self.layers[i].weights - v[i]

						else:
							#Gradient of the cost function with respect to weight
							dweights = (self.learning_rate
										* np.outer(gradients[i], np.hstack(([1.], x.flatten()))
										))

							#Calculate momentum
							if(epoch == 0):
								v[i] = dweights
							else:
								v[i] = v[i]*self.mu + dweights
								#v[i] = dweights

							#Update weights
							self.layers[i].weights = self.layers[i].weights - v[i]
					else:
						#Dont update weights on max pool layer, they update internally
						if(self.layers[i].__class__.__name__ == "MaxPool"):
							continue

						elif(self.layers[i].__class__.__name__ == "Convolutional"):
							dweights = (self.learning_rate 
									* self.layers[i].weightUpdate(gradients[i], self.outputs[i-1].flatten()))

							#Calculate momentum
							if(epoch == 0):
								v[i] = dweights
							else:
								v[i] = v[i]*self.mu + dweights
								#v[i] = dweights

							self.layers[i].weights = self.layers[i].weights - v[i]

						else:

							dweights = self.learning_rate*np.outer(gradients[i],
								 	np.hstack(([1.], self.outputs[i-1].flatten()))
								 	)

							#Calculate momentum
							if(epoch == 0):
								v[i] = dweights
							else:
								v[i] = v[i]*self.mu + dweights
								#v[i] = dweights

							self.layers[i].weights = self.layers[i].weights - v[i]

					#Weight decay
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

		for i in range(Y.shape[0]):
			sys.stdout.write("Progress: [%d / %d]\r" % (i+1, Y.shape[0]))
			sys.stdout.flush()

			x = X[i,:]
			predictions[i, :] = self.getOutputs(x)[self.depth-1].flatten()
		
			time.sleep(0.08) #Take this out if you can run at 100% CPU usage

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

