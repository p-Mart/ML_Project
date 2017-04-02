import numpy as np
from Nodes import *

class Sigmoid:

	def __init__(self, input_shape, nodes):
		self.input_shape = input_shape
		self.input_size = np.prod(np.array(input_shape)) + 1
		self.nodes = nodes

		self.weights = np.random.rand(nodes, self.input_size)

	def output(self, x):
		'''Returns the outputs of the nodes in this layer, of
		shape {nodes x 1}'''

		#Assuming that the input vector x comes in
		#with a shape of {1 x input_size}
		x = x.reshape((1, self.input_size - 1))
		x = np.hstack(([[1.]], x)) # Append the bias term

		return sigmoid(np.dot(self.weights, x.T))

	def derivative(self, x):
		return (self.output(x)*(1 - self.output(x)))

class Relu:

	def __init__(self, input_shape, nodes):
		self.input_shape = input_shape
		self.input_size = np.prod(np.array(input_shape)) + 1
		self.nodes = nodes
		self.weights = np.random.rand(nodes, self.input_size)


	def output(self, x):
		'''Returns the outputs of the nodes in this layer, of
		shape {nodes x 1}'''

		#Assuming that the input vector x comes in
		#with a shape of {1 x input_size}
		x = x.reshape((1, self.input_size - 1))
		x = np.hstack(([[1.]], x)) # Append the bias term

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
	def __init__(self, input_shape, nodes):
		self.input_shape = input_shape
		self.input_size = np.prod(np.array(input_shape)) + 1
		self.nodes = nodes

		self.weights = np.random.rand(nodes, self.input_size)

	def output(self, x):
		'''Returns the outputs of the nodes in this layer, of
		shape {nodes x 1}'''

		#Assuming that the input vector x comes in
		#with a shape of {1 x input_size}
		x = x.reshape((1, self.input_size - 1))
		x = np.hstack(([[1.]], x)) # Append the bias term
		
		return softmax(np.dot(self.weights, x.T))

	def derivative(self, x):
		#return (self.output(x)*(1 - self.output(x)))#This MIGHT be wrong
		return 1.

class MaxPool:

	def __init__(self, input_shape, receptive_field, stride):
		self.input_shape = input_shape
		self.input_size = np.prod(np.array(input_shape))

		self.f = receptive_field
		self.s = stride

		h, w, d = input_shape
		self.output_h = (h - self.f)/self.s + 1
		self.output_w = (w - self.f)/self.s + 1
		self.output_d = d

		self.output_size = self.output_h*self.output_w*self.output_d

		self.weights  = np.zeros((self.output_size, self.input_size))
		#self.weights = np.hstack((np.zeros((self.output_size,1)), self.weights))
		#self.gradient = np.zeros(self.input_shape)

	def output(self, x):

		f = self.f #Size of partitions being looked at
		s = self.s #Stride over the input

		'''Assuming x is some tensor of shape [h x w x d]
		Output of the form {nodes x 1} as per usual'''
		x = x.reshape(self.input_shape)

		#Reset the gradient
		self.weights = np.zeros((self.output_size, self.input_size))
		
		#Compute the output dimensions of the max pool layer
		h, w, d = x.shape

		pool = np.empty((self.output_h,self.output_w,self.output_d))
		#This could probably be significantly optimized	
		#Max pooling operation
		for k in range(d):
			for i in range(0,h-f+1,s):
				for j in range(0,w-f+1,s):
					#Get the max value in the receptive field
					max_value = np.max(x[i:(i+f),j:(j+f),k])
					#Get the index of the maximum value in the receptive field
					r_max_index = np.unravel_index(np.argmax(x[i:(i+f),j:(j+f),k]), 
													x[i:(i+f),j:(j+f),k].shape)


					pool[i/s,j/s,k] = max_value
					pool_node = np.ravel_multi_index((i/s, j/s, k), pool.shape)
					#self.gradient[i:(i+f),j:(j+f),k][max_index[0],max_index[1]] = 1.
					max_index = (r_max_index[0] + i, r_max_index[1] + j, k)
					forward_node = np.ravel_multi_index(max_index, x.shape)
					self.weights[pool_node, forward_node] = 1.

		self.weights = np.hstack((np.zeros((self.output_size, 1)), self.weights))

		#pool = pool.reshape((output_h*output_w*output_d, 1))
		return pool

	def derivative(self, x):
		#return self.gradient.flatten()
		return 1.


class Convolutional:

	def __init__(self, input_shape, number_filters,spatial_extent,stride,zero_padding):
		
		self.input_shape = input_shape

		self.k = number_filters
		self.f = spatial_extent
		self.s = stride
		self.p = zero_padding

		#Output shape calculation
		self.w = (input_shape[1] - self.f +2*self.p)/self.s + 1
		self.h = (input_shape[0] - self.f +2*self.p)/self.s + 1
		self.d = self.k

		#Parameter sharing of weights
		#The + 1 is the weight for the bias term in each feature map
		self.weights = np.random.rand(self.k, self.f * self.f*input_shape[2] + 1)

	def im2col(self, x):
		x = x.reshape(self.input_shape)
		total_fields = self.w*self.h
		x_h,x_w,x_d = x.shape
		#Unrolled input
		X_col = np.empty((self.f*self.f*x_d,total_fields))

		k = 0
		for i in range(0,x_h-self.f+1,self.s):
			for j in range(0,x_w-self.f+1,self.s):
				receptive_field = x[i:(i+self.f),j:(j+self.f),:]
				receptive_field=receptive_field.reshape(
							self.f*self.f*x_d)
				X_col[:,k] = receptive_field
				k += 1
		
		#Append the bias term
		X_col = np.vstack((np.ones((1,total_fields)), X_col))

		return X_col

	def output(self, x):
		x_col = self.im2col(x)
		out = np.dot(self.weights,x_col)
		print self.h*self.w*self.d
		out = out.reshape(self.h,self.w,self.d)
		return out

	def derivative(self, x):
		return x #lol


#Debug section
if __name__ == '__main__':
	x = np.random.rand(32,32,3)

	layer_1 = Convolutional(x.shape,12,1,1,0)
	layer_1_os = layer_1.output(x).shape
	print layer_1_os
	print np.prod(np.array(layer_1_os))
	layer_2 = Relu(layer_1_os, np.prod(np.array(layer_1_os)))
	layer_2_os = layer_2.output(layer_1.output(x)).shape
	print layer_2_os

	layer_3 = MaxPool(layer_1_os,2,2)
	layer_3_os = layer_3.output(layer_2.output(layer_1.output(x))).shape
	print layer_3_os

	layer_4 = Softmax(layer_3_os, 10)
	layer_4_os = layer_4.output(layer_3.output(layer_2.output(layer_1.output(x))))
	print layer_4_os
