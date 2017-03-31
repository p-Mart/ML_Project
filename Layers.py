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
		x = x.reshape((1, self.input_size))
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
		x = x.reshape((1, self.input_size))
		#x = np.hstack(([[1]], x)) #Temporary for debugging
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
		x = x.reshape((1, self.input_size))
		#x = np.hstack(([[1]], x)) #Temporary for debugging
		return softmax(np.dot(self.weights, x.T))

	def derivative(self, x):
		#return (self.output(x)*(1 - self.output(x)))#This MIGHT be wrong
		return 1.

class MaxPool:

	def __init__(self, input_shape, receptive_field, stride):
		self.input_shape = input_shape
		self.f = receptive_field
		self.s = stride

	def output(self, x):
		'''Assuming x is some tensor of shape [h x w x d]
		Output of the form {nodes x 1} as per usual'''
		x = x.reshape(self.input_shape)
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

		#pool = pool.reshape((output_h*output_w*output_d, 1))
		return pool


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
		self.weights = np.random.rand(self.k, self.f * self.f*input_shape[2])

	def im2col(self, x):
		x = x.reshape(self.input_shape)
		total_fields = self.w*self.h
		x_h,x_w,x_d = x.shape
		X_col = np.empty((self.f*self.f*x_d,total_fields))

		k = 0
		for i in range(0,x_h-self.f+1,self.s):
			for j in range(0,x_w-self.f+1,self.s):
				receptive_field = x[i:(i+self.f),j:(j+self.f),:]
				receptive_field=receptive_field.reshape(
							self.f*self.f*x_d)
				X_col[:,k] = receptive_field
				k += 1
		
		return X_col

	def output(self, x):
		x_col = self.im2col(x)
		out = np.dot(self.weights,x_col)
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
