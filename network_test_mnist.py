from Network import *
from Layers import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cPickle, gzip

def toLogit(y):
	Y = np.zeros((len(y), len(set(y))))
	for i in range(len(y)):
		Y[i, y[i]] = 1.

	return Y

#Load dataset
f = gzip.open("mnist.pkl.gz", 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

X_train = train_set[0]
Y_train = toLogit(train_set[1])

#Hyperparameters
n_filters = 5
n_classes = 10

layer_1 = Convolutional(input_shape=(28,28,1),
					number_filters=n_filters,
					spatial_extent=3,
					stride=1,zero_padding=0)
#layer_1_os = layer_1.output(x).shape
#print layer_1.weights.shape

layer_2 = Relu(layer_1.output_shape, np.prod(np.array(layer_1.output_shape)))
#layer_2_os = layer_2.output(layer_1.output(x)).shape
#print layer_2.weights.shape
layer_3 = MaxPool(layer_1.output_shape,2,2)
#layer_3_os = layer_3.output(layer_2.output(layer_1.output(x))).shape

layer_4 = Softmax(layer_3.output_shape, n_classes)
#layer_4_os = layer_4.output(layer_3.output(layer_2.output(layer_1.output(x))))
#print layer_4.weights.shape
model = Network(
				[layer_1, layer_2,layer_3,layer_4],
				learning_rate = 0.02,
				func = "categorical crossentropy"
				)

model.train(X_train, Y_train, number_epochs = 200)

del X_train, Y_train, train_set

X_valid = valid_set[0]
Y_valid = toLogit(valid_set[1])

predictions = model.predict(X_valid[:10,:], Y_valid[:10,:])
conv_layers_visual = model.outputs[0][:,:,0]
print model.outputs[3]
for i in range(n_filters-1):
	conv_layers_visual = np.concatenate(
							(conv_layers_visual, model.outputs[0][:,:,i+1]),
							axis = 1)

plt.imshow(conv_layers_visual)
plt.show()
print predictions
print Y_valid[:10,:]

