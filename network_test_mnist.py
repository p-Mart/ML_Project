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
#n_filters = 8
n_classes = 10

layer_1 = Convolutional(input_shape=(28,28,1),
					number_filters=8,
					spatial_extent=5,
					stride=1,zero_padding=0)
#layer_1_os = layer_1.output(x).shape
#print layer_1.weights.shape
#print layer_1.output_shape

layer_2 = MaxPool(input_shape=layer_1.output_shape,
					receptive_field=2,
					stride=2)
#print layer_2.output_shape
layer_3 = Convolutional(input_shape=layer_2.output_shape,
					number_filters=16,
					spatial_extent=5,
					stride=1,
					zero_padding=0)
#print layer_3.output_shape
#print layer_3.weights.shape
layer_4 = MaxPool(input_shape=layer_3.output_shape,
					receptive_field=3,
					stride=3)

layer_5 = Softmax(layer_4.output_shape, n_classes)


model = Network(
				[layer_1, layer_2,layer_3,layer_4, layer_5],
				learning_rate = 0.01,
				func = "categorical crossentropy"
				)

model.train(X_train[:1000], Y_train[:1000], number_epochs = 3)

del X_train, Y_train, train_set

X_valid = valid_set[0]
Y_valid = toLogit(valid_set[1])

predictions = model.predict(X_valid[:10,:], Y_valid[:10,:])
conv_layers_visual = model.outputs[0][:,:,0]
#print model.outputs[3]
for i in range(7):
	conv_layers_visual = np.concatenate(
							(conv_layers_visual, model.outputs[0][:,:,i+1]),
							axis = 1)

plt.imshow(conv_layers_visual)
plt.show()


conv_layers_2_visual = model.outputs[2][:,:,0]
for i in range(15):
	conv_layers_2_visual = np.concatenate(
							(conv_layers_2_visual, model.outputs[2][:,:,i+1]),
							axis = 1)


plt.imshow(conv_layers_2_visual)
plt.show()


for i in range(predictions.shape[0]):
	max_index = np.argmax(predictions[i, :])
	predictions[i, :] = np.zeros((1, predictions.shape[1]))
	predictions[i, :][max_index] = 1.

print predictions
print "\n"
print Y_valid[:10,:]

