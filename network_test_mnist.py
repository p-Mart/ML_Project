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

layer_2 = MaxPool(input_shape=layer_1.output_shape,
					receptive_field=2,
					stride=2)

layer_3 = Relu(layer_2.output_shape, layer_2.output_size)

layer_4 = Softmax(layer_3.output_shape, n_classes)
'''
layer_1 = Relu(input_shape=(28,28,1),nodes=784)
layer_2 = Softmax(input_shape=layer_1.output_shape, nodes=10)
'''
model = Network(
				[layer_1, layer_2, layer_3, layer_4],
				learning_rate = 0.01,
				batches=1,
				func = "categorical crossentropy"
				)

#Plot the losses
for i in range(10):
	losses = model.train(X_train[:100], Y_train[:100], number_epochs = 1)

#model.load("mnist_model")
model.save("mnist_model")

#plt.plot(np.linspace(1, len(losses), len(losses)), np.array(losses))
#plt.show()

del X_train, Y_train, train_set

#Predict on the validation set
X_valid = valid_set[0]
Y_valid = toLogit(valid_set[1])
predictions = model.predict(X_valid[:10,:], Y_valid[:10,:])


#Visualizing the output of conv layer
conv_layers_visual = model.outputs[0][:,:,0]
#print model.outputs[3]
for i in range(7):
	conv_layers_visual = np.concatenate(
							(conv_layers_visual, model.outputs[0][:,:,i+1]),
							axis = 1)

plt.imshow(conv_layers_visual)
plt.show()

accuracy = 0.
num_correct =0.
for i in range(predictions.shape[0]):
	max_index = np.argmax(predictions[i, :])
	predictions[i, :] = np.zeros((1, predictions.shape[1]))
	predictions[i, :][max_index] = 1.

	print predictions[i, :]

	if(Y_valid[i, :][max_index] == 1.):
		num_correct += 1.

accuracy = 100. * num_correct / float(10.)

print "\nAccuracy on validation set: ", accuracy , "%"