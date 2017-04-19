from Network import *
from Layers import *

import numpy as np

#Model architecture
n_classes = 5

layer_1 = Convolutional(input_shape=(40,26,1),
					number_filters=8,
					spatial_extent=5,
					stride=1,zero_padding=0)
layer_2 = MaxPool(input_shape=layer_1.output_shape,
					receptive_field=2,
					stride=2)
layer_3 = Relu(layer_2.output_shape, layer_2.output_size)
layer_4 = Softmax(layer_3.output_shape, n_classes)

model = Network(
				[layer_1, layer_2, layer_3, layer_4],
				learning_rate = 0.01,
				batches=1,
				func = "categorical crossentropy"
				)

model.train(X, Y, number_epochs = 10)