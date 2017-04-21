from WordWithTimeline import WordWithTimeline
import FeaturesExtractor
import os
from Network import *
from Layers import *
import numpy as np

def generateDicArray(words, word):
	result = np.zeros(len(words))

	i = 0
	for item in words:
		if words[item] == word:
			result[i] = 1
			return result
		else:
			i += 1
	return 0



# directory = '/home/genous/Downloads/LibriSpeech/dev-clean/84/121123/'
directory = '84/121123'

numOfWords = 5
feature_length = 625
words = {}

outputs = np.empty((5, 5))
features = np.empty((5,feature_length))

with open(os.path.join(directory, "cache-file"), 'rd') as f:

	while numOfWords > 0:
		line = f.readline()

		words[5 - numOfWords] = line.split(' ')[1]

		audioFilePath = os.path.join(directory, line.split(' ')[0])
		startTime = float(line.split(' ')[2])
		endTime = float(line.split(' ')[3])

		# with open(audioFilePath, 'rd') as a:
		timelinedWord = WordWithTimeline(line.split(' ')[1], startTime, endTime)
		features[5 - numOfWords,:] = FeaturesExtractor.getFeaturesFFT(timelinedWord, audioFilePath, feature_length)

		numOfWords -= 1


for i in range(5):
	outputs[i] = generateDicArray(words, words[i])

# print(words)
# print(outputs)
# print(features)

n_classes = 5
	
x = int(np.sqrt(feature_length))

layer_1 = Convolutional(input_shape=(x, x, 1),
					number_filters=1,
					spatial_extent=5,
					stride=1,zero_padding=0)
print layer_1.output_shape
layer_2 = MaxPool(input_shape=layer_1.output_shape,
					receptive_field=2,
					stride=1)
layer_3 = Relu(layer_2.output_shape, layer_2.output_size)
layer_4 = Softmax(layer_3.output_shape, n_classes)

model = Network(
				[layer_1, layer_2, layer_3, layer_4],
				learning_rate = 0.01,
				batches=1,
				func = "categorical crossentropy"
				)

model.train(features, outputs, number_epochs = 100000)

print(model.predict(features, outputs))