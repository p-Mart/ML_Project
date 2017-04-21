from __future__ import division
from WordWithTimeline import WordWithTimeline
from Network import *
from Layers import *
import numpy as np
import matplotlib.pyplot as plt
import FeaturesExtractor
import MainStoreDictFile
import os

feature_length = 625


#directory = '/home/dev-clean/'
# directory = '/home/genous/Downloads/LibriSpeech/dev-clean'
#directory = '/home/ubuntu/dev-clean/'
directory = '/home/bubbles/Documents/dev-clean/'

# You can assume that Word_To_Index_Cache has unique words
def load_dictionary_to_global_variable():
	word_to_index = {}
	index_to_word = {}
	size_of_output_labels_vector = 1
	
	index = 0
	if not os.path.exists(os.path.join(directory, "Word_To_Index_Cache")):
		MainStoreDictFile.run()

	with open(os.path.join(directory, "Word_To_Index_Cache")) as word_cache:
		for line in word_cache.read().splitlines():
			
			word_to_index[line] = index
			index_to_word[index] = line
			index += 1

	return word_to_index, index_to_word, len(word_to_index)


word_to_index, index_to_word, size_of_output_labels_vector = load_dictionary_to_global_variable()

n_classes = size_of_output_labels_vector
	
x = int(np.sqrt(feature_length))


model_name = "VoiceRecognitionModel_4"

layer_1 = Relu(feature_length, feature_length)
layer_2 = Relu(feature_length, feature_length//3 * 2)
layer_3 = Relu(layer_2.output_shape, layer_2.output_size //3 * 2)

layer_4 = Softmax(layer_3.output_shape, n_classes)

model = Network(
				[layer_1, layer_2, layer_3, layer_4],
				learning_rate = 0.01,
				reg=0.001,
				batches=1,
				func = "categorical crossentropy"
				)

model.load(model_name)

predictions = model.predict(all_features[:n_examples], all_output_labels[:n_examples])

for i in range(predictions.shape[0]):
	max_index = np.argmax(predictions[i, :])
	predictions[i, :] = np.zeros((1, predictions.shape[1]))
	predictions[i, :][max_index] = 1.

	print index_to_word[max_index],