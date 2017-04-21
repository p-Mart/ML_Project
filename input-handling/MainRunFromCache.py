from __future__ import division
from WordWithTimeline import WordWithTimeline
from Network import *
from Layers import *
import numpy as np
import FeaturesExtractor
import MainStoreDictFile
import os


directory = '/home/dev-clean/'

data_size = 1000
feature_length = 625

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


def initialize():
	index = 0
	
	all_features = np.empty((data_size, feature_length))
	all_output_labels = np.empty((data_size, size_of_output_labels_vector))  #  loop through all words, call word_to_index[word] and assign it to np.zeros

	for filename in os.listdir(directory):
		
		if os.path.isdir(directory + "/" + filename):
			
			for filename2 in os.listdir(os.path.join(directory, filename)):
				
				if os.path.isdir(directory + "/" + filename + "/" + filename2):
					
					for filename3 in os.listdir( os.path.join(directory, filename, filename2)):
					    
					    if filename3 == "cache-file":
					    	with open(os.path.join(directory, filename, filename2, filename3), 'rd') as f:
					    		for line in f:

					    			audioFilePath = os.path.join(directory, filename, filename2, line.split(' ')[0])
					    			word = line.split(' ')[1]
					    			startTime = float(line.split(' ')[2])
					    			endTime = float(line.split(' ')[3])

					    			with open(audioFilePath, 'rd') as a:
					    				timelinedWord = WordWithTimeline(word, startTime, endTime)
					    				

				    					all_features[index, :] = FeaturesExtractor.getFeaturesFFT(timelinedWord, audioFilePath, feature_length)
				    					all_output_labels[index, :] = np.zeros(size_of_output_labels_vector)
				    					
				    					all_output_labels[index, word_to_index[word]] = 1
				    					index += 1

				    					if index >= data_size:
				    						return all_output_labels, all_features


all_output_labels, all_features = initialize()

print("DONE INITIALIZING")


n_classes = size_of_output_labels_vector
	
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

model.train(all_features, all_output_labels, number_epochs = 20)

predictions = model.predict(all_features, all_output_labels)

for i in range(predictions.shape[0]):
	max_index = np.argmax(predictions[i, :])
	predictions[i, :] = np.zeros((1, predictions.shape[1]))
	predictions[i, :][max_index] = 1.

	print predictions[i, :]
