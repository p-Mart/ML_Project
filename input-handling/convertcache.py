from WordWithTimeline import WordWithTimeline
import numpy as np
import FeaturesExtractor
import MainStoreDictFile
import os
import fileinput

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

all_features = np.empty((data_size, feature_length))
all_output_labels = np.empty((data_size, size_of_output_labels_vector))  #  loop through all words, call word_to_index[word] and assign it to np.zeros


index = 0
for filename in os.listdir(directory):
	
	if os.path.isdir(directory + "/" + filename):
		
		for filename2 in os.listdir(os.path.join(directory, filename)):
			
			if os.path.isdir(directory + "/" + filename + "/" + filename2):
				
				for filename3 in os.listdir( os.path.join(directory, filename, filename2)):
				    
				    if filename3 == "cache-file":
				    	for line in fileinput.input(os.path.join(directory, filename, filename2, filename3), inplace=True):
				    		print line.replace("/home/genous/Downloads/LibriSpeech/", "/home/"),


