from WordWithTimeline import WordWithTimeline
import FeaturesExtractor
import os

directory = '/home/genous/Downloads/LibriSpeech/dev-clean/'

wordIndex = 0
wordToIndex = {}

wordToIndexFile = open(directory + "/" + "Word_To_Index_Cache", "w+")

for filename in os.listdir(directory):
	
	if os.path.isdir(directory + "/" + filename):
		
		for filename2 in os.listdir(os.path.join(directory, filename)):
			
			if os.path.isdir(directory + "/" + filename + "/" + filename2):
				
				for filename3 in os.listdir( os.path.join(directory, filename, filename2)):
				    
				    if filename3 == "cache-file":
				    	with open(os.path.join(directory, filename, filename2, filename3), 'rd') as f:
				    		for line in f:
				    			word = line.split(' ')[1]
			    				print(word)

			    				if word not in wordToIndex:
				    				wordToIndex[word] = wordIndex
				    				wordToIndexFile.write(word + "\n")