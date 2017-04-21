from __future__ import division
import WordTimeliner
import FeaturesExtractor
import os

directory = '/home/genous/Downloads/LibriSpeech/dev-clean/'

for filename in os.listdir(directory):
	for filename2 in os.listdir(os.path.join(directory, filename)):
		for filename3 in os.listdir( os.path.join(directory, filename, filename2)):
		    
		    if filename3.endswith(".txt"):
		    	with open(os.path.join(directory, filename, filename2, filename3), 'rd') as f:
		    		for line in f:
		    			audioFilePath = os.path.join(directory, filename, filename2, line.split(' ', 1)[0] + ".flac")
		    			print(audioFilePath)

		    			transcript = line.split(' ', 1)[1]
		    			print(transcript)


		    			with open(audioFilePath, 'rd') as a:
		    				timelinedWords = WordTimeliner.getListOfWords(transcript, a)

		    				for i in range(len(timelinedWords)):
	    					
		    					features = FeaturesExtractor.getFeatures(timelinedWords[i], audioFilePath)

		    					#!!!!!!!!!!!!!!!!!!!!!!! Do SOMETING WITH FEATURES HERE !!!!!!!!!!!!!!!!!!!!!!!
		    					# They have a variable size so we probably need to do something about that