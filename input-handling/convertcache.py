from WordWithTimeline import WordWithTimeline
import numpy as np
import FeaturesExtractor
import MainStoreDictFile
import os
import fileinput

directory = '/home/dev-clean/'

for filename in os.listdir(directory):
	
	if os.path.isdir(directory + "/" + filename):
		
		for filename2 in os.listdir(os.path.join(directory, filename)):
			
			if os.path.isdir(directory + "/" + filename + "/" + filename2):
				
				for filename3 in os.listdir( os.path.join(directory, filename, filename2)):
				    
				    if filename3 == "cache-file":
				    	for line in fileinput.input(os.path.join(directory, filename, filename2, filename3), inplace=True):
				    		print line.replace("/home/genous/Downloads/LibriSpeech/", "/home/"),


