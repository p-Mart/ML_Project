from WordWithTimeline import WordWithTimeline
import FeaturesExtractor
import os

directory = '/home/genous/Downloads/LibriSpeech/dev-clean/'

word_to_index = {}
index_to_word = {}

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
				    			
				    			if len(word_to_index) == 0:
				    				word_to_index[word] = 0
				    				index_to_word[0] = word
				    			else:
				    				if word not in word_to_index:
				    					word_to_index[word] = max(word_to_index.values()) + 1
				    					index_to_word[max(word_to_index.values()) + 1] = word

				    			startTime = float(line.split(' ')[2])
				    			endTime = float(line.split(' ')[3])

				    			with open(audioFilePath, 'rd') as a:
				    				timelinedWord = WordWithTimeline(word, startTime, endTime)
				    				
			    					features = FeaturesExtractor.getFeatures(timelinedWord, audioFilePath)
			    					print(audioFilePath)
			    					print(word)
			    					print("Start time " + str(startTime))
				    				print("End time " + str(endTime))
			    					print("Feature size: " + features.size)

			    					#!!!!!!!!!!!!!!!!!!!!!!! Do SOMETING WITH FEATURES HERE !!!!!!!!!!!!!!!!!!!!!!!
			    					# They have a variable size so we probably need to do something about that