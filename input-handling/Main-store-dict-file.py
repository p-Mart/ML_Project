from WordWithTimeline import WordWithTimeline
import FeaturesExtractor
import os

directory = '/home/genous/Downloads/LibriSpeech/dev-clean/'

wordIndex = 0
wordToIndex = {}
indexToWord = {}

wordToIndexFile = open(directory + "/" + "wordToIndex", "w+")
#indexToWordFile = open(directory + "/" + "indexToWord", "w+")

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
				    				
			    					features = FeaturesExtractor.getFeatures(timelinedWord, audioFilePath)
			    					#print(audioFilePath)
			    					print(word)
			    					#print("Start time " + str(startTime))
				    				#print("End time " + str(endTime))
			    					#print("Feature size: " + str(features.size))

			    					if word not in wordToIndex:
				    					wordToIndex[word] = wordIndex
				    					indexToWord[wordIndex] = word
				    					wordIndex += 1
				    					wordToIndexFile.write(word + "\n")


			    					#!!!!!!!!!!!!!!!!!!!!!!! Do SOMETING WITH FEATURES HERE !!!!!!!!!!!!!!!!!!!!!!!
			    					# They have a variable size so we probably need to do something about that