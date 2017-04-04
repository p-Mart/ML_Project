import WordTimeliner

with open('audio.flac', 'rd') as a:
	words = WordTimeliner.getListOfWords('Go do you hear', a)

	for i in words:
		print(words[i].getWord())
		print(words[i].getStartTime())
		print(words[i].getEndTime())