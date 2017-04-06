import requests
import json
from WordWithTimeline import WordWithTimeline

def getListOfWords(transcript, audio):

	r = requests.post('http://localhost:8765/transcriptions?async=false', files={'audio': audio, 'transcript': transcript})

	response = json.loads(r.content)

	words = {};

	i = 0;
	for word in response['words']:
		if 'word' in word and 'start' in word and 'end' in word:
			text = word['word']
			startTime = word['start']
			endTime = word['end']

			words[i] = WordWithTimeline(text, startTime, endTime)
			i += 1
		else:
			print('Failed aligning... skipping word')

	return words
