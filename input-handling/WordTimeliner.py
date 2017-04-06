import requests
import json
from WordWithTimeline import WordWithTimeline

def getListOfWords(transcript, audio):


	r = requests.post('http://localhost:8765/transcriptions?async=false', files={'audio': audio, 'transcript': transcript})

	response = json.loads(r.content)

	words = {};

	i = 0;
	for word in response['words']:
		text = word['word']
		startTime = word['start']
		endTime = word['end']

		words[i] = WordWithTimeline(text, startTime, endTime)
		i += 1

	return words
