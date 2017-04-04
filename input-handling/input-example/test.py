import requests

print('start')

with open('text.txt', 'rb') as t, open('audio.flac', 'rb') as a:
	r = requests.post('http://localhost:8765/transcriptions?async=false', files={'audio': a, 'transcript': t})
	print(r.content)

print('end')