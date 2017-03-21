import wave, sys


if __name__ == "__main__":
	in_wav = wave.open(sys.argv[1])
	
	total_frames = in_wav.getnframes()
	sample_width = in_wav.getsampwidth()
	channels = in_wav.getnchannels()
	framerate = in_wav.getframerate()
	
	buffer_size = 4096
	
	data = in_wav.readframes(buffer_size)
	while(data):
		sys.stdout.write(data)
		data = in_wav.readframes(buffer_size)

	in_wav.close()