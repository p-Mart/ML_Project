import sys
import soundfile as sf

if __name__ == "__main__":
	filename = sys.argv[1]
	filename = filename[:filename.index('.')] #Remove the extension

	sig, rate = sf.read(sys.argv[1])
	sf.write(filename + ".wav", sig, rate)
