from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import soundfile as sf

def getFeatures(timelinedWord, audioFilePath):	
	sig, rate = sf.read(audioFilePath)

	sigStart = int(timelinedWord.getStartTime() * rate)
	sigEnd = int(timelinedWord.getEndTime() * rate)

	sigOfCurrentWord = sig[sigStart:sigEnd]

	mfcc_feat = mfcc(sigOfCurrentWord,rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	fbank_feat = logfbank(sigOfCurrentWord,rate)

	return fbank_feat