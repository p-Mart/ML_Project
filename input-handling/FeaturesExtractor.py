from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import soundfile as sf

# (rate,sig) = wav.read("bird.wav")

# oneWord = len(sig)/8
# newSig = sig[:oneWord]


# mfcc_feat = mfcc(sig,rate)
# d_mfcc_feat = delta(mfcc_feat, 2)
# fbank_feat = logfbank(sig,rate)

# print(len(fbank_feat))

# print(fbank_feat[:])


def getFeatures(audioFilePath):
	
	print('self in getFeatures is:', self)
	print('Audio Path in getFeatures is', audioFilePath)
	
	sig, rate = sf.read(audioFilePath)
	oneWord = len(sig)
	newSig = sig[:oneWord]


	mfcc_feat = mfcc(sig,rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	fbank_feat = logfbank(sig,rate)

	return fbank_feat