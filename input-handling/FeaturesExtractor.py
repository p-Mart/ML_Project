from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import soundfile as sf

#uses FFT
def getFeaturesFFT(timelinedWord, audioFilePath, feature_vector_size):
	(sig, rate) = sf.read(audioFilePath)
	sig_start = int(timelinedWord.getStartTime() * rate)
	sig_end = int(timelinedWord.getEndTime() * rate)

	sig_of_current_word = sig[sig_start:sig_end]

	result = np.fft.fft(sig_of_current_word)
	normalized_non_mirrored_amplitudes = abs(result[:int(len(result) / 2)]) * (2 / len(sig_of_current_word))

	features = np.zeros(feature_vector_size)

	shrink_size = int(len(normalized_non_mirrored_amplitudes) / feature_vector_size)

	raw_index = 0
	for i in range(0, feature_vector_size):
		for j in range(shrink_size):
			features[i] += normalized_non_mirrored_amplitudes[raw_index]
			raw_index += 1

	return features

def getFeaturesMFCC(timelinedWord, audioFilePath):
	sig, rate = sf.read(audioFilePath)

	sigStart = int(timelinedWord.getStartTime() * rate)
	sigEnd = int(timelinedWord.getEndTime() * rate)

	sigOfCurrentWord = sig[sigStart:sigEnd]

	mfcc_feat = mfcc(sigOfCurrentWord,rate)
	d_mfcc_feat = delta(mfcc_feat, 2)
	fbank_feat = logfbank(sigOfCurrentWord,rate)

	return fbank_feat