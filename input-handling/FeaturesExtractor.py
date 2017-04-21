from __future__ import division
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import math
import soundfile as sf
import scipy
import matplotlib.pyplot as plt



def calculate_starting_bin_index(bin_frequency_increment):
    return int(20 / bin_frequency_increment)


def calculate_ending_bin_index(bin_frequency_increment, sampling_rate):
    ideal_ending_bin_index = 40000 / bin_frequency_increment

    if ideal_ending_bin_index > (sampling_rate / 2) / bin_frequency_increment:
        return int((sampling_rate / 2) / bin_frequency_increment)
    else:
        return int(ideal_ending_bin_index)


#uses FFT
def getFeaturesFFT(timelinedWord, audioFilePath, feature_vector_size):
	(sig, rate) = sf.read(audioFilePath)
	sig_start = int(timelinedWord.getStartTime() * rate)
	sig_end = int(timelinedWord.getEndTime() * rate)

	sig_of_current_word = sig[sig_start:sig_end]

	result = np.fft.fft(sig_of_current_word, feature_vector_size)

	#non_mirrored_result = np.resize(result, len(result) / 2)
	
	'''
	frequency_increment = rate / (len(sig_of_current_word) * 1.0)
	
	if frequency_increment == 0:
		print(rate)
		print(len(sig_of_current_word))
		print "Frequenct increments is 0!!!"

	non_mirrored_result = result[calculate_starting_bin_index(frequency_increment):calculate_ending_bin_index(frequency_increment, rate)]
	'''
	#normalized_amplitudes = np.abs(non_mirrored_result[:]) * (2.0 / feature_vector_size)
	#x = np.abs(non_mirrored_result[:])
	#normalized_amplitudes = (x - np.mean(x)) / np.std(x)


	#features = np.zeros(feature_vector_size)
	features = result

	'''
	pad_size = int(math.ceil(float(len(normalized_amplitudes))/feature_vector_size)*feature_vector_size - len(normalized_amplitudes))
	padded = np.append(normalized_amplitudes, np.zeros(pad_size)*np.NaN)
	shrink_size = int(len(padded) / feature_vector_size)
	
	raw_index = 0
	for i in range(0, feature_vector_size):
		for j in range(shrink_size):
			features[i] += padded[raw_index]
			raw_index += 1
	'''



	#run through a "mid-pass filter"
	range_of_filter = len(features)
	sin = np.zeros(range_of_filter)
	for i in range(range_of_filter):
		sin[i] = 20 * np.sin(i * (np.pi /range_of_filter))
		features[i] *= sin[i]


	features = (features - np.mean(features)) / np.std(features)

	

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