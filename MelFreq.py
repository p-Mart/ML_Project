import sys

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
	filename = sys.argv[1]

	(rate,sig) = wav.read(filename)  #Get sample rate and audio signal from input wav

	#These are the default values for the number of coefficients and number of filters,
	#respectively, used in the function mfcc()
	numcep = 13
	nfilt = 26
	mfcc_feat = mfcc(sig, rate,numcep=13,nfilt=26) #Calculate the mel-frequency spectral coefficients (MFCC)

	plt.imshow(mfcc_feat.T, cmap = cm.jet, aspect = 12.)#Plot the MFCC features for each window
	plt.show()