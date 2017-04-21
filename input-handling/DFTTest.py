from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


space = 500

sin = np.zeros(space)
for i in range(space):
	sin[i] = np.sin(i * (np.pi /space))

plot1 = plt.plot(sin)
plt.show(plot1)

# N = 8  # signal size
# signal = [0, 0.707, 1, 0.707, 0, -0.707, -1, -0.707]
# bit_rate = 8  # 8 samples per second

# sig = signal
# (sig, bit_rate2) = sf.read("sin_1000Hz_1s.wav")
# (sig, bit_rate2) = sf.read("fsew0_001.wav")


# feature_vector_size = 400

# result = np.fft.fft(sig)
# normalized_non_mirrored_amplitudes = abs(result[:int(len(result)/2)]) * (2/len(sig))

# features = np.zeros(feature_vector_size)

# shrink_size = int(len(normalized_non_mirrored_amplitudes) / feature_vector_size)

# raw_index = 0
# for i in range(0, feature_vector_size):
#     for j in range(shrink_size):
#         features[i] += normalized_non_mirrored_amplitudes[raw_index]
#         raw_index += 1

# #plot1 = plt.plot(normalized_non_mirrored_amplitudes)
# plot2 = plt.plot(features)
# #plt.show(plot1)
# plt.show(plot2)

