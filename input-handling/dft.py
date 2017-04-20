import cmath
import numpy as np

def calculate_starting_bin_index(bin_frequency_increment):
    return int(20 / bin_frequency_increment)


def calculate_ending_bin_index(bin_frequency_increment, sampling_rate):
    ideal_ending_bin_index = 20000 / bin_frequency_increment

    if ideal_ending_bin_index > (sampling_rate / 2) / bin_frequency_increment:
        return int((sampling_rate / 2) / bin_frequency_increment)
    else:
        return int(ideal_ending_bin_index)


def group_every_x_bins(target_increment, bin_frequency_increment):
    if bin_frequency_increment < target_increment:
        return int(target_increment / bin_frequency_increment)
    else:
        return 1


def calculate_dft_step(x_n, n, k, N):
    return (x_n * cmath.exp((-2j * np.pi * k * n) / N)) * (2 / N)


def extract_features_dft(signal, bit_rate, target_frequency_increment):
    feature_vector_size = (20000 - 20) / target_frequency_increment
    frequency_increment = bit_rate / len(signal)

    start_bin = calculate_starting_bin_index(frequency_increment)
    end_bin = calculate_ending_bin_index(frequency_increment, bit_rate)

    if end_bin > start_bin:
        bin = np.zeros(end_bin - start_bin, dtype=complex)
    else:
        return np.zeros(feature_vector_size)

    curr_bin_frequency = 0
    for k in range(start_bin, end_bin):
        for n in range(len(signal)):
            bin[k] += calculate_dft_step(signal[n], n, k, len(signal))

        curr_bin_frequency += frequency_increment
        print(k)

    result = np.zeros(feature_vector_size)

    for i in range(len(bin)):
        sum = 0
        for j in range(group_every_x_bins(1, frequency_increment)):
            sum += abs(bin[i + j])
        result[i] = abs(sum)

    for i in range(len(bin)):
        print(round(result[i]), frequency_increment * i, "Hz")
