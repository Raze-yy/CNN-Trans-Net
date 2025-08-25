# utils/preprocessing.py

import numpy as np

def process_input_spectra(input_spectra):
    """
    Extract and combine source and emission spectra.
    Returns shape: [N, 49, 13]
    """
    wave = input_spectra[:, :, 0:1]
    source = input_spectra[:, :, 1:12:2]
    emission = input_spectra[:, :, 2:13:2]
    return np.concatenate([wave, source, emission], axis=2)