import numpy as np

def get_hann_window(n):
    window = np.sin(np.pi * np.arange(n) / n) ** 2

    return window

def get_hamming_window(n):
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / n)

    return window

def get_rectangular_window(n):
    window = np.ones(n)

    return window

def get_window(window_type, n):
    """Returns a window of the specified type with the specified length"""
    if window_type == 'hann':
        return get_hann_window(n)
    elif window_type == 'hamming':
        return get_hamming_window(n)
    elif window_type == 'rectangular':
        return get_rectangular_window(n)
    else:
        raise ValueError('Invalid window type')
    
def check_window_correctness(audio, window, overlap):
    # pads the audio with zeroe's to make it compatible with the window with given overlap
    pad_length = (len(audio) - len(window)) % (len(window) - overlap)
    padded_audio = np.pad(audio, (0, pad_length))

    # generates windows of audio with the given window and overlap
    hop_length = len(window) - overlap
    windowed_audio = []
    for i in range(0, len(padded_audio) - len(window) + 1, hop_length):
        windowed_audio.append(padded_audio[i:i + len(window)] * window)

    # reconstructs the audio from the windowed audios
    final_audio = np.zeros(len(padded_audio))
    for i in range(0, len(windowed_audio)):
        final_audio[i * hop_length:i * hop_length + len(window)] += windowed_audio[i]

    # calculates rmse error between the original and reconstructed (from windowed audio) audio
    error = np.sum((padded_audio - final_audio) ** 2)
    error_percent = error / np.sum(padded_audio ** 2) * 100

    return error_percent