from utils.window import get_window, check_window_correctness
import librosa
from scipy.signal import stft
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging

def make_spectrogram(waveform, window, n, overlap, sr):
    """Function to create spectrograms using Short Time Fourier Transform"""
    _, _, Zxx = stft(waveform, window=window, nperseg=n, noverlap=overlap, fs=sr)
    spectrogram = np.abs(Zxx)

    return spectrogram

def save_spectrogram_img(spectrogram, output_path, sr):
    """Function to save the spectrogram as a png file"""
    fig, ax = plt.subplots()
    ax.set_axis_off()
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sr, y_axis='log', x_axis='time', ax=ax)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main(window_type, n, overlap):
    logging.info(f'Generating spectrograms with window type: {window_type}, window size: {n}, overlap: {overlap}')
    # gets the window
    window = get_window(window_type, n)
    
    # the file contains y_labels for each audio file
    metadata_file = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
    
    if os.path.exists(f'train_data/spectrograms/{window_type}'):
        os.system(f'rm -r train_data/spectrograms/{window_type}')
    
    # makes directory for each class so as to save spectrograms
    for file_class in metadata_file['class'].unique():
        os.makedirs(f'train_data/spectrograms/{window_type}/{file_class}')

    # variable to count the error in windowing
    all_error = 0

    # go through all the audio files
    for root, _, files in os.walk('UrbanSound8K/audio'):
        for file in tqdm(files):
            if not file.endswith('.wav'):
                continue
            
            # load the audio file. generate spectrogram and save the spectrogram as a png
            file_path = os.path.join(root, file)
            waveform, sr = librosa.load(file_path, sr=None)
            spectrogram = make_spectrogram(waveform, window, n, overlap, sr)
            file_class = metadata_file[metadata_file['slice_file_name'] == os.path.basename(file)]['class'].values[0]
            output_path = os.path.join('train_data/spectrograms', window_type, file_class, os.path.basename(file).replace('.wav', '.png'))
            save_spectrogram_img(spectrogram, output_path, sr)

            # add the error in windowing
            all_error += check_window_correctness(waveform, window, overlap)

    # print(f'Error: {all_error/len(files)} %')
    logging.info(f'Percentage error in windowing: {all_error/len(files)} %')
    logging.info('Finished generating spectrograms')
    logging.info(f'Spectrograms saved in train_data/spectrograms/{window_type}')
    logging.info('-----------------------------------\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a spectrogram')
    parser.add_argument('--window', type=str, default='hann', help='Window type')
    parser.add_argument('--n', type=int, default=1024, help='Window size')
    parser.add_argument('--overlap', type=int, default=512, help='Overlap size')
    args = parser.parse_args()

    window_type = args.window
    n = args.n
    overlap = args.overlap

    if not os.path.exists('logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename=f'logs/processing_{window_type}.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filemode='w')

    if window_type not in ['hann', 'hamming', 'rectangular', 'all']:
        raise ValueError('Invalid window type')
    
    # run through all the window types
    if window_type == 'all':
        for window_type in ['hann', 'hamming', 'rectangular']:
            main(window_type, n, overlap)
    else:
        main(window_type, n, overlap)
