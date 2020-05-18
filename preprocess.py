import os
import glob
import h5py
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile

DATA_PATH = './data'
SUBSET = 'train'
SLICE_DURATION = 5
SAVE_AS = 'h5'
SAMPLE_RATE = 16000
SLICE_STEP = SAMPLE_RATE * SLICE_DURATION // 2
WINDOW_MS = 30
WINDOW_LEN = SAMPLE_RATE * WINDOW_MS // 1000
STEP = WINDOW_LEN // 2
VORBIS_WINDOW = []
OUTPUT = ''
NOISE_DATAPATH = ''
NOISE_DATAPATH_TEST = ''
DIFFERENCE_FFT = True


def prepare_dataset(data_path,
                    subset='train',
                    split_valid=0,
                    slice_duration=5,
                    save_as='h5'):
    
    global DATA_PATH, SUBSET, SLICE_DURATION, SAVE_AS, OUTPUT, NOISE_DATAPATH, NOISE_DATAPATH_TEST, VORBIS_WINDOW

    DATA_PATH = data_path
    SUBSET = subset
    SLICE_DURATION = slice_duration
    SAVE_AS = save_as
    OUTPUT = DATA_PATH + '/' + SUBSET
    NOISE_DATAPATH = DATA_PATH + '/Applause'
    NOISE_DATAPATH_TEST = DATA_PATH + '/Applause_test'
    VORBIS_WINDOW = vorbis_window(WINDOW_LEN)


    print('data_path =', DATA_PATH)
    print('subset =', SUBSET)
    print('slice_duration =', SLICE_DURATION)
    print('slice_step =', SLICE_STEP)
    print('save_as =', SAVE_AS)
    print('sample_rate =', SAMPLE_RATE)
    print('window_ms =', WINDOW_MS)
    print('window_len =', WINDOW_LEN)
    print('step =', STEP)
    print('Starting preparing dataset...')

    process_function = process_signal_as_h5 if SAVE_AS == 'h5' else process_signal_as_wav
    music_list = glob.glob(os.path.join(DATA_PATH + '/Songs/' + SUBSET, "**/mixture16.wav"), recursive=True)
    
    # creating validation set
    if SUBSET == 'train' and split_valid > 0 and split_valid <= 1:
        np.random.shuffle(music_list)
        val_count = int(len(music_list) * split_valid)
        music_list_val = music_list[:val_count]
        music_list_train = music_list[val_count:]
        
        print("Creating train split")
        create_output_folder()
        for audio_path in tqdm(music_list_train):
            process_function(audio_path)
        
        print("Creating valid split")
        OUTPUT = DATA_PATH + '/' + 'valid'
        NOISE_DATAPATH = DATA_PATH + '/Applause_valid'
        create_output_folder()
        for audio_path in tqdm(music_list_val):
            process_function(audio_path)    
    else:
        create_output_folder()
        for audio_path in tqdm(music_list):
            process_function(audio_path)

    print('Preparing dataset has finished successfully.')


def process_signal_as_wav(audio_path):
    sr, signal = wavfile.read(audio_path)
    assert sr == 16000
    
    # signals should be float to avoid int overflow and mono
    signal = signal.astype('float32')
    signal = to_mono(signal)
    
    if len(signal) < SLICE_DURATION * SAMPLE_RATE:
        return

    for sl in range(len(signal) // SLICE_STEP - 1):
        file_path_signal = get_path(audio_path, 'song', sl)
        file_path_noisy = get_path(audio_path, 'noisy_song', sl)

        signal_slice = signal[sl * SLICE_STEP: sl * SLICE_STEP + SLICE_DURATION * SAMPLE_RATE]
        noisy_signal_slice, noise = get_noisy_signal(signal_slice)

        signal_slice_with_new_phase = np.zeros(len(signal_slice))
        noisy_signal_new_slice = np.zeros(len(noisy_signal_slice))
        frame_count = len(signal_slice) // STEP - 1

        for i in range(frame_count):
            # noisy
            noisy_cur_window = noisy_signal_slice[i * STEP: i * STEP + WINDOW_LEN]
            noisy_magnitude, noisy_phase, noisy_fft = get_magn_phase(noisy_cur_window)
            # signal
            signal_cur_window = signal_slice[i * STEP: i * STEP + WINDOW_LEN]
            signal_magnitude, signal_phase, signal_fft = get_magn_phase(signal_cur_window)
            # new signal
            signal_new_fft = signal_magnitude * noisy_phase
            signal_new_window = np.fft.irfft(signal_new_fft) * vorbis_window(WINDOW_LEN)
            signal_slice_with_new_phase[i * STEP: i * STEP + WINDOW_LEN] += signal_new_window
            # new noisy    
            noisy_new_fft = noisy_magnitude * noisy_phase
            noisy_new_window = np.fft.irfft(noisy_new_fft) * vorbis_window(WINDOW_LEN)
            noisy_signal_new_slice[i * STEP: i * STEP + WINDOW_LEN] += noisy_new_window

        write_wav(file_path_signal, signal_slice_with_new_phase)
        write_wav(file_path_noisy, noisy_signal_new_slice)


def process_signal_as_h5(audio_path):
    sr, signal = wavfile.read(audio_path)
    assert sr == 16000

    # signals should be float to avoid int overflow and mono
    signal = signal.astype('float32')
    signal = to_mono(signal)
    
    if len(signal) < SLICE_DURATION * SAMPLE_RATE:
        return

    for sl in range(len(signal) // SLICE_STEP - 1):
        signal_slice = signal[sl * SLICE_STEP: sl * SLICE_STEP + SLICE_DURATION * SAMPLE_RATE]

        # making noisy songs using different noises with the same song slice
        for k in range(3):
            noisy_signal_slice, noise = get_noisy_signal(signal_slice)

            signal_fft_slice = []
            noisy_signal_fft_slice = []
            noise_fft_slice = []
            frame_count = len(signal_slice) // STEP - 1

            for i in range(frame_count):
                # signal
                signal_cur_window = signal_slice[i * STEP: i * STEP + WINDOW_LEN]
                signal_magnitude, signal_phase, signal_fft = get_magn_phase(signal_cur_window)
                signal_fft_slice.append(signal_magnitude)
                # noisy
                noisy_cur_window = noisy_signal_slice[i * STEP: i * STEP + WINDOW_LEN]
                noisy_magnitude, noisy_phase, noisy_fft = get_magn_phase(noisy_cur_window)
                noisy_signal_fft_slice.append(noisy_magnitude)
                # noisy - signal
                noise_magnitude = np.abs(noisy_fft - signal_fft)
                noise_fft_slice.append(noise_magnitude)

            # making np array
            signal_fft_slice = np.asarray(signal_fft_slice) 
            noisy_signal_fft_slice = np.asarray(noisy_signal_fft_slice)
            noise_fft_slice = np.asarray(noise_fft_slice)  
       
            # wrire song       
            song_file_path = get_path(audio_path, 'song', sl, k)
            write_hdf5(signal_fft_slice, song_file_path)
            # write noisy_song
            noisy_file_path = get_path(audio_path, 'noisy', sl, k)
            write_hdf5(noisy_signal_fft_slice, noisy_file_path)

            if DIFFERENCE_FFT:
                # write noise_minus_signal
                noise_file_path = get_path(audio_path, 'noisy_minus_signal', sl, k)
                write_hdf5(noise_fft_slice, noise_file_path)


def write_hdf5(signal_fft, path):
    # handle clipping
    max_absolute_signal = np.max(np.abs(signal_fft))
    if max_absolute_signal > 32767:
        signal_fft = signal_fft * (32767 / max_absolute_signal)  

    with h5py.File(path, 'w') as hf:
        hf.create_dataset('dataset', data=signal_fft) 


def write_wav(signal, path):
    # handle clipping
    max_absolute_signal = np.max(np.abs(signal))
    if max_absolute_signal > 32767:
        signal = signal * (32767 / max_absolute_signal)  

    wavfile.write(path, SAMPLE_RATE, signal.astype("int16").flatten())                    


def get_noisy_signal(signal):
    noise_paths = glob.glob(os.path.join(NOISE_DATAPATH if SUBSET == 'train' else NOISE_DATAPATH_TEST, "*.wav"))
    noise_path = np.random.choice(noise_paths, 1)[0]
    print(noise_path)
    sr, noise = wavfile.read(noise_path)
    assert sr == 16000
    
    # signals should be float to avoid int overflow and mono
    noise = noise.astype('float32')
    noise = to_mono(noise)

    if len(signal) > len(noise):
        noise_modified = get_tiled_noise(noise, len(signal))
    else: 
        # get random noise slice
        interval = len(noise)-len(signal)+1
        rand_int = np.random.randint(interval)   
        noise_modified = noise[rand_int:rand_int+len(signal)]

    snr = get_SNR(signal, noise_modified)
    # multiplied with snr with noise because we get ratios of signal/noise to fixed signal energy
    noise_with_snr = snr * noise_modified
    noisy_signal = signal + noise_with_snr
    assert len(noisy_signal) == len(signal)

    return noisy_signal, noise_with_snr


def get_tiled_noise(noise, n):
    y = noise
    while len(y) < n:
        y = np.append(y, noise)
    y = y[0:n]

    return y


def get_SNR(signal, noise):
    # dB scale
    rand_dB = np.random.randint(-5, 15)

    # astype needed for not get negative powers
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noise))
    ratio = signal_power / (noise_power + 1e-7)

    # initial SNR is 10 * np.log10(signal_power / noise_power)
    # signal to noise scale factor
    # Attention to <<minus>> rand_dB
    db_lin = np.power(10, -rand_dB / 10)
    k = np.sqrt(ratio * db_lin)
    
    return k


def to_mono(signal):
    # stereo
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    return signal


def get_magn_phase(arr):
    VORBIS_WINDOW = vorbis_window(WINDOW_LEN)
    fft = np.fft.rfft(arr * VORBIS_WINDOW)
    magn = np.abs(fft) + 1e-7
    phase = fft / magn

    return magn, phase, fft


def vorbis_window(n):
    return np.sin((np.pi / 2) * (np.sin(np.pi * np.array(range(n)) / n)) ** 2)


def get_path(audio_path, folder, index, index_2=""):
    audio_name = audio_path.split('/')[-2]
    filename = audio_name + '_' + str(index) + ('_' + str(index_2) if index_2 else "")
    alias = 'wav' if SAVE_AS == 'wav' else 'h5'
    folder_path = OUTPUT + '/' + alias + 's/' + folder
    path = os.path.join(folder_path, (filename + '.' + alias))

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return path


def create_output_folder():
    folder = '/wavs' if SAVE_AS == 'wav' else '/h5s'

    if not os.path.exists(OUTPUT + folder):
        os.makedirs(OUTPUT + folder)


def get_signal_from_fft(magn, phase):
    VORBIS_WINDOW = vorbis_window(WINDOW_LEN)
    signal = np.zeros(SLICE_DURATION * SAMPLE_RATE)
    frame_count = len(signal) // STEP - 1

    for i in range(frame_count):
        cur_phase = phase[i]
        cur_magn = magn[i]
        cur_fft = cur_magn * cur_phase
        cur_window = np.fft.irfft(cur_fft) * VORBIS_WINDOW
        signal[i * STEP: i * STEP + WINDOW_LEN] += cur_window

    return np.array(signal)

def process_test_song(path):
    sr, noisy_song = wavfile.read(path)
    assert sr == 16000

    noisy_song = to_mono(noisy_song)
    noisy_song = noisy_song.astype('float32')

    song_len = len(noisy_song)
    magn_batch = []
    phase_batch = []

    for sl in range(song_len // SLICE_STEP - 1):
        # get slices
        noisy_song_slice = noisy_song[sl * SLICE_STEP: sl * SLICE_STEP + SLICE_DURATION * SAMPLE_RATE]

        # calculate fft magn and phase for each slice
        noisy_song_fft_magn = []
        noisy_song_fft_phase = []
        frame_count = len(noisy_song_slice) // STEP - 1

        for i in range(frame_count):
            noisy_cur_window = noisy_song_slice[i * STEP: i * STEP + WINDOW_LEN]
            noisy_magnitude, noisy_phase, noisy_fft = get_magn_phase(noisy_cur_window)
            # append
            noisy_song_fft_magn.append(noisy_magnitude)
            noisy_song_fft_phase.append(noisy_phase)

        magn_batch.append(noisy_song_fft_magn)
        phase_batch.append(noisy_song_fft_phase)    

    # converting to np.array
    magn_batch = np.array(magn_batch)
    phase_batch = np.array(phase_batch)

    return magn_batch, phase_batch

def generate_noisy_signal(signal_path):
    sr, signal = wavfile.read(signal_path)
    assert sr == 16000
    signal_len = len(signal)
    
    # signals should be float to avoid int overflow and mono
    signal = signal.astype('float32')
    signal = to_mono(signal)

    # config
    global SUBSET, NOISE_DATAPATH, NOISE_DATAPATH_TEST
    SUBSET = "train"
    NOISE_DATAPATH = DATA_PATH + '/Applause'
    NOISE_DATAPATH_TEST = DATA_PATH + '/Applause_test'
        
    if signal_len < SLICE_DURATION * SAMPLE_RATE:
        return

    noise = np.zeros(signal_len)
    noisy_signal = np.zeros(signal_len)
    
    for sl in range(signal_len // SLICE_STEP - 1):
        # get slices
        signal_slice = signal[sl * SLICE_STEP: sl * SLICE_STEP + SLICE_DURATION * SAMPLE_RATE]
        noisy_signal_slice, noise_slice = get_noisy_signal(signal_slice)
        assert len(signal_slice) == len(noise_slice)

        # get whole noise and noisy_signal
        noise[sl * SLICE_STEP: sl * SLICE_STEP + SLICE_DURATION * SAMPLE_RATE] += noise_slice * (vorbis_window(len(noise_slice)) ** 2)
        noisy_signal[sl * SLICE_STEP: sl * SLICE_STEP + SLICE_DURATION * SAMPLE_RATE] += noisy_signal_slice * (vorbis_window(len(noisy_signal_slice)))
  
    #save signals as wav
    signal_name = os.path.basename(signal_path)[:-4]
    write_wav(noisy_signal, os.path.join(os.path.dirname(signal_path), signal_name + '_noisy.wav'))
    
    print("Generated successfully")
