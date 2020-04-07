import os
import glob
import h5py
import shutil
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

DATA_PATH = './data'
SUBSET = 'train'
SLICE_DURATION = 5
SAVE_AS = 'h5'
SAMPLE_RATE = 16000
WINDOW_MS = 30
WINDOW_LEN = SAMPLE_RATE * WINDOW_MS // 1000
STEP = WINDOW_LEN // 2
VORBIS_WINDOW = []
OUTPUT = ''
NOISE_DATAPATH = ''
NOISE_DATAPATH_TEST = ''


def prepare_dataset(data_path,
                    subset='train',
                    split_valid=None,
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
    print('save_as =', SAVE_AS)
    print('sample_rate =', SAMPLE_RATE)
    print('window_ms =', WINDOW_MS)
    print('window_len =', WINDOW_LEN)
    print('step =', STEP)
    print('Starting preparing dataset...')

    process_function = process_signal_as_h5 if SAVE_AS == 'h5' else process_signal_as_wav
    music_list = glob.glob(os.path.join(DATA_PATH + '/Songs/' + SUBSET, "**/mixture16.wav"), recursive=True)
    create_output_folder()

    for audio_path in tqdm(music_list):
        process_function(audio_path)

    # creating validation set
    if SUBSET == 'train' and split_valid is not None:
        if split_valid <= 0 or split_valid >= 1:
            return 

        OUTPUT = DATA_PATH + '/' + 'valid'
        create_output_folder()   
        folder_names = ['noisy', 'song']

        for folder_name in tqdm(folder_names):
            path = OUTPUT + '/' + SAVE_AS + 's/' + folder_name

            if not os.path.exists(path):
                os.makedirs(path)

            source_path = DATA_PATH + '/train/' + SAVE_AS + 's/' + folder_name
            dest_path = path

            all_files = os.listdir(source_path)
            # get the first split
            perm = np.random.permutation(len(all_files))[:int(split_valid * len(all_files))]
            
            split_files = np.array(all_files)[perm]

            for f in split_files:
                shutil.move(source_path + '/' + f, dest_path + '/' + f)

    print('Preparing dataset has finished successfully.')


def process_signal_as_wav(audio_path):
    # sr = 16000
    sr, signal = wavfile.read(audio_path)

    if len(signal) < SLICE_DURATION:
        return

    for sl in range(len(signal) // (SLICE_DURATION * SAMPLE_RATE)):
        file_path_signal = get_path(audio_path, 'song', sl)
        file_path_noisy = get_path(audio_path, 'noisy_song', sl)

        signal_slice = signal[sl * SAMPLE_RATE * SLICE_DURATION : (sl + 1) * SLICE_DURATION * SAMPLE_RATE]
        noisy_signal_slice, noise = get_noisy_signal(signal_slice)

        signal_slice_with_new_phase = np.zeros(len(signal_slice))
        noisy_signal_new_slice = np.zeros(len(noisy_signal_slice))
        frame_count = len(signal_slice) // STEP - 1

        for i in range(frame_count):
            # noisy
            noisy_cur_window = noisy_signal_slice[i * STEP: i * STEP + WINDOW_LEN]
            noisy_magnitude, noisy_phase = get_magn_phase(noisy_cur_window)
            # signal
            signal_cur_window = signal_slice[i * STEP: i * STEP + WINDOW_LEN]
            signal_magnitude, signal_phase = get_magn_phase(signal_cur_window)
            # new signal
            signal_new_fft = signal_magnitude * noisy_phase
            signal_new_window = np.fft.irfft(signal_new_fft) * VORBIS_WINDOW
            signal_slice_with_new_phase[i * STEP: i * STEP + WINDOW_LEN] += signal_new_window
            # new noisy    
            noisy_new_fft = noisy_magnitude * noisy_phase
            noisy_new_window = np.fft.irfft(noisy_new_fft) * VORBIS_WINDOW
            noisy_signal_new_slice[i * STEP: i * STEP + WINDOW_LEN] += noisy_new_window

        wavfile.write(file_path_signal, SAMPLE_RATE, signal_slice_with_new_phase.astype("int16"))
        wavfile.write(file_path_noisy, SAMPLE_RATE, noisy_signal_new_slice.astype("int16"))


def process_signal_as_h5(audio_path):
    # sr = 16000
    sr, signal = wavfile.read(audio_path)

    if len(signal) < SLICE_DURATION:
        return

    for sl in range(len(signal) // (SLICE_DURATION * SAMPLE_RATE)):
        signal_slice = signal[sl * SAMPLE_RATE * SLICE_DURATION: (sl + 1) * SLICE_DURATION * SAMPLE_RATE]
        noisy_signal_slice, noise = get_noisy_signal(signal_slice)

        signal_fft_slice = []
        noisy_signal_fft_slice = []
        frame_count = len(signal_slice) // STEP - 1

        for i in range(frame_count):
            # signal
            signal_cur_window = signal_slice[i * STEP: i * STEP + WINDOW_LEN]
            signal_magnitude, signal_phase = get_magn_phase(signal_cur_window)
            signal_fft_slice.append(signal_magnitude)
            # noisy
            noisy_cur_window = noisy_signal_slice[i * STEP: i * STEP + WINDOW_LEN]
            noisy_magnitude, noisy_phase = get_magn_phase(noisy_cur_window)
            noisy_signal_fft_slice.append(noisy_magnitude)
  
        song_file_path = get_path(audio_path, 'song', sl)
        with h5py.File(song_file_path, 'w') as hf:
            hf.create_dataset('dataset', data=np.asarray(signal_fft_slice))

        noisy_file_path = get_path(audio_path, 'noisy', sl)
        with h5py.File(noisy_file_path, 'w') as hf:
            hf.create_dataset('dataset', data=np.asarray(noisy_signal_fft_slice))


def get_noisy_signal(signal):
    noise_paths = glob.glob(os.path.join(NOISE_DATAPATH if SUBSET == 'train' else NOISE_DATAPATH_TEST, "*.wav"))
    noise_path = np.random.choice(noise_paths, 1)[0]
    sr, noise = wavfile.read(noise_path)

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
    rand_dB = np.random.randint(-3, 20)

    # astype needed for not get negative powers
    signal_power = np.mean(np.square(signal.astype(np.int64)))
    noise_power = np.mean(np.square(noise.astype(np.int64))) + 1e-7
    ratio = signal_power / noise_power

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
    fft = np.fft.rfft(arr * VORBIS_WINDOW)
    magn = np.abs(fft) + 1e-7
    phase = fft / magn

    return magn, phase


def vorbis_window(n):
    return np.sin((np.pi / 2) * (np.sin(np.pi * np.array(range(n)) / n)) ** 2)


def get_path(audio_path, folder, index):
    audio_name = audio_path.split('/')[-2]
    filename = audio_name + '_' + str(index)
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
