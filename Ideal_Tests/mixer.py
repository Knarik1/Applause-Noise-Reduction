from scipy.io import wavfile
import numpy as np
import math
import glob
import os
from tqdm import tqdm

mix_paths = './mixes' 
if not os.path.exists(mix_paths):
    os.makedirs(mix_paths)
voice_paths = glob.glob('./voices/*')
noise_paths = glob.glob('./noises/*')
dbs = [0, 5, 10]
sample_rate = 16000

def compute_power(audio):
    if 0 == len(audio):
        return 0
    return np.sum(audio ** 2) / len(audio)

for pref_db in tqdm(dbs):
    mix_dir_db = os.path.join(mix_paths, "db_" + str(pref_db))
    for noise_path in noise_paths:
        # mix_dir = os.path.join(mix_dir_db, os.path.basename(noise_path)[:-4])
        if not os.path.exists(mix_dir_db):
            os.makedirs(mix_dir_db)
        for voice_path in voice_paths:
            mix_path = os.path.join(mix_dir_db, os.path.basename(voice_path)[:-4] + "_" + os.path.basename(noise_path))
            
            # reading audios, ensuring they have same sampling rate as the value set
            v_rate, voice_data = wavfile.read(voice_path)
            n_rate, noise_data = wavfile.read(noise_path)
            
            assert v_rate == sample_rate, 'Provided sample rate and the sample of following audio do not match ' + voice_path
            assert n_rate == sample_rate, 'Provided sample rate and the sample of following audio do not match ' + noise_path
            
            # audios should be float to avoid int overflow
            voice_data = voice_data.astype('float')
            noise_data = noise_data.astype('float')
            
            voice_len = len(voice_data)
            noise_len = len(noise_data)
            
            # tile noise if needed
            if noise_len < voice_len:
                tile_no = (int(2 * voice_len / noise_len) + 1)
                noise_data = np.tile(noise_data, tile_no)
                noise_len = len(noise_data)
            
            # pick voice length interval randomly from noise
            start = np.random.choice(range(noise_len - voice_len + 1))
            noise_data = noise_data[start: start + voice_len]
            
            assert len(noise_data) == len(voice_data), 'Trying to mix different length audios'
            
            voice_energy = np.sum(voice_data ** 2)
            noise_energy = np.sum(noise_data ** 2)
            
            SNR = voice_energy / noise_energy
            
            noise_coeff = math.sqrt(SNR * math.exp(-pref_db * np.log(10) / 10))
            noise_data *= noise_coeff
            
            mix_data = voice_data + noise_data
            
            # handle clipping
            max_absolute_mix = np.max(np.abs(mix_data))
            if max_absolute_mix > 32767:
                mix_data = mix_data * (32767 / max_absolute_mix)
            
            # save as int16 audio
            mix_data = np.array(mix_data).astype('int16').flatten()
            
            wavfile.write(mix_path, sample_rate, mix_data)

