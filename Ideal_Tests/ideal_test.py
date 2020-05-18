from scipy.io import wavfile
import numpy as np

mix_path = "../data/Songs/test_song/Alexander Ross - Velvet Curtain/mixture16_noisy_song.wav"
song_path = "../data/Songs/test_song/Alexander Ross - Velvet Curtain/mixture16.wav"
real_sr = 16000
big_window_len = real_sr * 5
big_step = int(big_window_len / 2)
small_window_len = int(real_sr * 30 / 1000)
small_step = int(small_window_len / 2)
small_frames_count = int(big_window_len / small_step) - 1

sr_mix, mix = wavfile.read(mix_path) 
assert sr_mix == real_sr
sr_song, song = wavfile.read(song_path) 
assert sr_song == real_sr
assert len(mix) == len(song)

big_frames_count = len(mix) // big_step - 1

def vorbis_window(N):
    return np.sin((np.pi / 2) * (np.sin(np.pi * np.arange(N) / N)) ** 2)

small_vorbis_window = vorbis_window(small_window_len)
big_vorbis_window_squared = vorbis_window(big_window_len) ** 2

new_song = np.zeros(len(mix))
for i in range(big_frames_count):
    cur_mix_big_frame = mix[i * big_step: i * big_step + big_window_len]
    cur_song_big_frame = song[i * big_step: i * big_step + big_window_len]

    assert len(cur_mix_big_frame) == big_window_len
    assert len(cur_song_big_frame) == big_window_len

    new_big_window = np.zeros(big_window_len)
    for j in range(small_frames_count):
        cur_mix_small_frame = cur_mix_big_frame[j * small_step: j * small_step + small_window_len]
        cur_song_small_frame = cur_song_big_frame[j * small_step: j * small_step + small_window_len]
        
        cur_mix_ft = np.fft.rfft(cur_mix_small_frame * small_vorbis_window)
        cur_song_ft = np.fft.rfft(cur_song_small_frame * small_vorbis_window)

        #clips
        # cur_ratio_mask = np.clip(np.abs(cur_song_ft)/ (np.abs(cur_mix_ft) + 1e-7), 0, 1)
        # cur_ratio_mask = np.clip(np.abs(cur_song_ft)/ (np.abs(cur_mix_ft) + 1e-7), 0, 5)

        #divide
        # cur_ratio_mask = np.abs(cur_mix_ft - cur_song_ft) / (np.abs(cur_mix_ft)+ 1e-7)
        # cur_ratio_mask_appl = np.abs(cur_mix_ft - cur_song_ft) / (np.abs(cur_mix_ft)+ 1e-7)

        #clips appl
        # cur_ratio_mask_appl = np.clip(np.abs(cur_mix_ft - cur_song_ft) / (np.abs(cur_mix_ft)+ 1e-7), 0, 1)
        # cur_ratio_mask_appl = np.clip(np.abs(cur_mix_ft - cur_song_ft) / (np.abs(cur_mix_ft)+ 1e-7), 0, 5)

        #sqr
        # cur_ratio_mask = np.sqrt(np.abs(cur_song_ft)**2/(np.abs(cur_mix_ft-cur_song_ft)**2 +  np.abs(cur_song_ft)**2))
        cur_ratio_mask = np.sqrt(np.abs(cur_song_ft)**2/ (np.abs(cur_mix_ft - cur_song_ft)**2 + np.abs(cur_song_ft)**2))
        # cur_ratio_mask = np.abs(cur_song_ft)**2/ (np.abs(cur_mix_ft - cur_song_ft)**2 + cur_song_ft**2)**2

        #sqr appl
        # cur_ratio_mask_appl = np.abs(cur_mix_ft - cur_song_ft)**2/ (np.abs(cur_mix_ft - cur_song_ft)**2 + cur_song_ft**2)
        # cur_ratio_mask_appl = np.abs(cur_mix_ft - cur_song_ft)**2/ (np.abs(cur_mix_ft - cur_song_ft)**2 + cur_song_ft**2)**2
# 
        new_cur_song_ft = cur_ratio_mask * cur_mix_ft
        # new_cur_song_ft = cur_mix_ft - cur_ratio_mask_appl * cur_mix_ft


        new_song_small_frame = np.fft.irfft(new_cur_song_ft) * small_vorbis_window
        new_big_window[j * small_step: j * small_step + small_window_len] += new_song_small_frame

    new_song[i * big_step: i * big_step + big_window_len] += new_big_window * big_vorbis_window_squared

    # handle clipping
max_absolute_new_song = np.max(np.abs(new_song))
if max_absolute_new_song > 32767:
    new_song = new_song * (32767 / max_absolute_new_song)

wavfile.write("./same_clap_velv/sqr_sqrt.wav", real_sr, new_song.astype("int16"))
