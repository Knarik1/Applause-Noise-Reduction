# Applause-Noise-Reduction

Goal is to clean concert audio record from applauses.

## Datasets
  * MUSDB18-HQ, for high quality music audios
  * Free Sound, for random concert noises
  
## Pre-proceesing
  * downsample all music audio from 44.kHz to 16kHz
  * make mono from stereo by averaging left and right audio signals
  * get STFT features
  
## Creating a noisy dataset from audio and noises datasets
  * adding random noises to audio with different SNR's

## Estimating the mask of clean audio signal
  * using Bi-GRU model

## Post-processing
  * from STFT going back to audio signal
  * adding phase info from noisy audio to estimated audio
