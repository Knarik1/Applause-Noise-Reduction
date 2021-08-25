# Applause-Noise-Reduction

Goal is to clean concert audio record from applauses.

## Datasets
  * MUSDB18-HQ, for high quality music audios
  * Free Sound, for random concert noises
  
  
## Defining Ideal Masks for clean audio estimtaion 
<img src="https://github.com/Knarik1/Applause-Noise-Reduction/blob/master/images/ideal_masks.png" width="700">
  
## Pre-proceesing
  * downsample all music audio from 44.kHz to 16kHz
  * make mono from stereo by averaging left and right audio signals
  <img src="https://github.com/Knarik1/Applause-Noise-Reduction/blob/master/images/downsample.png" width="700">
  
  * splitting audio into 5 sec intervals with 2.5 sec overlap 
  <img src="https://github.com/Knarik1/Applause-Noise-Reduction/blob/master/images/pre_processing.png" width="700">
  
  * get STFT features
  <img src="https://github.com/Knarik1/Applause-Noise-Reduction/blob/master/images/stft.png" width="700">
  
  
## Creating a noisy dataset from audio and noises datasets
  * adding random noises to audio with different SNR's
  <img src="https://github.com/Knarik1/Applause-Noise-Reduction/blob/master/images/noisy_dataset.png" width="700">


## Estimating the mask of clean audio signal
  * using Bi-GRU model


## Post-processing
  * from STFT going back to audio signal
  * adding phase info from noisy audio to estimated audio
  <img src="https://github.com/Knarik1/Applause-Noise-Reduction/blob/master/images/post_processing.png" width="700">
