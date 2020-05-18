import os
import glob
import h5py
import time
import numpy as np
import pandas as pd
import preprocess as preprocess

class DataLoader:
    def __init__(self, train_dir, val_dir, test_dir, train_batch_size, valid_batch_size,
                 test_batch_size, n_inputs, seq_length):

        self.config = {
            "train_batch_size": train_batch_size,
            "valid_batch_size": valid_batch_size,
            "test_batch_size": test_batch_size,
            "n_inputs": n_inputs,
            "seq_length": seq_length
        }

        # getting data file names
        # sort data to sure the match between signal and noisy signal paths
        x_train_paths = glob.glob(os.path.join(train_dir + '/h5s/noisy', "*.h5"))
        y_train_paths = glob.glob(os.path.join(train_dir + '/h5s/song', "*.h5"))
        diff_train_paths = glob.glob(os.path.join(train_dir + '/h5s/noisy_minus_signal', "*.h5"))
        self.train_paths = np.sort(np.array([x_train_paths, y_train_paths, diff_train_paths]), axis=1)
        self.m_train = self.train_paths.shape[1]

        x_val_paths = glob.glob(os.path.join(val_dir + '/h5s/noisy', "*.h5"))
        y_val_paths = glob.glob(os.path.join(val_dir + '/h5s/song', "*.h5"))
        diff_val_paths = glob.glob(os.path.join(val_dir + '/h5s/noisy_minus_signal', "*.h5"))
        self.val_paths = np.sort(np.array([x_val_paths, y_val_paths, diff_val_paths]), axis=1)
        self.m_val = self.val_paths.shape[1]

        x_test_paths = glob.glob(os.path.join(test_dir + '/h5s/noisy', "*.h5"))
        y_test_paths = glob.glob(os.path.join(test_dir + '/h5s/song', "*.h5"))
        diff_test_paths = glob.glob(os.path.join(test_dir + '/h5s/noisy_minus_signal', "*.h5"))
        self.test_paths = np.sort(np.array([x_test_paths, y_test_paths, diff_test_paths]), axis=1)
        self.m_test = self.test_paths.shape[1]

        print("----Data Loader init----")
        

    def batch_data_loader(self, batch_size, file_paths, index):
        # get mini-batch of paths
        x_paths_batch = file_paths[0][index*batch_size: (index+1)* batch_size]
        y_paths_batch = file_paths[1][index*batch_size: (index+1)* batch_size]
        diff_paths_batch = file_paths[2][index*batch_size: (index+1)* batch_size]
       
        x_batch = preprocess.read_hdf5(x_paths_batch)
        y_batch = preprocess.read_hdf5(y_paths_batch)
        diff_batch = preprocess.read_hdf5(diff_paths_batch)

        # reading stats values
        # df = pd.read_csv('norm_stats.csv', index_col=None) 
        # x_mean = df.iloc[0]['x_data']
        # x_std = df.iloc[1]['x_data']

        # normalizing
        # x_batch_norm = (x_batch - x_mean) / x_std 

        # mask is signal/noisy_signal
        # mask = np.clip(y_batch / (x_batch + 10e-7), 0, 1)
        # diff_batch**2 /(diff_batch**2 + y_batch**2 + 10e-7)
        mask = y_batch**2 /(diff_batch**2 + y_batch**2 + 10e-7)

        return x_batch, mask

    def test_song_data_loader (self, path):
        magn_batch = []
        phase_batch = []

        x_paths_batch = np.sort(glob.glob(os.path.join(path + '/h5s/magnitude', "*.h5")))
        y_paths_batch = np.sort(glob.glob(os.path.join(path + '/h5s/phase', "*.h5")))

        magn_batch = preprocess.read_hdf5(x_paths_batch)
        phase_batch = preprocess.read_hdf5(y_paths_batch)

        return magn_batch, phase_batch

    def train_data_loader(self, index):
        return self.batch_data_loader(self.config["train_batch_size"], self.train_paths, index)

    def val_data_loader(self, index):
        return self.batch_data_loader(self.config["valid_batch_size"], self.val_paths, index)

    def test_data_loader(self, index):
        return self.batch_data_loader(self.config["test_batch_size"], self.test_paths, index)
