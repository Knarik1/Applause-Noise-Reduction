import os
import glob
import h5py
import time
import numpy as np


class DataLoader:
    def __init__(self, train_dir, val_dir, test_dir, train_batch_size, val_batch_size,
                 test_batch_size, n_inputs, seq_length):

        self.config = {
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "test_batch_size": test_batch_size,
            "n_inputs": n_inputs,
            "seq_length": seq_length
        }

        # getting data file names
        x_train_paths = glob.glob(os.path.join(train_dir + '/h5s/noisy', "*.h5"))
        y_train_paths = glob.glob(os.path.join(train_dir + '/h5s/song', "*.h5"))
        self.train_paths = np.array([x_train_paths, y_train_paths])
        self.m_train = self.train_paths.shape[1]

        x_val_paths = glob.glob(os.path.join(val_dir + '/h5s/noisy', "*.h5"))
        y_val_paths = glob.glob(os.path.join(val_dir + '/h5s/song', "*.h5"))
        self.val_paths = np.array([x_val_paths, y_val_paths])
        self.m_val = self.val_paths.shape[1]

        x_test_paths = glob.glob(os.path.join(test_dir + '/h5s/noisy', "*.h5"))
        y_test_paths = glob.glob(os.path.join(test_dir + '/h5s/song', "*.h5"))
        self.test_paths = np.array([x_test_paths, y_test_paths])
        self.m_test = self.test_paths.shape[1]

        print("----Data Loader init----")

    def batch_data_loader(self, batch_size, file_paths, index, perm=None):
        x_batch = []
        y_batch = []

        # shuffle file_paths array
        if perm is not None:
            file_paths = file_paths.T[perm].T

        # get mini-batch of paths
        x_paths_batch = file_paths[0][index: index + batch_size]
        y_paths_batch = file_paths[1][index: index + batch_size]
       
        # reading files from paths
        for path in x_paths_batch:
            # getting signal array
            with h5py.File(path, 'r') as f:
                data = list(f['dataset'])
            x_batch.append(data)

        for path in y_paths_batch:
            # getting signal array
            with h5py.File(path, 'r') as f:
                data = list(f['dataset'])
            y_batch.append(data)

        # converting to np.array
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        # mask is signal/noisy_signal
        y_batch = np.clip(y_batch / (x_batch + 10e-7), 0, 1)

        return x_batch, y_batch

    def train_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.config["train_batch_size"], self.train_paths, index, perm=perm)

    def val_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.config["val_batch_size"], self.val_paths, index, perm=perm)

    def test_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.config["test_batch_size"], self.test_paths, index, perm=perm)

    def test_song_loader(self, path):
        magn_batch = []
        phase_batch = []
        
        # getting song file names
        magn_paths = glob.glob(os.path.join(path + '/h5s/magnitude', "*.h5"))
        phase_paths = glob.glob(os.path.join(path + '/h5s/phase', "*.h5"))
        self.m_test_song = len(magn_paths)

        # reading files from paths
        for path in magn_paths:
            # getting signal array
            with h5py.File(path, 'r') as f:
                data = list(f['dataset'])
            magn_batch.append(data)

        for path in phase_paths:
            # getting signal array
            with h5py.File(path, 'r') as f:
                data = list(f['dataset'])
            phase_batch.append(data)

        # converting to np.array
        magn_batch = np.array(magn_batch)
        phase_batch = np.array(phase_batch)

        return magn_batch, phase_batch