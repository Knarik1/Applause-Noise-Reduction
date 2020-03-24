import os
import glob
import h5py
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

        # normalizing
        range_x = np.max(x_batch) - np.min(x_batch)
        if range_x > 0:
            x_batch = (x_batch - np.min(x_batch)) / range_x

        range_y = np.max(y_batch) - np.min(y_batch)
        if range_y > 0:
            y_batch = (y_batch - np.min(y_batch)) / range_y

        return x_batch, y_batch

    def train_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.config["train_batch_size"], self.train_paths, index, perm=perm)

    def val_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.config["val_batch_size"], self.val_paths, index, perm=perm)

    def test_data_loader(self, index, perm=None):
        return self.batch_data_loader(self.config["test_batch_size"], self.test_paths, index, perm=perm)

    def get_test_song(self):
        x_path = self.test_paths.T[0][0]
        y_path = self.test_paths.T[0][1]

        print("Signal for test is\n", x_path, "\n", y_path)

        with h5py.File(x_path, 'r') as f:
            x_data = np.array(list(f['dataset']))

        with h5py.File(y_path, 'r') as f:
            y_data = np.array(list(f['dataset']))

        return x_data, y_data