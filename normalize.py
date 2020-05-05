import os
import glob
import h5py
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd


x_train_paths = glob.glob(os.path.join('./data/train/h5s/noisy', "*.h5"))

x_data = []
    
# reading files from paths
for path in tqdm(x_train_paths):
    # getting signal array
    with h5py.File(path, 'r') as f:
        data = list(f['dataset'])
    x_data.append(data)

# converting to np.array
x_data = np.array(x_data)

print(x_data.shape)

x_mean = np.mean(x_data)
x_std = np.std(x_data)

stat_dict = {
            'x_data': [x_mean, x_std],
          }

df = pd.DataFrame(stat_dict)
df.to_csv('norm_stats.csv')
