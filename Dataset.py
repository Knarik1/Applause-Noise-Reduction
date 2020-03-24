from torch.utils.data import Dataset
import librosa
import h5py


class WaveDataset(Dataset):
    def __init__(self, dataframe, transforms=None, use_log_scale=True):
        self.dataframe = dataframe
        self.dataframe = self.dataframe[['noisy', 'linear_mixture']]
        self.transforms = transforms
        self.use_log_scale = use_log_scale

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        paths = self.dataframe.iloc[item, :].values
        mel_specs = []
        for i, p in enumerate(paths):
            # Read saved numpy arrays that correspond to the initial music
            with h5py.File(p, 'r') as hf:
                data = hf['dataset'][:]
            mlc, phase = librosa.magphase(data)

            if self.use_log_scale:
                mlc = librosa.amplitude_to_db(mlc)
            #TODO find better way to handle even shape
            if mlc.shape[1]%2 == 0:
                mlc = mlc[:, :-1]
            mel_specs.append(mlc)

        # TODO add pipeline
        if self.transforms:
            for tr in self.transforms:
                # print(self.dataframe.iloc[item, :])
                mel_specs = tr(mel_specs)

        return mel_specs
