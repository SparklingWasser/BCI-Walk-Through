
# from braindecode.models.util import models_dict
# print(f'All the Braindecode models:\n{list(models_dict.keys())}')

import numpy as np
import scipy.io as sio

file_path = '/Volumes/SPARK_Data/Cloud/연구/Open Dataset/DEAP database/data_preprocessed_matlab/s10.mat'
data_loaded = sio.loadmat(file_path)

ch_idx = np.arange(0, 32)
t_pre_stim = 3
t_data_to_exclude = 30
fs = 128
val_dim = 0
valence_mid = 5

data = data_loaded['data'][:, ch_idx, (t_pre_stim+t_data_to_exclude)*fs:] # epochs, channel, time-series
data_list = [data[i] for i in range(data.shape[0])]

label = data_loaded['labels'][:, val_dim]
label = np.array((label > valence_mid)*1)
label_list = label.tolist()


from braindecode.datasets import create_from_X_y

ch_names = list(['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'])

win_len = 5

windows_dataset = create_from_X_y(
    data_list, label_list, drop_last_window=True, sfreq=fs, ch_names=ch_names,
    window_stride_samples=win_len*fs,
    window_size_samples=win_len*fs,
)

from skorch.callbacks import LRScheduler
from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split, SliceDataset
from skorch.dataset import Dataset
from skorch.dataset import ValidSplit

X = SliceDataset(windows_dataset, idx=0)
y = np.array([y for y in SliceDataset(windows_dataset, idx=1)])


import torch
from braindecode.util import set_random_seeds
import random

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = "cuda" if cuda else "cpu"

seed = 2
set_random_seeds(seed=seed, cuda=cuda)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

training_data, test_data, training_label, test_label = train_test_split(X, y, test_size=0.2, shuffle=False)


from braindecode.models import EEGConformer
from braindecode import EEGClassifier

n_classes = len(np.unique(label))
classes = list(range(n_classes))

n_channels = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]

class Conformer:
    def __init__(self):
        self.n_outputs = 2
        self.validation_set_piece = 4
        self.model = EEGConformer(
            n_outputs = self.n_outputs,
            n_chans = 32,
            n_times = 640,
            sfreq = fs,
            final_fc_length=1480,
        )
        # print(self.model)
        # print(EEGConformer.__doc__)
    
    def initialize(self):
        lr = 0.0000625
        weight_decay = 0.001
        batch_size = 32
        n_epochs = 20

        cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
        device = "cuda" if cuda else "cpu"
        if cuda:
            self.model.cuda()
            torch.backends.cudnn.benchmark = True

        clf = EEGClassifier(
            self.model,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            train_split=ValidSplit(self.validation_set_piece),
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=[
                "balanced_accuracy",
                ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
            ],
            device=device,
            classes=list(range(self.n_outputs)),
            max_epochs=n_epochs,
        )
        return clf
        
model = Conformer().initialize()
model.fit(training_data, training_label)

test_acc = model.score(test_data, test_label)
print(f"Test acc: {(test_acc * 100):.2f}%")

# print(f"Data balance in test set: {(np.sum(test_label)/len(test_label)):.1f} (0.5 means half and half)")
# print(f"Data balance in valid set: {(np.sum(y_valid)/len(y_valid)):.1f} (0.5 means half and half)")

