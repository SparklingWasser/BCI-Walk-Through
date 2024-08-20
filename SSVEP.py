

import scipy.io as sio
import numpy as np
from meegkit.trca import TRCA


path = '/Volumes/SPARK_Data/Cloud/연구/Open Dataset/40_Class_SSVEP/'

data_loaded = sio.loadmat(path + 'S1.mat')['data']
ch_sel_list = np.concatenate([np.arange(47, 48), np.arange(53, 58), np.arange(60, 63)]) # 47: Pz, 53: PO5, 54: PO3, 55: POz, 56: PO4, 57: PO6, 60: O1, 61: Oz, 62: O2
data_ch_sel = data_loaded[ch_sel_list].transpose(1, 0, 2, 3)
del data_loaded

n_samp_ori, n_ch, n_class, number_of_runs = data_ch_sel.shape
total_label = np.array([i for i in range(n_class)]*number_of_runs).reshape(number_of_runs, n_class).transpose()

stim_info = sio.loadmat(path + 'Freq_Phase.mat')
phase = stim_info['phases'].squeeze()
freq = stim_info['freqs'].squeeze()

fs = 250
t_preonset = 0.5
t_stim = 5
t_postonset = 0.5

t_delay = 0.14
t_data_in_use = 0.5

samp_start = round((t_preonset + t_delay)*fs)
samp_end = round((t_preonset + t_delay + t_data_in_use)*fs) 
samp_in_use = np.arange(samp_start, samp_end)
total_data = data_ch_sel[samp_in_use]
n_samp_epo = total_data.shape[0]

filterbank = [[(6, 90), (4, 100)],  # passband, stopband freqs [(Wp), (Ws)]
              [(14, 90), (10, 100)],
              [(22, 90), (16, 100)],
              [(30, 90), (24, 100)],
              [(38, 90), (32, 100)],
              [(46, 90), (40, 100)],
              [(54, 90), (48, 100)]]

is_ensemble = True  # True = ensemble TRCA method; False = TRCA method
trca = TRCA(fs, filterbank, is_ensemble)


class CrossValidation:
    def __init__(self):
        self.number_of_samples_epoched = 125
        self.number_of_channels = 9
        self.number_of_runs = number_of_runs
        self.acc = []

    def fold_assign(self, total_fold, test_fold):
        self.training_fold = [x for x in range(0, total_fold)]
        self.training_fold.remove(test_fold)
        self.test_fold = test_fold
        return self.training_fold, self.test_fold
    
    def data_assign(self, total_data, total_label, i):
        training_fold, test_fold = CV.fold_assign(self.number_of_runs, i)

        self.training_data = total_data[..., training_fold].reshape(self.number_of_samples_epoched, self.number_of_channels, -1)
        self.training_label = total_label[:, training_fold].reshape(-1)

        self.test_data = total_data[..., test_fold].reshape(self.number_of_samples_epoched, self.number_of_channels, -1)
        self.test_label = total_label[:, test_fold].reshape(-1)

        return self.training_data, self.training_label, self.test_data, self.test_label
    
    def score(self, predicted_label, true_label):
        is_correct = predicted_label == true_label
        self.acc.append(np.mean(is_correct)*100)
        
    def accuracy_print(self):
        print("Classification accuracy: %.1f%% \n" % np.mean(self.acc))


for i in range(number_of_runs):
    CV = CrossValidation()
    training_data, training_label, test_data, test_label = CV.data_assign(total_data, total_label, i)

    trca.fit(training_data, training_label)
    predicted_label = trca.predict(test_data)
    CV.score(predicted_label, test_label)

CV.accuracy_print()
