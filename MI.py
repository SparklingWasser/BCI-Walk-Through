

import mne
import scipy.io as sio
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch

from mne.decoding import CSP as CommonSpatialPattern

path_name = 'E:\\Project\\ETC_BCI-review\\Walkthrough\\dataset\\MI\\Subject 15\\with occular artifact'
data = sio.loadmat(path_name+'\\cnt.mat')['cnt'][0,0]['x'][0,0]
mrk = sio.loadmat(path_name+'\\mrk.mat')['mrk'][0,0]['y'][0,0]
trig = np.round(sio.loadmat(path_name+'\\mrk.mat')['mrk'][0,0]['time'][0,0]/1000*200)
mnt = sio.loadmat(path_name+'\\mnt.mat')['mnt'][0,0]['clab']

sampling_freq = 200
filtered = mne.filter.filter_data(
    data=data, sfreq=sampling_freq, l_freq=0.5, h_freq=30
)

epoched = np.zeros([20,32,2000])
for i in range(0,20):
    epoched[i,:,:] = filtered[int(trig[0,i]):int(trig[0,i])+2000,:].transpose()
   
ch_names = [None] * (mnt.shape)[1]
for i in range(0,(mnt.shape)[1]): ch_names[i] = mnt[0,i][0].item()
ch_types = ["eeg"] * len(ch_names)
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)

x_train_data = epoched[0:16,:,:]
x_test_data = epoched[16:20,:,:]
training_label = mrk[0,0:16]
test_label = mrk[0,16:20]

training_data = mne.EpochsArray(x_train_data, info).crop(tmin=0.0, tmax=5.0)
test_data = mne.EpochsArray(x_test_data, info).crop(tmin=0.0, tmax=5.0)

training_data = training_data.get_data(copy=False)
test_data = test_data.get_data(copy=False)

CSP = CommonSpatialPattern(n_components=2)
feature_vector_training = CSP.fit_transform(training_data, training_label)
feature_vector_test = CSP.transform(test_data)

model = LinearDiscriminantAnalysis()
model.fit(feature_vector_training, training_label)

score = float(model.score(feature_vector_test, test_label))*100
print("Classification accuracy: %.1f%% \n" % score)