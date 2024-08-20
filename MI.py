

import mne
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mne.decoding import CSP as CommonSpatialPattern

sampling_freq = 128  
ch_names = ["C3", "Cz", "C4"]
ch_types = ["eeg"] * len(ch_names)
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
info.set_montage("standard_1020")

path_name = '/Volumes/SPARK_Data/Cloud/연구/10. BCI Survey/Walk_Through_Example/MI/BCICII_3/'
data = sio.loadmat(path_name+'dataset.mat')
y_test_mat = sio.loadmat(path_name+'labels.mat')

x_train_data = data['x_train'].transpose()
x_test_data = data['x_test'].transpose()
training_label = data['y_train'].squeeze()
test_label = y_test_mat['y_test'].squeeze()
del data, y_test_mat

training_data = mne.EpochsArray(x_train_data, info).crop(tmin=4.0, tmax=7.0)
test_data = mne.EpochsArray(x_test_data, info).crop(tmin=4.0, tmax=7.0)
# del x_train_data, x_test_data

training_data = training_data.get_data(copy=False)
test_data = test_data.get_data(copy=False)

CSP = CommonSpatialPattern(n_components=2)
feature_vector_training = CSP.fit_transform(training_data, training_label)
feature_vector_test = CSP.transform(test_data)

model = LinearDiscriminantAnalysis()
model.fit(feature_vector_training, training_label)

score = float(model.score(feature_vector_test, test_label))*100
print("Classification accuracy: %.1f%% \n" % score)
