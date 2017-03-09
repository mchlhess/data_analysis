import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def load_clinical_eeg_data(datapath, sub):
  import pandas as pd
  alldata = pd.read_csv(os.path.join(datapath, sub + '.csv'))
  alldata.rename(columns={'Unnamed: 0': 'Index'})
  eegevents = alldata[['labels', 'chunks']]
  alldata.drop(['Unnamed: 0', 'labels', 'chunks'], axis=1, inplace=True)
  names = alldata.keys()
  return alldata.iloc[:].as_matrix(), eegevents, names

data, label_chunk, nodes = load_clinical_eeg_data('train/','chb01')

labels_and_chunks = label_chunk.as_matrix()
s_res = labels_and_chunks[:,0]
print s_res.shape

non_szr = []
pre_szr = []
szr = []

marker = 0
for i in range(0, len(s_res)):
  if s_res[i] == 0:
    marker = 0
    non_szr.append(data[i])
  if s_res[i] == 1 and marker == 0:
    marker = 1
    for n in range(1, 640):
      s_res[i - n] = 1
      pre_szr.append(data[i - n])
      print("changing")
  if s_res[i] == 1 and marker == 1:
    s_res[i] = 2
    szr.append(data[i])


np.savetxt('pre_szr.csv', pre_szr, delimiter=',', fmt='%f')
np.savetxt('non_szr.csv', non_szr, delimiter=',', fmt='%f')
np.savetxt('szr.csv', szr, delimiter=',', fmt='%f')
