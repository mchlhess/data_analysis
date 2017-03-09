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

print range(640, 1, -1)

def load_clinical_eeg_data(datapath, sub):
  import pandas as pd
  alldata = pd.read_csv(os.path.join(datapath, sub))
  alldata.rename(columns={'Unnamed: 0': 'Index'})
  eegevents = alldata[['labels']]
  alldata.drop(['Unnamed: 0', 'labels', 'chunks'], axis=1, inplace=True)
  names = alldata.keys()
  return alldata.iloc[:].as_matrix(), eegevents.as_matrix()

subjects = [f[:] for f in os.listdir('train')]
for i in subjects[1:]:
  print i

f_data, f_label = load_clinical_eeg_data('train/','chb01.csv')

for i in subjects[2:]:
  t_data, t_label = load_clinical_eeg_data('train', i)
  print t_data.shape
  print t_label
  f_data = np.vstack((f_data, t_data))
  f_label = np.vstack((f_label, t_label))

print(f_data)
print(f_data.shape)
print(f_label)
print(f_label.shape)

non_szr = []
pre_szr = []
szr = []

marker = 0
for i in range(0, len(f_label)):
  if f_label[i] == 0:
    marker = 0
    if i < len(f_label) - 640 and f_label[i + 640] != 1:
      non_szr.append(f_data[i])
    print("-----------")
  
  elif f_label[i] == 1 and marker == 0:
    marker = 1
    print("---BEGIN SEIZURE---")
    
    for n in range(640, 1, -1):
      pre_szr.append(f_data[i - n])
      print("pre-seizure")

  elif f_label[i] == 1 and marker == 1:
    szr.append(f_data[i])
    print("seizure")

pre_szr = np.asarray(pre_szr)
szr = np.asarray(szr)
non_szr = np.asarray(non_szr)

print pre_szr.shape
print szr.shape
print non_szr.shape

np.savetxt('pre_szr.csv', pre_szr, delimiter=',', fmt='%f')
np.savetxt('non_szr.csv', non_szr, delimiter=',', fmt='%f')
np.savetxt('szr.csv', szr, delimiter=',', fmt='%f')
