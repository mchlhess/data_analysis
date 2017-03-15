import os
import tensorflow as tf
import numpy as np

def load_clinical_eeg_data(datapath, sub):
  import pandas as pd
  alldata = pd.read_csv(os.path.join(datapath, sub))
  alldata.rename(columns={'Unnamed: 0': 'Index'})
  eegevents = alldata[['labels']]
  alldata.drop(['Unnamed: 0', 'labels', 'chunks'], axis=1, inplace=True)
  names = alldata.keys()
  return alldata.iloc[:].as_matrix(), eegevents.as_matrix()

def convert_data(filename, dir):
  f_data, f_label = load_clinical_eeg_data(dir, filename)

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

  return pre_szr, szr, non_szr
