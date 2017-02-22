import os

def load_clinical_eeg_data(datapath, sub):
  # input arguments:
  # datapath (string): path to the root directory
  # sub (string): subject ID (e.g. chb01, chb02, etc)
  
  # output:
  # eegdata (numpy array): samples x channels data matrix
  # eegevents (pandas dataframe): labels and chunks
  # channel_names (list): names of the channels
  import pandas as pd
  alldata = pd.read_csv(os.path.join(datapath, 'train', sub + '.csv'))
  alldata.rename(columns={'Unnamed: 0': 'Index'})
  eegevents = alldata[['labels', 'chunks']]
  alldata.drop(['Unnamed: 0', 'labels', 'chunks'], axis=1, inplace=True)
  names = alldata.keys()
  return alldata.iloc[:].as_matrix(), eegevents, names

matrix, eegchunks, chan_name = load_clinical_eeg_data('', 'chb01')
