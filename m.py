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

X_train, X_test, Y_train, Y_test = train_test_split(data, s_res, test_size=.2)

# create model
model = Sequential()
model.add(Dense(5, input_dim=23, init='uniform', activation='relu'))
model.add(Dense(5, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, Y_train, nb_epoch=1, batch_size=100)
# evaluate the model
scores = model.evaluate(X_train, Y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

proba = model.predict_proba(X_test, batch_size=32)
classes = np.argmax(proba, axis=1)
print(accuracy_score(Y_test, classes))
