
# coding: utf-8

# In[1]:

#load data, condensed as we did this previously
def load_clinical_eeg_data(datapath, sub):
    alldata = pd.read_csv(os.path.join(datapath, sub + '.csv')) #removed 'train' bc of how I saved
    alldata.rename(columns={'Unnamed: 0': 'Index'})
    eegevents = alldata[['labels', 'chunks']]
    alldata.drop(['Unnamed: 0', 'labels', 'chunks'], axis=1, inplace=True)
    names = alldata.keys()
    return alldata.iloc[:].as_matrix(), eegevents, names
import os
import numpy as np
import pandas as pd
os.chdir("C:\\Users\\adam1brownell\\Desktop\Winter2017\\188B Files\Project")
path = os.getcwd()

#Get subject names from appropriate dir, -4 for .csv suffux
#subjects = [f[:-4] for f in os.listdir(path)]

#data, label_chunk, nodes = load_clinical_eeg_data(path,subjects[3])
#print data
#print label_chunk
#data2, label_chunk2, nodes = load_clinical_eeg_data(path,subjects[4])


# In[2]:

def load_all_data(model):
    import os
    import numpy as np
    import pandas as pd
    os.chdir("C:\\Users\\adam1brownell\\Desktop\Winter2017\\188B Files\Project")
    path = os.getcwd()
    subjects = [f[:-4] for f in os.listdir(path)]
    i = 1
    j = len(subjects)
    
    for sub in subjects:
        print "Loading data from patient ", i, " of ", j
        
        data, label_chunk, nodes = load_clinical_eeg_data(path,sub)
        datalist.append(pd.DataFrame(data=data))
        labellist.append(label_chunk)
        i = i + 1
        
    print "Concatinating to single dataset..."
    big_data = pd.concat(datalist)
    big_labels = pd.concat(labellist)
    
    return np.array(big_data), np.array(big_labels)
#Get subject names from appropriate dir, -4 for .csv suffux
#x = pd.DataFrame(data=data)
#data2, label_chunk2, nodes = load_clinical_eeg_data(path,subjects[4])
#x2 = pd.DataFrame(data=data2)
#x3 = pd.concat([x,x2])
#x3


# In[3]:

# Our dataset is very one-sided, and wee need to sample evenly to avoid our models only predicting "no seizure"
def re_label3(data,labels):
    non_szr = []
    pre_szr = []
    szr = []

    marker = 0
    f_label = labels
    f_data = data
    for i in range(0, len(f_label)):
        if f_label[i] == 0:
            marker = 0
            if i < len(f_label) - 640 and f_label[i + 640] != 1:
                non_szr.append(f_data[i])

        elif f_label[i] == 1 and marker == 0:
            marker = 1

            for n in range(640, 1, -1):
                pre_szr.append(f_data[i - n])

        elif f_label[i] == 1 and marker == 1:
            szr.append(f_data[i])

    pre_szr = np.asarray(pre_szr)
    szr = np.asarray(szr)
    non_szr = np.asarray(non_szr)
    
    return non_szr, szr, pre_szr

def sample_sizer(labelList, n, *args):
    samples = []
    for arg in args:
        for i in range(n):
            samples.append(arg[i,:])
            
    labelr = []
    for labels in labelList:
        for i in range(n):
            labelr.append(labels)
    return np.array(samples), np.array(labelr)


# In[4]:

#build spectrogram dataset
def specdata(data,labels):
    i = 0
    specData = []
    specLabels = []
    j = .90
    while i + 640 < len(labels):
        for node in range(7): #Push all 7 important nodes into one
            specgram, freqs, t = mlab.specgram(data[i:i+640,node], NFFT=64, noverlap=63, Fs=64)
            specData.append(specgram.flatten())
            specLabels.append(labels[i+320])
        if i + (j*len(labels)) >= len(labels):
            print (1-j)*100, "% done"
            j = j - .10
        i = i + 64
    print "100% done"
    
    non, b4, seiz = re_label3(specData,specLabels)
    np.random.shuffle(non)
    np.random.shuffle(b4)
    np.random.shuffle(seiz)
    size = min([len(non),len(seiz),len(b4)])
    spec_data, spec_labels = sample_sizer([0,1], size, non,b4)
    
    return spec_data, spec_labels


# In[5]:

from matplotlib import mlab
from sklearn.decomposition import PCA
import sklearn.preprocessing as prepro

import os
import numpy as np
import pandas as pd
    
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils


# In[6]:

def nn_prepro(data,labels):
    from matplotlib import mlab
    from sklearn.decomposition import PCA
    import sklearn.preprocessing as prepro
    
    "Running PCA to pick most important Nodes...."
    pca = PCA(n_components = 7)
    node_data = pca.fit_transform(data)
    
    "Building Spectrograms..."
    spec_data, spec_labels = specdata(node_data,labels)

    "Running PCA to pick most important Frequencies..."
    pca = PCA(n_components = 1000)
    x_pca = pca.fit_transform(spec_data)
    print "Cleaning Finished"
    nn_labels = np_utils.to_categorical(spec_labels, 3)
    
    return x_pca, nn_labels


# In[7]:

#def load_train():
    
    #model = Sequential()
    #model.add(Dense(64, input_dim=1000, init='normal', activation='sigmoid'))
    #model.add(Dense(32, init='normal', activation='sigmoid'))
    #model.add(Dense(3, init='uniform', activation='softmax'))

    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #os.chdir("C:\\Users\\adam1brownell\\Desktop\Winter2017\\188B Files\Project")
    #path = os.getcwd()
    #subjects = [f[:-4] for f in os.listdir(path)]
    #i = 1
    #j = len(subjects)
    
    #for sub in subjects:
        #print "Loading data from patient ", i, " of ", j
        
        #data, label_chunk, nodes = load_clinical_eeg_data(path,sub)
        #labels = np.array(label_chunk)[:,0]
        
        #clean_data, clean_labels = nn_prepro(data, labels)
        
        #del data
        #del label_chunk
        #del labels
        
        #print "Train NN Model: "
        #model.fit(clean_data, clean_labels, nb_epoch=250, batch_size=100) #mini-batch
        
        
        #i = i+1
    #return model


# In[9]:

#### Run Once ####
model = Sequential()
model.add(Dense(64, input_dim=1000, init='normal', activation='sigmoid'))
model.add(Dense(32, init='normal', activation='sigmoid'))
model.add(Dense(3, init='uniform', activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
subjects = [f[:-4] for f in os.listdir(path)]


# In[7]:

from keras.models import load_model
modelname='spec_nn.h5'
model = load_model(modelname)
#model = model
#CHANGE i !!!! #######
i = 10
os.chdir("C:\\Users\\adam1brownell\\Desktop\Winter2017\\188B Files\Project")
path = os.getcwd()
subjects = [f[:-4] for f in os.listdir(path)]


print "Loading data from patient ", i+1, " of ", 11
sub = subjects[i]        
data, label_chunk, nodes = load_clinical_eeg_data(path,sub)
labels = np.array(label_chunk)[:,0]
clean_data, clean_labels = nn_prepro(data, labels)

print "Train NN Model: "
model.fit(clean_data, clean_labels, nb_epoch=250, batch_size=100) #mini-batch

model.save(modelname)
print "Model Saved"


# In[9]:

from sklearn.externals import joblib
pca1 = PCA(n_components = 7)
clfname = 'nn_pca1.pkl'  # CHANGE THIS 
joblib.dump(pca1, clfname)
pca2 = PCA(n_components = 1000)
clfname = 'nn_pca2.pkl'  # CHANGE THIS 
joblib.dump(pca2, clfname)


# In[ ]:

#from matplotlib import mlab
#from sklearn.decomposition import PCA
#import sklearn.preprocessing as prepro

#data, label_chunk = load_all_data()
#labels = np.array(label_chunk)[:,0]

#"Running PCA to pick most important Nodes...."
#pca = PCA(n_components = 7)
#node_data = pca.fit_transform(data)

#"Building Spectrograms..."
#spec_data, spec_labels = specdata(node_data,labels)

#"Running PCA to pick most important Frequencies..."
#pca = PCA(n_components = 1000)
#x_pca = pca.fit_transform(spec_data)
#print "PCA finished"


# In[20]:

# create model

#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.utils import np_utils

#model = Sequential()
#model.add(Dense(64, input_dim=1000, init='normal', activation='sigmoid'))
#model.add(Dense(32, init='normal', activation='sigmoid'))
#model.add(Dense(3, init='uniform', activation='softmax'))

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#import sklearn.cross_validation as ms

#nn_labels = np_utils.to_categorical(spec_labels, 3)

#x_train, x_test, y_train, y_test = ms.train_test_split(x_pca, nn_labels)
#print "Train NN Model: "
#model.fit(x_train, y_train, nb_epoch=500, batch_size=100) #mini-batch


# In[21]:

#score
#scores = model.evaluate(x_test, y_test)
#print scores[1]


# In[ ]:




# In[ ]:



