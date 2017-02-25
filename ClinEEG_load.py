
# coding: utf-8

# In[2]:

def load_clinical_eeg_data(datapath, sub):
    # input arguments:
    # datapath (string): path to the root directory
    # sub (string): subject ID (e.g. chb01, chb02, etc)
    
    # output:
    # eegdata (numpy array): samples x channels data matrix
    # eegevents (pandas dataframe): labels and chunks
    # channel_names (list): names of the channels
    import pandas as pd
    alldata = pd.read_csv(os.path.join(datapath, sub + '.csv')) #removed 'train' bc of how I saved
    alldata.rename(columns={'Unnamed: 0': 'Index'})
    eegevents = alldata[['labels', 'chunks']]
    alldata.drop(['Unnamed: 0', 'labels', 'chunks'], axis=1, inplace=True)
    names = alldata.keys()
    return alldata.iloc[:].as_matrix(), eegevents, names


# In[3]:

import os
#Appropriate dir
os.chdir("C:\\Users\\adam1brownell\\Desktop\Winter2017\\188B Files\Project")
path = os.getcwd()

#Get subject names from appropriate dir, -4 for .csv suffux
subjects = [f[:-4] for f in os.listdir(path)]

data, label_chunk, nodes = load_clinical_eeg_data(path,subjects[3])


# In[23]:

#Visualize Raw Data
import matplotlib.pyplot as plt

#Random Node choice
x = data[:,3]
print x.shape
time = [i for i in range(len(x))]
plt.plot(time,x)
plt.title(nodes[3] + " Raw Data")
plt.show()
plt.plot(time[1:325], x[1:325])
plt.show()


# In[26]:

#Line Noise Removal (Median Filtering)
from scipy import signal
window_size =  21/200 * len(x) + 1
med_x = signal.medfilt(x,65)
plt.plot(time,med_x)
plt.title(nodes[3] + " Median Filtered")
plt.show()
plt.plot(time[1:325], med_x[1:325])
plt.show()


# In[49]:

#Line Noise Removal (Bandpass Filtering)
#Nyquist Frequency = Half sampling rate (64hz)
A,B = signal.butter(3,0.05)
band_x = signal.filtfilt(A,B, x)
plt.plot(time,x)
plt.plot(time,band_x)
plt.show()
#plt.plot(time[1:325], x[1:325])
plt.plot(time[1:325], band_x[1:325])
plt.show()


# In[57]:

#Hilbert Transform
hil_x = signal.hilbert(band_x)
plt.plot(time,hil_x)
plt.show()
#plt.plot(time[1:325], x[1:325])
#plt.plot(time[1:325], band_x[1:325])
plt.plot(time[1:325], hil_x[1:325])
plt.show()


# In[ ]:




# In[53]:

#Load + Clean all data
for sub in subjects: #subjects is a list of subject names
    data, label_chunk, nodes = load_clinical_eeg_data(path,sub) #path is the directory of data


# In[ ]:



