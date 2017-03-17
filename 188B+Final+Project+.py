#Adam's code for Naive Baye's identification and prediction
#Filter Function
def apply_filters(x, order=4, fs=512.0, filt=None, btype='low', axis=0):
    from scipy.signal import butter, filtfilt
    nyq = .5 * fs
    
    if filt is None:
        return x
    if isinstance(filt, list):
        f = [i/nyq for i in filt]
        btype='band'
    else:
        f = filt/nyq
        
    b, a = butter(order, f, btype=btype, analog=False)
    x = filtfilt(b, a, x, axis=axis)
    return x


### Building Appropriate Sample Sizes ###     
# Our dataset is very one-sided, and wee need to sample evenly to avoid our models only predicting "no seizure"
def re_label3(data,labels):
    import numpy as np
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
    import numpy as np
    samples = []
    for arg in args:
        for i in range(n):
            samples.append(arg[i,:])
            
    labelr = []
    for labels in labelList:
        for i in range(n):
            labelr.append(labels)
    return np.array(samples), np.array(labelr)
    
#Build equally sized set
def equal_dataset(data,labels):
    import numpy as np
    nonData, seizData, preData = re_label3(data, labels)
    
    #Shuffle each list
    np.random.shuffle(nonData)
    np.random.shuffle(seizData)
    np.random.shuffle(preData)
    
    #generate a new dataset
    size = 64 * 30 # ~30 sec of each data type
    new_data, new_labels = sample_sizer([0,1,2], size, nonData,preData,seizData)
    
    return new_data, new_labels

#load data, condensed as we did this previously
def load_clinical_eeg_data(datapath, sub):
    import pandas as pd
    alldata = pd.read_csv(os.path.join(datapath, sub)) #removed 'train' bc of how I saved
    alldata.rename(columns={'Unnamed: 0': 'Index'})
    eegevents = alldata[['labels', 'chunks']]
    alldata.drop(['Unnamed: 0', 'labels', 'chunks'], axis=1, inplace=True)
    names = alldata.keys()
    return alldata.iloc[:].as_matrix(), eegevents, names    


#use the next two functions to preprocess and test the testing data for naive bayes prediction and identification, logistic regression prediction, and k-nearest neighbors prediction.


#############################################
# Run:
# clean_data = nb_prepro(filename, nb_pca)
# predictions = run_sk_model(clean_data, nb_model)

###############################################

#Process the data
def nb_prepro(filename,preprocname):
    
    ####### Jeff: Change path to directory
    path ='directory'
    data, label_chunk, nodes = load_clinical_eeg_data(path,filename)
    
    #Scale data before PCA
    import sklearn.preprocessing as prepro
    x_scaled = prepro.scale(data)
    
    
    #Run pkl PCA
    from sklearn.externals import joblib
    preproc = joblib.load(preprocname)
    preprocessed_x = preproc.fit_transform(x_scaled)
    
    #Run Bandpass
    x_bp = apply_filters(preprocessed_x, order=4, fs=64, filt=[12, 30], btype='band')
    return x_bp  
    
def run_sk_model(x, modelname):
    from sklearn.externals import joblib
    clf = joblib.load(modelname)  # load the saved sklearn model (so you don't have to retrain)
    return clf.predict(x)