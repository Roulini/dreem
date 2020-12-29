import numpy as np 
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew
import scipy.signal as sig
from scipy.special import entr
import time

def windows_split(X_data):
    """split the data in windows of length n for manipulation such as MMD (ref entropy.pdf figure 7"""
    n = len(X_data)
    return np.array([X_data[i*100:100*(i+1)] for i in range(n//100)])

def MMD(data_array):
    """return MMD of a signal"""
    n_pts = len(data_array)
    min_val = data_array[0]
    max_val = data_array[0]
    min_index = 0
    max_index = 0
    for i in range(n_pts):
        if data_array[i]<min_val:
            min_val = data_array[min_index]
            min_index = i
        if data_array[i]>max_val:
            max_val = data_array[max_index]
            min_index = i
        
    d = np.sqrt((max_index-min_index)**2 + (max_val-min_val)**2)
    return d


def MMD_epoch(data_epoch):
    """return sum of the MMD of each window of an epoch"""
    return sum([MMD(split) for split in windows_split(data_epoch)])

def get_mid_freq(freq_seq):
    """get the frequence where the amplitude of the integral at left and right are equal"""
    n_pts = len(freq_seq)
    integral = np.sum(np.abs(freq_seq[:n_pts//2]))
    cum_sum = 0 
    for i in range(len(freq_seq)):
        cum_sum = cum_sum + np.abs(freq_seq[i])
        if cum_sum>=integral/2:
            # print(cum_sum,integral)
            return i

def get_amplitude_range_freq(freq_seq,fmin,fmax):
    """get the sum of amplitude of frequencies in a given range"""
    return np.sum(np.abs(freq_seq[fmin:fmax]))
    



def esis(data_array):
    """retur sum of amplitude of the signal"""
    return np.sum(np.square(np.abs(data_array)))

def esis_epoch(data_epoch,freq_min=1,freq_max=1):
    """return sum of amplitude of the signal for an epoch"""
    freq_mid = (freq_min + freq_max) /2
    return sum([freq_mid * esis(split) for split in windows_split(data_epoch)])


def get_features_final(data,n=None):
    """extract de features from the data""" 
    n_eeg = 52
    
    if n !=None: #take only part of the data
        data_cut = {}
        for key in data.keys():
            data_cut[key] = data[key][:n]
        data = data_cut
    feature_dic={}
    for i in range(1,n_eeg+1) : feature_dic['eeg_' +str(i)+"MMD"] = [MMD_epoch(eeg_epoch) for eeg_epoch in data['eeg_' +str(i)][:]]
    feature_dic["xMMD"] = [MMD_epoch(pos_epoch) for pos_epoch in data['x'][:]]
    feature_dic["yMMD"] = [MMD_epoch(pos_epoch) for pos_epoch in data['y'][:]]
    feature_dic["zMMD"] = [MMD_epoch(pos_epoch) for pos_epoch in data['z'][:]]
    feature_dic["pulseMMD"] = [MMD_epoch(pos_epoch) for pos_epoch in data['pulse'][:]]

    for i in range(1,n_eeg+1) : feature_dic["eeg_"+str(i)+"esis"] = [esis_epoch(eeg_epoch) for eeg_epoch in data['eeg_'+str(i)][:]]
    feature_dic["xesis"] = [esis_epoch(pos_epoch) for pos_epoch in data['x'][:]]
    feature_dic["yesis"] = [esis_epoch(pos_epoch) for pos_epoch in data['y'][:]]
    feature_dic["zesis"] = [esis_epoch(pos_epoch) for pos_epoch in data['z'][:]]
    feature_dic["pulseesis"] = [esis_epoch(pos_epoch) for pos_epoch in data['pulse'][:]]

    for i in range(1,n_eeg+1): feature_dic["eeg_"+str(i)+"midfreq"] = [get_mid_freq(eeg_epoch) for eeg_epoch in data['eeg_'+str(i)][:]]
    
    
    return pd.DataFrame(feature_dic)

def extract_features(data,feature_dic,transfo,name,n_eeg):
    for i in range(1,n_eeg+1) : feature_dic['eeg_' +str(i)+name] = [transfo(eeg_epoch) for eeg_epoch in data['eeg_' +str(i)][:]]
    feature_dic["x" + name] = [transfo(pos_epoch) for pos_epoch in data['x'][:]]
    feature_dic["y" + name] = [transfo(pos_epoch) for pos_epoch in data['y'][:]]
    feature_dic["z" + name] = [transfo(pos_epoch) for pos_epoch in data['z'][:]]
    feature_dic["pulse" + name] = [transfo(pos_epoch) for pos_epoch in data['pulse'][:]]
    
    return feature_dic

def extract_fft_features(data,feature_dic,transfo,name,n_eeg):
    for i in range(1,n_eeg+1) : feature_dic['eeg_' +str(i)+name] = [transfo(np.fft.rfft(eeg_epoch)) for eeg_epoch in data['eeg_' +str(i)][:]]
    feature_dic["x" + name] = [transfo(np.fft.rfft(pos_epoch)) for pos_epoch in data['x'][:]]
    feature_dic["y" + name] = [transfo(np.fft.rfft(pos_epoch)) for pos_epoch in data['y'][:]]
    feature_dic["z" + name] = [transfo(np.fft.rfft(pos_epoch)) for pos_epoch in data['z'][:]]
    feature_dic["pulse" + name] = [transfo(np.fft.rfft(pos_epoch)) for pos_epoch in data['pulse'][:]]
    
    return feature_dic

def band_features_unit(x,fmin,fmax): 
    """extract band features for on signal"""   
    W = np.array([max(0.1,fmin),fmax])/25
    b, a = sig.butter(4, W, btype='bandpass')
    x_f = sig.lfilter(b, a, x)
    p = np.absolute(x_f)/np.absolute(x_f).sum()

    mmd = MMD(x_f)
    esis = esis_epoch(x_f, fmin, fmax)
    entropy =  entr(p).sum()

    return mmd,esis,entropy


def get_band_features(data,feature_dic,n_eeg=7):
    """return all the band features of the dataset"""
    band_features = {}

    sources = ["eeg_1","eeg_2","eeg_3","eeg_4","eeg_5","eeg_6","eeg_7"]
    bands = ["delta","theta","alpha","beta"]
    for i in range(n_eeg) :
        source = sources[i]
        tic = time.time()
        print("processing ", source)
        
        band_features["delta"] =  np.array([band_features_unit(x,0,4) for x in  data[source]])
        band_features["theta"]  = np.array([band_features_unit(x,4,8) for x in  data[source]])
        band_features["alpha"]  = np.array([band_features_unit(x,8,13) for x in  data[source]])
        band_features["beta"] = np.array([band_features_unit(x,13,22) for x in  data[source]])
        for band in bands : 
            feature_dic[source + band + "MMD"] = band_features[band][:,0]
            feature_dic[source + band + "esis"] = band_features[band][:,1]
            feature_dic[source + band + "entropy"] = band_features[band][:,2]
        tac = time.time()
        print("duration", tac-tic)
    return feature_dic

def extract_band(x,fmin,fmax):
    W = np.array([max(0.1,fmin),fmax])/25
    b, a = sig.butter(4, W, btype='bandpass')
    x_f = sig.lfilter(b, a, x)
    return x_f

def get_fmin_fmax(band):
    if band == "delta":
        return 0,4
    if band == "theta":
        return 4,8
    if band == "alpha":
        return 8,13
    if band == "beta":
        return 13,22

def get_band_fft_features(data,feature_dic,transfo,n_eeg=7):
    sources = ["eeg_1","eeg_2","eeg_3","eeg_4","eeg_5","eeg_6","eeg_7"]
    bands = ["delta","theta","alpha","beta"]

    for i in range(n_eeg) :
        source = sources[i] 
        tic = time.time()
        print("processing ", source)
        for band in bands :
            fmin,fmax = get_fmin_fmax(band) 
            data_freq = [np.abs(np.fft.rfft(extract_band(x,fmin,fmax))) for x in data[source][:]]
            feature_dic[source +"_"+ band +"_"+ "freq" +"_"+ "mean"] = [np.mean(f) for f in data_freq]
            feature_dic[source +"_"+ band +"_"+ "freq" +"_"+ "median"] =[np.median(f) for f in data_freq]
            feature_dic[source +"_"+ band +"_"+ "freq" +"_"+ "std"] = [np.std(f) for f in data_freq]
            feature_dic[source +"_"+ band +"_"+ "freq" +"_"+ "1q"] = [first_quantile(f) for f in data_freq]
            feature_dic[source +"_"+ band +"_"+ "freq" +"_"+ "3q"] = [last_quantile(f) for f in data_freq]
        
        tac = time.time()
        print("duration", tac-tic)
    return feature_dic

def first_quantile(x):
    return np.quantile(x, 0.25)
    
def last_quantile(x):
    return np.quantile(x, 0.75)

def get_features(data,n=None):
    """extract de features from the data""" 
    n_eeg = 5
    
    if n !=None: #take only part of the data
        data_cut = {}
        for key in data.keys():
            data_cut[key] = data[key][:n]
        data = data_cut
    feature_dic={}

    
    #MMD of signals
    print("process MMD")
    tic = time.time()
    feature_dic = extract_features(data,feature_dic,MMD_epoch,"MMD",n_eeg)
    tac = time.time()
    print("duration ", tac-tic)

    #esis of signals
    print("process esis")
    tic = time.time()
    feature_dic = extract_features(data,feature_dic,esis_epoch,"esis",n_eeg)
    tac = time.time()
    print("duration ", tac-tic)
    
    #midfreq of eeg
    print("process mid freq")
    tic = time.time()
    for i in range(1,n_eeg+1): feature_dic["eeg_"+str(i)+"midfreq"] = [get_mid_freq(eeg_epoch) for eeg_epoch in data['eeg_'+str(i)][:]]
    tac = time.time()
    print("duration ", tac-tic)

    #mean
    print("process mean")
    tic = time.time()

    feature_dic = extract_features(data,feature_dic,np.mean,"mean",n_eeg)
    tac = time.time()
    print("duration ", tac-tic)
    
    #std
    print("process std")
    tic = time.time()
    feature_dic = extract_features(data,feature_dic,np.std,"std",n_eeg)
    tac = time.time()
    print("duration ", tac-tic)
    

    #kurtosis
    print("process kurtosis")
    tic = time.time()
    feature_dic = extract_features(data,feature_dic,kurtosis,"kurtosis",n_eeg)
    tac = time.time()
    print("duration ", tac-tic)
    
    #skew
    print("process skew")
    tic = time.time()
    feature_dic = extract_features(data,feature_dic,skew,"skew",n_eeg)
    tac = time.time()
    print("duration ", tac-tic)
    
    #quantile    
    print("process quantiles")
    tic = time.time()
    feature_dic = extract_features(data,feature_dic,first_quantile,"quant1",n_eeg)
    feature_dic = extract_features(data,feature_dic,first_quantile,"quant3",n_eeg)
    tac = time.time()
    print("duration ", tac-tic)
    
    #band features
    print("process bands")
    tic = time.time()
    feature_dic = get_band_features(data,feature_dic)
    tac = time.time()
    print("duration ", tac-tic)
    

    ######
    #freq features
    ######
    print("process frequencies")
    tic = time.time()
    feature_dic =  get_band_fft_features(data,feature_dic,np.mean,7)
    tac = time.time()
    print("duration ", tac-tic)

    return pd.DataFrame(feature_dic)



