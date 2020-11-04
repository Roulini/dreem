import numpy as np 
import pandas as pd

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

def esis_epoch(data_epoch):
    """return sum of amplitude of the signal for an epoch"""
    return sum([esis(split) for split in windows_split(data_epoch)])



def get_features(data,n=None):
    """extract de features from the data""" 
    n_eeg = 2
    
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